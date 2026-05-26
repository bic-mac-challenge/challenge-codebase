import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # fragmentation issues on dante. also reduced cache_rate to 0.1 to reduce memory footprint
# https://docs.pytorch.org/docs/2.12/notes/cuda.html#environment-variables 
# has to be before importing torch

import torch
import yaml
import matplotlib.pyplot as plt
from monai.data import DataLoader, CacheDataset
from tqdm import tqdm

from dataset import get_dataset
from transforms import get_transforms
from unet import build_model


torch.backends.cudnn.benchmark = True


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_log(log_path):
    """Parse train_log.txt → (train_loss_history, val_loss_history, last_epoch).
    Returns empty histories and -1 if the file does not exist."""
    train_losses, val_losses = [], []
    last_epoch = -1
    if not os.path.exists(log_path):
        return train_losses, val_losses, last_epoch
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            epoch, t_loss, v_loss = line.split(",")
            train_losses.append(float(t_loss))
            val_losses.append(float(v_loss))
            last_epoch = int(epoch)
    return train_losses, val_losses, last_epoch


def main():

    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    all_data = get_dataset(cfg["data_dir"])
    val_data, train_data = all_data[:2], all_data[2:]

    train_transforms = get_transforms(cfg["patch_size"], cfg["train_num_samples"])
    val_transforms   = get_transforms(cfg["patch_size"], cfg["val_num_samples"])

    print("Caching train dataset...")
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_rate=0.1,
        num_workers=cfg["num_workers"],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    print("Caching val dataset...")
    val_dataset = CacheDataset(
        data=val_data,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
    )

    scaler  = torch.amp.GradScaler("cuda")
    l1_loss = torch.nn.L1Loss()

    out = cfg["output_dir"]
    os.makedirs(f"{out}/checkpoints", exist_ok=True)
    os.makedirs(f"{out}/logs",        exist_ok=True)
    os.makedirs(f"{out}/plots",       exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Resume from last checkpoint if one exists                          #
    # ------------------------------------------------------------------ #
    last_ckpt_path = f"{out}/checkpoints/last_model.pth"
    log_path       = f"{out}/logs/train_log.txt"

    train_loss_history, val_loss_history, last_epoch = load_log(log_path)
    start_epoch  = last_epoch + 1
    best_val_loss = min(val_loss_history) if val_loss_history else float("inf")

    if os.path.exists(last_ckpt_path):
        print(f"Resuming from checkpoint (epoch {last_epoch}) → {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)

        # Support both old format (bare state_dict) and new format (dict)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            # Sanity-check the stored epoch matches the log
            stored_epoch = ckpt.get("epoch", last_epoch)
            if stored_epoch != last_epoch:
                print(
                    f"  Warning: checkpoint epoch ({stored_epoch}) differs from "
                    f"log epoch ({last_epoch}). Using log epoch."
                )
        else:
            # Legacy bare state_dict — load weights only
            print("  (legacy checkpoint format — loading weights only)")
            model.load_state_dict(ckpt)
            # Fast-forward the scheduler to match elapsed epochs
            for _ in range(start_epoch):
                scheduler.step()
    else:
        print("No checkpoint found — starting from scratch.")

    if start_epoch >= cfg["epochs"]:
        print(
            f"Training already complete ({start_epoch} epochs done, "
            f"target is {cfg['epochs']}). Nothing to do."
        )
        return

    print(f"Starting training from epoch {start_epoch} → {cfg['epochs'] - 1}")

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, cfg["epochs"]):

        model.train()
        epoch_loss = 0
        pbar = tqdm(loader)

        for batch in pbar:

            x    = batch["input"].to(device)
            y    = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            y[~mask] = 0  # don't bother trying to predict the bed

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = l1_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_description(f"loss {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(loader)
        scheduler.step()

        # Release gradient memory
        optimizer.zero_grad(set_to_none=True)
        del x, y, mask, pred, loss
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["input"].to(device)
                y    = batch["ct"].to(device)
                mask = batch["prediction_mask"].bool().to(device)
                y[~mask] = 0

                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = l1_loss(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Best checkpoint — weights only (used for inference)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                f"{out}/checkpoints/best_model.pth",
            )

        # Last checkpoint — full training state for resuming
        torch.save(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
            },
            last_ckpt_path,
        )

        # Append to log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        # Plot loss (x-axis reflects true epoch numbers)
        epochs_so_far = list(range(len(train_loss_history)))
        plt.figure()
        plt.plot(epochs_so_far, train_loss_history, label="train")
        plt.plot(epochs_so_far, val_loss_history,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train / Val Loss")
        plt.legend()
        plt.savefig(f"{out}/plots/loss_curve.png")
        plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())
    print("training baseline model - PET only")
    main()