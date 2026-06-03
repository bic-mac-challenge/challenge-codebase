"""
train.py  –  Training loop for pseudo-CT prediction.

Input modalities, normalization strategies, and all hyperparameters are
controlled from config.yaml.  No code changes are needed when adding or
removing modalities.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # fragmentation issues on dante. also reduced cache_rate to 0.1 to reduce memory footprint
# https://docs.pytorch.org/docs/2.12/notes/cuda.html#environment-variables 
# has to be before importing torch


import matplotlib.pyplot as plt
import torch
import yaml
from monai.data import CacheDataset, DataLoader
from tqdm import tqdm
import gc

from dataset import get_dataset
from transforms import get_in_channels, get_transforms
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

    modalities      = cfg["input_modalities"]     # list of {name, type} dicts
    normalization   = cfg["normalization"]         # {type: strategy}
    in_channels     = get_in_channels(modalities)

    print(f"Input modalities ({in_channels} channels):")
    for m in modalities:
        print(f"  [{m['type']:10s}]  {m['name']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    all_data             = get_dataset(cfg["data_dir"])
    val_data, train_data = all_data[:2], all_data[2:]

    train_transforms = get_transforms(
        cfg["patch_size"], cfg["train_num_samples"],
        modalities, normalization,
    )
    val_transforms = get_transforms(
        cfg["patch_size"], cfg["val_num_samples"],
        modalities, normalization,
    )

    print("Caching train dataset...")
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_rate=0.1,   # changed from 1.0 to 0.1 for memory issues on dante
        num_workers=cfg["num_workers"],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0), # True causes memory issues on dante, set to False
    )

    print("Caching val dataset...")
    val_dataset = CacheDataset(
        data=val_data,
        transform=val_transforms,
        cache_rate=0.1,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False, # True causes memory issues on dante, set to False
        persistent_workers=(cfg["num_workers"] > 0), # True causes memory issues on dante, set to False if workers=0
    )

    model = build_model(in_channels=in_channels).to(device)
    print(f"Model: UNet3D  in_channels={in_channels}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"],
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
    best_val_loss = min(val_loss_history) if val_loss_history else float("inf")
    start_epoch = last_epoch + 1


    if os.path.exists(last_ckpt_path):
        print(f"Resuming from checkpoint (epoch {last_epoch}) → {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)

        stored_epoch = ckpt.get("epoch", last_epoch)
        start_epoch = stored_epoch + 1

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

    for epoch in range(start_epoch, cfg["epochs"]):

        # ── Training ──────────────────────────────────────────
        model.train()
        epoch_loss = 0
        train_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:04d}")

        for batch in pbar:
            x    = batch["input"].to(device)
            y    = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            y[~mask] = 0   # do not penalise predictions outside the body

            if not mask.any(): # Skip this batch entirely to avoid NaN
                continue 

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = l1_loss(pred[mask], y[mask])   # loss only inside mask

            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}")
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_steps += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = (
            epoch_loss / train_steps if train_steps > 0 else float("nan")
           )
        scheduler.step()

        # clean up memory after each epoch - memory failures when reachingvalidation on dante
        del x, y, mask, pred, loss
        torch.cuda.empty_cache()
        gc.collect() # free up cpu memory using garbage collection

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_steps = 0 # memory issues on dante when validating on full val set, so added a counter to process batches one by one. if more memory becomes available, can remove this and just average over all batches as before.
        with torch.no_grad():
            for batch in val_loader:
                x_batch    = batch["input"]
                y_batch    = batch["ct"]
                mask_batch = batch["prediction_mask"].bool()
                y_batch[~mask_batch] = 0

                # Process sequentially to avoid OOM.
                for i in range(x_batch.shape[0]): # x_batch.shape is [val_num_samples/batch size, channels, depth, height, width] or [8, C, 192, 192, 192] default
                    x = x_batch[i:i+1].to(device) 
                    y = y_batch[i:i+1].to(device)
                    mask = mask_batch[i:i+1].to(device)

                    if not mask.any():
                        del x, y, mask
                        continue

                    with torch.amp.autocast("cuda"):
                        pred = model(x)
                        loss = l1_loss(pred[mask], y[mask])
                    val_loss += loss.item()
                    val_steps += 1
            
                    # delete patch tensors to free up memory before processing the next patch
                    del x, y, mask, pred, loss
                
                del x_batch, y_batch, mask_batch
                gc.collect()



        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("nan") # avoid division by 0 nan
        # avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:04d}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        # clean up memory after validation - memory failures on dante
        torch.cuda.empty_cache()

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # ── Checkpoints ───────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{out}/checkpoints/best_model.pth")

        torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
            },
            last_ckpt_path,
        )

        # ── Logging ───────────────────────────────────────────
        with open(f"{out}/logs/train_log.txt", "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")

        epochs_so_far = list(range(len(train_loss_history)))

        plt.figure()
        plt.plot(epochs_so_far, train_loss_history, label="train")
        plt.plot(epochs_so_far, val_loss_history,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss (masked)")
        plt.title("Train / Val Loss")
        plt.legend()
        plt.savefig(f"{out}/plots/loss_curve.png")
        plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))  # ensure relative paths work
    print(os.getcwd())
    main()
