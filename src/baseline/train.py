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



def main():

    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    all_data = get_dataset(cfg["data_dir"])
    val_data, train_data = all_data[:2], all_data[2:]

    train_transforms = get_transforms(cfg["patch_size"], cfg["train_num_samples"])
    val_transforms = get_transforms(cfg["patch_size"], cfg["val_num_samples"])

    print("Caching train dataset...")
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_rate=0.1, # Change this to reduce memory footprint
        num_workers=cfg["num_workers"],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True
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
        persistent_workers=True
    )
    
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"]
    )

    scaler  = torch.amp.GradScaler("cuda")
    l1_loss = torch.nn.L1Loss()

    out = cfg["output_dir"]
    os.makedirs(f"{out}/checkpoints", exist_ok=True)
    os.makedirs(f"{out}/logs", exist_ok=True)
    os.makedirs(f"{out}/plots", exist_ok=True)

    best_val_loss = float("inf")

    train_loss_history = []
    val_loss_history = []

    print("Starting training...")

    for epoch in range(cfg["epochs"]):

        model.train()

        epoch_loss = 0

        pbar = tqdm(loader)

        for batch in pbar:

            x    = batch["input"].to(device)
            y    = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            y[~mask] = 0 # don't bother trying to predict the bed 
            optimizer.zero_grad(set_to_none=True) # release old gradients from memory - dante

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

        # release gradient memory from batch - dante
        optimizer.zero_grad(set_to_none=True) 
        # delete tensors from training loop
        del x, y, mask, pred, loss 
        # clear cache to avoid fragmentation errors in validation
        torch.cuda.empty_cache() 

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["input"].to(device)
                y    = batch["ct"].to(device)
                mask = batch["prediction_mask"].bool().to(device)
                y[~mask] = 0 # don't bother trying to predict the bed 

                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = l1_loss(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # best checkpoint (by val)
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                f"{out}/checkpoints/best_model.pth"
            )

        # last checkpoint
        torch.save(
            model.state_dict(),
            f"{out}/checkpoints/last_model.pth"
        )

        # log
        with open(f"{out}/logs/train_log.txt", "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        # plot loss
        plt.figure()
        plt.plot(train_loss_history, label="train")
        plt.plot(val_loss_history, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train / Val Loss")
        plt.legend()
        plt.savefig(f"{out}/plots/loss_curve.png")
        plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))  # ensure relative paths work
    print(os.getcwd())
    print("training baseline model - PET only")
    main()