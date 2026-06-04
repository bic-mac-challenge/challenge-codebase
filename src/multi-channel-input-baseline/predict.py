"""
predict.py  –  Sliding-window inference for pseudo-CT prediction.

The modality list and normalization config are loaded from config.yaml so that
inference automatically matches the training setup.  Make sure the config
points to the same modalities used when the checkpoint was trained.
"""

import argparse
from pathlib import Path

import nibabel as nib
import torch
import yaml
from monai.inferers import sliding_window_inference

from dataset import get_subject_features
from transforms import get_in_channels, get_predict_transforms
from unet import build_model


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Sliding-window settings (can be overridden via CLI)
DEFAULT_PATCH_SIZE = (192, 192, 192)
DEFAULT_SW_BATCH   = 1      # increase for speed at the cost of VRAM
DEFAULT_OVERLAP    = 0.5


def predict(features_dir: str, out_path: str, model_path: str | None = None):

    cfg          = load_config()
    modalities   = cfg["input_modalities"]
    normalization = cfg["normalization"]

    patch_size = tuple(cfg.get("patch_size", DEFAULT_PATCH_SIZE))
    in_channels = get_in_channels(modalities)

    print(f"Input modalities ({in_channels} channels):")
    for m in modalities:
        print(f"  [{m['type']:10s}]  {m['name']}")

    # ── Transforms ────────────────────────────────────────────
    transforms = get_predict_transforms(modalities, normalization)

    # ── Model ─────────────────────────────────────────────────
    device = "cuda"

    if model_path is None:
        model_path = Path(__file__).parent / "outputs/checkpoints/best_model.pth"

    model = build_model(in_channels=in_channels, base_features=16).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Data ──────────────────────────────────────────────────
    subject = get_subject_features(features_dir)
    data    = transforms(subject)

    x = data["input"].unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────
    print("Sliding window inference...")
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pred = sliding_window_inference(
            x, patch_size, DEFAULT_SW_BATCH, model,
            overlap=DEFAULT_OVERLAP,
            mode="gaussian",
            progress=True,
            sw_device="cuda",
            device="cpu",
        )

    # Invert CT normalization: pred ∈ [0, 1]  →  HU ∈ [-1000, 2000]
    pred_hu = pred.cpu().numpy()[0, 0] * 3000 - 1000

    # Use affine from the first (reference) modality
    ref_key = modalities[0]["name"]
    affine  = data[ref_key].meta["affine"].numpy()

    print("Saving...")
    nib.save(nib.Nifti1Image(pred_hu, affine), out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pseudo-CT from a subject's feature directory."
    )
    parser.add_argument(
        "--features_dir", required=True,
        help="Path to the subject's features/ folder",
    )
    parser.add_argument(
        "--output_ct", required=True,
        help="Path to save the predicted pseudo-CT (e.g. ./pseudo-ct.nii.gz)",
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Path to model checkpoint (default: outputs/checkpoints/best_model.pth)",
    )
    args = parser.parse_args()
    predict(args.features_dir, args.output_ct, args.model_path)
