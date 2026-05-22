import argparse
import torch
import nibabel as nib
from pathlib import Path

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

from unet import build_model
from dataset import get_subject_features


MODEL_PATH = Path(__file__).parent / "outputs/checkpoints/best_model.pth"
PATCH_SIZE = (192, 192, 192)
SW_BATCH = 1 # You can increase this if to speed up inference at the cost of VRAM
OVERLAP = 0.5


def predict(features_dir, out_path):

    transforms = Compose([
        LoadImaged(keys=["nacpet"]),
        EnsureChannelFirstd(keys=["nacpet"]),
        NormalizeIntensityd(keys=["nacpet"], nonzero=False, subtrahend=[0], channel_wise=True),
        ConcatItemsd(keys=["nacpet"], name="input"),
        EnsureTyped(keys=["input"]),
    ])

    device = "cuda"

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    subject = get_subject_features(features_dir)
    data = transforms(subject)

    x = data["input"].unsqueeze(0).to(device)
    print("Sliding window inference...")
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pred = sliding_window_inference(
            x, PATCH_SIZE, SW_BATCH, model,
            overlap=OVERLAP, mode="gaussian", progress=True,
            sw_device="cuda", device="cpu",
        )

    # Apply inverse of normalization to get HU
    pred_hu = pred.cpu().numpy()[0, 0] * 3000 - 1000
    affine = data["nacpet"].meta["affine"].numpy()

    print("Saving...")
    nib.save(nib.Nifti1Image(pred_hu, affine), out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, help="Path to the subject's features/ folder")
    parser.add_argument("--output_ct",    required=True, help="Path to save the predicted pseudo-CT (e.g. './psudeo-ct.nii.gz')")
    args = parser.parse_args()

    predict(args.features_dir, args.output_ct)
