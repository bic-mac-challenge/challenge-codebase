from metrics import mae_suv_pet
import sys
from pathlib import Path
from nibabel import nib

def evaluate_study(prediction_pet_path, label_dir_path):
    prediction_pet = nib.load(prediction_pet_path)
    label_dir = Path(label_dir_path)
    label_pet = nib.load(next(label_dir.glob("*pet.nii.gz")))
    label_seg = nib.load(next(label_dir.glob("*seg-body_dseg.nii.gz")))

    with open(label_dir / "suv.txt", "r") as f:
        suv = float(f.read().strip())

    assert prediction_pet.shape == label_pet.shape, "Prediction and label PET scans must have the same shape."

    mae = mae_suv_pet(prediction_pet, label_pet, label_seg, suv)
    return mae


if __name__ == "__main__":
    mae = evaluate_study(sys.argv[1], sys.argv[2])
    print(f"Mean Absolute Error (MAE) within body region: {mae}")