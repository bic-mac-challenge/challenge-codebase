from metrics import mae_ct
import sys
from pathlib import Path
from nibabel import nib

def evaluate_study(prediction_ct_path, label_dir_path):
    prediction_ct = nib.load(prediction_ct_path)
    label_dir = Path(label_dir_path)
    label_ct = nib.load(next(label_dir.glob("*ct.nii.gz")))
    label_seg = nib.load(next(label_dir.glob("*seg-body_dseg.nii.gz")))
    
    assert prediction_ct.shape == label_ct.shape, "Prediction and label CT scans must have the same shape."

    mae = mae_ct(prediction_ct, label_ct, label_seg)

    return mae

if __name__ == "__main__":
    mae = evaluate_study(sys.argv[1], sys.argv[2])
    print(f"Mean Absolute Error (MAE) within body region: {mae}")