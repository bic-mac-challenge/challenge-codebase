import sys
import nibabel as nib
from pathlib import Path

def mae_ct(prediction_ct, label_ct, label_seg):
    """Calculate Mean Absolute Error (MAE) between predicted CT and label CT within the body region."""

    pred_data = prediction_ct.get_fdata()
    label_data = label_ct.get_fdata()
    seg_data = label_seg.get_fdata()

    # Create a mask for the body region (assuming body is labeled with 1 in seg)
    body_mask = seg_data > 0

    # Calculate MAE only within the body region
    mae = abs(pred_data[body_mask] - label_data[body_mask]).mean()
    
    return mae


def mae_suv_pet(prediction_pet, label_pet, label_seg, suv_constant):
    """Calculate Mean Absolute Error (MAE) between predicted CT and label CT within the body region."""

    pred_data = prediction_pet.get_fdata() / suv_constant
    label_data = label_pet.get_fdata() / suv_constant
    seg_data = label_seg.get_fdata()

    # Create a mask for the body region (assuming body is labeled with 1 in seg)
    body_mask = seg_data > 0

    # Calculate MAE only within the body region
    mae = abs(pred_data[body_mask] - label_data[body_mask]).mean()
    
    return mae 

