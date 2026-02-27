"""
Whole-body SUV MAE metric.
"""

import numpy as np
import nibabel as nib
from .suv_utils import load_pet_as_suv, suv_sanity_check


def compute_whole_body_suv_mae(
    pred_pet_path,
    gt_pet_path,
    body_mask_path,
    liver_mask_path,
    json_path,
    pet_unit="kBq",
    exclusion_cm=4.0,
    debug=False,
):
    """
    Compute voxel-wise MAE of SUV inside body,
    excluding ±4 cm around superior liver slice.
    """

    pred = load_pet_as_suv(pred_pet_path, json_path, pet_unit)
    gt = load_pet_as_suv(gt_pet_path, json_path, pet_unit)

    body_mask = nib.load(body_mask_path).get_fdata() > 0
    liver_mask = nib.load(liver_mask_path).get_fdata() > 0

    if debug:
        suv_sanity_check(pred, body_mask, "Prediction")
        suv_sanity_check(gt, body_mask, "Ground Truth")

    slice_thickness_mm = nib.load(pred_pet_path).header.get_zooms()[2]
    exclusion_slices = int(round((exclusion_cm * 10.0) / slice_thickness_mm))

    superior_slice = np.max(np.where(liver_mask)[2])

    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(pred.shape[2], superior_slice + exclusion_slices)

    exclusion_mask = np.zeros_like(body_mask, dtype=bool)
    exclusion_mask[:, :, z_min:z_max] = True

    eval_mask = body_mask & (~exclusion_mask)

    return np.mean(np.abs(pred - gt)[eval_mask])