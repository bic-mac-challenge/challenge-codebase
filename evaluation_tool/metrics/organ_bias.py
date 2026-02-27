"""
Organ bias metric (SUV-mean MARE).
"""

import numpy as np
import nibabel as nib
from .suv_utils import load_pet_as_suv


def compute_organ_bias_from_totalseg(
    pred_path,
    gt_path,
    totalseg_path,
    organ_label_dict,
    json_path,
    pet_unit="kBq",
    epsilon=1e-6,
):
    """
    Compute mean absolute relative error (MARE)
    of SUV-mean across specified organs.
    """

    pred = load_pet_as_suv(pred_path, json_path, pet_unit)
    gt = load_pet_as_suv(gt_path, json_path, pet_unit)

    seg = nib.load(totalseg_path).get_fdata()

    mare_values = []

    for _, label_id in organ_label_dict.items():
        mask = seg == label_id
        if np.sum(mask) == 0:
            continue

        pred_mean = np.mean(pred[mask])
        gt_mean = np.mean(gt[mask])

        mare = np.abs(pred_mean - gt_mean) / (np.abs(gt_mean) + epsilon)
        mare_values.append(mare)

    if not mare_values:
        raise ValueError("No valid organs found.")

    return np.mean(mare_values)