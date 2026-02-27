"""
Common utilities used across metrics.
"""

import numpy as np
import nibabel as nib


# =========================================================
# Brain Outlier Utilities
# =========================================================

def compute_k_value(pred_path, gt_path, brain_mask_path,
                    threshold=0.05, epsilon=1e-6):
    """
    Compute k value for a single case.

    k = fraction of brain voxels with relative error < threshold
    """

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    brain_mask = nib.load(brain_mask_path).get_fdata() > 0

    # If 4D, use first frame (static metric)
    if pred.ndim == 4:
        pred = pred[..., 0]
        gt = gt[..., 0]

    valid_mask = brain_mask & (np.abs(gt) > epsilon)
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return 0.0

    relative_error = np.abs(pred - gt) / (np.abs(gt) + epsilon)

    return np.sum(relative_error[valid_mask] < threshold) / num_valid


def compute_auc_of_K(k_values, num_points=1000):
    """
    Compute AUC of K(x):
        K(x) = fraction of cases where k > x
    """

    x_values = np.linspace(0, 1, num_points)
    K_values = []

    for x in x_values:
        Kx = np.sum(np.array(k_values) > x) / len(k_values)
        K_values.append(Kx)

    return np.trapezoid(K_values, x_values)


# =========================================================
# TAC Utilities
# =========================================================

def integrate_tac(tac, frame_durations):
    """
    Compute time-integrated activity (AUC).
    """
    return np.sum(tac * frame_durations)


def compute_region_auc(pet_4d, mask, frame_durations):
    """
    Compute integrated TAC (AUC) for one region.
    """

    T = pet_4d.shape[-1]
    tac = []

    for t in range(T):
        tac.append(np.mean(pet_4d[..., t][mask]))

    tac = np.array(tac)

    return integrate_tac(tac, frame_durations)