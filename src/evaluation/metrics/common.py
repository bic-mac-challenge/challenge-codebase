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
    seg = nib.load(brain_mask_path).get_fdata()
    brain_mask = seg == 90

    valid_mask = brain_mask & (np.abs(gt) > epsilon)
    num_valid = np.sum(valid_mask)

    if num_valid == 0:
        return 0.0

    relative_error = np.abs(pred - gt) / (np.abs(gt) + epsilon)

    return np.sum(relative_error[valid_mask] < threshold) / num_valid


def compute_auc_of_K(k_values):
    """
    Compute AUC of K(x):
        K(x) = fraction of cases where k > x

    Analytically, AUC = mean(k_values), since ∫₀¹ I(k > x) dx = k for k ∈ [0, 1].
    """
    return np.mean(k_values)