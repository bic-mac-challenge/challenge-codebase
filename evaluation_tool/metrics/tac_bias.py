"""
TAC bias metric.
"""

import numpy as np
import nibabel as nib
from .common import compute_region_auc


def compute_tac_bias(
    pred_path,
    gt_path,
    totalseg_path,
    synthseg_path,
    frame_durations,
    aorta_label,
    brain_label_ids,
    epsilon=1e-6,
):
    """
    Compute TAC bias as MARE of integrated AUC values.
    """

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    ts_seg = nib.load(totalseg_path).get_fdata()
    synthseg = nib.load(synthseg_path).get_fdata()

    assert pred.ndim == 4
    assert len(frame_durations) == pred.shape[-1]

    mare_values = []

    # Aorta
    aorta_mask = ts_seg == aorta_label
    if np.sum(aorta_mask) > 0:
        auc_pred = compute_region_auc(pred, aorta_mask, frame_durations)
        auc_gt = compute_region_auc(gt, aorta_mask, frame_durations)
        mare_values.append(np.abs(auc_pred - auc_gt) / (np.abs(auc_gt) + epsilon))

    # Brain regions
    for label_id in brain_label_ids:
        region_mask = synthseg == label_id
        if np.sum(region_mask) == 0:
            continue

        auc_pred = compute_region_auc(pred, region_mask, frame_durations)
        auc_gt = compute_region_auc(gt, region_mask, frame_durations)
        mare_values.append(np.abs(auc_pred - auc_gt) / (np.abs(auc_gt) + epsilon))

    if not mare_values:
        raise ValueError("No valid TAC regions found.")

    return np.mean(mare_values)