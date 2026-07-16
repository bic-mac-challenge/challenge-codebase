"""
Whole-body SUV MAE metric.
"""

import numpy as np
import nibabel as nib
from .suv_utils import compute_suv_factor

LIVER_LABEL = 5


def compute_whole_body_suv_mae(
    pred_pet_path,
    gt_pet_path,
    body_seg_path,
    organ_seg_path,
    exclusion_cm=4.0,
    save_mask_path=None,
):
    """
    Voxel-wise MAE of SUV across the body, excluding ±4 cm around the superior liver
    slice to reduce sensitivity to respiratory misalignment.

    NOTE: The STIR-reconstructed PET images are not in Bq/mL, so SUV cannot be computed
    from the recorded injected dose and subject weight in the usual way. Instead, a common
    SUV scaling factor is estimated: injected dose is proxied by the integral of the
    ground-truth PET, and subject weight by the body volume. This factor is applied
    identically to both prediction and ground truth before computing the MAE.

    Parameters
    ----------
    pred_pet_path  : path to predicted PET NIfTI
    gt_pet_path    : pet-label/pet.nii.gz
    body_seg_path  : pet-label/body_seg.nii.gz
    organ_seg_path : pet-label/organ_seg.nii.gz
    save_mask_path : optional path to save the evaluation mask as NIfTI
    """

    body_seg_img = nib.load(body_seg_path)
    gt_img = nib.load(gt_pet_path)
    suv_factor = compute_suv_factor(gt_img, body_seg_img)
    pred = nib.load(pred_pet_path).get_fdata() * suv_factor
    gt = gt_img.get_fdata() * suv_factor

    body_mask = body_seg_img.get_fdata() > 0
    liver_mask = nib.load(organ_seg_path).get_fdata() == LIVER_LABEL

    slice_thickness_mm = nib.load(pred_pet_path).header.get_zooms()[2]
    exclusion_slices = int(round((exclusion_cm * 10.0) / slice_thickness_mm))

    # PET's z-axis (STIR-resampled, ring spacing 3.29114 mm) is stored in the
    # opposite direction from the CT grid: index increases toward inferior,
    # so the superior-most liver voxel is the *smallest* index here.
    superior_slice = np.min(np.where(liver_mask)[2])

    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(pred.shape[2], superior_slice + exclusion_slices)

    exclusion_mask = np.zeros_like(body_mask, dtype=bool)
    exclusion_mask[:, :, z_min:z_max] = True

    eval_mask = body_mask & (~exclusion_mask)

    if save_mask_path is not None:
        nib.save(nib.Nifti1Image(eval_mask.astype(np.uint8), body_seg_img.affine), save_mask_path)

    return np.mean(np.abs(pred - gt)[eval_mask])
