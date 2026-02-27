"""
Main Evaluation Script

Runs quantitative evaluation metrics for the PET attenuation
correction challenge.

Usage (from project root):

    python -m evaluation_tool.eval \
        --subject sub-000 \
        --root data \
        -all \
        --pet_unit kBq \
        --debug
"""

import argparse
import os
import json
import numpy as np
import nibabel as nib

from metrics import (
    compute_whole_body_suv_mae,
    compute_brain_outlier_score,
    compute_organ_bias_from_totalseg,
    compute_tac_bias,
)


# =========================================================
# Helper: Simulate 4D PET (for testing only)
# =========================================================

def expand_to_4d(pet_path, num_frames=8):
    """
    Expand 3D PET into 4D by repeating volume.
    Used only for local testing when dynamic PET
    is not available.
    """

    img = nib.load(pet_path)
    data = img.get_fdata()

    if data.ndim == 4:
        return pet_path  # already dynamic

    data_4d = np.stack([data] * num_frames, axis=-1)

    temp_path = pet_path.replace(".nii", "_4d.nii")
    nib.save(nib.Nifti1Image(data_4d, img.affine), temp_path)

    return temp_path


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="PET Evaluation Tool"
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject identifier (e.g., sub-000)"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root data directory"
    )

    parser.add_argument(
        "-all",
        action="store_true",
        help="Run all metrics"
    )

    parser.add_argument(
        "-specific_metric",
        choices=[
            "whole_body_mae",
            "brain_outlier",
            "organ_bias",
            "tac_bias"
        ],
        help="Run specific metric only"
    )

    parser.add_argument(
        "--pet_unit",
        default="kBq",
        choices=["kBq", "Bq"],
        help="Unit of PET images (default: kBq)"
    )

    parser.add_argument(
        "--test_4d",
        action="store_true",
        help="Simulate dynamic PET by repeating 3D volume"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (SUV sanity check)"
    )

    args = parser.parse_args()

    subject_path = os.path.join(args.root, args.subject)

    features_path = os.path.join(subject_path, "features")
    labels_path = os.path.join(subject_path, "labels")

    # -----------------------------------------------------
    # Define file paths (adjust if naming changes)
    # -----------------------------------------------------

    pred_pet = os.path.join(
        features_path,
        f"{args.subject}_ses-quadra_trc-18FFDG_rec-nacstatOSEM_pet.nii.gz"
    )

    gt_pet = os.path.join(
        labels_path,
        f"{args.subject}_ses-quadra_trc-18FFDG_rec-acstatOSEM_pet.nii.gz"
    )

    ts_body = os.path.join(
        labels_path,
        f"{args.subject}_ses-quadra_acq-LOWDOSE_ce-none_rec-ac_seg-body_space-individual_dseg.nii.gz"
    )

    ts_total = os.path.join(
        labels_path,
        f"{args.subject}_ses-quadra_acq-LOWDOSE_ce-none_rec-ac_seg-total_space-individual_dseg.nii.gz"
    )

    synthseg = os.path.join(
        labels_path,
        f"{args.subject}_ses-vida_task-rest_acq-MPRAGE_seg-synthsegparc_space-individual_dseg.nii.gz"
    )

    meta_json = os.path.join(features_path, "constants.json")

    # -----------------------------------------------------
    # Optionally simulate dynamic PET
    # -----------------------------------------------------

    if args.test_4d:
        pred_pet = expand_to_4d(pred_pet)
        gt_pet = expand_to_4d(gt_pet)

    results = {}

    # =====================================================
    # 1. Whole-body SUV MAE
    # =====================================================

    if args.all or args.specific_metric == "whole_body_mae":

        results["Whole-body SUV MAE"] = compute_whole_body_suv_mae(
            pred_pet_path=pred_pet,
            gt_pet_path=gt_pet,
            body_mask_path=ts_body,
            liver_mask_path=ts_total,
            json_path=meta_json,
            pet_unit=args.pet_unit,
            debug=args.debug
        )

    # =====================================================
    # 2. Brain Outlier Score
    # =====================================================

    if args.all or args.specific_metric == "brain_outlier":

        results["Brain Outlier Score"] = compute_brain_outlier_score(
            pred_paths=[pred_pet],
            gt_paths=[gt_pet],
            brain_mask_paths=[synthseg]
        )

    # =====================================================
    # 3. Organ Bias
    # =====================================================

    if args.all or args.specific_metric == "organ_bias":

        organ_labels = {
            "brain": 90,
            "liver": 5,
            "spleen": 1,
            "heart": 52,
            "pancreas": 10,
            "muscle": 200,
            "adipose": 201,
            "extremities": 300,
        }

        results["Organ Bias"] = compute_organ_bias_from_totalseg(
            pred_path=pred_pet,
            gt_path=gt_pet,
            totalseg_path=ts_total,
            organ_label_dict=organ_labels,
            json_path=meta_json,
            pet_unit=args.pet_unit
        )

    # =====================================================
    # 4. TAC Bias (Dynamic Only)
    # =====================================================

    if args.all or args.specific_metric == "tac_bias":

        pet_data = nib.load(pred_pet).get_fdata()

        if pet_data.ndim != 4:
            print("TAC Bias skipped: PET is not dynamic (4D).")
        else:
            frame_durations = np.array([4.0] * pet_data.shape[-1])

            results["TAC Bias"] = compute_tac_bias(
                pred_path=pred_pet,
                gt_path=gt_pet,
                totalseg_path=ts_total,
                synthseg_path=synthseg,
                frame_durations=frame_durations,
                aorta_label=52,
                brain_label_ids=[3, 42, 10, 49, 8, 47]
            )

    # =====================================================
    # Print Results
    # =====================================================

    print("\n================ Evaluation Results ================")
    print(f"Subject: {args.subject}")
    print("----------------------------------------------------")

    if not results:
        print("No metric selected.")
    else:
        for name, value in results.items():
            print(f"{name:<25}: {value:.6f}")

    print("====================================================\n")


if __name__ == "__main__":
    main()