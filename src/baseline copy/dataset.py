"""
dataset.py  –  Subject discovery and feature/label path resolution.

Returns dictionaries of file paths (and metadata) that MONAI's LoadImaged
can consume directly.  All keys must match what is used in transforms.py and
config.yaml.

To add a new feature modality:
  1. Add its path to get_subject_features() below.
  2. Add a corresponding entry under input_modalities in config.yaml.
  3. No other changes are needed — transforms.py and unet.py adapt automatically.
"""

import json
import os


def get_subject_features(features_dir: str) -> dict:
    """
    Return all available model inputs for a subject.

    Includes:
      - NAC-PET
      - Topogram  (NOTE: requires offline resampling to PET/MRI space before
                   use as a training channel; see transforms.py)
      - Dixon MRI: combined in/out-phase volumes
      - Dixon MRI: four bed-position chunks, in- and out-phase
                   (redundant with combined; use one set or the other)
      - MRI face mask
      - Metadata: sex, age, height, weight (for future metadata-conditioned models)
    """
    paths = {
        # PET
        "nacpet":                  os.path.join(features_dir, "nacpet.nii.gz"),
        # Topogram (CT-geometry localizer – needs resampling before use)
        "topogram":                os.path.join(features_dir, "topogram.nii.gz"),
        # Dixon MRI – combined (full FOV)
        "mri_combined_in_phase":   os.path.join(features_dir, "mri_combined_in_phase.nii.gz"),
        "mri_combined_out_phase":  os.path.join(features_dir, "mri_combined_out_phase.nii.gz"),
        # Dixon MRI – individual bed-position chunks
        **{
            f"mri_chunk_{i}_{phase}": os.path.join(features_dir, f"mri_chunk_{i}_{phase}.nii.gz")
            for i in range(4) for phase in ("in_phase", "out_phase")
        },
        # Auxiliary mask
        "mri_face_mask":           os.path.join(features_dir, "mri_face_mask.nii.gz"),
    }

    with open(os.path.join(features_dir, "metadata.json")) as f:
        metadata = json.load(f)   # keys: sex, age, height, weight

    return {**paths, **metadata}


def get_subject_ct_labels(ct_label_dir: str) -> dict:
    """
    Return ground-truth label paths for a subject.

    Available at training time only.  The prediction target is `ct` (HU values).
    Body / organ segmentations and the prediction mask are used to restrict the
    loss to the body region.
    """
    return {
        "ct":               os.path.join(ct_label_dir, "ct.nii.gz"),
        "body_seg":         os.path.join(ct_label_dir, "body_seg.nii.gz"),
        "organ_seg":        os.path.join(ct_label_dir, "organ_seg.nii.gz"),
        "prediction_mask":  os.path.join(ct_label_dir, "prediction_mask.nii.gz"),
    }


def get_dataset(data_dir: str) -> list:
    """
    Build a list of subject dicts from a directory tree.

    Expected layout per subject:
        <sub>/
          features/   ← get_subject_features()
          ct-label/   ← get_subject_ct_labels()
    """
    subjects = []
    for sub in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, sub)
        if not os.path.isdir(subject_dir):
            continue
        subject = get_subject_features(os.path.join(subject_dir, "features"))
        subject.update(get_subject_ct_labels(os.path.join(subject_dir, "ct-label")))
        subjects.append(subject)
    return subjects
