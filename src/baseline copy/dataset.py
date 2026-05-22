import json
import os


def get_subject_features(features_dir):
    """Return all available model inputs for a subject: NAC-PET, topogram, combined and chunked
    DIXON MRI (in/out-phase), MRI face mask, and metadata (sex, age, height, weight).
    These are everything a your model may use to predict the pseudo-CT."""
    paths = {
        "nacpet":                  os.path.join(features_dir, "nacpet.nii.gz"),
        "topogram":                os.path.join(features_dir, "topogram.nii.gz"),
        "mri_combined_in_phase":   os.path.join(features_dir, "mri_combined_in_phase.nii.gz"),
        "mri_combined_out_phase":  os.path.join(features_dir, "mri_combined_out_phase.nii.gz"),
        "mri_face_mask":           os.path.join(features_dir, "mri_face_mask.nii.gz"),
        **{f"mri_chunk_{i}_{phase}": os.path.join(features_dir, f"mri_chunk_{i}_{phase}.nii.gz")
           for i in range(4) for phase in ("in_phase", "out_phase")},
    }
    with open(os.path.join(features_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return {**paths, **metadata}


def get_subject_ct_labels(ct_label_dir):
    """Return label paths for a subject. The target to predict is `ct` (Hounsfield Units),
    along with body/organ segmentations and a prediction mask."""
    return {
        "ct":               os.path.join(ct_label_dir, "ct.nii.gz"),
        "body_seg":         os.path.join(ct_label_dir, "body_seg.nii.gz"),
        "organ_seg":        os.path.join(ct_label_dir, "organ_seg.nii.gz"),
        "prediction_mask":  os.path.join(ct_label_dir, "prediction_mask.nii.gz"),
    }


def get_dataset(data_dir):
    """Build a list of subjects from a directory of subjects, each combining features and CT labels."""
    subjects = []
    for sub in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, sub)
        subject = get_subject_features(os.path.join(subject_dir, "features"))
        subject.update(get_subject_ct_labels(os.path.join(subject_dir, "ct-label")))
        subjects.append(subject)
    return subjects
