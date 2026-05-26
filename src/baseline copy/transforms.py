"""
transforms.py  –  MONAI transform pipelines for training and inference.

Normalization is organised by modality type (MRI / PET / CT / topogram) so
that each type can evolve independently.  Placeholder strategies are documented
in the per-type functions below; implement them there when you are ready.

Co-registration reminder
────────────────────────
NAC-PET and all MRI volumes share the same coordinate frame on a simultaneous
PET/MRI scanner, so no registration is needed between those modalities.

The topogram lives in a different (CT-localizer) geometry.  It must be
resampled into the PET/MRI FOV as an offline pre-processing step before it can
be included as a training channel.  A typical workflow:

    import SimpleITK as sitk
    pet   = sitk.ReadImage("nacpet.nii.gz")
    topo  = sitk.ReadImage("topogram.nii.gz")
    topo_resampled = sitk.Resample(topo, pet, sitk.Transform(),
                                   sitk.sitkLinear, 0.0,
                                   topo.GetPixelID())
    sitk.WriteImage(topo_resampled, "topogram_resampled.nii.gz")

Update dataset.py to point at the resampled file once this is done.
"""

from __future__ import annotations

from typing import Dict, List

from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-type normalization factories
# ─────────────────────────────────────────────────────────────────────────────

def get_mri_normalization(keys: List[str], strategy: str) -> List:
    """
    Return a list of MONAI transforms that normalise MRI channels.

    Implemented
    -----------
    z_score
        Zero-mean, unit-variance per channel across all voxels (nonzero mask).
        Simple and generally robust for Dixon in/out-phase images.

    Planned (not yet implemented – uncomment and test when ready)
    -------------------------------------------------------------
    percentile
        Clip to the [p_low, p_high] percentile range, then scale to [0, 1].
        Useful when the background pulls the z-score distribution.

    histogram
        Histogram matching to a population reference image.
        Increases inter-subject consistency but requires a reference atlas.

    bias_field
        N4ITK bias field correction.  This is a slow per-volume operation and
        is best run as an offline pre-processing step (e.g. via SimpleITK or
        ANTs) rather than inside the training loop.
    """
    if strategy == "z_score":
        return [
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]

    # TODO: percentile normalization
    # if strategy == "percentile":
    #     return [
    #         ScaleIntensityRangePercentilesd(
    #             keys=keys, lower=1, upper=99,
    #             b_min=0.0, b_max=1.0, clip=True, channel_wise=True,
    #         ),
    #     ]

    # TODO: histogram normalization
    # if strategy == "histogram":
    #     from monai.transforms import HistogramNormalized
    #     return [
    #         HistogramNormalized(keys=keys, num_bins=256, min=0, max=1),
    #     ]

    # NOTE: bias_field

    raise ValueError(f"Unknown MRI normalization strategy: {strategy!r}. "
                     f"Available: z_score  (percentile, histogram: see TODOs)")


def get_pet_normalization(keys: List[str], strategy: str) -> List:
    """
    Return a list of MONAI transforms that normalise PET channels.

    Implemented
    -----------
    z_score
        Zero-mean, unit-variance (including zero voxels – background matters
        for PET count distributions).

    Planned
    -------
    suv_scale
        The image is already in SUV units; clip to [0, suv_max] and scale to
        [0, 1].  Preserves absolute uptake differences across subjects, which
        may be informative for attenuation estimation.
    """
    if strategy == "z_score":
        return [
            NormalizeIntensityd(keys=keys, nonzero=False, channel_wise=True,
                                subtrahend=[0]),
        ]

    # TODO: SUV scaling
    # if strategy == "suv_scale":
    #     SUV_MAX = 20.0
    #     return [
    #         ScaleIntensityRanged(
    #             keys=keys, a_min=0, a_max=SUV_MAX,
    #             b_min=0.0, b_max=1.0, clip=True,
    #         ),
    #     ]

    raise ValueError(f"Unknown PET normalization strategy: {strategy!r}. "
                     f"Available: z_score  (suv_scale: see TODO)")


def get_ct_normalization(keys: List[str], strategy: str) -> List:
    """
    Return a list of MONAI transforms that normalise CT channels.

    Applied to the ground-truth CT during training only (not at inference).

    Implemented
    -----------
    hu_window
        Clip HU to [-1000, 2000] (air to dense bone), scale to [0, 1].
        The inverse transform (pred * 3000 - 1000) is applied in predict.py.

    Planned
    -------
    soft_tissue_window
        Narrower window, e.g. [-200, 300] HU, focusing network capacity on
        soft-tissue contrast at the cost of bone detail.
    """
    if strategy == "hu_window":
        return [
            ScaleIntensityRanged(
                keys=keys, a_min=-1000, a_max=2000,
                b_min=0.0, b_max=1.0, clip=True,
            ),
        ]

    # TODO: soft tissue window
    # if strategy == "soft_tissue_window":
    #     return [
    #         ScaleIntensityRanged(
    #             keys=keys, a_min=-200, a_max=300,
    #             b_min=0.0, b_max=1.0, clip=True,
    #         ),
    #     ]

    raise ValueError(f"Unknown CT normalization strategy: {strategy!r}. "
                     f"Available: hu_window  (soft_tissue_window: see TODO)")


def get_topogram_normalization(keys: List[str], strategy: str) -> List:
    """
    Return a list of MONAI transforms that normalise topogram channels.

    The topogram is a 2D DRR-like projection resampled to 3D PET/MRI space.
    Pixel values are roughly proportional to line-integral attenuation.

    Implemented
    -----------
    z_score
        Zero-mean, unit-variance over nonzero voxels.

    NOTE: Enable topogram in config.yaml only after the offline resampling step
    described at the top of this file has been completed.
    """
    if strategy == "z_score":
        return [
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]

    raise ValueError(f"Unknown topogram normalization strategy: {strategy!r}. "
                     f"Available: z_score")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_norm_transforms(modalities: List[Dict], normalization_cfg: Dict) -> List:
    """Group modality keys by type and return their normalization transforms."""
    by_type: Dict[str, List[str]] = {}
    for m in modalities:
        by_type.setdefault(m["type"], []).append(m["name"])

    norm_transforms = []
    dispatch = {
        "mri":       get_mri_normalization,
        "pet":       get_pet_normalization,
        "topogram":  get_topogram_normalization,
    }
    for mod_type, keys in by_type.items():
        if mod_type not in dispatch:
            raise ValueError(f"Unknown modality type: {mod_type!r}. "
                             f"Supported: {list(dispatch)}")
        strategy = normalization_cfg.get(mod_type, "z_score")
        norm_transforms += dispatch[mod_type](keys, strategy)

    return norm_transforms


def get_in_channels(modalities: List[Dict]) -> int:
    """Return the number of input channels for the model (one per modality)."""
    return len(modalities)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(
    patch_size,
    num_samples: int,
    modalities: List[Dict],
    normalization_cfg: Dict,
) -> Compose:
    """
    Build the full training/validation transform pipeline.

    Parameters
    ----------
    patch_size:
        Spatial crop size, e.g. [192, 192, 192].
    num_samples:
        Number of random patches to sample per subject.
    modalities:
        List of dicts with keys ``name`` (matching dataset.py) and ``type``
        (one of pet | mri | topogram).  Order determines channel order.
    normalization_cfg:
        Dict mapping modality type → strategy string, e.g.
        ``{'pet': 'z_score', 'mri': 'z_score', 'ct': 'hu_window'}``.
    """
    input_keys   = [m["name"] for m in modalities]
    # CT and prediction_mask are always loaded during training
    all_load_keys = input_keys + ["ct", "prediction_mask"]

    load_t = [
        LoadImaged(keys=all_load_keys),
        EnsureChannelFirstd(keys=all_load_keys),
    ]

    norm_t = _build_norm_transforms(modalities, normalization_cfg)

    ct_strategy = normalization_cfg.get("ct", "hu_window")
    norm_t += get_ct_normalization(["ct"], ct_strategy)

    concat_t = [
        ConcatItemsd(keys=input_keys, name="input"),
    ]

    spatial_t = [
        RandSpatialCropSamplesd(
            keys=["input", "ct", "prediction_mask"],
            roi_size=patch_size,
            random_size=False,
            num_samples=num_samples,
        ),
        EnsureTyped(keys=["input", "ct", "prediction_mask"]),
    ]

    return Compose(load_t + norm_t + concat_t + spatial_t)


def get_predict_transforms(
    modalities: List[Dict],
    normalization_cfg: Dict,
) -> Compose:
    """
    Build the inference-time transform pipeline (no CT or prediction_mask).

    Parameters
    ----------
    modalities:
        Same list used during training (must match saved checkpoint).
    normalization_cfg:
        Same dict used during training.
    """
    input_keys = [m["name"] for m in modalities]

    load_t = [
        LoadImaged(keys=input_keys),
        EnsureChannelFirstd(keys=input_keys),
    ]

    norm_t = _build_norm_transforms(modalities, normalization_cfg)

    concat_t = [
        ConcatItemsd(keys=input_keys, name="input"),
        EnsureTyped(keys=["input"]),
    ]

    return Compose(load_t + norm_t + concat_t)
