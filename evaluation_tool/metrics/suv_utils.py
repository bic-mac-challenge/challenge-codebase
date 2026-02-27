"""
SUV Utilities

Provides conversion of PET activity concentration to SUV.
"""

import json
import nibabel as nib
import numpy as np


def load_pet_as_suv(pet_path, json_path, pet_unit="kBq"):
    """
    Convert PET activity concentration to SUV.

    Parameters
    ----------
    pet_path : str
        Path to PET NIfTI file.
    json_path : str
        Path to metadata JSON containing:
            - PatientWeight (kg)
            - InjectedRadioactivity (MBq)
    pet_unit : str
        'kBq'  → PET stored in kBq/mL (final challenge data)
        'Bq'   → PET stored in Bq/mL (training data)

    Returns
    -------
    pet_suv : ndarray
        PET volume converted to SUV.
    """

    with open(json_path, "r") as f:
        meta = json.load(f)

    weight_kg = meta["PatientWeight"]
    dose_mbq = meta["InjectedRadioactivity"]

    pet = nib.load(pet_path).get_fdata()

    if pet_unit == "kBq":
        scale = 1e3
    elif pet_unit == "Bq":
        scale = 1e6
    else:
        raise ValueError("pet_unit must be 'kBq' or 'Bq'")

    norm_factor = weight_kg / (dose_mbq * scale)

    return pet * norm_factor


def suv_sanity_check(pet_suv, body_mask, name="PET"):
    """
    Debug helper to verify SUV magnitude (~1 inside body).
    """
    mean_suv = np.mean(pet_suv[body_mask])
    print(f"[DEBUG] {name} mean SUV (body): {mean_suv:.4f}")

    if mean_suv < 0.01 or mean_suv > 50:
        print(
            "[WARNING] SUV mean appears incorrect. "
            "Check PET units or normalization."
        )