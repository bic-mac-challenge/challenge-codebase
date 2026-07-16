"""
BIC-MAC PDF Evaluation Report

Given a dataset directory and a prediction directory (same layout as
eval_dataset.py), renders a condensed per-subject PDF report. Currently shows,
for every subject with a predicted CT and/or PET: a size-normalized
sum-of-error projection (frontal + sagittal) of the CT mu-map absolute error
and/or the PET SUV absolute error, with a body contour for anatomical
reference, plus the whole-body mu-map / SUV MAE.
"""

import argparse
import os

import numpy as np
import nibabel as nib
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_pdf import PdfPages

try:
    from .eval_dataset import validate_pred_structure
    from .metrics.ct_whole_body_mae import hu_to_mu, compute_whole_body_mu_mae, LIVER_LABEL as CT_LIVER_LABEL
    from .metrics.pet_whole_body_mae import compute_whole_body_suv_mae, LIVER_LABEL as PET_LIVER_LABEL
    from .metrics.suv_utils import compute_suv_factor
except ImportError:
    from eval_dataset import validate_pred_structure
    from metrics.ct_whole_body_mae import hu_to_mu, compute_whole_body_mu_mae, LIVER_LABEL as CT_LIVER_LABEL
    from metrics.pet_whole_body_mae import compute_whole_body_suv_mae, LIVER_LABEL as PET_LIVER_LABEL
    from metrics.suv_utils import compute_suv_factor


# (view label, axis to project out)
VIEWS = [("Frontal", 1), ("Sagittal", 0)]

# Per-modality display/plotting config, in report order.
MODALITY_CONFIG = {
    "ct": {
        "display": "CT",
        "unit": "cm⁻¹",
        "metric_label": "Mu-map MAE",
        "colorbar_label": "Σ|Δμ|, size-normalized (cm⁻¹)",
        "cmap": "inferno",
    },
    "pet": {
        "display": "PET",
        "unit": "SUV",
        "metric_label": "SUV MAE",
        "colorbar_label": "Σ|ΔSUV|, size-normalized (SUV)",
        "cmap": "viridis",
    },
}


def _eval_mask_from_body_and_liver(body_mask, liver_mask, slice_thickness_mm, exclusion_cm):
    # Mirrors the whole-body MAE metrics' masking so the visualized error
    # region matches the reported metric exactly.
    exclusion_slices = int(round((exclusion_cm * 10.0) / slice_thickness_mm))
    superior_slice = np.max(np.where(liver_mask)[2])

    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(body_mask.shape[2], superior_slice + exclusion_slices)

    exclusion_mask = np.zeros_like(body_mask, dtype=bool)
    exclusion_mask[:, :, z_min:z_max] = True

    return body_mask & (~exclusion_mask)


def _load_mu_maps(pred_ct_path, gt_ct_path, body_seg_path, organ_seg_path, exclusion_cm):
    pred_img = hu_to_mu(pred_ct_path)
    gt_img = hu_to_mu(gt_ct_path)

    body_mask = nib.load(body_seg_path).get_fdata() > 0
    liver_mask = nib.load(organ_seg_path).get_fdata() == CT_LIVER_LABEL

    slice_thickness_mm = nib.load(pred_ct_path).header.get_zooms()[2]
    eval_mask = _eval_mask_from_body_and_liver(body_mask, liver_mask, slice_thickness_mm, exclusion_cm)

    return {
        "pred": pred_img.get_fdata(),
        "gt": gt_img.get_fdata(),
        "body_mask": body_mask,
        "eval_mask": eval_mask,
        "affine": gt_img.affine,
        "zooms": np.array(gt_img.header.get_zooms()[:3]),
    }


def _load_suv_maps(pred_pet_path, gt_pet_path, body_seg_path, organ_seg_path, exclusion_cm):
    body_seg_img = nib.load(body_seg_path)
    gt_img = nib.load(gt_pet_path)
    suv_factor = compute_suv_factor(gt_img, body_seg_img)

    pred = nib.load(pred_pet_path).get_fdata() * suv_factor
    gt = gt_img.get_fdata() * suv_factor

    body_mask = body_seg_img.get_fdata() > 0
    liver_mask = nib.load(organ_seg_path).get_fdata() == PET_LIVER_LABEL

    slice_thickness_mm = nib.load(pred_pet_path).header.get_zooms()[2]
    eval_mask = _eval_mask_from_body_and_liver(body_mask, liver_mask, slice_thickness_mm, exclusion_cm)

    return {
        "pred": pred,
        "gt": gt,
        "body_mask": body_mask,
        "eval_mask": eval_mask,
        "affine": gt_img.affine,
        "zooms": np.array(gt_img.header.get_zooms()[:3]),
    }


def _sum_projection(volume, mask, axis):
    """Sum-project `volume` along `axis`, restricted to `mask`. Rays with no
    valid voxels are NaN (rendered as a neutral gray "not evaluated" gap)."""
    valid = np.any(mask, axis=axis)
    proj = np.sum(np.where(mask, volume, 0.0), axis=axis)
    return np.where(valid, proj, np.nan)


def _to_view(arr2d):
    """Orient a projected (in-plane, S-I) 2D array for display: S-I vertical, superior at top."""
    return np.flipud(arr2d.T)


def _project_modality(pred, gt, eval_mask, body_mask, affine, zooms):
    """
    Stage-1 (heavy) step: reduce a subject's 3D volumes down to small 2D
    per-view projections, so the 3D arrays can be freed before a shared
    color scale is picked across the whole report.

    The raw voxel-wise sum along a ray is an extensive quantity: a bigger
    body has more voxels along any ray, so its sum inflates relative to a
    smaller body with identical per-voxel error. We correct for this with a
    single per-subject/per-view scale factor `area_pixels / volume_voxels`
    (both counted over the same eval_mask), which is invariant to isotropic
    body-size scaling — see plan for the derivation. This turns the raw sum
    back into a size-normalized quantity in the same units as the MAE, so a
    single fixed vmin/vmax works across the whole report.
    """
    ornt = nib.io_orientation(affine)

    def canon(arr):
        return nib.orientations.apply_orientation(arr, ornt)

    pred_c = canon(pred)
    gt_c = canon(gt)
    eval_mask_c = canon(eval_mask)
    body_mask_c = canon(body_mask)
    diff = np.abs(pred_c - gt_c)

    canon_zooms = zooms[ornt[:, 0].astype(int)]
    volume_voxels = eval_mask_c.sum()

    proj = {}
    silhouette = {}
    aspect = {}
    for view_name, axis in VIEWS:
        raw_sum = _sum_projection(diff, eval_mask_c, axis)
        area_pixels = np.any(eval_mask_c, axis=axis).sum()
        scale = area_pixels / volume_voxels if volume_voxels else 0.0

        proj[view_name] = _to_view(raw_sum * scale)
        silhouette[view_name] = _to_view(np.any(body_mask_c, axis=axis))
        aspect[view_name] = canon_zooms[2] / canon_zooms[0 if axis == 1 else 1]

    return proj, silhouette, aspect


def compute_subject_projections(subject_id, ct_maps=None, ct_mae=None, pet_maps=None, pet_mae=None):
    """Build the per-modality projection record for one subject. A modality
    is included only when its maps were computed (i.e. the prediction was
    available)."""
    data = {"subject_id": subject_id}

    if ct_maps is not None:
        proj, silhouette, aspect = _project_modality(
            ct_maps["pred"], ct_maps["gt"], ct_maps["eval_mask"], ct_maps["body_mask"],
            ct_maps["affine"], ct_maps["zooms"],
        )
        data["ct"] = {"mae": ct_mae, "proj": proj, "silhouette": silhouette, "aspect": aspect}

    if pet_maps is not None:
        proj, silhouette, aspect = _project_modality(
            pet_maps["pred"], pet_maps["gt"], pet_maps["eval_mask"], pet_maps["body_mask"],
            pet_maps["affine"], pet_maps["zooms"],
        )
        data["pet"] = {"mae": pet_mae, "proj": proj, "silhouette": silhouette, "aspect": aspect}

    return data


def build_subject_figure(subject_data, vmax):
    subject_id = subject_data["subject_id"]
    modalities = [m for m in ("ct", "pet") if m in subject_data]

    fig, axes = plt.subplots(
        len(modalities), 2, figsize=(6.5, 4.0 * len(modalities)),
        constrained_layout=True, squeeze=False,
    )

    title_parts = []
    for row, modality in enumerate(modalities):
        cfg = MODALITY_CONFIG[modality]
        mdata = subject_data[modality]

        cmap = plt.get_cmap(cfg["cmap"]).copy()
        cmap.set_bad("#dddddd")

        im = None
        for col, (view_name, _axis) in enumerate(VIEWS):
            ax = axes[row][col]
            im = ax.imshow(
                mdata["proj"][view_name], cmap=cmap, vmin=0, vmax=vmax[modality],
                aspect=mdata["aspect"][view_name],
            )
            cs = ax.contour(
                mdata["silhouette"][view_name].astype(float), levels=[0.5],
                colors="white", linewidths=0.8,
            )
            cs.set(path_effects=[patheffects.withStroke(linewidth=1.6, foreground="black")])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(view_name, fontsize=10)

        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.85, label=cfg["colorbar_label"])
        title_parts.append(f"[{cfg['display']}] {cfg['metric_label']}: {mdata['mae']:.4f} {cfg['unit']}")

    fig.suptitle(f"{subject_id}\n" + "    ".join(title_parts), fontsize=12, fontweight="bold")
    return fig


def build_summary_figure(mae_values, exclusion_cm, vmax, has_ct, has_pet):
    n = len(mae_values)
    ncols = 1 if n <= 30 else (2 if n <= 60 else 3)
    nrows = -(-n // ncols)

    modality_label = " + ".join(
        MODALITY_CONFIG[m]["display"] for m in ("ct", "pet") if (m == "ct" and has_ct) or (m == "pet" and has_pet)
    )

    # Absolute (inch-based) vertical layout so spacing stays correct regardless
    # of subject count, rather than fractions of a variable-height figure.
    title_space_in = 0.55
    row_height_in = 0.18
    footer_gap_in = 0.30
    footer_line_gap_in = 0.22
    bottom_margin_in = 0.20

    footer_line_count = int(has_ct) + int(has_pet)

    list_height_in = nrows * row_height_in
    fig_width = 6.5
    fig_height = (
        title_space_in + list_height_in + footer_gap_in
        + footer_line_count * footer_line_gap_in + footer_line_gap_in
        + bottom_margin_in
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def y_from_top(inches_from_top):
        return 1 - inches_from_top / fig_height

    ax.text(
        0.02, y_from_top(0.20), f"BIC-MAC {modality_label} Report",
        fontsize=14, fontweight="bold", va="top", transform=ax.transAxes,
    )

    col_width = 1.0 / ncols
    for i, (subject_id, values) in enumerate(mae_values.items()):
        col = i // nrows
        row = i % nrows
        x = col * col_width + 0.02
        y_in = title_space_in + row * row_height_in

        parts = [f"{subject_id:<14s}"]
        if has_ct:
            parts.append(f"CT {values['ct']:.4f} cm⁻¹")
        if has_pet:
            parts.append(f"PET {values['pet']:.4f} SUV")

        ax.text(
            x, y_from_top(y_in), "  ".join(parts), fontsize=8.5,
            family="monospace", transform=ax.transAxes, va="top",
        )

    footer_start_in = title_space_in + list_height_in + footer_gap_in

    footer_lines = []
    if has_ct:
        mean_ct = float(np.mean([v["ct"] for v in mae_values.values()]))
        footer_lines.append(f"Mean CT Mu-map MAE ({n} subjects): {mean_ct:.4f} cm⁻¹  (body mask, ±{exclusion_cm:.0f} cm liver exclusion)")
    if has_pet:
        mean_pet = float(np.mean([v["pet"] for v in mae_values.values()]))
        footer_lines.append(f"Mean PET SUV MAE ({n} subjects): {mean_pet:.4f} SUV  (body mask, ±{exclusion_cm:.0f} cm liver exclusion)")

    for i, line in enumerate(footer_lines):
        ax.text(
            0.02, y_from_top(footer_start_in + i * footer_line_gap_in), line,
            fontsize=9, family="monospace", transform=ax.transAxes,
            fontweight="bold", va="top",
        )

    scale_parts = []
    if has_ct:
        scale_parts.append(f"CT 0–{vmax['ct']:.4f} cm⁻¹")
    if has_pet:
        scale_parts.append(f"PET 0–{vmax['pet']:.4f} SUV")

    note_y_in = footer_start_in + len(footer_lines) * footer_line_gap_in
    ax.text(
        0.02, y_from_top(note_y_in),
        f"Scale ({', '.join(scale_parts)}) is area/volume-normalized, comparable across body sizes.",
        fontsize=7, style="italic", color="0.35", transform=ax.transAxes, va="top",
    )

    return fig


def _pooled_vmax(per_subject, modality, override):
    if override is not None:
        return override
    pooled = np.concatenate([
        data[modality]["proj"][view_name][~np.isnan(data[modality]["proj"][view_name])]
        for data in per_subject if modality in data
        for view_name, _ in VIEWS
    ])
    return float(np.percentile(pooled, 99)) if pooled.size else 1.0


def generate_report(dataset_dir, pred_dir, output_path, subjects=None, exclusion_cm=4.0,
                     vmax_ct=None, vmax_pet=None, quiet=False):
    if subjects is None:
        subjects = sorted(
            d for d in os.listdir(pred_dir)
            if os.path.isdir(os.path.join(pred_dir, d))
        )

    if not subjects:
        raise ValueError(f"No subject folders found in {pred_dir}")

    pred_layout = validate_pred_structure(pred_dir, subjects)
    report_subjects = [s for s, (has_pet, has_ct) in pred_layout.items() if has_pet or has_ct]

    if not report_subjects:
        raise ValueError(
            "This report requires ct.nii.gz and/or pet.nii.gz predictions. "
            f"None of the subjects in {pred_dir} have either."
        )

    # validate_pred_structure guarantees has_ct / has_pet are each uniform across subjects.
    has_pet, has_ct = pred_layout[report_subjects[0]]

    if not quiet:
        modalities = " + ".join(MODALITY_CONFIG[m]["display"] for m, present in (("ct", has_ct), ("pet", has_pet)) if present)
        print(f"Building {modalities} report for {len(report_subjects)} subject(s)\n")

    # Stage 1 (heavy): reduce each subject's 3D volumes to small 2D projections.
    per_subject = []
    mae_values = {}

    for subject_id in tqdm(report_subjects, desc="Computing subjects", disable=quiet):
        subject_path = os.path.join(dataset_dir, subject_id)

        ct_maps = ct_mae = None
        if has_ct:
            pred_ct = os.path.join(pred_dir, subject_id, "ct.nii.gz")
            gt_ct = os.path.join(subject_path, "ct-label", "ct.nii.gz")
            ct_body_seg = os.path.join(subject_path, "ct-label", "body_seg.nii.gz")
            ct_organ_seg = os.path.join(subject_path, "ct-label", "organ_seg.nii.gz")

            ct_mae = float(compute_whole_body_mu_mae(pred_ct, gt_ct, ct_body_seg, ct_organ_seg, exclusion_cm))
            ct_maps = _load_mu_maps(pred_ct, gt_ct, ct_body_seg, ct_organ_seg, exclusion_cm)

        pet_maps = pet_mae = None
        if has_pet:
            pred_pet = os.path.join(pred_dir, subject_id, "pet.nii.gz")
            gt_pet = os.path.join(subject_path, "pet-label", "pet.nii.gz")
            pet_body_seg = os.path.join(subject_path, "pet-label", "body_seg.nii.gz")
            pet_organ_seg = os.path.join(subject_path, "pet-label", "organ_seg.nii.gz")

            pet_mae = float(compute_whole_body_suv_mae(pred_pet, gt_pet, pet_body_seg, pet_organ_seg, exclusion_cm))
            pet_maps = _load_suv_maps(pred_pet, gt_pet, pet_body_seg, pet_organ_seg, exclusion_cm)

        per_subject.append(compute_subject_projections(subject_id, ct_maps, ct_mae, pet_maps, pet_mae))
        mae_values[subject_id] = {"ct": ct_mae, "pet": pet_mae}
        del ct_maps, pet_maps

    # Stage 2 (cheap): pick one shared color scale per modality, then render every figure.
    vmax = {}
    if has_ct:
        vmax["ct"] = _pooled_vmax(per_subject, "ct", vmax_ct)
    if has_pet:
        vmax["pet"] = _pooled_vmax(per_subject, "pet", vmax_pet)

    summary_fig = build_summary_figure(mae_values, exclusion_cm, vmax, has_ct, has_pet)

    with PdfPages(output_path) as pdf:
        pdf.savefig(summary_fig)
        plt.close(summary_fig)
        for data in per_subject:
            fig = build_subject_figure(data, vmax)
            pdf.savefig(fig)
            plt.close(fig)

    if not quiet:
        print(f"Report written to {output_path}\n")

    return mae_values


def main():

    parser = argparse.ArgumentParser(
        description="BIC-MAC PDF Evaluation Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected directory structures (same as eval_dataset.py):\n"
            "\n"
            "  --dataset_dir                       --pred_dir\n"
            "  bic-mac-data/train/                 pred_dir/\n"
            "  ├── sub-000/                         ├── sub-000/\n"
            "  │   ├── ct-label/                    │   ├── ct.nii.gz (optional)\n"
            "  │   ├── pet-label/                    │   └── pet.nii.gz (optional)\n"
            "  │   └── ...                           └── sub-001/ ...\n"
            "  └── sub-001/ ...\n"
            "\n"
            "  Each subject folder in pred_dir may contain ct.nii.gz only, pet.nii.gz only,\n"
            "  or both, but the choice must be consistent across all subjects.\n"
        ),
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to the BIC-MAC dataset split, e.g. bic-mac-data/train")
    parser.add_argument("--pred_dir", required=True, help="Root with predicted subject folders (ct.nii.gz and/or pet.nii.gz)")
    parser.add_argument("--output", default=None, help="Output PDF path (default: <pred_dir>/report.pdf)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Explicit list of subject IDs (default: all sub-folders in pred_dir)")
    parser.add_argument("--exclusion_cm", type=float, default=4.0, help="Axial exclusion band around the superior liver slice, in cm (default: 4.0)")
    parser.add_argument("--vmax_ct", type=float, default=None, help="Fixed colorbar max for the CT sum-projection panels (default: auto, 99th percentile across the report)")
    parser.add_argument("--vmax_pet", type=float, default=None, help="Fixed colorbar max for the PET sum-projection panels (default: auto, 99th percentile across the report)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.pred_dir, "report.pdf")
    generate_report(
        args.dataset_dir, args.pred_dir, output_path, args.subjects, args.exclusion_cm,
        args.vmax_ct, args.vmax_pet,
    )


if __name__ == "__main__":
    main()
