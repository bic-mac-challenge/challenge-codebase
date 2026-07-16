# BIC-MAC Challenge Codebase

> [!WARNING]
> A bug in the SUV MAE metric was discovered and has now been fixed (please see Update below from July 16). The function has also been fixed in the automatic CodaBench validation, so we advise everyone that has made PET submissions to press "resubmit" to get the updated and correct metrics.

**Big Cross-Modal Attenuation Correction** — synthesize pseudo-CT from multi-modal PET/MRI input to enable CT-less PET reconstruction.

🏆 [Challenge website](https://bic-mac-challenge.github.io/)
 | 🗂️ [Dataset](https://huggingface.co/datasets/DEPICT-RH/BIC-MAC)
 | 🏆 [CodaBench submission & leaderboard](https://www.codabench.org/competitions/12555/)

---
## Updates
- **July 16, 2026: [BUG FIX]**: The Whole-body SUV MAE metric excluded the wrong end of the liver: PET's z-axis runs in the opposite direction from CT's, so the superior-most liver slice was located incorrectly, shifting the ±4 cm exclusion band off the top of the liver. Please `git pull` for the corrected evaluation logic. Please see [PR #28](https://github.com/bic-mac-challenge/challenge-codebase/pull/28) for details.
- **July 8, 2026: [CHANGE]**: The fifth metric, TAC-Bias, has been removed from the final evaluation. Final submissions are scored on the four metrics defined in [Evaluation](#evaluation-srcevaluation). Please see [PR #23](https://github.com/bic-mac-challenge/challenge-codebase/pull/23) for details.
- **July 6, 2026: [BUG FIX]**: DRY-RUN and FINAL submission instructions listed a wrong email. All DRY-RUN and FINAL submissions should be sent to bic-mac-challenge@outlook.com. 
- **May 15, 2026: [ANNOUNCEMENT]**: NIfTI validation and leaderboards now live - see submission instructions at [https://www.codabench.org/competitions/12555](https://www.codabench.org/competitions/12555)
- **April 8, 2026 (13:00 GMT+2)**: [BUG FIX] A bug in the Carney HU-to-MU conversion formula has been fixed. Please make sure to run `docker pull ghcr.io/bic-mac-challenge/recon:latest` and `git pull` for updated recon and evaluation logic. The 8 ground truth PET train images have been updated on huggingface. 
- **April 7, 2026: [NEW DATA POLICY]**: The use of public datasets for pretraining and other use-cases is now allowed under certain conditions. Please see [docs/rules.md](docs/rules.md) for details. 
## Table of Contents

- [Overview](#overview)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Dataset Structure](#dataset-structure)
- [Pseudo-CT Baseline](#pseudo-ct-baseline-srcbaseline)
- [Reconstruction](#reconstruction-srcrecon)
- [Evaluation](#evaluation-srcevaluation)
- [Submission](#submission)

---

<a id="overview"></a>

## 🧠 Overview

Your algorithm receives the files under `features/` for each subject and must output a pseudo-CT volume as a NIfTI file in Hounsfield units (HU). Predictions are evaluated two ways (see [Evaluation](#evaluation-srcevaluation) for metric definitions):

1. **CT accuracy** — Predicted pseudo-CT is compared directly against the ground-truth CT
2. **PET accuracy** — Predicted pseudo-CT is fed into the reconstruction pipeline to produce an attenuation-corrected PET image, which is then compared against the ground-truth PET

Note that no PET reconstruction experience is needed to participate in the challenge, and the main purpose of the reconstruction is to enable clinically meaningful metrics. 

---

<a id="documentation"></a>

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Rules](docs/rules.md) | All rules for participating teams including training data policy, and pretraining policy |
| [Data Background](docs/data-background.md) | Details dat acquisition, preprocessing, and alignment of modalities |
| [Reconstruction Pipeline](docs/reconstruction.md) | How the pseudo-CT is turned into an AC-PET image; how to run it locally |
| [Submission Guide](docs/submission-guide.md) | Validation, dry-run, and final submission phases explained |
| [Docker Packaging](docs/docker-packaging.md) | How to containerize your model, with baseline as a worked example |
| [Tips & FAQ](docs/tips-and-faq.md) | Common questions, pitfalls, and practical advice |

---

<a id="getting-started"></a>

## 🚀 Getting Started

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv), Docker

```bash
uv sync
```

The `src/` directory contains three components:

```
src/
├── baseline/       # Baseline pseudo-CT model (patch-based 3D UNet)
├── evaluation/     # Metric defintinitions and scripts
└── recon/          # PET reconstruction script (and Docker)
```

---

<a id="dataset-structure"></a>

## 🗂️ Dataset structure

The dataset comprises 99 subject-unique studies, with 20 reserved for testing and the remaining 79 available on huggingface and split as follows:

| Split | Subjects | Contents |
|-------|----------|----------|
| `train/` (full) | 8 | `features/` + `ct-label/` + `recon/` + `pet-label/` |
| `train/` (no recon) | 67 | `features/` + `ct-label/` |
| `val/` | 4 | `features/` + `recon/` |

All train subjects have CT labels, but due to the size of the sinograms, only 8 include the `recon/` and `pet-label/` folders needed for closed loop reconstruction. Validation subjects have sinogram `recon/` data but no labels — submit predicted pseudo-CTs and reconstructed PETs to Codabench to get live leaderboard metrics during the challenge. The train subjects with `recon/` data are: `sub-000, sub-001, sub-002, sub-005, sub-006, sub-008, sub-013, sub-014`.


All images except those in `pet-label` are resampled to the label CT image (tensor size: 512x512x531, voxel size 1.52x1.52,2.00mm^3). NIfTI images are structured in four folders per subject. 
- `features/` All the files you can use as input to your pseudo-CT model during inference.
- `ct-label/` The CT target (`ct.nii.gz`) and segmentations for evaluation. 
- `pet-label/` The PET target (`pet.nii.gz`) and segmentations for evaluation. 
- `recon/` Sinograms and metadata for PET reconstructions. 


```
train/
└── sub-000/
    ├── features/                          # generative model inputs
    │   ├── nacpet.nii.gz                  # non-attenuation-corrected PET. 
    │   ├── topogram.nii.gz                # 2D scout X-ray
    │   ├── mri_chunk_{0-3}_{in/out}_phase.nii.gz    # MRI chunk (0-3), in- and out-phase
    │   ├── mri_combined_{in/out}_phase.nii.gz  # stitched whole-body MRI, in- and out-phase
    │   ├── mri_face_mask.nii.gz           # binary anonymization mask
    │   └── metadata.json                  # {sex, age, height, weight}
    ├── ct-label/                          # ground-truth CT
    │   ├── ct.nii.gz                      # CT in Hounsfield Units (reference)
    │   ├── body_seg.nii.gz                # TotalSegmentator body seg.
    │   ├── organ_seg.nii.gz               # TotalSegmentator organ seg.
    │   └── prediction_mask.nii.gz         # Within-body voxels (face excluded).
    ├── recon/                             # sinogram data for reconstruction
    │   ├── mult_nac_rd85.hs/.s            # multiplicative sinogram (NAC)
    │   ├── add_nac_rd85.hs/.s             # additive sinogram (NAC)
    │   ├── prompts_rd85.hs/.s             # raw sinogram 
    │   ├── offset.json                    # bed position and gantry offset
    │   ├── ct_face_and_bed.nii.gz         # inverse of prediction_mask.nii.gz
    │   └── face_and_bed_mask.nii.gz       # binary face + scanner bed mask
    └── pet-label/                         # ground-truth PET
        ├── pet.nii.gz                     # CT-attenuation-corrected PET (reference)
        ├── body_seg.nii.gz                # TotalSegmentator body seg. in PET space
        ├── tissue_seg.nii.gz              # TotalSegmentator tissue seg. in PET space
        └── organ_seg.nii.gz               # TotalSegmentator organ seg. in PET space
```

---

<a id="pseudo-ct-baseline-srcbaseline"></a>

## 📦 Pseudo-CT Baseline (`src/baseline/`)

A simple patch-based 3D UNet that predicts pseudo-CT from NAC-PET only. 

**Python usage:**

```bash
# Example:
python src/baseline/predict.py --features_dir data/sub-000/features/ --output_ct results/sub-000/ct.nii.gz
```

**Docker usage:**

```bash
docker pull ghcr.io/bic-mac-challenge/baseline:latest

docker run --rm \
  --gpus all \
  -v /path/to/sub-000/features:/data/features:ro \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/baseline:latest
```

The predicted CT is written to `/data/output/ct.nii.gz`. All weights and dependencies are baked into the image, and the same is expected for your final docker image submission.

You can re-train the baseline by running `train.py` and containerize it by running `docker build -t my-baseline .` (from inside the `src/baseline` folder)

---

<a id="reconstruction-srcrecon"></a>

## ⚙️ Reconstruction (`src/recon/`)

Converts a CT (ground-truth or pseudo-CT) and PET sinograms into an attenuation-corrected PET image using [STIR](http://stir.sourceforge.net/). See [docs/reconstruction.md](docs/reconstruction.md) for pipeline details and local usage instructions.

```bash
docker pull ghcr.io/bic-mac-challenge/recon:latest

docker run --rm \
  -v /path/to/sub-000/recon:/data/recon \
  -v /path/to/ct.nii.gz:/data/ct/ct.nii.gz \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
```

The reconstructed PET is written to `/data/output/pet.nii.gz`.

> [!WARNING]
> Running reconstruction requires **~20 GB of RAM** and takes **20–120 minutes** depending on CPU speed. The `intermediates/` folder uses **~50 GB** of additional disk space — consider deleting it after a successful reconstruction.

---

<a id="evaluation-srcevaluation"></a>

## 📊 Evaluation (`src/evaluation/`)

> [!IMPORTANT]
> The fifth metric `TAC Bias` has been removed from the final evaluation, please see announcement and [PR #23](https://github.com/bic-mac-challenge/challenge-codebase/pull/23). 

Four metrics compare predicted PET and CT outputs against the ground truth:

| Metric | Modality | Description | Region |
|--------|------|-------------|--------|
| Whole-body SUV MAE | `PET` | Mean absolute error in standardised uptake value (SUV = activity × weight / total dose) | Body mask, excluding ±4 cm around liver |
| Brain Outlier Score | `PET` | AUC of fraction of brain voxels within relative error thresholds (5%, 10%, 15%) [Ladefoged 2017 (fig. 8)](https://www.sciencedirect.com/science/article/pii/S1053811916307170)| Brain |
| Organ Bias | `PET` | Mean absolute relative error of mean SUV in 8 organs: brain, liver, spleen, heart, pancreas, muscle, adipose, extremities | TotalSegmentator organ labels |
| CT MU MAE | `CT` | Mean absolute error of attenuation coefficients (μ at 511 keV) between predicted and ground-truth CT after HU→μ conversion | Body mask, excluding ±4 cm axial slices at top of liver|

**Evaluate a single subject:**

```bash
python src/evaluation/eval_subject.py \
  --subject_dir <subject_dir> \
  --pred_pet <pred_pet.nii.gz> \
  --pred_ct <pred_ct.nii.gz>
```

`--pred_pet` and `--pred_ct` are both optional — omit either to skip PET or CT metrics.
Note: Brain Outlier Score is a dataset-level metric and requires multiple subjects (see below).

**Evaluate multiple subjects:**

```bash
python src/evaluation/eval_dataset.py \
  --dataset_dir <dataset_dir> \
  --pred_dir <predictions_dir>
```

`<predictions_dir>` must contain one sub-folder per subject with consistent contents — either CT only, PET only, or both:

```
predictions_dir/
├── sub-000/
│   ├── ct.nii.gz        
│   └── pet.nii.gz       
├── sub-001/
│   ├── ct.nii.gz
│   └── pet.nii.gz
└── ...
```

---

<a id="submission"></a>

## 📬 Submission

| Phase | What you submit |
|-------|----------------|
| **Validation** | Zip of NIfTI predictions for the 4 `val/` subjects (CT + optional PET) uploaded to Codabench — you run prediction and reconstruction locally using the provided data and tools |
| **Dry Run** | Docker container emailed to us — we run it on the 4 `val/` subjects on our hardware and return CT metrics, or error logs if the container failed |
| **Final Test** | Docker container emailed to us — we run prediction, reconstruction, and full evaluation on the unseen test set |

Validation and Dry Run open May 15. See [docs/submission-guide.md](docs/submission-guide.md) for full instructions, and [docs/docker-packaging.md](docs/docker-packaging.md) for how to build and test your container.


