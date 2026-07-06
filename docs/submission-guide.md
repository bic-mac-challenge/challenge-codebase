# Submission Guide

This guide explains how to submit predictions at each phase of the BIC-MAC Challenge.

---

## Phase Overview

| Phase | Period | What you submit | What we return |
|-------|--------|----------------|----------------|
| **Validation** | May 15 – Aug 15 | Zip of NIfTI predictions uploaded to Codabench | All metrics on the 4 validation subjects |
| **Dry Run** | May 15 – Aug 15 | Docker container via email | CT metrics on the 4 validation subjects (or error logs if the container failed) |
| **Final Test** | June 15 - Aug 15 | Docker container via email | Full evaluation on the unseen test set (September 1) |

Validation and Dry Run run **concurrently** throughout the challenge. Use them to iterate on your model before the final deadline. There is no limit on submissions during either phase.

---

## Phase 1: Validation — NIfTI Upload

Submit your predictions directly as NIfTI files. No Docker container needed.

### What to submit

Run your model on the 4 validation subjects (you have both `features/` and `recon/` for these) and produce predictions:

1. **Pseudo-CT** (`ct.nii.gz`) — run your model on `features/`
2. **Reconstructed PET** (`pet.nii.gz`, optional) — run the reconstruction pipeline on your pseudo-CT using the provided Docker image (see [reconstruction.md](reconstruction.md))

If you only submit `ct.nii.gz`, you will receive CT metrics only. Submitting both unlocks all four metrics.

### Output requirements

- NIfTI format (`.nii.gz`)
- Same shape and affine as `features/nacpet.nii.gz`
- CT values in Hounsfield units 

### Zip structure

```
submission.zip
├── sub-004/
│   ├── ct.nii.gz
│   └── pet.nii.gz   # optional
├── sub-009/
│   ├── ct.nii.gz
│   └── pet.nii.gz   # optional
├── sub-010/
│   ├── ct.nii.gz
│   └── pet.nii.gz   # optional
└── sub-018/
    ├── ct.nii.gz
    └── pet.nii.gz   # optional
```

Upload to the [Codabench competition page](https://www.codabench.org/competitions/12555/#/participate-tab).

---

## Phase 2: Dry Run — Container Check

The dry run verifies that your Docker container runs correctly on organizer hardware **before** the final deadline. Submit as early as possible to leave time to fix issues.

We run your container on the 4 validation subjects and return either:
- **CT metrics** — your container ran successfully and produced valid pseudo-CTs
- **Error logs** — the container failed, with details of what went wrong

See [docker-packaging.md](docker-packaging.md) for how to build and test your container locally before submitting.


### How to submit

Email **bic-mac-challenge@outlook.com** with subject line `[DRY-RUN] <TeamName>` and include:
- Team name, Docker image name and tag
- A link to your image using **one** of the options below

**Option A — Docker Hub (preferred):**
```bash
docker tag my-model:latest <dockerhub-username>/my-model:latest
docker push <dockerhub-username>/my-model:latest
```
Send us the full image name (e.g. `myteam/my-model:latest`).

**Option B — Compressed archive via file sharing:**
```bash
docker save my-model:latest | gzip > my-model.tar.gz
```
Upload `my-model.tar.gz` to Google Drive, Dropbox, or similar and share the download link.

Dry Run submissions are limited to two per month per team.

---

## Phase 3: Final Test

Submit your Docker container by **August 15, 2026**. The container does not need to be the same as the one used for the dry run — you can continue to improve your model right up to the deadline.

We will:
1. Run your container on each unseen test subject
2. Run the full reconstruction pipeline on each pseudo-CT
3. Evaluate all metrics against ground-truth CT and PET

Results and winner announcements: **September 1, 2026**.

### How to submit

Same as the dry run — email your container to **bic-mac-challenge@outlook.com** with subject `[FINAL] <TeamName>`, using Docker Hub or a compressed archive with a file sharing link. Make sure to also include a link to a short methedology paper describing your approach. This methedology paper must be uploaded to a public repository and it is a requirement to be considered eligeble for prizes. 

---

## Hardware Constraints (Phases 2 & 3)

| Resource | Specification |
|----------|--------------|
| GPU | 1× NVIDIA A40 |
| CPU | 2× Intel Xeon Gold 6346 @ 3.10 GHz |
| RAM | 128 GB |
| Wall-clock time per subject | 5 minutes |
| Network access | None (`--network none`) |

All weights and dependencies must be baked into the image. No downloads at inference time.
