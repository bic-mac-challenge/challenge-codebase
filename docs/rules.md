# BIC-MAC Challenge Rules

This document describes the rules governing participation in the Big Cross-Modal Attenuation Correction (BIC-MAC) Challenge at MICCAI 2026. All registered teams must comply with these rules. Violations may result in disqualification.

---

## Training Data and External Resources

**Additional training data is allowed.** as long as it was released, publicly available and accessible to all participants without restrictions prior the start of the challenge (April 1st). Private datasets are NOT allowed. The use of public datasets must be disclosed in the submitted methodology paper. Please see [tips-and-faq.md](tips-and-faq.md) for suggested public datasets. If you are unsure whether a particular dataset fulfills the above criteria, please send an email to bic-mac-challenge@outlook.com. 

**Pretrained networks are allowed** if they were publicly available and accessible to all participants without restrictions (e.g. on GitHub, Huggingface, Zenodo, or a comparable platform) *prior to the start of the challenge* (April 1st). You may use these as initialization, feature extractors or preprocessing, but the fine-tuning data must be limited to the provided dataset.

**Any preprocessing, manual labelling or augmentation of the BIC-MAC dataset and public datasets is allowed**, as long as it does not conflict with the other rules or the BIC-MAC Data User Agreement.

---

## Submission Limits

**Validation phase (CodaBench NIfTI upload):** up to 5 submissions per day per team. The validation phase starts May 15

**Dry-run (Docker container via email):** up to 2 submissions per month per team starting May 15. The organizers will verify that the container runs, respects hardware and time constraints, and produces correctly dimensioned output. The Dry-run is performed on the validation set. 

**Final test submission (Docker container via email):** each team is permitted **one** successful Docker submission on the unseen test set. If a submission fails on some or all test cases, the team will be notified and failed predictions will be replaced with outputs from the baseline U-Net model. Each team may submit up to **twice** for the final evaluation, with only the most recent submission used for scoring.

---

## Container Requirements

Your Docker container must satisfy the following constraints, which are enforced identically in both the dry-run and final evaluation:

- **GPU:** 1× NVIDIA A40
- **CPU:** 2× Intel Xeon Gold 6346 @ 3.10 GHz
- **RAM:** 128 GB
- **Time limit:** 5 minutes per subject (wall-clock)
- **Network access:** None — the container runs with `--network none`

All model weights and dependencies must be baked into the image at build time. Any attempt to download resources at inference time will fail.

Your container must read from `/data/features/` (mounted read-only) and write `ct.nii.gz` to `/data/output/`. The output must be a NIfTI file in Hounsfield units with the same shape and affine as `features/nacpet.nii.gz`.

---

## Methodology Paper Requirement

Each team must prepare a short methodology paper describing the technical approach underlying their submission. This paper must be uploaded to a public repository (e.g., arXiv) and included with the final submission email alongside the Docker container. 


## Publication Embargo

Participating teams may publish their own results independently, subject to a **three-month embargo period** after the conclusion of MICCAI 2026. This embargo allows the organizers and the invited participant coauthers to publish the challenge summary paper first. Up to four members from each of the top five performing teams will be invited as co-authors on the challenge summary paper. Teams may opt out of inclusion in the summary paper by notifying the organizing committee via email.

---

## Metrics, Ranking, and Prizes

Performance is ranked using a rank-based aggregation across five evaluation metrics. Each metric is averaged across all test cases, submissions are ranked per metric (1 = best), and the final score is the mean of the five metric ranks. The lowest final score wins.

Please see [src/evaluation/README.md](src/evaluation/README.md) for definitions of each metric.

In the event of a tied aggregated rank, teams share the corresponding placement following Olympic-style conventions (e.g., two first places, no second place, then third place). Prize money is split equally between tied teams.

**Prizes:** 1st place: €500 · 2nd place: €300 · 3rd place: €200

---

For questions about these rules, contact the organizing team at **bic-mac-challenge@outlook.com**.
