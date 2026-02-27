# PET Evaluation Tool

This tool implements four quantitative evaluation metrics for the
Big Cross-Modal Attenuation Correction Challenge.

---

## Implemented Metrics

1. **Whole-body SUV MAE**
2. **Brain outlier robustness score**
3. **Organ bias (SUV-mean MARE)**
4. **TAC bias (dynamic AUC MARE)**

All metrics quantify similarity between pseudo-CTAC-PET and CTAC-PET images.

---

## Installation

Requires:

- Python ≥ 3.10
- numpy
- nibabel

Install:

```bash
pip install numpy nibabel