"""
Evaluation Tool Package

This package implements quantitative evaluation metrics for
pseudo-CT attenuation correction challenge.

Metrics:
- Whole-body SUV MAE
- Brain outlier robustness score
- Organ bias
"""

from .eval_dataset import evaluate_dataset
from .eval_subject import evaluate_subject
from .metrics import (
    compute_whole_body_suv_mae,
    compute_brain_outlier_score,
    compute_organ_bias,
    compute_whole_body_mu_mae,
)