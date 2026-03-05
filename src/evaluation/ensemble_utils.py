"""Lightweight ensemble aggregation for uncertainty quantification.

Provides functions to aggregate predictions across multiple models (ensemble)
and compute confidence intervals via ensemble variance.
"""
from __future__ import annotations

import numpy as np


def ensemble_mean_std(
    predictions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate ensemble predictions (mean and std across models).

    Args:
        predictions: List of (N, D) predictions from K models

    Returns:
        mean: (N, D) ensemble mean
        std:  (N, D) ensemble standard deviation
    """
    pred_stack = np.stack(predictions, axis=0)  # (K, N, D)
    mean = pred_stack.mean(axis=0)  # (N, D)
    std = pred_stack.std(axis=0)    # (N, D)
    return mean, std


def ensemble_confidence_interval(
    predictions: list[np.ndarray],
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute confidence intervals from ensemble predictions.

    Args:
        predictions: List of (N, D) predictions from K models
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        lower: (N, D) lower bound
        upper: (N, D) upper bound
    """
    pred_stack = np.stack(predictions, axis=0)  # (K, N, D)
    alpha = (1 - confidence) / 2
    lower = np.percentile(pred_stack, alpha * 100, axis=0)
    upper = np.percentile(pred_stack, (1 - alpha) * 100, axis=0)
    return lower, upper
