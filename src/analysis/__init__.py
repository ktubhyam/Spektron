"""Spektron: Analysis tools for trained models.

Provides transfer function computation and visualization for D-LinOSS layers,
enabling interpretability analysis of learned spectral filters.
"""

from .transfer_function import (
    compute_scalar_frequency_response,
    extract_layer_responses,
    plot_filter_bank,
    plot_pole_zero,
    plot_layer_frequency_coverage,
)

__all__ = [
    "compute_scalar_frequency_response",
    "extract_layer_responses",
    "plot_filter_bank",
    "plot_pole_zero",
    "plot_layer_frequency_coverage",
]
