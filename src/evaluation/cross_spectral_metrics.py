"""
Cross-Spectral Prediction Metrics for Spektron.

Evaluation metrics specific to the IR <-> Raman cross-spectral prediction
task (Experiment E2). All functions operate on batched tensors (B, L) and
return per-sample metrics as 1-D tensors of shape (B,).

Metrics:
    - spectral_mse: Mean squared error per spectrum
    - spectral_cosine_similarity: Cosine similarity of full spectra
    - peak_position_recall: Fraction of target peaks recovered in prediction
    - peak_intensity_correlation: Pearson r at detected peak positions
    - spectral_sid: Spectral Information Divergence (KL-based)
    - detect_peaks: Scipy-based peak detection helper
"""
import numpy as np
import torch
from scipy.signal import find_peaks
from typing import Optional, Tuple


def spectral_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample mean squared error between predicted and target spectra.

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.

    Returns:
        mse: (B,) MSE for each sample.
    """
    return ((pred - target) ** 2).mean(dim=-1)


def spectral_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample cosine similarity between predicted and target spectra.

    Cosine similarity treats each spectrum as a vector and measures the
    angle between them, ignoring global scaling differences.

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.
        eps: Small constant for numerical stability.

    Returns:
        cosine_sim: (B,) cosine similarity in [-1, 1] for each sample.
    """
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=-1, eps=eps)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=-1, eps=eps)
    return (pred_norm * target_norm).sum(dim=-1)


def detect_peaks(
    spectrum: np.ndarray,
    prominence: float = 0.05,
    distance: int = 5,
) -> np.ndarray:
    """Detect peaks in a single spectrum using scipy.signal.find_peaks.

    Args:
        spectrum: (L,) 1-D numpy array (single spectrum).
        prominence: Minimum peak prominence (relative to spectrum range).
            Peaks with prominence below this threshold are discarded.
        distance: Minimum number of points between adjacent peaks.

    Returns:
        peak_indices: (N_peaks,) integer array of peak positions.
            Empty array if no peaks are found.
    """
    # Scale prominence to the spectrum's dynamic range
    spec_range = spectrum.max() - spectrum.min()
    abs_prominence = prominence * max(spec_range, 1e-10)

    peaks, _ = find_peaks(
        spectrum,
        prominence=abs_prominence,
        distance=distance,
    )
    return peaks


def _indices_to_wavenumbers(
    indices: np.ndarray,
    wavenumber_range: Tuple[float, float],
    n_points: int,
) -> np.ndarray:
    """Convert point indices to wavenumber values.

    Args:
        indices: (N,) integer peak indices.
        wavenumber_range: (low, high) wavenumber range in cm^-1.
        n_points: Total number of spectral points.

    Returns:
        wavenumbers: (N,) wavenumber values in cm^-1.
    """
    low, high = wavenumber_range
    return low + indices * (high - low) / (n_points - 1)


def peak_position_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold_cm1: float = 10.0,
    wavenumber_range: Tuple[float, float] = (500.0, 4000.0),
    n_points: int = 2048,
    prominence: float = 0.05,
) -> torch.Tensor:
    """Fraction of target peaks whose positions are recovered in the prediction.

    For each peak in the target spectrum, check whether there is a peak
    in the predicted spectrum within `threshold_cm1` wavenumbers.

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.
        threshold_cm1: Maximum allowed position error in cm^-1 for a
            target peak to be considered "recovered".
        wavenumber_range: (low, high) wavenumber range of the spectra.
        n_points: Number of spectral points (should match L).
        prominence: Peak detection prominence threshold.

    Returns:
        recall: (B,) fraction of target peaks recovered, in [0, 1].
            Returns 1.0 for samples where the target has no peaks.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    B = pred_np.shape[0]
    recalls = np.zeros(B, dtype=np.float32)

    for i in range(B):
        target_peaks = detect_peaks(target_np[i], prominence=prominence)
        if len(target_peaks) == 0:
            recalls[i] = 1.0  # No peaks to miss
            continue

        pred_peaks = detect_peaks(pred_np[i], prominence=prominence)
        if len(pred_peaks) == 0:
            recalls[i] = 0.0  # Missed all target peaks
            continue

        # Convert to wavenumber space
        target_wn = _indices_to_wavenumbers(
            target_peaks, wavenumber_range, n_points
        )
        pred_wn = _indices_to_wavenumbers(
            pred_peaks, wavenumber_range, n_points
        )

        # For each target peak, find closest predicted peak
        recovered = 0
        for t_wn in target_wn:
            min_dist = np.min(np.abs(pred_wn - t_wn))
            if min_dist <= threshold_cm1:
                recovered += 1

        recalls[i] = recovered / len(target_peaks)

    return torch.tensor(recalls, dtype=torch.float32, device=pred.device)


def peak_intensity_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_points: int = 2048,
    prominence: float = 0.05,
) -> torch.Tensor:
    """Pearson correlation of intensities at detected peak positions.

    Detects peaks in the TARGET spectrum, then computes the Pearson
    correlation coefficient between predicted and target intensities
    at those peak positions. This measures whether the model correctly
    predicts the relative intensities of spectral features.

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.
        n_points: Number of spectral points (should match L).
        prominence: Peak detection prominence threshold.

    Returns:
        correlations: (B,) Pearson r for each sample, in [-1, 1].
            Returns 0.0 for samples with fewer than 2 target peaks
            (correlation undefined).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    B = pred_np.shape[0]
    correlations = np.zeros(B, dtype=np.float32)

    for i in range(B):
        target_peaks = detect_peaks(target_np[i], prominence=prominence)
        if len(target_peaks) < 2:
            # Pearson r is undefined for fewer than 2 points
            correlations[i] = 0.0
            continue

        pred_intensities = pred_np[i][target_peaks]
        target_intensities = target_np[i][target_peaks]

        # Handle constant arrays (std == 0)
        pred_std = pred_intensities.std()
        target_std = target_intensities.std()
        if pred_std < 1e-10 or target_std < 1e-10:
            correlations[i] = 0.0
            continue

        # Pearson correlation
        pred_centered = pred_intensities - pred_intensities.mean()
        target_centered = target_intensities - target_intensities.mean()
        r = np.sum(pred_centered * target_centered) / (
            np.sqrt(np.sum(pred_centered ** 2))
            * np.sqrt(np.sum(target_centered ** 2))
            + 1e-10
        )
        correlations[i] = float(r)

    return torch.tensor(correlations, dtype=torch.float32, device=pred.device)


def spectral_sid(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Spectral Information Divergence (SID).

    SID is a symmetric divergence measure between two spectra treated as
    probability distributions. It is the sum of the two KL divergences:
        SID(p, q) = KL(p || q) + KL(q || p)

    Both spectra are first converted to probability distributions by
    shifting to non-negative values and normalizing to sum to 1.

    Lower SID indicates more similar spectra. SID >= 0, with SID = 0
    indicating identical distributions.

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.
        eps: Small constant added before log to prevent -inf.

    Returns:
        sid: (B,) SID values for each sample (non-negative).
    """
    # Shift to non-negative (subtract per-sample minimum)
    pred_shifted = pred - pred.min(dim=-1, keepdim=True).values + eps
    target_shifted = target - target.min(dim=-1, keepdim=True).values + eps

    # Normalize to probability distributions
    p = pred_shifted / pred_shifted.sum(dim=-1, keepdim=True)
    q = target_shifted / target_shifted.sum(dim=-1, keepdim=True)

    # KL divergences (both directions)
    kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)

    return kl_pq + kl_qp


def compute_cross_spectral_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold_cm1: float = 10.0,
    wavenumber_range: Tuple[float, float] = (500.0, 4000.0),
    n_points: int = 2048,
    prominence: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute all cross-spectral metrics in one pass.

    Convenience function that runs all metrics and returns a dictionary.
    All values are per-sample tensors of shape (B,).

    Args:
        pred: (B, L) predicted spectra.
        target: (B, L) ground truth spectra.
        threshold_cm1: Peak position matching threshold in cm^-1.
        wavenumber_range: Wavenumber range of the spectra.
        n_points: Number of spectral points.
        prominence: Peak detection prominence threshold.

    Returns:
        Dictionary with keys:
            - "mse": per-sample MSE
            - "cosine_similarity": per-sample cosine similarity
            - "peak_recall": per-sample peak position recall
            - "peak_intensity_corr": per-sample peak intensity Pearson r
            - "sid": per-sample spectral information divergence
    """
    return {
        "mse": spectral_mse(pred, target),
        "cosine_similarity": spectral_cosine_similarity(pred, target),
        "peak_recall": peak_position_recall(
            pred, target,
            threshold_cm1=threshold_cm1,
            wavenumber_range=wavenumber_range,
            n_points=n_points,
            prominence=prominence,
        ),
        "peak_intensity_corr": peak_intensity_correlation(
            pred, target,
            n_points=n_points,
            prominence=prominence,
        ),
        "sid": spectral_sid(pred, target),
    }
