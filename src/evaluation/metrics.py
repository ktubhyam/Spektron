"""
SpectralFM v2: Evaluation & Metrics

Comprehensive evaluation for calibration transfer:
- R², RMSEP, RPD, bias, slope
- Sample efficiency curves
- Cross-instrument analysis
- Uncertainty calibration
- Latent space visualization
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from scipy import stats


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_uncertainty: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive regression metrics.

    Args:
        y_true: (N,) ground truth
        y_pred: (N,) predictions
        y_uncertainty: (N,) predicted uncertainty (std)

    Returns:
        dict of metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    metrics = {}

    # Core metrics
    metrics["r2"] = r2_score(y_true, y_pred)
    metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmsep"] = metrics["rmse"]  # Same thing, chemometric convention

    # Ratio of Performance to Deviation (RPD)
    std_true = np.std(y_true)
    metrics["rpd"] = std_true / (metrics["rmse"] + 1e-10)

    # Bias and slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    metrics["slope"] = slope
    metrics["intercept"] = intercept
    metrics["bias"] = np.mean(y_pred - y_true)

    # Max absolute error
    metrics["max_ae"] = np.max(np.abs(y_true - y_pred))

    # Uncertainty calibration (if provided)
    if y_uncertainty is not None:
        y_uncertainty = y_uncertainty.flatten()
        # Expected calibration: ~68% of predictions within 1σ
        within_1sigma = np.mean(np.abs(y_true - y_pred) <= y_uncertainty)
        within_2sigma = np.mean(np.abs(y_true - y_pred) <= 2 * y_uncertainty)
        metrics["coverage_1sigma"] = within_1sigma
        metrics["coverage_2sigma"] = within_2sigma
        metrics["mean_uncertainty"] = np.mean(y_uncertainty)

    return metrics


def sample_efficiency_curve(model, source_spectra: np.ndarray,
                            target_spectra: np.ndarray,
                            targets: np.ndarray,
                            n_samples_list: List[int] = [5, 10, 20, 30, 50, 100],
                            n_repeats: int = 5,
                            seed: int = 42) -> Dict:
    """Compute transfer performance as a function of N transfer samples.

    This is the key figure for the paper: how many samples do we need?

    Returns:
        dict with n_samples → {mean_r2, std_r2, mean_rmse, std_rmse}
    """
    results = {}

    for n in n_samples_list:
        r2s = []
        rmses = []

        for repeat in range(n_repeats):
            rng = np.random.RandomState(seed + repeat)
            idx = rng.choice(len(targets), min(n, len(targets)), replace=False)

            # This would call fine-tune + evaluate
            # Placeholder for now
            r2s.append(0.0)
            rmses.append(0.0)

        results[n] = {
            "mean_r2": np.mean(r2s),
            "std_r2": np.std(r2s),
            "mean_rmse": np.mean(rmses),
            "std_rmse": np.std(rmses),
        }

    return results


# ============================================================
# Baseline Methods
# ============================================================

class PLSBaseline:
    """PLS regression baseline for comparison."""

    def __init__(self, n_components: int = 10):
        self.model = PLSRegression(n_components=n_components)

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)


class PiecewiseDirectStandardization:
    """PDS: Classical calibration transfer baseline.

    Maps each wavelength in the source to a window of wavelengths in the target.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.F = None  # Transfer matrix

    def fit(self, source: np.ndarray, target: np.ndarray):
        """Fit PDS transfer matrix.

        Args:
            source: (N, P) source instrument spectra (transfer standards)
            target: (N, P) target instrument spectra (same samples)
        """
        N, P = source.shape
        w = self.window_size
        half_w = w // 2

        self.F = np.zeros((P, P))

        for j in range(P):
            # Window around wavelength j
            start = max(0, j - half_w)
            end = min(P, j + half_w + 1)

            # Local PLS or least squares
            X_local = source[:, start:end]
            y_local = target[:, j]

            # Ridge regression for stability
            lam = 1e-4
            XtX = X_local.T @ X_local + lam * np.eye(end - start)
            Xty = X_local.T @ y_local
            coeffs = np.linalg.solve(XtX, Xty)

            self.F[start:end, j] = coeffs

        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transfer source spectra using fitted matrix."""
        return source @ self.F

    def fit_transform(self, source_train: np.ndarray,
                      target_train: np.ndarray,
                      source_test: np.ndarray) -> np.ndarray:
        self.fit(source_train, target_train)
        return self.transform(source_test)


class SlopeInterceptCorrection:
    """Simple slope/intercept (bias) correction baseline."""

    def fit_transform(self, source_train: np.ndarray,
                      target_train: np.ndarray,
                      source_test: np.ndarray) -> np.ndarray:
        # Compute per-wavelength slope and intercept
        n_wl = source_train.shape[1]
        slopes = np.zeros(n_wl)
        intercepts = np.zeros(n_wl)

        for i in range(n_wl):
            slope, intercept, _, _, _ = stats.linregress(
                source_train[:, i], target_train[:, i]
            )
            slopes[i] = slope
            intercepts[i] = intercept

        return source_test * slopes + intercepts


def run_baseline_comparison(source_train: np.ndarray, target_train: np.ndarray,
                            source_test: np.ndarray, target_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            n_pls_components: int = 10) -> Dict:
    """Run all baseline methods and compare.

    Returns:
        dict mapping method_name → metrics
    """
    results = {}

    # 1. Direct PLS (no transfer)
    pls = PLSBaseline(n_pls_components)
    pred = pls.fit_predict(source_train, y_train, source_test)
    results["PLS_no_transfer"] = compute_metrics(y_test, pred)

    # 2. PLS on target directly (upper bound)
    pls_target = PLSBaseline(n_pls_components)
    pred = pls_target.fit_predict(target_train, y_train, target_test)
    results["PLS_target_direct"] = compute_metrics(y_test, pred)

    # 3. PDS + PLS
    pds = PiecewiseDirectStandardization(window_size=5)
    transferred = pds.fit_transform(source_train, target_train, source_test)
    pred = pls.fit_predict(transferred, y_train, source_test)
    results["PDS_PLS"] = compute_metrics(y_test, pred)

    # 4. Slope/Intercept + PLS
    sic = SlopeInterceptCorrection()
    transferred = sic.fit_transform(source_train, target_train, source_test)
    pred = pls.fit_predict(transferred, y_train, source_test)
    results["SIC_PLS"] = compute_metrics(y_test, pred)

    return results
