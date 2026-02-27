"""Transfer function analysis for trained D-LinOSS layers.

Computes and visualizes the discrete frequency response H(z) of each
oscillator in the D-LinOSS backbone. This reveals what spectral filters
the model has learned, providing interpretability unique to SSMs.

The per-oscillator transfer function (without skip D) is:

    H_p(z) = dt^2 * b * z / [(1 + dt*g)*z^2 - (2 + dt*g - dt^2*a)*z + 1]

where:
    a = relu(A_diag[p])   -- squared natural frequency
    g = relu(G_diag[p])   -- damping coefficient
    dt = sigmoid(steps[p]) -- discretization step

The alpha parameter is soft-clamped for CFL stability:
    S = 1 + dt * g
    alpha = dt^2 * a / S
    alpha = 1.99 * tanh(alpha / 1.99)

The 2x2 transition matrix M for each oscillator is:
    M = [[1/S,       -alpha/dt],
         [dt/S,   1 - alpha   ]]

with eigenvalues that encode the resonant frequency and damping of the
learned filter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _compute_M_matrix(
    dt: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 2x2 transition matrix entries and clamped alpha.

    Args:
        dt: (P,) activated discretization steps (after sigmoid).
        a: (P,) activated squared frequencies (after relu).
        g: (P,) activated damping coefficients (after relu).

    Returns:
        Tuple of (M_11, M_12, M_21, M_22, alpha), each shape (P,).
    """
    S = 1.0 + dt * g
    alpha = dt ** 2 * a / S
    alpha = 1.99 * np.tanh(alpha / 1.99)  # CFL soft clamp

    M_11 = 1.0 / S
    M_12 = -alpha / dt
    M_21 = dt / S
    M_22 = 1.0 - alpha

    return M_11, M_12, M_21, M_22, alpha


def _compute_eigenvalues(
    M_11: np.ndarray,
    M_12: np.ndarray,
    M_21: np.ndarray,
    M_22: np.ndarray,
) -> np.ndarray:
    """Compute eigenvalues of the 2x2 block transition matrices.

    For a 2x2 matrix [[a, b], [c, d]], the eigenvalues are:
        lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2

    Args:
        M_11, M_12, M_21, M_22: (P,) matrix entries per oscillator.

    Returns:
        eigenvalues: (P, 2) complex array of eigenvalue pairs.
    """
    trace = M_11 + M_22
    det = M_11 * M_22 - M_12 * M_21
    discriminant = trace ** 2 - 4.0 * det

    # Use complex sqrt to handle both real and complex eigenvalue cases
    sqrt_disc = np.sqrt(discriminant.astype(np.complex128))
    lambda_plus = (trace + sqrt_disc) / 2.0
    lambda_minus = (trace - sqrt_disc) / 2.0

    return np.stack([lambda_plus, lambda_minus], axis=-1)  # (P, 2)


def compute_scalar_frequency_response(
    steps: np.ndarray,
    A_diag: np.ndarray,
    G_diag: np.ndarray,
    n_freq: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the scalar frequency response |H(e^{j*omega})| per oscillator.

    Uses the 2x2 block eigenvalue formulation. The transfer function for
    oscillator p (ignoring input/output coupling B, C) is evaluated by
    substituting z = e^{j*omega} into the characteristic polynomial:

        H_p(z) = dt^2 * z / [S*z^2 - (2 + dt*g - dt^2*a_clamped)*z + 1]

    where a_clamped incorporates the CFL soft clamp on alpha.

    Args:
        steps: (P,) activated discretization steps (already sigmoid-applied).
        A_diag: (P,) activated squared frequencies (already relu-applied).
        G_diag: (P,) activated damping coefficients (already relu-applied).
        n_freq: Number of frequency points in [0, pi].

    Returns:
        omega: (n_freq,) frequency grid from 0 to pi.
        H_scalar: (P, n_freq) complex frequency response per oscillator.
    """
    P = steps.shape[0]
    omega = np.linspace(0, np.pi, n_freq, endpoint=True)  # (n_freq,)
    z = np.exp(1j * omega)  # (n_freq,) on the unit circle

    S = 1.0 + steps * G_diag  # (P,)
    alpha = steps ** 2 * A_diag / S
    alpha = 1.99 * np.tanh(alpha / 1.99)

    # Reconstruct the effective clamped frequency for the denominator.
    # From the recurrence: M_12 = -alpha/dt, and the characteristic polynomial
    # of M is z^2 - trace*z + det = 0.
    # The transfer function numerator comes from the forcing vector:
    #   F1 = dt/S * Bu,  F2 = dt^2/S * Bu
    # The scalar transfer function (unit B) is:
    #   H(z) = [dt/S * z + dt^2/S] * z / det(zI - M)
    # But we can also write it in the standard rational form using the
    # denominator polynomial from the recurrence relation.

    # Denominator: S*z^2 - (2 + dt*g - S*alpha)*z + 1
    # Note: 2 + dt*g = 1 + S, and S*alpha = dt^2 * a_clamped_raw
    # But we use the M-matrix formulation directly.
    M_11 = 1.0 / S
    M_22 = 1.0 - alpha

    trace = M_11 + M_22  # (P,)
    det = M_11 * M_22 - (-alpha / steps) * (steps / S)  # (P,)
    # det = M_11*M_22 + alpha/S = (1-alpha)/S + alpha/S = 1/S

    # Expand for broadcasting: (P, 1) vs (1, n_freq)
    trace_exp = trace[:, np.newaxis]
    det_exp = det[:, np.newaxis]
    dt_exp = steps[:, np.newaxis]
    S_exp = S[:, np.newaxis]

    z_exp = z[np.newaxis, :]  # (1, n_freq)

    # Denominator: z^2 - trace*z + det
    denom = z_exp ** 2 - trace_exp * z_exp + det_exp  # (P, n_freq)

    # Numerator: the forcing enters as F = [dt/S, dt^2/S]^T * Bu at each step.
    # Through the resolvent, the transfer function from u to x2 (position) is:
    #   H_x2(z) = (dt^2/S * z + cross_term) / (z^2 - trace*z + det)
    # The output y = C * x2, so scalar H (unit B, unit C) picks up x2.
    #
    # From the state-space: x_{k+1} = M * x_k + F * u_{k+1}
    # Taking z-transform: (zI - M) X(z) = F * z * U(z)  (one-step delay in F)
    # X(z) = (zI - M)^{-1} * F * z * U(z)
    # Y(z) = C^T X(z) = C^T (zI - M)^{-1} F z U(z)
    #
    # For scalar analysis, C = [0, 1] (reading x2 = position), F = [dt/S, dt^2/S].
    # (zI - M)^{-1} = adj(zI-M) / det(zI-M)
    # adj = [[z - M_22, M_12], [M_21, z - M_11]]
    # C^T * adj * F = [M_21, z - M_11] . [dt/S, dt^2/S]
    #               = M_21 * dt/S + (z - M_11) * dt^2/S
    #               = (dt/S)^2 + (z - 1/S) * dt^2/S
    #               = dt^2/S * [dt/(S) * 1/dt + z - 1/S]
    #   Wait, let me redo this carefully.
    #
    # C = [0, 1] selects the second row of adj:
    #   row2 of adj(zI - M) = [M_21, z - M_11]
    #
    # C^T * adj * F = M_21 * F_1 + (z - M_11) * F_2
    #              = (dt/S) * (dt/S) + (z - 1/S) * (dt^2/S)
    #              = dt^2/S^2 + dt^2/S * z - dt^2/S^2
    #              = dt^2/S * z
    #
    # Therefore: H(z) = dt^2 * z / (S * (z^2 - trace*z + det))
    #                  = dt^2 * z / (S*z^2 - S*trace*z + S*det)
    #
    # With S*det = 1 (since det = 1/S), this gives:
    # H(z) = dt^2 * z / (S*z^2 - S*trace*z + 1)
    #
    # This matches the formula in the docstring.

    numer = dt_exp ** 2 * z_exp  # (P, n_freq)
    H_scalar = numer / (S_exp * denom)  # (P, n_freq)

    return omega, H_scalar


def _extract_activated_params(
    layer_module: "torch.nn.Module",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract activated (sigmoid/relu) parameters from a DampedLayer.

    Args:
        layer_module: A DampedLayer instance (the .layer attribute of a LinOSSBlock).

    Returns:
        Tuple of (steps, A_diag, G_diag) as numpy arrays with activations applied.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        steps = torch.sigmoid(layer_module.steps).cpu().numpy()
        A_diag = F.relu(layer_module.A_diag).cpu().numpy()
        G_diag = F.relu(layer_module.G_diag).cpu().numpy()

    return steps, A_diag, G_diag


def extract_layer_responses(
    model: "torch.nn.Module",
    layer_idx: int = 0,
    direction: str = "fwd",
    n_freq: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frequency response data from a trained D-LinOSS layer.

    Navigates the model hierarchy:
        Spektron.backbone (DLinOSSBackbone)
            .layers[i] (LinOSSBlock) -- forward direction
            .bwd_layers[i] (LinOSSBlock) -- backward direction
                .layer (DampedLayer)
                    .steps, .A_diag, .G_diag

    Args:
        model: A Spektron or DLinOSSBackbone instance.
        layer_idx: Which layer to analyze (0-indexed).
        direction: 'fwd' for forward layers, 'bwd' for backward layers.
        n_freq: Number of frequency evaluation points.

    Returns:
        omega: (n_freq,) frequency grid [0, pi].
        H_magnitude: (P, n_freq) magnitude of transfer function per oscillator.
        resonant_frequencies: (P,) resonant frequency (angle of dominant eigenvalue)
            for each oscillator.
        damping_ratios: (P,) damping ratio (1 - |eigenvalue|) for each oscillator.

    Raises:
        ValueError: If the model does not have a DLinOSSBackbone or the layer
            index is out of range.
        ValueError: If direction is 'bwd' but the backbone is not bidirectional.
    """
    # Navigate to the backbone
    backbone = model
    if hasattr(model, "backbone"):
        backbone = model.backbone

    # Validate backbone type
    backbone_cls_name = type(backbone).__name__
    if backbone_cls_name != "DLinOSSBackbone":
        raise ValueError(
            f"Expected DLinOSSBackbone, got {backbone_cls_name}. "
            "Transfer function analysis only applies to D-LinOSS models."
        )

    # Select the layer list
    if direction == "fwd":
        layer_list = backbone.layers
    elif direction == "bwd":
        if not hasattr(backbone, "bwd_layers"):
            raise ValueError(
                "Backbone is not bidirectional; no backward layers available."
            )
        layer_list = backbone.bwd_layers
    else:
        raise ValueError(f"direction must be 'fwd' or 'bwd', got '{direction}'.")

    if layer_idx < 0 or layer_idx >= len(layer_list):
        raise ValueError(
            f"layer_idx={layer_idx} out of range [0, {len(layer_list) - 1}]."
        )

    # Extract the DampedLayer from LinOSSBlock
    block = layer_list[layer_idx]
    damped_layer = block.layer  # LinOSSBlock.layer is the SSM (DampedLayer)

    steps, A_diag, G_diag = _extract_activated_params(damped_layer)

    # Compute frequency response
    omega, H_scalar = compute_scalar_frequency_response(steps, A_diag, G_diag, n_freq)
    H_magnitude = np.abs(H_scalar)  # (P, n_freq)

    # Compute eigenvalues for resonant frequencies and damping
    M_11, M_12, M_21, M_22, _ = _compute_M_matrix(steps, A_diag, G_diag)
    eigenvalues = _compute_eigenvalues(M_11, M_12, M_21, M_22)  # (P, 2)

    # Resonant frequency: angle of the eigenvalue with larger imaginary part
    # (both eigenvalues are complex conjugates when the discriminant < 0,
    #  so we pick the one with positive imaginary part)
    angles = np.abs(np.angle(eigenvalues))  # (P, 2)
    resonant_frequencies = np.max(angles, axis=-1)  # (P,)

    # Damping ratio: how far inside the unit circle
    # |lambda| = 1/sqrt(S) for underdamped, so damping = 1 - |lambda|
    magnitudes = np.abs(eigenvalues)  # (P, 2)
    dominant_magnitude = np.max(magnitudes, axis=-1)  # (P,)
    damping_ratios = 1.0 - dominant_magnitude  # (P,)

    return omega, H_magnitude, resonant_frequencies, damping_ratios


def _omega_to_wavenumber(
    omega: np.ndarray,
    wavenumber_range: tuple[float, float] = (500, 4000),
    n_points: int = 2048,
) -> np.ndarray:
    """Map discrete frequency omega to wavenumber in cm^-1.

    The discrete signal has N = n_points samples spanning [nu_min, nu_max].
    Frequency omega in [0, pi] corresponds to spatial frequency k in the
    token sequence, with:
        k = omega / (2*pi)  cycles per token
        spatial period = 1/k = 2*pi/omega tokens
        wavenumber per token = delta_nu = (nu_max - nu_min) / (N - 1)

    The corresponding wavenumber modulation frequency is:
        nu_freq = k * (N - 1) * delta_nu / (something)

    More precisely: omega = 0 corresponds to DC (no variation), and
    omega = pi corresponds to Nyquist (alternating every token). The
    wavenumber associated with the omega-th frequency bin for a DFT of
    the spectral signal is:
        bin index = omega * (N / (2*pi))  (for omega in [0, pi])
        wavenumber = nu_min + bin_index * delta_nu

    But this conflates two different frequency domains. The correct mapping
    for visualization is simply linear interpolation: omega=0 maps to
    the lowest spatial frequency of the spectrum (nu_min), and omega=pi
    maps to the Nyquist rate of the token grid. We map omega to a
    wavenumber axis via the relationship:
        nu = nu_min + (omega / pi) * (nu_max - nu_min) / 2

    Actually, for a signal sampled at points nu_0, nu_0+dnu, ..., nu_0+(N-1)*dnu,
    the DFT bin k corresponds to a modulation with period N/k samples,
    i.e., wavenumber resolution dnu * N/k. The omega variable in [0, pi]
    is omega = 2*pi*k/N for k in [0, N/2], so k = omega*N/(2*pi).

    For a cleaner mapping, we just produce a wavenumber axis that linearly
    maps the omega range to the wavenumber range, which is the standard
    approach for filter response visualization over the spectral domain.

    Args:
        omega: (n_freq,) angular frequency array in [0, pi].
        wavenumber_range: (nu_min, nu_max) in cm^-1.
        n_points: Number of spectral points in the original signal.

    Returns:
        wavenumber: (n_freq,) wavenumber values in cm^-1.
    """
    nu_min, nu_max = wavenumber_range
    # omega in [0, pi] maps linearly to [nu_min, nu_max]
    # omega=0 -> DC (uniform, maps to nu_min end of spectrum)
    # omega=pi -> Nyquist (alternating every sample, maps to nu_max end)
    wavenumber = nu_min + (omega / np.pi) * (nu_max - nu_min)
    return wavenumber


def plot_filter_bank(
    omega: np.ndarray,
    H_magnitude: np.ndarray,
    wavenumber_range: tuple[float, float] = (500, 4000),
    n_points: int = 2048,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a heatmap of |H(e^{j*omega})| per oscillator.

    Oscillators are sorted by resonant frequency so the heatmap reveals
    which frequency bands are covered by learned filters.

    Args:
        omega: (n_freq,) frequency grid from compute_scalar_frequency_response.
        H_magnitude: (P, n_freq) magnitude response per oscillator.
        wavenumber_range: (nu_min, nu_max) for x-axis mapping.
        n_points: Number of spectral points (for wavenumber mapping).
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    P, n_freq = H_magnitude.shape

    # Sort oscillators by peak frequency
    peak_indices = np.argmax(H_magnitude, axis=1)
    sort_order = np.argsort(peak_indices)
    H_sorted = H_magnitude[sort_order]

    # Map omega to wavenumber
    wavenumber = _omega_to_wavenumber(omega, wavenumber_range, n_points)

    # Normalize each row for visibility (otherwise a few dominant oscillators
    # wash out the rest)
    row_max = np.max(H_sorted, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    H_normalized = H_sorted / row_max

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(
        wavenumber,
        np.arange(P),
        H_normalized,
        cmap="inferno",
        shading="auto",
        rasterized=True,
    )

    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=13)
    ax.set_ylabel("Oscillator index (sorted by resonant freq.)", fontsize=13)
    ax.set_title("D-LinOSS Learned Filter Bank", fontsize=15)

    cbar = fig.colorbar(im, ax=ax, label=r"Normalized $|H(e^{j\omega})|$", pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Add reference lines for common functional group absorptions
    reference_bands = {
        r"O-H": 3400,
        r"N-H": 3300,
        r"C-H": 2900,
        r"C$\equiv$N": 2200,
        r"C=O": 1700,
        r"C=C": 1600,
        r"C-O": 1100,
    }
    for label, nu in reference_bands.items():
        if wavenumber_range[0] <= nu <= wavenumber_range[1]:
            ax.axvline(
                nu, color="white", linestyle="--", alpha=0.5, linewidth=0.8
            )
            ax.text(
                nu + 20,
                P * 0.95,
                label,
                color="white",
                fontsize=8,
                rotation=90,
                va="top",
                ha="left",
                alpha=0.7,
            )

    ax.set_xlim(wavenumber_range)
    ax.invert_xaxis()  # IR convention: high wavenumber on left
    ax.tick_params(labelsize=11)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_pole_zero(
    model: "torch.nn.Module",
    layer_indices: Optional[list[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot eigenvalues of the M transition matrix in the complex plane.

    Shows all oscillator eigenvalues overlaid on the unit circle. Points
    inside the circle indicate damped modes; on the circle, undamped modes.
    The angle from the positive real axis gives the discrete resonant frequency.

    Args:
        model: A Spektron or DLinOSSBackbone instance.
        layer_indices: Which layers to plot. If None, plots all layers.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    # Navigate to backbone
    backbone = model
    if hasattr(model, "backbone"):
        backbone = model.backbone

    backbone_cls_name = type(backbone).__name__
    if backbone_cls_name != "DLinOSSBackbone":
        raise ValueError(
            f"Expected DLinOSSBackbone, got {backbone_cls_name}."
        )

    n_layers = len(backbone.layers)
    has_bwd = hasattr(backbone, "bwd_layers") and backbone.bidirectional

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # Collect eigenvalues per layer + direction
    all_eigs: list[tuple[str, np.ndarray]] = []
    for idx in layer_indices:
        for direction, layer_list in [("fwd", backbone.layers)] + (
            [("bwd", backbone.bwd_layers)] if has_bwd else []
        ):
            block = layer_list[idx]
            steps, A_diag, G_diag = _extract_activated_params(block.layer)
            M_11, M_12, M_21, M_22, _ = _compute_M_matrix(steps, A_diag, G_diag)
            eigs = _compute_eigenvalues(M_11, M_12, M_21, M_22)  # (P, 2)
            label = f"Layer {idx} ({direction})"
            all_eigs.append((label, eigs))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.0, alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Color map for layers
    cmap = plt.cm.tab10
    for i, (label, eigs) in enumerate(all_eigs):
        color = cmap(i % 10)
        eigs_flat = eigs.flatten()
        ax.scatter(
            eigs_flat.real,
            eigs_flat.imag,
            c=[color],
            s=12,
            alpha=0.6,
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("Re", fontsize=13)
    ax.set_ylabel("Im", fontsize=13)
    ax.set_title("D-LinOSS Eigenvalues (Pole Locations)", fontsize=15)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.tick_params(labelsize=11)

    # Add radial distance annotations
    for r in [0.9, 0.95, 0.99]:
        circle = plt.Circle(
            (0, 0), r, fill=False, linestyle=":", color="gray",
            linewidth=0.5, alpha=0.4,
        )
        ax.add_patch(circle)
        ax.text(
            r * np.cos(np.pi / 4),
            r * np.sin(np.pi / 4),
            f"|z|={r}",
            fontsize=7,
            color="gray",
            alpha=0.6,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_layer_frequency_coverage(
    model: "torch.nn.Module",
    wavenumber_range: tuple[float, float] = (500, 4000),
    n_points: int = 2048,
    n_freq: int = 1024,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-layer aggregate frequency coverage.

    For each layer, computes sum_p |H_p(e^{j*omega})| and plots the result,
    showing whether different layers specialize in different frequency ranges.

    Args:
        model: A Spektron or DLinOSSBackbone instance.
        wavenumber_range: (nu_min, nu_max) for x-axis mapping.
        n_points: Number of spectral points (for wavenumber mapping).
        n_freq: Number of frequency evaluation points.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    backbone = model
    if hasattr(model, "backbone"):
        backbone = model.backbone

    backbone_cls_name = type(backbone).__name__
    if backbone_cls_name != "DLinOSSBackbone":
        raise ValueError(
            f"Expected DLinOSSBackbone, got {backbone_cls_name}."
        )

    n_layers = len(backbone.layers)
    has_bwd = hasattr(backbone, "bwd_layers") and backbone.bidirectional

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(12, 3 * n_layers), sharex=True, squeeze=False
    )

    # Reference bands for annotation
    reference_bands = {
        r"O-H": 3400,
        r"C-H": 2900,
        r"C$\equiv$N": 2200,
        r"C=O": 1700,
        r"C=C": 1600,
        r"C-O": 1100,
    }

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]

        # Forward direction
        block_fwd = backbone.layers[layer_idx]
        steps_f, A_f, G_f = _extract_activated_params(block_fwd.layer)
        omega, H_fwd = compute_scalar_frequency_response(steps_f, A_f, G_f, n_freq)
        H_fwd_agg = np.sum(np.abs(H_fwd), axis=0)  # (n_freq,)

        wavenumber = _omega_to_wavenumber(omega, wavenumber_range, n_points)

        ax.plot(wavenumber, H_fwd_agg, color="C0", linewidth=1.2, label="Forward")

        # Backward direction
        if has_bwd:
            block_bwd = backbone.bwd_layers[layer_idx]
            steps_b, A_b, G_b = _extract_activated_params(block_bwd.layer)
            _, H_bwd = compute_scalar_frequency_response(steps_b, A_b, G_b, n_freq)
            H_bwd_agg = np.sum(np.abs(H_bwd), axis=0)
            ax.plot(
                wavenumber, H_bwd_agg, color="C1", linewidth=1.2,
                linestyle="--", label="Backward",
            )

        # Reference lines
        for label, nu in reference_bands.items():
            if wavenumber_range[0] <= nu <= wavenumber_range[1]:
                ax.axvline(nu, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

        ax.set_ylabel(r"$\sum_p |H_p|$", fontsize=11)
        ax.set_title(f"Layer {layer_idx}", fontsize=12, loc="left")
        ax.legend(fontsize=9, loc="upper right")
        ax.tick_params(labelsize=10)

    # Shared x-axis label and formatting
    axes[-1, 0].set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=13)
    axes[-1, 0].set_xlim(wavenumber_range)
    axes[-1, 0].invert_xaxis()

    # Add reference band labels to top subplot only
    for label, nu in reference_bands.items():
        if wavenumber_range[0] <= nu <= wavenumber_range[1]:
            axes[0, 0].text(
                nu, axes[0, 0].get_ylim()[1] * 0.9, label,
                fontsize=7, rotation=90, va="top", ha="right",
                color="gray", alpha=0.7,
            )

    fig.suptitle(
        "Per-Layer Frequency Coverage (D-LinOSS)", fontsize=15, y=1.01
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
