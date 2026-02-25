"""
SpectralFM v2: Transfer Heads

1. FNO (Fourier Neural Operator) — resolution-independent spectral transfer
2. VIB (Variational Information Bottleneck) — disentangle chemistry vs instrument
3. MLP baseline head for comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================
# Fourier Neural Operator Transfer Head
# ============================================================

class SpectralConv1d(nn.Module):
    """1D Spectral Convolution (core of FNO).

    Operates in Fourier domain: learns to mix Fourier modes.
    This is natural for spectroscopy where features ARE frequency components.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        # Complex-valued weight for Fourier domain
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def _complex_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication using real tensors (last dim = 2 for real/imag)."""
        # a: (B, C_in, modes, 2), b: (C_in, C_out, modes, 2)
        real = a[..., 0].unsqueeze(2) * b[..., 0].unsqueeze(0) - \
               a[..., 1].unsqueeze(2) * b[..., 1].unsqueeze(0)
        imag = a[..., 0].unsqueeze(2) * b[..., 1].unsqueeze(0) + \
               a[..., 1].unsqueeze(2) * b[..., 0].unsqueeze(0)
        # Sum over in_channels
        return torch.stack([real.sum(1), imag.sum(1)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) real-valued input
        Returns:
            out: (B, C_out, L) real-valued output
        """
        B, C, L = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1) complex
        x_ft_real = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (B, C, L//2+1, 2)

        # Multiply with weight in Fourier domain (only first `modes` modes)
        n_modes = min(self.modes, x_ft_real.size(2))
        out_ft = self._complex_mul(
            x_ft_real[:, :, :n_modes],
            self.weight[:, :, :n_modes]
        )  # (B, C_out, n_modes, 2)

        # Pad remaining modes with zeros
        out_ft_padded = torch.zeros(
            B, self.out_channels, L // 2 + 1, 2,
            device=x.device, dtype=x.dtype
        )
        out_ft_padded[:, :, :n_modes] = out_ft

        # iFFT
        out_ft_complex = torch.complex(out_ft_padded[..., 0], out_ft_padded[..., 1])
        return torch.fft.irfft(out_ft_complex, n=L, dim=-1)


class FNOBlock(nn.Module):
    """Single FNO block: spectral convolution + pointwise transform + residual."""

    def __init__(self, width: int, modes: int, activation: str = "gelu"):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral path + pointwise path (residual)
        x1 = self.spectral_conv(x)
        x2 = self.pointwise(x)
        return self.act(self.norm(x1 + x2))


class FNOTransferHead(nn.Module):
    """Fourier Neural Operator for calibration transfer.

    Maps latent representations to transferred spectra in a
    RESOLUTION-INDEPENDENT manner — the FNO operates on functions,
    not fixed-size vectors.

    Also accepts instrument parameters for conditioned transfer.
    """

    def __init__(self, d_latent: int = 128, out_channels: int = 2048,
                 width: int = 64, modes: int = 32, n_layers: int = 4,
                 n_instrument_params: int = 8):
        super().__init__()
        self.width = width
        self.out_channels = out_channels

        # Project latent to initial field
        self.lift = nn.Linear(d_latent, width * 64)  # (B, width, 64)
        self.lift_shape = (width, 64)

        # Instrument conditioning
        self.inst_proj = nn.Sequential(
            nn.Linear(n_instrument_params, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(width, modes) for _ in range(n_layers)
        ])

        # Project to output
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv1d(width * 2, 1, 1),
        )

    def forward(self, z: torch.Tensor,
                instrument_params: Optional[torch.Tensor] = None,
                output_length: int = None) -> torch.Tensor:
        """
        Args:
            z: (B, d_latent) latent representation
            instrument_params: (B, n_params) target instrument characteristics
            output_length: desired output length (for resolution independence)
        Returns:
            spectrum: (B, output_length or out_channels)
        """
        B = z.size(0)
        out_len = output_length or self.out_channels

        # Lift to initial field
        x = self.lift(z).view(B, *self.lift_shape)  # (B, width, 64)

        # Interpolate to desired output length
        x = F.interpolate(x, size=out_len, mode='linear', align_corners=False)

        # Condition on instrument parameters
        if instrument_params is not None:
            inst_cond = self.inst_proj(instrument_params)  # (B, width)
            x = x + inst_cond.unsqueeze(-1)

        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)

        # Project to 1D output
        spectrum = self.project(x).squeeze(1)  # (B, out_len)

        return spectrum


# ============================================================
# Variational Information Bottleneck (VIB) Disentanglement
# ============================================================

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    Forward: identity. Backward: negate and scale gradients.
    This makes z_chem learn to NOT encode instrument info,
    while the classifier still trains normally.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha) if isinstance(alpha, torch.Tensor) else alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class VIBHead(nn.Module):
    """Variational Information Bottleneck for disentangling
    chemistry-invariant vs instrument-specific features.

    z_chem: should capture chemical composition (transferable)
    z_inst: should capture instrument characteristics (discardable)

    Objective:
        max I(z_chem; Y) - β·I(z_chem; X) + γ·I(z_inst; D_instrument)
    """

    def __init__(self, d_input: int = 256, z_chem_dim: int = 128,
                 z_inst_dim: int = 64, beta: float = 1e-3):
        super().__init__()
        self.z_chem_dim = z_chem_dim
        self.z_inst_dim = z_inst_dim
        self.beta = beta

        # Chemistry-invariant encoder → mean, logvar
        self.chem_mean = nn.Linear(d_input, z_chem_dim)
        self.chem_logvar = nn.Linear(d_input, z_chem_dim)

        # Instrument-specific encoder → mean, logvar
        self.inst_mean = nn.Linear(d_input, z_inst_dim)
        self.inst_logvar = nn.Linear(d_input, z_inst_dim)

        # Instrument classifier (for z_inst — should predict instrument)
        self.inst_classifier = nn.Sequential(
            nn.Linear(z_inst_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),  # max 16 instruments
        )

        # Adversarial classifier on z_chem
        # (z_chem should NOT predict instrument → gradient reversal)
        self.chem_inst_classifier = nn.Sequential(
            nn.Linear(z_chem_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
        )

    def reparameterize(self, mean: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def kl_divergence(self, mean: torch.Tensor,
                      logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)) where p(z) = N(0, I)."""
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(self, h: torch.Tensor) -> dict:
        """
        Args:
            h: (B, d_input) encoder output (e.g., CLS token)
        Returns:
            dict with z_chem, z_inst, kl losses, classifier logits
        """
        # Chemistry-invariant
        chem_mean = self.chem_mean(h)
        chem_logvar = self.chem_logvar(h)
        z_chem = self.reparameterize(chem_mean, chem_logvar)

        # Instrument-specific
        inst_mean = self.inst_mean(h)
        inst_logvar = self.inst_logvar(h)
        z_inst = self.reparameterize(inst_mean, inst_logvar)

        # KL divergences
        kl_chem = self.kl_divergence(chem_mean, chem_logvar)
        kl_inst = self.kl_divergence(inst_mean, inst_logvar)

        # Classifier outputs
        inst_from_inst = self.inst_classifier(z_inst)
        # Gradient reversal: classifier trains to predict instrument from z_chem,
        # but reversed gradients make z_chem learn to NOT encode instrument info
        z_chem_reversed = GradientReversal.apply(z_chem, 1.0)
        inst_from_chem = self.chem_inst_classifier(z_chem_reversed)

        return {
            "z_chem": z_chem,
            "z_inst": z_inst,
            "chem_mean": chem_mean,
            "chem_logvar": chem_logvar,
            "inst_mean": inst_mean,
            "inst_logvar": inst_logvar,
            "kl_chem": kl_chem,
            "kl_inst": kl_inst,
            "kl_loss": kl_chem + kl_inst,  # Combined KL loss for convenience
            "inst_from_inst": inst_from_inst,
            "inst_from_chem": inst_from_chem,
        }


# ============================================================
# Prediction Heads
# ============================================================

class RegressionHead(nn.Module):
    """Simple MLP for property prediction (baseline comparison)."""

    def __init__(self, d_input: int = 128, n_targets: int = 1,
                 d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReconstructionHead(nn.Module):
    """Spectrum reconstruction head for MSRP pretraining."""

    def __init__(self, d_input: int = 256, n_patches: int = 127,
                 patch_size: int = 32, d_hidden: int = 512):
        super().__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_patches, d_model) token representations
        Returns:
            reconstructed: (B, N_patches, patch_size) per-patch reconstruction
        """
        return self.proj(x)
