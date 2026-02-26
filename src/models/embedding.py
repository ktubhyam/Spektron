"""
Spektron: Spectral Embeddings

Two embedding modes:
1. WaveletEmbedding: DWT-based multi-scale tokenization with patching (stride=16, 127 tokens)
2. RawSpectralEmbedding: No patching, all 2048 points as tokens (for D-LinOSS backbone)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


class WavenumberPositionalEncoding(nn.Module):
    """Physics-aware positional encoding based on wavenumber values.

    Instead of learned or sinusoidal PE based on token position,
    we encode the actual wavenumber (cm⁻¹) or wavelength (nm) values,
    so the model knows WHERE in the spectrum each token comes from.
    """

    def __init__(self, d_model: int, max_len: int = 2048, base_freq: float = 10000.0):
        super().__init__()
        self.d_model = d_model

        # Standard sinusoidal encoding (as fallback / base)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(base_freq) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        # Learnable projection for actual wavenumber values
        self.wavenumber_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, wavenumbers: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, L, D) token embeddings
            wavenumbers: (B, L) or (L,) actual wavenumber values
        """
        if wavenumbers is not None:
            if wavenumbers.dim() == 1:
                wavenumbers = wavenumbers.unsqueeze(0).expand(x.size(0), -1)
            # Normalize wavenumbers to [-1, 1]
            wn_norm = wavenumbers.unsqueeze(-1)  # (B, L, 1)
            wn_pe = self.wavenumber_proj(wn_norm)  # (B, L, D)
            return x + wn_pe
        else:
            return x + self.pe[:, :x.size(1)]


class WaveletEmbedding(nn.Module):
    """Multi-scale wavelet decomposition embedding.

    Decomposes input spectrum via DWT into approximation + detail coefficients
    at multiple scales, then projects each scale to d_model and concatenates.

    This captures:
    - Approximation: baseline, broad features
    - Detail level 1: sharp peaks (highest frequency)
    - Detail level 2: medium peaks
    - Detail level 3: broad peaks, shoulders
    - Detail level 4: very broad features
    """

    def __init__(self, d_model: int = 256, n_channels: int = 2048,
                 wavelet_levels: int = 4, patch_size: int = 32,
                 stride: int = 16, dropout: float = 0.1,
                 wavelet_name: str = "db4"):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.wavelet_levels = wavelet_levels
        self.patch_size = patch_size
        self.stride = stride
        self.wavelet_name = wavelet_name

        # Compute number of patches at each scale
        # DWT halves the length at each level
        self.scale_lengths = [n_channels]
        for i in range(wavelet_levels):
            self.scale_lengths.append((self.scale_lengths[-1] + 1) // 2)

        # 1D convolution for patching at each scale
        # Scale 0: original resolution
        self.patch_embed_main = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=stride
        )

        # Multi-scale projections for wavelet coefficients
        # Each level has different length, so use adaptive pooling + projection
        self.scale_projections = nn.ModuleList()
        for level in range(wavelet_levels + 1):  # approx + details
            self.scale_projections.append(nn.Sequential(
                nn.Linear(1, d_model // (wavelet_levels + 1)),
                nn.GELU(),
            ))

        # Fusion layer: combine all scales
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // (wavelet_levels + 1) * (wavelet_levels + 1),
                      d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = WavenumberPositionalEncoding(d_model, max_len=512)

        # Domain token embeddings
        self.domain_embeddings = nn.Embedding(4, d_model)  # NIR, IR, RAMAN, UNKNOWN
        self.domain_map = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _wavelet_decompose_batch(self, x: torch.Tensor) -> dict:
        """DWT decomposition using pywt (db4, configurable levels).

        Uses actual Daubechies-4 wavelet via PyWavelets for proper
        multi-scale decomposition. Not differentiable, but the model
        learns FROM the coefficients, not THROUGH the decomposition.

        pywt output lengths for 2048 input with db4, 4 levels:
            cA4: 134, cD4: 134, cD3: 262, cD2: 517, cD1: 1027
        These differ from Haar (which always halves), but downstream
        F.interpolate(..., size=N) handles the size mismatch.
        """
        import pywt

        device = x.device
        x_np = x.detach().cpu().numpy()

        all_approx = []
        all_details = [[] for _ in range(self.wavelet_levels)]

        for i in range(x_np.shape[0]):
            coeffs = pywt.wavedec(x_np[i], self.wavelet_name, level=self.wavelet_levels)
            # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]  (pywt convention)
            all_approx.append(coeffs[0])
            for level in range(self.wavelet_levels):
                all_details[level].append(coeffs[level + 1])

        result = {
            "approx": torch.tensor(np.stack(all_approx), dtype=torch.float32, device=device),
            "details": [
                torch.tensor(np.stack(d), dtype=torch.float32, device=device)
                for d in all_details
            ],
        }
        return result

    def _resolve_domain_indices(self, domain, batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """Resolve domain argument to per-sample index tensor.

        Args:
            domain: str, list[str], torch.Tensor, or None
            batch_size: B
            device: target device

        Returns:
            (B,) long tensor of domain indices
        """
        if domain is None:
            return torch.full((batch_size,), 3, dtype=torch.long, device=device)
        if isinstance(domain, torch.Tensor):
            return domain[:batch_size].to(device)
        if isinstance(domain, str):
            idx = self.domain_map.get(domain, 3)
            return torch.full((batch_size,), idx, dtype=torch.long, device=device)
        # list of strings — per-sample domains
        indices = [self.domain_map.get(d, 3) for d in domain[:batch_size]]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, spectrum: torch.Tensor,
                domain=None,
                instrument_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            spectrum: (B, L) raw spectrum
            domain: str, list[str], torch.Tensor, or None
            instrument_id: (B,) instrument indices

        Returns:
            tokens: (B, N_tokens, d_model)
        """
        B = spectrum.size(0)

        # 1. Main patch embedding (original resolution)
        x_1d = spectrum.unsqueeze(1)  # (B, 1, L)
        main_tokens = self.patch_embed_main(x_1d)  # (B, d_model, N)
        main_tokens = main_tokens.transpose(1, 2)  # (B, N, d_model)
        N = main_tokens.size(1)

        # 2. Wavelet decomposition
        coeffs = self._wavelet_decompose_batch(spectrum)

        # 3. Project each scale and pool to match main sequence length
        scale_features = []

        # Approximation coefficients
        approx = coeffs["approx"]  # (B, L_approx)
        # Interpolate to match N tokens
        approx_interp = F.interpolate(
            approx.unsqueeze(1), size=N, mode='linear', align_corners=False
        ).squeeze(1)  # (B, N)
        approx_proj = self.scale_projections[0](approx_interp.unsqueeze(-1))
        scale_features.append(approx_proj)

        # Detail coefficients at each level
        for i, detail in enumerate(coeffs["details"]):
            detail_interp = F.interpolate(
                detail.unsqueeze(1), size=N, mode='linear', align_corners=False
            ).squeeze(1)  # (B, N)
            detail_proj = self.scale_projections[i + 1](detail_interp.unsqueeze(-1))
            scale_features.append(detail_proj)

        # 4. Concatenate main + all scale features
        all_features = torch.cat([main_tokens] + scale_features, dim=-1)

        # 5. Fusion
        tokens = self.fusion(all_features)  # (B, N, d_model)

        # 6. Positional encoding
        tokens = self.pos_enc(tokens)

        # 7. Prepend domain token (per-sample)
        domain_indices = self._resolve_domain_indices(domain, B, tokens.device)
        domain_tok = self.domain_embeddings(domain_indices).unsqueeze(1)  # (B, 1, d_model)

        # 8. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, domain_tok, tokens], dim=1)  # (B, N+2, d_model)

        return tokens


class RawSpectralEmbedding(nn.Module):
    """Raw spectral embedding WITHOUT patching.

    Each spectral point becomes a token, preserving full resolution.
    Uses a local Conv1d (stride=1) for neighborhood context extraction
    without reducing the sequence length.

    This is the key architectural choice for D-LinOSS:
    - No information loss from patching
    - D-LinOSS processes the full 2048-point sequence in O(n)
    - Each token has local spectral context via the Conv1d
    - Wavenumber positional encoding provides physics-aware position info

    Input:  (B, 2048) raw spectrum
    Output: (B, 2050, d_model) = [CLS, MODALITY, 2048 tokens]
    """

    def __init__(self, d_model: int = 256, n_channels: int = 2048,
                 kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels

        # Local feature extraction (stride=1, preserves sequence length)
        # kernel_size=15 gives each token awareness of ~15 neighboring points
        padding = kernel_size // 2
        self.local_conv = nn.Conv1d(
            1, d_model, kernel_size=kernel_size, stride=1, padding=padding
        )

        # Layer norm after projection
        self.norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_enc = WavenumberPositionalEncoding(d_model, max_len=n_channels + 2)

        # Domain token embeddings
        self.domain_embeddings = nn.Embedding(4, d_model)  # NIR, IR, RAMAN, UNKNOWN
        self.domain_map = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _resolve_domain_indices(self, domain, batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """Resolve domain argument to per-sample index tensor.

        Args:
            domain: str, list[str], torch.Tensor, or None
            batch_size: B
            device: target device

        Returns:
            (B,) long tensor of domain indices
        """
        if domain is None:
            return torch.full((batch_size,), 3, dtype=torch.long, device=device)
        if isinstance(domain, torch.Tensor):
            return domain[:batch_size].to(device)
        if isinstance(domain, str):
            idx = self.domain_map.get(domain, 3)
            return torch.full((batch_size,), idx, dtype=torch.long, device=device)
        # list of strings — per-sample domains
        indices = [self.domain_map.get(d, 3) for d in domain[:batch_size]]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, spectrum: torch.Tensor,
                domain=None,
                instrument_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            spectrum: (B, L) raw spectrum
            domain: str, list[str], or None — per-sample domain labels
            instrument_id: (B,) instrument indices (unused, for API compat)

        Returns:
            tokens: (B, L+2, d_model) where L = n_channels
        """
        B = spectrum.size(0)

        # 1. Local feature extraction (stride=1, no downsampling)
        x = spectrum.unsqueeze(1)  # (B, 1, L)
        tokens = self.local_conv(x)  # (B, d_model, L)
        tokens = tokens.transpose(1, 2)  # (B, L, d_model)

        # Ensure correct sequence length (padding might add 1)
        if tokens.size(1) > self.n_channels:
            tokens = tokens[:, :self.n_channels, :]

        tokens = self.norm(tokens)

        # 2. Positional encoding
        tokens = self.pos_enc(tokens)

        # 3. Domain token (per-sample)
        domain_indices = self._resolve_domain_indices(domain, B, tokens.device)
        domain_tok = self.domain_embeddings(domain_indices).unsqueeze(1)  # (B, 1, d_model)

        # 4. CLS token
        cls = self.cls_token.expand(B, -1, -1)

        # 5. Concatenate: [CLS, DOMAIN, tokens...]
        tokens = torch.cat([cls, domain_tok, tokens], dim=1)  # (B, L+2, d_model)

        return self.dropout(tokens)
