"""
Spektron: Mamba Backbone (Selective State Space Model)

Pure PyTorch implementation of the core Mamba architecture for portability.
For production, swap with mamba-ssm CUDA kernels for 5-10x speedup.

Key innovation: Selective mechanism filters noise while propagating chemical signals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange


class SelectiveSSM(nn.Module):
    """Core Selective State Space Model.

    The key insight from Mamba: make the SSM parameters (Δ, B, C) INPUT-DEPENDENT.
    This allows the model to selectively propagate or forget information based
    on the input content — crucial for spectroscopy where noise should be
    forgotten but chemical signals should propagate.

    Discrete SSM equations:
        h_t = Ā h_{t-1} + B̄ x_t
        y_t = C h_t + D x_t

    Where Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB ≈ ΔB (simplified)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input projection: x -> (z, x_inner) where z is gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        # SSM parameters: input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Δ (discretization step) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize Δ bias for stability
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A matrix: structured initialization (diagonal, negative)
        # HiPPO-inspired: A_n = -(n+1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # Store in log space for stability

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_scan(self, x: torch.Tensor, dt: torch.Tensor,
                  A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  D: torch.Tensor) -> torch.Tensor:
        """Sequential SSM scan (for clarity; use parallel scan in production).

        Args:
            x: (B, L, D_inner) input
            dt: (B, L, D_inner) discretization steps
            A: (D_inner, N) state matrix
            B: (B, L, N) input matrix
            C: (B, L, N) output matrix
            D: (D_inner,) skip connection

        Returns:
            y: (B, L, D_inner) output
        """
        batch, length, d_inner = x.shape
        n = A.shape[1]

        # Discretize: Ā = exp(Δ * A)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D_inner, N) broadcast

        # Sequential scan
        h = torch.zeros(batch, d_inner, n, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(length):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)
        y = y + x * D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, D = x.shape

        # Input projection → (x_inner, z_gate)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Causal convolution
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal: trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_proj = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt_proj_input, B_proj, C_proj = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ (discretization step) — input-dependent!
        dt = F.softplus(self.dt_proj(dt_proj_input))  # (B, L, d_inner)

        # A matrix (always negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # SSM scan
        y = self._ssm_scan(x_conv, dt, A, B_proj, C_proj, self.D)

        # Gated output
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)

        return out


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and layer norm."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


class MambaBackbone(nn.Module):
    """Stacked Mamba blocks forming the backbone.

    Processes spectral tokens with O(n) complexity and selective filtering.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) token embeddings
        Returns:
            x: (B, L, d_model) processed tokens
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
