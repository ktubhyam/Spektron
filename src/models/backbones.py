"""
Spektron: CNN and S4D Backbone Baselines

Provides additional backbone architectures for the architecture benchmark:
- CNN1DBackbone: 1D convolutional network (FIR filters)
- S4DBackbone: Diagonal SSM without oscillatory structure (ablation)

All backbones share the interface: (B, L, d_model) -> (B, L, d_model)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DBlock(nn.Module):
    """Single 1D CNN block with residual connection.

    Conv1D -> BatchNorm -> GELU -> Conv1D -> Dropout -> Residual
    """

    def __init__(self, d_model: int, kernel_size: int = 7,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand
        padding = (kernel_size - 1) // 2  # same padding

        self.norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_inner, kernel_size,
                               padding=padding, groups=1)
        self.conv2 = nn.Conv1d(d_inner, d_model, 1)  # pointwise
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, d_model). Returns: (B, L, d_model)."""
        residual = x
        x = self.norm(x)
        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = x.transpose(1, 2)
        return x + residual


class CNN1DBackbone(nn.Module):
    """Stacked 1D CNN blocks forming a backbone.

    Uses depthwise-separable-style convolutions with residual connections.
    Matches the (B, L, d_model) -> (B, L, d_model) interface of other backbones.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6,
                 kernel_size: int = 7, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            CNN1DBlock(d_model, kernel_size, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, d_model). Returns: (B, L, d_model)."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class S4DBlock(nn.Module):
    """Diagonal State Space Model block (S4D-style, no oscillatory structure).

    Uses real-valued diagonal A matrix with exponential decay only.
    This is the ablation control for D-LinOSS: proves whether the
    oscillatory structure matters or if any SSM would do.

    Discrete equations:
        h_t = diag(exp(A * dt)) * h_{t-1} + B * x_t
        y_t = Re(C * h_t) + D * x_t
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)

        # A: real negative diagonal (exponential decay, NO oscillation)
        # Initialize with HiPPO-inspired spacing
        A_init = -torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(-A_init))  # store log for positivity

        # dt: learnable discretization step
        dt_init = torch.rand(d_model) * 0.1 + 0.001
        self.log_dt = nn.Parameter(torch.log(dt_init))

        # B, C: input/output projections (real-valued)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, d_model). Returns: (B, L, d_model)."""
        residual = x
        x = self.norm(x)

        B_batch, L, D = x.shape

        # Discretize
        dt = torch.exp(self.log_dt)  # (d_model,)
        A = -torch.exp(self.A_log)   # (d_state,) — always negative
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (d_model, d_state)

        # Sequential scan (simple; parallel scan for production)
        h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            # x_t: (B, D) -> project to state: (B, D, N)
            h = dA.unsqueeze(0) * h + self.B.unsqueeze(0) * x[:, t].unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = self.dropout(y)
        return y + residual


class S4DBackbone(nn.Module):
    """Stacked S4D blocks as ablation baseline.

    Real diagonal SSM without oscillatory structure.
    If D-LinOSS outperforms this, the oscillatory inductive bias matters.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4,
                 d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            S4DBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, d_model). Returns: (B, L, d_model)."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
