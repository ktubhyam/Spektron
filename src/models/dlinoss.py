"""
Spektron: D-LinOSS Backbone

Drop-in replacement for MambaBackbone using Damped Linear Oscillatory
State-Space Models. The key physics insight: D-LinOSS dynamics are
mathematically identical to damped harmonic oscillators, which is
exactly the physics of molecular vibrations.

Architecture:
    Input (B, L, d_model) -> [DampedBlock x N] -> LayerNorm -> Output (B, L, d_model)

Each DampedBlock = BatchNorm -> DampedSSM -> GELU -> Dropout -> GLU -> Dropout -> Residual

Properties vs Mamba:
    - 2nd-order oscillatory dynamics (not 1st-order exponential decay)
    - IMEX symplectic discretization (not ZOH)
    - O(n) via parallel associative scan (same as Mamba)
    - Diagonal A: squared natural frequencies omega_k^2
    - Diagonal G: damping coefficients gamma_k
    - Proven universal approximator (Theorem 3.3, Rusch & Rus 2025)
"""
import math

import torch
import torch.nn as nn

from .linoss import LinOSSBlock, DampedLayer


class DLinOSSBackbone(nn.Module):
    """Stacked D-LinOSS blocks forming the backbone.

    Processes spectral tokens with O(n) complexity and oscillatory dynamics
    that match molecular vibrations.

    Bidirectional mode (default): runs each layer forward AND backward,
    then averages the results. This is critical for spectra which are
    non-causal — peaks at any position depend on global molecular modes.

    Interface matches MambaBackbone exactly:
        Input: (B, L, d_model) -> Output: (B, L, d_model)
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4,
                 d_state: int = 128, r_min: float = 0.9,
                 r_max: float = 1.0, theta_max: float = math.pi,
                 dropout: float = 0.05, layer_name: str = "Damped",
                 bidirectional: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.bidirectional = bidirectional

        # Forward layers
        self.layers = nn.ModuleList([
            LinOSSBlock(
                layer_name=layer_name,
                state_dim=d_state,
                hidden_dim=d_model,
                r_min=r_min,
                r_max=r_max,
                theta_max=theta_max,
                drop_rate=dropout,
            )
            for _ in range(n_layers)
        ])

        # Backward layers (separate parameters for richer representation)
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                LinOSSBlock(
                    layer_name=layer_name,
                    state_dim=d_state,
                    hidden_dim=d_model,
                    r_min=r_min,
                    r_max=r_max,
                    theta_max=theta_max,
                    drop_rate=dropout,
                )
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
        if self.bidirectional:
            for fwd_layer, bwd_layer in zip(self.layers, self.bwd_layers):
                # Forward scan (left-to-right)
                x_fwd = fwd_layer(x)
                # Backward scan: only flip content tokens (skip CLS/DOMAIN at 0,1)
                # This preserves special token positions while reversing spectral order
                special = x[:, :2]        # (B, 2, d_model) — CLS + DOMAIN
                content = x[:, 2:]        # (B, L, d_model) — spectral tokens
                x_bwd_in = torch.cat([special, torch.flip(content, [1])], dim=1)
                x_bwd_out = bwd_layer(x_bwd_in)
                # Flip content back to original order
                x_bwd = torch.cat([
                    x_bwd_out[:, :2],
                    torch.flip(x_bwd_out[:, 2:], [1])
                ], dim=1)
                # Average forward and backward
                x = (x_fwd + x_bwd) * 0.5
        else:
            for layer in self.layers:
                x = layer(x)
        return self.norm(x)

    @property
    def learned_frequencies(self) -> list[torch.Tensor]:
        """Extract learned natural frequencies from all layers.

        Returns a list of tensors, one per layer, each of shape (d_state,).
        These can be compared against physical vibrational frequencies
        to verify the architecture-physics alignment claim.
        """
        freqs = []
        all_layers = list(self.layers)
        if self.bidirectional:
            all_layers += list(self.bwd_layers)
        for layer in all_layers:
            if hasattr(layer.layer, 'learned_frequencies'):
                freqs.append(layer.layer.learned_frequencies)
        return freqs

    @property
    def learned_damping(self) -> list[torch.Tensor]:
        """Extract learned damping coefficients from all layers."""
        dampings = []
        all_layers = list(self.layers)
        if self.bidirectional:
            all_layers += list(self.bwd_layers)
        for layer in all_layers:
            if hasattr(layer.layer, 'learned_damping'):
                dampings.append(layer.layer.learned_damping)
        return dampings
