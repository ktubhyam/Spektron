"""
D-LinOSS layer implementations (vendored from TritonLinOSS, MIT License).

Import paths fixed from `src.damped_linoss.` to relative `.scan`.
Only the pure PyTorch scan is used (no Triton on macOS).
On GPU training, Triton kernels can be added as a drop-in.

Core equation (Damped):
    x''(t) = -A·x(t) - G·x'(t) + B·u(t)
    y(t)   = C·x(t) + D·u(t)

This is mathematically identical to a system of damped harmonic oscillators,
which is exactly the physics of molecular vibrations.
"""
import abc
import warnings
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .scan import associative_scan

# Triton is not available on macOS; will use PyTorch fallback
TRITON_AVAILABLE = False
ParallelScanFunction = None

SCAN_TYPE = Tuple[torch.Tensor, torch.Tensor]


def binary_operator(q_i: SCAN_TYPE, q_j: SCAN_TYPE) -> SCAN_TYPE:
    """Binary operator for parallel scan of 2x2 linear recurrence.

    Composes two (transition_matrix, forcing) pairs:
        (M_j, F_j) * (M_i, F_i) = (M_j @ M_i, M_j @ F_i + F_j)

    M is stored as flattened 4*P (four P-vectors: [M_11, M_12, M_21, M_22]).
    F is stored as (2*P, 2) where the last dim holds real/imag components.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    iA, iB, iC, iD = torch.chunk(A_i, 4, dim=-1)
    jA, jB, jC, jD = torch.chunk(A_j, 4, dim=-1)

    A_new_part = jA * iA + jB * iC
    B_new_part = jA * iB + jB * iD
    C_new_part = jC * iA + jD * iC
    D_new_part = jC * iB + jD * iD

    A_new = torch.cat([A_new_part, B_new_part, C_new_part, D_new_part], dim=-1)

    b_i1, b_i2 = torch.chunk(b_i, 2, dim=-2)

    jA_bs = jA.unsqueeze(-1)
    jB_bs = jB.unsqueeze(-1)
    jC_bs = jC.unsqueeze(-1)
    jD_bs = jD.unsqueeze(-1)

    new_b1 = jA_bs * b_i1 + jB_bs * b_i2
    new_b2 = jC_bs * b_i1 + jD_bs * b_i2

    new_b = torch.cat([new_b1, new_b2], dim=-2)

    return A_new, new_b + b_j


class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.sigmoid(self.w2(x))


class _AbstractLinOSSLayer(nn.Module):
    """Base class for all LinOSS layer variants."""

    def __init__(self, use_triton: bool = None):
        super().__init__()
        self.use_triton = use_triton

    @abc.abstractmethod
    def _recurrence(self, *args, **kwargs):
        raise NotImplementedError

    def _should_use_triton(self, tensor: torch.Tensor) -> bool:
        """Always False on macOS (no Triton). On Linux+CUDA, check availability."""
        if self.use_triton is None:
            return TRITON_AVAILABLE and tensor.is_cuda
        else:
            if self.use_triton and not TRITON_AVAILABLE:
                raise RuntimeError("Triton requested but not available.")
            return self.use_triton


class IMLayer(_AbstractLinOSSLayer):
    """LinOSS with Implicit Midpoint (IM) discretization.

    Energy-conserving (symplectic), no damping.
    Eigenvalues lie exactly on the unit circle.
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 r_min: float, r_max: float, theta_max: float,
                 use_triton: bool = None):
        super().__init__(use_triton)

        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / math.sqrt(hidden_dim)
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / math.sqrt(state_dim)
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B_mat, input_sequence, step):
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B_mat)

        schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
        M_11 = 1.0 - step**2.0 * A_diag * schur_comp
        M_12 = -1.0 * step * A_diag * schur_comp
        M_21 = step * schur_comp
        M_22 = schur_comp

        M = torch.cat([M_11, M_12, M_21, M_22])
        L = input_sequence.shape[-2]

        if input_sequence.dim() == 3:
            batch_size = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(batch_size, -1)
        else:
            M_elements = M

        view_shape = (1, -1, 1) if input_sequence.dim() == 2 else (1, 1, -1, 1)

        M_11_b = M_11.view(view_shape)
        M_21_b = M_21.view(view_shape)
        step_b = step.view(view_shape)

        F1 = M_11_b * Bu_elements * step_b
        F2 = M_21_b * Bu_elements * step_b
        F_mat = torch.cat((F1, F2), dim=-2)

        if input_sequence.dim() == 3:
            M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
            scan_axis = 1
        else:
            M_expanded = M_elements.unsqueeze(0).expand(L, -1)
            scan_axis = 0

        _, xs = associative_scan(
            binary_operator, (M_expanded, F_mat),
            reverse=False, axis=scan_axis,
        )

        return xs[..., A_diag.shape[0]:, :]

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        orig_dtype = input_sequence.dtype
        with torch.amp.autocast('cuda', enabled=False):
            input_f32 = input_sequence.float()
            steps = torch.sigmoid(self.steps)
            A_diag = F.relu(self.A_diag)

            ys = self._recurrence(A_diag, self.B, input_f32, steps)

            Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
            Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
            Du = input_f32 * self.D
            output = Cy + Du
        return output.to(orig_dtype)


class IMEXLayer(_AbstractLinOSSLayer):
    """LinOSS with Implicit-Explicit (IMEX) discretization.

    Better stability properties than IM for stiff systems.
    Still energy-conserving for undamped case.
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 r_min: float, r_max: float, theta_max: float,
                 use_triton: bool = None):
        super().__init__(use_triton)

        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / math.sqrt(hidden_dim)
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / math.sqrt(state_dim)
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B_mat, input_sequence, step):
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B_mat)

        M_11 = torch.ones_like(A_diag)
        M_12 = -1.0 * step * A_diag
        M_21 = step
        M_22 = 1.0 - (step**2.0) * A_diag

        M = torch.cat([M_11, M_12, M_21, M_22])
        L = input_sequence.shape[-2]

        if input_sequence.dim() == 3:
            batch_size = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(batch_size, -1)
        else:
            M_elements = M

        view_shape = (1, -1, 1) if input_sequence.dim() == 2 else (1, 1, -1, 1)
        step_b = step.view(view_shape)

        F1 = Bu_elements * step_b
        F2 = Bu_elements * (step_b**2.0)
        F_mat = torch.cat((F1, F2), dim=-2)

        if input_sequence.dim() == 3:
            M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
            scan_axis = 1
        else:
            M_expanded = M_elements.unsqueeze(0).expand(L, -1)
            scan_axis = 0

        _, xs = associative_scan(
            binary_operator, (M_expanded, F_mat),
            reverse=False, axis=scan_axis,
        )

        return xs[..., A_diag.shape[0]:, :]

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        orig_dtype = input_sequence.dtype
        with torch.amp.autocast('cuda', enabled=False):
            input_f32 = input_sequence.float()
            steps = torch.sigmoid(self.steps)
            A_diag = F.relu(self.A_diag)

            ys = self._recurrence(A_diag, self.B, input_f32, steps)

            Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
            Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
            Du = input_f32 * self.D
            output = Cy + Du
        return output.to(orig_dtype)


class DampedLayer(_AbstractLinOSSLayer):
    """D-LinOSS: Damped Linear Oscillatory State-Space Model.

    The key contribution from Boyer et al. (arXiv 2505.12171):
    adds a learnable diagonal damping matrix G to the oscillatory dynamics.

    Physics correspondence:
        x''(t) = -A·x(t) - G·x'(t) + B·u(t)

    where:
        A_diag >= 0: squared natural frequencies (omega_k^2)
        G_diag >= 0: damping coefficients (gamma_k)
        B: input coupling (dipole/polarizability derivatives)
        C: output projection (measurement)
        D: skip connection

    The IMEX discretization yields:
        S = I + dt*G
        z_{k+1} = S^{-1} * [z_k + dt*(-A*x_k + B*u_{k+1})]
        x_{k+1} = x_k + dt*z_{k+1}

    Eigenvalue magnitude: |lambda_i| = 1/sqrt(1 + dt_i*G_i)
    G=0: undamped (energy-conserving), G>0: damped
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 r_min: float, r_max: float, theta_max: float,
                 use_triton: bool = None):
        super().__init__(use_triton)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)
        steps = torch.sigmoid(self.steps)

        # Initialize damping from spectral radius bounds
        mags = torch.sqrt(torch.rand(state_dim) * (r_max**2 - r_min**2) + r_min**2)
        G_diag_init = (1 - mags**2) / (steps.detach() * mags**2)
        self.G_diag = nn.Parameter(G_diag_init)

        # Initialize A from angle (frequency) bounds
        theta = torch.rand(state_dim) * theta_max
        A_diag_init = self._map_theta_to_A(
            theta, F.relu(self.G_diag.detach()), steps.detach()
        )
        self.A_diag = nn.Parameter(A_diag_init)

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / math.sqrt(hidden_dim)
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / math.sqrt(state_dim)
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _map_theta_to_A(self, thetas: torch.Tensor, G_diag: torch.Tensor,
                        steps: torch.Tensor) -> torch.Tensor:
        """Map oscillation angles theta to A diagonal values.

        Given (theta, G, step), solve for A such that the discrete
        eigenvalue has angle theta and magnitude 1/sqrt(1+step*G).
        """
        cos_theta = torch.cos(thetas)
        tan_theta = torch.tan(thetas)
        tan_theta_sq = tan_theta**2

        sqrt_term = 4 * torch.sqrt(
            steps**4 * cos_theta**(-2) + steps**5 * G_diag * cos_theta**(-2)
        )

        common_term = steps**2 * (
            -4
            - 2 * steps * G_diag
            - 4 * tan_theta_sq
            - 2 * steps * G_diag * tan_theta_sq
        )

        denominator = 2 * steps**4 * (1 + tan_theta_sq)

        A_plus = (sqrt_term - common_term) / denominator
        A_minus = (-sqrt_term - common_term) / denominator

        A_diag = torch.where(thetas > math.pi / 2, A_plus, A_minus)
        return A_diag

    def _recurrence(self, A_diag, G_diag, B_mat, input_sequence, step):
        """Compute the recurrence with damping.

        Args:
            A_diag: (P,) squared frequencies
            G_diag: (P,) damping coefficients
            B_mat: (P, H, 2) input matrix
            input_sequence: (L, H) or (B, L, H)
            step: (P,) discretization steps

        Returns:
            states: (..., L, P, 2) SSM states
        """
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B_mat)

        Identity = torch.ones_like(A_diag)
        S = Identity + step * G_diag

        # CFL stability condition: alpha = step²*A/S must be < 2
        # to keep transition matrix eigenvalues inside the unit circle.
        # Without this, A_diag can grow during training making M_22 < -1,
        # causing exponential divergence in the 2048-step associative scan.
        alpha = step**2 * A_diag / S
        alpha = 1.99 * torch.tanh(alpha / 1.99)  # Soft clamp: differentiable everywhere

        M_11 = 1.0 / S
        M_12 = -alpha / step          # was: -step/S * A_diag = -alpha/step
        M_21 = step / S
        M_22 = Identity - alpha        # was: 1 - step²*A/S = 1 - alpha

        M = torch.cat([M_11, M_12, M_21, M_22])
        L = input_sequence.shape[-2]

        if input_sequence.dim() == 3:
            batch_size = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(batch_size, -1)
        else:
            M_elements = M

        view_shape = (1, -1, 1) if input_sequence.dim() == 2 else (1, 1, -1, 1)

        step_b = step.view(view_shape)
        S_b = S.view(view_shape)

        F1 = step_b * (1.0 / S_b) * Bu_elements
        F2 = (step_b**2) * (1.0 / S_b) * Bu_elements
        F_mat = torch.cat((F1, F2), dim=-2)

        if input_sequence.dim() == 3:
            M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
            scan_axis = 1
        else:
            M_expanded = M_elements.unsqueeze(0).expand(L, -1)
            scan_axis = 0

        _, xs = associative_scan(
            binary_operator, (M_expanded, F_mat),
            reverse=False, axis=scan_axis,
        )

        return xs[..., A_diag.shape[0]:, :]

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the damped oscillatory SSM.

        Forces float32 for the recurrence scan to prevent numerical
        instability under AMP mixed precision (the 2048-step associative
        scan accumulates 11 levels of matrix composition, which is
        catastrophic in float16).

        Args:
            input_sequence: (L, H) or (B, L, H)

        Returns:
            output: same shape as input
        """
        orig_dtype = input_sequence.dtype

        # Force float32 for the scan — AMP float16 causes NaN after
        # ~11 levels of binary composition in the associative scan.
        with torch.amp.autocast('cuda', enabled=False):
            input_f32 = input_sequence.float()
            steps = torch.sigmoid(self.steps)
            G_diag = F.relu(self.G_diag)
            A_diag = F.relu(self.A_diag)  # Must be ≥0 (squared frequencies)

            ys = self._recurrence(A_diag, G_diag, self.B, input_f32, steps)

            Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
            Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
            Du = input_f32 * self.D
            output = Cy + Du

        return output.to(orig_dtype)

    @property
    def learned_frequencies(self) -> torch.Tensor:
        """Extract learned natural frequencies (omega_k) from A_diag.

        Returns sqrt(A_diag) for positive values, useful for comparing
        against physical vibrational frequencies.
        """
        with torch.no_grad():
            a = self.A_diag.data
            return torch.sqrt(F.relu(a))

    @property
    def learned_damping(self) -> torch.Tensor:
        """Extract learned damping coefficients."""
        with torch.no_grad():
            return F.relu(self.G_diag.data)


class LinOSSBlock(nn.Module):
    """Single LinOSS block: Norm -> SSM -> GELU -> Dropout -> GLU -> Dropout -> Residual.

    This is the building block for the D-LinOSS backbone.
    Uses BatchNorm (following the original paper) instead of LayerNorm.
    """

    def __init__(self, layer_name: str, state_dim: int, hidden_dim: int,
                 r_min: float, r_max: float, theta_max: float,
                 drop_rate: float):
        super().__init__()

        layer_map = {
            "IM": IMLayer,
            "IMEX": IMEXLayer,
            "Damped": DampedLayer,
        }
        if layer_name not in layer_map:
            raise KeyError(f"Layer name {layer_name} not defined. Use: {list(layer_map.keys())}")

        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)
        self.layer = layer_map[layer_name](
            state_dim, hidden_dim, r_min, r_max, theta_max,
        )
        self.glu = GLU(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (L, H) or (B, L, H)
        Returns:
            output: same shape as input (with residual)
        """
        orig_dtype = x.dtype

        # Force float32 for the entire block — the SSM scan can produce
        # values up to ±200K which overflow float16 in the GLU's linears.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            skip = x

            # BatchNorm expects (N, C, L) format
            x_t = x
            if x.dim() == 2:
                x_t = x_t.unsqueeze(0)
            x_norm = self.norm(x_t.permute(0, 2, 1)).permute(0, 2, 1)
            if x.dim() == 2:
                x_norm = x_norm.squeeze(0)
            x = x_norm

            x = self.layer(x)
            x = F.gelu(x)
            x = self.drop(x)
            x = self.glu(x)
            x = self.drop(x)
            x = skip + x

        return x.to(orig_dtype)
