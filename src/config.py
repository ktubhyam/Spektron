"""
Spektron: Configuration
Hybrid Mamba-Transformer with OT, Physics, Wavelets, MoE, TTT, FNO, KAN, VIB
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class WaveletConfig:
    """Wavelet multi-scale embedding configuration."""
    wavelet: str = "db4"           # Daubechies-4 (good for spectral data)
    levels: int = 4                 # Decomposition levels
    mode: str = "symmetric"         # Boundary handling
    trainable_filters: bool = False # Learn wavelet filters


@dataclass
class MambaConfig:
    """Selective State Space Model (Mamba) configuration."""
    d_model: int = 256
    d_state: int = 16              # SSM state dimension
    d_conv: int = 4                # Local convolution width
    expand: int = 2                # Block expansion factor
    n_layers: int = 4              # Number of Mamba blocks
    dt_rank: str = "auto"          # Rank of Δ projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True             # Parallel scan (faster)


@dataclass
class DLinOSSConfig:
    """Damped Linear Oscillatory State-Space Model configuration.

    D-LinOSS dynamics are mathematically identical to damped harmonic
    oscillators, matching the physics of molecular vibrations.
    """
    d_model: int = 256
    d_state: int = 128             # Number of oscillators (P)
    n_layers: int = 4              # Number of D-LinOSS blocks
    r_min: float = 0.9             # Min spectral radius (init)
    r_max: float = 1.0             # Max spectral radius (init)
    theta_max: float = 3.14159     # Max oscillation angle (pi)
    dropout: float = 0.05          # Dropout rate
    layer_name: str = "Damped"     # "Damped", "IM", or "IMEX"


@dataclass
class CNN1DConfig:
    """1D CNN backbone configuration."""
    d_model: int = 256
    n_layers: int = 6              # More layers needed (smaller receptive field)
    kernel_size: int = 7           # Conv kernel size
    expand: int = 2                # Channel expansion factor
    dropout: float = 0.1


@dataclass
class S4DConfig:
    """Diagonal SSM (S4D) configuration — ablation control for D-LinOSS.

    Real-valued diagonal A (exponential decay only, NO oscillation).
    If D-LinOSS beats S4D, the oscillatory structure matters.
    """
    d_model: int = 256
    d_state: int = 64              # State dimension
    n_layers: int = 4
    dropout: float = 0.1


@dataclass
class TransformerConfig:
    """Transformer block configuration."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2              # Fewer layers (Mamba handles bulk)
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    n_experts: int = 4             # NIR, IR, Raman, Cross-modal
    top_k: int = 2                 # Activate top-k experts per input
    d_expert: int = 512            # Expert hidden dimension
    use_kan: bool = False          # Use KAN activations in experts
    balance_loss_weight: float = 0.01  # Load balancing
    noise_std: float = 0.1        # Gating noise for exploration


@dataclass
class FNOConfig:
    """Fourier Neural Operator transfer head configuration."""
    modes: int = 32                # Number of Fourier modes to keep
    width: int = 64                # Hidden channel width
    n_layers: int = 4
    activation: str = "gelu"
    use_spectral_conv: bool = True


@dataclass
class KANConfig:
    """Kolmogorov-Arnold Network configuration."""
    grid_size: int = 5             # B-spline grid points
    spline_order: int = 3          # Cubic B-splines
    grid_range: Tuple[float, float] = (-1.0, 1.0)
    base_activation: str = "silu"


@dataclass
class VIBConfig:
    """Variational Information Bottleneck configuration."""
    z_chem_dim: int = 128          # Chemistry-invariant latent
    z_inst_dim: int = 64           # Instrument-specific latent
    beta: float = 1e-3             # KL weight (final value after annealing)
    beta_start: float = 0.1        # Initial KL weight (high → diverse representations)
    disentangle_weight: float = 0.1


@dataclass
class OTConfig:
    """Optimal Transport alignment configuration."""
    reg: float = 1.0               # Sinkhorn regularization (>0.5 for 128-dim embeddings)
    n_iter: int = 50               # Sinkhorn iterations
    method: str = "sinkhorn"       # sinkhorn or emd
    weight: float = 0.1            # Loss weight


@dataclass
class PhysicsConfig:
    """Physics-informed loss configuration."""
    beer_lambert_weight: float = 0.05
    smoothness_weight: float = 0.05
    non_negativity_weight: float = 0.02
    peak_shape_weight: float = 0.02
    derivative_smoothness_weight: float = 0.03
    smoothness_kernel_size: int = 11


@dataclass
class TTTConfig:
    """Test-Time Training configuration."""
    n_steps: int = 10              # Gradient steps at test time
    lr: float = 1e-4               # TTT learning rate
    mask_ratio: float = 0.15       # MSRP mask ratio for TTT
    adapt_layers: str = "norm"     # Which layers to adapt: norm, all, lora


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )


@dataclass
class PretrainConfig:
    """Pretraining configuration."""
    # MSRP
    mask_ratio: float = 0.40       # Target mask ratio (higher = harder task)
    mask_ratio_start: float = 0.15 # Initial mask ratio (progressive schedule)
    mask_type: str = "contiguous"  # contiguous, random, peak_aware
    mask_patch_size: int = 3       # Contiguous patches to mask together

    # Loss weights
    msrp_weight: float = 1.0
    contrastive_weight: float = 0.3
    denoise_weight: float = 0.2
    ot_weight: float = 0.1
    physics_weight: float = 0.1
    vib_weight: float = 0.15
    moe_balance_weight: float = 0.01

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000
    grad_clip: float = 5.0
    grad_accumulation_steps: int = 1  # Accumulate gradients over N steps
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # Augmentation
    noise_std: float = 0.01
    baseline_drift_scale: float = 0.005
    wavelength_shift_max: int = 3
    intensity_scale_range: Tuple[float, float] = (0.95, 1.05)


@dataclass
class FinetuneConfig:
    """Fine-tuning / calibration transfer configuration."""
    n_transfer_samples: List[int] = field(
        default_factory=lambda: [5, 10, 20, 30, 50, 100]
    )
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 100
    patience: int = 15
    use_lora: bool = True
    use_ttt: bool = True


@dataclass
class SpektronConfig:
    """Master configuration for Spektron."""

    # Model name
    name: str = "Spektron"
    seed: int = 42

    # Backbone: "mamba", "dlinoss", "transformer", "cnn1d", "s4d"
    backbone: str = "mamba"

    # Input
    n_channels: int = 2048         # Resample all spectra to this
    patch_size: int = 32
    stride: int = 16

    # Embedding: "wavelet" (patched) or "raw" (no patching, for D-LinOSS)
    embedding_type: str = "wavelet"
    raw_embed_kernel: int = 15     # Local Conv1d kernel for raw embedding

    # Domain tokens
    domain_tokens: List[str] = field(
        default_factory=lambda: ["NIR", "IR", "RAMAN", "UNKNOWN"]
    )

    # Sub-configs
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    dlinoss: DLinOSSConfig = field(default_factory=DLinOSSConfig)
    cnn1d: CNN1DConfig = field(default_factory=CNN1DConfig)
    s4d: S4DConfig = field(default_factory=S4DConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)
    kan: KANConfig = field(default_factory=KANConfig)
    vib: VIBConfig = field(default_factory=VIBConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @property
    def d_model(self) -> int:
        backend_configs = {
            "dlinoss": self.dlinoss,
            "mamba": self.mamba,
            "cnn1d": self.cnn1d,
            "s4d": self.s4d,
            "transformer": self.transformer,
        }
        return backend_configs.get(self.backbone, self.mamba).d_model

    @property
    def n_patches(self) -> int:
        """Number of patches for wavelet/patched embedding."""
        return (self.n_channels - self.patch_size) // self.stride + 1

    @property
    def total_latent_dim(self) -> int:
        return self.vib.z_chem_dim + self.vib.z_inst_dim

    @property
    def use_raw_embedding(self) -> bool:
        """Whether to use raw (non-patched) embedding."""
        return self.embedding_type == "raw"

    @property
    def seq_len(self) -> int:
        """Effective sequence length after embedding (excluding special tokens)."""
        if self.use_raw_embedding:
            return self.n_channels  # 2048 raw points
        return self.n_patches       # 127 patches


def get_light_config() -> SpektronConfig:
    """Get a lightweight configuration for fast CPU testing (Mamba backbone)."""
    cfg = SpektronConfig()

    # Smaller model dimensions
    cfg.mamba.d_model = 64
    cfg.mamba.d_state = 8
    cfg.mamba.n_layers = 2
    cfg.mamba.expand = 1

    cfg.transformer.d_model = 64
    cfg.transformer.n_heads = 4
    cfg.transformer.n_layers = 1
    cfg.transformer.d_ff = 128

    cfg.moe.n_experts = 2
    cfg.moe.d_expert = 128

    cfg.vib.z_chem_dim = 32
    cfg.vib.z_inst_dim = 16

    cfg.fno.width = 16
    cfg.fno.modes = 8
    cfg.fno.n_layers = 2

    return cfg


def get_dlinoss_config() -> SpektronConfig:
    """Get D-LinOSS configuration for Paper 1.

    Uses raw spectral embedding (no patching) + D-LinOSS backbone.
    Processes all 2048 spectral points directly with O(n) complexity.
    """
    cfg = SpektronConfig()

    # D-LinOSS backbone
    cfg.backbone = "dlinoss"
    cfg.embedding_type = "raw"

    cfg.dlinoss.d_model = 256
    cfg.dlinoss.d_state = 128      # 128 oscillators per layer
    cfg.dlinoss.n_layers = 4
    cfg.dlinoss.dropout = 0.05

    cfg.transformer.d_model = 256
    cfg.transformer.n_heads = 8
    cfg.transformer.n_layers = 2
    cfg.transformer.d_ff = 1024

    cfg.moe.n_experts = 4
    cfg.moe.d_expert = 512

    cfg.vib.z_chem_dim = 128
    cfg.vib.z_inst_dim = 64

    cfg.fno.width = 64
    cfg.fno.modes = 32
    cfg.fno.n_layers = 4

    return cfg


def get_benchmark_config(backbone: str = "dlinoss", seed: int = 42) -> SpektronConfig:
    """Get param-matched config for architecture benchmark (E1).

    All backbones target ~2M backbone parameters.
    Uses raw embedding for all to ensure fair comparison.

    Args:
        backbone: "dlinoss", "mamba", "transformer", "cnn1d", "s4d"
        seed: random seed for reproducibility
    """
    cfg = SpektronConfig()
    cfg.backbone = backbone
    cfg.embedding_type = "raw"
    cfg.seed = seed

    # Shared training config
    cfg.pretrain.batch_size = 64
    cfg.pretrain.lr = 3e-4
    cfg.pretrain.max_steps = 50000
    cfg.pretrain.warmup_steps = 2000
    cfg.pretrain.mask_ratio = 0.40
    cfg.pretrain.mask_ratio_start = 0.15
    cfg.pretrain.mask_type = "contiguous"

    # VIB (shared across all)
    cfg.vib.z_chem_dim = 128
    cfg.vib.z_inst_dim = 64

    # Param-matched backbone configs (~2M backbone params each)
    # D-LinOSS is bidirectional (4 fwd + 4 bwd = 8 blocks) → ~2.1M
    # All others tuned to match within ~25%
    if backbone == "dlinoss":
        cfg.dlinoss.d_model = 256
        cfg.dlinoss.d_state = 128
        cfg.dlinoss.n_layers = 4       # 4 fwd + 4 bwd = 8 blocks total
        cfg.dlinoss.dropout = 0.05
    elif backbone == "mamba":
        cfg.mamba.d_model = 256
        cfg.mamba.d_state = 16
        cfg.mamba.n_layers = 4          # ~1.75M
        cfg.mamba.expand = 2
        cfg.mamba.d_conv = 4
    elif backbone == "transformer":
        cfg.transformer.d_model = 256
        cfg.transformer.n_heads = 8
        cfg.transformer.n_layers = 4    # ~1.6M (NOTE: shared with post-backbone)
        cfg.transformer.d_ff = 512
        cfg.transformer.dropout = 0.1
    elif backbone == "cnn1d":
        cfg.cnn1d.d_model = 256
        cfg.cnn1d.n_layers = 2          # 2 layers × ~1.05M each → ~2.1M
        cfg.cnn1d.kernel_size = 7
        cfg.cnn1d.expand = 2
        cfg.cnn1d.dropout = 0.1
    elif backbone == "s4d":
        cfg.s4d.d_model = 256
        cfg.s4d.d_state = 256           # Higher state dim for more params
        cfg.s4d.n_layers = 8            # 8 layers → ~1.06M
        cfg.s4d.dropout = 0.1
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # MoE (shared across all backbones)
    cfg.moe.n_experts = 4
    cfg.moe.d_expert = 512

    # Post-backbone Transformer (skipped when backbone="transformer")
    if backbone != "transformer":
        cfg.transformer.d_model = 256
        cfg.transformer.n_heads = 8
        cfg.transformer.n_layers = 2
        cfg.transformer.d_ff = 1024

    return cfg


def get_light_dlinoss_config() -> SpektronConfig:
    """Get a lightweight D-LinOSS configuration for fast CPU testing."""
    cfg = SpektronConfig()

    cfg.backbone = "dlinoss"
    cfg.embedding_type = "raw"
    cfg.n_channels = 256           # Shorter sequence for CPU testing

    cfg.dlinoss.d_model = 64
    cfg.dlinoss.d_state = 32
    cfg.dlinoss.n_layers = 2
    cfg.dlinoss.dropout = 0.05

    cfg.transformer.d_model = 64
    cfg.transformer.n_heads = 4
    cfg.transformer.n_layers = 1
    cfg.transformer.d_ff = 128

    cfg.moe.n_experts = 2
    cfg.moe.d_expert = 128

    cfg.vib.z_chem_dim = 32
    cfg.vib.z_inst_dim = 16

    cfg.fno.width = 16
    cfg.fno.modes = 8
    cfg.fno.n_layers = 2

    return cfg
