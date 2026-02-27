"""
Minimal CNN-Transformer-VIB model for spectrum-to-structure identification.

Architecture:
    IR (3501) + Raman (3501)
      -> CNN Tokenizer (per-modality 1D CNN -> patch embeddings)
      -> Cross-Attention Fusion (IR attends to Raman and vice versa)
      -> Transformer Encoder (2 layers, 4 heads)
      -> VIB Head (z_chem + z_inst with reparameterization)
      -> Retrieval via cosine similarity on z_chem

Paper: "Can One Hear the Shape of a Molecule?"
Dataset: QM9S (~130K molecules, broadened IR + Raman spectra)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ============================================================
# CNN Tokenizer: converts raw spectrum to patch embeddings
# ============================================================

class CNNTokenizer(nn.Module):
    """1D CNN that converts a raw spectrum into a sequence of patch embeddings.

    Uses strided convolutions to reduce sequence length while extracting
    local spectral features (peaks, shoulders, baselines).

    Input:  (B, 1, L)  where L = 3501
    Output: (B, n_patches, embed_dim)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128,
                 input_length: int = 3501):
        super().__init__()
        self.embed_dim = embed_dim

        # Progressive downsampling: 3501 -> ~875 -> ~219 -> ~55
        self.conv_layers = nn.Sequential(
            # Block 1: capture narrow peaks (kernel=7)
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            # Block 2: capture broader features (kernel=5)
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Block 3: aggregate to embed_dim
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Compute output sequence length
        self._n_patches = self._compute_n_patches(input_length)

    def _compute_n_patches(self, L: int) -> int:
        """Compute output length after all conv layers."""
        # Simulate the conv layers
        with torch.no_grad():
            x = torch.zeros(1, 1, L)
            x = self.conv_layers(x)
            return x.size(2)

    @property
    def n_patches(self) -> int:
        return self._n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) raw spectrum
        Returns:
            patches: (B, n_patches, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        out = self.conv_layers(x)  # (B, embed_dim, n_patches)
        return out.transpose(1, 2)  # (B, n_patches, embed_dim)


# ============================================================
# Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, L, D)"""
        return x + self.pe[:, :x.size(1)]


# ============================================================
# Cross-Attention Fusion
# ============================================================

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention to fuse IR and Raman modalities.

    IR tokens attend to Raman tokens and vice versa, then the results
    are concatenated and projected.
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # IR attends to Raman
        self.ir_to_raman = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        # Raman attends to IR
        self.raman_to_ir = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ir = nn.LayerNorm(embed_dim)
        self.norm_raman = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Project concatenated [IR_fused; Raman_fused] back to embed_dim
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, ir_tokens: torch.Tensor,
                raman_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_tokens: (B, N, D) IR patch embeddings
            raman_tokens: (B, N, D) Raman patch embeddings
        Returns:
            fused: (B, N, D) fused representation
        """
        # Cross-attention with residual connections
        ir_attended, _ = self.ir_to_raman(
            query=ir_tokens, key=raman_tokens, value=raman_tokens
        )
        ir_fused = self.norm_ir(ir_tokens + ir_attended)

        raman_attended, _ = self.raman_to_ir(
            query=raman_tokens, key=ir_tokens, value=ir_tokens
        )
        raman_fused = self.norm_raman(raman_tokens + raman_attended)

        # Concatenate and project
        combined = torch.cat([ir_fused, raman_fused], dim=-1)  # (B, N, 2D)
        fused = self.fuse_proj(combined)  # (B, N, D)
        fused = self.norm_out(fused)

        return fused


# ============================================================
# Transformer Encoder
# ============================================================

class TransformerEncoder(nn.Module):
    """Lightweight Transformer encoder for global reasoning over fused tokens."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) token sequence
        Returns:
            cls_output: (B, D) pooled CLS representation
            all_tokens: (B, N+1, D) all token representations
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = self.encoder(x)  # (B, N+1, D)
        cls_output = x[:, 0]  # (B, D)
        return cls_output, x


# ============================================================
# VIB Head: Variational Information Bottleneck
# ============================================================

class VIBHead(nn.Module):
    """Variational Information Bottleneck for disentangling
    chemistry-invariant (z_chem) vs instrument-specific (z_inst) features.

    z_chem: captures molecular identity (used for retrieval)
    z_inst: captures nuisance variation (discarded at test time)

    Uses reparameterization trick with KL divergence to N(0,I).
    """

    def __init__(self, d_input: int = 128, z_chem_dim: int = 64,
                 z_inst_dim: int = 32):
        super().__init__()
        self.z_chem_dim = z_chem_dim
        self.z_inst_dim = z_inst_dim

        # Chemistry encoder: input -> (mu, logvar)
        self.chem_encoder = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
        )
        self.chem_mu = nn.Linear(d_input, z_chem_dim)
        self.chem_logvar = nn.Linear(d_input, z_chem_dim)

        # Instrument encoder: input -> (mu, logvar)
        self.inst_encoder = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.GELU(),
        )
        self.inst_mu = nn.Linear(d_input // 2, z_inst_dim)
        self.inst_logvar = nn.Linear(d_input // 2, z_inst_dim)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def kl_divergence(self, mu: torch.Tensor,
                      logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (B, d_input) encoder output (CLS token)
        Returns:
            dict with z_chem, z_inst, means, logvars, KL losses
        """
        # Chemistry branch
        h_chem = self.chem_encoder(h)
        chem_mu = self.chem_mu(h_chem)
        chem_logvar = self.chem_logvar(h_chem)
        z_chem = self.reparameterize(chem_mu, chem_logvar)

        # Instrument branch
        h_inst = self.inst_encoder(h)
        inst_mu = self.inst_mu(h_inst)
        inst_logvar = self.inst_logvar(h_inst)
        z_inst = self.reparameterize(inst_mu, inst_logvar)

        # KL divergences
        kl_chem = self.kl_divergence(chem_mu, chem_logvar)
        kl_inst = self.kl_divergence(inst_mu, inst_logvar)

        return {
            "z_chem": z_chem,           # (B, z_chem_dim) -- used for retrieval
            "z_inst": z_inst,           # (B, z_inst_dim) -- discarded at test time
            "chem_mu": chem_mu,         # (B, z_chem_dim)
            "chem_logvar": chem_logvar, # (B, z_chem_dim)
            "inst_mu": inst_mu,         # (B, z_inst_dim)
            "inst_logvar": inst_logvar, # (B, z_inst_dim)
            "kl_chem": kl_chem,         # scalar
            "kl_inst": kl_inst,         # scalar
            "kl_total": kl_chem + kl_inst,  # scalar
        }


# ============================================================
# Full Model: SpectrumIdentifier
# ============================================================

class SpectrumIdentifier(nn.Module):
    """CNN-Transformer-VIB model for molecular identification from
    IR + Raman vibrational spectra.

    Pipeline:
        IR spectrum  -> CNN Tokenizer -> IR patch embeddings
        Raman spectrum -> CNN Tokenizer -> Raman patch embeddings
        -> Cross-Attention Fusion -> Fused tokens
        -> Positional Encoding
        -> Transformer Encoder -> CLS token
        -> VIB Head -> z_chem (for retrieval) + z_inst (discarded)
    """

    def __init__(
        self,
        input_length: int = 3501,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        ff_dim: int = 256,
        z_chem_dim: int = 64,
        z_inst_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.z_chem_dim = z_chem_dim

        # Separate CNN tokenizers for IR and Raman (shared weights optional)
        self.ir_tokenizer = CNNTokenizer(
            in_channels=1, embed_dim=embed_dim, input_length=input_length
        )
        self.raman_tokenizer = CNNTokenizer(
            in_channels=1, embed_dim=embed_dim, input_length=input_length
        )

        n_patches = self.ir_tokenizer.n_patches

        # Positional encoding (applied after fusion)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=n_patches + 16)

        # Modality embeddings (learned, added before cross-attention)
        self.ir_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.raman_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim, n_heads=n_heads, dropout=dropout
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim, n_heads=n_heads,
            n_layers=n_transformer_layers, ff_dim=ff_dim, dropout=dropout
        )

        # VIB head
        self.vib = VIBHead(
            d_input=embed_dim, z_chem_dim=z_chem_dim, z_inst_dim=z_inst_dim
        )

    def encode(self, ir: torch.Tensor, raman: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass: spectra -> VIB outputs.

        Args:
            ir: (B, 3501) IR spectrum
            raman: (B, 3501) Raman spectrum
        Returns:
            dict with z_chem, z_inst, KL losses, etc.
        """
        # Tokenize each modality
        ir_tokens = self.ir_tokenizer(ir)       # (B, N, D)
        raman_tokens = self.raman_tokenizer(raman)  # (B, N, D)

        # Add modality embeddings
        ir_tokens = ir_tokens + self.ir_modality_emb
        raman_tokens = raman_tokens + self.raman_modality_emb

        # Cross-attention fusion
        fused = self.fusion(ir_tokens, raman_tokens)  # (B, N, D)

        # Positional encoding
        fused = self.pos_enc(fused)

        # Transformer encoder (with CLS token)
        cls_output, all_tokens = self.transformer(fused)  # (B, D), (B, N+1, D)

        # VIB head
        vib_out = self.vib(cls_output)

        return vib_out

    def forward(self, ir: torch.Tensor, raman: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for encode()."""
        return self.encode(ir, raman)

    def get_embedding(self, ir: torch.Tensor, raman: torch.Tensor) -> torch.Tensor:
        """Get the chemistry embedding for retrieval (deterministic at eval time).

        Args:
            ir: (B, 3501) IR spectrum
            raman: (B, 3501) Raman spectrum
        Returns:
            z_chem: (B, z_chem_dim) L2-normalized chemistry embedding
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            vib_out = self.encode(ir, raman)
            z_chem = F.normalize(vib_out["chem_mu"], dim=-1)  # Use mean (no sampling)
        if was_training:
            self.train()
        return z_chem

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Loss functions for training
# ============================================================

class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss for retrieval training.

    Each molecule is its own class. We create positive pairs via
    augmentation (add noise) and treat all other molecules in the
    batch as negatives.

    loss = -log( exp(sim(z, z+) / tau) / sum_j exp(sim(z, z_j) / tau) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) embeddings of original spectra
            z2: (B, D) embeddings of augmented spectra
        Returns:
            loss: scalar InfoNCE loss
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        B = z1.size(0)

        # Cosine similarity matrix: (2B, 2B)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss over molecule IDs.

    Treats each molecule as its own class. Uses a learnable linear
    classifier (or prototype-based classification with cosine similarity).
    """

    def __init__(self, embed_dim: int, n_classes: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        # Prototype matrix: one prototype per molecule
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.02)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) embeddings
            labels: (B,) molecule IDs
        Returns:
            loss: scalar cross-entropy
        """
        z = F.normalize(z, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = torch.mm(z, prototypes.t()) / self.temperature  # (B, n_classes)
        return F.cross_entropy(logits, labels)


# ============================================================
# Utility: parameter count summary
# ============================================================

def model_summary(model: nn.Module) -> str:
    """Print a summary of model parameters."""
    lines = []
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            lines.append(f"  {name}: {list(param.shape)} = {n:,}")
    lines.insert(0, f"Total trainable parameters: {total:,}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check
    print("Building SpectrumIdentifier...")
    model = SpectrumIdentifier(
        input_length=3501,
        embed_dim=128,
        n_heads=4,
        n_transformer_layers=2,
        ff_dim=256,
        z_chem_dim=64,
        z_inst_dim=32,
    )
    print(f"Parameters: {model.count_parameters():,}")
    print(f"CNN output patches: {model.ir_tokenizer.n_patches}")

    # Forward pass with random data
    B = 4
    ir = torch.randn(B, 3501)
    raman = torch.randn(B, 3501)

    model.train()
    out = model(ir, raman)
    print(f"z_chem shape: {out['z_chem'].shape}")
    print(f"z_inst shape: {out['z_inst'].shape}")
    print(f"KL chem: {out['kl_chem'].item():.4f}")
    print(f"KL inst: {out['kl_inst'].item():.4f}")

    # Test embedding extraction
    z = model.get_embedding(ir, raman)
    print(f"Embedding shape: {z.shape}")
    print(f"Embedding norm: {z.norm(dim=-1)}")  # Should be ~1.0

    print("\nSanity check passed!")
