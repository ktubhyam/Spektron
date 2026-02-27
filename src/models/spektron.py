"""
Spektron: Full Model Assembly

Supports two backbone modes:
1. Mamba backbone (original): Wavelet embed → Mamba(×4) → MoE → Transformer(×2) → VIB
2. D-LinOSS backbone (Paper 1): Raw embed → D-LinOSS(×4) → MoE → Transformer(×2) → VIB

The D-LinOSS mode processes all 2048 spectral points without patching,
using oscillatory dynamics that match molecular vibrations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import copy

from .embedding import WaveletEmbedding, RawSpectralEmbedding
from .mamba import MambaBackbone
from .dlinoss import DLinOSSBackbone
from .backbones import CNN1DBackbone, S4DBackbone
from .moe import MixtureOfExperts
from .transformer import TransformerEncoder
from .heads import VIBHead, ReconstructionHead, RegressionHead, FNOTransferHead


class Spektron(nn.Module):
    """Spektron: Physics-Informed State Space Foundation Model.

    Supports both Mamba and D-LinOSS backbones via config.backbone.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        # ========== Learnable mask token ==========
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # ========== Embedding ==========
        if config.use_raw_embedding:
            self.embedding = RawSpectralEmbedding(
                d_model=d,
                n_channels=config.n_channels,
                kernel_size=config.raw_embed_kernel,
            )
        else:
            self.embedding = WaveletEmbedding(
                d_model=d,
                n_channels=config.n_channels,
                wavelet_levels=config.wavelet.levels,
                patch_size=config.patch_size,
                stride=config.stride,
            )

        # ========== Backbone ==========
        if config.backbone == "dlinoss":
            self.backbone = DLinOSSBackbone(
                d_model=d,
                n_layers=config.dlinoss.n_layers,
                d_state=config.dlinoss.d_state,
                r_min=config.dlinoss.r_min,
                r_max=config.dlinoss.r_max,
                theta_max=config.dlinoss.theta_max,
                dropout=config.dlinoss.dropout,
                layer_name=config.dlinoss.layer_name,
            )
        elif config.backbone == "cnn1d":
            self.backbone = CNN1DBackbone(
                d_model=d,
                n_layers=config.cnn1d.n_layers,
                kernel_size=config.cnn1d.kernel_size,
                expand=config.cnn1d.expand,
                dropout=config.cnn1d.dropout,
            )
        elif config.backbone == "s4d":
            self.backbone = S4DBackbone(
                d_model=d,
                n_layers=config.s4d.n_layers,
                d_state=config.s4d.d_state,
                dropout=config.s4d.dropout,
            )
        elif config.backbone == "transformer":
            # Use Transformer as backbone (separate from the post-backbone transformer)
            self.backbone = TransformerEncoder(
                d_model=d,
                n_layers=config.transformer.n_layers,
                n_heads=config.transformer.n_heads,
                d_ff=config.transformer.d_ff,
                dropout=config.transformer.dropout,
            )
        else:
            self.backbone = MambaBackbone(
                d_model=d,
                n_layers=config.mamba.n_layers,
                d_state=config.mamba.d_state,
                d_conv=config.mamba.d_conv,
                expand=config.mamba.expand,
            )

        # ========== Mixture of Experts ==========
        self.moe = MixtureOfExperts(
            d_model=d,
            n_experts=config.moe.n_experts,
            top_k=config.moe.top_k,
            d_expert=config.moe.d_expert,
            use_kan=config.moe.use_kan,
            noise_std=config.moe.noise_std,
        )

        # ========== Post-backbone Transformer ==========
        if config.backbone == "transformer":
            # When backbone IS a transformer, skip the redundant post-backbone
            # transformer — backbone already provides global attention
            self.transformer = nn.Identity()
        else:
            self.transformer = TransformerEncoder(
                d_model=d,
                n_layers=config.transformer.n_layers,
                n_heads=config.transformer.n_heads,
                d_ff=config.transformer.d_ff,
                dropout=config.transformer.dropout,
            )

        # ========== VIB Disentanglement ==========
        self.vib = VIBHead(
            d_input=d,
            z_chem_dim=config.vib.z_chem_dim,
            z_inst_dim=config.vib.z_inst_dim,
            beta=config.vib.beta,
        )

        # ========== Task Heads ==========
        # Pretraining: reconstruction (with skip connection from embedding)
        if config.use_raw_embedding:
            # For raw embedding: reconstruct each spectral point
            # d_input = 2*d because we concatenate embedding + backbone tokens
            self.reconstruction_head = PointwiseReconstructionHead(
                d_input=d * 2,
                n_points=config.n_channels,
            )
        else:
            # For patched embedding: reconstruct each patch
            self.reconstruction_head = ReconstructionHead(
                d_input=d * 2,
                n_patches=config.n_patches,
                patch_size=config.patch_size,
            )

        # Fine-tuning: property prediction from z_chem
        self.regression_head = RegressionHead(
            d_input=config.vib.z_chem_dim,
            n_targets=1,
        )

        # Transfer: FNO-based spectral transfer
        self.fno_head = FNOTransferHead(
            d_latent=config.vib.z_chem_dim,
            out_channels=config.n_channels,
            width=config.fno.width,
            modes=config.fno.modes,
            n_layers=config.fno.n_layers,
        )

        # MC Dropout for uncertainty
        self.mc_dropout = nn.Dropout(0.1)

        # Track parameters
        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_name = self.config.backbone.upper()
        print(f"Spektron [{backbone_name}]: {total:,} total params, {trainable:,} trainable")

    def encode(self, spectrum: torch.Tensor,
               domain=None,
               instrument_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Full encoding pipeline: spectrum → disentangled latent.

        Args:
            spectrum: (B, L) raw spectrum
            domain: str, list[str], or None — per-sample domain labels
            instrument_id: (B,) instrument indices

        Returns:
            dict with tokens, z_chem, z_inst, vib outputs, moe_loss
        """
        # 1. Embedding (raw or wavelet)
        tokens = self.embedding(spectrum, domain, instrument_id)

        # 2. Backbone (D-LinOSS or Mamba)
        tokens = self.backbone(tokens)

        # 3. Mixture of Experts
        tokens, moe_loss = self.moe(tokens)

        # 4. Transformer
        tokens = self.transformer(tokens)

        # 5. Extract CLS token for global representation
        cls_token = tokens[:, 0]  # (B, d_model)
        patch_tokens = tokens[:, 2:]  # (B, N, d_model) — skip CLS + domain

        # 6. VIB disentanglement
        vib_out = self.vib(cls_token)

        return {
            "tokens": tokens,
            "patch_tokens": patch_tokens,
            "cls_token": cls_token,
            "z_chem": vib_out["z_chem"],
            "z_inst": vib_out["z_inst"],
            "vib": vib_out,
            "moe_loss": moe_loss,
        }

    def pretrain_forward(self, spectrum: torch.Tensor,
                         mask: torch.Tensor,
                         domain=None,
                         instrument_id: Optional[torch.Tensor] = None) -> Dict:
        """Forward pass for pretraining (MSRP + auxiliary losses).

        Applies mask BEFORE encoding so the model must predict masked
        positions from context (BERT-style), not just copy the input.

        Args:
            spectrum: (B, L) input spectrum
            mask: (B, N) binary mask (1 = masked, 0 = visible)
                  N = n_channels for raw embedding, n_patches for patched
            domain: domain string
            instrument_id: (B,) instrument IDs

        Returns:
            dict with reconstructions, latents, losses
        """
        # 1. Embed the full spectrum to get token representations
        tokens = self.embedding(spectrum, domain, instrument_id)
        # tokens: (B, N+2, d_model) where first 2 are [CLS, DOMAIN]

        # 2. Apply mask to content tokens (skip CLS + DOMAIN at positions 0,1)
        #    Replace masked token embeddings with learnable [MASK] token
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        content_tokens = tokens[:, 2:]      # (B, N, d_model)
        mask_tok = self.mask_token.expand_as(content_tokens)
        # Where mask==1, use mask_token; where mask==0, keep original
        content_tokens = content_tokens * (1 - mask_expanded) + mask_tok * mask_expanded
        tokens = torch.cat([tokens[:, :2], content_tokens], dim=1)

        # Save embedding-level tokens for skip connection to reconstruction head
        embed_tokens = content_tokens.detach().clone()
        # Re-attach to graph for gradient flow through mask_token
        embed_tokens = content_tokens

        # 3. Run masked tokens through backbone → MoE → Transformer → VIB
        tokens = self.backbone(tokens)
        tokens, moe_loss = self.moe(tokens)
        tokens = self.transformer(tokens)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 2:]

        vib_out = self.vib(cls_token)

        # 4. Reconstruct from concatenated [embedding, backbone] tokens
        #    Skip connection lets the decoder access local spectral features
        #    from the embedding alongside global context from the backbone
        recon_input = torch.cat([embed_tokens, patch_tokens], dim=-1)
        reconstruction = self.reconstruction_head(recon_input)

        return {
            "reconstruction": reconstruction,
            "z_chem": vib_out["z_chem"],
            "z_inst": vib_out["z_inst"],
            "vib": vib_out,
            "moe_loss": moe_loss,
            "patch_tokens": patch_tokens,
            "cls_token": cls_token,
        }

    def transfer_forward(self, source_spectrum: torch.Tensor,
                         source_domain: str = "NIR",
                         target_instrument_params: Optional[torch.Tensor] = None,
                         output_length: Optional[int] = None) -> Dict:
        """Forward pass for calibration transfer."""
        enc = self.encode(source_spectrum, source_domain)
        transferred = self.fno_head(
            enc["z_chem"], target_instrument_params, output_length
        )
        prediction = self.regression_head(enc["z_chem"])

        return {
            "transferred_spectrum": transferred,
            "prediction": prediction,
            "z_chem": enc["z_chem"],
            "z_inst": enc["z_inst"],
        }

    def predict(self, spectrum: torch.Tensor,
                domain: str = "NIR",
                mc_samples: int = 1) -> Dict:
        """Prediction with optional MC Dropout uncertainty."""
        if mc_samples <= 1:
            enc = self.encode(spectrum, domain)
            pred = self.regression_head(enc["z_chem"])
            return {"prediction": pred, "z_chem": enc["z_chem"]}

        self.train()
        preds = []
        for _ in range(mc_samples):
            enc = self.encode(spectrum, domain)
            z = self.mc_dropout(enc["z_chem"])
            pred = self.regression_head(z)
            preds.append(pred)
        self.eval()

        preds = torch.stack(preds)
        return {
            "prediction": preds.mean(0),
            "uncertainty": preds.std(0),
            "z_chem": enc["z_chem"],
        }

    def test_time_train(self, test_spectra: torch.Tensor,
                        n_steps: int = 10, lr: float = 1e-4,
                        mask_ratio: float = 0.15,
                        adapt_layers: str = "norm") -> None:
        """Test-Time Training for zero-shot adaptation."""
        if adapt_layers == "norm":
            params = [p for n, p in self.named_parameters()
                      if "norm" in n or "ln" in n]
        elif adapt_layers == "lora":
            params = [p for n, p in self.named_parameters()
                      if "lora" in n.lower()]
        else:
            params = list(self.parameters())

        if not params:
            params = [p for n, p in self.named_parameters() if "norm" in n]

        optimizer = torch.optim.Adam(params, lr=lr)

        self.train()
        for step in range(n_steps):
            batch_size = min(32, len(test_spectra))
            idx = torch.randperm(len(test_spectra))[:batch_size]
            batch = test_spectra[idx]

            # Create mask appropriate to the embedding type
            seq_len = self.config.seq_len
            n_mask = int(seq_len * mask_ratio)
            mask = torch.zeros(batch_size, seq_len, device=batch.device)
            for i in range(batch_size):
                mask_idx = torch.randperm(seq_len)[:n_mask]
                mask[i, mask_idx] = 1

            output = self.pretrain_forward(batch, mask)

            # Self-supervised loss
            target = self._create_reconstruction_target(batch)
            recon_loss = F.mse_loss(
                output["reconstruction"] * mask.unsqueeze(-1),
                target * mask.unsqueeze(-1)
            )

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

        self.eval()

    def _create_reconstruction_target(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Create reconstruction target matching the embedding type.

        For raw embedding: target is the raw spectrum reshaped to (B, L, 1)
        For patched embedding: target is spectrum.unfold() → (B, N_patches, patch_size)
        """
        if self.config.use_raw_embedding:
            return spectrum.unsqueeze(-1)  # (B, L, 1)
        else:
            return self._patchify(spectrum)

    def _patchify(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Convert spectrum to patches for reconstruction target."""
        B, L = spectrum.shape
        p = self.config.patch_size
        s = self.config.stride
        return spectrum.unfold(1, p, s)

    def get_lora_params(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "lora" in n.lower()]

    def freeze_backbone(self):
        """Freeze everything except task heads and LoRA adapters."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.regression_head.parameters():
            param.requires_grad = True
        for param in self.fno_head.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class PointwiseReconstructionHead(nn.Module):
    """Reconstruction head for raw (non-patched) spectral input.

    Each token corresponds to a single spectral point.
    Reconstructs the scalar value at each position.
    """

    def __init__(self, d_input: int = 256, n_points: int = 2048,
                 d_hidden: int = 128):
        super().__init__()
        self.n_points = n_points
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) token representations
        Returns:
            reconstructed: (B, L, 1) per-point reconstruction
        """
        return self.proj(x)


class SpektronForPretraining(nn.Module):
    """Wrapper for pretraining with masking strategy."""

    def __init__(self, model: Spektron, config):
        super().__init__()
        self.model = model
        self.config = config
        self._current_step = 0
        self._max_steps = config.pretrain.max_steps

    def create_mask(self, batch_size: int, seq_len: int,
                    mask_ratio: float = 0.20,
                    mask_type: str = "contiguous",
                    mask_patch_size: int = 3,
                    device: torch.device = None) -> torch.Tensor:
        """Create masking pattern for MSRP.

        Args:
            batch_size: B
            seq_len: sequence length (n_patches or n_channels depending on mode)
            mask_ratio: fraction to mask
            mask_type: "contiguous", "random", or "peak_aware"
            mask_patch_size: size of contiguous mask blocks
            device: target device (avoids CPU→GPU sync)

        Returns:
            mask: (B, seq_len) binary mask (1 = masked)
        """
        mask = torch.zeros(batch_size, seq_len, device=device)
        n_mask = int(seq_len * mask_ratio)

        if mask_type == "random":
            for i in range(batch_size):
                idx = torch.randperm(seq_len, device=device)[:n_mask]
                mask[i, idx] = 1

        elif mask_type == "contiguous":
            for i in range(batch_size):
                n_blocks = max(1, n_mask // mask_patch_size)
                for _ in range(n_blocks):
                    start = torch.randint(0, max(1, seq_len - mask_patch_size + 1), (1,))
                    end = min(start.item() + mask_patch_size, seq_len)
                    mask[i, start:end] = 1

        elif mask_type == "peak_aware":
            for i in range(batch_size):
                idx = torch.randperm(seq_len, device=device)[:n_mask]
                mask[i, idx] = 1

        return mask

    def _get_current_mask_ratio(self) -> float:
        """Progressive mask schedule: ramp from mask_ratio_start to mask_ratio.

        Uses linear warmup over the first 40% of training, then holds at target.
        """
        start = getattr(self.config.pretrain, 'mask_ratio_start',
                        self.config.pretrain.mask_ratio)
        target = self.config.pretrain.mask_ratio
        if start >= target:
            return target
        ramp_steps = int(self._max_steps * 0.4)
        if self._current_step >= ramp_steps:
            return target
        progress = self._current_step / max(ramp_steps, 1)
        return start + (target - start) * progress

    def forward(self, spectrum: torch.Tensor,
                domain=None,
                instrument_id: Optional[torch.Tensor] = None) -> Dict:
        """Pretraining forward pass."""
        B = spectrum.size(0)
        seq_len = self.config.seq_len
        device = spectrum.device

        # Scale mask_patch_size for raw embedding: 3-point blocks are
        # trivially solvable when each token is a single spectral point.
        # Use ~32-point blocks (equivalent information to one patched token).
        mask_patch_size = self.config.pretrain.mask_patch_size
        if self.config.use_raw_embedding:
            mask_patch_size = max(mask_patch_size, self.config.patch_size)

        # Progressive mask schedule
        current_mask_ratio = self._get_current_mask_ratio()

        # Create mask (directly on device to avoid CPU→GPU sync)
        mask = self.create_mask(
            B, seq_len,
            current_mask_ratio,
            self.config.pretrain.mask_type,
            mask_patch_size,
            device=device,
        )

        # Forward
        output = self.model.pretrain_forward(spectrum, mask, domain, instrument_id)
        output["mask"] = mask

        # Target
        target = self.model._create_reconstruction_target(spectrum)
        output["target_patches"] = target

        return output
