#!/usr/bin/env python3
"""
SpectralFM v2 — Smoke Test Suite

Run: python tests/smoke_test.py
Tests each module independently, then tests the full forward/backward pass.
Fix issues module by module until all tests pass.
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️"

results = []

def test(name, fn):
    """Run a test function and record result."""
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()
        print()


def test_imports():
    """Test that all modules import correctly."""
    import torch
    print(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

def test_config():
    from src.config import SpectralFMConfig
    cfg = SpectralFMConfig()
    assert cfg.d_model == 256
    assert cfg.n_channels == 2048
    assert cfg.mamba.n_layers == 4
    assert cfg.transformer.n_layers == 2

def test_data_corn():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "processed", "corn")
    m5 = np.load(os.path.join(data_dir, "m5_spectra.npy"))
    mp6 = np.load(os.path.join(data_dir, "mp6_spectra.npy"))
    props = np.load(os.path.join(data_dir, "properties.npy"))
    wl = np.load(os.path.join(data_dir, "wavelengths.npy"))
    assert m5.shape == (80, 700), f"m5 shape: {m5.shape}"
    assert mp6.shape == (80, 700), f"mp6 shape: {mp6.shape}"
    assert props.shape == (80, 4), f"props shape: {props.shape}"
    assert wl.shape == (700,), f"wavelengths shape: {wl.shape}"

def test_data_tablet():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "processed", "tablet")
    cal1 = np.load(os.path.join(data_dir, "calibrate_1.npy"))
    cal2 = np.load(os.path.join(data_dir, "calibrate_2.npy"))
    calY = np.load(os.path.join(data_dir, "calibrate_Y.npy"))
    assert cal1.shape == (155, 650), f"cal1 shape: {cal1.shape}"
    assert cal2.shape == (155, 650), f"cal2 shape: {cal2.shape}"
    assert calY.shape == (155, 3), f"calY shape: {calY.shape}"

def test_wavelet_embedding():
    import torch
    from src.config import SpectralFMConfig
    from src.models.embedding import WaveletEmbedding
    
    cfg = SpectralFMConfig()
    embed = WaveletEmbedding(
        d_model=cfg.d_model,
        n_channels=cfg.n_channels,
        wavelet_levels=cfg.wavelet.levels,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
    )
    x = torch.randn(2, 2048)
    tokens = embed(x, domain="NIR")
    print(f"    WaveletEmbedding output: {tokens.shape}")
    # Should be (2, N+2, 256) where N = num patches
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == cfg.d_model
    actual_n_patches = tokens.shape[1] - 2  # minus CLS + domain
    print(f"    Actual n_patches: {actual_n_patches} (config says {cfg.n_patches})")

def test_mamba():
    import torch
    from src.models.mamba import MambaBackbone
    
    mamba = MambaBackbone(d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 64, 256)  # (B, L, D)
    y = mamba(x)
    assert y.shape == x.shape, f"Mamba output {y.shape} != input {x.shape}"

def test_moe():
    import torch
    from src.models.moe import MixtureOfExperts
    
    moe = MixtureOfExperts(d_model=256, n_experts=4, top_k=2, d_expert=512)
    x = torch.randn(2, 64, 256)
    y, balance_loss = moe(x)
    assert y.shape == x.shape, f"MoE output {y.shape} != input {x.shape}"
    assert balance_loss.dim() == 0, "Balance loss should be scalar"

def test_transformer():
    import torch
    from src.models.transformer import TransformerEncoder
    
    tf = TransformerEncoder(d_model=256, n_layers=2, n_heads=8, d_ff=1024)
    x = torch.randn(2, 64, 256)
    y = tf(x)
    assert y.shape == x.shape, f"Transformer output {y.shape} != input {x.shape}"

def test_vib_head():
    import torch
    from src.models.heads import VIBHead
    
    vib = VIBHead(d_input=256, z_chem_dim=128, z_inst_dim=64)
    x = torch.randn(2, 256)
    out = vib(x)
    assert out["z_chem"].shape == (2, 128), f"z_chem: {out['z_chem'].shape}"
    assert out["z_inst"].shape == (2, 64), f"z_inst: {out['z_inst'].shape}"
    assert "kl_loss" in out

def test_reconstruction_head():
    import torch
    from src.models.heads import ReconstructionHead
    from src.config import SpectralFMConfig
    
    cfg = SpectralFMConfig()
    head = ReconstructionHead(d_input=256, n_patches=cfg.n_patches, patch_size=cfg.patch_size)
    x = torch.randn(2, cfg.n_patches, 256)
    y = head(x)
    print(f"    ReconstructionHead output: {y.shape}")
    assert y.shape == (2, cfg.n_patches, cfg.patch_size)

def test_fno_head():
    import torch
    from src.models.heads import FNOTransferHead
    
    fno = FNOTransferHead(d_latent=128, out_channels=2048, width=64, modes=32, n_layers=4)
    z = torch.randn(2, 128)
    y = fno(z)
    print(f"    FNO output: {y.shape}")
    assert y.shape == (2, 2048), f"FNO output {y.shape} expected (2, 2048)"

def test_losses():
    import torch
    from src.losses.losses import MSRPLoss, PhysicsLoss
    
    # MSRP
    msrp = MSRPLoss()
    pred = torch.randn(2, 64, 32)
    target = torch.randn(2, 64, 32)
    mask = torch.zeros(2, 64)
    mask[:, :10] = 1  # Mask first 10 patches
    loss = msrp(pred, target, mask)
    assert loss.dim() == 0 and loss.item() > 0

def test_full_forward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    x = torch.randn(2, 2048)
    
    # Encode
    enc = model.encode(x, domain="NIR")
    print(f"    Encode output keys: {list(enc.keys())}")
    print(f"    z_chem: {enc['z_chem'].shape}, z_inst: {enc['z_inst'].shape}")
    print(f"    tokens: {enc['tokens'].shape}")

def test_full_backward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    x = torch.randn(2, 2048)
    
    enc = model.encode(x, domain="NIR")
    loss = enc["z_chem"].sum() + enc["moe_loss"]
    loss.backward()
    
    # Check gradients exist
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"    Gradients: {n_grad}/{n_total} parameters have gradients")
    assert n_grad > 0, "No gradients!"

def test_pretrain_forward():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
    
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)
    pretrain_model = SpectralFMForPretraining(model, cfg)
    
    x = torch.randn(2, 2048)
    output = pretrain_model(x, domain="NIR")
    
    print(f"    Pretrain output keys: {list(output.keys())}")
    print(f"    Reconstruction: {output['reconstruction'].shape}")
    print(f"    Target patches: {output['target_patches'].shape}")
    print(f"    Mask: {output['mask'].shape}")

def test_ttt():
    import torch
    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM

    cfg = SpectralFMConfig()
    model = SpectralFM(cfg)

    test_spectra = torch.randn(10, 2048)
    model.test_time_train(test_spectra, n_steps=2, lr=1e-4)
    print("    TTT completed without error")


def test_wavelet_pywt():
    """Test that wavelet embedding uses pywt and produces correct shapes."""
    import torch
    from src.models.embedding import WaveletEmbedding

    emb = WaveletEmbedding(d_model=64, n_channels=2048, wavelet_levels=4,
                           patch_size=32, stride=16)
    x = torch.randn(2, 2048)
    tokens = emb(x)
    expected_n_patches = (2048 - 32) // 16 + 1  # 127
    expected_shape = (2, expected_n_patches + 2, 64)  # +2 for CLS + domain
    assert tokens.shape == expected_shape, f"Wrong shape: {tokens.shape}, expected {expected_shape}"
    assert torch.isfinite(tokens).all()
    print(f"    pywt wavelet output: {tokens.shape}")


def test_lora_injection():
    """Test LoRA injection and forward pass."""
    import torch
    from src.config import get_light_config
    from src.models.spectral_fm import SpectralFM
    from src.models.lora import inject_lora, get_lora_state_dict

    config = get_light_config()
    model = SpectralFM(config)
    total_before = sum(p.numel() for p in model.parameters())

    inject_lora(model, ["q_proj", "k_proj", "v_proj"], rank=4, alpha=8)

    total_after = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    assert lora_params > 0, "No LoRA params found!"
    print(f"    LoRA params: {lora_params:,} (added {total_after - total_before:,})")

    # Forward still works
    x = torch.randn(2, 2048)
    model.eval()
    with torch.no_grad():
        out = model.encode(x)
    assert out["z_chem"].shape == (2, config.vib.z_chem_dim)

    # Freeze + LoRA stays trainable
    model.freeze_backbone()
    lora_trainable = sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and 'lora_' in n)
    assert lora_trainable > 0, "No LoRA params trainable after freeze"

    # State dict extraction
    lora_sd = get_lora_state_dict(model)
    assert len(lora_sd) > 0
    print(f"    LoRA state dict: {len(lora_sd)} keys")


def test_logger():
    """Test dual logging (JSON-only mode)."""
    import tempfile, json
    from src.utils.logging import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_logger = ExperimentLogger(
            project="test", run_name="smoke",
            use_wandb=False, log_dir=tmpdir
        )
        exp_logger.log({"loss": 1.0}, step=0)
        exp_logger.log({"loss": 0.5}, step=1)
        exp_logger.finish()

        log_file = f"{tmpdir}/smoke.jsonl"
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"
        entry = json.loads(lines[0])
        assert "loss" in entry
        assert "_step" in entry
    print("    JSON logging OK")


# ===== D-LinOSS Tests =====

def test_dlinoss_config():
    """Test D-LinOSS config creation."""
    from src.config import get_dlinoss_config, get_light_dlinoss_config
    cfg = get_dlinoss_config()
    assert cfg.backbone == "dlinoss"
    assert cfg.embedding_type == "raw"
    assert cfg.d_model == 256
    assert cfg.dlinoss.d_state == 128
    assert cfg.use_raw_embedding == True
    assert cfg.seq_len == 2048

    light = get_light_dlinoss_config()
    assert light.backbone == "dlinoss"
    assert light.n_channels == 256
    assert light.seq_len == 256
    print(f"    D-LinOSS config: d_model={cfg.d_model}, d_state={cfg.dlinoss.d_state}, layers={cfg.dlinoss.n_layers}")


def test_linoss_scan():
    """Test the vendored parallel associative scan."""
    import torch
    from src.models.linoss.scan import associative_scan

    # Simple prefix sum test
    def add(a, b):
        return a + b

    x = torch.arange(1, 9, dtype=torch.float32)  # [1, 2, 3, 4, 5, 6, 7, 8]
    result = associative_scan(add, x, axis=0)
    expected = torch.cumsum(x, dim=0)  # [1, 3, 6, 10, 15, 21, 28, 36]
    assert torch.allclose(result, expected), f"Scan failed: {result} vs {expected}"
    print(f"    Prefix sum scan: {result.tolist()}")


def test_damped_layer():
    """Test DampedLayer forward pass."""
    import torch
    from src.models.linoss.layers import DampedLayer

    layer = DampedLayer(state_dim=32, hidden_dim=64, r_min=0.9, r_max=1.0, theta_max=3.14159)
    x = torch.randn(2, 50, 64)  # (B, L, H)
    y = layer(x)
    assert y.shape == x.shape, f"DampedLayer: {y.shape} != {x.shape}"
    assert torch.isfinite(y).all(), "DampedLayer has non-finite outputs"

    # Check learned frequency extraction
    freqs = layer.learned_frequencies
    assert freqs.shape == (32,), f"Frequencies shape: {freqs.shape}"
    print(f"    DampedLayer: {x.shape} -> {y.shape}, {freqs.shape[0]} frequencies")


def test_dlinoss_backbone():
    """Test full DLinOSSBackbone."""
    import torch
    from src.models.dlinoss import DLinOSSBackbone

    backbone = DLinOSSBackbone(d_model=64, n_layers=2, d_state=32, dropout=0.05)
    x = torch.randn(2, 100, 64)  # (B, L, H)
    y = backbone(x)
    assert y.shape == x.shape, f"DLinOSSBackbone: {y.shape} != {x.shape}"

    # Check frequency extraction
    freqs = backbone.learned_frequencies
    assert len(freqs) == 2, f"Expected 2 layers of frequencies, got {len(freqs)}"
    print(f"    DLinOSSBackbone: {x.shape} -> {y.shape}")


def test_raw_embedding():
    """Test RawSpectralEmbedding (no patching)."""
    import torch
    from src.models.embedding import RawSpectralEmbedding

    embed = RawSpectralEmbedding(d_model=64, n_channels=256, kernel_size=15)
    x = torch.randn(2, 256)
    tokens = embed(x, domain="IR")
    expected_shape = (2, 256 + 2, 64)  # +2 for CLS + domain
    assert tokens.shape == expected_shape, f"Raw embed: {tokens.shape}, expected {expected_shape}"
    assert torch.isfinite(tokens).all()
    print(f"    RawSpectralEmbedding: (2, 256) -> {tokens.shape}")


def test_dlinoss_full_forward():
    """Test full SpectralFM with D-LinOSS backbone."""
    import torch
    from src.config import get_light_dlinoss_config
    from src.models.spectral_fm import SpectralFM

    cfg = get_light_dlinoss_config()
    model = SpectralFM(cfg)
    x = torch.randn(2, cfg.n_channels)  # (B, 256)

    enc = model.encode(x, domain="IR")
    print(f"    D-LinOSS encode: z_chem={enc['z_chem'].shape}, tokens={enc['tokens'].shape}")
    assert enc['z_chem'].shape == (2, cfg.vib.z_chem_dim)
    assert enc['z_inst'].shape == (2, cfg.vib.z_inst_dim)


def test_dlinoss_full_backward():
    """Test full backward pass through D-LinOSS model."""
    import torch
    from src.config import get_light_dlinoss_config
    from src.models.spectral_fm import SpectralFM

    cfg = get_light_dlinoss_config()
    model = SpectralFM(cfg)
    x = torch.randn(2, cfg.n_channels)

    enc = model.encode(x, domain="RAMAN")
    loss = enc["z_chem"].sum() + enc["moe_loss"]
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"    D-LinOSS backward: {n_grad}/{n_total} params have gradients")
    assert n_grad > 0, "No gradients in D-LinOSS model!"


def test_dlinoss_pretrain():
    """Test D-LinOSS pretraining forward pass."""
    import torch
    from src.config import get_light_dlinoss_config
    from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining

    cfg = get_light_dlinoss_config()
    model = SpectralFM(cfg)
    pretrain = SpectralFMForPretraining(model, cfg)

    x = torch.randn(2, cfg.n_channels)
    output = pretrain(x, domain="IR")

    print(f"    D-LinOSS pretrain: recon={output['reconstruction'].shape}, mask={output['mask'].shape}")
    assert output['reconstruction'].shape[0] == 2
    assert output['mask'].shape == (2, cfg.seq_len)


def test_per_sample_domain():
    """Test per-sample domain embedding with mixed modality batches."""
    import torch
    from src.models.embedding import RawSpectralEmbedding, WaveletEmbedding

    # Raw embedding with list of domains
    embed_raw = RawSpectralEmbedding(d_model=64, n_channels=256, kernel_size=15)
    x = torch.randn(4, 256)
    domains = ["IR", "RAMAN", "IR", "RAMAN"]
    tokens = embed_raw(x, domain=domains)
    assert tokens.shape == (4, 258, 64)
    # Domain tokens at position 1 should differ between IR and RAMAN
    assert not torch.allclose(tokens[0, 1], tokens[1, 1]), "IR and RAMAN domain tokens should differ"
    print(f"    Per-sample domain (raw): (4, 256) + {domains} -> {tokens.shape}")

    # Wavelet embedding with list of domains
    embed_wav = WaveletEmbedding(d_model=64, n_channels=2048, wavelet_levels=4, patch_size=32, stride=16)
    x2 = torch.randn(2, 2048)
    domains2 = ["NIR", "IR"]
    tokens2 = embed_wav(x2, domain=domains2)
    assert tokens2.shape[0] == 2
    assert not torch.allclose(tokens2[0, 1], tokens2[1, 1]), "NIR and IR domain tokens should differ"
    print(f"    Per-sample domain (wavelet): (2, 2048) + {domains2} -> {tokens2.shape}")


def test_mask_scaling():
    """Test that raw mode auto-scales mask_patch_size."""
    import torch
    from src.config import get_light_dlinoss_config
    from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining

    cfg = get_light_dlinoss_config()
    assert cfg.pretrain.mask_patch_size == 3, "Default should be 3"

    model = SpectralFM(cfg)
    pretrain = SpectralFMForPretraining(model, cfg)

    x = torch.randn(2, cfg.n_channels)
    output = pretrain(x, domain="IR")

    # Mask should exist and have contiguous blocks >= patch_size
    mask = output['mask']
    assert mask.shape == (2, cfg.n_channels)
    # Check that masked regions form blocks of at least patch_size
    # (not isolated 3-point blocks)
    for i in range(mask.shape[0]):
        masked_indices = torch.where(mask[i] == 1)[0]
        if len(masked_indices) > 1:
            diffs = masked_indices[1:] - masked_indices[:-1]
            max_consecutive = 1
            current = 1
            for d in diffs:
                if d == 1:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 1
            assert max_consecutive >= cfg.patch_size, \
                f"Expected contiguous blocks >= {cfg.patch_size}, got {max_consecutive}"
            break
    print(f"    Mask scaling: patch_size={cfg.patch_size}, mask blocks are contiguous")


def test_n_patches_computed():
    """Test that n_patches is correctly computed from n_channels/patch_size/stride."""
    from src.config import SpectralFMConfig, get_light_dlinoss_config

    cfg = SpectralFMConfig()
    assert cfg.n_patches == 127, f"Default: expected 127, got {cfg.n_patches}"

    cfg2 = get_light_dlinoss_config()
    expected = (256 - 32) // 16 + 1  # = 15
    assert cfg2.n_patches == expected, f"Light D-LinOSS: expected {expected}, got {cfg2.n_patches}"
    print(f"    n_patches computed: default={SpectralFMConfig().n_patches}, light_dlinoss={cfg2.n_patches}")


if __name__ == "__main__":
    print("=" * 60)
    print("SpectralFM v2 — Smoke Test Suite")
    print("=" * 60)

    print("\n--- 1. Imports ---")
    test("imports", test_imports)

    print("\n--- 2. Config ---")
    test("config", test_config)

    print("\n--- 3. Data ---")
    test("corn data", test_data_corn)
    test("tablet data", test_data_tablet)

    print("\n--- 4. Individual Modules (Mamba) ---")
    test("wavelet_embedding", test_wavelet_embedding)
    test("mamba", test_mamba)
    test("moe", test_moe)
    test("transformer", test_transformer)
    test("vib_head", test_vib_head)
    test("reconstruction_head", test_reconstruction_head)
    test("fno_head", test_fno_head)

    print("\n--- 5. Losses ---")
    test("losses", test_losses)

    print("\n--- 6. Full Model (Mamba) ---")
    test("full_forward", test_full_forward)
    test("full_backward", test_full_backward)
    test("pretrain_forward", test_pretrain_forward)

    print("\n--- 7. Test-Time Training ---")
    test("ttt", test_ttt)

    print("\n--- 8. P2: Architecture Fixes ---")
    test("wavelet_pywt", test_wavelet_pywt)
    test("lora_injection", test_lora_injection)
    test("logger", test_logger)

    print("\n--- 9. D-LinOSS Backbone ---")
    test("dlinoss_config", test_dlinoss_config)
    test("linoss_scan", test_linoss_scan)
    test("damped_layer", test_damped_layer)
    test("dlinoss_backbone", test_dlinoss_backbone)
    test("raw_embedding", test_raw_embedding)
    test("dlinoss_full_forward", test_dlinoss_full_forward)
    test("dlinoss_full_backward", test_dlinoss_full_backward)
    test("dlinoss_pretrain", test_dlinoss_pretrain)

    print("\n--- 10. Bug Fixes ---")
    test("per_sample_domain", test_per_sample_domain)
    test("mask_scaling", test_mask_scaling)
    test("n_patches_computed", test_n_patches_computed)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    print(f"  {PASS} Passed: {passed}")
    print(f"  {FAIL} Failed: {failed}")
    print(f"  Total: {passed + failed}")

    if failed > 0:
        print(f"\nFailed tests:")
        for name, status, err in results:
            if status == FAIL:
                print(f"  {FAIL} {name}: {err}")

    sys.exit(0 if failed == 0 else 1)
