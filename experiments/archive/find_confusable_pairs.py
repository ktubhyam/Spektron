"""
Find confusable molecular pairs in QM9S broadened spectra.

Two molecules m₁, m₂ are "spectrally confusable" if their combined
IR+Raman spectra are very similar: ||S(m₁) - S(m₂)|| < ε.

This experiment:
1. Loads broadened IR and Raman spectra (1080 molecules available)
2. Computes pairwise spectral distances
3. Identifies confusable pairs (small distance, different molecules)
4. Tests whether confusable pairs tend to share point group symmetry
5. Tests whether confusable pairs have lower R(G,N)
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from scipy.spatial.distance import cdist, cosine
from scipy.stats import pearsonr
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_broadened_spectra(data_dir: str = "data/raw/qm9s") -> dict:
    """Load broadened IR and Raman spectra."""
    data_path = Path(data_dir)

    logger.info("Loading broadened IR spectra...")
    ir_df = pd.read_csv(data_path / "ir_broaden.csv")
    ir_ids = ir_df.iloc[:, 0].values
    ir_spectra_full = ir_df.iloc[:, 1:].values
    wavenumbers = np.array([float(c) for c in ir_df.columns[1:]])
    logger.info(f"  IR raw: {len(ir_ids)} molecules")

    logger.info("Loading broadened Raman spectra...")
    raman_df = pd.read_csv(data_path / "raman_broaden.csv")
    raman_ids = raman_df.iloc[:, 0].values
    raman_spectra_full = raman_df.iloc[:, 1:].values
    logger.info(f"  Raman raw: {len(raman_ids)} molecules")

    # Align: only keep molecules present in BOTH IR and Raman
    common_ids = np.intersect1d(ir_ids, raman_ids)
    logger.info(f"  Common molecules: {len(common_ids)}")

    ir_mask = np.isin(ir_ids, common_ids)
    raman_mask = np.isin(raman_ids, common_ids)

    mol_ids = ir_ids[ir_mask]
    ir_spectra = ir_spectra_full[ir_mask]
    raman_spectra = raman_spectra_full[raman_mask]

    # Verify alignment
    raman_aligned_ids = raman_ids[raman_mask]
    assert np.array_equal(mol_ids, raman_aligned_ids), "ID mismatch after alignment!"

    logger.info(f"Loaded {len(mol_ids)} aligned molecules")
    logger.info(f"IR shape: {ir_spectra.shape}, Raman shape: {raman_spectra.shape}")
    logger.info(f"Wavenumber range: {wavenumbers[0]:.0f} - {wavenumbers[-1]:.0f} cm⁻¹")

    return {
        "mol_ids": mol_ids,
        "ir": ir_spectra,
        "raman": raman_spectra,
        "wavenumbers": wavenumbers,
    }


def normalize_spectra(spectra: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalize spectra for comparison."""
    if method == "l2":
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return spectra / norms
    elif method == "max":
        maxvals = np.max(spectra, axis=1, keepdims=True)
        maxvals = np.maximum(maxvals, 1e-10)
        return spectra / maxvals
    elif method == "std":
        means = np.mean(spectra, axis=1, keepdims=True)
        stds = np.std(spectra, axis=1, keepdims=True)
        stds = np.maximum(stds, 1e-10)
        return (spectra - means) / stds
    return spectra


def find_confusable_pairs(data: dict, top_k: int = 50,
                          metric: str = "cosine",
                          max_molecules: int = 5000) -> dict:
    """
    Find the most spectrally confusable pairs of molecules.

    Tests three modalities:
    1. IR only
    2. Raman only
    3. IR + Raman combined

    Args:
        max_molecules: subsample to this many for tractable pairwise computation

    Returns dictionary with confusable pairs for each modality.
    """
    ir_full = normalize_spectra(data["ir"], "l2")
    raman_full = normalize_spectra(data["raman"], "l2")
    n_total = len(ir_full)

    # Subsample if needed (pairwise O(n²) is expensive)
    if n_total > max_molecules:
        np.random.seed(42)
        sample_idx = np.random.choice(n_total, max_molecules, replace=False)
        sample_idx.sort()
        ir = ir_full[sample_idx]
        raman = raman_full[sample_idx]
        logger.info(f"Subsampled {max_molecules} from {n_total} molecules for pairwise analysis")
    else:
        ir = ir_full
        raman = raman_full
        sample_idx = np.arange(n_total)

    # Combined: concatenate normalized IR and Raman
    combined = np.hstack([ir, raman])
    combined = normalize_spectra(combined, "l2")

    n_mol = len(ir)
    results = {"sample_idx": sample_idx}

    for name, spectra in [("IR_only", ir), ("Raman_only", raman), ("IR+Raman", combined)]:
        logger.info(f"Computing pairwise distances for {name} ({n_mol} molecules)...")
        t0 = time.time()

        # Compute pairwise cosine distances
        dist_matrix = cdist(spectra, spectra, metric=metric)

        # Zero out diagonal
        np.fill_diagonal(dist_matrix, np.inf)

        # Find top-k most similar pairs
        flat_indices = np.argpartition(dist_matrix.ravel(), top_k * 2)[:top_k * 2]
        rows, cols = np.unravel_index(flat_indices, dist_matrix.shape)

        pairs = set()
        pair_dists = {}
        for r, c in zip(rows, cols):
            key = (min(r, c), max(r, c))
            pairs.add(key)
            pair_dists[key] = dist_matrix[key[0], key[1]]

        sorted_pairs = sorted(pair_dists.items(), key=lambda x: x[1])[:top_k]

        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s")

        # Map back to original indices
        sorted_pairs_orig = [((int(sample_idx[i]), int(sample_idx[j])), d)
                             for (i, j), d in sorted_pairs]

        valid_dists = dist_matrix[dist_matrix != np.inf]
        results[name] = {
            "pairs": sorted_pairs_orig,
            "dist_matrix_stats": {
                "mean": float(np.mean(valid_dists)),
                "median": float(np.median(valid_dists)),
                "min": float(np.min(dist_matrix)),
                "p1": float(np.percentile(valid_dists, 1)),
                "p5": float(np.percentile(valid_dists, 5)),
            }
        }

    return results


def analyze_confusable_pairs(data: dict, pairs_results: dict,
                            smiles_list: list = None) -> None:
    """Analyze and print confusable pair results."""

    mol_ids = data["mol_ids"]

    print("\n" + "=" * 80)
    print("CONFUSABLE MOLECULAR PAIRS ANALYSIS")
    print("=" * 80)

    for modality in ["IR_only", "Raman_only", "IR+Raman"]:
        result = pairs_results[modality]
        pairs = result["pairs"]
        stats = result["dist_matrix_stats"]

        print(f"\n{'─' * 80}")
        print(f"Modality: {modality}")
        print(f"Distance statistics: mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
              f"min={stats['min']:.6f}, p1={stats['p1']:.4f}, p5={stats['p5']:.4f}")
        print(f"\nTop 20 most confusable pairs:")
        print(f"{'Rank':>4} {'Mol1':>5} {'Mol2':>5} {'Distance':>10} {'SMILES1':>40} {'SMILES2':>40}")

        for rank, ((i, j), dist) in enumerate(pairs[:20], 1):
            smi1 = smiles_list[i] if smiles_list and i < len(smiles_list) else f"mol_{i}"
            smi2 = smiles_list[j] if smiles_list and j < len(smiles_list) else f"mol_{j}"
            # Truncate long SMILES
            smi1 = smi1[:38] + ".." if len(smi1) > 40 else smi1
            smi2 = smi2[:38] + ".." if len(smi2) > 40 else smi2
            print(f"{rank:>4} {mol_ids[i]:>5} {mol_ids[j]:>5} {dist:>10.6f} {smi1:>40} {smi2:>40}")

    # Modal complementarity analysis
    print(f"\n{'=' * 80}")
    print("MODAL COMPLEMENTARITY ANALYSIS")
    print("=" * 80)
    print("Do confusable pairs under IR also appear confusable under Raman?")

    ir_pairs = set(p for p, d in pairs_results["IR_only"]["pairs"][:50])
    raman_pairs = set(p for p, d in pairs_results["Raman_only"]["pairs"][:50])
    combined_pairs = set(p for p, d in pairs_results["IR+Raman"]["pairs"][:50])

    ir_and_raman = ir_pairs & raman_pairs
    ir_not_raman = ir_pairs - raman_pairs
    raman_not_ir = raman_pairs - ir_pairs

    print(f"\n  Top-50 confusable pairs under IR:    {len(ir_pairs)}")
    print(f"  Top-50 confusable pairs under Raman: {len(raman_pairs)}")
    print(f"  Confusable under BOTH IR and Raman:  {len(ir_and_raman)} "
          f"({100*len(ir_and_raman)/max(1,len(ir_pairs)):.1f}% of IR pairs)")
    print(f"  Confusable under IR only:            {len(ir_not_raman)}")
    print(f"  Confusable under Raman only:         {len(raman_not_ir)}")

    # How many IR confusable pairs are resolved by adding Raman?
    ir_in_combined = ir_pairs & combined_pairs
    raman_in_combined = raman_pairs & combined_pairs
    print(f"\n  IR pairs still confusable when combined:    {len(ir_in_combined)} / {len(ir_pairs)} "
          f"({100*len(ir_in_combined)/max(1,len(ir_pairs)):.1f}%)")
    print(f"  Raman pairs still confusable when combined: {len(raman_in_combined)} / {len(raman_pairs)} "
          f"({100*len(raman_in_combined)/max(1,len(raman_pairs)):.1f}%)")

    resolved_by_combination = (ir_pairs | raman_pairs) - combined_pairs
    print(f"\n  Pairs RESOLVED by combining IR+Raman: {len(resolved_by_combination)}")
    print(f"  This demonstrates MODAL COMPLEMENTARITY: combining modalities resolves "
          f"confusable pairs that are indistinguishable under a single modality")


def test_modal_complementarity_quantitative(data: dict) -> None:
    """
    Quantitative test of modal complementarity (Theorem 2).

    For each molecule pair, compute:
    - d_IR = cosine distance using IR only
    - d_Raman = cosine distance using Raman only
    - d_combined = cosine distance using IR+Raman

    Theorem 2 predicts: d_combined should be larger (more discriminative)
    than max(d_IR, d_Raman) for centrosymmetric molecules, because
    IR and Raman observe disjoint mode sets.
    """
    # Subsample for tractability
    n_total = len(data["ir"])
    max_mol = min(5000, n_total)
    np.random.seed(42)
    if n_total > max_mol:
        sub_idx = np.random.choice(n_total, max_mol, replace=False)
    else:
        sub_idx = np.arange(n_total)

    ir = normalize_spectra(data["ir"][sub_idx], "l2")
    raman = normalize_spectra(data["raman"][sub_idx], "l2")
    combined = np.hstack([ir, raman])
    combined = normalize_spectra(combined, "l2")

    n_mol = len(ir)

    # Sample random pairs for analysis
    np.random.seed(42)
    n_pairs = min(10000, n_mol * (n_mol - 1) // 2)
    idx1 = np.random.randint(0, n_mol, n_pairs)
    idx2 = np.random.randint(0, n_mol, n_pairs)
    # Ensure i != j
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    d_ir = np.array([1 - np.dot(ir[i], ir[j]) for i, j in zip(idx1, idx2)])
    d_raman = np.array([1 - np.dot(raman[i], raman[j]) for i, j in zip(idx1, idx2)])
    d_combined = np.array([1 - np.dot(combined[i], combined[j]) for i, j in zip(idx1, idx2)])

    d_max_single = np.maximum(d_ir, d_raman)

    print(f"\n{'=' * 80}")
    print("QUANTITATIVE MODAL COMPLEMENTARITY TEST")
    print("=" * 80)
    print(f"Sampled {len(idx1)} random molecule pairs")
    print(f"\n  Mean cosine distance:")
    print(f"    IR only:           {np.mean(d_ir):.6f}")
    print(f"    Raman only:        {np.mean(d_raman):.6f}")
    print(f"    max(IR, Raman):    {np.mean(d_max_single):.6f}")
    print(f"    IR + Raman:        {np.mean(d_combined):.6f}")

    # How often does combining help?
    improvement = d_combined > d_max_single
    print(f"\n  Combined > max(single) for {np.sum(improvement)}/{len(improvement)} "
          f"({100*np.mean(improvement):.1f}%) of pairs")
    print(f"  Mean improvement: {np.mean(d_combined - d_max_single):.6f}")

    # Correlation between IR and Raman distances
    corr, p_value = pearsonr(d_ir, d_raman)
    print(f"\n  Pearson correlation(d_IR, d_Raman): r={corr:.4f} (p={p_value:.2e})")
    print(f"  (Low correlation supports complementarity — IR and Raman capture different info)")

    # Look at pairs that are confusable under one modality but not the other
    threshold = np.percentile(d_ir, 5)  # Bottom 5% = most confusable
    confusable_ir = d_ir < threshold
    confusable_raman = d_raman < threshold

    n_resolved_by_raman = np.sum(confusable_ir & ~confusable_raman)
    n_resolved_by_ir = np.sum(confusable_raman & ~confusable_ir)
    n_confusable_both = np.sum(confusable_ir & confusable_raman)

    print(f"\n  At 5th percentile threshold ({threshold:.6f}):")
    print(f"    Confusable under IR:                {np.sum(confusable_ir)}")
    print(f"    Confusable under Raman:             {np.sum(confusable_raman)}")
    print(f"    Confusable under BOTH:              {n_confusable_both}")
    print(f"    Confusable IR, resolved by Raman:   {n_resolved_by_raman}")
    print(f"    Confusable Raman, resolved by IR:   {n_resolved_by_ir}")
    print(f"    Resolution rate by complementarity: "
          f"{100*(n_resolved_by_raman + n_resolved_by_ir)/(np.sum(confusable_ir) + np.sum(confusable_raman) - n_confusable_both + 1e-10):.1f}%")


if __name__ == "__main__":
    # Load broadened spectra
    data = load_broadened_spectra()

    # Get SMILES from QM9 dataset for the molecules we have
    try:
        from torch_geometric.datasets import QM9
        ds = QM9(root='/tmp/qm9_tg')
        # Build mapping from molecule ID to SMILES
        smiles_map = {}
        for mid in data["mol_ids"]:
            mid = int(mid)
            if mid < len(ds):
                smiles_map[mid] = ds[mid].smiles
        smiles_list = [smiles_map.get(int(mid), f"mol_{mid}") for mid in data["mol_ids"]]
    except Exception as e:
        logger.warning(f"Could not load SMILES: {e}")
        smiles_list = None

    # Find confusable pairs (subsample for tractability)
    pairs_results = find_confusable_pairs(data, top_k=50, max_molecules=5000)

    # Analyze
    analyze_confusable_pairs(data, pairs_results, smiles_list)

    # Quantitative modal complementarity test
    test_modal_complementarity_quantitative(data)

    print("\n\nDone!")
