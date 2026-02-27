"""
Cross-validate confusable pairs with molecular symmetry.

Key questions:
1. Are confusable pairs more likely to share the same point group?
2. Does R(G,N) < 1.0 predict higher confusability?
3. Does centrosymmetry correlate with spectral confusion?

This provides evidence for Theorem 1 (R(G,N) determines observable modes)
and Theorem 2 (modal complementarity for centrosymmetric molecules).
"""

import json
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_point_group_data(pg_path: str, rgn_path: str) -> dict:
    """Load point group assignments and R(G,N) data."""
    with open(pg_path) as f:
        pg_data = json.load(f)

    rgn_data = np.load(rgn_path)

    return {
        "R": rgn_data["R"],              # (130831,) R(G,N) values
        "N": rgn_data["N"],              # (130831,) atom counts
        "d": rgn_data["d"],              # (130831,) DOF = 3N-6
        "n_silent": rgn_data["n_silent"], # (130831,) silent mode counts
        "is_centro": rgn_data["is_centrosymmetric"],  # (130831,) bool
        "pg_counts": pg_data["point_group_counts"],
        "hardest": pg_data["hardest_molecules"],
    }


def load_broadened_spectra(data_dir: str = "data/raw/qm9s") -> dict:
    """Load broadened IR and Raman spectra."""
    data_path = Path(data_dir)

    logger.info("Loading broadened IR spectra...")
    ir_df = pd.read_csv(data_path / "ir_broaden.csv")
    ir_ids = ir_df.iloc[:, 0].values
    ir_spectra = ir_df.iloc[:, 1:].values.astype(np.float32)
    logger.info(f"  IR: {len(ir_ids)} molecules, {ir_spectra.shape[1]} points")

    logger.info("Loading broadened Raman spectra...")
    raman_df = pd.read_csv(data_path / "raman_broaden.csv")
    raman_ids = raman_df.iloc[:, 0].values
    raman_spectra = raman_df.iloc[:, 1:].values.astype(np.float32)
    logger.info(f"  Raman: {len(raman_ids)} molecules, {raman_spectra.shape[1]} points")

    # Align on common IDs
    common_ids = np.intersect1d(ir_ids, raman_ids)
    logger.info(f"  Common molecules: {len(common_ids)}")

    ir_mask = np.isin(ir_ids, common_ids)
    raman_mask = np.isin(raman_ids, common_ids)

    mol_ids = ir_ids[ir_mask]
    ir_aligned = ir_spectra[ir_mask]
    raman_aligned = raman_spectra[raman_mask]

    raman_aligned_ids = raman_ids[raman_mask]
    assert np.array_equal(mol_ids, raman_aligned_ids), "ID mismatch!"

    return {
        "mol_ids": mol_ids,
        "ir": ir_aligned,
        "raman": raman_aligned,
    }


def normalize_l2(spectra: np.ndarray) -> np.ndarray:
    """L2-normalize each spectrum."""
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return spectra / norms


def compute_pairwise_and_analyze(spectra_data: dict, pg_data: dict,
                                  n_sample: int = 5000, n_top_pairs: int = 200):
    """
    Main analysis: compute pairwise distances, find confusable pairs,
    cross-reference with point group data.
    """
    mol_ids = spectra_data["mol_ids"]
    ir = normalize_l2(spectra_data["ir"])
    raman = normalize_l2(spectra_data["raman"])
    n_total = len(mol_ids)

    # Subsample for tractable O(n²)
    np.random.seed(42)
    if n_total > n_sample:
        sample_idx = np.random.choice(n_total, n_sample, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n_total)
        n_sample = n_total

    ir_sub = ir[sample_idx]
    raman_sub = raman[sample_idx]
    ids_sub = mol_ids[sample_idx]

    # Combined IR+Raman
    combined = np.hstack([ir_sub, raman_sub])
    combined = normalize_l2(combined)

    logger.info(f"Analyzing {n_sample} molecules (subsampled from {n_total})")

    # Get R(G,N) and point group info for sampled molecules
    R_values = pg_data["R"][ids_sub.astype(int)]
    N_values = pg_data["N"][ids_sub.astype(int)]
    is_centro = pg_data["is_centro"][ids_sub.astype(int)]
    n_silent = pg_data["n_silent"][ids_sub.astype(int)]

    results = {}

    for name, spec in [("IR_only", ir_sub), ("Raman_only", raman_sub), ("IR+Raman", combined)]:
        logger.info(f"\nComputing pairwise cosine distances for {name}...")
        t0 = time.time()
        dist_matrix = cdist(spec, spec, metric="cosine")
        np.fill_diagonal(dist_matrix, np.inf)
        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s")

        # Find top confusable pairs
        n_pairs_to_find = n_top_pairs * 2  # factor of 2 for (i,j) and (j,i)
        flat_idx = np.argpartition(dist_matrix.ravel(), n_pairs_to_find)[:n_pairs_to_find]
        rows, cols = np.unravel_index(flat_idx, dist_matrix.shape)

        pair_dists = {}
        for r, c in zip(rows, cols):
            key = (min(r, c), max(r, c))
            pair_dists[key] = dist_matrix[key[0], key[1]]

        sorted_pairs = sorted(pair_dists.items(), key=lambda x: x[1])[:n_top_pairs]

        # Distance statistics
        valid_dists = dist_matrix[dist_matrix != np.inf]

        results[name] = {
            "pairs": sorted_pairs,
            "mean_dist": float(np.mean(valid_dists)),
            "median_dist": float(np.median(valid_dists)),
            "p1_dist": float(np.percentile(valid_dists, 1)),
            "p5_dist": float(np.percentile(valid_dists, 5)),
        }

    # ================================================================
    # ANALYSIS 1: Do confusable pairs share properties?
    # ================================================================
    print("\n" + "=" * 80)
    print("SYMMETRY-CONFUSABILITY CROSS-VALIDATION")
    print("=" * 80)

    for modality in ["IR_only", "Raman_only", "IR+Raman"]:
        pairs = results[modality]["pairs"]
        n_pairs = len(pairs)

        print(f"\n{'─' * 80}")
        print(f"Modality: {modality} (top {n_pairs} confusable pairs)")
        print(f"Distance stats: mean={results[modality]['mean_dist']:.4f}, "
              f"p1={results[modality]['p1_dist']:.6f}")

        # For confusable pairs, check:
        same_N = 0       # Same atom count
        same_R = 0       # Same R(G,N)
        both_R1 = 0      # Both have R=1.0
        any_low_R = 0    # At least one has R < 1.0
        both_centro = 0  # Both centrosymmetric
        any_centro = 0   # At least one centrosymmetric
        same_silent = 0  # Same number of silent modes
        delta_N_list = []
        delta_R_list = []

        for (i, j), d in pairs:
            Ni, Nj = N_values[i], N_values[j]
            Ri, Rj = R_values[i], R_values[j]
            ci, cj = is_centro[i], is_centro[j]
            si, sj = n_silent[i], n_silent[j]

            if Ni == Nj:
                same_N += 1
            if abs(Ri - Rj) < 1e-6:
                same_R += 1
            if Ri == 1.0 and Rj == 1.0:
                both_R1 += 1
            if Ri < 1.0 or Rj < 1.0:
                any_low_R += 1
            if ci and cj:
                both_centro += 1
            if ci or cj:
                any_centro += 1
            if si == sj:
                same_silent += 1
            delta_N_list.append(abs(int(Ni) - int(Nj)))
            delta_R_list.append(abs(float(Ri) - float(Rj)))

        print(f"\n  Property sharing in confusable pairs:")
        print(f"    Same atom count (N):     {same_N}/{n_pairs} ({100*same_N/n_pairs:.1f}%)")
        print(f"    Same R(G,N):             {same_R}/{n_pairs} ({100*same_R/n_pairs:.1f}%)")
        print(f"    Both R=1.0:              {both_R1}/{n_pairs} ({100*both_R1/n_pairs:.1f}%)")
        print(f"    Any R<1.0 (silent modes): {any_low_R}/{n_pairs} ({100*any_low_R/n_pairs:.1f}%)")
        print(f"    Both centrosymmetric:    {both_centro}/{n_pairs} ({100*both_centro/n_pairs:.1f}%)")
        print(f"    Any centrosymmetric:     {any_centro}/{n_pairs} ({100*any_centro/n_pairs:.1f}%)")
        print(f"    Same #silent modes:      {same_silent}/{n_pairs} ({100*same_silent/n_pairs:.1f}%)")
        print(f"    Mean |ΔN|:               {np.mean(delta_N_list):.1f}")
        print(f"    Mean |ΔR|:               {np.mean(delta_R_list):.4f}")

        # Compare to random baseline
        np.random.seed(123)
        n_random = 10000
        ri = np.random.randint(0, n_sample, n_random)
        rj = np.random.randint(0, n_sample, n_random)
        mask = ri != rj
        ri, rj = ri[mask], rj[mask]

        random_same_N = np.sum(N_values[ri] == N_values[rj])
        random_same_R = np.sum(np.abs(R_values[ri] - R_values[rj]) < 1e-6)
        random_both_R1 = np.sum((R_values[ri] == 1.0) & (R_values[rj] == 1.0))
        random_any_low_R = np.sum((R_values[ri] < 1.0) | (R_values[rj] < 1.0))
        random_any_centro = np.sum(is_centro[ri] | is_centro[rj])
        random_delta_N = np.mean(np.abs(N_values[ri].astype(int) - N_values[rj].astype(int)))

        n_rand = len(ri)
        print(f"\n  Random pair baseline ({n_rand} pairs):")
        print(f"    Same N:     {random_same_N}/{n_rand} ({100*random_same_N/n_rand:.1f}%)")
        print(f"    Same R:     {random_same_R}/{n_rand} ({100*random_same_R/n_rand:.1f}%)")
        print(f"    Both R=1:   {random_both_R1}/{n_rand} ({100*random_both_R1/n_rand:.1f}%)")
        print(f"    Any R<1:    {random_any_low_R}/{n_rand} ({100*random_any_low_R/n_rand:.1f}%)")
        print(f"    Any centro: {random_any_centro}/{n_rand} ({100*random_any_centro/n_rand:.1f}%)")
        print(f"    Mean |ΔN|:  {random_delta_N:.1f}")

        # Fisher exact test: are confusable pairs more likely to share N?
        # Contingency table: [confusable & same_N, confusable & diff_N;
        #                     random & same_N, random & diff_N]
        table = [[same_N, n_pairs - same_N],
                 [int(random_same_N), n_rand - int(random_same_N)]]
        odds, p_val = fisher_exact(table)
        print(f"\n  Fisher exact test (same N enrichment in confusable pairs):")
        print(f"    Odds ratio: {odds:.2f}, p-value: {p_val:.2e}")

    # ================================================================
    # ANALYSIS 2: Distance as a function of R(G,N) and centrosymmetry
    # ================================================================
    print(f"\n{'=' * 80}")
    print("DISTANCE CORRELATIONS WITH MOLECULAR PROPERTIES")
    print("=" * 80)

    # For each molecule, compute its minimum distance to any other molecule
    for name in ["IR_only", "Raman_only", "IR+Raman"]:
        spec = {"IR_only": ir_sub, "Raman_only": raman_sub, "IR+Raman": combined}[name]
        dist_matrix = cdist(spec, spec, metric="cosine")
        np.fill_diagonal(dist_matrix, np.inf)

        min_dists = np.min(dist_matrix, axis=1)  # (n_sample,)
        mean_dists = np.mean(dist_matrix, axis=1)

        print(f"\n  {name}:")

        # Correlation between R(G,N) and minimum distance
        # Only meaningful if there's variance in R
        r_var = np.var(R_values)
        if r_var > 1e-10:
            corr_R_mindist, p_R_mindist = pearsonr(R_values, min_dists)
            print(f"    Corr(R(G,N), min_dist): r={corr_R_mindist:.4f}, p={p_R_mindist:.2e}")
        else:
            print(f"    Corr(R(G,N), min_dist): insufficient variance in R (var={r_var:.6f})")

        # Correlation between atom count and min distance
        corr_N, p_N = pearsonr(N_values, min_dists)
        print(f"    Corr(N_atoms, min_dist): r={corr_N:.4f}, p={p_N:.2e}")

        # Centrosymmetric vs non-centrosymmetric
        centro_mask = is_centro.astype(bool)
        if np.sum(centro_mask) > 0 and np.sum(~centro_mask) > 0:
            centro_mindists = min_dists[centro_mask]
            noncentro_mindists = min_dists[~centro_mask]
            stat, p_mw = mannwhitneyu(centro_mindists, noncentro_mindists, alternative='less')
            print(f"    Centrosymmetric min_dist: {np.mean(centro_mindists):.6f} (n={len(centro_mindists)})")
            print(f"    Non-centrosymmetric:      {np.mean(noncentro_mindists):.6f} (n={len(noncentro_mindists)})")
            print(f"    Mann-Whitney U (centro < non-centro): p={p_mw:.2e}")
        else:
            print(f"    No centrosymmetric molecules in sample")

    # ================================================================
    # ANALYSIS 3: Modal complementarity for centrosymmetric molecules
    # ================================================================
    print(f"\n{'=' * 80}")
    print("MODAL COMPLEMENTARITY BY CENTROSYMMETRY")
    print("=" * 80)

    # For centrosymmetric molecules: IR and Raman should be more complementary
    # (mutual exclusion rule). Compare IR-Raman distance correlation
    # for centro vs non-centro molecules.

    ir_dists = cdist(ir_sub, ir_sub, metric="cosine")
    raman_dists = cdist(raman_sub, raman_sub, metric="cosine")
    np.fill_diagonal(ir_dists, np.inf)
    np.fill_diagonal(raman_dists, np.inf)

    # Sample random pairs
    np.random.seed(42)
    n_pairs_sample = min(50000, n_sample * (n_sample - 1) // 2)
    pi = np.random.randint(0, n_sample, n_pairs_sample)
    pj = np.random.randint(0, n_sample, n_pairs_sample)
    valid = pi != pj
    pi, pj = pi[valid], pj[valid]

    ir_d = ir_dists[pi, pj]
    raman_d = raman_dists[pi, pj]

    # Split pairs by centrosymmetry
    both_centro = is_centro[pi] & is_centro[pj]
    neither_centro = ~is_centro[pi] & ~is_centro[pj]

    if np.sum(both_centro) > 10:
        corr_centro, p_centro = pearsonr(ir_d[both_centro], raman_d[both_centro])
        print(f"\n  Both centrosymmetric pairs ({np.sum(both_centro)}):")
        print(f"    Corr(d_IR, d_Raman): r={corr_centro:.4f}, p={p_centro:.2e}")
    else:
        print(f"\n  Too few both-centrosymmetric pairs ({np.sum(both_centro)})")

    corr_noncentro, p_noncentro = pearsonr(ir_d[neither_centro], raman_d[neither_centro])
    print(f"  Neither centrosymmetric pairs ({np.sum(neither_centro)}):")
    print(f"    Corr(d_IR, d_Raman): r={corr_noncentro:.4f}, p={p_noncentro:.2e}")

    corr_all, p_all = pearsonr(ir_d, raman_d)
    print(f"  All pairs ({len(pi)}):")
    print(f"    Corr(d_IR, d_Raman): r={corr_all:.4f}, p={p_all:.2e}")

    print(f"\n  Interpretation:")
    print(f"  For centrosymmetric molecules, IR and Raman observe DISJOINT mode sets")
    print(f"  (mutual exclusion rule), so d_IR and d_Raman should be LESS correlated.")
    print(f"  Lower correlation = stronger complementarity.")

    # ================================================================
    # ANALYSIS 4: Atom count distribution in confusable pairs
    # ================================================================
    print(f"\n{'=' * 80}")
    print("CONFUSABLE PAIR STRUCTURE ANALYSIS")
    print("=" * 80)

    combined_pairs = results["IR+Raman"]["pairs"]

    print(f"\nTop 20 most confusable pairs (IR+Raman combined):")
    print(f"{'Rank':>4} {'ID_1':>7} {'ID_2':>7} {'Dist':>10} {'N1':>3} {'N2':>3} "
          f"{'R1':>5} {'R2':>5} {'Centro':>8}")
    for rank, ((i, j), d) in enumerate(combined_pairs[:20], 1):
        id1, id2 = ids_sub[i], ids_sub[j]
        n1, n2 = int(N_values[i]), int(N_values[j])
        r1, r2 = R_values[i], R_values[j]
        c1, c2 = is_centro[i], is_centro[j]
        centro_str = ""
        if c1 and c2:
            centro_str = "both"
        elif c1 or c2:
            centro_str = "one"
        else:
            centro_str = "none"
        print(f"{rank:>4} {id1:>7} {id2:>7} {d:>10.6f} {n1:>3} {n2:>3} "
              f"{r1:>5.2f} {r2:>5.2f} {centro_str:>8}")

    # Atom count difference distribution for confusable vs random
    confusable_dN = [abs(int(N_values[i]) - int(N_values[j]))
                     for (i, j), d in combined_pairs]
    random_dN = np.abs(N_values[pi].astype(int) - N_values[pj].astype(int))

    print(f"\n  Atom count difference distribution:")
    print(f"    Confusable pairs: mean |ΔN| = {np.mean(confusable_dN):.2f}, "
          f"median = {np.median(confusable_dN):.0f}")
    print(f"    Random pairs:     mean |ΔN| = {np.mean(random_dN):.2f}, "
          f"median = {np.median(random_dN):.0f}")

    for dN_threshold in [0, 1, 2]:
        confusable_frac = np.mean(np.array(confusable_dN) <= dN_threshold)
        random_frac = np.mean(random_dN <= dN_threshold)
        print(f"    |ΔN| ≤ {dN_threshold}: confusable={confusable_frac:.1%}, random={random_frac:.1%}")

    print("\n\nDone!")


if __name__ == "__main__":
    # Load data
    pg_data = load_point_group_data(
        "experiments/results/qm9_point_groups.json",
        "experiments/results/qm9_rgn_data.npz",
    )

    spectra_data = load_broadened_spectra()

    # Run analysis
    compute_pairwise_and_analyze(spectra_data, pg_data, n_sample=5000, n_top_pairs=200)
