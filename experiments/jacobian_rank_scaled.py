"""
Scaled Jacobian rank analysis using QM9 molecular geometries.

Extends the basic jacobian_rank.py by:
1. Using real QM9 molecular geometries (not random G matrices)
2. Building G matrices from actual bond/angle internal coordinates
3. Testing 1000+ molecules across varied symmetries
4. Analyzing rank as a function of molecule size and symmetry
"""

import numpy as np
from numpy.linalg import eigh, svd
import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Atomic masses (amu)
ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.904,
}


def build_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix."""
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def find_bonds(positions: np.ndarray, atomic_numbers: list,
               threshold_factor: float = 1.3) -> list:
    """
    Find bonds based on interatomic distances.
    A bond exists if distance < threshold_factor * sum_of_covalent_radii.
    """
    COVALENT_RADII = {
        1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
    }

    n = len(atomic_numbers)
    dist_mat = build_distance_matrix(positions)
    bonds = []

    for i in range(n):
        for j in range(i + 1, n):
            r_i = COVALENT_RADII.get(atomic_numbers[i], 1.0)
            r_j = COVALENT_RADII.get(atomic_numbers[j], 1.0)
            threshold = threshold_factor * (r_i + r_j)
            if dist_mat[i, j] < threshold:
                bonds.append((i, j, dist_mat[i, j]))

    return bonds


def find_angles(bonds: list) -> list:
    """Find bond angles from bond list. Returns (i, j, k) where j is the center atom."""
    from collections import defaultdict
    neighbors = defaultdict(set)
    for i, j, _ in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    angles = []
    for center in neighbors:
        nbrs = sorted(neighbors[center])
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                angles.append((nbrs[a_idx], center, nbrs[b_idx]))

    return angles


def build_b_matrix_row_bond(positions: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Build B-matrix row for a bond stretch between atoms i and j.
    B_s,x = ∂s/∂x where s is the bond length.
    """
    n = len(positions)
    b_row = np.zeros(3 * n)
    rij = positions[j] - positions[i]
    dist = np.linalg.norm(rij)
    e_ij = rij / dist

    b_row[3*i:3*i+3] = -e_ij
    b_row[3*j:3*j+3] = e_ij

    return b_row


def build_b_matrix_row_angle(positions: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """
    Build B-matrix row for a bond angle i-j-k (j is center atom).
    """
    n = len(positions)
    b_row = np.zeros(3 * n)

    rji = positions[i] - positions[j]
    rjk = positions[k] - positions[j]
    r_ji = np.linalg.norm(rji)
    r_jk = np.linalg.norm(rjk)
    e_ji = rji / r_ji
    e_jk = rjk / r_jk

    cos_theta = np.clip(np.dot(e_ji, e_jk), -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)
    sin_theta = max(sin_theta, 1e-10)

    # Wilson's formulas for angle bending B-matrix elements
    b_i = (cos_theta * e_ji - e_jk) / (r_ji * sin_theta)
    b_k = (cos_theta * e_jk - e_ji) / (r_jk * sin_theta)
    b_j = -(b_i + b_k)

    b_row[3*i:3*i+3] = b_i
    b_row[3*j:3*j+3] = b_j
    b_row[3*k:3*k+3] = b_k

    return b_row


def build_g_matrix(positions: np.ndarray, atomic_numbers: list,
                   bonds: list, angles: list) -> np.ndarray:
    """
    Build the Wilson G matrix: G = B M⁻¹ Bᵀ

    Args:
        positions: (N, 3) array
        atomic_numbers: list of Z values
        bonds: list of (i, j, dist) tuples
        angles: list of (i, j, k) tuples

    Returns:
        G: (d x d) kinetic energy matrix where d = len(bonds) + len(angles)
    """
    n = len(positions)
    n_bonds = len(bonds)
    n_angles = len(angles)
    d = n_bonds + n_angles

    if d == 0:
        return np.zeros((0, 0))

    # Build B matrix
    B = np.zeros((d, 3 * n))
    for idx, (i, j, _) in enumerate(bonds):
        B[idx] = build_b_matrix_row_bond(positions, i, j)
    for idx, (i, j, k) in enumerate(angles):
        B[n_bonds + idx] = build_b_matrix_row_angle(positions, i, j, k)

    # Mass matrix M⁻¹ (diagonal, repeated 3x per atom)
    M_inv = np.zeros(3 * n)
    for a, z in enumerate(atomic_numbers):
        mass = ATOMIC_MASSES.get(z, 12.0)
        M_inv[3*a:3*a+3] = 1.0 / mass

    # G = B M⁻¹ Bᵀ
    G = B @ np.diag(M_inv) @ B.T

    return G


def compute_observables(G, F, dipole_derivs, polar_derivs):
    """Compute spectroscopic observables from G and F matrices."""
    d = G.shape[0]
    GF = G @ F

    eigenvalues, L = eigh(GF)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    L = L[:, idx]

    freqs = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)

    ir_intensities = np.zeros(d)
    raman_activities = np.zeros(d)
    depol_ratios = np.zeros(d)

    for k in range(d):
        # IR
        dmu_dQ = L[:, k] @ dipole_derivs
        ir_intensities[k] = np.sum(dmu_dQ**2)

        # Raman
        dalpha_dQ = L[:, k] @ polar_derivs
        alpha_bar = (dalpha_dQ[0] + dalpha_dQ[1] + dalpha_dQ[2]) / 3
        gamma_sq = 0.5 * ((dalpha_dQ[0]-dalpha_dQ[1])**2 +
                          (dalpha_dQ[1]-dalpha_dQ[2])**2 +
                          (dalpha_dQ[2]-dalpha_dQ[0])**2 +
                          6*(dalpha_dQ[3]**2 + dalpha_dQ[4]**2 + dalpha_dQ[5]**2))
        raman_activities[k] = 45 * alpha_bar**2 + 7 * gamma_sq

        denom = 45 * alpha_bar**2 + 4 * gamma_sq
        depol_ratios[k] = 3 * gamma_sq / denom if denom > 1e-15 else 0.75

    return np.concatenate([freqs, ir_intensities, raman_activities, depol_ratios])


def compute_jacobian_and_rank(G, F_params, dipole_derivs, polar_derivs, h=1e-6):
    """
    Compute Jacobian and its rank for given G matrix and force constants.

    Returns: dict with rank, d_F, d_S, condition_number, full_rank
    """
    d = G.shape[0]
    d_F = len(F_params)

    F0 = np.diag(F_params)
    phi_0 = compute_observables(G, F0, dipole_derivs, polar_derivs)
    d_S = len(phi_0)

    # Central finite differences
    J = np.zeros((d_S, d_F))
    for j in range(d_F):
        F_plus = F_params.copy()
        F_plus[j] += h
        F_minus = F_params.copy()
        F_minus[j] -= h

        phi_plus = compute_observables(G, np.diag(F_plus), dipole_derivs, polar_derivs)
        phi_minus = compute_observables(G, np.diag(F_minus), dipole_derivs, polar_derivs)
        J[:, j] = (phi_plus - phi_minus) / (2 * h)

    # SVD for rank analysis
    U, sigma, Vt = svd(J, full_matrices=False)
    threshold = sigma[0] * 1e-8
    rank = int(np.sum(sigma > threshold))
    cond = sigma[0] / sigma[-1] if sigma[-1] > 0 else np.inf

    return {
        "rank": rank,
        "d_F": d_F,
        "d_S": d_S,
        "d": d,
        "condition_number": float(cond),
        "full_rank": rank == d_F,
        "overdetermination": d_S / d_F,
        "min_sv": float(sigma[-1]),
        "max_sv": float(sigma[0]),
    }


def analyze_molecule(positions: np.ndarray, atomic_numbers: list,
                     rng: np.random.Generator) -> Optional[dict]:
    """
    Full Jacobian rank analysis for a single molecule.

    Returns None if molecule is too small or has issues.
    """
    n = len(atomic_numbers)
    if n < 3:
        return None

    bonds = find_bonds(positions, atomic_numbers)
    if len(bonds) < 2:
        return None

    angles = find_angles(bonds)
    d = len(bonds) + len(angles)

    if d < 2:
        return None

    # Build G matrix
    G = build_g_matrix(positions, atomic_numbers, bonds, angles)

    # Check G is valid (positive semi-definite, no NaN)
    if np.any(np.isnan(G)) or np.any(np.isinf(G)):
        return None

    eigvals_G = np.linalg.eigvalsh(G)
    if np.any(eigvals_G < -1e-10):
        return None

    # Random force constants and derivatives
    F_params = rng.uniform(1.0, 10.0, size=d)
    dipole_derivs = rng.standard_normal((d, 3))
    polar_derivs = rng.standard_normal((d, 6))

    try:
        result = compute_jacobian_and_rank(G, F_params, dipole_derivs, polar_derivs)
        result["n_atoms"] = n
        result["n_bonds"] = len(bonds)
        result["n_angles"] = len(angles)
        return result
    except Exception as e:
        return None


def run_scaled_analysis(max_molecules: int = 2000, output_dir: str = "experiments/results"):
    """
    Run Jacobian rank analysis on QM9 molecular geometries.
    """
    from torch_geometric.datasets import QM9

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading QM9 dataset...")
    ds = QM9(root='/tmp/qm9_tg')

    # Sample molecules of varying sizes
    n_total = min(max_molecules, len(ds))
    logger.info(f"Analyzing {n_total} molecules...")

    rng = np.random.default_rng(2026)
    indices = rng.choice(len(ds), n_total, replace=False)

    results = []
    errors = 0
    t_start = time.time()

    for count, idx in enumerate(indices):
        if count % 200 == 0 and count > 0:
            elapsed = time.time() - t_start
            rate = count / elapsed
            logger.info(f"  Progress: {count}/{n_total} ({100*count/n_total:.1f}%) "
                       f"[{elapsed:.0f}s, {rate:.1f} mol/s]")

        mol = ds[int(idx)]
        positions = mol.pos.numpy()
        atomic_numbers = mol.z.tolist()

        result = analyze_molecule(positions, atomic_numbers, rng)
        if result is not None:
            result["qm9_idx"] = int(idx)
            result["smiles"] = mol.smiles
            results.append(result)
        else:
            errors += 1

    elapsed = time.time() - t_start
    logger.info(f"Done: {len(results)} successful, {errors} errors in {elapsed:.1f}s")

    # ============================================================
    # Analysis
    # ============================================================

    print("\n" + "=" * 70)
    print(f"SCALED JACOBIAN RANK ANALYSIS ({len(results)} molecules)")
    print("=" * 70)

    n_full = sum(1 for r in results if r["full_rank"])
    n_deficient = sum(1 for r in results if not r["full_rank"])
    print(f"\n  Full rank:      {n_full} ({100*n_full/len(results):.1f}%)")
    print(f"  Rank deficient: {n_deficient} ({100*n_deficient/len(results):.1f}%)")

    # By molecule size (N)
    by_n = defaultdict(list)
    for r in results:
        by_n[r["n_atoms"]].append(r)

    print(f"\n  {'N':>3} {'count':>6} {'full_rank':>10} {'pct':>6} {'avg_d':>6} {'avg_cond':>12} {'avg_ratio':>10}")
    print("  " + "-" * 60)
    for n in sorted(by_n.keys()):
        rs = by_n[n]
        n_fr = sum(1 for r in rs if r["full_rank"])
        avg_d = np.mean([r["d"] for r in rs])
        finite_conds = [r["condition_number"] for r in rs if np.isfinite(r["condition_number"])]
        avg_cond = np.mean(finite_conds) if finite_conds else float('inf')
        avg_ratio = np.mean([r["overdetermination"] for r in rs])
        print(f"  {n:>3} {len(rs):>6} {n_fr:>10} {100*n_fr/len(rs):>5.1f}% {avg_d:>6.1f} {avg_cond:>12.1f} {avg_ratio:>10.1f}x")

    # Overdetermination analysis
    all_ratios = [r["overdetermination"] for r in results]
    print(f"\n  Overdetermination ratio: mean={np.mean(all_ratios):.2f}x, "
          f"min={np.min(all_ratios):.2f}x, max={np.max(all_ratios):.2f}x")

    # Condition number analysis
    finite_conds = [r["condition_number"] for r in results if np.isfinite(r["condition_number"])]
    if finite_conds:
        print(f"  Condition number: median={np.median(finite_conds):.1f}, "
              f"p95={np.percentile(finite_conds, 95):.1f}, "
              f"max={np.max(finite_conds):.1f}")

    # Show rank-deficient molecules
    if n_deficient > 0:
        print(f"\n  RANK-DEFICIENT MOLECULES ({n_deficient}):")
        for r in results:
            if not r["full_rank"]:
                print(f"    idx={r['qm9_idx']} N={r['n_atoms']} d={r['d']} "
                      f"rank={r['rank']}/{r['d_F']} cond={r['condition_number']:.1e} "
                      f"SMILES={r.get('smiles', 'N/A')}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    if n_full == len(results):
        print("  ALL molecules have full-rank Jacobians with real QM9 geometries.")
        print("  This strongly supports Conjecture 3 (generic identifiability).")
        print("  The 4x overdetermination ratio makes rank deficiency a codimension-4 event.")
    else:
        print(f"  {n_deficient} molecules have rank-deficient Jacobians.")
        print("  These may correspond to special symmetry configurations.")

    # Save results
    save_data = {
        "n_molecules": len(results),
        "n_full_rank": n_full,
        "n_rank_deficient": n_deficient,
        "pct_full_rank": 100 * n_full / len(results),
        "overdetermination_stats": {
            "mean": float(np.mean(all_ratios)),
            "min": float(np.min(all_ratios)),
            "max": float(np.max(all_ratios)),
        },
        "condition_stats": {
            "median": float(np.median(finite_conds)) if finite_conds else None,
            "p95": float(np.percentile(finite_conds, 95)) if finite_conds else None,
        },
        "rank_deficient_molecules": [r for r in results if not r["full_rank"]],
    }

    output_file = output_path / "jacobian_scaled_results.json"
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of molecules")
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    run_scaled_analysis(max_molecules=args.n, output_dir=args.output)
