"""
Jacobian rank analysis for the vibrational forward map Φ.

Computes rank(∂Φ/∂F) where:
  - Φ maps force constants F to observables (frequencies, IR intensities, etc.)
  - F is a diagonal valence force field (one FC per internal coordinate)

This provides numerical evidence for Conjecture 3 (Generic Identifiability).

For a C1 molecule with d = 3N-6 vibrational DOF:
  - F has d independent parameters
  - Φ produces 4d observables (frequencies, IR ints, Raman acts, depol ratios)
  - If rank(J) = d at a generic point, the forward map is locally injective

We test this on simple model systems where we can construct the GF matrix
analytically or semi-analytically.
"""

import numpy as np
from numpy.linalg import eigh, svd, norm


def make_triatomic_gf(m1, m2, m3, r12, r23, theta, f_r1, f_r2, f_theta):
    """
    Build the G and F matrices for a bent triatomic molecule (like H2O).

    Internal coordinates: r12 (bond 1), r23 (bond 2), theta (angle)
    Force constants: f_r1, f_r2 (stretches), f_theta (bend)

    Uses Wilson's method for the B matrix.

    Args:
        m1, m2, m3: atomic masses (amu)
        r12, r23: bond lengths (Angstrom)
        theta: bond angle (radians)
        f_r1, f_r2, f_theta: force constants (mdyne/A or mdyne*A/rad²)

    Returns:
        G: 3x3 kinetic energy matrix
        F: 3x3 force constant matrix (diagonal)
        eigenvalues: vibrational frequencies (proxy: eigenvalues of GF)
    """
    # G matrix elements for bent triatomic (Wilson's formulas)
    # G_rr = 1/m_end + 1/m_center
    # G_rr' = cos(theta) / m_center
    # G_thetatheta = 1/r1² (1/m1 + 1/m2) + 1/r2² (1/m2 + 1/m3) - 2cos(theta)/(r1*r2*m2)
    # G_rtheta = sin(theta) / (r * m_center)

    mu1, mu2, mu3 = 1.0/m1, 1.0/m2, 1.0/m3

    G = np.zeros((3, 3))
    G[0, 0] = mu1 + mu2  # G_r1r1
    G[1, 1] = mu2 + mu3  # G_r2r2
    G[0, 1] = np.cos(theta) * mu2  # G_r1r2
    G[1, 0] = G[0, 1]

    G[2, 2] = (mu1 + mu2) / r12**2 + (mu2 + mu3) / r23**2 - 2 * np.cos(theta) * mu2 / (r12 * r23)  # G_thetatheta
    G[0, 2] = -np.sin(theta) * mu2 / r12  # G_r1theta
    G[2, 0] = G[0, 2]
    G[1, 2] = -np.sin(theta) * mu2 / r23  # G_r2theta
    G[2, 1] = G[1, 2]

    # Diagonal force constant matrix
    F = np.diag([f_r1, f_r2, f_theta])

    return G, F


def compute_observables(G, F, dipole_derivs=None, polar_derivs=None):
    """
    Compute the full set of spectroscopic observables from G and F matrices.

    Returns: dict with frequencies, IR intensities, Raman activities, depol ratios
    """
    d = G.shape[0]
    GF = G @ F

    # Eigenvalues and eigenvectors
    eigenvalues, L = eigh(GF)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    L = L[:, idx]

    # Frequencies (proportional to sqrt of eigenvalues)
    # Using proxy: ν_k ∝ sqrt(λ_k)
    freqs = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)

    # For IR and Raman intensities, we need dipole and polarizability derivatives
    # If not provided, use random (generic) projections
    if dipole_derivs is None:
        # Random dipole derivative vectors (3 components per internal coord)
        rng = np.random.default_rng(42)
        dipole_derivs = rng.standard_normal((d, 3))

    if polar_derivs is None:
        # Random polarizability derivative (6 unique components per internal coord)
        rng = np.random.default_rng(123)
        polar_derivs = rng.standard_normal((d, 6))

    # IR intensities: a_k = |∂μ/∂Q_k|² = |L_k^T ∂μ/∂s|²
    ir_intensities = np.zeros(d)
    for k in range(d):
        dmu_dQ = L[:, k] @ dipole_derivs  # 3-vector
        ir_intensities[k] = np.sum(dmu_dQ**2)

    # Raman activities: b_k = 45(α̅')² + 7(γ')²
    # Simplified: use |L_k^T ∂α/∂s|² as proxy
    raman_activities = np.zeros(d)
    for k in range(d):
        dalpha_dQ = L[:, k] @ polar_derivs  # 6-vector
        # Separate isotropic and anisotropic parts
        # α̅' = (α_xx' + α_yy' + α_zz') / 3
        alpha_bar = (dalpha_dQ[0] + dalpha_dQ[1] + dalpha_dQ[2]) / 3
        # γ'² = 0.5 * [(α_xx'-α_yy')² + (α_yy'-α_zz')² + (α_zz'-α_xx')²
        #              + 6(α_xy'² + α_xz'² + α_yz'²)]
        gamma_sq = 0.5 * ((dalpha_dQ[0]-dalpha_dQ[1])**2 +
                          (dalpha_dQ[1]-dalpha_dQ[2])**2 +
                          (dalpha_dQ[2]-dalpha_dQ[0])**2 +
                          6*(dalpha_dQ[3]**2 + dalpha_dQ[4]**2 + dalpha_dQ[5]**2))
        raman_activities[k] = 45 * alpha_bar**2 + 7 * gamma_sq

    # Depolarization ratios: ρ_k = 3γ'² / (45α̅'² + 4γ'²)
    depol_ratios = np.zeros(d)
    for k in range(d):
        dalpha_dQ = L[:, k] @ polar_derivs
        alpha_bar = (dalpha_dQ[0] + dalpha_dQ[1] + dalpha_dQ[2]) / 3
        gamma_sq = 0.5 * ((dalpha_dQ[0]-dalpha_dQ[1])**2 +
                          (dalpha_dQ[1]-dalpha_dQ[2])**2 +
                          (dalpha_dQ[2]-dalpha_dQ[0])**2 +
                          6*(dalpha_dQ[3]**2 + dalpha_dQ[4]**2 + dalpha_dQ[5]**2))
        denom = 45 * alpha_bar**2 + 4 * gamma_sq
        depol_ratios[k] = 3 * gamma_sq / denom if denom > 1e-15 else 0.75

    return {
        "freqs": freqs,
        "ir_intensities": ir_intensities,
        "raman_activities": raman_activities,
        "depol_ratios": depol_ratios,
    }


def compute_jacobian(G_func, F_params, dipole_derivs=None, polar_derivs=None, h=1e-6):
    """
    Compute the Jacobian ∂Φ/∂F by central finite differences.

    Args:
        G_func: function that returns G matrix (constant w.r.t. F for diagonal VFF)
        F_params: array of force constant parameters
        h: finite difference step size

    Returns:
        J: (4d × d_F) Jacobian matrix
        observables_0: observables at the nominal point
    """
    d_F = len(F_params)
    G = G_func()
    d = G.shape[0]

    # Compute observables at nominal point
    F0 = np.diag(F_params)
    obs_0 = compute_observables(G, F0, dipole_derivs, polar_derivs)

    # Stack all observables into a single vector
    phi_0 = np.concatenate([obs_0["freqs"], obs_0["ir_intensities"],
                            obs_0["raman_activities"], obs_0["depol_ratios"]])
    d_S = len(phi_0)

    # Compute Jacobian by central differences
    J = np.zeros((d_S, d_F))
    for j in range(d_F):
        F_plus = F_params.copy()
        F_plus[j] += h
        F_minus = F_params.copy()
        F_minus[j] -= h

        obs_plus = compute_observables(G, np.diag(F_plus), dipole_derivs, polar_derivs)
        obs_minus = compute_observables(G, np.diag(F_minus), dipole_derivs, polar_derivs)

        phi_plus = np.concatenate([obs_plus["freqs"], obs_plus["ir_intensities"],
                                   obs_plus["raman_activities"], obs_plus["depol_ratios"]])
        phi_minus = np.concatenate([obs_minus["freqs"], obs_minus["ir_intensities"],
                                    obs_minus["raman_activities"], obs_minus["depol_ratios"]])

        J[:, j] = (phi_plus - phi_minus) / (2 * h)

    return J, phi_0


def analyze_jacobian(J):
    """Analyze the rank, condition number, and singular values of the Jacobian."""
    U, sigma, Vt = svd(J, full_matrices=False)

    # Numerical rank (threshold = max(σ) × 1e-8)
    threshold = sigma[0] * 1e-8
    rank = np.sum(sigma > threshold)

    # Condition number
    if sigma[-1] > 0:
        cond = sigma[0] / sigma[-1]
    else:
        cond = np.inf

    return {
        "rank": rank,
        "d_F": J.shape[1],
        "d_S": J.shape[0],
        "condition_number": cond,
        "singular_values": sigma,
        "normalized_rank": rank / J.shape[1],
    }


def test_h2o():
    """Test Jacobian analysis for H2O (C2v, d=3)."""
    print("=" * 60)
    print("TEST: H2O (C2v, N=3, d=3)")
    print("=" * 60)

    # H2O parameters
    m_H, m_O = 1.008, 15.999
    r_OH = 0.9572  # Angstrom
    theta = np.radians(104.52)  # bond angle

    # Force constants (typical values in mdyne/A)
    f_r1 = 8.45  # O-H stretch 1
    f_r2 = 8.45  # O-H stretch 2
    f_theta = 0.76  # H-O-H bend

    F_params = np.array([f_r1, f_r2, f_theta])

    def G_func():
        G, _ = make_triatomic_gf(m_H, m_O, m_H, r_OH, r_OH, theta, f_r1, f_r2, f_theta)
        return G

    J, phi_0 = compute_jacobian(G_func, F_params)
    result = analyze_jacobian(J)

    print(f"  d_F = {result['d_F']} (force constant parameters)")
    print(f"  d_S = {result['d_S']} (observables = 4d = {4*3})")
    print(f"  rank(J) = {result['rank']}")
    print(f"  normalized rank = {result['normalized_rank']:.4f}")
    print(f"  condition number = {result['condition_number']:.2f}")
    print(f"  singular values = {result['singular_values']}")
    print(f"  overdetermination ratio = {result['d_S'] / result['d_F']:.1f}×")
    print(f"  STATUS: {'FULL RANK ✓' if result['rank'] == result['d_F'] else 'RANK DEFICIENT ✗'}")
    print()
    return result


def test_random_molecules(n_molecules=100, d_range=(3, 15)):
    """
    Test Jacobian analysis on random "molecules" (random G and F matrices).

    This tests whether generic force constant configurations yield full-rank Jacobians.
    """
    print("=" * 60)
    print(f"TEST: {n_molecules} random molecules (d ∈ [{d_range[0]}, {d_range[1]}])")
    print("=" * 60)

    rng = np.random.default_rng(2026)
    results = []

    for i in range(n_molecules):
        d = rng.integers(d_range[0], d_range[1] + 1)

        # Random symmetric positive definite G matrix
        A = rng.standard_normal((d, d))
        G = A @ A.T + np.eye(d) * 0.1  # ensure PD

        # Random positive force constants (diagonal VFF)
        F_params = rng.uniform(0.5, 10.0, size=d)

        # Random dipole and polarizability derivatives
        dipole_derivs = rng.standard_normal((d, 3))
        polar_derivs = rng.standard_normal((d, 6))

        def G_func(G=G):
            return G

        try:
            J, phi_0 = compute_jacobian(G_func, F_params, dipole_derivs, polar_derivs)
            result = analyze_jacobian(J)
            result["d"] = d
            result["full_rank"] = result["rank"] == result["d_F"]
            results.append(result)
        except Exception as e:
            print(f"  Molecule {i}: ERROR - {e}")

    # Summarize
    n_full_rank = sum(1 for r in results if r["full_rank"])
    n_total = len(results)

    print(f"\n  Total tested: {n_total}")
    print(f"  Full rank: {n_full_rank} ({100*n_full_rank/n_total:.1f}%)")
    print(f"  Rank deficient: {n_total - n_full_rank} ({100*(n_total-n_full_rank)/n_total:.1f}%)")

    # Stats by d
    from collections import defaultdict
    by_d = defaultdict(list)
    for r in results:
        by_d[r["d"]].append(r)

    print(f"\n  {'d':>3} {'n':>5} {'full_rank':>10} {'avg_cond':>12} {'avg_ratio':>10}")
    print("  " + "-" * 45)
    for d in sorted(by_d.keys()):
        rs = by_d[d]
        n_fr = sum(1 for r in rs if r["full_rank"])
        avg_cond = np.mean([r["condition_number"] for r in rs if np.isfinite(r["condition_number"])])
        avg_ratio = np.mean([r["d_S"] / r["d_F"] for r in rs])
        print(f"  {d:>3} {len(rs):>5} {n_fr:>10} {avg_cond:>12.1f} {avg_ratio:>10.1f}×")

    print()
    if n_full_rank == n_total:
        print("  RESULT: ALL generic configurations have full-rank Jacobian.")
        print("  This is consistent with Conjecture 3 (generic identifiability).")
    else:
        print(f"  RESULT: {n_total - n_full_rank} configurations have rank-deficient Jacobian.")
        print("  Investigate whether these correspond to eigenvalue degeneracies.")

    return results


def test_near_degenerate():
    """
    Test Jacobian near eigenvalue degeneracies.
    These are the configurations where Conjecture 3 might fail.
    """
    print("=" * 60)
    print("TEST: Near-degenerate configurations (stress test)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    d = 5
    n_tests = 50
    n_rank_deficient = 0

    for i in range(n_tests):
        # Create G matrix
        A = rng.standard_normal((d, d))
        G = A @ A.T + np.eye(d) * 0.1

        # Create force constants that produce near-degenerate eigenvalues
        # Make two FCs very similar
        base = rng.uniform(2.0, 8.0, size=d)
        # Make FC[0] ≈ FC[1] (near-degenerate)
        base[1] = base[0] + rng.uniform(-0.001, 0.001)

        F_params = base
        dipole_derivs = rng.standard_normal((d, 3))
        polar_derivs = rng.standard_normal((d, 6))

        def G_func(G=G):
            return G

        J, _ = compute_jacobian(G_func, F_params, dipole_derivs, polar_derivs, h=1e-7)
        result = analyze_jacobian(J)

        if not (result["rank"] == result["d_F"]):
            n_rank_deficient += 1
            print(f"  Config {i}: RANK DEFICIENT (rank={result['rank']}/{result['d_F']}, "
                  f"κ={result['condition_number']:.1e})")

    print(f"\n  Near-degenerate configs tested: {n_tests}")
    print(f"  Rank deficient: {n_rank_deficient}")
    print(f"  Full rank: {n_tests - n_rank_deficient}")
    if n_rank_deficient == 0:
        print("  Even near-degenerate configs maintain full rank.")
        print("  von Neumann-Wigner codim-2 prediction: degeneracies are isolated.")
    print()


if __name__ == "__main__":
    # Test 1: H2O (known molecule)
    test_h2o()

    # Test 2: Random generic molecules
    test_random_molecules(n_molecules=200, d_range=(3, 20))

    # Test 3: Near-degenerate stress test
    test_near_degenerate()
