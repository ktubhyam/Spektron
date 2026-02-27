"""
Compute the Information Completeness Ratio R(G,N) from character tables.

R(G,N) = |M_IR ∪ M_Raman| / (3N - 6)

This is the fraction of vibrational degrees of freedom observable by
combined IR and Raman spectroscopy, as defined in Theorem 1.

Usage:
    python experiments/compute_rgn.py

Tests against known values from spectroscopy textbooks.
"""

# Character table data for common point groups.
# For each irrep: (name, dimension, is_IR_active, is_Raman_active)
# IR active = transforms as x, y, or z (translation)
# Raman active = transforms as x², y², z², xy, xz, yz (quadratic)

CHARACTER_TABLES = {
    "C1": {
        "irreps": [
            ("A", 1, True, True),
        ],
    },
    "Cs": {
        "irreps": [
            ("A'", 1, True, True),   # x, y; x², y², z², xy
            ("A''", 1, True, True),  # z; xz, yz
        ],
    },
    "Ci": {
        "irreps": [
            ("Ag", 1, False, True),  # x², y², z², xy, xz, yz
            ("Au", 1, True, False),  # x, y, z
        ],
    },
    "C2": {
        "irreps": [
            ("A", 1, True, True),    # z; x², y², z², xy
            ("B", 1, True, True),    # x, y; xz, yz
        ],
    },
    "C2v": {
        "irreps": [
            ("A1", 1, True, True),   # z; x², y², z²
            ("A2", 1, False, True),  # Rz; xy
            ("B1", 1, True, True),   # x; xz
            ("B2", 1, True, True),   # y; yz
        ],
    },
    "C3v": {
        "irreps": [
            ("A1", 1, True, True),   # z; x², y², z²
            ("A2", 1, False, False), # Rz — SILENT
            ("E", 2, True, True),    # (x,y); (x²-y², xy), (xz, yz)
        ],
    },
    "C2h": {
        "irreps": [
            ("Ag", 1, False, True),  # x², y², z², xy
            ("Bg", 1, False, True),  # xz, yz
            ("Au", 1, True, False),  # z
            ("Bu", 1, True, False),  # x, y
        ],
    },
    "D2h": {
        "irreps": [
            ("Ag", 1, False, True),   # x², y², z²
            ("B1g", 1, False, True),  # xy
            ("B2g", 1, False, True),  # xz
            ("B3g", 1, False, True),  # yz
            ("Au", 1, False, False),  # SILENT
            ("B1u", 1, True, False),  # z
            ("B2u", 1, True, False),  # y
            ("B3u", 1, True, False),  # x
        ],
    },
    "D3h": {
        "irreps": [
            ("A1'", 1, False, True),  # x²+y², z²
            ("A2'", 1, True, False),  # z  — wait, need to check
            ("E'", 2, True, True),    # (x,y); (x²-y², xy)
            ("A1''", 1, False, False),# SILENT
            ("A2''", 1, True, False), # z
            ("E''", 2, False, True),  # (xz, yz)
        ],
    },
    "D6h": {
        "irreps": [
            ("A1g", 1, False, True),  # x²+y², z²
            ("A2g", 1, False, False), # SILENT (Rz)
            ("B1g", 1, False, False), # SILENT
            ("B2g", 1, False, False), # SILENT
            ("E1g", 2, False, True),  # (xz, yz)
            ("E2g", 2, False, True),  # (x²-y², xy)
            ("A1u", 1, False, False), # SILENT
            ("A2u", 1, True, False),  # z
            ("B1u", 1, False, False), # SILENT
            ("B2u", 1, False, False), # SILENT
            ("E1u", 2, True, False),  # (x, y)
            ("E2u", 2, False, False), # SILENT
        ],
    },
    "Td": {
        "irreps": [
            ("A1", 1, False, True),  # x²+y²+z²
            ("A2", 1, False, False), # SILENT
            ("E", 2, False, True),   # (2z²-x²-y², x²-y²)
            ("T1", 3, False, False), # (Rx, Ry, Rz) — SILENT for vib
            ("T2", 3, True, True),   # (x,y,z); (xy, xz, yz)
        ],
    },
    "Oh": {
        "irreps": [
            ("A1g", 1, False, True),  # x²+y²+z²
            ("A2g", 1, False, False), # SILENT
            ("Eg", 2, False, True),   # (2z²-x²-y², x²-y²)
            ("T1g", 3, False, False), # (Rx,Ry,Rz) — SILENT for vib
            ("T2g", 3, False, True),  # (xy, xz, yz)
            ("A1u", 1, False, False), # SILENT
            ("A2u", 1, False, False), # SILENT
            ("Eu", 2, False, False),  # SILENT
            ("T1u", 3, True, False),  # (x, y, z)
            ("T2u", 3, False, False), # SILENT
        ],
    },
    "Dinfh": {
        # D∞h for linear molecules (3N-5 modes)
        "irreps": [
            ("Sigma_g+", 1, False, True),  # z²; x²+y²
            ("Sigma_g-", 1, False, False),  # SILENT
            ("Pi_g", 2, False, True),       # (xz, yz)
            ("Sigma_u+", 1, True, False),   # z
            ("Sigma_u-", 1, False, False),  # SILENT
            ("Pi_u", 2, True, False),       # (x, y)
        ],
        "linear": True,
    },
}


def compute_rgn_from_decomposition(point_group: str, vib_decomposition: dict) -> dict:
    """
    Compute R(G,N) given a vibrational representation decomposition.

    Args:
        point_group: Name of the point group (key in CHARACTER_TABLES)
        vib_decomposition: dict mapping irrep name -> multiplicity
            e.g., {"A1": 2, "B2": 1} for H2O

    Returns:
        dict with N_IR, N_Raman, N_silent, N_both, d, R
    """
    ct = CHARACTER_TABLES[point_group]
    irrep_info = {name: (dim, ir, raman) for name, dim, ir, raman in ct["irreps"]}

    n_ir = 0
    n_raman = 0
    n_both = 0
    n_silent = 0
    d_total = 0

    for irrep_name, multiplicity in vib_decomposition.items():
        if irrep_name not in irrep_info:
            raise ValueError(f"Unknown irrep {irrep_name} for {point_group}")
        dim, is_ir, is_raman = irrep_info[irrep_name]
        n_modes = multiplicity * dim

        d_total += n_modes
        if is_ir and is_raman:
            n_both += n_modes
            n_ir += n_modes
            n_raman += n_modes
        elif is_ir:
            n_ir += n_modes
        elif is_raman:
            n_raman += n_modes
        else:
            n_silent += n_modes

    n_observable = n_ir + n_raman - n_both  # = |M_IR ∪ M_Raman|
    R = n_observable / d_total if d_total > 0 else 0.0

    return {
        "point_group": point_group,
        "d": d_total,
        "N_IR": n_ir,
        "N_Raman": n_raman,
        "N_both": n_both,
        "N_silent": n_silent,
        "N_observable": n_observable,
        "R": R,
    }


# Known vibrational decompositions for test molecules
TEST_MOLECULES = {
    "H2O": {
        "point_group": "C2v",
        "N": 3,
        "vib_decomp": {"A1": 2, "B2": 1},
        "expected_R": 1.0,
    },
    "CO2": {
        "point_group": "Dinfh",
        "N": 3,
        "vib_decomp": {"Sigma_g+": 1, "Sigma_u+": 1, "Pi_u": 1},
        "expected_R": 1.0,
        "linear": True,
    },
    "NH3": {
        "point_group": "C3v",
        "N": 4,
        "vib_decomp": {"A1": 2, "E": 2},
        "expected_R": 1.0,
    },
    "CH4": {
        "point_group": "Td",
        "N": 5,
        "vib_decomp": {"A1": 1, "E": 1, "T2": 2},
        "expected_R": 1.0,
    },
    "C2H4_ethylene": {
        "point_group": "D2h",
        "N": 6,
        # Γ_vib = 3Ag + B1g + 2B2g + B3g + Au + 2B1u + B2u + 2B3u
        # Wait, let me be more careful. Ethylene D2h:
        # Γ_vib = 3Ag + B1g + 2B2g + B3g + Au + 2B1u + B2u + B3u
        # Actually standard decomposition:
        # Ag: 3, B1g: 1, B2g: 1, B3g: 1, Au: 1, B1u: 2, B2u: 1, B3u: 2
        "vib_decomp": {
            "Ag": 3, "B1g": 1, "B2g": 1, "B3g": 1,
            "Au": 1, "B1u": 2, "B2u": 1, "B3u": 2,
        },
        "expected_R": 11/12,  # 1 Au silent mode out of 12
    },
    "C6H6_benzene": {
        "point_group": "D6h",
        "N": 12,
        # Γ_vib for benzene (D6h, 30 modes):
        # 2A1g + A2g + 4E2g + E1g + A2u + 2B1u + 2B2u + 3E1u + 2E2u + B2g
        # Let me use the standard decomposition:
        # Gerade: 2A1g + A2g + B2g + E1g + 4E2g
        # Ungerade: A2u + 2B1u + 2B2u + 3E1u + 2E2u
        # Γ_vib(benzene) = 2A1g + A2g + 2B2g + E1g + 4E2g + A2u + 2B1u + 2B2u + 3E1u + 2E2u
        # Total: 2+1+2+2+8+1+2+2+6+4 = 30 ✓
        "vib_decomp": {
            "A1g": 2, "A2g": 1, "B2g": 2, "E1g": 1, "E2g": 4,
            "A2u": 1, "B1u": 2, "B2u": 2, "E1u": 3, "E2u": 2,
        },
        # IR active: A2u (1) + 3×E1u (6) = 7
        # Raman active: 2×A1g (2) + E1g (2) + 4×E2g (8) = 12
        # Silent: A2g (1) + 2×B2g (2) + 2×B1u (2) + 2×B2u (2) + 2×E2u (4) = 11
        # Wait — that's 7+12+11 = 30, but observable = 7+12 = 19, silent = 11
        # Hmm, let me recheck: some sources say 10 silent modes for benzene
        # The discrepancy is B2g count. Let me use the standard:
        # Actually B1g doesn't appear in Γ_vib for benzene.
        "expected_R": 19/30,
    },
    "SF6": {
        "point_group": "Oh",
        "N": 7,
        # Γ_vib = A1g + Eg + 2T1u + T2g + T2u
        "vib_decomp": {
            "A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1,
        },
        "expected_R": 12/15,  # T2u (3 modes) silent → 12/15 = 0.80
    },
}


def run_tests():
    """Test R(G,N) computation against known values."""
    print("=" * 70)
    print("INFORMATION COMPLETENESS RATIO R(G,N) — Theorem 1 Validation")
    print("=" * 70)
    print()

    all_pass = True
    for mol_name, mol_data in TEST_MOLECULES.items():
        pg = mol_data["point_group"]
        vib = mol_data["vib_decomp"]
        expected = mol_data["expected_R"]

        result = compute_rgn_from_decomposition(pg, vib)

        # Check d = 3N-6 (or 3N-5 for linear)
        N = mol_data["N"]
        is_linear = mol_data.get("linear", False)
        expected_d = 3 * N - 5 if is_linear else 3 * N - 6

        d_ok = result["d"] == expected_d
        r_ok = abs(result["R"] - expected) < 0.02  # Allow small rounding

        status = "PASS" if (d_ok and r_ok) else "FAIL"
        if not (d_ok and r_ok):
            all_pass = False

        print(f"[{status}] {mol_name} ({pg}, N={N})")
        print(f"  d = {result['d']} (expected {expected_d}) {'✓' if d_ok else '✗ MISMATCH'}")
        print(f"  N_IR = {result['N_IR']}, N_Raman = {result['N_Raman']}, "
              f"N_both = {result['N_both']}, N_silent = {result['N_silent']}")
        print(f"  N_observable = {result['N_observable']}")
        print(f"  R(G,N) = {result['R']:.4f} (expected {expected:.4f}) "
              f"{'✓' if r_ok else '✗ MISMATCH'}")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"{'Molecule':<20} {'G':<10} {'N':>3} {'d':>4} {'N_IR':>5} {'N_Ram':>5} "
          f"{'N_sil':>5} {'R(G,N)':>8}")
    print("-" * 70)
    for mol_name, mol_data in TEST_MOLECULES.items():
        pg = mol_data["point_group"]
        result = compute_rgn_from_decomposition(pg, mol_data["vib_decomp"])
        print(f"{mol_name:<20} {pg:<10} {mol_data['N']:>3} {result['d']:>4} "
              f"{result['N_IR']:>5} {result['N_Raman']:>5} "
              f"{result['N_silent']:>5} {result['R']:>8.4f}")

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED — check decompositions.")

    return all_pass


if __name__ == "__main__":
    run_tests()
