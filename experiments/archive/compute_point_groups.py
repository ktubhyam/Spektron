"""
Compute point groups and R(G,N) for all QM9 molecules.

Uses pymatgen's PointGroupAnalyzer on 3D coordinates from torch_geometric QM9 dataset.
Outputs:
  - Point group distribution histogram
  - R(G,N) distribution
  - Symmetry-stratified statistics
  - Identifies molecules with lowest R(G,N) (hardest for inverse problem)
"""

import numpy as np
import json
import time
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

# Pymatgen for point group analysis
from pymatgen.core.structure import Molecule as PmgMolecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# Torch geometric for QM9 dataset
from torch_geometric.datasets import QM9

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Atomic number to element symbol mapping
ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

# ============================================================
# Character table data for R(G,N) computation
# For each point group, define which irreducible representations
# are IR-active, Raman-active, or silent.
# ============================================================

# Map from point group string to classification function
# Returns: (is_IR_active, is_Raman_active) for each irrep

POINT_GROUP_DATA = {
    # --- Trivial and low symmetry ---
    "C1": {
        "irreps": ["A"],
        "IR_active": {"A"},
        "Raman_active": {"A"},
    },
    "Cs": {
        "irreps": ["A'", "A''"],
        "IR_active": {"A'", "A''"},
        "Raman_active": {"A'", "A''"},
    },
    "Ci": {
        "irreps": ["Ag", "Au"],
        "IR_active": {"Au"},
        "Raman_active": {"Ag"},
    },
    # --- Cn groups ---
    "C2": {
        "irreps": ["A", "B"],
        "IR_active": {"A", "B"},
        "Raman_active": {"A", "B"},
    },
    "C3": {
        "irreps": ["A", "E"],
        "IR_active": {"A", "E"},
        "Raman_active": {"A", "E"},
        "degeneracies": {"E": 2},
    },
    # --- Cnv groups ---
    "C2v": {
        "irreps": ["A1", "A2", "B1", "B2"],
        "IR_active": {"A1", "B1", "B2"},
        "Raman_active": {"A1", "A2", "B1", "B2"},
    },
    "C3v": {
        "irreps": ["A1", "A2", "E"],
        "IR_active": {"A1", "E"},
        "Raman_active": {"A1", "E"},
        "degeneracies": {"E": 2},
    },
    "C4v": {
        "irreps": ["A1", "A2", "B1", "B2", "E"],
        "IR_active": {"A1", "E"},
        "Raman_active": {"A1", "B1", "B2", "E"},
        "degeneracies": {"E": 2},
    },
    # --- Cnh groups ---
    "C2h": {
        "irreps": ["Ag", "Bg", "Au", "Bu"],
        "IR_active": {"Au", "Bu"},
        "Raman_active": {"Ag", "Bg"},
    },
    "C3h": {
        "irreps": ["A'", "A''", "E'", "E''"],
        "IR_active": {"A'", "E'"},  # z-component in A'', but A'' is IR active too in some conventions
        "Raman_active": {"A'", "E'", "E''"},
        "degeneracies": {"E'": 2, "E''": 2},
    },
    # --- Dn groups ---
    "D2": {
        "irreps": ["A", "B1", "B2", "B3"],
        "IR_active": {"B1", "B2", "B3"},
        "Raman_active": {"A", "B1", "B2", "B3"},
    },
    "D3": {
        "irreps": ["A1", "A2", "E"],
        "IR_active": {"A2", "E"},
        "Raman_active": {"A1", "E"},
        "degeneracies": {"E": 2},
    },
    # --- Dnh groups ---
    "D2h": {
        "irreps": ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
        "IR_active": {"B1u", "B2u", "B3u"},
        "Raman_active": {"Ag", "B1g", "B2g", "B3g"},
    },
    "D3h": {
        "irreps": ["A1'", "A2'", "E'", "A1''", "A2''", "E''"],
        "IR_active": {"A2''", "E'"},
        "Raman_active": {"A1'", "E'", "E''"},
        "degeneracies": {"E'": 2, "E''": 2},
    },
    "D4h": {
        "irreps": ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"],
        "IR_active": {"A2u", "Eu"},
        "Raman_active": {"A1g", "B1g", "B2g", "Eg"},
        "degeneracies": {"Eg": 2, "Eu": 2},
    },
    "D5h": {
        "irreps": ["A1'", "A2'", "E1'", "E2'", "A1''", "A2''", "E1''", "E2''"],
        "IR_active": {"A2''", "E1'"},
        "Raman_active": {"A1'", "E1'", "E2'", "E1''"},  # E2' Raman active
        "degeneracies": {"E1'": 2, "E2'": 2, "E1''": 2, "E2''": 2},
    },
    "D6h": {
        "irreps": ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g",
                    "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"],
        "IR_active": {"A2u", "E1u"},
        "Raman_active": {"A1g", "E1g", "E2g"},
        "degeneracies": {"E1g": 2, "E2g": 2, "E1u": 2, "E2u": 2},
    },
    # --- Dnd groups ---
    "D2d": {
        "irreps": ["A1", "A2", "B1", "B2", "E"],
        "IR_active": {"B2", "E"},
        "Raman_active": {"A1", "B1", "B2", "E"},
        "degeneracies": {"E": 2},
    },
    "D3d": {
        "irreps": ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
        "IR_active": {"A2u", "Eu"},
        "Raman_active": {"A1g", "Eg"},
        "degeneracies": {"Eg": 2, "Eu": 2},
    },
    # --- Tetrahedral ---
    "Td": {
        "irreps": ["A1", "A2", "E", "T1", "T2"],
        "IR_active": {"T2"},
        "Raman_active": {"A1", "E", "T2"},
        "degeneracies": {"E": 2, "T1": 3, "T2": 3},
    },
    # --- Octahedral ---
    "Oh": {
        "irreps": ["A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"],
        "IR_active": {"T1u"},
        "Raman_active": {"A1g", "Eg", "T2g"},
        "degeneracies": {"Eg": 2, "T1g": 3, "T2g": 3, "Eu": 2, "T1u": 3, "T2u": 3},
    },
    # --- Linear ---
    "C*v": {
        # Sigma+, Sigma-, Pi, Delta, ...
        # All modes are both IR and Raman active for C∞v
        # This is an approximation — need to handle case-by-case for linear molecules
        "all_active": True,  # Special handling
    },
    "D*h": {
        # Sigma_g+, Sigma_g-, Pi_g, ..., Sigma_u+, Sigma_u-, Pi_u, ...
        # Mutual exclusion applies (centrosymmetric)
        "centrosymmetric_linear": True,  # Special handling
    },
}

# Aliases for pymatgen point group names
PG_ALIASES = {
    "C∞v": "C*v",
    "Cinf_v": "C*v",
    "C*v": "C*v",
    "D∞h": "D*h",
    "Dinf_h": "D*h",
    "D*h": "D*h",
    "S2": "Ci",  # S2 = Ci (inversion symmetry)
    "S4": "S4",
    "S6": "S6",
    "D1d": "C2v",  # D1d doesn't exist; pymatgen sometimes returns this — approximate as C2v
}

# Additional point groups that pymatgen may return
POINT_GROUP_DATA["C4"] = {
    "irreps": ["A", "B", "E"],
    "IR_active": {"A", "E"},
    "Raman_active": {"A", "B", "E"},
    "degeneracies": {"E": 2},
}
POINT_GROUP_DATA["S4"] = {
    "irreps": ["A", "B", "E"],
    "IR_active": {"B", "E"},
    "Raman_active": {"A", "B", "E"},
    "degeneracies": {"E": 2},
}
POINT_GROUP_DATA["S6"] = {
    "irreps": ["Ag", "Eg", "Au", "Eu"],
    "IR_active": {"Au", "Eu"},
    "Raman_active": {"Ag", "Eg"},
    "degeneracies": {"Eg": 2, "Eu": 2},
}


def compute_rgn_for_point_group(pg_name: str, n_atoms: int) -> dict:
    """
    Compute R(G,N) for a given point group and number of atoms.

    For non-linear molecules: d = 3N - 6 vibrational DOF
    For linear molecules: d = 3N - 5 vibrational DOF

    Returns dict with:
        d: total vibrational modes
        R: information completeness ratio
        is_centrosymmetric: bool
        fraction_IR: fraction IR-active
        fraction_Raman: fraction Raman-active
    """
    # Normalize point group name
    pg = PG_ALIASES.get(pg_name, pg_name)

    is_linear = pg in ("C*v", "D*h")
    d = 3 * n_atoms - 5 if is_linear else 3 * n_atoms - 6

    if d <= 0:
        return {"d": d, "R": 1.0, "is_centrosymmetric": False,
                "n_IR": 0, "n_Raman": 0, "n_silent": 0}

    # Handle linear molecules
    if pg == "C*v":
        # All modes IR and Raman active for C∞v
        return {
            "d": d, "R": 1.0, "is_centrosymmetric": False,
            "n_IR": d, "n_Raman": d, "n_silent": 0,
            "fraction_IR": 1.0, "fraction_Raman": 1.0,
        }
    elif pg == "D*h":
        # For D∞h linear molecules:
        # Σ_g+ modes: Raman only (N_center - 1 of them for symmetric stretch)
        # Σ_u+ modes: IR only
        # Π_g modes: Raman only (doubly degenerate)
        # Π_u modes: IR only (doubly degenerate)
        # No silent modes in practice for small linear molecules
        # Approximate: all modes are either IR or Raman active (mutual exclusion)
        n_IR = d // 2 + (d % 2)  # Rough approximation
        n_Raman = d - n_IR + (d % 2)  # Some overlap for small molecules
        # For CO2 (D∞h, N=3): d=4, modes = Σ_g+(1, Raman) + Σ_u+(1, IR) + Π_u(2, IR)
        # So n_IR=3, n_Raman=1, n_observable=4, R=1.0
        # Actually for D∞h, mutual exclusion means R=1.0 typically
        return {
            "d": d, "R": 1.0, "is_centrosymmetric": True,
            "n_IR": n_IR, "n_Raman": n_Raman, "n_silent": 0,
            "fraction_IR": n_IR / d, "fraction_Raman": n_Raman / d,
        }

    # Handle non-linear molecules
    if pg not in POINT_GROUP_DATA:
        # Unknown point group — assume C1 (all active)
        logger.warning(f"Unknown point group '{pg}' — treating as C1 (all active)")
        return {
            "d": d, "R": 1.0, "is_centrosymmetric": False,
            "n_IR": d, "n_Raman": d, "n_silent": 0,
            "fraction_IR": 1.0, "fraction_Raman": 1.0,
            "unknown_pg": True,
        }

    pg_data = POINT_GROUP_DATA[pg]
    ir_active = pg_data.get("IR_active", set())
    raman_active = pg_data.get("Raman_active", set())

    # Check centrosymmetry (has inversion center)
    centrosymmetric_groups = {"Ci", "C2h", "C3h", "D2h", "D3d", "D4h", "D5h", "D6h", "Oh", "D*h"}
    is_centro = pg in centrosymmetric_groups

    # For the purpose of R(G,N), we need to know what fraction of modes
    # of each irrep type are observable (IR or Raman active).
    # Since we don't know the exact vibrational decomposition without
    # the actual Hessian, we compute the FRACTION of irrep space that
    # is observable.
    #
    # Key insight: For a "typical" molecule of this point group,
    # the vibrational modes will be distributed across irreps roughly
    # proportional to the dimension of each irrep. So R(G,N) can be
    # estimated as the weighted fraction of observable irreps.

    irreps = pg_data.get("irreps", [])
    degeneracies = pg_data.get("degeneracies", {})

    total_dim = sum(degeneracies.get(irrep, 1) for irrep in irreps)
    observable_dim = 0
    ir_dim = 0
    raman_dim = 0

    for irrep in irreps:
        dim = degeneracies.get(irrep, 1)
        is_ir = irrep in ir_active
        is_raman = irrep in raman_active
        if is_ir or is_raman:
            observable_dim += dim
        if is_ir:
            ir_dim += dim
        if is_raman:
            raman_dim += dim

    # R(G,N) ≈ observable_dim / total_dim (fraction of representation space that is observable)
    R = observable_dim / total_dim if total_dim > 0 else 1.0

    # Estimate actual mode counts
    n_IR = int(round(d * ir_dim / total_dim))
    n_Raman = int(round(d * raman_dim / total_dim))
    n_observable = int(round(d * observable_dim / total_dim))
    n_silent = d - n_observable

    return {
        "d": d,
        "R": R,
        "is_centrosymmetric": is_centro,
        "n_IR": n_IR,
        "n_Raman": n_Raman,
        "n_silent": max(0, n_silent),
        "fraction_IR": ir_dim / total_dim,
        "fraction_Raman": raman_dim / total_dim,
    }


def get_point_group(positions: np.ndarray, atomic_numbers: list,
                    tolerance: float = 0.3) -> str:
    """
    Compute point group of a molecule using pymatgen.

    Args:
        positions: (N, 3) array of atomic positions in Angstrom
        atomic_numbers: list of atomic numbers
        tolerance: symmetry tolerance in Angstrom

    Returns:
        Point group string (e.g., "C2v", "D3h", "Td")
    """
    species = [ATOMIC_NUM_TO_SYMBOL.get(z, 'X') for z in atomic_numbers]
    mol = PmgMolecule(species, positions)

    analyzer = PointGroupAnalyzer(mol, tolerance=tolerance)
    pg = analyzer.sch_symbol

    return pg


def process_qm9_dataset(max_molecules: Optional[int] = None,
                        tolerance: float = 0.3,
                        output_dir: str = "results") -> dict:
    """
    Process all QM9 molecules: compute point groups and R(G,N).

    Args:
        max_molecules: if set, only process this many molecules
        tolerance: symmetry tolerance for point group detection
        output_dir: directory for output files

    Returns:
        Dictionary with full results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading QM9 dataset...")
    ds = QM9(root='/tmp/qm9_tg')
    n_total = min(len(ds), max_molecules) if max_molecules else len(ds)
    logger.info(f"Processing {n_total} molecules (tolerance={tolerance})")

    # Results storage
    results = []
    pg_counter = Counter()
    rgn_by_pg = defaultdict(list)
    errors = []

    t_start = time.time()

    for i in range(n_total):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (n_total - i) / rate
            logger.info(f"  Progress: {i}/{n_total} ({100*i/n_total:.1f}%) "
                       f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        mol = ds[i]
        positions = mol.pos.numpy()
        atomic_numbers = mol.z.tolist()
        smiles = mol.smiles
        n_atoms = len(atomic_numbers)

        try:
            pg = get_point_group(positions, atomic_numbers, tolerance=tolerance)
            rgn_data = compute_rgn_for_point_group(pg, n_atoms)

            result = {
                "idx": i,
                "smiles": smiles,
                "N": n_atoms,
                "point_group": pg,
                "d": rgn_data["d"],
                "R": rgn_data["R"],
                "is_centrosymmetric": rgn_data["is_centrosymmetric"],
                "n_IR": rgn_data.get("n_IR", 0),
                "n_Raman": rgn_data.get("n_Raman", 0),
                "n_silent": rgn_data.get("n_silent", 0),
            }
            results.append(result)
            pg_counter[pg] += 1
            rgn_by_pg[pg].append(rgn_data["R"])

        except Exception as e:
            errors.append({"idx": i, "smiles": smiles, "error": str(e)})
            if len(errors) <= 10:
                logger.warning(f"  Error on mol {i} ({smiles}): {e}")

    elapsed = time.time() - t_start
    logger.info(f"Done processing {n_total} molecules in {elapsed:.1f}s "
               f"({n_total/elapsed:.0f} mol/s)")
    logger.info(f"Errors: {len(errors)}")

    # ============================================================
    # Analysis
    # ============================================================

    print("\n" + "=" * 70)
    print("POINT GROUP DISTRIBUTION")
    print("=" * 70)

    for pg, count in pg_counter.most_common():
        pct = 100 * count / len(results)
        R_vals = rgn_by_pg[pg]
        R_mean = np.mean(R_vals)
        print(f"  {pg:>8s}: {count:>6d} ({pct:>5.1f}%)  R(G,N) = {R_mean:.4f}")

    print(f"\n  Total classified: {len(results)}")
    print(f"  Total errors: {len(errors)}")

    # R(G,N) statistics
    all_R = [r["R"] for r in results]
    print("\n" + "=" * 70)
    print("R(G,N) DISTRIBUTION")
    print("=" * 70)
    print(f"  Mean R(G,N):   {np.mean(all_R):.4f}")
    print(f"  Median R(G,N): {np.median(all_R):.4f}")
    print(f"  Min R(G,N):    {np.min(all_R):.4f}")
    print(f"  Max R(G,N):    {np.max(all_R):.4f}")
    print(f"  Std R(G,N):    {np.std(all_R):.4f}")

    # Fraction with R < 1
    n_full = sum(1 for r in all_R if r >= 0.999)
    n_partial = sum(1 for r in all_R if r < 0.999)
    print(f"\n  R ≈ 1.0 (all modes observable): {n_full} ({100*n_full/len(all_R):.1f}%)")
    print(f"  R < 1.0 (some silent modes):   {n_partial} ({100*n_partial/len(all_R):.1f}%)")

    # Centrosymmetric fraction
    n_centro = sum(1 for r in results if r["is_centrosymmetric"])
    print(f"\n  Centrosymmetric molecules: {n_centro} ({100*n_centro/len(results):.1f}%)")

    # Hardest molecules (lowest R)
    print("\n" + "=" * 70)
    print("HARDEST MOLECULES (lowest R(G,N))")
    print("=" * 70)
    sorted_results = sorted(results, key=lambda x: x["R"])
    for r in sorted_results[:20]:
        print(f"  R={r['R']:.4f}  PG={r['point_group']:>6s}  N={r['N']:>2d}  "
              f"d={r['d']:>2d}  silent={r['n_silent']:>2d}  SMILES={r['smiles']}")

    # Save results
    save_data = {
        "n_molecules": len(results),
        "n_errors": len(errors),
        "tolerance": tolerance,
        "point_group_counts": dict(pg_counter.most_common()),
        "rgn_stats": {
            "mean": float(np.mean(all_R)),
            "median": float(np.median(all_R)),
            "min": float(np.min(all_R)),
            "max": float(np.max(all_R)),
            "std": float(np.std(all_R)),
        },
        "n_centrosymmetric": n_centro,
        "n_full_observable": n_full,
        "n_partial_observable": n_partial,
        "hardest_molecules": sorted_results[:50],
        "errors": errors[:50],
    }

    output_file = output_path / "qm9_point_groups.json"
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Also save per-molecule results as numpy arrays for fast loading
    np.savez(
        output_path / "qm9_rgn_data.npz",
        R=np.array(all_R),
        N=np.array([r["N"] for r in results]),
        d=np.array([r["d"] for r in results]),
        n_silent=np.array([r["n_silent"] for r in results]),
        is_centrosymmetric=np.array([r["is_centrosymmetric"] for r in results]),
    )

    return save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max molecules to process")
    parser.add_argument("--tolerance", type=float, default=0.3, help="Symmetry tolerance (Angstrom)")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    args = parser.parse_args()

    process_qm9_dataset(
        max_molecules=args.max,
        tolerance=args.tolerance,
        output_dir=args.output,
    )
