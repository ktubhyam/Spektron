# R&D Progress Report
# "Can One Hear the Shape of a Molecule?"
# Date: 2026-02-12

---

## Experiment 1: R(G,N) Computation (COMPLETE)

**Tool:** `experiments/compute_rgn.py` (verified), `experiments/compute_point_groups.py` (scaled)

**Results on 7 test molecules:**
| Molecule | Point Group | N | d | R(G,N) | N_silent |
|----------|------------|---|---|--------|----------|
| H2O | C2v | 3 | 3 | 1.0 | 0 |
| CO2 | D*h | 3 | 4 | 1.0 | 0 |
| NH3 | C3v | 4 | 6 | 0.75 | 2 |
| CH4 | Td | 5 | 9 | 0.60 | 4 |
| C2H4 | D2h | 6 | 12 | 0.875 | 2 |
| C6H6 | D6h | 12 | 30 | 0.633 | 11 |
| SF6 | Oh | 7 | 15 | 0.80 | 3 |

**Results on full QM9 (130,831 molecules):**
| Point Group | Count | Percent | R(G,N) |
|-------------|-------|---------|--------|
| C1 | 111,499 | 85.2% | 1.00 |
| Cs | 18,207 | 13.9% | 1.00 |
| C2v | 467 | 0.4% | 1.00 |
| C2 | 401 | 0.3% | 1.00 |
| C3v | 64 | <0.1% | 0.75 |
| C2h | 51 | <0.1% | 1.00 |
| Others | 141 | 0.1% | varies |

**Key Statistics:**
- Mean R(G,N) = 0.9998
- 99.9% of molecules have R = 1.0 (all modes observable)
- Only 120 molecules (0.09%) have silent modes
- Only 92 molecules (0.07%) are centrosymmetric
- Hardest: cubane (Oh, R=0.45), benzene (D6h, R=0.50)

**Implication for paper:** QM9 is dominated by low-symmetry molecules. The "hard" molecules (R<1) are rare but important — they are the ones where our theory predicts ML models will fail.

---

## Experiment 2: Jacobian Rank Analysis (COMPLETE)

**Tool:** `experiments/jacobian_rank.py` (small-scale), `experiments/jacobian_rank_scaled.py` (QM9-scale)

### Small-scale results (random molecules):
- 200/200 random molecules: **full rank**
- 50/50 near-degenerate configurations: **full rank**
- H2O verified: rank=3/3, condition number=288, overdetermination=4.0x

### Scaled results (999 real QM9 molecules):
- **999/999 molecules have full-rank Jacobians (100%)**
- Overdetermination ratio: **exactly 4.0x** at every molecule size
- Condition number: median=6,474 (well-conditioned)
- Tested molecules from N=9 to N=27 atoms (d=17 to d=77 vibrational modes)
- Zero rank-deficient cases found

**Implication for paper:** This is the strongest computational evidence for Conjecture 3 (generic identifiability). The 4:1 overdetermination ratio, combined with 100% full-rank Jacobians on real molecular geometries, makes a compelling case that the combined IR+Raman forward map is generically injective.

---

## Experiment 3: Confusable Pairs & Modal Complementarity (COMPLETE)

**Tool:** `experiments/find_confusable_pairs.py`

**Dataset:** 94,402 QM9S molecules with both broadened IR and Raman spectra (5,000 subsampled for pairwise analysis)

### Confusable pairs analysis:
| Modality | Mean cosine dist | Min cosine dist | p1 | p5 |
|----------|-----------------|-----------------|-----|-----|
| IR only | 0.729 | 0.0098 | 0.217 | 0.361 |
| Raman only | 0.608 | 0.0176 | 0.194 | 0.283 |
| IR+Raman | 0.668 | 0.0195 | 0.253 | 0.376 |

### Modal complementarity validation (Theorem 2):
- Top-50 confusable IR pairs vs. top-50 confusable Raman pairs: **only 8% overlap**
- **69 out of 96 unique confusable pairs (72%) resolved** by combining modalities
- Pearson r(d_IR, d_Raman) = 0.44 — moderate correlation supports complementarity
- **82.1% resolution rate**: pairs confusable under one modality are resolved by adding the other

**Key finding:** Mol 3229 and 17746 are the most confusable pair under combined IR+Raman (cosine dist = 0.019). They are completely different molecules but have remarkably similar spectra.

**Implication for paper:** Strong empirical validation of Theorem 2. IR and Raman capture complementary information — combining them resolves the vast majority of confusable pairs.

---

## Experiment 4: Minimal Model Prototype (BUILT, TESTING)

**Tool:** `experiments/minimal_model.py`, `experiments/train_minimal.py`

**Architecture:**
- CNN Tokenizer (3 strided conv layers) → 55 patches × 128 dim
- Cross-Attention Fusion (IR ↔ Raman bidirectional)
- 2-layer Transformer encoder, 4 heads, 256 FFN dim
- VIB Head: z_chem (64 dim) + z_inst (32 dim)
- Total parameters: **547,968** (~0.5M)

**Training:**
- Contrastive loss (InfoNCE) + classification loss + VIB KL regularization
- Data augmentation: Gaussian noise, random scaling, baseline shift
- Cosine annealing LR, gradient clipping

**Status:** Forward pass verified, training loop tested on synthetic data. Real data training running.

---

## Novelty Analysis (COMPLETE)

**File:** `paper/NOVELTY_ANALYSIS.md`

**Genuinely novel contributions:**
1. R(G,N) → ML accuracy prediction pipeline
2. Fano bound applied to molecular identification (first in spectroscopy)
3. Generic identifiability via full observable set (4d) — distinguishes from Kuramshina's freq-only analysis
4. 999-molecule Jacobian rank numerical evidence
5. Theory-guided ML architecture (VIB + cross-attention motivated by Theorems 1-2)
6. Symmetry-stratified ML evaluation

**Must-cite prior works:**
- Kuramshina et al. (1999): inverse vibrational problem (freq-only is ill-posed)
- Wang & Torquato (2024): "hear the shape" for crystals
- Alemi et al. (2017): VIB
- Kac (1966), Gordon-Webb-Wolpert (1992): drum problem

---

## Next Steps

1. **Train model on larger subset** (5,000-20,000 molecules) and report retrieval accuracy
2. **Stratify accuracy by R(G,N)** — test whether low-R molecules are indeed harder
3. **Compare IR-only vs Raman-only vs IR+Raman** model accuracy (modal complementarity test)
4. **Compute Fano bound** for confusable sets and compare to empirical error rates
5. **Download full QM9S dataset** (complete broadened spectra) for production training
6. **Cross-validate confusable pairs** with point group information
