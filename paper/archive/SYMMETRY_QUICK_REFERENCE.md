# Symmetry & Identifiability Quick Reference

**For Spektron Paper — Equations, Tables, and Key Facts**

---

## Key Equations

### Vibrational Degrees of Freedom
```
Non-linear molecules:  N_vib = 3N - 6
Linear molecules:      N_vib = 3N - 5
```
where N = number of atoms

### Vibrational Representation Decomposition
```
Γ_vib = Γ_total - Γ_trans - Γ_rot
```
where:
- Γ_total: representation for all 3N degrees of freedom
- Γ_trans: translational modes (x, y, z)
- Γ_rot: rotational modes (Rx, Ry, Rz)

### Selection Rules

**IR-active:** Mode transforms as (x, y, or z)
- For centrosymmetric molecules: ungerade (u) only

**Raman-active:** Mode transforms as (x², y², z², xy, xz, or yz)
- For centrosymmetric molecules: gerade (g) only

**Silent modes:** Neither IR nor Raman active
- Do not transform as translations OR quadratic functions

### Mutual Exclusion Principle
For molecules with inversion symmetry (i):
```
IR-active ∩ Raman-active = ∅
```
No mode can be both IR and Raman active.

### Identifiability Bound (Proposed)
```
I(M|S) ≤ I_IR + I_Raman - I_overlap - I_silent + I_degeneracy_penalty

where:
  I_IR     = information from IR spectrum
  I_Raman  = information from Raman spectrum
  I_overlap = redundant information (zero for centrosymmetric molecules)
  I_silent = unobservable modes
  I_degeneracy_penalty = information loss due to degenerate modes
```

### Complementarity for Centrosymmetric Molecules
```
I(M | S_IR, S_Raman) > I(M | S_IR) + I(M | S_Raman)
```
Synergy term due to mutual exclusion → perfect complementarity

---

## Point Group Classification Tables

### Table 1: Common Point Groups by Symmetry Level

| Symmetry Level | Point Groups | Inversion? | Mutual Exclusion? | Examples |
|----------------|--------------|------------|-------------------|----------|
| None           | C₁           | No         | No                | CHFClBr  |
| Low            | Cs, C₂, C₂ᵥ  | No         | No                | H₂O, H₂O₂ |
| Moderate       | C₃ᵥ, D₂ₕ, D₃ₕ | D₂ₕ, D₃ₕ have i | D₂ₕ, D₃ₕ: Yes | NH₃, C₂H₄ |
| High           | D₄ₕ, D₆ₕ, Tₐ  | D₄ₕ, D₆ₕ have i | D₄ₕ, D₆ₕ: Yes | C₆H₆, CH₄ |
| Very High      | Oₕ, Iₕ       | Yes        | Yes               | SF₆, C₆₀  |

### Table 2: Centrosymmetric Point Groups (Have Inversion i)

**Linear:** D∞ₕ (CO₂, acetylene)

**Planar:** D₂ₕ, D₃ₕ, D₄ₕ, D₆ₕ, D₈ₕ

**Trigonal/Rhombohedral:** D₃ᵈ, D₅ᵈ, S₆ (= C₃ᵢ)

**Cubic:** Tₕ, Oₕ

**Other:** Ci, C₂ₕ, S₂ (= Ci), S₄, S₈

### Table 3: Degeneracy Notation (Mulliken Symbols)

| Symbol | Dimension | Degeneracy | Example Point Groups |
|--------|-----------|------------|----------------------|
| A, B   | 1         | Non-degenerate | All groups |
| E      | 2         | Doubly degenerate | C₃ᵥ, D₃ₕ, D₆ₕ |
| T (or F) | 3       | Triply degenerate | Tₐ, Tₕ, Oₕ |
| G      | 4         | Quadruply degenerate | Rare (icosahedral) |

**Subscripts:**
- **g/u:** gerade (even) / ungerade (odd) under inversion
- **1/2:** symmetric / antisymmetric under C₂ perpendicular to principal axis
- **'/":** symmetric / antisymmetric under horizontal mirror plane σₕ

---

## Molecular Examples Summary

### Water (H₂O) — C₂ᵥ, 3 atoms, 3 modes

| Mode | Irrep | Type | Frequency (cm⁻¹) | IR | Raman |
|------|-------|------|------------------|----|----|
| ν₁   | A₁    | Symmetric stretch | 3657 | Yes | Yes |
| ν₂   | A₁    | Bend | 1595 | Yes | Yes |
| ν₃   | B₁    | Asymmetric stretch | 3756 | Yes | Yes |

**All modes both IR and Raman active (no mutual exclusion)**

---

### Carbon Dioxide (CO₂) — D∞ₕ, 3 atoms, 4 modes

| Mode | Irrep | Type | Frequency (cm⁻¹) | Degeneracy | IR | Raman |
|------|-------|------|------------------|------------|----|-------|
| ν₁   | Σg⁺   | Symmetric stretch | 1480 | 1 | No | Yes (pol.) |
| ν₂   | Πᵤ    | Bend | 526 | 2 | Yes (weak) | No |
| ν₃   | Σᵤ⁺   | Asymmetric stretch | 2565 | 1 | Yes (strong) | No |

**Perfect mutual exclusion (centrosymmetric)**
**Perfect complementarity: IR + Raman = all modes**

---

### Methane (CH₄) — Tₐ, 5 atoms, 9 modes

| Mode | Irrep | Type | Frequency (cm⁻¹) | Degeneracy | IR | Raman |
|------|-------|------|------------------|------------|----|-------|
| ν₁   | A₁    | Symmetric stretch | 2917 | 1 | No | Yes (pol.) |
| ν₂   | E     | Bend | 1534 | 2 | No | Yes |
| ν₃   | T₂    | Asymmetric stretch | 3019 | 3 | Yes | Yes |
| ν₄   | T₂    | Asymmetric bend | 1306 | 3 | Yes | Yes |

**9 modes → 4 observable frequencies (high degeneracy)**
**T₂ modes both IR and Raman active (partial overlap)**

---

### Benzene (C₆H₆) — D₆ₕ, 12 atoms, 30 modes

**IR-active modes:**
- A₂ᵤ: 1 mode at 675 cm⁻¹
- E₁ᵤ: 3×2 = 6 modes at 1035, 1479, 3036 cm⁻¹
- **Total: 4 IR peaks (representing 7 modes)**

**Raman-active modes:**
- 2A₁g + E₁g + 4E₂g
- **Total: 7 Raman peaks (representing 13 modes)**

**Silent modes:**
- B₁ᵤ, B₂ᵤ, E₂ᵤ
- **~10 modes completely unobservable**

**30 total modes → 11 observable frequencies (63% information loss)**
**Perfect mutual exclusion (centrosymmetric)**

---

### SF₆ — Oₕ, 7 atoms, 15 modes

| Mode | Irrep | Frequency (cm⁻¹) | Degeneracy | IR | Raman | Silent |
|------|-------|------------------|------------|----|-------|--------|
| ν₁   | A₁g   | 775 | 1 | No | Yes (pol.) | No |
| ν₂   | Eg    | 643 | 2 | No | Yes | No |
| ν₃   | T₁ᵤ   | 948 | 3 | Yes | No | No |
| ν₄   | T₁ᵤ   | 615 | 3 | Yes | No | No |
| ν₅   | T₂g   | 524 | 3 | No | Yes | No |
| ν₆   | T₂ᵤ   | 346 | 3 | No | No | **Yes** |

**15 modes → 5 observable frequencies (67% information loss)**
**IR: 2 peaks (6 modes)**
**Raman: 3 peaks (6 modes)**
**Silent: 3 modes**

---

### Ethylene (C₂H₄) — D₂ₕ, 6 atoms, 12 modes

**Raman-active only (g):** Ag, B₁g, B₂g, B₃g

**IR-active only (u):** B₁ᵤ, B₂ᵤ, B₃ᵤ

**Silent:** Aᵤ (H-C-H out-of-plane twist at 875 cm⁻¹)

**Perfect mutual exclusion (centrosymmetric)**
**Has silent mode (neither IR nor Raman)**

---

## Symmetry → Information Content Scaling

### General Trend
```
C₁ >> C₂ᵥ > D₂ₕ > D₆ₕ ≈ Tₐ > Oₕ
(Maximum information) → (Minimum information)
```

### Quantitative Metrics

**Observable Mode Fraction:**
```
f_obs = (# IR peaks + # Raman peaks - # overlapping peaks) / (3N - 6)
```

| Point Group | Example | f_obs (approx) | Identifiability |
|-------------|---------|----------------|-----------------|
| C₁          | CHFClBr | ~1.0           | Maximum         |
| C₂ᵥ         | H₂O     | ~1.0           | High            |
| D₂ₕ         | C₂H₄    | ~0.8           | Medium-High     |
| D₆ₕ         | C₆H₆    | ~0.37          | Low-Medium      |
| Tₐ          | CH₄     | ~0.44          | Low-Medium      |
| Oₕ          | SF₆     | ~0.33          | Low             |

---

## Key Theorems for Paper

### Theorem 1: Spectroscopic Identifiability Bound
**Statement:** The information content of a vibrational spectrum is bounded by the number of observable modes, accounting for degeneracy and mutual exclusion.

**Mathematical form:**
```
H(M | S_IR, S_Raman) ≥ H_silent + H_degeneracy + H_symmetry

where:
  H_silent     = entropy of silent modes (unobservable)
  H_degeneracy = entropy within degenerate subspaces (indistinguishable)
  H_symmetry   = entropy over orbits of point group G (quotient uncertainty)
```

### Theorem 2: Mutual Exclusion and Complementarity
**Statement:** For centrosymmetric molecules, IR and Raman spectra are perfectly complementary, providing superadditive information gain.

**Mathematical form:**
```
For point group G with inversion i:
  I(M; S_IR, S_Raman) > I(M; S_IR) + I(M; S_Raman)

The synergy term:
  ΔI = I(M; S_IR, S_Raman) - I(M; S_IR) - I(M; S_Raman) > 0
```

### Theorem 3: Symmetry Orbit Non-Identifiability
**Statement:** Molecules differing only by a symmetry transformation g ∈ G produce identical spectra and are fundamentally indistinguishable.

**Mathematical form:**
```
For molecular configurations M₁, M₂ in the same G-orbit:
  S(M₁) = S(M₂)  ⟹  P(M₁ | S) = P(M₂ | S)

The inverse map is defined only on the quotient space:
  S⁻¹: Spectra → Structures/G
```

---

## Computational Tools Quick List

1. **MolSym** (Python)
   - Point group detection
   - Symmetry element generation
   - Character table construction

2. **RDKit** (Python)
   - Atom symmetry classes (graph automorphism)
   - No direct point group detection

3. **QSym²** (Quantum chemistry)
   - Symbolic character tables
   - Symmetry-orbit analysis
   - Continuous symmetry measures

4. **Online databases:**
   - WebQC: https://www.webqc.org/symmetry.php
   - Gernot Katzer: http://gernot-katzers-spice-pages.com/character_tables/

---

## Paper Writing Checklist

### Introduction
- [ ] Mention symmetry as fundamental constraint
- [ ] State that Spektron is first foundation model accounting for symmetry

### Background
- [ ] 1-page subsection: "Group Theory and Selection Rules"
- [ ] Define point groups, irreps, selection rules
- [ ] State mutual exclusion principle
- [ ] Give benzene as concrete example

### Methods
- [ ] Justify VIB as separating G-equivariant (chemistry) from G-invariant (instrument)
- [ ] Explain OT alignment respects symmetry orbits
- [ ] Justify multi-modal (IR + Raman) pretraining via complementarity

### Experiments
- [ ] E-Symmetry-1: Ablation by point group class
- [ ] E-Symmetry-2: IR vs. Raman vs. IR+Raman on centrosymmetric molecules
- [ ] E-Symmetry-3: Degeneracy analysis in latent space
- [ ] E-Symmetry-4: VIB disentanglement quality vs. symmetry

### Results
- [ ] Table: Performance stratified by point group
- [ ] Figure: IR-only vs. Raman-only vs. IR+Raman (show superadditivity for centrosymmetric)

### Discussion
- [ ] Deep dive: Why symmetry matters for foundation models
- [ ] Connection to identifiability theory (cite arXiv:2511.08995)
- [ ] Limitations: Silent modes fundamentally unobservable

### References (Must Cite)
- [ ] arXiv:2511.08995 (Group-Theoretic Identifiability)
- [ ] arXiv:2003.09077 (Inverse Problems and Symmetry Breaking)
- [ ] Chemistry LibreTexts on selection rules
- [ ] Rule of mutual exclusion (Wikipedia or primary sources)

---

## Key Messages for Abstract

1. "Molecular point group symmetry fundamentally constrains spectroscopic identifiability through degeneracy and selection rules"

2. "For centrosymmetric molecules, IR and Raman spectra exhibit perfect mutual exclusion, making multi-modal pretraining essential"

3. "Spektron is the first foundation model for vibrational spectroscopy that explicitly respects symmetry constraints through VIB disentanglement and optimal transport"

4. "We demonstrate that high-symmetry molecules (D₆ₕ, Oₕ) have 60-70% lower observable information content than low-symmetry molecules (C₁, C₂ᵥ), explaining domain-specific transfer performance"

---

**End of Quick Reference**
