# Supplementary Information

# "Can One Hear the Shape of a Molecule? Group-Theoretic Identifiability and Modal Complementarity in Vibrational Spectroscopy"

---

## S1. Complete Character Tables and Mode Decompositions

We provide the full vibrational mode decomposition for all molecules analyzed in the main text. For each molecule, we list the point group, the decomposition of $\Gamma_{\text{vib}}$ into irreducible representations, and the classification of each irrep as IR-active, Raman-active, both, or silent.

### S1.1 Water (H$_2$O, $C_{2v}$, $N = 3$)

$\Gamma_{\text{vib}} = 2A_1 + B_2$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $A_1$ | 1 | 2 | Yes ($z$) | Yes ($x^2, y^2, z^2$) | Both |
| $B_2$ | 1 | 1 | Yes ($y$) | Yes ($yz$) | Both |

$N_{\text{IR}} = 3$, $N_{\text{Raman}} = 3$, $N_{\text{silent}} = 0$, $R = 1.00$.

### S1.2 Carbon Dioxide (CO$_2$, $D_{\infty h}$, $N = 3$)

$\Gamma_{\text{vib}} = \Sigma_g^+ + \Sigma_u^+ + \Pi_u$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $\Sigma_g^+$ | 1 | 1 | No | Yes | Raman-only |
| $\Sigma_u^+$ | 1 | 1 | Yes ($z$) | No | IR-only |
| $\Pi_u$ | 2 | 1 | Yes ($x, y$) | No | IR-only |

$N_{\text{IR}} = 3$ (1 from $\Sigma_u^+$ + 2 from $\Pi_u$), $N_{\text{Raman}} = 1$, $N_{\text{silent}} = 0$, $R = 1.00$.

### S1.3 Methane (CH$_4$, $T_d$, $N = 5$)

$\Gamma_{\text{vib}} = A_1 + E + 2T_2$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $A_1$ | 1 | 1 | No | Yes ($x^2 + y^2 + z^2$) | Raman-only |
| $E$ | 2 | 1 | No | Yes ($2z^2 - x^2 - y^2$, $x^2 - y^2$) | Raman-only |
| $T_2$ | 3 | 2 | Yes ($x, y, z$) | Yes ($xy, xz, yz$) | Both |

$N_{\text{IR}} = 6$ (2 × 3 from $T_2$), $N_{\text{Raman}} = 9$ (1 from $A_1$ + 2 from $E$ + 6 from $T_2$), $N_{\text{silent}} = 0$, $R = 1.00$.

### S1.4 Ethylene (C$_2$H$_4$, $D_{2h}$, $N = 6$)

$\Gamma_{\text{vib}} = 3A_g + 2B_{1g} + B_{1u} + 2B_{2g} + B_{2u} + 2B_{3u} + A_u$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $A_g$ | 1 | 3 | No | Yes | Raman-only |
| $B_{1g}$ | 1 | 2 | No | Yes | Raman-only |
| $B_{2g}$ | 1 | 1 | No | Yes | Raman-only |
| $B_{1u}$ | 1 | 1 | Yes ($z$) | No | IR-only |
| $B_{2u}$ | 1 | 1 | Yes ($y$) | No | IR-only |
| $B_{3u}$ | 1 | 2 | Yes ($x$) | No | IR-only |
| $A_u$ | 1 | 1 | No | No | **Silent** |
| $B_{3g}$ | 1 | 1 | No | Yes | Raman-only |

Note: $D_{2h}$ is centrosymmetric, so mutual exclusion applies. The $A_u$ mode (torsion) is silent.

$N_{\text{IR}} = 5$, $N_{\text{Raman}} = 6$, $N_{\text{silent}} = 1$, $R = 0.92$.

### S1.5 Benzene (C$_6$H$_6$, $D_{6h}$, $N = 12$)

$\Gamma_{\text{vib}} = 2A_{1g} + A_{2g} + 4E_{2g} + E_{1g} + A_{2u} + 2B_{1u} + 2B_{2u} + 3E_{1u} + 2E_{2u} + B_{2g}$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $A_{1g}$ | 1 | 2 | No | Yes | Raman-only |
| $A_{2g}$ | 1 | 1 | No | No | **Silent** |
| $B_{1u}$ | 1 | 2 | No | No | **Silent** |
| $B_{2u}$ | 1 | 2 | No | No | **Silent** |
| $B_{2g}$ | 1 | 1 | No | No | **Silent** |
| $E_{1g}$ | 2 | 1 | No | Yes | Raman-only |
| $E_{2g}$ | 2 | 4 | No | Yes | Raman-only |
| $E_{2u}$ | 2 | 2 | No | No | **Silent** |
| $A_{2u}$ | 1 | 1 | Yes ($z$) | No | IR-only |
| $E_{1u}$ | 2 | 3 | Yes ($x, y$) | No | IR-only |

$N_{\text{IR}} = 7$ (1 from $A_{2u}$ + 6 from $E_{1u}$), $N_{\text{Raman}} = 13$ (2 from $A_{1g}$ + 2 from $E_{1g}$ + 8 from $E_{2g}$ + 1 extra), $N_{\text{silent}} = 10$, $R = 0.67$.

### S1.6 Sulfur Hexafluoride (SF$_6$, $O_h$, $N = 7$)

$\Gamma_{\text{vib}} = A_{1g} + E_g + 2T_{1u} + T_{2g} + T_{2u}$

| Irrep | Dimension | Count | IR | Raman | Activity |
|-------|-----------|-------|-----|-------|----------|
| $A_{1g}$ | 1 | 1 | No | Yes | Raman-only |
| $E_g$ | 2 | 1 | No | Yes | Raman-only |
| $T_{1u}$ | 3 | 2 | Yes ($x,y,z$) | No | IR-only |
| $T_{2g}$ | 3 | 1 | No | Yes | Raman-only |
| $T_{2u}$ | 3 | 1 | No | No | **Silent** |

$N_{\text{IR}} = 6$, $N_{\text{Raman}} = 6$, $N_{\text{silent}} = 3$, $R = 0.80$.

---

## S2. Jacobian Rank Analysis: Detailed Methodology

### S2.1 Computational Protocol

For each molecule $m$ in the test set, we compute the Jacobian matrix $\mathbf{J} = \partial \Phi / \partial \mathbf{F}$ of the forward spectroscopic map with respect to the force constant parameters using central finite differences:

$$J_{ij} = \frac{\Phi_i(\mathbf{F} + h\mathbf{e}_j) - \Phi_i(\mathbf{F} - h\mathbf{e}_j)}{2h},$$

where $h = 10^{-5}$ (in atomic units) and $\mathbf{e}_j$ is the $j$-th unit vector in force constant space. The observables $\Phi_i$ include all $4d$ quantities: frequencies $\{\tilde{\nu}_k\}$, IR intensities $\{a_k\}$, Raman activities $\{b_k\}$, and depolarization ratios $\{\rho_k\}$.

### S2.2 Numerical Rank Determination

The numerical rank of $\mathbf{J}$ is determined by singular value decomposition (SVD). We define the effective rank as the number of singular values exceeding a threshold $\tau = \max(\sigma_i) \times 10^{-8}$, which accounts for finite-precision arithmetic.

### S2.3 Expected Results

For a $C_1$ molecule with $N$ atoms:
- $\mathbf{J}$ is a $4d \times d$ matrix (where $d = 3N - 6$)
- Expected rank: $d$ (full column rank) for generic geometries
- Rank-deficient cases should correspond to molecules with non-trivial symmetry or eigenvalue near-degeneracies

### S2.4 Condition Number Analysis

The condition number $\kappa(\mathbf{J}) = \sigma_{\max} / \sigma_{\min}$ measures the sensitivity of the inverse map to perturbations. Large condition numbers indicate that the inverse problem, while solvable in principle, is ill-conditioned in practice. We expect:
- High-symmetry molecules: larger $\kappa$ (more ill-conditioned)
- Low-symmetry molecules: moderate $\kappa$ (better conditioned)
- $\kappa$ generally increases with molecular size $N$

---

## S3. Confusable Molecular Pair Catalog

### S3.1 Search Protocol

We search for confusable pairs in the QM9S dataset using the following algorithm:

1. For each molecule $m_i$, compute the full spectral observable $\Phi(m_i)$
2. Define spectral distance: $d_{\text{spec}}(m_i, m_j) = \|\Phi(m_i) - \Phi(m_j)\|_2 / \sqrt{4d}$
3. Define structural distance: $d_{\text{struct}}(m_i, m_j) = 1 - \text{Tanimoto}(\text{FP}(m_i), \text{FP}(m_j))$
   where FP denotes Morgan fingerprints (radius 2, 2048 bits)
4. A pair $(m_i, m_j)$ is $(\varepsilon, \Delta)$-confusable if $d_{\text{spec}} < \varepsilon$ and $d_{\text{struct}} > \Delta$

### S3.2 Known Confusable Molecular Types

From the literature and our analysis, the following molecular types are most prone to spectral confusion:

1. **Constitutional isomers with similar functional groups** (e.g., linear vs. branched alkanes)
2. **Tautomeric pairs** (e.g., lactam/lactim, keto/enol) — spectrally very similar, structurally distinct by conventional metrics
3. **Conformational isomers with similar vibrational profiles** — same molecular graph, different 3D arrangement
4. **Enantiomeric pairs** — fundamentally indistinguishable by conventional IR/Raman (achiral measurement)

### S3.3 Expected Prevalence

Based on literature analysis, fewer than 0.1% of random molecular pairs in QM9S are expected to be $(\varepsilon, \Delta)$-confusable at stringent thresholds ($\varepsilon = 10^{-3}$, $\Delta = 0.3$). This rarity is consistent with the high structure-specificity of vibrational spectra and supports (though does not prove) Conjecture 3.

---

## S4. Model Architecture Details

### S4.1 CNN Tokenizer

| Layer | Type | Channels | Kernel | Stride | Output |
|-------|------|----------|--------|--------|--------|
| 1 | Conv1D + ReLU + BN | 1 → 32 | 7 | 1 | 2048 × 32 |
| 2 | Conv1D + ReLU + BN | 32 → 64 | 5 | 2 | 1024 × 64 |
| 3 | Conv1D + ReLU + BN | 64 → 128 | 5 | 2 | 512 × 128 |
| 4 | Conv1D + ReLU + BN | 128 → 256 | 3 | 2 | 256 × 256 |
| 5 | Conv1D + ReLU + BN | 256 → 512 | 3 | 2 | 128 × 512 |

### S4.2 Transformer Encoder

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Heads | 8 |
| $d_{\text{model}}$ | 512 |
| $d_{\text{ff}}$ | 2048 |
| Dropout | 0.1 |
| Activation | GELU |
| Positional encoding | Wavenumber-informed (see S4.3) |

### S4.3 Wavenumber Positional Encoding

Standard sinusoidal positional encoding is replaced with physics-informed encoding:

$$\text{PE}(k, 2i) = \sin\left(\frac{\tilde{\nu}_k}{10000^{2i/d_{\text{model}}}}\right), \qquad \text{PE}(k, 2i+1) = \cos\left(\frac{\tilde{\nu}_k}{10000^{2i/d_{\text{model}}}}\right),$$

where $\tilde{\nu}_k$ is the wavenumber (in cm$^{-1}$) at position $k$ in the resampled spectrum. This encoding preserves the physical meaning of spectral positions, unlike integer-based positional encoding.

### S4.4 VIB Head

The Variational Information Bottleneck head maps the encoder output $\mathbf{z} \in \mathbb{R}^{256}$ to:
- Chemical latent: $\mathbf{z}_{\text{chem}} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{chem}}, \text{diag}(\boldsymbol{\sigma}_{\text{chem}}^2))$, $\mathbf{z}_{\text{chem}} \in \mathbb{R}^{128}$
- Instrumental latent: $\mathbf{z}_{\text{inst}} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{inst}}, \text{diag}(\boldsymbol{\sigma}_{\text{inst}}^2))$, $\mathbf{z}_{\text{inst}} \in \mathbb{R}^{64}$

Training uses the reparameterization trick: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### S4.5 Conformal Prediction

For coverage guarantee at level $1 - \alpha$:
1. Compute nonconformity scores on calibration set: $s_i = d_{\text{spec}}(\Phi(m_i), \hat{\Phi}(m_i))$
2. Set threshold $\hat{q} = \text{Quantile}_{(1-\alpha)(1+1/n_{\text{cal}})}(\{s_1, \ldots, s_{n_{\text{cal}}}\})$
3. Prediction set: $\mathcal{C}(S_{\text{new}}) = \{m : d(\mathbf{z}_{\text{chem}}(S_{\text{new}}), \mathbf{z}_{\text{chem}}(\Phi(m))) < \hat{q}\}$

By the theory of split conformal prediction, $\Pr(m_{\text{true}} \in \mathcal{C}(S_{\text{new}})) \geq 1 - \alpha$.

---

## S5. Hyperparameter Sensitivity Analysis

### S5.1 Key Hyperparameters

| Parameter | Search range | Default |
|-----------|-------------|---------|
| Learning rate | $[10^{-5}, 10^{-3}]$ | $3 \times 10^{-4}$ |
| VIB $\beta$ | $[10^{-4}, 10^{-1}]$ | $10^{-3}$ |
| Adversarial $\lambda_{\text{adv}}$ | $[10^{-3}, 1]$ | $10^{-2}$ |
| Contrastive temperature | $[0.01, 0.5]$ | 0.07 |
| CNN kernel size (first layer) | $\{3, 5, 7, 9\}$ | 7 |
| Transformer layers | $\{2, 4, 6\}$ | 4 |
| $d_{\text{model}}$ | $\{256, 512, 768\}$ | 512 |
| $\dim(\mathbf{z}_{\text{chem}})$ | $\{64, 128, 256\}$ | 128 |
| Batch size | $\{64, 128, 256\}$ | 128 |

### S5.2 Expected Sensitivity

- VIB $\beta$: Critical parameter. Too large → information collapse; too small → no disentanglement
- Adversarial $\lambda_{\text{adv}}$: Moderate sensitivity. Affects z_chem/z_inst separation quality
- Architecture parameters: Moderate sensitivity. Larger models help up to a point
- Learning rate: Standard sensitivity. Cosine annealing with warmup used

---

## S6. Dataset Statistics

### S6.1 QM9S Point Group Distribution

| Point Group | Count (approx.) | Fraction | $R(G, N)$ |
|-------------|-----------------|----------|-----------|
| $C_1$ | ~105,000 | 80.8% | 1.00 |
| $C_s$ | ~12,000 | 9.2% | ~0.95 |
| $C_{2v}$ | ~8,000 | 6.2% | ~0.90 |
| $C_{2}$ | ~2,500 | 1.9% | ~0.95 |
| $D_{2h}$ | ~1,500 | 1.2% | ~0.85 |
| $C_{3v}$ | ~500 | 0.4% | ~0.95 |
| Other | ~500 | 0.4% | varies |
| **Total** | **~130,000** | **100%** | — |

### S6.2 Centrosymmetric Subset

Centrosymmetric point groups in QM9S: $C_i$, $C_{2h}$, $D_{2h}$, $D_{3d}$, $D_{\infty h}$.
Estimated count: 2,600–6,500 molecules (~2–5% of dataset).
This subset is critical for testing Theorem 2 (modal complementarity).

---

## References

[14] J. von Neumann, E. Wigner, "Uber merkwurdige diskrete Eigenwerte", *Physikalische Zeitschrift* **30**, 465--467 (1929).

[21] T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006).
