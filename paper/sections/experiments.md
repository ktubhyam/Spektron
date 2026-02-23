# Section 5: Experiments and Results

The theoretical framework of Section 3 makes specific, testable predictions about the identifiability of molecular structure from vibrational spectra. In this section we describe a suite of experiments designed to validate Theorems 1 and 2, to provide computational support for Conjecture 3, and to probe the practical implications of Proposition 1. We also evaluate the downstream consequences for calibration transfer and uncertainty quantification. Throughout, we describe the experimental design and protocol in detail and present predicted outcomes grounded in the theory; since the full experimental campaign is ongoing, we frame results using language appropriate for a pre-registered study, noting where theoretical predictions are sharp and where the outcomes will be most informative.


## 5.1 Experiment 1: Symmetry Stratification (Validating Theorem 1)

**Objective.** Theorem 1 establishes that the information completeness ratio $R(G, N)$ quantifies the fraction of a molecule's vibrational degrees of freedom accessible to IR and Raman spectroscopy. A direct corollary is that molecules with lower $R(G, N)$ possess fewer spectroscopic discriminants and should therefore be harder to identify from their spectra, regardless of the identification algorithm employed. This experiment tests whether $R(G, N)$ is predictive of empirical identification accuracy.

**Dataset.** We use the QM9S dataset [22], which contains approximately 130,000 small organic molecules (up to 9 heavy atoms: C, N, O, F) with DFT-computed vibrational spectra (IR and Raman) at the B3LYP/def2-TZVP level of theory. Each molecule has 3,000 spectral points spanning 0--4,000 cm$^{-1}$. Point group assignments are computed for all molecules using the MolSym package [23] applied to the DFT-optimized geometries, with a symmetry tolerance of 0.01 \AA.

**Protocol.** A single retrieval model is trained on the full QM9S dataset using both IR and Raman spectra as input (the architecture and training procedure are described in Section 4). No symmetry information is provided to the model at training or inference time; the model receives only spectral data. After training, we evaluate top-$k$ identification accuracy ($k = 1, 5, 10$) on a held-out test set, stratified by the point group $G$ of the query molecule. For each point group class, we compute $R(G, N)$ from the character table of $G$ and the vibrational representation $\Gamma_{\text{vib}}$, as described in Section 3.2. Molecules with fewer than 10 representatives in the test set are excluded from the stratified analysis to ensure statistical reliability.

**Predicted outcomes.** Based on Theorem 1, we expect the following rank ordering of identification accuracy across point groups:

**Table 2. Predicted relationship between $R(G, N)$ and identification accuracy.**

| Point Group $G$ | Representative Molecule | $R(G, N)$ | Predicted Accuracy Rank | Reasoning |
|-----------------|------------------------|-----------|------------------------|-----------|
| $C_1$ | Generic asymmetric | 1.00 | Highest | All modes observable; no silent modes; no degeneracy |
| $C_s$, $C_2$ | Methanol, H$_2$O$_2$ | $\approx 0.95$--$1.00$ | High | Minimal degeneracy; few or no silent modes |
| $C_{2v}$ | H$_2$O, formaldehyde | $\approx 0.90$--$1.00$ | Medium-High | All modes typically active, but degeneracy possible for larger molecules |
| $D_{2h}$ | Ethylene, naphthalene | $\approx 0.85$--$0.92$ | Medium | Mutual exclusion active; some silent modes ($A_u$ in ethylene) |
| $D_{6h}$ | Benzene | $\approx 0.67$ | Low | 10 silent modes; substantial information loss |
| $O_h$ | SF$_6$ | $\approx 0.60$--$0.80$ | Medium-Low | Extensive degeneracy; silent modes ($T_{2u}$) |

The central prediction is a positive monotonic relationship between $R(G, N)$ and top-$k$ accuracy: point groups with higher $R$ values should yield higher identification accuracy. Quantitatively, we expect the correlation to be strong but not perfect, because $R(G, N)$ captures only one aspect of the identification problem (mode observability) and does not account for other factors such as the number of distinct molecules in each symmetry class or the distribution of spectral distances within a class.

**Key figure (Figure 4): Scatter plot of $R(G, N)$ versus top-1 accuracy.** Each point in the scatter plot represents one point group class, with the horizontal axis showing $R(G, N)$ and the vertical axis showing the mean top-1 accuracy for molecules in that class. Error bars indicate 95% confidence intervals computed by bootstrapping over molecules within each class. A least-squares regression line is overlaid to quantify the trend. We predict a Pearson correlation coefficient of $\rho \geq 0.75$ between $R(G, N)$ and mean top-1 accuracy.

**Distribution of symmetry classes.** Since the QM9S dataset consists primarily of small organic molecules, we expect the distribution of point groups to be highly skewed: approximately 70--80% of molecules will belong to $C_1$ (no non-trivial symmetry), with smaller populations in $C_s$, $C_2$, $C_{2v}$, and $C_{3v}$. High-symmetry groups ($D_{6h}$, $O_h$) will have very few representatives, as they require specific molecular architectures (e.g., hexagonal rings, octahedral coordination) that are rare among small organic molecules. This skew is itself informative: the prevalence of $C_1$ molecules in chemical databases means that the "typical" molecule has $R = 1.0$ and is, from the standpoint of Theorem 1, maximally identifiable.

**Controls.** To verify that the observed accuracy differences are attributable to $R(G, N)$ rather than confounding factors (e.g., molecule size, functional group distribution), we will report accuracy as a function of $R(G, N)$ after controlling for molecular weight (by binning molecules into weight classes) and chemical composition (by reporting results separately for subsets matched on heavy atom count). If the $R(G, N)$--accuracy correlation persists after these controls, this constitutes strong evidence that symmetry-induced information loss, as quantified by Theorem 1, is a genuine driver of identification difficulty.


## 5.2 Experiment 2: Modal Complementarity (Validating Theorem 2)

**Objective.** Theorem 2 establishes that for centrosymmetric molecules, IR and Raman spectroscopy observe strictly disjoint sets of vibrational modes (the mutual exclusion rule), and consequently that combining the two modalities strictly increases the number of observable vibrational degrees of freedom. This experiment tests whether this theoretical prediction translates into a measurable difference in identification accuracy when comparing single-modality and multi-modal models.

**Models.** Three models are trained on the QM9S dataset, identical in architecture and hyperparameters except for the input modality:

- **Model A (IR-only):** Receives only the simulated IR spectrum as input.
- **Model B (Raman-only):** Receives only the simulated Raman spectrum as input.
- **Model C (IR+Raman):** Receives both IR and Raman spectra, processed through the cross-attention fusion mechanism described in Section 4.

All three models are trained for the same number of epochs with the same learning rate schedule, and evaluated on the same held-out test set.

**Stratification.** The test set is partitioned into two groups:

1. **Centrosymmetric molecules:** Those whose point group contains the inversion operation $i$ (groups $C_i$, $C_{2h}$, $D_{2h}$, $D_{3d}$, $D_{4h}$, $D_{6h}$, $D_{\infty h}$, $O_h$, etc.).
2. **Non-centrosymmetric molecules:** All remaining molecules.

We note that only approximately 2--5% of QM9S molecules are expected to be centrosymmetric, reflecting the rarity of centrosymmetric structures among small organic molecules with 9 or fewer heavy atoms. This small sample size will be addressed by supplementing the analysis with molecules from the larger QMe14S dataset [24] (186,000 molecules with 14 elements) and the ChEMBL IR-Raman dataset [25] (220,000 molecules with vibrational mode symmetry labels), which provide a richer pool of centrosymmetric structures.

**Metric.** The primary metric is the **complementarity gain**, defined as:

$$\Delta_{\text{comp}} = \text{Acc}(\text{IR+Raman}) - \max\bigl(\text{Acc}(\text{IR}), \text{Acc}(\text{Raman})\bigr),$$

where $\text{Acc}(\cdot)$ denotes top-1 identification accuracy. The complementarity gain measures how much the second modality improves identification beyond the better of the two single-modality models. Theorem 2 predicts that $\Delta_{\text{comp}}$ should be significantly larger for centrosymmetric molecules than for non-centrosymmetric molecules.

**Predicted outcomes.** We expect the following pattern of results:

**Table 3. Predicted identification accuracy by modality and symmetry class.**

| Metric | Non-centrosymmetric | Centrosymmetric |
|--------|---------------------|-----------------|
| Acc(IR-only) | Moderate | Low--Moderate |
| Acc(Raman-only) | Moderate | Low--Moderate |
| Acc(IR+Raman) | Moderate--High | High |
| $\Delta_{\text{comp}}$ | Small ($\leq 5$ pp) | Large ($\geq 10$ pp) |

The physical reasoning is as follows. For centrosymmetric molecules, IR and Raman observe strictly non-overlapping mode sets (Theorem 2, part (a)). Adding the second modality therefore provides access to an entirely new set of vibrational degrees of freedom that were invisible to the first modality alone. This should produce a large accuracy gain. For non-centrosymmetric molecules, by contrast, most modes are active in both IR and Raman (as exemplified by water in Example 1 of Section 2.2), so the second modality provides largely redundant coverage of already-observed modes. The accuracy gain should therefore be smaller.

**Key figure (Figure 5): Grouped bar chart of identification accuracy.** The figure contains two groups of bars (centrosymmetric and non-centrosymmetric), with three bars in each group corresponding to Models A, B, and C. The expected visual pattern is that the three bars are similar in height for non-centrosymmetric molecules, but Model C towers above Models A and B for centrosymmetric molecules. An inset panel shows the complementarity gain $\Delta_{\text{comp}}$ for each group with error bars, making the predicted difference in gain directly visible.

**What this experiment does NOT test.** We emphasize that Theorem 2 concerns *mode counting* --- the number of observable vibrational degrees of freedom --- and not *mutual information*. The complementarity gain in accuracy is a proxy for the increase in observable structural information, but it is not a direct measurement of information-theoretic quantities. In particular, a large complementarity gain is consistent with Theorem 2 but does not demonstrate superadditivity of mutual information (which would be mathematically impossible, as noted in Section 3.3). The second modality provides complementary *coverage* of the vibrational mode space, not synergistic *amplification* of information already present in the first modality.

**Secondary analysis: mode-level complementarity.** For molecules where the vibrational mode assignments are known (from the DFT calculations underlying QM9S), we will additionally compute the mode-level complementarity by examining which specific modes contribute to correct identifications. For centrosymmetric molecules, we expect that correct identifications by Model C will frequently depend on modes that are invisible to either Model A or Model B alone --- that is, the combined model succeeds precisely because it can "see" both the gerade and ungerade portions of the vibrational spectrum.


## 5.3 Experiment 3: Confusable Set Analysis (Validating Proposition 1)

**Objective.** Proposition 1 provides a Fano-inequality-based lower bound on the identification error for confusable molecular sets: collections of molecules that are spectrally similar but structurally distinct. This experiment constructs confusable sets from real spectroscopic data and evaluates whether the Fano bound is predictive of empirical identification failure.

**Confusable set construction.** Following Definition 2 (Section 3.5), we construct confusable sets from the QM9S dataset as follows:

1. **Spectral distance computation.** For all pairs of molecules in the dataset (or a computationally feasible subsample), we compute the spectral distance $d_{\text{spec}}$ as the Wasserstein-1 distance between the combined IR+Raman spectra, treated as probability distributions after area normalization. We also compute the Pearson correlation distance $d_{\text{corr}} = 1 - r$ as a complementary metric.

2. **Structural distance computation.** Structural distance $d_{\text{struct}}$ is computed as the Tanimoto distance ($1 - T$) between Morgan fingerprints (radius 2, 2048 bits) computed from the molecular graphs.

3. **Pair classification.** Molecular pairs are classified as:
   - **Confusable:** $d_{\text{spec}} < \varepsilon$ and $d_{\text{struct}} > \Delta$, where $\varepsilon$ and $\Delta$ are threshold parameters.
   - **Well-separated:** $d_{\text{spec}} > 3\varepsilon$ and $d_{\text{struct}} > \Delta$.

Based on prior analysis of vibrational spectroscopy databases [20], we expect truly confusable pairs (with high spectral similarity and low structural similarity) to be rare. The most promising sources of confusable molecules are:

- **Tautomeric pairs** (e.g., lactam-lactim tautomers such as 2-hydroxypyridine/2-pyridone, where the spectral similarity can reach 0.88--0.92 and the Tanimoto similarity is 0.65--0.75).
- **Positional isomers** (e.g., ortho/meta/para-substituted aromatics, though these are typically distinguishable in the fingerprint region).
- **Structural isomers with similar functional group composition** (e.g., $n$-propanol versus isopropanol).

If the natural frequency of confusable pairs is too low (fewer than 50 pairs at a given threshold), we will relax the spectral tolerance to $\varepsilon$ corresponding to a spectral similarity of 0.85 (rather than 0.90), or supplement with computationally generated near-isospectral pairs obtained by small perturbations of DFT geometries.

**Fano bound evaluation.** For each confusable set $\mathcal{C} = \{m_1, \ldots, m_K\}$, we compute the Fano lower bound:

$$P_{\text{error}} \geq 1 - \frac{I(M; S_{\text{IR}}, S_{\text{Raman}}) + \log 2}{\log K},$$

where the mutual information $I(M; S_{\text{IR}}, S_{\text{Raman}})$ is estimated using the Gaussian copula mutual information (GCMI) estimator [26] applied to the spectral embeddings of the molecules in $\mathcal{C}$. The empirical identification error of the model on $\mathcal{C}$ is then compared to the Fano bound.

**Predicted outcomes.** We expect the following:

1. **Confusable pairs are rare but exist.** Consistent with the findings of Varmuza and Karlovits [20] and with the high structure-specificity of the IR/Raman fingerprint region, we expect that the vast majority of molecular pairs are spectrally distinguishable. However, specific chemical classes --- particularly tautomeric pairs and closely related structural isomers --- should yield confusable sets of modest size ($K = 2$--$10$).

2. **Model accuracy degrades on confusable sets.** We predict that the model's top-1 accuracy on confusable sets will drop substantially below its overall accuracy (e.g., from approximately 40--45% overall to approximately 10--20% on confusable sets).

3. **Empirical error is consistent with the Fano bound.** For confusable sets where the spectral distance is small, the mutual information $I(M; S)$ will be low relative to $\log K$, and the Fano bound will predict a high error probability. We expect the model's empirical error to lie at or above the Fano bound, consistent with the bound being a genuine lower limit.

**Key figure (Figure 6): Spectral distance versus structural distance scatter plot.** Each point represents a molecular pair, with the horizontal axis showing $d_{\text{spec}}$ and the vertical axis showing $d_{\text{struct}}$. A shaded rectangular region in the upper-left corner marks the confusable zone (low spectral distance, high structural distance). Points are colored by the model's identification outcome (correct or incorrect). The expected visual pattern is that incorrect identifications concentrate in or near the confusable zone, with a gradient of increasing error probability as one moves toward smaller spectral distances and larger structural distances. An inset panel shows the empirical error probability versus the Fano bound for confusable sets of varying size $K$, with the Fano bound plotted as a dashed diagonal reference.

**Connection to Theorems 1 and 2.** We will additionally stratify the confusable set analysis by symmetry class and modality. Theorem 1 predicts that confusable sets should be larger (more molecules per set) for high-symmetry point groups, where fewer spectral discriminants are available. Theorem 2 predicts that confusable sets should be smaller when both IR and Raman data are used for centrosymmetric molecules, since the two modalities probe entirely non-overlapping mode sets and thus provide maximal discriminative coverage.


## 5.4 Experiment 4: Jacobian Rank Analysis (Supporting Conjecture 3)

**Objective.** Conjecture 3 posits that for generic molecules (those outside a measure-zero exceptional set), the combined IR+Raman spectrum determines the molecular force constants uniquely up to symmetry equivalence. A necessary condition for this is that the Jacobian $\mathbf{J} = \partial \Phi / \partial \mathbf{F}$ has full column rank at generic configurations. This experiment computes the Jacobian rank and condition number for a large sample of molecules and examines whether rank deficiency correlates with the symmetry-based exceptional set predicted by the theory.

**Setup.** For each molecule in a randomly selected test set of 2,000 molecules from QM9S (stratified by point group to ensure representation of both low- and high-symmetry species), we compute the Jacobian of the observation map $\Phi$ with respect to the force constant parameters $\mathbf{F}$. The computation proceeds as follows:

1. **Force constant parametrization.** We adopt a diagonal valence force field (DVFF), in which one independent force constant $F_{ii}$ is assigned to each internal coordinate $s_i$. This yields $d_F = d = 3N - 6$ independent parameters. For selected molecules, we also consider the nearest-neighbor valence force field (NNVFF), which includes nearest-neighbor cross-terms and has $d_F \approx 2d$--$3d$ parameters.

2. **Jacobian computation via finite differences.** For each force constant $F_{ij}$, the Jacobian element $\partial \Phi_k / \partial F_{ij}$ is approximated by central finite differences:

$$\frac{\partial \Phi_k}{\partial F_{ij}} \approx \frac{\Phi_k(\mathbf{F} + h \, \mathbf{e}_{ij}) - \Phi_k(\mathbf{F} - h \, \mathbf{e}_{ij})}{2h},$$

where $h = 10^{-4}$ (in atomic units) and $\mathbf{e}_{ij}$ is the unit vector in the $F_{ij}$ direction. For each perturbation, the eigenvalues and eigenvectors of $\mathbf{GF}$ are recomputed, and the full observable vector $\Phi = \{(\tilde{\nu}_k, a_k, b_k, \rho_k)\}$ is reconstructed.

3. **Rank and condition number.** The numerical rank of $\mathbf{J}$ is determined as the number of singular values exceeding a threshold of $10^{-8}$ times the largest singular value. The condition number $\kappa(\mathbf{J}) = \sigma_{\max} / \sigma_{\min}$ is computed from the singular value decomposition, where $\sigma_{\min}$ is the smallest non-negligible singular value.

**Predicted outcomes.** Based on the evidence presented in Section 3.4, we expect the following:

1. **Full rank at generic configurations.** For molecules with trivial symmetry ($C_1$), the Jacobian $\mathbf{J}$ should have full column rank ($\text{rank}(\mathbf{J}) = d_F$) in all tested cases. This is because the $4d$ observables generically overdetermine the $d$ force constant parameters by a factor of 4, and the Hellmann-Feynman derivatives $\partial \lambda_k / \partial F_{ij}$ are generically nonzero for non-degenerate eigenvalue configurations.

2. **Rank deficiency at high-symmetry configurations.** For molecules with non-trivial symmetry, we expect rank deficiency ($\text{rank}(\mathbf{J}) < d_F$) at configurations where eigenvalue degeneracies occur. The rank deficiency should equal the number of "hidden" force constant degrees of freedom that cannot be resolved from the available spectral data --- typically the off-diagonal force constants coupling modes within a degenerate set.

3. **Condition number correlates with symmetry.** The condition number $\kappa(\mathbf{J})$ should be systematically larger for high-symmetry molecules (reflecting the ill-conditioning introduced by near-degeneracies) and smaller for low-symmetry molecules (reflecting the well-conditioned overdetermined system).

**Key figure (Figure 7): Histogram of normalized Jacobian rank.** The figure displays a histogram of the ratio $\text{rank}(\mathbf{J}) / d_F$ across the 2,000 test molecules. We predict that the histogram will show a sharp peak at 1.0 (corresponding to full rank at generic configurations) with a small tail extending to values below 1.0. This tail should consist predominantly of molecules with non-trivial point group symmetry. An inset scatter plot shows $\kappa(\mathbf{J})$ versus the point group order $|G|$, where we expect a positive trend (higher symmetry $\to$ larger condition number).

**Additional analysis: the exceptional set.** For each molecule where $\text{rank}(\mathbf{J}) < d_F$, we will examine whether the rank deficiency can be explained by the symmetry structure predicted by the theory. Specifically, we will check whether the null space of $\mathbf{J}$ corresponds to force constant perturbations that preserve the eigenvalue degeneracy pattern of the molecule's point group --- that is, perturbations within the "kernel of the observation" associated with the symmetry. If this correspondence holds consistently, it provides strong computational support for the conjecture that the exceptional set (where $\tilde{\Phi}$ fails to be injective) is contained within the set of symmetric configurations, and hence has measure zero in $\mathcal{M}/G$.

**Limitations.** We emphasize that a positive result in this experiment --- full Jacobian rank at all tested generic configurations --- constitutes *evidence for* Conjecture 3, not a *proof*. The conjecture asserts a measure-theoretic statement (injectivity outside a measure-zero set) that cannot be established by testing a finite sample, however large. Moreover, the Jacobian analysis concerns only *local* injectivity (whether $\Phi$ is a local diffeomorphism); establishing *global* injectivity (whether $\Phi^{-1}(s)$ contains a single point for almost every $s$) requires ruling out the existence of distinct molecular configurations that map to the same spectral point through different branches of the forward map.


## 5.5 Experiment 5: Calibration Transfer and Uncertainty Quantification

The preceding experiments address the fundamental identifiability question using simulated DFT spectra. In this section, we evaluate the practical consequences of our theoretical framework for two applied tasks: calibration transfer across instruments and uncertainty quantification for spectral identification.

### 5.5.1 Calibration Transfer (Experiment E5)

**Objective.** Calibration transfer --- the task of adapting a spectroscopic model trained on one instrument to make accurate predictions on spectra from a different instrument --- is a central challenge in applied vibrational spectroscopy. From the perspective of our framework, instrument variability acts as a nuisance transformation that does not alter the molecular chemistry (encoded in $z_{\text{chem}}$) but does alter the observed spectrum through instrument-specific response functions (encoded in $z_{\text{inst}}$). The VIB disentanglement of Section 4, which factorizes the latent representation into chemical and instrumental components, is designed to facilitate transfer by isolating the chemistry-bearing signal from the instrument-specific artifact.

**Datasets.** We use two standard calibration transfer benchmarks:

- **Corn dataset [27].** 80 corn samples measured on 3 NIR instruments (m5, mp5, mp6), with 700 spectral channels each. Properties: moisture, oil, protein, starch. This is the standard benchmark in calibration transfer literature.
- **Tablet dataset [28].** 655 pharmaceutical tablet samples measured on 2 NIR instruments, with 650 channels each. Properties: active pharmaceutical ingredient content, tablet weight, hardness. The dataset is partitioned into calibration (155 samples), validation (40 samples), and test (460 samples) subsets.

**Protocol.** The model is first pretrained on the large-scale QM9S/ChEMBL spectral corpus using the self-supervised objectives described in Section 4 (masked spectral reconstruction, contrastive learning, and denoising). The pretrained backbone is then fine-tuned on the source instrument (e.g., m5 for corn) using LoRA adapters [29] (rank $r = 8$, $\alpha = 16$, dropout 0.05), training only 1--2% of parameters. Transfer to the target instrument (e.g., mp5 or mp6) is evaluated at $k = \{0, 1, 5, 10, 25\}$ labeled transfer samples from the target domain. For $k = 0$ (zero-shot), the model relies entirely on the pretrained VIB disentanglement and optional test-time training (TTT): at inference, $K_{\text{TTT}} = 50$ steps of masked spectral reconstruction are performed on the unlabeled target spectra to adapt the model's instrument embedding without any labeled target data.

**Baselines.** We compare against established calibration transfer methods:

- **PDS (Piecewise Direct Standardization) [30]:** Constructs a local transfer matrix from a small window of spectral channels.
- **SBC (Slope/Bias Correction) [31]:** Corrects for linear shifts in the prediction space.
- **di-PLS (Domain-Invariant Partial Least Squares) [32]:** Projects spectra into a domain-invariant subspace.
- **LoRA-CT [33]:** Fine-tunes low-rank adapters for calibration transfer (current state-of-the-art, $R^2 = 0.952$ on corn moisture with $k = 10$).

**Predicted outcomes.** We predict the following results for corn moisture prediction (the primary benchmark in calibration transfer literature):

**Table 4. Predicted $R^2$ for corn moisture transfer (m5 $\to$ mp5).**

| Method | $k = 0$ | $k = 1$ | $k = 5$ | $k = 10$ | $k = 25$ |
|--------|---------|---------|---------|----------|----------|
| PDS | -- | 0.72 | 0.88 | 0.93 | 0.95 |
| SBC | -- | 0.65 | 0.82 | 0.90 | 0.94 |
| di-PLS | 0.60 | 0.75 | 0.90 | 0.94 | 0.96 |
| LoRA-CT | -- | 0.82 | 0.92 | 0.952 | 0.97 |
| Ours (VIB) | 0.75 | 0.88 | 0.94 | **0.960** | **0.98** |
| Ours (VIB + TTT) | **0.82** | **0.91** | **0.95** | **0.970** | **0.98** |

The key predictions are:

1. **Zero-shot transfer.** The VIB disentanglement, which separates $z_{\text{chem}}$ from $z_{\text{inst}}$, should enable meaningful zero-shot transfer ($R^2 \geq 0.75$) by discarding the instrument-specific component. Test-time training should further improve zero-shot performance to approximately $R^2 = 0.82$ by adapting the model to the target instrument's spectral characteristics through self-supervised learning on unlabeled target data.

2. **Few-shot transfer.** With $k = 10$ labeled transfer samples, we expect to match or exceed the current state-of-the-art LoRA-CT result of $R^2 = 0.952$. The VIB pretraining provides a richer initialization for transfer than LoRA-CT's architecture-only approach.

3. **Sample efficiency.** We predict that our method with $k = 5$ samples will match LoRA-CT's performance at $k = 10$, representing a 2$\times$ improvement in sample efficiency for calibration transfer.

**Key figure (Figure 8): Sample efficiency curves.** The figure plots $R^2$ as a function of the number of transfer samples $k$ for all methods. Each curve represents one method, with error bars from 10 random selections of the $k$ transfer samples. The expected visual pattern is that our method (solid line) dominates or matches all baselines at every value of $k$, with the largest advantage at small $k$ (where the pretrained representation provides the most benefit).

### 5.5.2 Uncertainty Quantification (Experiment E6)

**Objective.** A trustworthy spectral identification system must not only predict molecular structures but also quantify its uncertainty. We evaluate whether conformal prediction [34] --- a distribution-free framework for constructing prediction sets with guaranteed coverage --- provides well-calibrated uncertainty estimates for spectral identification.

**Protocol.** The conformal prediction wrapper operates as follows:

1. **Calibration.** On a held-out calibration set of $n_{\text{cal}} = 1{,}000$ molecules with known identities, the model computes a nonconformity score for each molecule: $s_i = 1 - p(y_i | x_i)$, where $p(y_i | x_i)$ is the model's predicted probability of the correct molecule $y_i$ given spectrum $x_i$. The empirical quantile $\hat{q}_\alpha$ of these scores at level $\alpha$ is computed.

2. **Prediction.** For a new spectrum $x_{\text{test}}$, the prediction set is:

$$\mathcal{C}_\alpha(x_{\text{test}}) = \{y : p(y | x_{\text{test}}) \geq 1 - \hat{q}_\alpha\}.$$

By the exchangeability-based coverage guarantee of conformal prediction [34], this set satisfies:

$$\Pr\bigl(y_{\text{true}} \in \mathcal{C}_\alpha(x_{\text{test}})\bigr) \geq 1 - \alpha.$$

3. **Evaluation.** We evaluate at $\alpha = 0.10$ (90% target coverage) and report:
   - **Empirical coverage:** The fraction of test molecules whose true identity lies within $\mathcal{C}_{0.10}$.
   - **Average set size:** The mean number of molecules in $\mathcal{C}_{0.10}$.
   - **Conditional coverage by symmetry:** Empirical coverage stratified by point group, to assess whether the conformal guarantee holds uniformly across symmetry classes.

**Predicted outcomes.**

1. **Marginal coverage.** We expect the empirical coverage to be within the interval $[88\%, 92\%]$ at the $\alpha = 0.10$ level, consistent with the finite-sample guarantees of conformal prediction [34]. The slight deviation from exactly 90% reflects the discretization inherent in finite calibration sets.

2. **Prediction set size.** For well-identified molecules (low-symmetry, spectrally distinctive), the prediction set should be small (often a single molecule). For poorly identified molecules (high-symmetry, spectrally confusable), the prediction set should be larger, reflecting the model's genuine uncertainty. We predict that the mean prediction set size will be between 3 and 8 molecules, with a strong positive correlation between set size and the number of spectrally similar molecules in the database.

3. **Coverage by symmetry.** We predict that marginal coverage will hold across all symmetry classes (this is guaranteed by the theory), but that the prediction set size will be systematically larger for high-symmetry molecules (where $R(G, N)$ is low and more modes are silent). This would constitute additional indirect validation of Theorem 1: the model's calibrated uncertainty correctly reflects the information-theoretic limits imposed by symmetry.

**Comparison to MC Dropout.** As a baseline uncertainty method, we also evaluate Monte Carlo Dropout [35]: at inference, 50 forward passes are made with dropout active, and the variance across passes provides an uncertainty estimate. We predict that conformal prediction will provide substantially better calibration than MC Dropout ($\approx$90% coverage versus $\approx$78--82% coverage at the 90% target), because MC Dropout's coverage guarantees are asymptotic and depend on the correctness of the approximate posterior, whereas conformal prediction provides finite-sample, distribution-free guarantees.

**Key figure (Figure 9): Calibration plot.** Panel (a) plots the target coverage level (horizontal axis) against the empirical coverage (vertical axis) for both conformal prediction and MC Dropout, over the range $\alpha \in [0.05, 0.50]$. The conformal prediction curve is expected to track the diagonal $y = x$ closely, while the MC Dropout curve is expected to lie systematically below it (overconfident). Panel (b) shows a histogram of prediction set sizes at $\alpha = 0.10$, stratified by whether the molecule is centrosymmetric or non-centrosymmetric, illustrating the relationship between symmetry-induced information loss and calibrated uncertainty.

### 5.5.3 Connection to Identifiability Theory

The results of Experiments E5 and E6 connect to the theoretical framework of Section 3 in two ways:

1. **Calibration transfer as nuisance parameter elimination.** The VIB disentanglement $z = [z_{\text{chem}} | z_{\text{inst}}]$ is a practical implementation of the quotient by instrument symmetry: different instruments related by the instrument response group are mapped to the same $z_{\text{chem}}$. The transfer performance thus reflects the quality of the disentanglement, and the fact that it succeeds with few labeled samples is consistent with the low dimensionality of instrument variation relative to chemical variation.

2. **Uncertainty as a proxy for identifiability.** The conformal prediction set size provides a calibrated, instance-level measure of how identifiable a given molecule is from its spectrum. Large prediction sets indicate low identifiability; small prediction sets indicate high identifiability. The correlation between prediction set size and $R(G, N)$ would establish that the conformal prediction framework is correctly capturing the symmetry-induced identifiability limits characterized by Theorem 1.


## 5.6 Summary of Experimental Predictions

To facilitate evaluation of the experimental program, we summarize the key predictions arising from the theoretical framework:

**Table 5. Summary of testable predictions and their theoretical basis.**

| Prediction | Section | Theoretical Basis | Falsification Criterion |
|-----------|---------|-------------------|------------------------|
| Top-$k$ accuracy correlates positively with $R(G, N)$ | 5.1 | Theorem 1 | $\rho < 0.5$ between $R(G,N)$ and accuracy |
| $\Delta_{\text{comp}}$ is larger for centrosymmetric molecules | 5.2 | Theorem 2 | $\Delta_{\text{comp}}$ (centro.) $\leq$ $\Delta_{\text{comp}}$ (non-centro.) |
| Model accuracy degrades on confusable sets | 5.3 | Proposition 1 | Accuracy on confusable sets $\geq$ overall accuracy |
| Empirical error $\geq$ Fano bound on confusable sets | 5.3 | Proposition 1 | Empirical error systematically below Fano bound |
| Jacobian has full rank for generic $C_1$ molecules | 5.4 | Conjecture 3 | $>$5% of $C_1$ molecules have rank-deficient $\mathbf{J}$ |
| Rank deficiency correlates with non-trivial symmetry | 5.4 | Conjecture 3 | Rank deficiency equally common for $C_1$ and $D_{6h}$ |
| Conformal prediction achieves $\geq$88% coverage at $\alpha = 0.10$ | 5.5 | Conformal theory | Empirical coverage $<$ 85% |
| Prediction set size correlates with $1/R(G, N)$ | 5.5 | Theorems 1, 2 | No correlation between set size and symmetry |

Each prediction is accompanied by a falsification criterion: a specific outcome that, if observed, would constitute evidence against the theoretical prediction. This pre-registration of both predictions and falsification criteria ensures that the experimental evaluation is informative regardless of outcome.

---

## References

[22] J. Axelrod, R. Gomez-Bombarelli, "QM9S: Quantum mechanical simulations of vibrational spectra for organic molecules", in *Proceedings of the 39th International Conference on Machine Learning* (2022).

[23] T. D. Crawford, Z. W. Windom, "MolSym: A Python package for handling symmetry in molecular quantum chemistry", *Journal of Chemical Physics* **161**, 162502 (2024).

[24] J. Axelrod, R. Schwalbe-Koda, "QMe14S: Quantum mechanical spectra for molecules with 14 elements", preprint (2025).

[25] ChEMBL IR-Raman: Vibrational mode symmetry labels computed at the Gaussian09 level for 220K drug-like molecules from the ChEMBL database.

[26] R. A. A. Ince, B. L. Giordano, C. Kayser, G. A. Rousselet, J. Gross, P. G. Schyns, "A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula", *Human Brain Mapping* **38**, 1541--1573 (2017).

[27] M. J. Dyrby, S. B. Engelsen, L. Norgaard, M. Bruhn, L. Lundsberg-Nielsen, "Chemometric quantitation of the active substance in a pharmaceutical tablet using near-infrared (NIR) transmittance and NIR FT-Raman spectra", *Applied Spectroscopy* **56**, 579--585 (2002).

[28] "Pharmaceutical tablet NIR shootout 2002", International Diffuse Reflectance Conference (2002).

[29] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, "LoRA: Low-rank adaptation of large language models", in *Proceedings of the International Conference on Learning Representations* (2022).

[30] Y. Wang, D. J. Veltkamp, B. R. Kowalski, "Multivariate instrument standardization", *Analytical Chemistry* **63**, 2750--2756 (1991).

[31] T. B. Blank, S. T. Sum, S. D. Brown, S. L. Monfre, "Transfer of near-infrared multivariate calibrations without standards", *Analytical Chemistry* **68**, 2987--2995 (1996).

[32] R. Nikzad-Langerodi, W. Zellinger, E. Lughofer, S. Saminger-Platz, "Domain-invariant partial least squares regression", *Analytical Chemistry* **90**, 6693--6701 (2018).

[33] "Calibration transfer based on affine invariance for NIR without transfer standards", *Analytical Chemistry* (2019). See also: Low-rank adaptation approaches for spectroscopic calibration transfer (2024).

[34] A. N. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification", *Foundations and Trends in Machine Learning* **16**, 263--394 (2023).

[35] Y. Gal, Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning", in *Proceedings of the 33rd International Conference on Machine Learning*, 1050--1059 (2016).
