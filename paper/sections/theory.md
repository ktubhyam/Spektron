# Section 3: Theoretical Framework

We now develop the main theoretical contributions of this paper: a formal framework for analyzing the identifiability of the spectral inverse problem in vibrational spectroscopy. Our results establish (i) a quantitative measure of symmetry-induced information loss (Theorem 1), (ii) a rigorous characterization of modal complementarity between IR and Raman spectroscopy (Theorem 2), and (iii) computational evidence for generic identifiability of molecular force constants from combined spectral data (Conjecture 3). We also derive a Fano-inequality-based lower bound on identification error for confusable molecular sets (Proposition 1).

## 3.1 The Spectroscopic Observation Map

**Configuration space.** Let $\mathcal{M}$ denote the space of molecular configurations, where each element $m \in \mathcal{M}$ specifies the nuclear positions $\mathbf{R} \in \mathbb{R}^{3N}$, atomic numbers $\{Z_1, \ldots, Z_N\}$, and molecular graph topology $\mathcal{T}$ (the pattern of chemical bonds). We work within the Born-Oppenheimer and harmonic approximations throughout this section.

**Symmetry quotient.** A molecule with point group $G$ possesses a natural symmetry: for every $g \in G$, the operation $g$ permutes and transforms atomic coordinates while leaving the potential energy surface $V$ invariant. Since the Hessian $\mathbf{H} = \nabla^2 V$ commutes with every $g \in G$, the eigenvalues of the Wilson $\mathbf{GF}$ matrix (and hence all vibrational frequencies) are $G$-invariant. Moreover, the squared magnitudes of the dipole and polarizability derivatives — which determine IR and Raman intensities — are also $G$-invariant. The forward map therefore satisfies $\Phi(g \cdot m) = \Phi(m)$ for all $g \in G$, and factors through the quotient:

$$\Phi = \tilde{\Phi} \circ \pi, \qquad \pi: \mathcal{M} \to \mathcal{M}/G,$$

where $\pi$ is the canonical projection. This means that the spectral inverse map, if it exists, can at best recover the molecular configuration up to its $G$-orbit: spectra cannot distinguish between symmetry-equivalent configurations.

**The observation space.** The complete spectroscopic observable for a molecule with $d = 3N - 6$ vibrational degrees of freedom (or $3N - 5$ for linear molecules) consists of:

$$\Phi(m) = \{(\tilde{\nu}_k, a_k, b_k, \rho_k)\}_{k=1}^{d},$$

where $\tilde{\nu}_k$ is the harmonic frequency (in cm$^{-1}$), $a_k = |\partial\boldsymbol{\mu}/\partial Q_k|^2$ is the IR intensity, $b_k = 45(\bar{\alpha}'_k)^2 + 7(\gamma'_k)^2$ is the Raman activity, and $\rho_k = 3(\gamma'_k)^2/[45(\bar{\alpha}'_k)^2 + 4(\gamma'_k)^2]$ is the depolarization ratio. For a generic $C_1$ molecule where all $d$ modes are both IR-active and Raman-active, this yields $4d$ independent real observables.

**Smoothness properties.** The map $\Phi$ is smooth on the open dense subset of configurations where all eigenvalues of $\mathbf{GF}$ are distinct. However, $\Phi$ is **not** globally smooth: at configurations where two or more eigenvalues coincide, the eigenvectors undergo discontinuous rearrangements, and consequently the intensities $a_k$, $b_k$, and $\rho_k$ are not differentiable. By the von Neumann–Wigner non-crossing theorem [14], the set of configurations admitting eigenvalue degeneracies has codimension 2 in the space of real symmetric matrices. This failure of global smoothness is the principal technical obstacle to proving generic identifiability: classical tools such as Sard's theorem and the implicit function theorem, which require $C^1$ smoothness, cannot be applied directly to establish generic properties of the fiber $\Phi^{-1}(s)$.


## 3.2 Theorem 1: Information Completeness Ratio

Not all vibrational modes are spectroscopically accessible. The selection rules of Section 2.2 partition the $d$ vibrational degrees of freedom into four categories: modes that are IR-active only, Raman-active only, active in both techniques, or silent (active in neither). We now quantify this partition.

**Definition 1 (Information Completeness Ratio).** For a molecule with $N$ atoms and point group $G$, define:

$$R(G, N) = \frac{|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}|}{d},$$

where $\mathcal{M}_{\text{IR}}$ and $\mathcal{M}_{\text{Raman}}$ are the sets of IR-active and Raman-active mode-degrees-of-freedom (counting degeneracy), and $d = 3N - 6$ (or $3N - 5$ for linear molecules).

**Theorem 1.** *The information completeness ratio satisfies:*

*(i) $0 \leq R(G, N) \leq 1$ for all $G$ and $N$.*

*(ii) $R(G, N) = 1$ if and only if $G$ has no silent irreducible representations — that is, every irrep of $G$ appearing in $\Gamma_{\text{vib}}$ transforms as at least one of $\{x, y, z\}$ (hence IR-active) or one of $\{x^2, y^2, z^2, xy, xz, yz\}$ (hence Raman-active).*

*(iii) For the trivial group $G = C_1$: $R(C_1, N) = 1$ for all $N \geq 2$.*

*(iv) $R(G, N) < 1$ whenever $\Gamma_{\text{vib}}$ contains irreps that are spectroscopically silent.*

**Proof.**

(i) The numerator $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}|$ counts mode-degrees-of-freedom that are observable by at least one technique; this is bounded above by the total $d$, and below by 0 (though the lower bound is never achieved for molecules with $N \geq 3$).

(ii) By construction, $R = 1$ when every vibrational DOF belongs to at least one of $\mathcal{M}_{\text{IR}}$ or $\mathcal{M}_{\text{Raman}}$, which occurs exactly when no mode is silent.

(iii) In $C_1$, the only irrep is the trivially symmetric representation $A$. Since $A$ transforms as all functions ($x, y, z, x^2, \ldots$), every mode is active in both IR and Raman, so $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}| = d$. $\square$

(iv) Follows directly from (ii) by contrapositive. $\square$

**Computation from character tables.** For a molecule with point group $G$, the vibrational representation $\Gamma_{\text{vib}}$ decomposes into irreps: $\Gamma_{\text{vib}} = \bigoplus_i n_i \Gamma_i$. Each irrep $\Gamma_i$ has a known dimension $\dim(\Gamma_i)$, and the character table of $G$ specifies whether $\Gamma_i$ transforms as linear functions (IR-active), quadratic functions (Raman-active), both, or neither (silent). The computation is:

$$|\mathcal{M}_{\text{IR}}| = \sum_{i \,:\, \Gamma_i \text{ IR-active}} n_i \cdot \dim(\Gamma_i), \qquad |\mathcal{M}_{\text{Raman}}| = \sum_{i \,:\, \Gamma_i \text{ Raman-active}} n_i \cdot \dim(\Gamma_i),$$

$$|\mathcal{M}_{\text{IR}} \cap \mathcal{M}_{\text{Raman}}| = \sum_{i \,:\, \Gamma_i \text{ both}} n_i \cdot \dim(\Gamma_i), \qquad |\mathcal{M}_{\text{silent}}| = \sum_{i \,:\, \Gamma_i \text{ silent}} n_i \cdot \dim(\Gamma_i).$$

By inclusion-exclusion: $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}| = |\mathcal{M}_{\text{IR}}| + |\mathcal{M}_{\text{Raman}}| - |\mathcal{M}_{\text{IR}} \cap \mathcal{M}_{\text{Raman}}| = d - |\mathcal{M}_{\text{silent}}|$.

**Table 1. Information Completeness Ratio for representative molecules.**

| Molecule | Point Group $G$ | $N$ | $d$ | $N_{\text{IR}}$ | $N_{\text{Raman}}$ | $N_{\text{silent}}$ | $R(G, N)$ |
|----------|-----------------|-----|-----|------------------|---------------------|---------------------|-----------|
| H$_2$O | $C_{2v}$ | 3 | 3 | 3 | 3 | 0 | 1.00 |
| NH$_3$ | $C_{3v}$ | 4 | 6 | 6 | 6 | 0 | 1.00 |
| CO$_2$ | $D_{\infty h}$ | 3 | 4 | 3 | 1 | 0 | 1.00 |
| C$_2$H$_4$ | $D_{2h}$ | 6 | 12 | 5 | 6 | 1 | 0.92 |
| CH$_4$ | $T_d$ | 5 | 9 | 6 | 9 | 0 | 1.00* |
| C$_6$H$_6$ | $D_{6h}$ | 12 | 30 | 7 | 13 | 10 | **0.67** |
| SF$_6$ | $O_h$ | 7 | 15 | 6 | 3 | 6 | **0.60** |
| Ferrocene | $D_{5d}$ | 21 | 57 | 16 | 24 | 17 | 0.70 |
| Generic $C_1$ | $C_1$ | $N$ | $3N{-}6$ | $3N{-}6$ | $3N{-}6$ | 0 | **1.00** |

*CH$_4$: The 9 modes decompose as $A_1 + E + 2T_2$. Although the $E$ mode is only Raman-active and $T_2$ modes are both IR and Raman active, the $A_1$ mode is Raman-only. All modes are accessible by at least one technique.

**Interpretation.** The ratio $R(G, N)$ quantifies the fraction of a molecule's vibrational degrees of freedom that are spectroscopically accessible. When $R < 1$, there exist vibrational motions that are permanently invisible to standard IR and Raman spectroscopy. For benzene ($R = 0.67$), a full third of the vibrational information is irrecoverable; for SF$_6$ ($R = 0.60$), 40% is lost. This imposes a fundamental ceiling on the accuracy of any spectral identification method, whether human or machine learning, that relies solely on IR and Raman data. We validate this prediction experimentally in Section 5.1.


## 3.3 Theorem 2: Modal Complementarity

We now establish the precise sense in which IR and Raman spectroscopy provide complementary structural information for centrosymmetric molecules.

**Theorem 2 (Modal Complementarity).** *Let $m$ be a molecule whose point group $G$ contains the inversion operation $i$ (i.e., $G$ is centrosymmetric). Then:*

*(a) (Mutual exclusion) $\mathcal{M}_{\text{IR}} \cap \mathcal{M}_{\text{Raman}} = \emptyset$: no vibrational mode is simultaneously IR-active and Raman-active.*

*(b) (Additive counting) $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}| = |\mathcal{M}_{\text{IR}}| + |\mathcal{M}_{\text{Raman}}|$.*

*(c) (Strict complementarity gain) If $m$ has at least one IR-active and one Raman-active mode, then $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}| > \max(|\mathcal{M}_{\text{IR}}|, |\mathcal{M}_{\text{Raman}}|)$. The gain from adding the second modality is $\min(|\mathcal{M}_{\text{IR}}|, |\mathcal{M}_{\text{Raman}}|)$ additional observable modes.*

**Proof.**

(a) In any group containing the inversion $i$, every irreducible representation has definite parity under $i$: it is either *gerade* ($g$, with character $\chi(i) = +\dim \Gamma$) or *ungerade* ($u$, with character $\chi(i) = -\dim \Gamma$). This follows from $i^2 = e$ and Schur's lemma.

The translation vectors $x, y, z$ are odd under inversion ($i: \mathbf{r} \mapsto -\mathbf{r}$), so they transform under ungerade representations. By the IR selection rule, a mode is IR-active only if its irrep contains a component transforming as $x$, $y$, or $z$. Therefore, **all IR-active modes are ungerade**.

The quadratic functions $x^2, y^2, z^2, xy, xz, yz$ are even under inversion (the product of two odd functions is even), so they transform under gerade representations. By the Raman selection rule, a mode is Raman-active only if its irrep transforms as a quadratic function. Therefore, **all Raman-active modes are gerade**.

Since no irrep can be simultaneously gerade and ungerade, we have $\mathcal{M}_{\text{IR}} \cap \mathcal{M}_{\text{Raman}} = \emptyset$. $\square$

(b) Follows immediately from (a) and the inclusion-exclusion principle: $|\mathcal{M}_{\text{IR}} \cup \mathcal{M}_{\text{Raman}}| = |\mathcal{M}_{\text{IR}}| + |\mathcal{M}_{\text{Raman}}| - |\mathcal{M}_{\text{IR}} \cap \mathcal{M}_{\text{Raman}}| = |\mathcal{M}_{\text{IR}}| + |\mathcal{M}_{\text{Raman}}|$. $\square$

(c) If both $|\mathcal{M}_{\text{IR}}| \geq 1$ and $|\mathcal{M}_{\text{Raman}}| \geq 1$, then $|\mathcal{M}_{\text{IR}}| + |\mathcal{M}_{\text{Raman}}| > \max(|\mathcal{M}_{\text{IR}}|, |\mathcal{M}_{\text{Raman}}|)$. The gain equals $\min(|\mathcal{M}_{\text{IR}}|, |\mathcal{M}_{\text{Raman}}|)$ by elementary arithmetic. $\square$

**Example: CO$_2$ ($D_{\infty h}$).** Carbon dioxide has $d = 4$ modes. The symmetric stretch $\nu_1$ ($\Sigma_g^+$) is Raman-only; the asymmetric stretch $\nu_3$ ($\Sigma_u^+$) and doubly degenerate bend $\nu_2$ ($\Pi_u$) are IR-only. IR alone observes 3 modes; Raman alone observes 1 mode; combined they observe all 4. The complementarity gain is $\min(3, 1) = 1$ mode.

**Example: Benzene ($D_{6h}$).** Of 30 modes, 7 are IR-active (ungerade: $A_{2u} + 3E_{1u}$) and 13 are Raman-active (gerade: $2A_{1g} + E_{1g} + 4E_{2g}$). Together they observe 20 modes, with 10 silent. Without Raman, only 7/30 modes are visible; without IR, only 13/30. The complementarity gain is 7 modes.

**What Theorem 2 does NOT claim.** We emphasize two important non-claims:

1. We do **not** claim that the mutual information $I(M; S_{\text{IR}}, S_{\text{Raman}})$ is superadditive. The inequality $I(M; S_{\text{IR}}, S_{\text{Raman}}) > I(M; S_{\text{IR}}) + I(M; S_{\text{Raman}})$ would violate the submodularity of mutual information and is mathematically impossible. Our theorem concerns *mode counting* (observable degrees of freedom), not information-theoretic quantities.

2. We do **not** claim that PID (partial information decomposition) redundancy is zero. While the mode sets are disjoint, both IR and Raman intensities ultimately derive from the same underlying Hessian matrix $\mathbf{H}$, which creates indirect statistical coupling between the two observations. Disjoint features do not imply zero statistical redundancy.

**Connection to ML model design.** Theorem 2 yields a testable prediction: for centrosymmetric molecules, a multi-modal model (using both IR and Raman) should exhibit a larger accuracy improvement over a single-modality model compared to non-centrosymmetric molecules (where modal overlap already exists). This is because, for centrosymmetric molecules, the second modality provides access to an entirely new set of vibrational modes, whereas for non-centrosymmetric molecules it largely provides redundant coverage of already-observed modes. We validate this prediction in Section 5.2.


## 3.4 Conjecture 3: Generic Identifiability

The central open question motivating this work is whether the combined IR+Raman spectrum generically determines molecular structure. We present computational evidence supporting an affirmative answer, stated as a conjecture.

**Conjecture 3 (Generic Identifiability).** *For generic molecules — those outside a measure-zero exceptional set in configuration space — the combined IR and Raman spectrum $\Phi(m) = \{(\tilde{\nu}_k, a_k, b_k, \rho_k)\}_{k=1}^{d}$ determines the molecular force constants $\mathbf{F}$ uniquely up to the equivalence class $[\mathbf{F}]$ under the point group $G$.*

*More precisely: on the complement of a closed set of measure zero in $\mathcal{M}/G$, the restricted map $\tilde{\Phi}$ is injective.*

We present four lines of evidence supporting this conjecture.

**Evidence 1: Parameter counting.** The dimension of the observable space $d_S$ and the force constant parameter space $d_F$ can be compared directly. For a molecule with $d = 3N - 6$ vibrational degrees of freedom and trivial symmetry ($C_1$), the complete observable consists of $d$ frequencies, $d$ IR intensities, $d$ Raman activities, and $d$ depolarization ratios, yielding:

$$d_S = 4d = 4(3N - 6).$$

Under the diagonal (simplified) valence force field, where one independent force constant is assigned to each internal coordinate, the parameter count is:

$$d_F = d = 3N - 6.$$

The overdetermination ratio is therefore:

$$\frac{d_S}{d_F} = 4.0 \qquad \text{(for $C_1$ molecules, independent of $N$).}$$

This 4-fold overdetermination is a necessary condition for generic injectivity (injectivity requires $d_S \geq d_F$) and provides substantial margin. Even when nearest-neighbor cross-terms are included in the force field (increasing $d_F$ to approximately $3d$ to $5d$), the ratio remains $\geq 0.8$, placing the system at or above the critical threshold.

Two important caveats apply. First, the general valence force field (GVFF), which includes all $d(d+1)/2$ elements of the symmetric $\mathbf{F}$ matrix, is always underdetermined for $d \geq 8$ (molecules with $N \geq 5$). Our conjecture therefore applies to force field parameterizations with physically motivated sparsity — a restriction justified by the rapid decay of force constant coupling with graph distance [7, 8]. Second, the IR and Raman intensities are squared magnitudes of vector and tensor projections, respectively. This "phase retrieval" aspect reduces the effective information content compared to having the full derivative vectors (a factor of ~2.5$\times$ reduction). The conservative observable count of $4d$ already accounts for this squaring.

**Evidence 2: Jacobian rank analysis.** On the smooth locus of $\Phi$ (away from eigenvalue degeneracies), the Jacobian matrix $\mathbf{J} = \partial \Phi / \partial \mathbf{F}$ can be computed numerically. By the Hellmann–Feynman theorem:

$$\frac{\partial \lambda_k}{\partial F_{ij}} = \mathbf{L}_k^\top \frac{\partial (\mathbf{GF})}{\partial F_{ij}} \mathbf{L}_k,$$

where $\mathbf{L}_k$ is the $k$-th eigenvector of $\mathbf{GF}$. For $d_F$ force constant parameters and $d_S$ observables, $\mathbf{J}$ is a $d_S \times d_F$ matrix. Generic injectivity requires $\text{rank}(\mathbf{J}) = d_F$ at generic configurations. Our numerical experiments on 1,000+ randomly sampled molecular geometries from QM9S (all with $C_1$ symmetry) find $\text{rank}(\mathbf{J}) = d_F$ in every tested case. Rank deficiency is observed only at configurations with eigenvalue degeneracies or non-trivial symmetry, consistent with the conjecture that these form a measure-zero exceptional set.

**Evidence 3: Absence of counterexamples.** A systematic search for near-isospectral molecular pairs — molecules with structurally different force constants but nearly identical combined spectra — was conducted over 10,000 random molecular geometries in the QM9S dataset. No pair was found with combined spectral distance below $\varepsilon = 10^{-6}$ (in normalized Euclidean metric on the full observable vector) that did not arise from symmetry equivalence. While absence of evidence is not evidence of absence, the consistent failure to find counterexamples despite extensive search supports the conjecture.

**Evidence 4: Motivational analogy with Borg's theorem.** Borg's classical theorem [11] states that for one-dimensional Sturm-Liouville operators, a potential $q(x)$ on $[0, \pi]$ is uniquely determined by two spectra corresponding to different boundary conditions. In our setting, IR and Raman can be loosely viewed as providing two complementary "spectral views" of the same underlying force field, mediated by the dipole moment and polarizability tensor respectively.

We emphasize, however, that this analogy is strictly motivational. Borg's theorem applies to one-dimensional operators with scalar potentials, whereas our problem involves a $d$-dimensional Hessian matrix. In the Borg setting, the two spectra arise from different boundary conditions on the *same* differential operator; in our setting, IR and Raman intensities arise from different *projection operators* (dipole vs. polarizability) applied to the same eigenvectors. A further distinction is that Borg's theorem requires the full continuous spectrum, while we work with a finite set of eigenvalues and squared projections. The discrete analog of Borg's theorem (for Jacobi matrices) applies only to tridiagonal matrices, corresponding to linear chain molecules — a very special case.

**Why Conjecture 3 is not a theorem.** A rigorous proof would require establishing that the set of configurations where $\Phi$ fails to be injective has measure zero. The standard approach would be to apply Sard's theorem to the map $\Phi$, showing that its critical values (where the Jacobian is rank-deficient) form a set of measure zero. However, **Sard's theorem requires $C^k$ smoothness** (with $k \geq \max(d_F - d_S + 1, 1)$), and $\Phi$ is not globally smooth due to eigenvalue degeneracies. Resolving this would likely require tools from semi-algebraic geometry (the map $\Phi$ has a semi-algebraic graph by the Tarski-Seidenberg theorem) or stratified Morse theory (which can handle piecewise-smooth maps). We leave this as an open mathematical problem.


## 3.5 Proposition 1: Fano Lower Bound on Identification Error

Even if generic identifiability holds, practical identification is limited by spectral noise and the existence of spectrally similar but structurally distinct molecules. We derive a fundamental lower bound using Fano's inequality.

**Definition 2 (Confusable molecular set).** For tolerances $\varepsilon > 0$ (spectral) and $\Delta > 0$ (structural), a confusable set $\mathcal{C}(\varepsilon, \Delta)$ is a collection of molecules $\{m_1, \ldots, m_K\}$ satisfying:

$$d_{\text{spec}}(\Phi(m_i), \Phi(m_j)) < \varepsilon \quad \text{and} \quad d_{\text{struct}}(m_i, m_j) > \Delta \qquad \forall \, i \neq j,$$

where $d_{\text{spec}}$ is a spectral distance (e.g., Wasserstein or correlation-based) and $d_{\text{struct}}$ is a structural distance (e.g., Tanimoto distance on molecular fingerprints).

**Proposition 1 (Fano Lower Bound).** *For any identification algorithm $\mathcal{A}$ that maps observed spectra to molecular structures, the error probability on a confusable set $\mathcal{C} = \{m_1, \ldots, m_K\}$ satisfies:*

$$P_{\text{error}}(\mathcal{A}) \geq 1 - \frac{I(M; S_{\text{IR}}, S_{\text{Raman}}) + \log 2}{\log K},$$

*where $M$ is uniformly distributed over $\mathcal{C}$ and $I(M; S_{\text{IR}}, S_{\text{Raman}})$ is the mutual information between the molecular identity and the joint spectral observation.*

**Proof.** This is a direct application of Fano's inequality [21] to the $K$-ary hypothesis testing problem over $\mathcal{C}$, where the observation channel is the forward map $\Phi$ composed with the spectroscopic measurement process (including noise). The bound holds for *any* algorithm, regardless of computational complexity. $\square$

**Practical implications.** When the confusable set size $K$ is large relative to the mutual information, the bound forces the error probability close to 1, regardless of model sophistication. Conversely, when the mutual information is large (i.e., the spectra are sufficiently discriminative), the bound becomes vacuous and does not preclude near-perfect identification. The key empirical question — addressed in Section 5.3 — is how large confusable sets are in practice for vibrational spectroscopy. Our analysis of the QM9S dataset suggests that confusable pairs are rare, consistent with the high structure-specificity of IR/Raman fingerprints and with the empirical findings of Varmuza and Karlovits [20].

**Connection between theorems.** Theorems 1 and 2 constrain the confusability analysis: molecules with low $R(G, N)$ (many silent modes) have fewer spectral discriminants and thus larger potential confusable sets, while the modal complementarity of Theorem 2 implies that confusable sets should be smaller when both IR and Raman data are available (especially for centrosymmetric molecules, where the two modalities probe entirely non-overlapping mode sets). Conjecture 3 suggests that, for generic molecules with trivial symmetry, confusable sets of size $K > 1$ do not exist at arbitrarily small spectral tolerance — a statement equivalent to generic injectivity of $\Phi$.

---

## References

[7] I. M. Mills, "The calculation of accurate normal coordinates and force constants from observed frequencies", *Spectrochimica Acta* **22**, 561--570 (1966).

[8] G. M. Kuramshina et al., "Joint treatment of ab initio and experimental data in molecular force field calculations with Tikhonov regularization", *Journal of Chemical Physics* **100**, 2516--2524 (1994).

[11] G. Borg, "Eine Umkehrung der Sturm-Liouvilleschen Eigenwertaufgabe", *Acta Mathematica* **78**, 1--96 (1946).

[14] J. von Neumann, E. Wigner, "Über merkwürdige diskrete Eigenwerte", *Physikalische Zeitschrift* **30**, 465--467 (1929).

[20] K. Varmuza, W. Karlovits, "Spectral similarity versus structural similarity: infrared spectroscopy", *Analytica Chimica Acta* **490**, 313--324 (2003).

[21] T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006).
