# Section 2: Background

## 2.1 The Forward Vibrational Map (Wilson GF Method)

We begin by defining the forward map $\Phi$ from molecular structure to vibrational spectrum, which constitutes the physical process that any inverse spectroscopic method must implicitly or explicitly invert. The formalism we present is the Wilson GF matrix method [12], which remains the standard framework for vibrational analysis in molecular spectroscopy after more than seven decades of use.

Consider a molecule with $N$ atoms, atomic masses $\{m_1, \ldots, m_N\}$, and equilibrium geometry specified by Cartesian coordinates $\mathbf{R}_0 \in \mathbb{R}^{3N}$. Within the Born-Oppenheimer approximation, the potential energy surface $V(\mathbf{R})$ is a function of nuclear coordinates alone. At a stationary point $\mathbf{R}_0$ (where $\nabla V = 0$), expanding $V$ to second order yields the harmonic approximation:

$$V(\mathbf{R}) \approx V(\mathbf{R}_0) + \frac{1}{2} (\mathbf{R} - \mathbf{R}_0)^\top \mathbf{H} (\mathbf{R} - \mathbf{R}_0),$$

where $\mathbf{H} = \nabla^2 V \big|_{\mathbf{R}_0}$ is the $3N \times 3N$ Hessian matrix of Cartesian force constants.

It is often more natural to work in internal coordinates --- bond lengths, bond angles, and dihedral angles --- which eliminate the six (or five, for linear molecules) trivial degrees of freedom corresponding to translation and rotation. Let $\mathbf{s} \in \mathbb{R}^d$ denote a complete set of $d = 3N - 6$ internal coordinates (or $3N - 5$ for linear molecules). The relationship between infinitesimal internal and Cartesian displacements is given by the Wilson $\mathbf{B}$ matrix [12]:

$$\delta \mathbf{s} = \mathbf{B} \, \delta \mathbf{R}, \qquad B_{ij} = \frac{\partial s_i}{\partial R_j}.$$

The kinetic energy in internal coordinates involves the $\mathbf{G}$ matrix, defined as

$$\mathbf{G} = \mathbf{B} \mathbf{M}^{-1} \mathbf{B}^\top,$$

where $\mathbf{M} = \operatorname{diag}(m_1, m_1, m_1, m_2, \ldots, m_N)$ is the $3N \times 3N$ diagonal mass matrix. The potential energy in internal coordinates is characterized by the force constant matrix $\mathbf{F}$, with entries $F_{ij} = \partial^2 V / \partial s_i \, \partial s_j$. The classical equations of motion for small vibrations then lead to the Wilson secular equation:

$$|\mathbf{G}\mathbf{F} - \lambda \mathbf{I}| = 0,$$

whose $d$ eigenvalues $\{\lambda_1, \ldots, \lambda_d\}$ are related to the harmonic vibrational frequencies by $\lambda_k = 4\pi^2 \nu_k^2$. Equivalently, in wavenumber units $\tilde{\nu}_k = \nu_k / c$, which are standard in vibrational spectroscopy. The corresponding eigenvectors $\{\mathbf{L}_1, \ldots, \mathbf{L}_d\}$ define the normal mode displacement patterns --- the collective atomic motions associated with each vibrational frequency.

The vibrational frequencies alone do not constitute the full spectroscopic observable. Each normal mode $Q_k$ has an associated IR intensity and Raman activity, determined respectively by how the mode modulates the molecular dipole moment $\boldsymbol{\mu}$ and the polarizability tensor $\boldsymbol{\alpha}$:

$$a_k = \left| \frac{\partial \boldsymbol{\mu}}{\partial Q_k} \right|^2 \qquad \text{(IR intensity)},$$

$$b_k = \left| \frac{\partial \boldsymbol{\alpha}}{\partial Q_k} \right|^2 \qquad \text{(Raman activity)}.$$

More precisely, the IR intensity involves the square of the transition dipole moment derivative (a vector quantity), while the Raman activity involves derivatives of the polarizability tensor (a rank-2 tensor), with the precise form depending on the scattering geometry and polarization conditions [13]. A mode is IR-active if $a_k > 0$ (i.e., the vibration modulates the dipole moment) and Raman-active if $b_k > 0$ (i.e., the vibration modulates the polarizability). We return to the group-theoretic conditions governing these activities in Section 2.2.

The complete forward map can now be written as

$$\Phi: (\mathbf{R}_0, \{Z_i\}, \mathcal{T}) \;\longmapsto\; \{(\tilde{\nu}_k, a_k, b_k)\}_{k=1}^{d},$$

where $\{Z_i\}$ are atomic numbers and $\mathcal{T}$ denotes the molecular graph topology. This map has several important properties that bear on the inverse problem:

1. **$G$-equivariance.** If $G$ denotes the molecular point group (the group of spatial symmetry operations that leave the nuclear framework invariant), then $\Phi$ is $G$-invariant: $\Phi(g \cdot \mathbf{R}_0) = \Phi(\mathbf{R}_0)$ for all $g \in G$. This is because the Hessian commutes with every symmetry operation $g \in G$ (since $V$ is $G$-invariant), and consequently the eigenvalues and their multiplicities are $G$-invariant. The forward map therefore factors through the quotient $\Phi = \tilde{\Phi} \circ \pi$, where $\pi: \mathcal{M} \to \mathcal{M}/G$ is the canonical projection.

2. **Semi-algebraic structure.** The map $\Phi$ is a composition of polynomial operations (construction of $\mathbf{G}$ and $\mathbf{F}$ from coordinates) and radical operations (square roots of eigenvalues). By the Tarski-Seidenberg theorem, the image of $\Phi$ is a semi-algebraic set.

3. **Piecewise smoothness.** The map $\Phi$ is smooth on the open dense subset of configurations where all eigenvalues of $\mathbf{GF}$ are distinct. However, $\Phi$ is **not** globally smooth: at configurations where two or more eigenvalues coincide (eigenvalue degeneracies), the eigenvectors --- and hence the intensities $a_k$ and $b_k$ --- undergo discontinuous rearrangements. By the von Neumann-Wigner non-crossing theorem [14], the set of configurations admitting eigenvalue degeneracies has codimension 2 in the space of real symmetric matrices. This failure of global smoothness has a direct consequence for the inverse problem: classical tools such as Sard's theorem, which require the forward map to be smooth, cannot be applied directly to establish generic properties of the fiber $\Phi^{-1}(s)$.

4. **Generically full-rank Jacobian.** On the smooth locus (away from degeneracies), the Jacobian of $\Phi$ with respect to the force constants $F_{ij}$ can be computed via the Hellmann-Feynman theorem: $\partial \lambda_k / \partial F_{ij} = \mathbf{L}_k^\top (\partial(\mathbf{GF})/\partial F_{ij}) \mathbf{L}_k$, which is generically nonzero. This suggests --- but does not prove --- that the forward map is generically a local diffeomorphism onto its image, a point we return to in our discussion of identifiability (Section 3).


## 2.2 Selection Rules from Representation Theory

The question of which vibrational modes are observable via IR spectroscopy, Raman spectroscopy, both, or neither is answered completely by the representation theory of the molecular point group $G$. This classical framework, developed by Wigner, Herzberg, and others [13, 15], provides the mathematical foundation for our information-theoretic analysis.

**Vibrational representation decomposition.** The $3N$ Cartesian displacement coordinates of a molecule transform under the point group $G$ according to a (generally reducible) representation $\Gamma_\text{total}$. This representation decomposes into irreducible representations (irreps) of $G$:

$$\Gamma_\text{total} = \bigoplus_i n_i \, \Gamma_i,$$

where $n_i$ is the multiplicity of the irrep $\Gamma_i$. Six of these degrees of freedom correspond to rigid-body motions: three translations transforming as $(x, y, z)$ and three rotations transforming as $(R_x, R_y, R_z)$. The vibrational representation is the remainder:

$$\Gamma_\text{vib} = \Gamma_\text{total} - \Gamma_\text{trans} - \Gamma_\text{rot}.$$

Each irrep $\Gamma_i$ appearing in $\Gamma_\text{vib}$ corresponds to one or more vibrational normal modes. Modes belonging to the same irrep share identical symmetry properties, and modes belonging to degenerate irreps (two-dimensional $E$, three-dimensional $T$, etc.) are constrained by symmetry to have exactly the same frequency within the harmonic approximation.

**IR selection rule.** A vibrational mode is IR-active if and only if the transition integral $\langle v = 0 | \boldsymbol{\mu} | v = 1 \rangle$ is nonzero for at least one component of the dipole moment vector. In group-theoretic terms, this requires that the irrep of the normal mode contains (upon reduction) a component that also appears in the representation of the translation vectors $(x, y, z)$. Equivalently, a mode belonging to irrep $\Gamma_i$ is IR-active if and only if $\Gamma_i$ transforms as $x$, $y$, or $z$ under the operations of $G$ [13].

**Raman selection rule.** A vibrational mode is Raman-active if and only if the vibration modulates at least one component of the polarizability tensor $\boldsymbol{\alpha}$. Since the polarizability is a symmetric rank-2 tensor, its components transform as quadratic functions of the coordinates: $x^2$, $y^2$, $z^2$, $xy$, $xz$, $yz$ (and their linear combinations, such as $x^2 - y^2$). A mode belonging to irrep $\Gamma_i$ is Raman-active if and only if $\Gamma_i$ transforms as one of these quadratic basis functions [13].

**Mutual exclusion rule.** For centrosymmetric molecules --- those whose point group $G$ contains the inversion operation $i$ --- the selection rules exhibit a striking dichotomy. Every irrep of such a group has definite parity: *gerade* ($g$, symmetric under inversion) or *ungerade* ($u$, antisymmetric under inversion). Since the translation vectors $x, y, z$ are odd under inversion, all IR-active modes must be ungerade. Since the quadratic functions $x^2, xy, \ldots$ are even under inversion, all Raman-active modes must be gerade. Because an irrep cannot be simultaneously gerade and ungerade, we arrive at the mutual exclusion rule [13, 15]:

> **Mutual Exclusion Rule.** *In a centrosymmetric molecule, no vibrational mode can be simultaneously IR-active and Raman-active.*

The centrosymmetric point groups include $C_i$, $C_{2h}$, $D_{2h}$, $D_{3d}$, $D_{4h}$, $D_{6h}$, $D_{\infty h}$, $O_h$, and others. For molecules belonging to these groups, IR and Raman spectroscopy probe strictly complementary subsets of the vibrational degrees of freedom.

**Silent modes.** Not all vibrational modes are accessible to either technique. A mode is *spectroscopically silent* if its irrep transforms as neither a linear nor a quadratic function of the coordinates --- that is, if it is neither IR-active nor Raman-active. Silent modes represent vibrational degrees of freedom that are permanently inaccessible to conventional IR and Raman spectroscopy, constituting an irreducible source of information loss in the inverse problem. (Such modes can sometimes be observed by other techniques, such as neutron inelastic scattering or hyper-Raman spectroscopy, but these are not part of the standard analytical toolkit.)

These concepts are best illustrated through concrete molecular examples that span a range of symmetries:

**Example 1: Water ($\text{H}_2\text{O}$, point group $C_{2v}$, $N = 3$, $d = 3$).** The vibrational representation decomposes as $\Gamma_\text{vib} = 2A_1 + B_1$, yielding three normal modes: the symmetric stretch $\nu_1$ ($A_1$, 3657 cm$^{-1}$), the bending mode $\nu_2$ ($A_1$, 1595 cm$^{-1}$), and the asymmetric stretch $\nu_3$ ($B_1$, 3756 cm$^{-1}$). In $C_{2v}$, the irreps $A_1$ and $B_1$ both transform as linear *and* quadratic functions: $A_1$ contains $z$ (hence IR-active) and $x^2, y^2, z^2$ (hence Raman-active); $B_1$ contains $x$ (IR-active) and $xz$ (Raman-active). Consequently, all three modes are active in both IR and Raman. There are no silent modes, and the information completeness ratio is $R(C_{2v}, 3) = (3 + 3)/3 = 1.0$ (after accounting for the overlap of doubly-counted modes that are active in both techniques, $R$ is defined as the fraction of modes observable by at least one technique, which here is $3/3 = 1.0$). Water illustrates the generic situation for low-symmetry molecules: all vibrational information is spectroscopically accessible.

**Example 2: Carbon dioxide ($\text{CO}_2$, point group $D_{\infty h}$, $N = 3$, $d = 4$).** As a linear molecule, $\text{CO}_2$ has $3(3) - 5 = 4$ vibrational degrees of freedom. The vibrational representation decomposes as $\Gamma_\text{vib} = \Sigma_g^+ + \Sigma_u^+ + \Pi_u$, where $\Pi_u$ is doubly degenerate. The symmetric stretch $\nu_1$ ($\Sigma_g^+$, 1388 cm$^{-1}$) is Raman-active but IR-inactive (the symmetric stretch of a centrosymmetric molecule does not change the dipole moment). The asymmetric stretch $\nu_3$ ($\Sigma_u^+$, 2349 cm$^{-1}$) is IR-active but Raman-inactive. The doubly degenerate bending mode $\nu_2$ ($\Pi_u$, 667 cm$^{-1}$) is IR-active but Raman-inactive. Mutual exclusion holds perfectly: the IR and Raman spectra are completely non-overlapping, yet together they account for all four vibrational degrees of freedom. There are no silent modes, and $R(D_{\infty h}, 3) = 1.0$. Carbon dioxide illustrates how mutual exclusion forces strict complementarity between the two techniques while preserving full information coverage.

**Example 3: Benzene ($\text{C}_6\text{H}_6$, point group $D_{6h}$, $N = 12$, $d = 30$).** Benzene is the paradigmatic example of symmetry-induced information loss. Of its 30 vibrational modes, 7 are IR-active (belonging to irreps $A_{2u}$ and $E_{1u}$, totaling 7 mode-degrees-of-freedom), and 13 are Raman-active (belonging to irreps $A_{1g}$, $E_{1g}$, and $E_{2g}$, totaling 13 mode-degrees-of-freedom). Mutual exclusion holds strictly: no mode is active in both techniques. The remaining 10 modes, belonging to irreps $B_{1u}$, $B_{2g}$, $B_{2u}$, and $E_{2u}$, are silent --- they transform as neither linear nor quadratic functions of the coordinates. (For instance, $E_{2u}$ modes involve out-of-plane ring deformations that modulate neither the dipole moment nor the polarizability.) The information completeness ratio is $R(D_{6h}, 12) = (7 + 13)/30 = 20/30 \approx 0.67$. A full third of benzene's vibrational degrees of freedom are spectroscopically invisible. This represents a fundamental, symmetry-imposed ceiling on the amount of structural information extractable from IR and Raman data for highly symmetric molecules.

These three examples reveal a systematic pattern: as molecular symmetry increases, the fraction of observable vibrational information generally decreases, due both to the emergence of silent modes and to degeneracy collapsing multiple independent modes onto a single frequency. We formalize this pattern as Theorem 1 (Information Completeness Ratio) in Section 3.


## 2.3 Related Work

The question of what can be determined about a molecular structure from its vibrational spectrum sits at the intersection of several classical and modern research programs. We organize the relevant prior work into five categories.

**The classical inverse vibrational problem.** The mathematical study of force constant recovery from spectroscopic data has a long history. Crawford and Fletcher [16] provided the first systematic analysis of the inverse vibrational problem, establishing the basic counting arguments for when a set of observed frequencies might suffice to determine molecular force constants. Mills [7] demonstrated in 1966 that vibrational frequencies alone are generally insufficient for unique determination of force constants --- the system of equations relating eigenvalues of $\mathbf{GF}$ to the entries of $\mathbf{F}$ is typically underdetermined when only frequency data are available. This motivated extensive work on incorporating additional constraints. Kuramshina, Weinhold, Kochikov, Yagola, and Pentin [8] introduced Tikhonov regularization to the ill-posed inverse problem, showing that joint treatment of *ab initio* and experimental data could yield stable force field solutions even when the purely experimental inverse problem is underdetermined. Kuramshina [17] subsequently developed a comprehensive framework for inverse problems in vibrational spectroscopy, establishing conditions under which unique recovery is possible when isotopic frequency shifts and Coriolis coupling constants supplement the fundamental frequencies. A key distinction between this classical program and the present work is that the classical approach seeks to recover the *force constant matrix* $\mathbf{F}$ from a known molecular topology, whereas we study the identifiability of the *molecular structure itself* (topology and geometry) from the complete spectroscopic observable $\{(\tilde{\nu}_k, a_k, b_k)\}$, including intensities.

**Machine learning for spectrum-to-structure prediction.** A recent wave of machine learning models has attacked the forward inverse problem --- predicting molecular structure directly from computed or experimental spectra --- with increasingly impressive results. DiffSpectra [4] introduced a diffusion-based generative model that reconstructs molecular graphs from simulated IR and Raman spectra, achieving 40.76% top-1 accuracy on a benchmark of over 100,000 molecules from the QM9S dataset. Vib2Mol [5] proposed a multi-task encoder-decoder framework combining spectral retrieval with molecular property prediction, reaching approximately 87% top-10 accuracy on experimental NIST spectra. VibraCLIP [6] applied contrastive learning between spectral and molecular representations, achieving 81.7% top-1 accuracy when molecular mass is provided as an auxiliary input, demonstrating that even modest side information can substantially improve identification. Most recently, SpectrumWorld [18] introduced SpectrumBench, a large-scale benchmark encompassing 1.2 million substances with paired spectroscopic data across multiple modalities, providing the first standardized evaluation framework for spectrum-to-structure methods at scale. MolSpectLLM [19] bridged vibrational spectroscopy with three-dimensional molecular structure prediction by leveraging large language model architectures, demonstrating that joint modeling of spectral and geometric information improves performance on both forward and inverse tasks. While these results represent genuine and substantial progress, they share a common limitation: they are purely empirical, and cannot distinguish between accuracy ceilings that arise from algorithmic limitations (and are thus improvable) and those that arise from fundamental physical constraints (and are thus insurmountable). The theoretical framework we develop in Section 3 addresses precisely this gap.

**Kac's question and its analogs.** Our work draws direct inspiration from Kac's celebrated 1966 question, "Can one hear the shape of a drum?" [1], which asked whether the eigenvalue spectrum of the Laplacian on a bounded planar domain uniquely determines the domain geometry. Gordon, Webb, and Wolpert [2] answered this question in the negative in 1992, constructing pairs of non-isometric planar domains with identical Laplacian spectra. Their construction exploits a group-theoretic technique (Sunada's method) that transplants eigenfunctions between domains related by a specific algebraic structure. Most recently, Wang and Torquato [3] posed the crystallographic analog, "Can one hear the shape of a crystal?", investigating whether diffraction patterns (Fourier moduli of atomic arrangements) uniquely determine crystal structures --- a question with deep connections to the phase retrieval problem in X-ray crystallography. Our molecular question differs from both the drum problem and the crystal problem in a crucial respect: we have access not only to eigenvalues (frequencies) but also to eigenvector projections (intensities), through both the dipole ($\partial \boldsymbol{\mu} / \partial Q_k$) and polarizability ($\partial \boldsymbol{\alpha} / \partial Q_k$) channels. This additional information fundamentally changes the character of the inverse problem, and is one reason we conjecture that generic identifiability may hold for molecules even though it fails for drums.

**Inverse problems, identifiability, and symmetry.** The role of symmetry in constraining the solutions of inverse problems has been studied in several contexts. Borg [11] proved in 1946 that for one-dimensional Sturm-Liouville operators, a potential is uniquely determined by two spectra corresponding to different boundary conditions --- a foundational result in inverse spectral theory. We note that while Borg's theorem provides a suggestive analogy (IR and Raman as two complementary "spectral views" of the same underlying force field), the analogy is strictly motivational: the one-dimensional Sturm-Liouville problem is mathematically quite different from the $d$-dimensional Hessian eigenvalue problem arising in molecular vibrations, and no direct extension of Borg's theorem applies in our setting. Chen and Davies [10] studied the interaction between symmetry and deep learning in the context of inverse problems, demonstrating that when the forward map is equivariant under a group $G$, the inverse problem inherits fundamental non-uniqueness (solutions are determined only up to $G$-orbits), and that explicit symmetry breaking in the training data or architecture can improve reconstruction quality. More recently, Arai and Itano [9] developed a general group-representation-theoretic framework governing identifiability in inverse problems, formalizing how the structure of the symmetry group determines the quotient space on which the inverse map is well-defined. Our work applies and extends this perspective to the specific setting of vibrational spectroscopy, where the relevant symmetry group is the molecular point group and the observables are constrained by the representation-theoretic selection rules described in Section 2.2.

**Spectral similarity and structural similarity.** An important empirical question underlying the inverse spectral problem is whether spectrally similar molecules are necessarily structurally similar. Varmuza and Karlovits [20] conducted a systematic study of 13,484 organic compounds from the Sadtler IR spectral database, computing both spectral similarity (using correlation coefficients between discretized IR spectra) and structural similarity (using various topological descriptors). They found a statistically significant positive correlation between spectral and structural similarity, but the correlation was far from perfect: many structurally dissimilar molecules exhibited moderately similar spectra, and conversely, some structurally similar molecules (e.g., positional isomers differing in substitution pattern) showed notably different spectra. This empirical finding is consistent with the theoretical picture developed in this paper: the forward map $\Phi$ is not injective in general, but its non-injectivity is structured and predictable from symmetry and group-theoretic considerations. The confusable molecular sets we identify in Section 5 can be understood as the extreme cases of the Varmuza-Karlovits observation --- pairs of molecules that are maximally similar spectrally while being maximally dissimilar structurally.

---

## References

[1] M. Kac, "Can one hear the shape of a drum?", *American Mathematical Monthly* **73**(4), 1--23 (1966).

[2] C. Gordon, D. L. Webb, S. Wolpert, "One cannot hear the shape of a drum", *Bulletin of the American Mathematical Society* **27**(1), 134--138 (1992).

[3] J. Wang, S. Torquato, "Can one hear the shape of a crystal?", preprint (2025).

[4] DiffSpectra: Diffusion-based molecular generation from IR and Raman spectra, arXiv:2507.06853 (2025).

[5] Vib2Mol: A multi-task framework for vibrational spectrum-to-molecule translation, arXiv:2503.07014 (2025).

[6] VibraCLIP: Contrastive learning for vibrational spectroscopy and molecular structure, *RSC Digital Discovery* (2025).

[7] I. M. Mills, "The calculation of accurate normal coordinates and force constants from observed frequencies", *Spectrochimica Acta* **22**, 561--570 (1966).

[8] G. M. Kuramshina, F. A. Weinhold, I. V. Kochikov, A. G. Yagola, Yu. A. Pentin, "Joint treatment of ab initio and experimental data in molecular force field calculations with Tikhonov regularization", *Journal of Chemical Physics* **100**, 2516--2524 (1994).

[9] "Group-theoretic structure governing identifiability in inverse problems", arXiv:2511.08995 (2025).

[10] Y. Chen, M. E. Davies, "Inverse problems, deep learning, and symmetry breaking", arXiv:2003.09077 (2020).

[11] G. Borg, "Eine Umkehrung der Sturm-Liouvilleschen Eigenwertaufgabe. Bestimmung der Differentialgleichung durch die Eigenwerte", *Acta Mathematica* **78**, 1--96 (1946).

[12] E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations: The Theory of Infrared and Raman Vibrational Spectra*, McGraw-Hill, New York (1955).

[13] G. Herzberg, *Molecular Spectra and Molecular Structure, Vol. II: Infrared and Raman Spectra of Polyatomic Molecules*, Van Nostrand, New York (1945).

[14] J. von Neumann, E. Wigner, "Uber merkwurdige diskrete Eigenwerte", *Physikalische Zeitschrift* **30**, 465--467 (1929).

[15] F. A. Cotton, *Chemical Applications of Group Theory*, 3rd ed., Wiley, New York (1990).

[16] B. L. Crawford, W. H. Fletcher, "The determination of accurate normal coordinates", *Journal of Chemical Physics* **19**, 141--142 (1951).

[17] G. M. Kuramshina, "Inverse problems of vibrational spectroscopy", in *Vibrational Spectroscopy* (1999).

[18] SpectrumWorld: SpectrumBench with 1.2 million substances, arXiv:2508.01188 (2025).

[19] MolSpectLLM: Bridging spectroscopy and 3D molecular structure, arXiv:2509.21861 (2025).

[20] K. Varmuza, W. Karlovits, "Spectral similarity versus structural similarity: infrared spectroscopy", *Analytica Chimica Acta* **490**, 313--324 (2003).
