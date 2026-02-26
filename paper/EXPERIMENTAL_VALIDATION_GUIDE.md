# Experimental Validation of Information-Theoretic Theorems: A Comprehensive Guide

## Executive Summary

This document provides a complete methodology for experimentally validating the three main information-theoretic theorems in Spektron. It synthesizes state-of-the-art methods from 2023-2025 research for measuring abstract information-theoretic quantities (mutual information, Wasserstein distance, KL divergence, PID) from finite samples, with specific protocols, software recommendations, and validation procedures.

---

## Table of Contents

1. [Our Theorems and Measurement Challenges](#our-theorems)
2. [Mutual Information Estimation](#mutual-information)
3. [Partial Information Decomposition (PID)](#pid)
4. [Wasserstein Distance Estimation](#wasserstein)
5. [KL Divergence Estimation](#kl-divergence)
6. [Sample Complexity Measurement](#sample-complexity)
7. [Dimensionality Reduction Strategy](#dimensionality-reduction)
8. [Null Models and Statistical Testing](#null-models)
9. [Confidence Intervals and Uncertainty Quantification](#confidence-intervals)
10. [Validation with Synthetic Data](#synthetic-validation)
11. [Software Stack](#software-stack)
12. [Reporting Standards](#reporting-standards)
13. [Complete Experimental Protocols](#experimental-protocols)
14. [Common Pitfalls and Solutions](#pitfalls)

---

## 1. Our Theorems and Measurement Challenges {#our-theorems}

### Theorem 1: Sample Complexity with Equivariance
**Abstract claim:** N_equivariant ≤ (1/|G|) N_standard + O(log|G|)

**Concrete measurement:** Learning curves with varying |G| (point group order)

**Challenge:** How to control for confounders (molecular size, complexity varies with G)? Dataset imbalance (90% C₁, 1% Oₕ)?

### Theorem 2: Centrosymmetric Synergy
**Abstract claim:** For centrosymmetric molecules, I(M; S_IR, S_Raman) has zero redundancy and positive synergy

**Concrete measurement:** PID estimation from finite samples of (S_IR, S_Raman, M) tuples

**Challenge:** High dimensionality (D=128 latent), PID estimator disagreement, discretization artifacts

### Theorem 3: Transfer Learning Bound
**Abstract claim:** ε_transfer ≤ ε_src + C·W₂(P_chem) + D·KL(P_inst)

**Concrete measurement:** Estimate W₂ and KL from spectral distributions

**Challenge:** Sample complexity for Wasserstein in D=128, choosing regularization for Sinkhorn

---

## 2. Mutual Information Estimation {#mutual-information}

### 2.1 Available Methods

#### Method A: KSG Estimator (k-Nearest Neighbor)

**Reference:** [Beyond Normal: On the Evaluation of Mutual Information Estimators](https://proceedings.neurips.cc/paper_files/paper/2023/file/36b80eae70ff629d667f210e13497edf-Paper-Conference.pdf) (NeurIPS 2023)

**Pros:**
- Works for continuous variables
- Bias: O(1/N), variance: O(1/N)
- Well-established with theoretical guarantees
- Performs well in low-moderate dimensions

**Cons:**
- Requires N >> 2^D samples (curse of dimensionality)
- KSG struggles in high-dimensional and sparse interaction settings
- Systematically underestimates MI on real-world datasets
- Numerical overflow when D > several hundred

**Sample complexity:** N = O(1/ε²) for ε-accurate estimate

**Implementation:**
```python
from sklearn.feature_selection import mutual_info_regression
# or use specialized libraries like gcmi, dit

mi = mutual_info_regression(X, y, n_neighbors=5)
```

**Recommended for:** D ≤ 20, N > 1000

**Recent advances (2024):** [Logarithmic transformation](https://arxiv.org/html/2410.07642v1) mitigates overflow while maintaining precision for high-dimensional cases.

#### Method B: MINE (Mutual Information Neural Estimation)

**Reference:** [MINE: Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (ICML 2018)

**Pros:**
- Linearly scalable in dimensionality and sample size
- Works for high dimensions (D > 50)
- Trainable through backprop, strongly consistent
- Recent work: [Mutual Information Estimation via Normalizing Flows](https://proceedings.neurips.cc/paper_files/paper/2024/file/05a2d9ef0ae6f249737c1e4cce724a0c-Paper-Conference.pdf) (NeurIPS 2024)

**Cons:**
- Unstable, needs careful tuning
- Significant bias/variance at high MI
- Slow convergence at low MI
- Sample complexity: N = O(D/ε²) depends on dimension

**Recommended for:** D > 50, when neural network training is acceptable

**2024 findings:** Flow-based generative estimators show reduced bias/variance at high MI but suffer from slow convergence at low MI.

#### Method C: Gaussian Copula MI (GCMI)

**Reference:** [A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula](https://pmc.ncbi.nlm.nih.gov/articles/PMC5324576/) (Human Brain Mapping 2017)

**Library:** [robince/gcmi](https://github.com/robince/gcmi)

**Pros:**
- Assumes copula structure (not marginals) is Gaussian
- Robust, fast, computationally efficient
- Lower bound approximation to true MI
- Excellent for multivariate calculations
- Similar sensitivity to rank correlation

**Cons:**
- Biased if Gaussian copula assumption violated
- Only captures Gaussian-modellable dependence

**Implementation:**
```python
from gcmi import gcmi_cc, gcmi_model_cd, gcmi_ccc

# Continuous-continuous MI
mi = gcmi_cc(x, y)

# Continuous-discrete MI (ANOVA style)
mi = gcmi_model_cd(x, y_discrete)

# Conditional MI: I(X;Y|Z)
mi = gcmi_ccc(x, y, z)
```

**Validation:** Use QQ plots to check Gaussian copula assumption

**Recommended for:** D = 10-50, when assumptions hold, neuroscience applications

#### Method D: Geodesic k-NN

**Reference:** [Estimating Mutual Information via Geodesic kNN](https://arxiv.org/pdf/2110.13883) (2021)

**Innovation:** Uses geodesic distances on manifolds instead of Euclidean

**Recommended for:** Data lying on low-dimensional manifolds

### 2.2 Dimensionality Challenges

**Problem:** For D=128 latent space (z_chem), which estimator works with N ≈ 1000-10000?

**Solution:** Dimensionality reduction before estimation (see Section 7)

### 2.3 Recommended Protocol for Spektron

**For z_chem (D=128, N~5000):**

1. **Reduce dimensionality:** PCA to D=10-20
2. **Primary estimator:** GCMI (fast, robust)
3. **Validation estimator:** KSG with k=5
4. **Check assumption:** QQ plot for Gaussian copula
5. **Compute CI:** Bootstrap with 1000 iterations

**Code template:**
```python
from sklearn.decomposition import PCA
from gcmi import gcmi_cc
import numpy as np

# Reduce dimensionality
pca = PCA(n_components=10)
z_reduced = pca.fit_transform(z_chem)

# Estimate MI
mi_estimate = gcmi_cc(z_reduced, labels)

# Bootstrap CI
mi_bootstrap = []
for _ in range(1000):
    idx = np.random.choice(len(z_reduced), size=len(z_reduced), replace=True)
    mi_boot = gcmi_cc(z_reduced[idx], labels[idx])
    mi_bootstrap.append(mi_boot)

ci_lower = np.percentile(mi_bootstrap, 2.5)
ci_upper = np.percentile(mi_bootstrap, 97.5)
```

---

## 3. Partial Information Decomposition (PID) {#pid}

### 3.1 What is PID?

PID decomposes the information shared between a set of input variables and an output variable:

I(M; S_IR, S_Raman) = Redundancy + Unique_IR + Unique_Raman + Synergy

- **Redundancy:** Information both sources provide independently
- **Unique:** Information only one source provides
- **Synergy:** Information only available when both sources are combined

### 3.2 Available Libraries

#### dit (Discrete Information Theory)

**Reference:** [dit: discrete information theory](https://dit.readthedocs.io/en/latest/)

**Pros:**
- Multiple PID measures (I_ccs, I_mmi, I_broja)
- Well-documented, actively maintained
- Supports various information-theoretic measures

**Cons:**
- Requires discrete variables (need to bin continuous data)
- Binning loses information
- Different estimators can disagree by 2-5×

**Implementation:**
```python
from dit import Distribution, PID
from dit.multivariate import coinformation

# Create distribution from data
d = Distribution.from_data([S_IR_discrete, S_Raman_discrete, M_discrete])

# Compute PID
pid = PID(d, ['S_IR', 'S_Raman'], 'M')
print(f"Redundancy: {pid.redundancy}")
print(f"Unique IR: {pid.unique_A}")
print(f"Unique Raman: {pid.unique_B}")
print(f"Synergy: {pid.synergy}")
```

#### pidpy

**Reference:** [pietromarchesi/pidpy](https://github.com/pietromarchesi/pidpy)

**Features:** Computes pure PID terms for arbitrary number of sources using Williams & Beer proposal

#### GCMI for Gaussian PID

**Reference:** [Gaussian Partial Information Decomposition: Bias Correction](https://proceedings.neurips.cc/paper_files/paper/2023/file/ec0bff8bf4b11e36f874790046dfdb65-Paper-Conference.pdf) (NeurIPS 2023)

**Pros:**
- Restricts search space to Gaussian distributions
- Reduces optimization from exponential to quadratic in dimensionality
- Enables computation for much higher dimensions

**Cons:**
- Assumes Gaussian structure
- Lower bound on true PID

#### IDTxl (Information Dynamics Toolkit xl)

**Reference:** [IDTxl: The Information Dynamics Toolkit xl](https://github.com/pwollstadt/IDTxl) (JOSS 2019)

**Features:**
- Multivariate transfer entropy, MI, PID
- Parallel computing (GPU + CPU)
- Designed for neuroscience time series

### 3.3 Recent Advances (2024)

**Continuous PID:**
- [Partial Information Decomposition for Continuous Variables based on Shared Exclusions](https://link.aps.org/doi/10.1103/PhysRevE.110.014115) (Phys. Rev. E 2024)
- Nearest-neighbor-based estimator for continuous variables
- No discretization needed

**Mixed Discrete-Continuous:**
- [Partial information decomposition for mixed discrete and continuous random variables](https://arxiv.org/html/2409.13506) (2024)
- Non-parametric procedures based on nearest-neighbor entropy estimation

**Information Bottleneck Approach:**
- [Partial Information Decomposition: Redundancy as Information Bottleneck](https://www.mdpi.com/1099-4300/26/7/546) (Entropy 2024)
- Reformulates redundancy as information bottleneck
- Iterative algorithm for larger systems

### 3.4 Recommended Protocol for Centrosymmetric Synergy

**Theorem 2 validation workflow:**

```python
# Step 1: Train model and extract latents
z_chem_IR, z_chem_Raman = extract_latents(model, spectra_IR, spectra_Raman)
molecule_labels = get_molecule_ids()

# Step 2: Reduce dimensionality
from sklearn.decomposition import PCA
pca_IR = PCA(n_components=10)
pca_Raman = PCA(n_components=10)
z_IR_red = pca_IR.fit_transform(z_chem_IR)
z_Raman_red = pca_Raman.fit_transform(z_chem_Raman)

# Step 3: Discretize (for dit library)
def discretize(X, n_bins=5):
    return np.digitize(X, bins=np.linspace(X.min(), X.max(), n_bins))

z_IR_disc = discretize(z_IR_red)
z_Raman_disc = discretize(z_Raman_red)
M_disc = discretize(molecule_labels)

# Step 4: Compute PID (multiple estimators for robustness)
from dit import Distribution, PID

# Create joint distribution
data = np.stack([z_IR_disc, z_Raman_disc, M_disc], axis=-1)
d = Distribution.from_ndarray(data)

# Compute PID with different measures
pid_ccs = PID(d, sources=['IR', 'Raman'], target='M', measure='I_ccs')
pid_broja = PID(d, sources=['IR', 'Raman'], target='M', measure='I_broja')

# Step 5: Statistical testing
# Null hypothesis: Synergy is due to chance (permutation test)
synergy_obs = pid_ccs.synergy

synergies_null = []
for _ in range(1000):
    M_shuffled = np.random.permutation(M_disc)
    data_null = np.stack([z_IR_disc, z_Raman_disc, M_shuffled], axis=-1)
    d_null = Distribution.from_ndarray(data_null)
    pid_null = PID(d_null, sources=['IR', 'Raman'], target='M', measure='I_ccs')
    synergies_null.append(pid_null.synergy)

p_value = (np.array(synergies_null) >= synergy_obs).mean()

# Step 6: Bootstrap confidence intervals
synergies_boot = []
for _ in range(1000):
    idx = np.random.choice(len(data), size=len(data), replace=True)
    data_boot = data[idx]
    d_boot = Distribution.from_ndarray(data_boot)
    pid_boot = PID(d_boot, sources=['IR', 'Raman'], target='M', measure='I_ccs')
    synergies_boot.append(pid_boot.synergy)

ci_lower = np.percentile(synergies_boot, 2.5)
ci_upper = np.percentile(synergies_boot, 97.5)

print(f"Observed synergy: {synergy_obs:.3f} bits")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"p-value: {p_value:.4f}")
```

### 3.5 Validation with Toy Examples

**XOR gate (maximal synergy):**
```python
# Generate XOR data
X1 = np.random.binomial(1, 0.5, size=1000)
X2 = np.random.binomial(1, 0.5, size=1000)
Y = X1 ^ X2  # XOR

# Known: Redundancy ≈ 0, Synergy > 0
# Use to validate your PID pipeline
```

**AND gate (redundancy + synergy):**
```python
Y = X1 & X2
# Known: Redundancy > 0, Synergy > 0
```

**COPY gate (pure redundancy):**
```python
Y = X1  # X2 is irrelevant
# Known: Redundancy > 0, Synergy ≈ 0
```

**Reference:** [Synergy, redundancy, and multivariate information measures](http://www.beggslab.com/uploads/1/0/1/7/101719922/29timmeetal2013.pdf)

### 3.6 Sample Size Requirements

**Empirical finding (from neuroscience):** N ≈ 1000-5000 for D=10

**For Spektron:** With N ≈ 1000-10000 molecules, reduce to D=10-20 before PID estimation

---

## 4. Wasserstein Distance Estimation {#wasserstein}

### 4.1 Available Methods

#### Method A: Exact OT (for small N < 1000)

**Library:** POT (Python Optimal Transport)

**Reference:** [POT: Python Optimal Transport](https://pythonot.github.io/) (JMLR 2021)

**Complexity:** O(N³ log N)

```python
import ot

# Empirical distributions
a = ot.unif(N_src)  # uniform weights
b = ot.unif(N_tgt)
M = ot.dist(X_src, X_tgt)  # cost matrix (Euclidean distances)

# Exact Wasserstein-2
W2 = ot.emd2(a, b, M)
```

**Recommended for:** N < 1000

#### Method B: Entropic Regularization (Sinkhorn)

**Complexity:** O(N² / reg)

**Bias:** Regularization parameter introduces bias

```python
# Sinkhorn divergence
W2_sinkhorn = ot.sinkhorn2(a, b, M, reg=0.01)
```

**Choosing reg:**
- Small reg (0.001-0.01): Less bias, slower convergence
- Large reg (0.1-1.0): More bias, faster convergence
- Rule of thumb: reg = 0.01 * median(M)

**Recommended for:** N = 1000-10000

#### Method C: Sliced Wasserstein (for high-D)

**Reference:** [Sliced Wasserstein Estimation with Control Variates](https://proceedings.iclr.cc/paper_files/paper/2024/file/08f628998ca37c9df8c6a0df3570db86-Paper-Conference.pdf) (ICLR 2024)

**Pros:**
- Projects to 1D, computes W₂ on projections
- Fast: O(N log N)
- Unbiased: converges to true W₂ as n_projections → ∞
- Dimension-free sample complexity: N = O(1/ε²)

**Cons:**
- Requires many projections for convergence (100-1000)

```python
from ot.sliced import sliced_wasserstein_distance

SW = sliced_wasserstein_distance(X_src, X_tgt, n_projections=100)
```

**2024 advance:** Control variates reduce variance by fitting Gaussian approximations to projected measures.

**Recommended for:** D > 20, N > 1000

#### Method D: GeomLoss (PyTorch, GPU-accelerated)

**Reference:** [GeomLoss: Geometric Loss functions](https://www.kernel-operations.io/geomloss/)

**Pros:**
- Fully differentiable (for gradient flows)
- Linear memory footprint (via KeOps)
- 50-100× faster than standard Sinkhorn on GPU
- Log-domain stabilization (no overflow)
- ε-scaling heuristic

```python
from geomloss import SamplesLoss

loss = SamplesLoss(loss="sinkhorn", blur=0.01)
W2 = loss(X_src, X_tgt)
```

**Recommended for:** Deep learning integration, large-scale problems

#### Method E: Bures-Wasserstein (Gaussian assumption)

**Reference:** [Wasserstein distance between two Gaussians](https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/)

**Closed-form formula:**

If P = N(μ₁, Σ₁) and Q = N(μ₂, Σ₂):

W₂²(P, Q) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁^(1/2) Σ₂ Σ₁^(1/2)))

**Pros:**
- Exact, fast (no sampling error)
- No hyperparameters (reg, n_projections)

**Cons:**
- Assumes Gaussian distributions
- Matrix square root can be numerically unstable

```python
import numpy as np
from scipy.linalg import sqrtm

def bures_wasserstein(mu1, Sigma1, mu2, Sigma2):
    mean_diff = np.linalg.norm(mu1 - mu2)**2

    # Covariance term
    sqrt_Sigma1 = sqrtm(Sigma1)
    M = sqrtm(sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1)
    cov_term = np.trace(Sigma1 + Sigma2 - 2*M)

    return np.sqrt(mean_diff + cov_term)

# POT implementation
from ot.gaussian import bures_wasserstein_distance
W2 = bures_wasserstein_distance(mu1, Sigma1, mu2, Sigma2)
```

**Validation:** Check Gaussian assumption with QQ plots, Shapiro-Wilk test

**Recommended for:** When Gaussian assumption holds (test first!)

### 4.2 Sample Complexity

**Theoretical result:** [Fournier & Guillin 2015](https://arxiv.org/abs/1312.3334)

- Empirical OT: N = O(D/ε²)
- Sliced Wasserstein: N = O(1/ε²) (dimension-free!)

**For Spektron:** With D=128, N=1000-5000:
- Use Sliced Wasserstein or Gaussian assumption
- If using Sinkhorn, reduce dimensionality first

### 4.3 Confidence Intervals

**Challenge:** POT library doesn't provide CIs natively

**Solution:** Bootstrap

```python
def wasserstein_bootstrap_ci(X_src, X_tgt, n_bootstrap=1000):
    from ot.sliced import sliced_wasserstein_distance

    estimates = []
    for _ in range(n_bootstrap):
        idx_src = np.random.choice(len(X_src), size=len(X_src), replace=True)
        idx_tgt = np.random.choice(len(X_tgt), size=len(X_tgt), replace=True)

        W2_boot = sliced_wasserstein_distance(X_src[idx_src], X_tgt[idx_tgt], n_projections=100)
        estimates.append(W2_boot)

    return np.percentile(estimates, [2.5, 97.5])
```

### 4.4 Recommended Protocol for Transfer Learning Bound

**Theorem 3 validation:**

```python
# Extract latent distributions
z_chem_src = extract_z_chem(model, spectra_src)
z_chem_tgt = extract_z_chem(model, spectra_tgt)

# Option 1: Gaussian assumption
mu_src, Sigma_src = z_chem_src.mean(0), np.cov(z_chem_src.T)
mu_tgt, Sigma_tgt = z_chem_tgt.mean(0), np.cov(z_chem_tgt.T)

# Check Gaussian assumption
from scipy.stats import shapiro
_, p_value = shapiro(z_chem_src[:, 0])  # Check first dimension
if p_value > 0.05:
    # Use Bures-Wasserstein
    from ot.gaussian import bures_wasserstein_distance
    W2 = bures_wasserstein_distance(mu_src, Sigma_src, mu_tgt, Sigma_tgt)
else:
    # Use Sliced Wasserstein
    from ot.sliced import sliced_wasserstein_distance
    W2 = sliced_wasserstein_distance(z_chem_src, z_chem_tgt, n_projections=100)

# Compute CI
ci_lower, ci_upper = wasserstein_bootstrap_ci(z_chem_src, z_chem_tgt)

print(f"W₂(P_chem_src, P_chem_tgt): {W2:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

---

## 5. KL Divergence Estimation {#kl-divergence}

### 5.1 Available Methods

#### Method A: KNN Estimator (Pérez-Cruz 2008)

**Reference:** [Estimation of mutual information for real-valued data](https://link.aps.org/accepted/10.1103/PhysRevE.100.022404)

**Formula:**

KL(P || Q) ≈ (D/N) Σᵢ log(rₖ(Xᵢ) / sₖ(Xᵢ))

where rₖ is k-NN distance in P, sₖ is k-NN distance in Q

**Pros:**
- Works for continuous data
- Consistent estimator
- Theoretical guarantees

**Cons:**
- Curse of dimensionality (needs N >> 2^D)

```python
from scipy.spatial import cKDTree

def kl_knn(X, Y, k=5):
    """KL(P_X || P_Y) via k-NN"""
    N, D = X.shape

    tree_X = cKDTree(X)
    tree_Y = cKDTree(Y)

    r = tree_X.query(X, k=k+1)[0][:, -1]  # k-NN distance in X
    s = tree_Y.query(X, k=k)[0][:, -1]    # k-NN distance in Y

    return (D/N) * np.sum(np.log(s / r)) + np.log(len(Y) / (len(X) - 1))
```

**Recommended for:** D ≤ 20, N > 1000

#### Method B: Parametric (Gaussian assumption)

**Closed-form for Gaussians:**

If P = N(μ_p, Σ_p) and Q = N(μ_q, Σ_q):

KL(P || Q) = 0.5 * [tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - D + log(det Σ_q / det Σ_p)]

```python
def kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q):
    D = len(mu_p)

    Sigma_q_inv = np.linalg.inv(Sigma_q)
    mu_diff = mu_q - mu_p

    term1 = np.trace(Sigma_q_inv @ Sigma_p)
    term2 = mu_diff.T @ Sigma_q_inv @ mu_diff
    term3 = np.log(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))

    return 0.5 * (term1 + term2 - D + term3)
```

**Recommended for:** When Gaussian assumption holds (always test first!)

#### Method C: Neural Estimator

**Library:** Based on MINE framework

**Pros:** Works for high dimensions

**Cons:** Unstable, needs careful tuning

**Not recommended** unless other methods fail

### 5.2 Recommended Protocol for Instrument Distribution

**For P_inst (instrument-specific latent, D ≈ 10-20):**

```python
# Extract instrument latents
z_inst_src = extract_z_inst(model, spectra_src)
z_inst_tgt = extract_z_inst(model, spectra_tgt)

# Check Gaussian assumption
from scipy.stats import shapiro
is_gaussian = all(shapiro(z_inst_src[:, i])[1] > 0.05 for i in range(z_inst_src.shape[1]))

if is_gaussian:
    # Gaussian KL
    mu_src, Sigma_src = z_inst_src.mean(0), np.cov(z_inst_src.T)
    mu_tgt, Sigma_tgt = z_inst_tgt.mean(0), np.cov(z_inst_tgt.T)
    kl = kl_gaussian(mu_src, Sigma_src, mu_tgt, Sigma_tgt)
else:
    # k-NN KL
    kl = kl_knn(z_inst_src, z_inst_tgt, k=5)

# Bootstrap CI
kl_bootstrap = []
for _ in range(1000):
    idx_src = np.random.choice(len(z_inst_src), size=len(z_inst_src), replace=True)
    idx_tgt = np.random.choice(len(z_inst_tgt), size=len(z_inst_tgt), replace=True)

    if is_gaussian:
        mu_src_boot = z_inst_src[idx_src].mean(0)
        Sigma_src_boot = np.cov(z_inst_src[idx_src].T)
        mu_tgt_boot = z_inst_tgt[idx_tgt].mean(0)
        Sigma_tgt_boot = np.cov(z_inst_tgt[idx_tgt].T)
        kl_boot = kl_gaussian(mu_src_boot, Sigma_src_boot, mu_tgt_boot, Sigma_tgt_boot)
    else:
        kl_boot = kl_knn(z_inst_src[idx_src], z_inst_tgt[idx_tgt], k=5)

    kl_bootstrap.append(kl_boot)

ci_lower, ci_upper = np.percentile(kl_bootstrap, [2.5, 97.5])
```

---

## 6. Sample Complexity Measurement {#sample-complexity}

### 6.1 Standard Protocol for Learning Curves

**Reference:** [Explaining neural scaling laws](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228526/) (PNAS 2024)

**Method:**

1. Vary training set size: N ∈ {10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
2. For each N, train model with 5-10 random seeds
3. Evaluate on fixed test set
4. Plot: Learning curve (N vs error)
5. Fit: Power law error(N) = A·N^(-α) + B

**Power law characteristics:**
- Large models are more sample-efficient (steeper α)
- Test loss scales as power-law with model size, dataset size, compute
- Trends span 7+ orders of magnitude

```python
import numpy as np
from scipy.optimize import curve_fit

def power_law(N, A, alpha, B):
    return A * N**(-alpha) + B

# Generate learning curve
sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
errors = []

for N in sample_sizes:
    errors_per_seed = []
    for seed in range(10):
        model = train_model(data[:N], seed=seed)
        error = evaluate(model, test_data)
        errors_per_seed.append(error)
    errors.append(np.mean(errors_per_seed))

# Fit power law
params, _ = curve_fit(power_law, sample_sizes, errors)
A, alpha, B = params

print(f"Sample complexity exponent α: {alpha:.3f}")
print(f"Error(N) = {A:.3f} * N^(-{alpha:.3f}) + {B:.3f}")
```

### 6.2 Equivariance and Sample Complexity

**Theorem 1 specific protocol:**

**Challenge:** Vary |G| (point group order), not just N

**Solution:**

1. **Dataset stratification:** Group molecules by point group
   - C₁ (order 1): No symmetry
   - Cₛ (order 2): Mirror plane
   - C₂ᵥ (order 4): 2 mirrors, 1 rotation
   - D₃ₕ (order 12): High symmetry
   - Oₕ (order 48): Cubic symmetry

2. **Balanced sampling:** Ensure equal representation per group

3. **Comparison:**
   - Baseline model: No equivariance
   - Equivariant model: E(3) or SO(3) equivariance

4. **Measure:** N required to reach target error (e.g., R² = 0.90)

**Reference:** [On the Sample Complexity of One Hidden Layer Networks with Equivariance](https://arxiv.org/html/2411.14288) (2024)

**Theoretical finding:** Sample complexity bounds depend on norm of filters, dimension-independent for equivariant networks

```python
# Protocol
point_groups = {
    'C1': molecules_c1,
    'Cs': molecules_cs,
    'C2v': molecules_c2v,
    'D3h': molecules_d3h,
    'Oh': molecules_oh
}

results = {}

for group_name, molecules in point_groups.items():
    G_order = get_point_group_order(group_name)

    # Learning curve for standard model
    N_standard = measure_sample_complexity(
        model=StandardModel(),
        data=molecules,
        target_r2=0.90
    )

    # Learning curve for equivariant model
    N_equivariant = measure_sample_complexity(
        model=EquivariantModel(),
        data=molecules,
        target_r2=0.90
    )

    results[group_name] = {
        'order': G_order,
        'N_standard': N_standard,
        'N_equivariant': N_equivariant,
        'ratio': N_standard / N_equivariant
    }

# Verify theorem: N_equivariant ≈ N_standard / |G|
for group_name, res in results.items():
    predicted_ratio = res['order']
    observed_ratio = res['ratio']
    print(f"{group_name}: Predicted {predicted_ratio:.1f}×, Observed {observed_ratio:.1f}×")
```

### 6.3 Controlling for Confounders

**Problem:** Molecular complexity varies with symmetry

**Solutions:**

1. **Matching:** Pair molecules with similar properties but different symmetries
2. **Regression adjustment:** Control for molecular weight, # atoms, # bonds
3. **Stratification:** Analyze within molecular weight bins

**Example:**
```python
from sklearn.linear_model import LinearRegression

# Control for confounders
X = np.column_stack([
    point_group_order,
    molecular_weight,
    num_atoms,
    num_bonds
])
y = sample_complexity

# Fit regression
model = LinearRegression()
model.fit(X, y)

# Coefficient for point_group_order (controlling for other variables)
point_group_effect = model.coef_[0]
print(f"Point group order effect (controlled): {point_group_effect:.3f}")
```

---

## 7. Dimensionality Reduction Strategy {#dimensionality-reduction}

### 7.1 The Tradeoff

**Problem:** High-D (D=128) requires huge N, Low-D (D=10) loses information

**Rule of thumb:** Reduce to D such that D ≤ N / 10

**For Spektron:** N=1000-10000 → use D=10-20

### 7.2 Information Preservation

**Question:** Does MI(X, Y) ≈ MI(PCA(X), PCA(Y))?

**Theory:** MI is preserved by invertible transforms, but PCA is not invertible (loses info)

**Validation protocol:**

```python
from sklearn.decomposition import PCA
from gcmi import gcmi_cc

# Original MI (infeasible for D=128, but compute on subset)
mi_original = gcmi_cc(X, y)

# Reduced MI
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
mi_reduced = gcmi_cc(X_reduced, y)

# Information loss
info_loss = (mi_original - mi_reduced) / mi_original
print(f"Information loss: {info_loss:.1%}")

# Variance explained
var_explained = pca.explained_variance_ratio_.sum()
print(f"Variance explained: {var_explained:.1%}")
```

**Target:** Retain 90-95% of variance with PCA

### 7.3 PCA vs UMAP vs Autoencoders

**Reference:** [Towards One Model for Classical Dimensionality Reduction: A Probabilistic Perspective on UMAP and t-SNE](https://arxiv.org/html/2405.17412v1) (2024)

| Method | Pros | Cons | Use case |
|--------|------|------|----------|
| **PCA** | Linear, fast, preserves global structure | Assumes linear relationships | First choice, baseline |
| **UMAP** | Nonlinear, preserves local + global structure | Stochastic, hyperparameters | Complex manifolds |
| **Autoencoder** | Learnable, task-specific | Requires training, overfitting risk | Deep integration |
| **Information Bottleneck** | Theoretically motivated, preserves MI | Computationally expensive | When MI preservation critical |

**Information Bottleneck reference:** [Data Efficiency, Dimensionality Reduction, and the Generalized Symmetric Information Bottleneck](https://direct.mit.edu/neco/article-abstract/36/7/1353/120664/) (Neural Computation 2024)

**Recommendation for Spektron:**
1. Start with PCA (fast, interpretable)
2. Validate with UMAP (check if nonlinear structure matters)
3. Use Information Bottleneck if MI preservation is critical

### 7.4 Optimal Dimensionality Selection

**Method 1: Scree plot (PCA)**

```python
pca_full = PCA()
pca_full.fit(X)

# Plot explained variance
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.axhline(0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()

# Choose D where 95% variance is explained
D_optimal = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
```

**Method 2: Cross-validation (task-specific)**

```python
from sklearn.model_selection import cross_val_score

dimensions = [5, 10, 15, 20, 30, 50]
scores = []

for D in dimensions:
    pca = PCA(n_components=D)
    X_reduced = pca.fit_transform(X)

    # Evaluate on downstream task
    score = cross_val_score(model, X_reduced, y, cv=5).mean()
    scores.append(score)

# Choose D with best downstream performance
D_optimal = dimensions[np.argmax(scores)]
```

---

## 8. Null Models and Statistical Testing {#null-models}

### 8.1 Permutation Tests

**Null hypothesis:** Observed information quantity is due to chance

**Reference:** [Review about the Permutation Approach in Hypothesis Testing](https://www.mdpi.com/2227-7390/12/17/2617) (Mathematics 2024)

**Standard protocol:**

```python
def permutation_test(X, y, estimator_func, n_permutations=1000):
    """
    X: features
    y: labels
    estimator_func: function that computes statistic (MI, synergy, etc.)
    """
    # Observed statistic
    stat_obs = estimator_func(X, y)

    # Null distribution
    stats_null = []
    for _ in range(n_permutations):
        y_shuffled = np.random.permutation(y)
        stat_null = estimator_func(X, y_shuffled)
        stats_null.append(stat_null)

    # p-value
    p_value = (np.array(stats_null) >= stat_obs).mean()

    return stat_obs, p_value, stats_null
```

**Application to PID synergy:**

```python
def compute_synergy(S_IR, S_Raman, M):
    # ... PID computation
    return pid.synergy

stat_obs, p_value, null_dist = permutation_test(
    X=np.stack([S_IR, S_Raman], axis=-1),
    y=M,
    estimator_func=lambda X, y: compute_synergy(X[:, 0], X[:, 1], y),
    n_permutations=1000
)

print(f"Observed synergy: {stat_obs:.3f} bits")
print(f"p-value: {p_value:.4f}")
```

**How many permutations?**
- Typical: 1000-10000
- For p < 0.001: Need at least 10000 permutations
- Rule of thumb: n_permutations ≥ 100 / target_p_value

### 8.2 NuMIT: Null Models for Information Theory

**Reference:** [Null models for comparing information decomposition across complex systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC12614810/) (PLOS Comp Bio 2024)

**Innovation:** Non-linear normalization for information-theoretic measures

**Problem addressed:** Information measures vary significantly across systems with similar properties

**Solution:** Compare observed PID against null model that preserves certain properties but destroys others

**Example null models:**

1. **Label shuffling:** Preserves marginal distributions, destroys dependencies
2. **Time shuffling (for time series):** Preserves autocorrelation, destroys cross-correlations
3. **Phase randomization:** Preserves power spectrum, destroys phase relationships

```python
def phase_randomization_null(signal):
    """Preserve power spectrum, randomize phase"""
    fft = np.fft.rfft(signal)
    amplitudes = np.abs(fft)
    random_phases = np.random.uniform(0, 2*np.pi, len(fft))
    fft_shuffled = amplitudes * np.exp(1j * random_phases)
    return np.fft.irfft(fft_shuffled, n=len(signal))
```

**NuMIT protocol:**

```python
# Observed PID
pid_obs = compute_pid(S_IR, S_Raman, M)

# Generate null ensemble
pid_null = []
for _ in range(1000):
    # Null model: shuffle molecule labels
    M_shuffled = np.random.permutation(M)
    pid_null_i = compute_pid(S_IR, S_Raman, M_shuffled)
    pid_null.append(pid_null_i)

# Normalized PID
pid_mean_null = np.mean(pid_null)
pid_std_null = np.std(pid_null)
pid_normalized = (pid_obs - pid_mean_null) / pid_std_null

# Significance
p_value = (np.array(pid_null) >= pid_obs).mean()
```

**Recommendation:** Always report both raw and normalized values

### 8.3 Multiple Hypothesis Correction

**Problem:** Testing multiple point groups / molecules / conditions → inflated false positive rate

**Solution:** Bonferroni, FDR (Benjamini-Hochberg)

```python
from statsmodels.stats.multitest import multipletests

# p-values from multiple tests
p_values = [0.03, 0.01, 0.15, 0.002, 0.08]

# Bonferroni correction (conservative)
reject_bonf, p_adjusted_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# FDR correction (less conservative)
reject_fdr, p_adjusted_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("Bonferroni-adjusted p-values:", p_adjusted_bonf)
print("FDR-adjusted p-values:", p_adjusted_fdr)
```

---

## 9. Confidence Intervals and Uncertainty Quantification {#confidence-intervals}

### 9.1 Bootstrap Method

**Reference:** [Bootstrap confidence intervals: A comparative simulation study](https://arxiv.org/html/2404.12967v1) (2024)

**Standard bootstrap:**

```python
def bootstrap_ci(data, estimator_func, n_bootstrap=1000, confidence=0.95):
    """
    data: dataset (can be tuple for multivariate)
    estimator_func: function that computes statistic from data
    """
    estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        if isinstance(data, tuple):
            n = len(data[0])
            idx = np.random.choice(n, size=n, replace=True)
            data_boot = tuple(d[idx] for d in data)
        else:
            idx = np.random.choice(len(data), size=len(data), replace=True)
            data_boot = data[idx]

        estimate = estimator_func(data_boot)
        estimates.append(estimate)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper, estimates
```

**Bootstrap variants:**

1. **Percentile bootstrap** (above): Simple, most common
2. **BCa (Bias-corrected accelerated)**: Accounts for bias and skewness
3. **Parametric bootstrap**: Assume distribution, resample from fitted model

**When to use which:**
- Percentile: Default choice
- BCa: When distribution is skewed or estimator is biased
- Parametric: When you trust the parametric assumption

### 9.2 Analytical CIs (when available)

**KSG estimator:** Known asymptotic variance

**Formula:** Var(MI_KSG) ≈ ψ(k) / N, where ψ is digamma function

```python
from scipy.special import digamma

def ksg_ci(mi_estimate, N, k=5, confidence=0.95):
    """Analytical CI for KSG estimator"""
    var = digamma(k) / N
    std = np.sqrt(var)

    z = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = mi_estimate - z * std
    ci_upper = mi_estimate + z * std

    return ci_lower, ci_upper
```

**Limitation:** Assumes asymptotic regime (large N)

### 9.3 CI Width Interpretation

**How wide is too wide?**

- **Narrow CI** (width < 10% of estimate): Estimate is reliable
- **Moderate CI** (width 10-30%): Estimate is informative but uncertain
- **Wide CI** (width > 30% or spans zero): Estimate is unreliable, need more data

**Example:**
- MI = 2.5 bits, CI = [2.3, 2.7]: Narrow, reliable
- MI = 2.5 bits, CI = [2.0, 3.0]: Moderate, informative
- MI = 2.5 bits, CI = [0.5, 4.5]: Wide, unreliable

### 9.4 Sample Size for Target CI Width

**Question:** How much data to get CI width < ε?

**Bootstrap approach:**

```python
def estimate_required_sample_size(data, estimator_func, target_width, max_N=10000):
    """Estimate N needed for CI width < target_width"""
    sample_sizes = np.logspace(2, np.log10(max_N), 10).astype(int)
    widths = []

    for N in sample_sizes:
        data_subset = data[:N]
        ci_lower, ci_upper, _ = bootstrap_ci(data_subset, estimator_func, n_bootstrap=100)
        width = ci_upper - ci_lower
        widths.append(width)

    # Fit power law: width(N) = a * N^b
    from scipy.optimize import curve_fit
    params, _ = curve_fit(lambda n, a, b: a * n**b, sample_sizes, widths)

    # Solve for N: target_width = a * N^b
    N_required = (target_width / params[0])**(1 / params[1])

    return int(N_required)
```

---

## 10. Validation with Synthetic Data {#synthetic-validation}

### 10.1 Why Synthetic Validation?

**Purpose:** Verify estimators work before applying to real data

**Protocol:**
1. Generate synthetic data with known ground truth
2. Estimate using your pipeline
3. Compare estimate vs truth
4. Iterate on method until accurate

### 10.2 Synthetic Data for MI

**Bivariate Gaussian with known MI:**

```python
def generate_correlated_gaussian(N, rho):
    """
    Generate X, Y with correlation rho
    True MI = -0.5 * log(1 - rho^2)
    """
    Sigma = np.array([[1, rho], [rho, 1]])
    XY = np.random.multivariate_normal([0, 0], Sigma, size=N)
    return XY[:, 0], XY[:, 1]

# Test
rho = 0.6
true_MI = -0.5 * np.log(1 - rho**2)  # = 0.222 bits

X, Y = generate_correlated_gaussian(N=1000, rho=rho)

# Estimate with GCMI
from gcmi import gcmi_cc
estimated_MI = gcmi_cc(X, Y)

print(f"True MI: {true_MI:.3f} bits")
print(f"Estimated MI: {estimated_MI:.3f} bits")
print(f"Error: {abs(estimated_MI - true_MI):.3f} bits")
```

**High-dimensional Gaussian:**

```python
def generate_high_d_gaussian(N, D, signal_dims=5, rho=0.6):
    """
    D-dimensional Gaussian where only first signal_dims correlate with y
    """
    X = np.random.randn(N, D)

    # Generate y as weighted sum of first signal_dims
    weights = np.random.randn(signal_dims)
    y = X[:, :signal_dims] @ weights + np.random.randn(N)

    return X, y

# Test dimensionality reduction + MI
X, y = generate_high_d_gaussian(N=1000, D=128, signal_dims=5)

# Reduce
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

# Estimate
mi_full = gcmi_cc(X, y)  # May fail or be inaccurate for D=128
mi_reduced = gcmi_cc(X_reduced, y)

print(f"MI (full D=128): {mi_full:.3f} bits")
print(f"MI (reduced D=10): {mi_reduced:.3f} bits")
```

### 10.3 Synthetic Data for PID

**Classic toy examples:**

**XOR (maximal synergy):**
```python
def generate_xor(N):
    X1 = np.random.binomial(1, 0.5, size=N)
    X2 = np.random.binomial(1, 0.5, size=N)
    Y = X1 ^ X2

    # Known: Redundancy ≈ 0, Synergy > 0
    return X1, X2, Y

X1, X2, Y = generate_xor(1000)
pid = compute_pid(X1, X2, Y)
assert pid.redundancy < 0.1  # Should be near zero
assert pid.synergy > 0.5     # Should be positive
```

**AND (redundancy + synergy):**
```python
X1, X2 = np.random.binomial(1, 0.5, size=(2, 1000))
Y = X1 & X2
# Known: Both redundancy and synergy > 0
```

**COPY (pure redundancy):**
```python
X1 = np.random.binomial(1, 0.5, size=1000)
X2 = np.random.binomial(1, 0.5, size=1000)
Y = X1  # X2 is irrelevant
# Known: Redundancy > 0, Synergy ≈ 0
```

**Validation protocol:**
```python
test_cases = {
    'XOR': (generate_xor, {'redundancy': 'low', 'synergy': 'high'}),
    'AND': (generate_and, {'redundancy': 'high', 'synergy': 'high'}),
    'COPY': (generate_copy, {'redundancy': 'high', 'synergy': 'low'})
}

for name, (generator, expected) in test_cases.items():
    X1, X2, Y = generator(1000)
    pid = compute_pid(X1, X2, Y)

    print(f"\n{name}:")
    print(f"  Redundancy: {pid.redundancy:.3f} (expected {expected['redundancy']})")
    print(f"  Synergy: {pid.synergy:.3f} (expected {expected['synergy']})")
```

### 10.4 Synthetic Data for Wasserstein Distance

**Two Gaussians with known W₂:**

```python
def test_wasserstein_estimator(N, D):
    # Two Gaussians shifted by distance δ
    delta = 2.0

    X = np.random.randn(N, D)
    Y = np.random.randn(N, D) + delta  # Shifted

    # True W₂ for Gaussians with same covariance: W₂ = ||μ_X - μ_Y||
    true_W2 = delta * np.sqrt(D)

    # Estimate
    from ot.sliced import sliced_wasserstein_distance
    estimated_W2 = sliced_wasserstein_distance(X, Y, n_projections=100)

    print(f"True W₂: {true_W2:.3f}")
    print(f"Estimated W₂: {estimated_W2:.3f}")
    print(f"Error: {abs(estimated_W2 - true_W2):.3f}")

    return abs(estimated_W2 - true_W2) / true_W2  # Relative error

# Test across dimensions
for D in [10, 50, 100]:
    error = test_wasserstein_estimator(N=1000, D=D)
    print(f"D={D}: Relative error {error:.1%}\n")
```

### 10.5 Monte Carlo Validation

**Reference:** [A survey of Monte Carlo methods for parameter estimation](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-020-00675-6)

**Protocol:**

```python
def monte_carlo_validation(true_parameter, estimator_func, data_generator,
                          n_trials=100, sample_sizes=[100, 500, 1000, 5000]):
    """
    true_parameter: known ground truth
    estimator_func: your estimator
    data_generator: function that generates synthetic data
    """
    results = []

    for N in sample_sizes:
        estimates = []
        for trial in range(n_trials):
            data = data_generator(N)
            estimate = estimator_func(data)
            estimates.append(estimate)

        bias = np.mean(estimates) - true_parameter
        variance = np.var(estimates)
        mse = bias**2 + variance

        results.append({
            'N': N,
            'mean_estimate': np.mean(estimates),
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'std': np.sqrt(variance)
        })

    return pd.DataFrame(results)

# Example: Validate MI estimator
true_MI = 1.0  # bits
results = monte_carlo_validation(
    true_parameter=true_MI,
    estimator_func=lambda data: gcmi_cc(data[0], data[1]),
    data_generator=lambda N: generate_correlated_gaussian(N, rho=0.8),
    n_trials=100
)

print(results)
# Check: bias → 0 and variance → 0 as N increases
```

---

## 11. Software Stack {#software-stack}

### 11.1 Core Libraries

| Library | Purpose | Installation | Documentation |
|---------|---------|--------------|---------------|
| **POT** | Optimal Transport | `pip install POT` | [pythonot.github.io](https://pythonot.github.io/) |
| **gcmi** | Gaussian Copula MI | Clone from [github.com/robince/gcmi](https://github.com/robince/gcmi) | GitHub README |
| **dit** | Discrete Information Theory, PID | `pip install dit` | [dit.readthedocs.io](https://dit.readthedocs.io/) |
| **IDTxl** | Transfer Entropy, PID | `pip install idtxl` | [pwollstadt.github.io/IDTxl](https://pwollstadt.github.io/IDTxl/) |
| **GeomLoss** | GPU-accelerated OT | `pip install geomloss` | [kernel-operations.io/geomloss](https://www.kernel-operations.io/geomloss/) |
| **scikit-learn** | PCA, MI estimators | `pip install scikit-learn` | [scikit-learn.org](https://scikit-learn.org/) |

### 11.2 Installation Script

```bash
# Create environment
conda create -n spectral_fm_validation python=3.10
conda activate spectral_fm_validation

# Install core libraries
pip install POT geomloss dit scikit-learn scipy numpy pandas matplotlib seaborn

# Install gcmi (manual)
git clone https://github.com/robince/gcmi.git
cd gcmi/python
# Add to PYTHONPATH or copy gcmi.py to project

# Install IDTxl
pip install idtxl

# For GPU support (optional)
pip install torch torchvision  # For GeomLoss GPU acceleration
pip install pykeops  # For KeOps (GeomLoss backend)
```

### 11.3 Requirements File

```txt
# requirements_validation.txt
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
POT>=0.9.5
geomloss>=0.2.6
dit>=1.2.3
idtxl>=1.5.1
torch>=2.0.0  # Optional, for GPU
statsmodels>=0.14.0  # For multiple hypothesis testing
```

### 11.4 Version Compatibility

**Tested combinations:**
- Python 3.10 + POT 0.9.5 + NumPy 1.24
- Python 3.11 + POT 0.9.6 + NumPy 1.26 (latest as of 2024)

**Known issues:**
- dit requires Python < 3.12 (as of 2024)
- gcmi is not pip-installable, must clone

---

## 12. Reporting Standards {#reporting-standards}

### 12.1 What to Report

**For every information-theoretic quantity, report:**

1. **Point estimate** (e.g., MI = 0.45 bits)
2. **Confidence interval** (95% CI: [0.38, 0.52])
3. **Sample size** (N = 2000 molecules)
4. **Estimator used** (KSG with k=5, or GCMI)
5. **Preprocessing** (PCA to D=10, then standardized)
6. **Statistical test** (p < 0.001 vs null model)
7. **Effect size** (if applicable)

### 12.2 Example Table Format

**Table: Information-Theoretic Quantities for Centrosymmetric Molecules**

| Quantity | Estimate | 95% CI | N | Estimator | Preprocessing | p-value |
|----------|----------|--------|---|-----------|---------------|---------|
| I(M; S_IR) | 2.34 bits | [2.21, 2.48] | 2000 | GCMI | PCA to D=10 | <0.001* |
| I(M; S_Raman) | 2.18 bits | [2.05, 2.32] | 2000 | GCMI | PCA to D=10 | <0.001* |
| I(M; S_IR, S_Raman) | 4.80 bits | [4.61, 5.01] | 2000 | GCMI | PCA to D=10 | <0.001* |
| Redundancy | 0.08 bits | [0.00, 0.18] | 2000 | Gaussian PID | PCA to D=10 | 0.082 |
| Synergy | 0.28 bits | [0.15, 0.41] | 2000 | Gaussian PID | PCA to D=10 | 0.002** |
| Unique (IR) | 0.26 bits | [0.14, 0.39] | 2000 | Gaussian PID | PCA to D=10 | 0.005** |
| Unique (Raman) | 0.10 bits | [0.02, 0.21] | 2000 | Gaussian PID | PCA to D=10 | 0.041* |

*p < 0.05, **p < 0.01 vs permutation null model (10000 permutations)

### 12.3 Example Figure

**Figure: Sample Complexity of Equivariant vs Non-Equivariant Models**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Learning curves
ax = axes[0]
for model_type in ['Standard', 'Equivariant']:
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    errors = get_learning_curve(model_type)
    errors_std = get_learning_curve_std(model_type)

    ax.loglog(sample_sizes, errors, 'o-', label=model_type)
    ax.fill_between(sample_sizes,
                     errors - errors_std,
                     errors + errors_std,
                     alpha=0.2)

ax.set_xlabel('Training set size (N)')
ax.set_ylabel('Test error (RMSEP)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('A. Learning Curves')

# Panel B: Sample complexity vs point group order
ax = axes[1]
point_groups = ['C1', 'Cs', 'C2v', 'D3h', 'Oh']
orders = [1, 2, 4, 12, 48]
sample_complexities = get_sample_complexities(point_groups)

ax.scatter(orders, sample_complexities['standard'], label='Standard', s=100)
ax.scatter(orders, sample_complexities['equivariant'], label='Equivariant', s=100)

# Theoretical prediction: N ∝ 1/|G|
ax.plot(orders, sample_complexities['standard'][0] / np.array(orders),
        '--', color='gray', label='Theory: N ∝ 1/|G|')

ax.set_xlabel('Point group order |G|')
ax.set_ylabel('Sample complexity N (for R² = 0.90)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('B. Sample Complexity vs Symmetry')

plt.tight_layout()
plt.savefig('sample_complexity_equivariance.pdf', dpi=300)
```

### 12.4 Methods Section Template

**Template for paper Methods section:**

> **Information-Theoretic Analysis**
>
> Mutual information between latent representations and molecular properties was estimated using the Gaussian copula mutual information (GCMI) method [cite]. Latent vectors (D=128) were first reduced to D=10 using PCA, retaining 94.2% of variance. GCMI assumes the copula structure is Gaussian; we validated this assumption using quantile-quantile plots (Supplementary Fig. X).
>
> Partial information decomposition (PID) was computed using the Gaussian PID estimator [cite]. For centrosymmetric molecules (N=2000), we decomposed I(M; S_IR, S_Raman) into redundancy, synergy, and unique information components. Statistical significance was assessed using permutation tests with 10,000 permutations, where molecule labels were shuffled to create a null distribution.
>
> Wasserstein distances between instrument-specific latent distributions were estimated using the Sliced Wasserstein metric with 100 random projections [cite]. For Gaussian-distributed latents, we used the closed-form Bures-Wasserstein distance [cite]. Gaussianity was tested using the Shapiro-Wilk test (α = 0.05).
>
> All confidence intervals were computed using bootstrap resampling (1,000 iterations) with the percentile method. Multiple hypothesis testing was corrected using the Benjamini-Hochberg FDR procedure (q = 0.05).

### 12.5 Supplementary Material Checklist

**Include in supplementary:**

- [ ] Full hyperparameter settings for all estimators
- [ ] QQ plots validating Gaussian assumptions
- [ ] Convergence plots for iterative methods (Sinkhorn, MINE)
- [ ] Bootstrap distributions for all estimates
- [ ] Null distributions from permutation tests
- [ ] Sensitivity analysis (varying k for KNN, reg for Sinkhorn, etc.)
- [ ] Synthetic validation results
- [ ] Code repository link (GitHub/Zenodo)

---

## 13. Complete Experimental Protocols {#experimental-protocols}

### 13.1 Protocol for Theorem 1: Sample Complexity with Equivariance

**Objective:** Verify N_equivariant ≤ (1/|G|) N_standard + O(log|G|)

**Steps:**

1. **Dataset Preparation**
   ```python
   # Stratify molecules by point group
   molecules_by_group = {
       'C1': load_molecules(point_group='C1'),
       'Cs': load_molecules(point_group='Cs'),
       'C2v': load_molecules(point_group='C2v'),
       'D3h': load_molecules(point_group='D3h'),
       'Oh': load_molecules(point_group='Oh')
   }

   # Balance dataset
   min_count = min(len(mols) for mols in molecules_by_group.values())
   for group in molecules_by_group:
       molecules_by_group[group] = molecules_by_group[group][:min_count]
   ```

2. **Learning Curve Generation**
   ```python
   sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]

   for group_name, molecules in molecules_by_group.items():
       results_standard = []
       results_equivariant = []

       for N in sample_sizes:
           # Train both model types
           for seed in range(10):
               # Standard model
               model_std = StandardModel(seed=seed)
               model_std.fit(molecules[:N])
               error_std = evaluate(model_std, test_molecules)
               results_standard.append({'N': N, 'seed': seed, 'error': error_std})

               # Equivariant model
               model_eq = EquivariantModel(point_group=group_name, seed=seed)
               model_eq.fit(molecules[:N])
               error_eq = evaluate(model_eq, test_molecules)
               results_equivariant.append({'N': N, 'seed': seed, 'error': error_eq})
   ```

3. **Power Law Fitting**
   ```python
   from scipy.optimize import curve_fit

   def power_law(N, A, alpha, B):
       return A * N**(-alpha) + B

   # Fit for each model type
   for model_type, results in [('Standard', results_standard),
                                ('Equivariant', results_equivariant)]:
       df = pd.DataFrame(results)
       grouped = df.groupby('N')['error'].mean()

       params, _ = curve_fit(power_law, grouped.index, grouped.values)
       A, alpha, B = params

       print(f"{model_type}: error(N) = {A:.3f} * N^(-{alpha:.3f}) + {B:.3f}")
   ```

4. **Statistical Analysis**
   ```python
   # Compute sample complexity ratio
   target_error = 0.05

   N_standard = ((target_error - B_std) / A_std)**(1 / -alpha_std)
   N_equivariant = ((target_error - B_eq) / A_eq)**(1 / -alpha_eq)

   ratio = N_standard / N_equivariant
   theoretical_ratio = get_point_group_order(group_name)

   print(f"Observed ratio: {ratio:.2f}")
   print(f"Theoretical ratio (|G|): {theoretical_ratio}")
   print(f"Relative error: {abs(ratio - theoretical_ratio) / theoretical_ratio:.1%}")
   ```

5. **Control for Confounders**
   ```python
   # Regression controlling for molecular properties
   from sklearn.linear_model import LinearRegression

   X = np.column_stack([
       [get_point_group_order(g) for g in group_labels],
       molecular_weights,
       num_atoms,
       num_bonds
   ])
   y = sample_complexities

   model = LinearRegression()
   model.fit(X, y)

   print("Effect of point group order (controlled):", model.coef_[0])
   ```

**Success criteria:**
- Ratio within 20% of theoretical prediction
- p < 0.05 for point group effect (controlling for confounders)

### 13.2 Protocol for Theorem 2: Centrosymmetric Synergy

**Objective:** Verify I(M; S_IR, S_Raman) has zero redundancy and positive synergy for centrosymmetric molecules

**Steps:**

1. **Data Preparation**
   ```python
   # Filter centrosymmetric molecules
   centrosym_molecules = filter_molecules(has_inversion_center=True)
   non_centrosym_molecules = filter_molecules(has_inversion_center=False)

   # Extract latent representations
   z_IR_centro, z_Raman_centro = extract_latents(model, centrosym_molecules)
   z_IR_non, z_Raman_non = extract_latents(model, non_centrosym_molecules)
   ```

2. **Dimensionality Reduction**
   ```python
   from sklearn.decomposition import PCA

   # Reduce to D=10
   pca_IR = PCA(n_components=10)
   pca_Raman = PCA(n_components=10)

   z_IR_red = pca_IR.fit_transform(z_IR_centro)
   z_Raman_red = pca_Raman.fit_transform(z_Raman_centro)

   print(f"Variance explained: {pca_IR.explained_variance_ratio_.sum():.1%}")
   ```

3. **PID Computation**
   ```python
   # Discretize for dit library
   def discretize(X, n_bins=5):
       bins = [np.percentile(X[:, i], np.linspace(0, 100, n_bins))
               for i in range(X.shape[1])]
       return np.array([np.digitize(X[:, i], bins[i]) for i in range(X.shape[1])]).T

   z_IR_disc = discretize(z_IR_red)
   z_Raman_disc = discretize(z_Raman_red)
   M_disc = discretize(molecule_features)

   # Compute PID
   from dit import Distribution, PID

   data = np.concatenate([z_IR_disc, z_Raman_disc, M_disc], axis=1)
   # ... (PID computation)

   pid = PID(...)
   synergy_centro = pid.synergy
   redundancy_centro = pid.redundancy
   ```

4. **Statistical Testing**
   ```python
   # Permutation test for synergy
   synergies_null = []
   for _ in range(10000):
       M_shuffled = np.random.permutation(M_disc)
       pid_null = compute_pid(z_IR_disc, z_Raman_disc, M_shuffled)
       synergies_null.append(pid_null.synergy)

   p_value_synergy = (np.array(synergies_null) >= synergy_centro).mean()

   # Test for zero redundancy (two-sided test)
   redundancies_null = [compute_pid(...).redundancy for _ in range(10000)]
   p_value_redundancy = 2 * min(
       (np.array(redundancies_null) <= redundancy_centro).mean(),
       (np.array(redundancies_null) >= redundancy_centro).mean()
   )
   ```

5. **Comparison with Non-Centrosymmetric**
   ```python
   # Repeat for non-centrosymmetric molecules
   pid_non = compute_pid(z_IR_non_red, z_Raman_non_red, M_non_disc)

   # Compare synergies
   from scipy.stats import mannwhitneyu
   stat, p_value = mannwhitneyu(
       bootstrap_synergies_centro,
       bootstrap_synergies_non
   )

   print(f"Centrosymmetric synergy: {synergy_centro:.3f} ± {synergy_centro_std:.3f}")
   print(f"Non-centrosymmetric synergy: {synergy_non:.3f} ± {synergy_non_std:.3f}")
   print(f"Difference p-value: {p_value:.4f}")
   ```

6. **Bootstrap CIs**
   ```python
   synergies_boot_centro = []
   redundancies_boot_centro = []

   for _ in range(1000):
       idx = np.random.choice(len(data), size=len(data), replace=True)
       pid_boot = compute_pid(z_IR_disc[idx], z_Raman_disc[idx], M_disc[idx])
       synergies_boot_centro.append(pid_boot.synergy)
       redundancies_boot_centro.append(pid_boot.redundancy)

   synergy_ci = np.percentile(synergies_boot_centro, [2.5, 97.5])
   redundancy_ci = np.percentile(redundancies_boot_centro, [2.5, 97.5])
   ```

**Success criteria:**
- Synergy > 0 with p < 0.05
- Redundancy not significantly different from 0 (|z| < 2 or CI contains 0)
- Synergy significantly higher for centrosymmetric vs non-centrosymmetric

### 13.3 Protocol for Theorem 3: Transfer Learning Bound

**Objective:** Verify ε_transfer ≤ ε_src + C·W₂(P_chem) + D·KL(P_inst)

**Steps:**

1. **Extract Latent Distributions**
   ```python
   # Source and target instruments
   z_chem_src, z_inst_src = extract_latents(model, spectra_src)
   z_chem_tgt, z_inst_tgt = extract_latents(model, spectra_tgt)
   ```

2. **Estimate W₂(P_chem)**
   ```python
   # Check Gaussian assumption
   from scipy.stats import shapiro

   is_gaussian = all(shapiro(z_chem_src[:, i])[1] > 0.05
                     for i in range(min(5, z_chem_src.shape[1])))

   if is_gaussian:
       # Bures-Wasserstein
       from ot.gaussian import bures_wasserstein_distance
       mu_src = z_chem_src.mean(0)
       Sigma_src = np.cov(z_chem_src.T)
       mu_tgt = z_chem_tgt.mean(0)
       Sigma_tgt = np.cov(z_chem_tgt.T)
       W2_chem = bures_wasserstein_distance(mu_src, Sigma_src, mu_tgt, Sigma_tgt)
   else:
       # Sliced Wasserstein
       from ot.sliced import sliced_wasserstein_distance
       W2_chem = sliced_wasserstein_distance(z_chem_src, z_chem_tgt, n_projections=100)

   # Bootstrap CI
   W2_chem_ci = wasserstein_bootstrap_ci(z_chem_src, z_chem_tgt)
   ```

3. **Estimate KL(P_inst)**
   ```python
   # Similar Gaussian check
   if is_gaussian:
       kl_inst = kl_gaussian(
           z_inst_src.mean(0), np.cov(z_inst_src.T),
           z_inst_tgt.mean(0), np.cov(z_inst_tgt.T)
       )
   else:
       kl_inst = kl_knn(z_inst_src, z_inst_tgt, k=5)

   # Bootstrap CI
   kl_inst_ci = bootstrap_ci(
       (z_inst_src, z_inst_tgt),
       lambda data: kl_knn(data[0], data[1], k=5)
   )
   ```

4. **Measure Transfer Error**
   ```python
   # Source error
   preds_src = model.predict(spectra_src_test)
   eps_src = np.sqrt(np.mean((preds_src - y_src_test)**2))

   # Transfer error (zero-shot)
   preds_tgt_zeroshot = model.predict(spectra_tgt_test)
   eps_transfer = np.sqrt(np.mean((preds_tgt_zeroshot - y_tgt_test)**2))

   # After adaptation (few-shot)
   model_adapted = model.adapt(spectra_tgt_fewshot, y_tgt_fewshot)
   preds_tgt_adapted = model_adapted.predict(spectra_tgt_test)
   eps_adapted = np.sqrt(np.mean((preds_tgt_adapted - y_tgt_test)**2))
   ```

5. **Fit Constants C, D**
   ```python
   # Collect data from multiple transfer pairs
   transfer_pairs = [
       ('mp5', 'mp6'),
       ('mp5', 'm5'),
       ('mp6', 'm5'),
       # ... more pairs
   ]

   data = []
   for src, tgt in transfer_pairs:
       eps_src_i = compute_source_error(src)
       eps_transfer_i = compute_transfer_error(src, tgt)
       W2_i = compute_wasserstein(src, tgt)
       KL_i = compute_kl(src, tgt)

       data.append({
           'eps_transfer': eps_transfer_i,
           'eps_src': eps_src_i,
           'W2': W2_i,
           'KL': KL_i
       })

   df = pd.DataFrame(data)

   # Fit: eps_transfer = eps_src + C*W2 + D*KL
   from sklearn.linear_model import LinearRegression

   X = df[['eps_src', 'W2', 'KL']].values
   y = df['eps_transfer'].values

   model_fit = LinearRegression()
   model_fit.fit(X, y)

   intercept, C, D = model_fit.intercept_, model_fit.coef_[1], model_fit.coef_[2]
   r2 = model_fit.score(X, y)

   print(f"ε_transfer = {intercept:.3f} + {C:.3f}·W₂ + {D:.3f}·KL")
   print(f"R² = {r2:.3f}")
   ```

6. **Validate Bound**
   ```python
   # Check if bound holds for all pairs
   df['bound'] = df['eps_src'] + C * df['W2'] + D * df['KL']
   df['slack'] = df['bound'] - df['eps_transfer']

   violations = (df['slack'] < 0).sum()
   print(f"Violations: {violations} / {len(df)} ({violations/len(df):.1%})")

   # Tightest bound
   print(f"Min slack: {df['slack'].min():.3f}")
   print(f"Mean slack: {df['slack'].mean():.3f}")
   ```

**Success criteria:**
- Bound holds for ≥ 95% of transfer pairs
- R² > 0.7 for fitted model
- C, D > 0 (positive coefficients)

---

## 14. Common Pitfalls and Solutions {#pitfalls}

### 14.1 Insufficient Samples

**Symptom:** Wide confidence intervals, unstable estimates, p-values > 0.05

**Diagnosis:**
```python
# Check CI width
ci_width = ci_upper - ci_lower
relative_width = ci_width / estimate

if relative_width > 0.3:
    print("WARNING: CI width > 30% of estimate")
    print("Recommendation: Collect more data or reduce dimensionality")
```

**Solutions:**
1. Increase N (collect more data)
2. Reduce D (more aggressive PCA)
3. Use parametric estimator if assumptions hold (Gaussian)
4. Pool data across conditions (if justified)

### 14.2 Wrong Assumptions

**Symptom:** Estimator gives implausible results (e.g., negative MI)

**Diagnosis:**
```python
# Check Gaussian assumption
from scipy.stats import shapiro, anderson

for i in range(min(5, X.shape[1])):
    stat, p = shapiro(X[:, i])
    if p < 0.05:
        print(f"Dimension {i}: Not Gaussian (p={p:.3f})")

# QQ plot
import matplotlib.pyplot as plt
from scipy.stats import probplot

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < X.shape[1]:
        probplot(X[:, i], dist="norm", plot=ax)
        ax.set_title(f'Dimension {i}')
plt.tight_layout()
```

**Solutions:**
1. Use robust estimator (KNN instead of Gaussian)
2. Transform data (log, Box-Cox) to make more Gaussian
3. Use non-parametric methods
4. Report assumption violations in paper

### 14.3 Discretization Artifacts

**Symptom:** PID results change dramatically with number of bins

**Diagnosis:**
```python
# Vary bin count
bin_counts = [3, 5, 7, 10, 15]
synergies = []

for n_bins in bin_counts:
    X_disc = discretize(X, n_bins=n_bins)
    pid = compute_pid(X_disc[:, 0], X_disc[:, 1], y)
    synergies.append(pid.synergy)

plt.plot(bin_counts, synergies, 'o-')
plt.xlabel('Number of bins')
plt.ylabel('Synergy (bits)')

# Check stability
if np.std(synergies) / np.mean(synergies) > 0.2:
    print("WARNING: Synergy unstable across bin counts")
```

**Solutions:**
1. Use continuous PID estimator (Gaussian PID, continuous shared exclusions)
2. Report results for multiple bin counts
3. Use adaptive binning (equal-frequency, not equal-width)
4. Increase sample size before discretizing

### 14.4 Outliers

**Symptom:** Wasserstein distance dominated by single data point

**Diagnosis:**
```python
# Detect outliers
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.05)
outliers = clf.fit_predict(X)

print(f"Outliers: {(outliers == -1).sum()} / {len(X)}")

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=outliers, cmap='coolwarm')
plt.colorbar(label='Outlier (-1) / Inlier (1)')
```

**Solutions:**
1. Remove outliers (justify in methods)
2. Use robust preprocessing (RobustScaler, winsorization)
3. Use robust distance (e.g., Sliced Wasserstein less sensitive than exact OT)
4. Report with and without outliers

### 14.5 Computational Issues

**Symptom:** POT library fails, MINE doesn't converge, KNN is too slow

**Solutions:**

**POT failures:**
```python
# Problem: Sinkhorn diverges with small reg
# Solution: Use larger reg or Sliced Wasserstein
try:
    W2 = ot.sinkhorn2(a, b, M, reg=0.01)
except:
    print("Sinkhorn failed, using Sliced Wasserstein")
    W2 = ot.sliced_wasserstein_distance(X, Y, n_projections=100)
```

**MINE non-convergence:**
```python
# Use learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# Early stopping
best_mi = -float('inf')
patience = 50
for epoch in range(1000):
    mi_estimate = train_step(...)

    if mi_estimate > best_mi:
        best_mi = mi_estimate
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
```

**KNN too slow:**
```python
# Use approximate nearest neighbors
from sklearn.neighbors import BallTree

tree = BallTree(X, metric='euclidean')
distances, _ = tree.query(X, k=k+1)
# Much faster for large N
```

### 14.6 Multiple Testing Issues

**Symptom:** Many significant results, suspicious

**Diagnosis:**
```python
# How many tests did you run?
n_tests = len(p_values)
expected_false_positives = n_tests * 0.05

print(f"Expected false positives at α=0.05: {expected_false_positives:.1f}")
print(f"Observed significant results: {(np.array(p_values) < 0.05).sum()}")
```

**Solution:**
```python
from statsmodels.stats.multitest import multipletests

# FDR correction
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Report both
for i, (p_raw, p_adj, sig) in enumerate(zip(p_values, p_adjusted, reject)):
    print(f"Test {i}: p={p_raw:.3f}, p_adj={p_adj:.3f}, significant={sig}")
```

---

## 15. Summary: Complete Validation Pipeline

### 15.1 End-to-End Checklist

**Before experiments:**
- [ ] Install all required libraries
- [ ] Validate estimators on synthetic data
- [ ] Determine required sample size via power analysis
- [ ] Choose dimensionality reduction (PCA to D=10-20)

**For each theorem:**
- [ ] Extract/prepare data
- [ ] Check assumptions (Gaussian, etc.) with QQ plots
- [ ] Compute point estimate
- [ ] Compute bootstrap CI (1000 iterations)
- [ ] Run permutation test (≥1000 permutations)
- [ ] Multiple hypothesis correction if needed
- [ ] Visualize (learning curves, distributions, etc.)

**Reporting:**
- [ ] Table with estimates, CIs, N, estimator, p-values
- [ ] Figure with error bars/shaded regions
- [ ] Methods section describing all choices
- [ ] Supplementary with validation, sensitivity analysis
- [ ] Code repository (GitHub/Zenodo)

### 15.2 Timeline Estimate

**For a single theorem (e.g., Theorem 2 on synergy):**

| Task | Time | Notes |
|------|------|-------|
| Data preparation | 1-2 days | Extract latents, filter molecules |
| Synthetic validation | 1 day | Test PID on XOR/AND gates |
| Dimensionality selection | 0.5 days | PCA scree plot, variance explained |
| Main computation | 1 day | PID with bootstrap |
| Permutation testing | 0.5 days | 10000 permutations |
| Sensitivity analysis | 1 day | Vary bins, estimators, D |
| Visualization | 0.5 days | Plots, tables |
| Writing | 1-2 days | Methods, results sections |
| **Total** | **6-9 days** | For one experienced researcher |

**For all three theorems + baselines:** ~1 month

### 15.3 Recommended Order of Execution

1. **Theorem 3 (Transfer bound)** — Easiest, uses standard MI/Wasserstein
2. **Theorem 1 (Sample complexity)** — Medium difficulty, requires many training runs
3. **Theorem 2 (Synergy)** — Hardest, PID is most finicky

---

## 16. Key References

### Mutual Information Estimation

1. [Beyond Normal: On the Evaluation of Mutual Information Estimators](https://proceedings.neurips.cc/paper_files/paper/2023/file/36b80eae70ff629d667f210e13497edf-Paper-Conference.pdf) (NeurIPS 2023)
2. [Mutual Information Estimation via Normalizing Flows](https://proceedings.neurips.cc/paper_files/paper/2024/file/05a2d9ef0ae6f249737c1e4cce724a0c-Paper-Conference.pdf) (NeurIPS 2024)
3. [MINE: Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (ICML 2018)
4. [A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula](https://pmc.ncbi.nlm.nih.gov/articles/PMC5324576/) (Human Brain Mapping 2017)
5. [Estimating Mutual Information via Geodesic kNN](https://arxiv.org/pdf/2110.13883) (2021)
6. [Improving Numerical Stability of Normalized Mutual Information Estimator on High Dimensions](https://arxiv.org/html/2410.07642v1) (2024)

### Partial Information Decomposition

7. [Gaussian Partial Information Decomposition: Bias Correction](https://proceedings.neurips.cc/paper_files/paper/2023/file/ec0bff8bf4b11e36f874790046dfdb65-Paper-Conference.pdf) (NeurIPS 2023)
8. [Partial Information Decomposition: Redundancy as Information Bottleneck](https://www.mdpi.com/1099-4300/26/7/546) (Entropy 2024)
9. [Partial information decomposition for continuous variables based on shared exclusions](https://link.aps.org/doi/10.1103/PhysRevE.110.014115) (Phys. Rev. E 2024)
10. [Partial information decomposition for mixed discrete and continuous random variables](https://arxiv.org/html/2409.13506) (2024)
11. [dit: discrete information theory](https://dit.readthedocs.io/)
12. [IDTxl: The Information Dynamics Toolkit xl](https://github.com/pwollstadt/IDTxl) (JOSS 2019)
13. [Synergy, redundancy, and multivariate information measures](http://www.beggslab.com/uploads/1/0/1/7/101719922/29timmeetal2013.pdf)

### Optimal Transport and Wasserstein Distance

14. [POT: Python Optimal Transport](https://pythonot.github.io/) (JMLR 2021)
15. [Sliced Wasserstein Estimation with Control Variates](https://proceedings.iclr.cc/paper_files/paper/2024/file/08f628998ca37c9df8c6a0df3570db86-Paper-Conference.pdf) (ICLR 2024)
16. [GeomLoss: Geometric Loss functions](https://www.kernel-operations.io/geomloss/)
17. [Wasserstein distance between two Gaussians](https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/)

### Sample Complexity and Learning Curves

18. [Explaining neural scaling laws](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228526/) (PNAS 2024)
19. [On the Sample Complexity of One Hidden Layer Networks with Equivariance](https://arxiv.org/html/2411.14288) (2024)
20. [E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials](https://www.nature.com/articles/s41467-022-29939-5) (Nature Comm. 2022)
21. [Equivariant score-based generative models provably learn distributions with symmetries efficiently](https://arxiv.org/html/2410.01244) (2024)

### Statistical Testing and Validation

22. [Null models for comparing information decomposition across complex systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC12614810/) (PLOS Comp Bio 2024)
23. [Review about the Permutation Approach in Hypothesis Testing](https://www.mdpi.com/2227-7390/12/17/2617) (Mathematics 2024)
24. [Bootstrap confidence intervals: A comparative simulation study](https://arxiv.org/html/2404.12967v1) (2024)
25. [Discrete Information Dynamics with Confidence via the Computational Mechanics Bootstrap](https://pmc.ncbi.nlm.nih.gov/articles/PMC7517337/)

### Dimensionality Reduction

26. [Data Efficiency, Dimensionality Reduction, and the Generalized Symmetric Information Bottleneck](https://direct.mit.edu/neco/article-abstract/36/7/1353/120664/) (Neural Computation 2024)
27. [Towards One Model for Classical Dimensionality Reduction: A Probabilistic Perspective on UMAP and t-SNE](https://arxiv.org/html/2405.17412v1) (2024)
28. [ACCURATE ESTIMATION OF MUTUAL INFORMATION IN HIGH DIMENSIONAL DATA](https://arxiv.org/pdf/2506.00330) (2024)

### Reporting Standards

29. [Sample Size Analysis for Machine Learning Clinical Validation Studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC10045793/)
30. [Improving statistical reporting in psychology](https://www.nature.com/articles/s44271-025-00356-w) (2025)

---

## Appendix: Quick Reference Card

### Estimator Selection Matrix

| Quantity | D | N | Assumption | Estimator | Library |
|----------|---|---|------------|-----------|---------|
| MI | ≤20 | >1000 | None | KSG | sklearn |
| MI | ≤50 | >1000 | Gaussian copula | GCMI | gcmi |
| MI | >50 | >5000 | None | MINE | Custom |
| PID | ≤10 | >1000 | Gaussian | Gaussian PID | gcmi |
| PID | ≤10 | >1000 | None | I_broja/I_ccs | dit |
| PID | >10 | >2000 | Continuous | Shared exclusion | Custom |
| W₂ | ≤20 | <1000 | None | Exact OT | POT |
| W₂ | ≤20 | >1000 | None | Sinkhorn | POT |
| W₂ | >20 | >1000 | None | Sliced W | POT |
| W₂ | Any | >100 | Gaussian | Bures-W | POT |
| KL | ≤20 | >1000 | None | k-NN | Custom |
| KL | Any | >100 | Gaussian | Closed-form | Custom |

### Default Hyperparameters

| Estimator | Parameter | Default | Range | Notes |
|-----------|-----------|---------|-------|-------|
| KSG | k | 5 | 3-10 | Larger k = less variance, more bias |
| GCMI | - | - | - | No hyperparameters |
| Sinkhorn | reg | 0.01 | 0.001-0.1 | Smaller = less bias, slower |
| Sliced W | n_proj | 100 | 50-1000 | More = less variance |
| k-NN KL | k | 5 | 3-10 | Same as KSG |
| Bootstrap | n_iter | 1000 | 500-5000 | More = better CI |
| Permutation | n_perm | 1000 | 1000-10000 | For p<0.001, use 10000 |
| PCA | n_comp | 10 | 5-20 | Choose via scree plot |
| Discretize | n_bins | 5 | 3-10 | Fewer bins = less overfitting |

### Python Code Snippets

```python
# COMPLETE VALIDATION PIPELINE IN 50 LINES

import numpy as np
from sklearn.decomposition import PCA
from gcmi import gcmi_cc
from ot.sliced import sliced_wasserstein_distance
from scipy.stats import shapiro

# 1. Load data
X, y = load_latent_representations()  # (N, D), (N,)

# 2. Check Gaussian assumption
is_gaussian = all(shapiro(X[:, i])[1] > 0.05 for i in range(min(5, X.shape[1])))

# 3. Dimensionality reduction
pca = PCA(n_components=10)
X_red = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# 4. Estimate MI
mi_est = gcmi_cc(X_red, y)

# 5. Bootstrap CI
mi_boot = []
for _ in range(1000):
    idx = np.random.choice(len(X_red), size=len(X_red), replace=True)
    mi_boot.append(gcmi_cc(X_red[idx], y[idx]))
mi_ci = np.percentile(mi_boot, [2.5, 97.5])

# 6. Permutation test
mi_null = []
for _ in range(1000):
    y_shuf = np.random.permutation(y)
    mi_null.append(gcmi_cc(X_red, y_shuf))
p_value = (np.array(mi_null) >= mi_est).mean()

# 7. Report
print(f"MI: {mi_est:.3f} bits")
print(f"95% CI: [{mi_ci[0]:.3f}, {mi_ci[1]:.3f}]")
print(f"p-value: {p_value:.4f}")

# 8. Wasserstein distance (if comparing two distributions)
X_src, X_tgt = load_two_distributions()
if is_gaussian:
    from ot.gaussian import bures_wasserstein_distance
    mu1, Sigma1 = X_src.mean(0), np.cov(X_src.T)
    mu2, Sigma2 = X_tgt.mean(0), np.cov(X_tgt.T)
    W2 = bures_wasserstein_distance(mu1, Sigma1, mu2, Sigma2)
else:
    W2 = sliced_wasserstein_distance(X_src, X_tgt, n_projections=100)

print(f"W₂: {W2:.3f}")
```

---

**Document version:** 1.0
**Last updated:** 2026-02-10
**Contact:** See PROJECT_STATUS.md for research team
