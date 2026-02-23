# Section 4: Methods

We now describe the computational framework used to validate the theoretical predictions of Section 3. Our approach has three components: (i) a symmetry-aware encoder architecture that maps vibrational spectra to a disentangled latent representation, (ii) a multi-objective training procedure that leverages both self-supervised and physics-informed learning signals, and (iii) an evaluation protocol designed to test the specific predictions of Theorems 1 and 2 and Conjecture 3. Throughout, we emphasize design choices motivated by the theory --- in particular, the multi-modal fusion mechanism (motivated by Theorem 2), the symmetry-stratified evaluation (motivated by Theorem 1), and the retrieval-based decoder with conformal prediction guarantees (motivated by Proposition 1).

## 4.1 Model Architecture

### 4.1.1 Overview

The model receives a vibrational spectrum $S \in \mathbb{R}^{L}$ (with $L = 2048$ after resampling; see Section 4.2.2) and produces a latent representation $z_{\text{chem}} \in \mathbb{R}^{128}$ intended to encode the chemical identity of the source molecule while discarding instrument-specific artifacts. Molecular identification is then performed by nearest-neighbor retrieval in latent space against a reference database of known molecules. The full architecture, depicted in Figure 7, consists of four stages:

$$S \;\xrightarrow{\text{Embedding}}\; \mathbf{T} \in \mathbb{R}^{(N+2) \times d} \;\xrightarrow{\text{Encoder}}\; \mathbf{h} \in \mathbb{R}^{d} \;\xrightarrow{\text{VIB}}\; (z_{\text{chem}}, z_{\text{inst}}) \;\xrightarrow{\text{Retrieval}}\; \hat{m} \in \mathcal{M}, \tag{1}$$

where $N$ is the number of spectral tokens, $d = 512$ is the model dimension, and VIB denotes the variational information bottleneck (Section 4.1.4).

### 4.1.2 Spectral Embedding

Raw spectra are tokenized via a hybrid wavelet-convolutional embedding that captures multi-scale spectral features. This design is motivated by the structure of vibrational spectra, which contain information at multiple characteristic scales: narrow peaks (individual vibrational modes, width $\sim$5--20 cm$^{-1}$), medium features (composite band envelopes, $\sim$50--200 cm$^{-1}$), and broad baselines ($\sim$500+ cm$^{-1}$).

**Wavelet decomposition.** The input spectrum $S \in \mathbb{R}^{L}$ is first decomposed via the discrete wavelet transform (DWT) using the Daubechies-4 (db4) wavelet at $J = 4$ levels, yielding one approximation coefficient vector $c_{A_4}$ and four detail coefficient vectors $\{c_{D_j}\}_{j=1}^{4}$. The approximation coefficients capture the spectral baseline and broad features, while the detail coefficients at successive levels encode progressively finer spectral features --- from broad shoulders ($c_{D_4}$) down to the sharpest peaks ($c_{D_1}$). This decomposition is computed via the PyWavelets library [22] and is not itself differentiable; the model learns from the wavelet coefficients rather than through the decomposition.

**Patch embedding.** In parallel, the raw spectrum is tokenized via a 1D convolutional layer with kernel size $p = 32$ and stride $s = 16$, producing $N = \lfloor(L - p)/s\rfloor + 1 = 127$ patch tokens, each projected to dimension $d$. Each wavelet coefficient vector is linearly interpolated to match the $N$-token sequence length, projected to dimension $d/(J+1)$, and concatenated with the patch tokens. A fusion layer (linear projection followed by layer normalization, GELU activation, and dropout) maps the concatenated representation to the final token dimension $d$.

**Positional encoding.** Rather than standard sinusoidal or learned positional encodings indexed by token position, we employ a physics-informed wavenumber encoding. If the wavenumber axis $\{\tilde{\nu}_1, \ldots, \tilde{\nu}_N\}$ is available, each token receives a positional encoding computed as a learned linear projection of its associated wavenumber value $\tilde{\nu}_i$:

$$\text{PE}(i) = W_{\text{pe}} \cdot \tilde{\nu}_i + b_{\text{pe}}, \qquad W_{\text{pe}} \in \mathbb{R}^{d}, \; b_{\text{pe}} \in \mathbb{R}^{d}. \tag{2}$$

This encoding ensures that the model is aware of the physical frequency associated with each token, which is essential for interpreting selection rules and comparing spectra measured at different resolutions or over different wavenumber ranges. When wavenumber information is unavailable, the model falls back to standard sinusoidal positional encoding.

**Special tokens.** Two special tokens are prepended to the sequence: a classification token [CLS], whose output serves as the global spectral representation, and a domain token [DOMAIN], drawn from a learned embedding table indexed by spectral modality (IR, Raman, NIR, or unknown). The domain token enables the shared encoder to condition its processing on the spectral type, which is important because IR and Raman spectra encode fundamentally different physical observables (dipole derivatives versus polarizability derivatives; see Section 2.1).

### 4.1.3 Encoder

The encoder maps the token sequence $\mathbf{T} \in \mathbb{R}^{(N+2) \times d}$ to a contextualized representation from which the [CLS] token output $\mathbf{h} \in \mathbb{R}^{d}$ is extracted as the global spectral embedding. We employ a hybrid architecture consisting of a 1D convolutional backbone for local feature extraction followed by a transformer for global reasoning.

**1D CNN tokenizer.** A stack of five 1D convolutional layers (channel progression $1 \to 32 \to 64 \to 128 \to 256 \to 512$, kernel size 3, GELU activations, batch normalization) processes the embedded tokens to build hierarchical local features. Each layer applies a convolution, activation, and optional max-pooling step. This CNN tokenizer has been shown to improve performance by 8--10\% over directly applying a transformer to raw spectral tokens [23], and it is particularly well-suited to vibrational spectra where local features (individual absorption bands) carry much of the discriminative information.

**Transformer encoder.** The CNN output is passed to a transformer encoder consisting of $L_T = 4$ layers, each with $H = 8$ attention heads, model dimension $d = 512$, feedforward dimension $d_{\text{ff}} = 2048$, and pre-norm residual connections [24]. The transformer provides global receptive field over the full spectrum, enabling the model to capture long-range correlations between distant spectral features --- for instance, the correlation between C-H stretching modes near 3000 cm$^{-1}$ and C-H bending modes near 1450 cm$^{-1}$ in organic molecules.

**Multi-modal fusion.** For joint IR+Raman identification --- the setting most directly relevant to Theorem 2 --- both spectra are processed by a shared encoder with modality-specific domain tokens:

$$\mathbf{h}_{\text{IR}} = \text{Encoder}(S_{\text{IR}}, \text{domain}=\text{IR}), \qquad \mathbf{h}_{\text{Raman}} = \text{Encoder}(S_{\text{Raman}}, \text{domain}=\text{Raman}).$$

The two embeddings are fused via a cross-attention mechanism:

$$\mathbf{h}_{\text{fused}} = \text{CrossAttention}(\mathbf{h}_{\text{IR}}, \mathbf{h}_{\text{Raman}}) = \text{softmax}\!\left(\frac{Q_{\text{IR}} K_{\text{Raman}}^{\top}}{\sqrt{d_k}}\right) V_{\text{Raman}}, \tag{3}$$

where $Q_{\text{IR}} = W_Q \mathbf{h}_{\text{IR}}$, $K_{\text{Raman}} = W_K \mathbf{h}_{\text{Raman}}$, $V_{\text{Raman}} = W_V \mathbf{h}_{\text{Raman}}$ (and symmetrically for the reverse direction). The bidirectional cross-attention outputs are averaged to produce $\mathbf{h}_{\text{fused}}$. Weight sharing between the IR and Raman encoders encourages the model to learn modality-invariant spectral features while the domain tokens and cross-attention allow modality-specific processing. This design is motivated by Theorem 2: for centrosymmetric molecules, the IR and Raman spectra probe strictly disjoint sets of vibrational modes, so the cross-attention mechanism must learn to combine genuinely complementary information rather than averaging redundant features.

### 4.1.4 Variational Information Bottleneck

The global representation $\mathbf{h}$ (or $\mathbf{h}_{\text{fused}}$ in the multi-modal case) is passed through a variational information bottleneck (VIB) [25] that disentangles the latent space into two components:

$$\mathbf{h} \;\xrightarrow{}\; (\boldsymbol{\mu}_{\text{chem}}, \boldsymbol{\sigma}_{\text{chem}}) \;\xrightarrow{\text{reparam.}}\; z_{\text{chem}} \in \mathbb{R}^{128}, \tag{4}$$
$$\mathbf{h} \;\xrightarrow{}\; (\boldsymbol{\mu}_{\text{inst}}, \boldsymbol{\sigma}_{\text{inst}}) \;\xrightarrow{\text{reparam.}}\; z_{\text{inst}} \in \mathbb{R}^{64}. \tag{5}$$

Here, $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are produced by separate linear projections, and the reparameterization trick [26] is used for differentiable sampling: $z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. At inference time, $z = \boldsymbol{\mu}$ (the mean is used deterministically).

The latent variables are trained with three objectives:

1. **Task loss.** The chemistry latent $z_{\text{chem}}$ should be predictive of molecular identity: $\mathcal{L}_{\text{task}}(z_{\text{chem}})$.

2. **KL regularization.** Both latent distributions are regularized toward a standard normal prior:

$$\mathcal{L}_{\text{KL}} = \text{KL}(q(z_{\text{chem}} | S) \| p(z)) + \text{KL}(q(z_{\text{inst}} | S) \| p(z)), \tag{6}$$

where $p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and the KL divergence has the closed-form expression $\text{KL} = -\frac{1}{2} \sum_{j} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$.

3. **Adversarial disentanglement.** An adversarial classifier attempts to predict the instrument identity from $z_{\text{chem}}$; the encoder is trained to make this prediction fail (i.e., to push the instrument-conditioned distribution of $z_{\text{chem}}$ toward uniformity over instruments):

$$\mathcal{L}_{\text{adv}} = \text{KL}\left(\text{softmax}(f_{\text{adv}}(z_{\text{chem}})) \;\|\; \mathcal{U}\right), \tag{7}$$

where $f_{\text{adv}}$ is a two-layer MLP classifier and $\mathcal{U}$ is the uniform distribution over instrument classes. Simultaneously, a separate classifier is trained to predict instrument identity from $z_{\text{inst}}$, ensuring that instrument-specific information is captured in $z_{\text{inst}}$ rather than lost entirely.

The combined VIB loss is:

$$\mathcal{L}_{\text{VIB}} = \mathcal{L}_{\text{task}}(z_{\text{chem}}) + \beta \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adv}} + \lambda_{\text{inst}} \cdot \mathcal{L}_{\text{inst\_cls}}, \tag{8}$$

with $\beta = 10^{-3}$ (annealed from 0 during warmup to prevent posterior collapse), $\lambda_{\text{adv}} = 0.1$, and $\lambda_{\text{inst}} = 0.1$.

We emphasize that the VIB disentanglement is a *practical engineering tool*, not a theoretically guaranteed separation. The decomposition into $z_{\text{chem}}$ and $z_{\text{inst}}$ is encouraged by the training objectives but is not provably achieved; residual instrument information may persist in $z_{\text{chem}}$, and some chemical information may leak into $z_{\text{inst}}$. The degree of disentanglement is an empirical quantity that we measure and report in Section 5.

### 4.1.5 Retrieval Decoder and Conformal Prediction

Molecular identification is performed via retrieval rather than generation. Given a query spectrum $S$ with encoded latent $z_{\text{chem}}$, we compute cosine similarity against a reference database $\mathcal{D} = \{(z_{\text{chem}}^{(i)}, m_i)\}_{i=1}^{|\mathcal{D}|}$ of precomputed latent representations for known molecules:

$$\text{sim}(S, m_i) = \frac{z_{\text{chem}} \cdot z_{\text{chem}}^{(i)}}{\|z_{\text{chem}}\| \; \|z_{\text{chem}}^{(i)}\|}, \tag{9}$$

and return the top-$k$ most similar molecules ranked by this score. This retrieval-based approach has several advantages over generative decoders (e.g., diffusion-based molecular generation [4]): it guarantees that all returned molecules are chemically valid (they exist in the database), it provides a natural confidence score (the similarity margin between the top-1 and top-2 candidates), and it is interpretable (the user can inspect the retrieved candidates and their spectral matches).

To provide formal uncertainty quantification, we wrap the retrieval system with split conformal prediction [27]. Given a calibration set of $n_{\text{cal}}$ labeled examples, we define the nonconformity score for a query spectrum $S$ with true molecule $m$ as:

$$\alpha(S, m) = 1 - \text{sim}(S, m), \tag{10}$$

and construct a prediction set $\mathcal{C}(S)$ at confidence level $1 - \varepsilon$ by including all molecules whose nonconformity score does not exceed the $\lceil (1 - \varepsilon)(1 + n_{\text{cal}}) \rceil / n_{\text{cal}}$ quantile of the calibration scores. By the exchangeability guarantee of conformal prediction [27], this procedure satisfies the finite-sample coverage property:

$$\mathbb{P}(m^* \in \mathcal{C}(S)) \geq 1 - \varepsilon, \tag{11}$$

for any new test point $(S, m^*)$, with no distributional assumptions beyond exchangeability. We evaluate coverage at $\varepsilon = 0.1$ (90\% confidence) in Section 5.5.

### 4.1.6 Model Scale

The full model contains approximately 85M parameters. In the single-modality (IR-only or Raman-only) configuration, the encoder processes one spectrum per forward pass; in the multi-modal configuration, two spectra are processed through the shared encoder and fused via cross-attention, roughly doubling the computational cost of the encoder stage. All components are implemented in PyTorch 2.x.

## 4.2 Training

### 4.2.1 Pretraining Objectives

The model is pretrained using a multi-objective self-supervised framework consisting of three complementary learning signals.

**Masked spectral reconstruction (MSR).** Following the masked autoencoding paradigm [28], we randomly mask 20\% of spectral tokens using contiguous blocks of 3 tokens each (to prevent trivial interpolation from adjacent unmasked tokens) and train the model to reconstruct the masked regions:

$$\mathcal{L}_{\text{MSR}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|S_i^{\text{pred}} - S_i^{\text{true}}\|_2^2, \tag{12}$$

where $\mathcal{M}$ denotes the set of masked token indices. This objective forces the encoder to learn the internal structure of vibrational spectra --- the correlations between peak positions, intensities, and widths that reflect underlying molecular physics.

**Contrastive learning.** When paired spectra from different instruments or modalities are available for the same molecule, we employ an InfoNCE contrastive loss [29] on the chemistry latent:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_{\text{chem}}^{(1)}, z_{\text{chem}}^{(2)}) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(z_{\text{chem}}^{(1)}, z_{\text{chem}, j}^{(2)}) / \tau)}, \tag{13}$$

where $(z_{\text{chem}}^{(1)}, z_{\text{chem}}^{(2)})$ is a positive pair (same molecule, different instruments or modalities), $B$ is the batch size, and $\tau = 0.1$ is the temperature. This objective encourages instrument-invariant representations: the same molecule measured on different instruments should map to similar latent points. For the IR+Raman setting, it also encourages cross-modal alignment --- a key ingredient for leveraging the complementarity established by Theorem 2.

**Denoising.** Spectra are corrupted with Gaussian noise ($\sigma = 0.01$), baseline drift (random low-order polynomial trends), and wavenumber jitter ($\pm 3$ cm$^{-1}$), and the model is trained to reconstruct the clean spectrum:

$$\mathcal{L}_{\text{denoise}} = \|S_{\text{clean}} - f_\theta(S_{\text{noisy}})\|_2^2. \tag{14}$$

This objective builds robustness to the spectral artifacts commonly encountered in experimental data.

**Physics-informed regularization.** Four soft physics constraints are applied to the reconstructed spectra:

- *Smoothness*: penalizes high total variation, $\mathcal{L}_{\text{smooth}} = \frac{1}{L-1}\sum_{i=1}^{L-1}(S_{i+1} - S_i)^2$;
- *Non-negativity*: penalizes negative absorbance values, $\mathcal{L}_{\text{nn}} = \frac{1}{L}\sum_i \text{ReLU}(-S_i)$;
- *Derivative smoothness*: penalizes rough second derivatives, encouraging smooth Lorentzian/Gaussian peak shapes;
- *Peak symmetry*: at detected local maxima, penalizes asymmetry between left and right shoulders.

These are soft regularizers weighted by small coefficients (0.02--0.05) and are not intended to encode hard physical constraints. They encode prior knowledge that vibrational spectra consist of smooth, non-negative, approximately symmetric peaks --- a weak but broadly applicable inductive bias.

**Combined pretraining loss.** The total pretraining objective is:

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{MSR}} + 0.3 \cdot \mathcal{L}_{\text{contrast}} + 0.2 \cdot \mathcal{L}_{\text{denoise}} + 0.1 \cdot \mathcal{L}_{\text{physics}} + 0.05 \cdot \mathcal{L}_{\text{VIB}} + 0.01 \cdot \mathcal{L}_{\text{MoE}}, \tag{15}$$

where $\mathcal{L}_{\text{MoE}}$ is a load-balancing loss that encourages uniform utilization across expert sub-networks (described in Supplementary Section S5), and $\mathcal{L}_{\text{VIB}}$ is defined in Equation (8). When multi-instrument data is available, an optimal transport alignment term $\mathcal{L}_{\text{OT}}$ (Sinkhorn divergence [30] between latent distributions of different instruments, weighted by 0.1) is added to encourage instrument-invariant representations.

### 4.2.2 Data

**Primary dataset: QM9S.** Our primary evaluation dataset is QM9S [31], which contains 130,831 small organic molecules (up to 9 heavy atoms: C, N, O, F) with simulated IR and Raman spectra computed at the B3LYP/def2-TZVP level of density functional theory. Each molecule has 3,000 spectral points covering the range 0--4,000 cm$^{-1}$, which we resample to $L = 2048$ points via linear interpolation to obtain uniform input dimensionality. QM9S is uniquely suited to our purposes because (a) it provides both IR and Raman spectra for every molecule, enabling multi-modal experiments; (b) the molecules span a range of point group symmetries, enabling symmetry-stratified evaluation; and (c) the spectra are computed from first principles, eliminating experimental noise and instrument variation as confounding factors in the identifiability analysis.

**Pretraining corpus.** For self-supervised pretraining, we augment QM9S with the ChEMBL IR-Raman dataset (approximately 220,000 computed spectra) [32]. All spectra are resampled to $L = 2048$ points, intensity-normalized to $[0, 1]$, and stored in HDF5 format with molecular identifiers, point group labels, and SMILES strings as metadata. The combined pretraining corpus contains approximately 350,000 spectra.

**Data augmentation.** During pretraining, we apply stochastic augmentations to each spectrum: multiplicative intensity scaling (uniformly sampled from $[0.95, 1.05]$), additive Gaussian noise (SNR $\in [50, 200]$), random baseline drift (linear and quadratic polynomial trends with coefficients $\sim \mathcal{N}(0, 0.005)$), and wavenumber jitter ($\pm 3$ cm$^{-1}$ uniform offset). These augmentations simulate the dominant sources of experimental variability (instrument response functions, detector noise, and wavelength calibration drift) and encourage the model to learn representations that are robust to these artifacts.

### 4.2.3 Optimization

Pretraining uses AdamW [33] with learning rate $3 \times 10^{-4}$, weight decay 0.01, and gradient clipping at norm 1.0. The learning rate follows a linear warmup over 1,000 steps followed by cosine decay over the remaining steps. The VIB coefficient $\beta$ is annealed linearly from 0 to $10^{-3}$ over 5,000 steps to prevent posterior collapse early in training [26]. Training proceeds for 50,000 steps with batch size 64. Mixed-precision training (FP16) is used throughout.

### 4.2.4 Fine-tuning via LoRA

For downstream tasks (property prediction, calibration transfer), the pretrained encoder is adapted via Low-Rank Adaptation (LoRA) [34]. LoRA injects trainable low-rank decompositions into the query, key, and value projection matrices of the transformer attention layers:

$$W' = W_{\text{frozen}} + \frac{\alpha}{r} B A, \tag{16}$$

where $W_{\text{frozen}} \in \mathbb{R}^{d \times d}$ is the frozen pretrained weight, $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are the trainable low-rank matrices, $r = 8$ is the rank, and $\alpha = 16$ is a scaling factor. $B$ is initialized to zero, so the model output is unchanged at the start of fine-tuning. This approach adapts only $\sim$0.5\% of the total parameters while preserving the pretrained representations, and is particularly suited to the low-data transfer settings we consider (as few as 5--10 labeled samples).

### 4.2.5 Test-Time Training

For zero-shot calibration transfer --- adaptation to a new instrument without any labeled data from that instrument --- we employ test-time training (TTT) [35]. Given a batch of unlabeled spectra $\{S_i\}_{i=1}^{N_{\text{test}}}$ from the target instrument, the model performs $K$ steps of self-supervised adaptation using the masked reconstruction objective (Equation 12) on the test data alone:

$$\theta^* = \theta - \eta \sum_{t=1}^{K} \nabla_\theta \mathcal{L}_{\text{MSR}}(S_{\text{test}}; \theta), \tag{17}$$

with $K = 10$ steps, learning rate $\eta = 10^{-4}$, mask ratio 0.15, and adaptation restricted to normalization layer parameters (to prevent catastrophic forgetting of pretrained knowledge). This procedure enables the model to adapt its internal representations to the statistical properties of the new instrument's spectra without requiring any labeled data --- a form of zero-shot calibration transfer.

### 4.2.6 Hardware

All experiments are conducted on a workstation equipped with 4 NVIDIA RTX 5090 GPUs (32 GB VRAM each). Multi-GPU training uses PyTorch DataParallel. Pretraining the full model on the 350K-spectrum corpus requires approximately 36 hours. Fine-tuning with LoRA requires 10--30 minutes per configuration.

## 4.3 Evaluation Protocol

### 4.3.1 Molecular Identification Metrics

**Top-$k$ accuracy.** The primary metric is top-$k$ retrieval accuracy: the fraction of test molecules for which the correct molecular structure appears among the $k$ nearest neighbors in the reference database. We report $k \in \{1, 5, 10, 25\}$. Top-1 accuracy measures exact identification; top-10 and top-25 measure whether the correct molecule is within a manageable shortlist for expert review, which is the practical use case in analytical chemistry.

**Tanimoto similarity.** For cases where exact identification fails, we measure the structural similarity between the predicted (top-1 retrieved) molecule and the true molecule using the Tanimoto coefficient on Morgan fingerprints (radius 2, 2048 bits) [36]:

$$T(m_{\text{pred}}, m_{\text{true}}) = \frac{|F(m_{\text{pred}}) \cap F(m_{\text{true}})|}{|F(m_{\text{pred}}) \cup F(m_{\text{true}})|}, \tag{18}$$

where $F(m)$ denotes the fingerprint bit-set. A high Tanimoto similarity ($T > 0.85$) indicates that the prediction is structurally close to the true molecule even when exact identification fails --- an important distinction for practical applications.

### 4.3.2 Symmetry-Stratified Evaluation (Theorem 1 Validation)

To test the prediction of Theorem 1 --- that the information completeness ratio $R(G, N)$ should correlate with identification accuracy --- we stratify the test set by molecular point group. For each point group $G$ represented in the dataset, we compute $R(G, N)$ from character tables (as described in Section 3.2) and the corresponding top-$k$ accuracy. We then fit a linear regression:

$$\text{Accuracy}(G) = a \cdot R(G, N) + b, \tag{19}$$

and report the coefficient of determination $R^2$ between the predicted and observed accuracy across symmetry classes. A strong positive correlation ($R^2 > 0.5$) would constitute empirical support for Theorem 1's prediction that symmetry-induced information loss limits identification performance.

### 4.3.3 Complementarity Gain (Theorem 2 Validation)

To test Theorem 2's prediction that multi-modal fusion should yield a larger accuracy gain for centrosymmetric molecules, we train three model variants: IR-only, Raman-only, and IR+Raman (fused). For each variant and for each molecule in the test set, we record the identification outcome. We then compute the complementarity gain:

$$\Delta_{\text{comp}}(G) = \text{Acc}_{\text{IR+Raman}}(G) - \max(\text{Acc}_{\text{IR}}(G), \text{Acc}_{\text{Raman}}(G)), \tag{20}$$

separately for centrosymmetric point groups ($C_i$, $C_{2h}$, $D_{2h}$, $D_{3d}$, $D_{4h}$, $D_{6h}$, $D_{\infty h}$, $O_h$) and non-centrosymmetric point groups. Theorem 2 predicts $\Delta_{\text{comp}}(\text{centrosymmetric}) > \Delta_{\text{comp}}(\text{non-centrosymmetric})$, because for centrosymmetric molecules the second modality provides access to an entirely non-overlapping set of vibrational modes (by the mutual exclusion rule), whereas for non-centrosymmetric molecules both modalities probe largely overlapping mode sets.

We test this prediction via a two-sample $t$-test on $\Delta_{\text{comp}}$ between the centrosymmetric and non-centrosymmetric subgroups, with significance level $\alpha = 0.05$.

### 4.3.4 Confusable Set Analysis (Proposition 1 Validation)

To evaluate the sharpness of the Fano lower bound (Proposition 1), we construct confusable molecular sets from the QM9S test split as follows. For each pair of molecules $(m_i, m_j)$ in the test set, we compute the spectral distance $d_{\text{spec}}(\Phi(m_i), \Phi(m_j))$ (Euclidean distance between $\ell_2$-normalized spectral vectors) and the structural distance $d_{\text{struct}}(m_i, m_j) = 1 - T(m_i, m_j)$ (complement of Tanimoto similarity). A confusable pair is one satisfying $d_{\text{spec}} < \varepsilon$ and $d_{\text{struct}} > \Delta$; we scan over $\varepsilon \in \{0.01, 0.05, 0.1\}$ and $\Delta \in \{0.3, 0.5, 0.7\}$.

For each identified confusable set $\mathcal{C} = \{m_1, \ldots, m_K\}$, we compute the Fano bound (Proposition 1) by estimating $I(M; S_{\text{IR}}, S_{\text{Raman}})$ from the within-set spectral variance, and compare it to the empirical identification error of the model on $\mathcal{C}$. Agreement between the empirical error and the Fano prediction would indicate that the model is approaching the fundamental information-theoretic limits of spectral identification for these difficult cases.

### 4.3.5 Conformal Prediction Coverage

We evaluate the conformal prediction wrapper (Section 4.1.5) by measuring empirical coverage at the 90\% confidence level ($\varepsilon = 0.1$). Good calibration requires that the empirical coverage be close to 90\% --- specifically, within the interval $[88\%, 92\%]$ for our test set sizes. We also report the average prediction set size, which measures the informativeness of the uncertainty estimates (smaller sets are more useful).

### 4.3.6 Jacobian Rank Computation (Conjecture 3 Evidence)

As computational evidence for Conjecture 3 (generic identifiability), we compute the numerical rank of the Jacobian $\mathbf{J} = \partial \Phi / \partial \mathbf{F}$ for a random sample of 1,000 molecules from the QM9S dataset. For each molecule, we perturb each force constant parameter $F_{ij}$ by $\delta = 10^{-4}$ and recompute the observables via finite differences to construct the Jacobian matrix. The numerical rank is determined by counting singular values exceeding $10^{-8}$ times the largest singular value. We report the distribution of $\text{rank}(\mathbf{J}) / d_F$ across the sample, where $d_F$ is the number of force constant parameters. Full rank ($\text{rank}(\mathbf{J}) = d_F$) at a given molecular configuration is consistent with local injectivity of the forward map $\Phi$ at that point.

### 4.3.7 Baselines

We compare against the following methods:

- **DiffSpectra** [4]: a diffusion-based generative model for molecular graph reconstruction from IR and Raman spectra, representing the current state of the art for top-1 accuracy on QM9S (40.76\% top-1, 99.49\% top-10);
- **Vib2Mol** [5]: a multi-task encoder-decoder framework combining retrieval and molecular property prediction ($\sim$87\% top-10, 98.1\% with reranking);
- **VibraCLIP** [6]: a contrastive learning approach aligning spectral and molecular representations (81.7\% top-1 with molecular mass hint, 98.9\% top-25);
- **MolSpectLLM** [19]: a large language model-based approach (0.53 average accuracy on standard benchmarks);
- **Random baseline**: uniform random retrieval from the reference database, providing a lower bound.

All baselines are evaluated on the same QM9S test split. Where possible, we use published results; otherwise, we reproduce models using publicly available code and default hyperparameters.

---

## References

[22] G. R. Lee, R. Gommers, F. Wasilewski, K. Wohlfahrt, A. O'Leary, "PyWavelets: A Python package for wavelet analysis", *Journal of Open Source Software* **4**(36), 1237 (2019).

[23] S. Li, Y. Jin, "Spectral transformers: Learning long-range dependencies in vibrational spectra", in *Proceedings of the AAAI Conference on Artificial Intelligence* (2024).

[24] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, "Attention is all you need", in *Advances in Neural Information Processing Systems* **30**, 5998--6008 (2017).

[25] A. A. Alemi, I. Fischer, J. V. Dillon, K. Murphy, "Deep variational information bottleneck", in *Proceedings of the International Conference on Learning Representations* (2017).

[26] D. P. Kingma, M. Welling, "Auto-encoding variational Bayes", in *Proceedings of the International Conference on Learning Representations* (2014).

[27] V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer (2005). See also: A. N. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification", *Foundations and Trends in Machine Learning* **16**(4), 494--591 (2023).

[28] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, R. Girshick, "Masked autoencoders are scalable vision learners", in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16000--16009 (2022).

[29] A. van den Oord, Y. Li, O. Vinyals, "Representation learning with contrastive predictive coding", arXiv:1807.03748 (2018).

[30] M. Cuturi, "Sinkhorn distances: Lightspeed computation of optimal transport", in *Advances in Neural Information Processing Systems* **26**, 2292--2300 (2013).

[31] K. Liang, S. Liao, Y. Guo, "QM9S: A dataset of vibrational spectra for 130K molecules from quantum mechanical calculations", *Scientific Data* (2023). See also: DetaNet, arXiv:2303.09394.

[32] ChEMBL IR-Raman computed spectral database. Available at https://www.ebi.ac.uk/chembl/.

[33] I. Loshchilov, F. Hutter, "Decoupled weight decay regularization", in *Proceedings of the International Conference on Learning Representations* (2019).

[34] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, "LoRA: Low-rank adaptation of large language models", in *Proceedings of the International Conference on Learning Representations* (2022).

[35] Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, M. Hardt, "Test-time training with self-supervision for generalization under distribution shifts", in *Proceedings of the International Conference on Machine Learning*, 9229--9248 (2020).

[36] D. Rogers, M. Hahn, "Extended-connectivity fingerprints", *Journal of Chemical Information and Modeling* **50**(5), 742--754 (2010).
