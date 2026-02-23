# Competitor Analysis: ML Models for Spectrum-to-Structure Identification
# Reference Document for "Can One Hear the Shape of a Molecule?"
# Last updated: 2026-02-11

---

## 1. DiffSpectra (2025)

**Paper:** "DiffSpectra: Diffusion-based molecular generation from IR and Raman spectra"
**arXiv:** 2507.06853
**Approach:** Diffusion-based generative model that reconstructs molecular graphs from simulated IR and Raman spectra.

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 40.76% |
| Top-10 accuracy | 99.49% |
| Dataset | QM9S subset (~100K molecules) |
| Input modalities | IR + Raman (simulated) |
| Molecular representation | 2D molecular graph |

**Architecture:** Conditional diffusion model. Encodes IR and Raman spectra jointly, then generates molecular graphs via iterative denoising. Uses a GNN decoder for graph generation.

**Key insight:** The large gap between top-1 (40.76%) and top-10 (99.49%) suggests the model captures the right "neighborhood" of molecules but struggles to pinpoint the exact structure. This is consistent with our theoretical prediction: molecules within a confusable set share similar spectra, and the model correctly identifies the confusable set but cannot resolve within it.

**Strengths:**
- Generative approach can propose novel structures not in the training database
- Multi-modal (IR + Raman)
- Near-perfect top-10 accuracy

**Weaknesses:**
- Low top-1 accuracy (40.76%)
- No theoretical framework for understanding failure cases
- No symmetry awareness
- No uncertainty quantification

**Our advantage:** We explain WHY top-1 accuracy is low (information completeness ratio R(G,N) < 1 for symmetric molecules; confusable sets). Our theory predicts which molecules DiffSpectra will fail on.

---

## 2. Vib2Mol (2025)

**Paper:** "Vib2Mol: A multi-task framework for vibrational spectrum-to-molecule translation"
**arXiv:** 2503.07014
**Approach:** Multi-task framework combining coarse-to-fine retrieval with molecular property prediction and SMILES generation.

| Metric | Value |
|--------|-------|
| Top-10 accuracy (retrieval) | ~87% (base), 98.1% (with reranking) |
| Top-1 accuracy | Not reported separately |
| Dataset | QM9S, custom enlarged set |
| Input modalities | IR + Raman |
| Output | Molecular fingerprint → SMILES |

**Architecture:** Two-stage pipeline. Stage 1: Coarse retrieval using spectral fingerprints (CNN encoder → embedding → nearest-neighbor search). Stage 2: Fine reranking using multi-task heads that predict molecular properties (formula, weight, functional groups) alongside structure.

**Key insight:** The coarse-to-fine approach implicitly addresses the confusable set problem: the coarse retrieval identifies a candidate set (analogous to our confusable set), then fine reranking uses auxiliary predictions to disambiguate.

**Strengths:**
- Multi-task learning provides auxiliary discriminants
- Coarse-to-fine retrieval is efficient and interpretable
- High top-10 accuracy with reranking

**Weaknesses:**
- Requires molecular property labels for training
- No theoretical justification for why multi-task helps
- No symmetry-aware evaluation
- Performance degrades significantly without reranking

**Our advantage:** We provide theoretical justification for why retrieval works (generic identifiability for most molecules) and why reranking helps (confusable pairs need additional discriminants beyond spectral similarity).

---

## 3. VibraCLIP (2025)

**Paper:** "VibraCLIP: Contrastive learning for vibrational spectroscopy and molecular structure"
**Published in:** RSC Digital Discovery
**Approach:** Contrastive learning (CLIP-style) between spectral embeddings and molecular structure embeddings.

| Metric | Value |
|--------|-------|
| Top-1 accuracy (with mass) | 81.7% |
| Top-1 accuracy (no mass) | ~65% (estimated) |
| Top-25 accuracy (with mass) | 98.9% |
| Dataset | QM9S |
| Input modalities | IR + Raman + molecular mass (auxiliary) |
| Molecular representation | SELFIES encoding |

**Architecture:** Dual-encoder contrastive learning. Spectral encoder (Transformer-based) maps spectra to shared embedding space; molecular encoder (SELFIES tokenizer + Transformer) maps structures to same space. Training maximizes cosine similarity for matched spectrum-structure pairs, minimizes for mismatched pairs.

**Key insight:** Providing molecular mass as an auxiliary "hint" dramatically improves top-1 accuracy (~65% → 81.7%). This is consistent with our framework: molecular mass eliminates many confusable pairs (molecules of different mass cannot be confusable), effectively reducing the confusable set size K and thus the Fano lower bound.

**Strengths:**
- Highest reported top-1 accuracy (81.7%)
- Contrastive learning provides interpretable similarity metric
- Efficient retrieval via pre-computed embeddings

**Weaknesses:**
- Requires molecular mass as auxiliary input (not always available)
- CLIP-style training requires large batch sizes
- No symmetry-aware design
- SELFIES representation limits structural diversity

**Our advantage:** We explain WHY mass helps (reduces confusable set cardinality), and our theory predicts that mass should help more for symmetric molecules (where confusable sets are larger).

---

## 4. DetaNet (2024-2025)

**Paper:** Introduced the QM9S dataset. E(3)-equivariant neural network for predicting vibrational spectra from 3D molecular structure (forward direction).
**Approach:** Forward model (structure → spectrum), not inverse. Used as baseline in QM9S benchmark.

| Metric | Value |
|--------|-------|
| Forward prediction MAE | ~10-30 cm⁻¹ (frequencies) |
| Dataset | QM9S (130K molecules) |
| Direction | Forward (structure → spectrum) |
| Equivariance | E(3)-equivariant |

**Relevance to our work:**
- Introduced QM9S as the standard benchmark
- E(3) equivariance validates our G-invariance framework (Theorem 1)
- Forward model accuracy sets upper bound on inverse problem tractability
- If forward prediction has ~10 cm⁻¹ error, inverse must contend with this uncertainty

---

## 5. SpectrumWorld (2025)

**Paper:** Comprehensive benchmark for spectral analysis tasks.
**Approach:** Multi-task benchmark spanning 14 spectroscopic tasks across multiple modalities.

| Feature | Value |
|---------|-------|
| Tasks | 14 spectral analysis tasks |
| Substances | 1.2M unique substances |
| Modalities | IR, Raman, NMR, MS, UV-Vis |
| Task types | Identification, property prediction, classification |

**Architecture:** Benchmark framework, not a single model. Evaluates multiple architectures (CNN, Transformer, GNN) across all tasks.

**Key insight:** Establishes that spectrum-to-structure is among the hardest spectral tasks, with accuracy varying significantly by molecular complexity. This is consistent with our theoretical prediction that identification difficulty depends on symmetry (via R(G,N)).

**Relevance:** Our experiments should be benchmarked on SpectrumWorld tasks where applicable, particularly the IR/Raman identification subtasks. The large-scale nature (1.2M substances) provides a stress test beyond QM9S.

---

## 6. MolSpectLLM (2025)

**Paper:** LLM-based approach to spectral interpretation.
**Approach:** Fine-tunes Qwen2.5-7B for multi-modal spectroscopic reasoning (IR + NMR + MS).

| Metric | Value |
|--------|-------|
| Average accuracy | 0.53 across all tasks |
| Dataset | Multi-modal spectral datasets |
| Base model | Qwen2.5-7B |
| Modalities | IR, NMR, MS (text-based descriptions) |

**Architecture:** LLM fine-tuned with spectral data represented as text (peak positions, intensities as tokenized sequences). Multi-task training on identification, functional group prediction, and structure elucidation.

**Key insight:** LLM-based approaches achieve modest accuracy (0.53), suggesting that the spectrum-to-structure problem benefits more from physics-informed architectures than from general-purpose language modeling. The spectral inverse problem has intrinsic mathematical structure (group theory, symmetry) that is not easily captured by text-based reasoning.

**Relevance:** Represents the "LLM for everything" approach. Our physics-informed model should significantly outperform this baseline, and our theoretical framework explains why: spectroscopy requires representation-theoretic structure that LLMs cannot learn from text alone.

---

## 7. Additional Relevant Models

### IBM Transformer (2025)
- 63.8% top-1, 83.9% top-10
- Pure transformer on IR spectra
- No multi-modal fusion

### SMAE / RamanMAE (2024-2025)
- Self-supervised spectral models
- Masked autoencoder pretraining for spectral representations
- Foundation model approach (similar to our pretraining strategy)
- Not specifically targeting spectrum-to-structure

---

## Comparative Summary

| Model | Top-1 | Top-10 | Theory | Multi-modal | Symmetry | Uncertainty |
|-------|-------|--------|--------|-------------|----------|-------------|
| DiffSpectra | 40.8% | 99.5% | No | Yes (IR+R) | No | No |
| Vib2Mol | — | 98.1% | No | Yes (IR+R) | No | No |
| VibraCLIP | 81.7%* | — | No | Yes (IR+R+mass) | No | No |
| IBM Transformer | 63.8% | 83.9% | No | No (IR only) | No | No |
| MolSpectLLM | ~53% | — | No | Yes (IR+NMR+MS) | No | No |
| **Ours** | **TBD** | **TBD** | **Yes** | **Yes (IR+R)** | **Yes** | **Yes** |

*With molecular mass as auxiliary input

## Our Unique Positioning

1. **Only model with theoretical framework**: We are the first to provide formal identifiability theory (Theorems 1-2, Conjecture 3) that explains why models succeed and fail.

2. **Symmetry-aware evaluation**: No prior work stratifies results by molecular point group. Our R(G,N) framework provides the first symmetry-aware ML benchmark.

3. **Theory predicts empirical findings of competitors**:
   - DiffSpectra's 40% top-1 vs. 99% top-10 → confusable sets
   - VibraCLIP's mass hint boost → confusable set reduction
   - All models' difficulty with symmetric molecules → low R(G,N)

4. **Uncertainty quantification**: Conformal prediction with Fano-bound-informed calibration. No competitor offers provable coverage guarantees.

5. **Theory-guided architecture**: VIB disentanglement and cross-attention fusion are designed based on Theorem 2 (modal complementarity), not ad hoc choices.

---

## Key Unanswered Questions from Competitor Analysis

1. **Why does no model exceed ~82% top-1?** Our answer: confusable molecular pairs + silent modes impose a fundamental ceiling for QM9S molecules (prediction from Theorem 1 and Proposition 1).

2. **Why does multi-modal help?** Our answer: Theorem 2 — for centrosymmetric molecules, IR and Raman observe disjoint mode sets.

3. **Will accuracy ever reach 100%?** Our answer: Conjecture 3 says generically yes (for C₁ molecules with full spectra). But for the QM9S distribution (containing ~5% centrosymmetric molecules with R < 1), a ceiling exists.

4. **Which molecules are fundamentally hard?** Our answer: Those with low R(G,N) (many silent modes) and those in large confusable sets. We provide the first method to predict difficulty a priori.
