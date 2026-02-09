# SpectralFM — Comprehensive Reference Document
## Full Details for Paper Development

---

## FULL COMPETITOR ANALYSIS

### Mass Spec Foundation Models (Different domain, but key references)

**DreaMS (Nature Biotechnology 2025)**
- Bushuiev et al.
- 24M mass spectra, self-supervised
- Structure elucidation task
- Proves foundation model concept works for spectral data
- OUR ANGLE: We do the same for vibrational spectroscopy

**PRISM (Enveda, 2025)**
- 1.2B mass spectra, 85B tokens
- Largest spectral foundation model
- Metabolomics applications
- OUR ANGLE: Different domain, but validates approach at scale

**DSCF (Nature Machine Intelligence 2025)**
- Zero-shot spectral denoising via foundation model
- Mass spec only
- OUR ANGLE: Shows foundation models can do zero-shot tasks on spectra

### Vibrational Spectroscopy ML (Closest competitors)

**LoRA-CT — Lai et al., Anal. Chem. 2025, 97(35), 19009-19018** ★ PRIMARY DL BASELINE ★
- Low-Rank Adaptation for Raman calibration transfer
- Key results: R² = 0.952 on methanol mixture (vs PDS 0.846, full FT 0.863)
- 600× fewer parameters than full fine-tuning
- Tested on: solvent mixtures, blended oils, 3 spectrometers
- Plug-and-play LoRA modules for different instruments
- LIMITATION: No self-supervised pretraining — starts from supervised model on source instrument
- OUR ADVANTAGE: Pretrained representations already encode spectral physics → better sample efficiency

**RamanMAE — bioRxiv 2025.05.18.654618**
- Masked autoencoder for Raman spectroscopy
- Biological imaging: cancer cells, tissue classification, denoising
- Self-supervised pretraining → fine-tuning for classification
- NOT calibration transfer
- OUR ADVANTAGE: CT task, multi-modality, foundation model scale

**SMAE — Ren et al., Expert Syst. Appl. 2025 (arXiv 2504.16130)**
- Spectral Masked Autoencoder for Raman bacterial classification
- Bacteria-ID dataset, 30 classes
- Random masking → reconstruction → fine-tune for classification
- NOT calibration transfer
- OUR ADVANTAGE: CT task, much larger pretraining corpus, physics-informed design

**BDSER-InceptionNet — Chen et al., Sensors 2025, 25(13), 4008**
- CNN with SE attention + Balanced Distribution Adaptation (BDA)
- Tested on corn and tablet datasets
- Outperforms PLS, SVR, traditional CNNs in cross-instrument prediction
- LIMITATION: No self-supervised pretraining, CNN architecture (no long-range dependencies)
- OUR ADVANTAGE: Transformer + self-supervised pretraining

**ACT — Huang et al., AAAI 2025**
- Analytical Chemistry-Informed Transformer
- Baseline correction, not foundation model for CT
- OUR ADVANTAGE: Different task focus entirely

**Vib2Mol — Lu et al., arXiv 2503.07014 (v4 Jan 2026)**
- Encoder-decoder transformer
- Spectrum → molecular structure prediction
- IR and Raman, NIST experimental data (12K spectra)
- OUR ADVANTAGE: Different task (forward prediction vs calibration transfer)

**MACE4IR — Bhatia et al., arXiv 2508.19118**
- Foundation model for IR using MACE architecture
- Molecule → spectrum prediction
- OUR ADVANTAGE: Different task (forward vs inverse/transfer)

**Federated NIR — CILS Jan 2026**
- Decentralized FL for cross-instrument NIR
- 30 clients, FedProx regularization
- 52% error reduction in cross-instrument tasks
- Corn and Gasoline datasets
- OUR ADVANTAGE: Centralized foundation model, self-supervised pretraining, different paradigm

### Classical Baselines

**di-PLS — Nikzad-Langerodi et al., Anal. Chem. 2018, 90(11), 6693-6701**
- Extends PLS with domain-invariant regularizer
- Aligns source/target distributions in latent space
- Supports unsupervised/semi-supervised/supervised adaptation
- Python: diPLSlib (GitHub: B-Analytics/diPLSlib)
- KEY: Include as baseline — strong classical method

**mdi-PLS — Mikulasek et al., J. Chemom. 2023, 37(5), e3477**
- Generalizes di-PLS to multiple source domains
- Also in diPLSlib

---

## ANALYTICAL CHEMISTRY FORMATTING GUIDE

### Article Structure (based on recent ML papers in AC)
1. Abstract (150-250 words)
2. Introduction (1.5-2 pages)
   - Problem statement (calibration transfer cost/limitations)
   - Current approaches and limitations
   - Foundation models in other domains (DreaMS, PRISM)
   - Our contribution (3 bullet points)
3. Methods (2-2.5 pages)
   - Pretraining architecture
   - Masking strategy
   - Fine-tuning approach
   - Baseline methods
4. Results and Discussion (3-3.5 pages)
   - Pretraining effectiveness (E1)
   - Sample efficiency (E3 — key figure)
   - Baseline comparison (E4)
   - Cross-modality transfer (E5)
   - Interpretability (E6)
5. Conclusions (0.5 page)
6. Associated Content (code/data availability)
7. References (~40-60)

### ACS Reference Format
Author1; Author2; Author3. Title. *Journal* **Year**, *Volume* (Issue), Pages. DOI.

### TOC Graphic
- Size: 8.25 cm × 4.45 cm
- Should show: spectrum + transformer + transfer arrow + result
- Clean, colorful, communicates key idea at a glance

---

## REVIEWER OBJECTION MATRIX

### Chemometrics Reviewer Objections

| Objection | Response | Experiment |
|-----------|----------|------------|
| "Why not just use more samples?" | Each sample = $50-200 lab time; we reduce from 50+ to 10 | E3: sample efficiency curve |
| "PDS works fine" | PDS fails at N<20; we show failure cases | E4: comparison at low N |
| "Computed spectra aren't real" | NIST/RRUFF experimental data included; ablation shows both help | E5 + ablation |
| "Only tested on corn/tablet — too narrow" | 3 datasets, multiple analytes, Raman + NIR | E4 across all datasets |
| "How handle different spectral ranges?" | Wavenumber PE + resampling to common grid | Architecture design |

### ML Reviewer Objections

| Objection | Response | Experiment |
|-----------|----------|------------|
| "Just BERT for spectra" | Physics-informed PE, contiguous masking, spectral augmentation | E2: masking ablation |
| "Corpus too small (400K vs 24M)" | Vib. spec. chemical space smaller; scaling curve shows diminishing returns | Corpus size ablation |
| "Why transformer not CNN?" | Long-range dependencies (overtones); ablation shows transformer > CNN | E4: CNN baseline |
| "Overfitting with 10 samples?" | LoRA (50K params, not 8M); pretrained representations generalize | E1 + E3 |
| "No theoretical justification" | Domain adaptation theory (Ben-David et al.); pretraining reduces domain gap | Discussion section |

### AC-Specific Objections

| Objection | Response |
|-----------|----------|
| "Incremental improvement" | FIRST foundation model for vib. spec. CT; paradigm shift from classical → foundation model |
| "Not enough real-world impact" | Pharmaceutical QC, food safety, environmental monitoring all need this |
| "Black box" | Attention maps, GradCAM show model attends to chemistry (E6) |
| "No code/data" | GitHub repo with full code + all benchmark data publicly available |

---

## FIELD CONTEXT (For Introduction)

### Key Quotes to Reference

From Spectroscopy Online Dec 2025:
- "foundation-scale spectroscopy AI systems" identified as emerging paradigm
- "generalist spectral LLMs capable of zero-shot interpretation across modalities" as future
- "cross-instrument transfer using foundation models" called out as needed

From Spectroscopy Online Jan/Feb 2026:
- "challenges in data quality and calibration transferability remain"
- "2025 marked a turning point" — AI becoming core enabler, not ancillary tool

From Chemical Science review (Westermayr & Marquetand 2025):
- "foundational models will dramatically advance simulation of optical spectra"
- "first attempt for a foundational model for IR spectroscopy" (MACE4IR) noted as "early stages"

### Key Research Groups to Know
- Nikzad-Langerodi (TU Wien): di-PLS, domain adaptation for spectroscopy
- Mishra (CSIRO/Wageningen): Deep learning for NIR, transfer learning review
- Ren group (Xiamen): Vib2Mol, spectral structure prediction
- Enveda Biosciences: PRISM foundation model (mass spec)
- Bushuiev group: DreaMS (mass spec foundation model)

---

## IMPLEMENTATION PRIORITY ORDER

1. Download corn dataset (smallest, most used benchmark)
2. Implement SpectralBERT encoder (the core)
3. Implement MSRP pretraining with ChEMBL data
4. Implement PDS baseline on corn
5. Implement LoRA fine-tuning
6. Run E3 (sample efficiency) — this tells us if the approach works
7. If E3 looks good → full experimental suite
8. If E3 disappoints → diagnose and iterate
