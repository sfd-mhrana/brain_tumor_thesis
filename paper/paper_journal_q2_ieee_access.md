# Attention-Enhanced EfficientNet with CBAM for Brain Tumor MRI Classification:
# A Benchmarking and Ablation Study

**Target:** IEEE Access / Computers in Biology and Medicine / Biomedical Signal Processing and Control (Q2)
**Format:** Full research article · ~6,000–8,000 words · 30–40 references · Structured abstract

> **STATUS NOTE (delete before submission):** Numbers in all tables are
> literature-consistent placeholders. Run `scripts/02`–`07`, replace every
> *(placeholder)* value with your own results, then remove this note.

---

## Abstract

**Background:** Accurate classification of brain tumour type from magnetic resonance
imaging (MRI) is essential for treatment planning. Deep learning has achieved strong
results but existing studies typically compare only a few architectures under
inconsistent protocols, rarely isolate the contribution of individual design choices,
and almost never report statistical significance.

**Methods:** We benchmark ten deep-learning configurations — a from-scratch CNN,
six ImageNet-pretrained backbones (VGG16, ResNet50, InceptionV3, DenseNet121,
EfficientNetB0/B3), a ResNet50+SE attention variant, a soft-voting ensemble, and
the proposed **EfficientNetB3 + CBAM** — on the 7,023-image Kaggle Brain Tumor MRI
dataset (four classes: glioma, meningioma, pituitary, no-tumour). All models share
one CLAHE preprocessing pipeline, one stratified 70/15/15 split, identical
augmentation, and two-stage AdamW/focal-loss training. Six single-factor ablation
studies isolate the contribution of preprocessing, augmentation, fine-tuning
strategy, attention type, loss function, and optimiser.

**Results (placeholder):** EfficientNetB3+CBAM achieved **98.1 %** accuracy and
**97.7 %** macro-F1, significantly outperforming the strongest baseline
(EfficientNetB3, 96.9 % macro-F1; McNemar *p* < 0.05). Ablations confirmed additive
contributions from all six components. Grad-CAM++ verified tumour-focused attention.

**Conclusion:** A lightweight attention enhancement to EfficientNetB3, trained with
a reproducible two-stage focal-loss protocol, sets a competitive benchmark.
Code and split CSVs are publicly released.

**Keywords:** brain tumour; MRI classification; deep learning; EfficientNet; CBAM;
transfer learning; attention mechanism; focal loss; Grad-CAM; ablation study.

---

## 1. Introduction

Brain tumours — principally gliomas, meningiomas, and pituitary adenomas — require
different treatment pathways, making accurate pre-surgical classification from MRI
a high-stakes clinical task. Manual MRI reading is time-consuming and subject to
inter-observer variability; in many regions, radiologist shortages further delay
diagnosis. Automated deep learning classification offers fast, consistent, scalable
support for radiologists.

Convolutional neural networks (CNNs) and transfer learning have driven large
accuracy improvements on brain-MRI benchmarks. Yet three weaknesses recur in the
published literature: (i) most comparisons cover only two to four architectures
under slightly different training conditions, making conclusions unreliable;
(ii) ablation analysis is rare — few papers isolate how much each component
(preprocessing, attention, loss design) contributes; and (iii) statistical
significance testing is almost universally absent.

This paper addresses all three by holding the entire pipeline fixed and varying
only the model, running six single-factor ablations, and validating key results
with McNemar's test and 5-fold cross-validation.

**Contributions:**
1. A **fair 10-model benchmark** under one identical pipeline.
2. A **proposed EfficientNetB3 + CBAM** model achieving state-of-the-art
   single-model performance on the 4-class Kaggle dataset.
3. **Six controlled ablations** isolating each component's contribution.
4. **Grad-CAM++ visualisation** and **statistical testing** (McNemar + CV).

---

## 2. Related Work

### 2.1 Transfer learning for brain-MRI classification

Fine-tuning ImageNet-pretrained CNNs became the dominant approach because
brain-MRI datasets are small relative to natural image corpora. Deepak and Ameer
[10] showed GoogLeNet features transfer well; Swati et al. [11] demonstrated
block-wise VGG19 fine-tuning. Sultan et al. and Sajjad et al. reported 95–98 %
accuracy with ResNet/Inception ensembles. EfficientNet [4] extended these results
with significantly better accuracy-per-parameter through compound scaling.

### 2.2 Attention mechanisms

SE blocks [2] reweight feature channels via global pooling and a bottleneck MLP.
CBAM [3] extends SE with a complementary spatial-attention gate that highlights
tumour-relevant locations. Both have been applied to medical imaging with modest
but consistent gains (+0.5–2 % accuracy), though direct comparisons under identical
conditions are rare.

### 2.3 Class imbalance and loss design

Focal loss [5] concentrates gradient on hard misclassified examples by modulating
cross-entropy with (1–pₜ)ᵞ. It has been shown to improve recall on
under-represented or visually ambiguous classes (here, meningioma).

### 2.4 Research gap

Few studies (i) compare six or more architectures under identical preprocessing
and splits, (ii) report ablations isolating each component, or (iii) confirm
results with significance testing. This paper targets all three gaps.

*Table 1 summarises representative prior work (fill from your literature matrix).*

**Table 1.** Representative related work on brain-MRI classification.

| Study | Year | Dataset | Method | Accuracy | Limitation |
|---|---|---|---|---|---|
| Deepak & Ameer [10] | 2019 | CE-MRI / 3 cls | GoogLeNet TL | 97.1 % | single dataset |
| Swati et al. [11] | 2019 | CE-MRI / 4 cls | VGG19 fine-tune | 94.8 % | 1 model, no stats |
| [fill] | 2022 | Kaggle / 4 cls | EfficientNet | ~97 % | no ablation |
| **This work** | 2026 | Kaggle / 4 cls | EfficientNetB3+CBAM | *(your result)* | — |

---

## 3. Methodology

### 3.1 Dataset

The Kaggle Brain Tumor MRI dataset (Nickparvar [12]) contains 7,023 T1-weighted
contrast MRI images across four classes: glioma (1,621), meningioma (1,645),
pituitary (1,757), and no-tumour (2,000). A single stratified split —
**70 % train / 15 % val / 15 % test** — is generated once (`random_state = 42`)
and shared by all models. Duplicates are removed by SHA-256 hashing.

### 3.2 Preprocessing

Each image is resized to 224 × 224. **CLAHE** (clip limit 2.0, 8×8 tiles) is
applied to the L channel of the LAB colour space to enhance tumour boundary contrast.
Grayscale MRIs are replicated to three channels; all images are normalised with
ImageNet mean/std. Raw and CLAHE versions are stored separately for ablation AB-1.

### 3.3 Augmentation

Training-only augmentation uses anatomically valid transforms: rotation (±15°),
translation (±8 %), zoom (±10 %), brightness jitter (±15 %), and horizontal flip.
Vertical flips and heavy distortion are excluded (anatomically invalid for brain MRI).

### 3.4 Architectures

**Table 2.** Architectures benchmarked.

| # | Model | Params (M) | Role |
|---|---|---|---|
| 1 | Custom CNN | ~1 | performance floor |
| 2 | VGG16 | 138 | classic baseline |
| 3 | ResNet50 | 25.6 | core baseline |
| 4 | InceptionV3 | 23 | multi-scale baseline |
| 5 | DenseNet121 | 8 | data-efficient baseline |
| 6 | EfficientNetB0 | 5.3 | efficient baseline |
| 7 | EfficientNetB3 | 12 | strong baseline |
| 8 | ResNet50 + SE | 26 | channel-attention control |
| 9 | **EfficientNetB3 + CBAM** | 13 | **proposed** |
| 10 | Ensemble (top-3) | — | upper bound |

All transfer-learning models share a classifier head: GAP → BN → Dense(512) →
Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax).

### 3.5 Proposed model: EfficientNetB3 + CBAM

CBAM [3] is inserted after the final convolutional feature map of EfficientNetB3,
before global pooling. It applies channel attention (shared MLP over average- and
max-pooled channel descriptors, sigmoid-gated) followed by spatial attention
(channel-wise pooling → 7 × 7 convolution → sigmoid). The module adds ~1 M
parameters (~8 % overhead). Ablation AB-4 confirms that gains come from the
attention mechanism, not the additional capacity.

### 3.6 Training protocol

Two-stage fine-tuning: Stage 1 freezes the backbone and trains the head (10 epochs,
LR 1e-3); Stage 2 unfreezes the top-30 layers and fine-tunes the whole network
(up to 30 epochs, LR 1e-5). Shared settings for all models: AdamW (weight decay
1e-4), cosine LR annealing, batch size 32, label smoothing 0.1, focal loss
(γ = 2.0, α = 0.25) + inverse-frequency class weighting, early stopping
(val_loss, patience 12), gradient clipping (max-norm 1.0), seed 42.

### 3.7 Evaluation

Primary metric: **macro-F1** (weights all four classes equally under imbalance).
Also reported: accuracy, macro precision/recall, weighted-F1, per-class F1,
AUC-ROC (OvR), Cohen's κ, confusion matrix, inference latency (ms), and
parameter count. McNemar's test (exact, α = 0.05) compares each model pair;
5-fold stratified CV provides mean ± SD macro-F1 with 95 % confidence intervals
for the top models.

---

## 4. Results

> Values below are **placeholders** — replace with your experimental results.

### 4.1 Overall comparison

**Table 3.** Test-set performance, all models *(placeholder)*.

| Model | Accuracy | Macro-F1 | AUC-ROC | κ | Params (M) | Lat (ms) |
|---|---|---|---|---|---|---|
| Custom CNN | 0.885 | 0.870 | 0.943 | 0.845 | 1.0 | 8 |
| VGG16 | 0.942 | 0.936 | 0.981 | 0.921 | 138 | 25 |
| InceptionV3 | 0.951 | 0.946 | 0.987 | 0.934 | 23 | 14 |
| DenseNet121 | 0.957 | 0.952 | 0.989 | 0.941 | 8 | 13 |
| ResNet50 | 0.961 | 0.956 | 0.991 | 0.946 | 25.6 | 15 |
| EfficientNetB0 | 0.966 | 0.962 | 0.993 | 0.952 | 5.3 | 10 |
| ResNet50 + SE | 0.969 | 0.965 | 0.994 | 0.956 | 26 | 16 |
| EfficientNetB3 | 0.973 | 0.969 | 0.995 | 0.961 | 12 | 12 |
| **EffNetB3+CBAM (ours)** | **0.981** | **0.977** | **0.998** | **0.971** | 13 | 14 |
| Ensemble (top-3) | 0.984 | 0.981 | 0.998 | 0.977 | — | 40 |

EfficientNet architectures occupy the Pareto-optimal region of accuracy vs.
parameter count. SE attention (+0.9 pp over ResNet50) and CBAM (+0.8 pp over
EfficientNetB3) demonstrate consistent attention gains.

### 4.2 Ablation studies

**Table 4.** Six single-factor ablations — macro-F1 *(placeholder)*.

| Ablation | Baseline condition | Macro-F1 | Proposed | Macro-F1 | Δ |
|---|---|---|---|---|---|
| AB-1 Preprocessing | No CLAHE | 0.961 | CLAHE | 0.977 | +0.016 |
| AB-2 Augmentation | None | 0.948 | Standard | 0.977 | +0.029 |
| AB-3 Fine-tuning | 1-stage | 0.960 | 2-stage | 0.977 | +0.017 |
| AB-4 Attention | No attention | 0.969 | CBAM | 0.977 | +0.008 |
| AB-5 Loss | Cross-entropy | 0.966 | Focal | 0.977 | +0.011 |
| AB-6 Optimiser | SGD | 0.962 | AdamW | 0.977 | +0.015 |

All six components contribute positively; no component is redundant. CLAHE and
augmentation show the largest absolute gains (AB-1: +1.6 pp; AB-2: +2.9 pp).

### 4.3 Per-class and attention analysis

Meningioma is the hardest class across all models (lowest per-class recall,
highest confusion with glioma). CBAM's spatial attention produces the largest
improvement in meningioma recall (+0.9 pp over EfficientNetB3), and focal loss
provides an additional +1.0 pp meningioma recall gain (AB-5), consistent with
training emphasis on hard examples.

**Table 5.** Attention ablation on EfficientNetB3 *(placeholder)*.

| Attention type | Macro-F1 | Meningioma recall | Δ vs. none |
|---|---|---|---|
| None | 0.969 | 0.946 | — |
| SE (channel only) | 0.973 | 0.955 | +0.004 |
| CBAM (channel + spatial) | 0.977 | 0.964 | +0.008 |

### 4.4 Explainability

Grad-CAM++ activations for the proposed model concentrate on tumour tissue across
all four classes; for the ResNet50 baseline, activations are more diffuse and
occasionally attend to scanner artefacts. The 4 × 3 Grad-CAM++ grid (classes ×
original/ResNet50/proposed; Figure [fill]) provides qualitative evidence that CBAM
improves spatial focus on clinically relevant anatomy.

### 4.5 Statistical significance

McNemar's test confirms that the proposed model significantly outperforms
EfficientNetB3 (*p* < 0.05) and all weaker baselines (*p* < 0.01 or better).
5-fold CV: EfficientNetB3 = 0.965 ± 0.003 (95 % CI: [0.961, 0.969]);
proposed = **0.972 ± 0.003** (95 % CI: **[0.968, 0.976]**) — non-overlapping
intervals confirm a robust improvement.

---

## 5. Comparison with State of the Art

**Table 6.** Comparison with published work on the Kaggle 4-class dataset *(fill
from your literature review)*.

| Study | Year | Method | Accuracy | Macro-F1 |
|---|---|---|---|---|
| [fill] | 2021 | ResNet50 TL | ~96 % | — |
| [fill] | 2022 | EfficientNetB3 | ~97 % | — |
| [fill] | 2023 | ViT fine-tune | ~97.5 % | — |
| **This work** | 2026 | EfficientNetB3+CBAM | **98.1 %** | **97.7 %** |

---

## 6. Discussion

The proposed model's gains are attributable to the combination of CLAHE (boundary
enhancement), two-stage fine-tuning (preserves low-level ImageNet representations),
focal loss (mitigates meningioma underperformance), and CBAM (spatial tumour focus).
Ablations show the effects are additive, not redundant. The ensemble adds only
+0.4 pp over the single proposed model while tripling inference latency, suggesting
the proposed model is the better deployment choice.

**Limitations:** single dataset; 2D slice-level only; no clinical validation;
limited scanner diversity. Cross-dataset generalisation (Figshare CE-MRI) and
prospective clinical evaluation are recommended future steps.

---

## 7. Conclusion

We benchmarked ten deep-learning configurations on the 4-class Kaggle Brain Tumor
MRI dataset under a rigorously identical protocol and proposed EfficientNetB3+CBAM
as a new state-of-the-art single model. Six ablation studies isolate individual
design contributions. McNemar's test and 5-fold CV confirm statistical significance.
Grad-CAM++ verifies tumour-focused spatial attention.

**Future work:** cross-dataset validation; 3D volumetric models; multi-modal fusion;
transformer hybrids; lightweight deployment; SHAP-based explainability.

---

## Reproducibility and Ethics

All seeds fixed at 42. Code, config, and split CSVs released at [GitHub URL].
Dataset (Nickparvar [12]) is publicly available for research; no new human data
were collected; no IRB approval required.

## Conflict of Interest

The authors declare no conflict of interest.

---

## References

[1]–[13]: same as paper_Q1_journal.md (He et al.; Hu et al.; Woo et al.; Tan & Le;
Lin et al.; Dosovitskiy et al.; Huang et al.; Szegedy et al.; Simonyan & Zisserman;
Deepak & Ameer; Swati et al.; Nickparvar; Cheng et al.)

*(Add 20–25 further references from your literature review to reach 30–40 total,
prioritising 2020–2026 work, formatted to the target journal's style — IEEE or
Elsevier as appropriate.)*
