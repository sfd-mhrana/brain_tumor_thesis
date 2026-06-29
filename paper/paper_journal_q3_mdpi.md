# CBAM-Enhanced EfficientNet for Brain Tumor MRI Classification

**Target:** MDPI Diagnostics / MDPI Applied Sciences / MDPI Electronics (Q3)
**Format:** Research article · ~4,500–6,000 words · 20–30 references · Unstructured abstract

> **STATUS NOTE (delete before submission):** All quantitative results are
> literature-consistent placeholders. Run the pipeline, replace with your own
> numbers, and remove this note before submission.

---

## Abstract

Automated brain tumour classification from magnetic resonance imaging (MRI) using
deep learning can support radiologists in resource-constrained settings. In this
paper, we propose an attention-enhanced deep learning model — EfficientNetB3
augmented with a Convolutional Block Attention Module (CBAM) — and evaluate it
alongside nine other architectures on the publicly available Kaggle Brain Tumor
MRI dataset (7,023 images, four classes: glioma, meningioma, pituitary tumour,
no-tumour). To ensure a fair comparison, all models share one preprocessing
pipeline (CLAHE at 224 × 224), one stratified 70/15/15 split, and identical
two-stage transfer-learning training with AdamW optimisation and focal loss.
The proposed EfficientNetB3+CBAM model achieved [fill your result] accuracy and
[fill your result] macro-F1, outperforming all baselines. Ablation experiments
confirm that CLAHE preprocessing, two-stage fine-tuning, CBAM attention, and
focal loss each contribute positively. Grad-CAM++ visualisations verify that the
model focuses on tumour tissue. Results demonstrate strong potential for
AI-assisted clinical decision support.

**Keywords:** brain tumour; MRI; deep learning; CBAM; EfficientNet; transfer
learning; attention mechanism; Grad-CAM; medical image classification.

---

## 1. Introduction

Brain tumours, including gliomas, meningiomas, and pituitary adenomas, vary
significantly in prognosis and treatment. Magnetic resonance imaging (MRI) is the
standard diagnostic modality, but manual interpretation requires specialist expertise
and is time-consuming. In low-resource settings, a shortage of radiologists makes
automated classification valuable.

Deep learning, particularly transfer learning from ImageNet-pretrained convolutional
neural networks (CNNs), has achieved accuracy exceeding 95 % on multi-class
brain-MRI benchmarks. However, most published comparisons evaluate only a few
architectures and rarely isolate the contribution of individual design choices.

In this paper, we address these gaps by: (1) benchmarking ten architectures under
a single, identical pipeline; (2) proposing EfficientNetB3+CBAM as an attention-
enhanced model; (3) running four ablation studies on the key design choices; and
(4) verifying spatial decision-making with Grad-CAM++ and confirming significance
with McNemar's test.

---

## 2. Related Work

Transfer learning from ImageNet-pretrained backbones has become the dominant
approach for brain-MRI classification, with VGG, ResNet, and DenseNet models
reporting 94–98 % accuracy [10,11]. EfficientNet [4] improved upon these with
compound depth/width/resolution scaling. Attention mechanisms — SE blocks [2]
and CBAM [3] — have shown consistent gains in medical imaging by directing
network focus towards diagnostically relevant regions. Focal loss [5] addresses
class imbalance by emphasising hard misclassified examples. Despite these advances,
few studies compare more than three models under identical conditions, and ablation
analysis is rare.

*Table 1. Representative prior work (fill from your literature review).*

| Study | Year | Dataset | Method | Accuracy |
|---|---|---|---|---|
| Deepak & Ameer [10] | 2019 | CE-MRI / 3 cls | GoogLeNet TL | 97.1 % |
| Swati et al. [11] | 2019 | CE-MRI | VGG19 | 94.8 % |
| [fill] | 2022 | Kaggle / 4 cls | EfficientNet | ~97 % |
| **This work** | 2026 | Kaggle / 4 cls | EfficientNetB3+CBAM | *(your result)* |

---

## 3. Materials and Methods

### 3.1 Dataset and preprocessing

We use the Kaggle Brain Tumor MRI dataset [12] (7,023 images; 4 classes). A single
stratified split — 70 % train / 15 % validation / 15 % test (`random_state = 42`)
— is shared by all models. Each image is resized to 224 × 224 and enhanced with
**CLAHE** (clip limit 2.0, 8 × 8 tiles on the L channel of LAB colour space) to
improve tumour boundary contrast. Grayscale images are replicated to three channels;
all images are normalised with ImageNet mean/std.

Training augmentation includes: rotation (±15°), translation (±8 %), zoom (±10 %),
brightness jitter (±15 %), and horizontal flip. Vertical flips are excluded
(anatomically invalid for brain MRI).

### 3.2 Models

We evaluate ten configurations (Table 2). All transfer-learning models share a
classifier head: GAP → BN → Dense(512, ReLU) → Dropout(0.4) → Dense(256, ReLU)
→ Dropout(0.3) → Dense(4, softmax).

**Table 2.** Architectures evaluated.

| # | Model | Params (M) | Role |
|---|---|---|---|
| 1 | Custom CNN | ~1 | performance floor |
| 2 | VGG16 | 138 | classic baseline |
| 3 | ResNet50 | 25.6 | core baseline |
| 4 | InceptionV3 | 23 | multi-scale |
| 5 | DenseNet121 | 8 | data-efficient |
| 6 | EfficientNetB0 | 5.3 | efficient |
| 7 | EfficientNetB3 | 12 | strong baseline |
| 8 | ResNet50 + SE | 26 | channel attention |
| 9 | **EfficientNetB3 + CBAM** | 13 | **proposed** |
| 10 | Ensemble (top-3) | — | upper bound |

### 3.3 Proposed model: EfficientNetB3 + CBAM

A CBAM block [3] is inserted after the final convolutional feature map of
EfficientNetB3 before global pooling. CBAM first applies **channel attention**
(average- and max-pooled channel descriptors, shared MLP, sigmoid gate) to
highlight informative feature channels, then **spatial attention** (channel-wise
average/max pooling concatenated, 7 × 7 convolution, sigmoid gate) to highlight
informative spatial locations. The module adds ~1 M parameters (~8 % overhead).

### 3.4 Training

Two-stage transfer learning is used for all pretrained models:
- **Stage 1** — backbone frozen, head trained (10 epochs, LR 1 × 10⁻³).
- **Stage 2** — top-30 backbone layers unfrozen, fine-tuned (up to 30 epochs,
  LR 1 × 10⁻⁵).

Common hyperparameters: AdamW (weight decay 1 × 10⁻⁴), cosine LR schedule,
batch size 32, focal loss (γ = 2.0), early stopping (patience 12), seed 42.

### 3.5 Evaluation

Primary metric: **macro-F1**. Also reported: accuracy, AUC-ROC (OvR), Cohen's κ,
confusion matrix, and inference latency. McNemar's test (α = 0.05) compares the
proposed model vs baselines; 5-fold CV is reported for the top two models.

---

## 4. Results

> All numbers below are **placeholders** — replace with your experimental results.

### 4.1 Comparison across models

**Table 3.** Test-set performance *(placeholder)*.

| Model | Accuracy | Macro-F1 | AUC-ROC | Params (M) | Lat (ms) |
|---|---|---|---|---|---|
| Custom CNN | 88.5 % | 87.0 % | 0.943 | 1.0 | 8 |
| VGG16 | 94.2 % | 93.6 % | 0.981 | 138 | 25 |
| InceptionV3 | 95.1 % | 94.6 % | 0.987 | 23 | 14 |
| DenseNet121 | 95.7 % | 95.2 % | 0.989 | 8 | 13 |
| ResNet50 | 96.1 % | 95.6 % | 0.991 | 25.6 | 15 |
| EfficientNetB0 | 96.6 % | 96.2 % | 0.993 | 5.3 | 10 |
| ResNet50 + SE | 96.9 % | 96.5 % | 0.994 | 26 | 16 |
| EfficientNetB3 | 97.3 % | 96.9 % | 0.995 | 12 | 12 |
| **EffNetB3+CBAM (ours)** | **98.1 %** | **97.7 %** | **0.998** | 13 | 14 |
| Ensemble (top-3) | 98.4 % | 98.1 % | 0.998 | — | 40 |

The proposed model leads all single-model baselines. EfficientNet architectures
achieve the best accuracy-per-parameter ratio. The ensemble adds only +0.4 pp
over the proposed model at 3× inference cost.

### 4.2 Ablation studies

**Table 4.** Ablation results on macro-F1 *(placeholder)*.

| Ablation | Without | With (proposed) | Δ F1 |
|---|---|---|---|
| AB-1 CLAHE | 96.1 % | 97.7 % | +1.6 pp |
| AB-2 Augmentation | 94.8 % | 97.7 % | +2.9 pp |
| AB-3 Two-stage fine-tune | 96.0 % | 97.7 % | +1.7 pp |
| AB-4 CBAM attention | 96.9 % | 97.7 % | +0.8 pp |
| AB-5 Focal loss | 96.6 % | 97.7 % | +1.1 pp |

All components contribute; augmentation and CLAHE show the largest individual gains.
Meningioma recall improves most from focal loss (AB-5) and spatial attention (AB-4).

### 4.3 Explainability

Grad-CAM++ activations for the proposed model concentrate on tumour tissue in all
four classes. Compared with the ResNet50 baseline, the proposed model shows tighter
activation maps aligned with tumour boundaries (Figure [fill]: 4 × 3 grid — rows
are classes, columns are original / ResNet50 / proposed).

### 4.4 Statistical significance

McNemar's test: proposed vs EfficientNetB3, *p* < 0.05 (significant). 5-fold CV:
EfficientNetB3 = 96.5 ± 0.3 %; proposed EfficientNetB3+CBAM = **97.2 ± 0.3 %**
(non-overlapping 95 % CIs confirm a robust improvement).

---

## 5. Discussion

EfficientNetB3+CBAM outperforms all baselines through the combined effect of CLAHE
contrast enhancement, two-stage fine-tuning, focal loss, and spatial attention.
Ablations confirm these contributions are additive. The meningioma class remains
the most challenging (lowest per-class recall), and CBAM and focal loss together
produce the largest recall gains on this class.

Compared with published work on the same dataset (Table 5), the proposed model
achieves competitive or superior accuracy while providing more thorough ablation
evidence and statistical validation than most prior studies.

**Table 5.** Comparison with published work *(fill from literature)*.

| Study | Year | Method | Accuracy |
|---|---|---|---|
| [fill] | 2022 | EfficientNet | ~97 % |
| [fill] | 2023 | ViT fine-tune | ~97.5 % |
| **This work** | 2026 | EfficientNetB3+CBAM | **98.1 %** |

**Limitations:** single dataset, 2D slices only, no clinical validation, limited
scanner diversity.

---

## 6. Conclusion

We proposed EfficientNetB3+CBAM for four-class brain-tumour MRI classification
and demonstrated its superiority over nine baselines under a rigorously fair
protocol. Ablation studies confirm additive contributions from each design choice;
Grad-CAM++ verifies tumour-focused decision making; McNemar's test confirms
statistical significance. The model is lightweight (13 M parameters, 14 ms/image)
and suitable for deployment in resource-constrained clinical settings.

**Future work:** cross-dataset validation; 3D/volumetric models; clinical evaluation;
lightweight deployment; multi-modal fusion.

---

## Author Contributions

[Your Name]: conceptualisation, methodology, software, experiments, writing.
[Supervisor]: supervision, review and editing.

## Conflicts of Interest

The authors declare no conflict of interest.

## Data Availability

The dataset is publicly available at [12]. Code is released at [GitHub URL].

---

## References

[1]–[13]: He et al. (ResNet); Hu et al. (SE); Woo et al. (CBAM); Tan & Le
(EfficientNet); Lin et al. (Focal Loss); Dosovitskiy et al. (ViT); Huang et al.
(DenseNet); Szegedy et al. (InceptionV3); Simonyan & Zisserman (VGG); Deepak &
Ameer; Swati et al.; Nickparvar (Kaggle dataset); Cheng et al. (CE-MRI dataset).

*(Add 10–15 further references from your literature review to reach 20–30 total.
Use MDPI citation style — numbered, IEEE-like format.)*
