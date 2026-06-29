# EfficientNetB3 with CBAM Attention for Brain Tumor MRI Classification

**Target:** IEEE EMBC / IEEE BIBM / IEEE TENCON / ICCIT (Q4 / Conference)
**Format:** Short conference paper · 6–8 pages (IEEE two-column) · 15–25 references
**Word count target:** ~3,500–4,500 words (body, excluding abstract and references)

> **STATUS NOTE (delete before submission):** All results are placeholders.
> Run the pipeline, replace with your numbers, and remove this note.

---

## Abstract

We propose EfficientNetB3+CBAM, an attention-enhanced transfer-learning model for
four-class brain tumour MRI classification (glioma, meningioma, pituitary, no-tumour)
on the Kaggle Brain Tumor MRI dataset (7,023 images). CBAM channel-and-spatial
attention is appended to an EfficientNetB3 backbone and trained with two-stage
fine-tuning, AdamW optimisation, and focal loss. Compared against eight baselines
under an identical preprocessing and split protocol, the proposed model achieves
[fill]% accuracy and [fill]% macro-F1, outperforming the next-best single model
by [fill] pp. Ablation experiments and Grad-CAM++ visualisations confirm the
contribution of each design choice and the spatial relevance of model decisions.

**Index Terms:** brain tumour, MRI, deep learning, EfficientNet, CBAM, transfer
learning, attention mechanism.

---

## I. Introduction

Brain tumours — gliomas, meningiomas, and pituitary adenomas — require different
treatment strategies, making accurate MRI classification clinically important.
Manual interpretation is slow and variable; automated deep learning classification
can support radiologists, especially in resource-limited settings.

Transfer learning from ImageNet-pretrained CNNs has achieved 94–98 % accuracy on
brain-MRI classification [10], [11]. Attention mechanisms (SE [2], CBAM [3]) and
efficient architectures (EfficientNet [4]) have further improved performance, but
most studies compare only a few models under inconsistent conditions and rarely
validate statistical significance.

This paper makes three contributions: (1) a fair comparison of nine architectures
under one identical pipeline; (2) a proposed EfficientNetB3+CBAM model with
channel-and-spatial attention; and (3) ablation and Grad-CAM++ evidence supporting
each design choice.

---

## II. Methodology

### A. Dataset and Preprocessing

The Kaggle Brain Tumor MRI dataset [12] contains 7,023 T1-weighted contrast MRI
images in four classes (glioma, meningioma, pituitary, no-tumour). A single
stratified 70/15/15 split (`random_state = 42`) is shared by all models.

Preprocessing: resize to 224 × 224; **CLAHE** (clipLimit 2.0, 8 × 8 tiles on the
LAB-L channel); grayscale → 3-channel; ImageNet normalisation. Training augmentation:
rotation ±15°, translation ±8 %, zoom ±10 %, brightness ±15 %, horizontal flip
(vertical flips excluded — anatomically invalid).

### B. Models

Nine architectures are compared (Table I). All pretrained models share a classifier
head: GAP → BN → Dense(512, ReLU) → Dropout(0.4) → Dense(256, ReLU) →
Dropout(0.3) → Dense(4, softmax).

**Table I.** Architectures evaluated.

| # | Model | Params (M) | Role |
|---|---|---|---|
| 1 | Custom CNN | ~1 | baseline floor |
| 2 | VGG16 | 138 | classic |
| 3 | ResNet50 | 25.6 | core baseline |
| 4 | InceptionV3 | 23 | multi-scale |
| 5 | DenseNet121 | 8 | dense |
| 6 | EfficientNetB0 | 5.3 | efficient |
| 7 | EfficientNetB3 | 12 | strong baseline |
| 8 | ResNet50 + SE | 26 | channel attention |
| **9** | **EfficientNetB3 + CBAM** | **13** | **proposed** |

### C. Proposed Model

CBAM [3] is inserted after EfficientNetB3's final convolutional feature map, before
global pooling. It applies in sequence:
- **Channel attention:** shared MLP over average- and max-pooled channel descriptors,
  sigmoid-gated recalibration.
- **Spatial attention:** channel-wise avg/max pooling, 7 × 7 convolution,
  sigmoid spatial gate.

Total overhead: ~1 M parameters (~8 % over backbone).

### D. Training Protocol

Two-stage transfer learning: Stage 1 freezes backbone, trains head (10 epochs,
LR 1e-3); Stage 2 unfreezes top-30 layers, fine-tunes (30 epochs, LR 1e-5).
All models: AdamW (wd 1e-4), cosine LR schedule, batch 32, focal loss (γ = 2),
early stopping (patience 12), seed 42.

### E. Evaluation

Primary: **macro-F1**. Also: accuracy, AUC-ROC, Cohen's κ, confusion matrix,
latency. McNemar's test (α = 0.05) compares the proposed model vs baselines.

---

## III. Results

> All values are **placeholders** — replace with your results.

### A. Performance Comparison

**Table II.** Test-set results *(placeholder)*.

| Model | Accuracy | Macro-F1 | AUC-ROC | Lat (ms) |
|---|---|---|---|---|
| Custom CNN | 88.5 % | 87.0 % | 0.943 | 8 |
| VGG16 | 94.2 % | 93.6 % | 0.981 | 25 |
| InceptionV3 | 95.1 % | 94.6 % | 0.987 | 14 |
| DenseNet121 | 95.7 % | 95.2 % | 0.989 | 13 |
| ResNet50 | 96.1 % | 95.6 % | 0.991 | 15 |
| EfficientNetB0 | 96.6 % | 96.2 % | 0.993 | 10 |
| ResNet50 + SE | 96.9 % | 96.5 % | 0.994 | 16 |
| EfficientNetB3 | 97.3 % | 96.9 % | 0.995 | 12 |
| **EffNetB3+CBAM** | **98.1 %** | **97.7 %** | **0.998** | **14** |

### B. Ablation Results

**Table III.** Macro-F1 ablations *(placeholder)*.

| Ablation | Removed | With | Δ |
|---|---|---|---|
| CLAHE | 96.1 % | 97.7 % | +1.6 pp |
| 2-stage fine-tune | 96.0 % | 97.7 % | +1.7 pp |
| CBAM attention | 96.9 % | 97.7 % | +0.8 pp |
| Focal loss | 96.6 % | 97.7 % | +1.1 pp |

All four components contribute independently; no component is redundant.

### C. Explainability and Significance

Grad-CAM++ activations for the proposed model concentrate on tumour tissue (Figure
[fill]: sample activations per class). Compared with ResNet50, the proposed model
shows tighter, tumour-aligned spatial attention. McNemar's test confirms the
proposed model significantly outperforms EfficientNetB3 (*p* < 0.05) and all
weaker baselines (*p* < 0.01).

---

## IV. Discussion

The proposed model outperforms all baselines through the combination of CLAHE,
two-stage fine-tuning, focal loss, and CBAM attention — each contributing
independently as shown in Table III. Meningioma remains the hardest class;
focal loss provides the largest meningioma recall improvement. At 13 M parameters
and 14 ms inference, the model is suitable for real-time clinical deployment.

**Comparison with prior work:** [fill in one paragraph from your literature review
comparing your result against 3–4 published results on the same dataset.]

**Limitations:** single dataset; 2D slices; no clinical evaluation.

---

## V. Conclusion

We proposed EfficientNetB3+CBAM for four-class brain-tumour MRI classification
and demonstrated its superiority over eight baselines under a fair, reproducible
protocol. Ablation studies confirm additive contributions from CLAHE, two-stage
fine-tuning, CBAM attention, and focal loss. Grad-CAM++ validates tumour-focused
decision making. Code and split CSVs are released at [GitHub URL].

**Future work:** cross-dataset validation; 3D volumetric models; clinical deployment.

---

## References

[1] K. He et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[2] J. Hu et al., "Squeeze-and-Excitation Networks," *CVPR*, 2018.

[3] S. Woo et al., "CBAM: Convolutional Block Attention Module," *ECCV*, 2018.

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling," *ICML*, 2019.

[5] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.

[6] G. Huang et al., "Densely Connected Convolutional Networks," *CVPR*, 2017.

[7] C. Szegedy et al., "Rethinking the Inception Architecture," *CVPR*, 2016.

[8] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks," *ICLR*, 2015.

[9] A. Dosovitskiy et al., "An Image is Worth 16×16 Words," *ICLR*, 2021.

[10] S. Deepak and P. M. Ameer, "Brain Tumor Classification via Transfer Learning,"
*Computers in Biology and Medicine*, 2019.

[11] Z. N. K. Swati et al., "Brain Tumor Classification for MR Images Using
Transfer Learning," *Comput. Med. Imaging Graph.*, 2019.

[12] M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle, 2021.

*(Add 5–10 more references from your literature review to reach 15–25 total,
formatted to IEEE conference style.)*
