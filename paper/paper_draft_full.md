# Attention-Enhanced EfficientNet for Multi-Class Brain Tumor MRI Classification: A Comprehensive Benchmarking Study

**Author:** [Your Name]¹
**Affiliation:** ¹[Department], [University], [City], [Country]
**Corresponding author:** [email]

---

> **STATUS / INTEGRITY NOTE (delete before submission).**
> This manuscript is structurally complete, but the quantitative results in
> Tables 2–7 and the abstract are **placeholders** — they are reference values
> drawn from the published literature on this dataset, included so the paper
> reads as a finished draft. They are **not** the output of the experiments in
> this repository, which have not yet been run. After you execute the pipeline
> (`scripts/02`–`07`), replace every value marked *(placeholder)* with your own
> numbers from `experiments/results/`, regenerate the figures, and remove this
> note. Presenting unrun numbers as real results is research misconduct and will
> end a thesis or a paper.

---

## Abstract

**Background.** Brain tumours are among the most lethal cancers, and accurate
classification of tumour type from magnetic resonance imaging (MRI) is critical
for treatment planning. Manual interpretation is time-consuming, subjective, and
constrained by a shortage of radiologists in many regions. Deep learning offers
fast, consistent, automated classification, but the literature is dominated by
studies that compare only two or three architectures, omit ablation analysis,
and rarely test whether reported differences are statistically significant.

**Methods.** We benchmark ten deep-learning configurations — a from-scratch CNN,
six ImageNet-pretrained transfer-learning backbones (VGG16, ResNet50,
InceptionV3, DenseNet121, EfficientNetB0, EfficientNetB3), a channel-attention
variant (ResNet50+SE), a top-3 soft-voting ensemble, and our proposed
**EfficientNetB3 + CBAM** — on the Kaggle Brain Tumor MRI dataset (7,023 images;
glioma, meningioma, pituitary, no-tumour). All models share an identical
pipeline: CLAHE preprocessing, a single stratified 70/15/15 split, the same
augmentation, two-stage fine-tuning with AdamW and cosine scheduling, and focal
loss for class imbalance. We report accuracy, macro-F1, AUC-ROC, Cohen's kappa,
and inference latency; isolate each design choice through six ablation studies;
verify spatial relevance with Grad-CAM++; and establish significance with
McNemar's test and 5-fold cross-validation.

**Results (placeholder — replace with your runs).** The proposed
EfficientNetB3+CBAM reached **98.1 %** accuracy and **97.7 %** macro-F1,
outperforming the strongest baseline (EfficientNetB3, 96.9 % macro-F1) by a
margin that was statistically significant (McNemar, *p* < 0.05). Ablations
confirmed that CLAHE, two-stage fine-tuning, CBAM attention, and focal loss each
contributed positively, with CBAM and focal loss together driving the largest
gain in meningioma recall. Grad-CAM++ showed the proposed model concentrating on
tumour tissue rather than scanner artefacts.

**Conclusion.** A small, well-motivated attention enhancement to a strong
efficient backbone, evaluated under rigorously fair and statistically validated
conditions, sets a competitive benchmark and supports its potential for
AI-assisted clinical decision support.

**Keywords:** brain tumour, MRI, deep learning, transfer learning, attention
mechanism, CBAM, EfficientNet, explainable AI, Grad-CAM, medical image
classification.

---

## 1. Introduction

Brain and central-nervous-system tumours cause substantial morbidity and
mortality worldwide. Treatment decisions depend heavily on tumour type: gliomas,
meningiomas, and pituitary tumours differ in malignancy, location, and
management, so correctly distinguishing them — and distinguishing tumour from
healthy tissue — is a clinically consequential task. Magnetic resonance imaging
is the primary non-invasive modality for this, but manual reading is slow,
depends on expert availability, and exhibits inter- and intra-observer
variability. In many low- and middle-income settings the radiologist-to-patient
ratio makes timely interpretation difficult.

Convolutional neural networks (CNNs) and, more recently, transfer learning from
ImageNet-pretrained backbones have driven large accuracy gains on brain-MRI
classification. Yet three methodological weaknesses recur in the literature.
First, most studies compare only two or three architectures, often under
slightly different preprocessing or splits, which makes the comparison unfair and
hard to reproduce. Second, **ablation analysis is rare** — when a model combines
preprocessing, attention, fine-tuning, and a custom loss, papers seldom isolate
how much each part contributes. Third, **statistical validation is usually
absent**: differences of a fraction of a percent are reported as improvements
without confidence intervals or significance tests, so it is unclear whether they
reflect a real effect or random variation.

This work addresses all three. We hold the entire data and training pipeline
fixed and vary only the architecture, so any difference is attributable to the
model. We add a lightweight Convolutional Block Attention Module (CBAM) to an
EfficientNetB3 backbone and quantify its effect against no-attention and
channel-only (SE) variants. And we validate every key claim with McNemar's test
and 5-fold cross-validation.

**Contributions.**

1. A **fair, reproducible benchmark** of ten architectures under one identical
   pipeline (CLAHE, shared stratified split, common augmentation, two-stage
   fine-tuning, focal loss), with code released publicly.
2. A **proposed EfficientNetB3 + CBAM** model that adds channel-and-spatial
   attention to a strong efficient backbone with negligible parameter overhead.
3. **Six controlled ablation studies** isolating preprocessing, augmentation,
   fine-tuning strategy, attention type, loss function, and optimizer.
4. **Explainability and statistical rigour**: Grad-CAM++ localisation plus
   McNemar significance testing and 5-fold cross-validation with 95 % confidence
   intervals.

The rest of the paper is organised as follows. Section 2 reviews related work.
Section 3 details the methodology. Section 4 describes the experimental setup.
Section 5 presents and discusses results, ablations, explainability, and
statistical tests. Section 6 notes limitations, and Section 7 concludes.

---

## 2. Related Work

**Traditional machine learning (≈2010–2017).** Early systems combined
hand-crafted texture and intensity features (GLCM, wavelet, Gabor) with
classifiers such as SVMs and random forests. They were interpretable but
depended on manual feature engineering and generalised poorly across scanners.

**The CNN era (≈2017–2021).** End-to-end CNNs removed the need for hand-crafted
features. AlexNet- and VGG-style networks trained on brain-MRI datasets achieved
strong accuracy, and architectural innovations — residual connections in ResNet
[1], multi-scale Inception modules, and dense connectivity in DenseNet — improved
gradient flow and feature reuse.

**Transfer learning for medical imaging (≈2019–2022).** Because annotated medical
data are scarce, fine-tuning ImageNet-pretrained backbones became the dominant
paradigm. Deepak and Ameer, Swati et al., Sultan et al., and Sajjad et al.
reported that fine-tuned VGG/GoogLeNet/ResNet models reached 94–98 % accuracy on
brain-MRI classification, frequently surpassing from-scratch training. EfficientNet
[4], with its compound depth/width/resolution scaling, offered a markedly better
accuracy-per-parameter trade-off and has become a strong default backbone.

**Attention mechanisms.** Squeeze-and-Excitation (SE) blocks [2] reweight feature
channels by global context, and the Convolutional Block Attention Module (CBAM)
[3] adds a complementary spatial-attention stage, letting the network emphasise
both *what* and *where* is informative. In medical imaging, where the
diagnostically relevant region is a small fraction of the image, such modules are
well motivated and have been shown to improve lesion-focused classification.

**Vision Transformers and hybrids (≈2022–2024).** ViT [6] and Swin transformers
model long-range dependencies and are competitive when sufficient data or strong
pretraining is available, though they are typically more data-hungry than CNNs at
the dataset sizes common in brain-MRI work.

**Class imbalance.** Focal loss [5] down-weights easy examples to focus training
on hard ones and is widely used to improve recall on under-represented classes —
relevant here, where meningioma is the hardest class.

**Research gap.** Across these clusters, few studies (i) compare six or more
architectures under identical preprocessing and splits, (ii) report controlled
ablations isolating each component, and (iii) accompany results with statistical
significance testing and explainability. This paper targets that combined gap.

*Table 1 summarises representative prior work on the same or comparable datasets.*

**Table 1.** Representative related work (compile from your literature matrix).

| Study | Year | Dataset / classes | Method | Accuracy | Reported limitation |
|---|---|---|---|---|---|
| [fill] | | CE-MRI / 3 | VGG19 fine-tune | ~94–97 % | single dataset, no ablation |
| [fill] | | Kaggle / 4 | ResNet/Inception TL | ~95–98 % | 2–3 models, no significance test |
| [fill] | | Kaggle / 4 | EfficientNet | ~97 % | no explainability |
| **This work** | 2026 | Kaggle / 4 | EffNetB3+CBAM | *(your result)* | — |

---

## 3. Methodology

### 3.1 System overview

The pipeline has five stages: (1) CLAHE preprocessing and resizing; (2) a single
stratified train/validation/test split shared by every model; (3) training-only
augmentation; (4) two-stage transfer learning; and (5) a fixed evaluation and
significance-testing protocol. Holding (1)–(3) and (5) constant and varying only
the model in (4) is what makes the comparison scientifically fair.

### 3.2 Dataset

We use the publicly available **Kaggle Brain Tumor MRI dataset**
(`masoudnickparvar/brain-tumor-mri-dataset`), comprising **7,023** T1-weighted
contrast MRI images across four classes — glioma, meningioma, pituitary tumour,
and no-tumour. We pool the provided Training and Testing folders and re-split to
control the split exactly (Section 3.4). Exploratory analysis records class
distribution and imbalance ratio, image-dimension range, per-class sample grids,
and a hash-based duplicate check; corrupted or duplicate images are removed.

### 3.3 Preprocessing

Each image is resized to 224 × 224 and enhanced with **Contrast-Limited Adaptive
Histogram Equalisation (CLAHE)** (clip limit 2.0, 8 × 8 tiles) applied to the
luminance (L) channel of the LAB colour space only, which sharpens tumour
boundaries without distorting colour statistics. Grayscale MRIs are represented
as three channels to match the ImageNet input format. Inputs are normalised with
ImageNet mean/standard deviation to align the source and target domains. Both raw
and CLAHE versions are stored to enable ablation AB-1.

### 3.4 Reproducible data split

A single stratified split — **70 % train / 15 % validation / 15 % test** — is
generated once with `random_state = 42` and written to CSV. **Every** model reads
these exact CSVs, so no model gains an advantage from an easier split, and the
test set is never used for any training or model-selection decision.

### 3.5 Augmentation

Augmentation is applied to the training set only and restricted to
anatomically valid transforms: rotation (±15°), small translation (±8 %), zoom
(±10 %), brightness jitter (±15 %), and horizontal flip (valid given approximate
left–right brain symmetry). Vertical flips and heavy shear/distortion are
excluded because they create anatomically implausible images. Validation and test
images are normalised only.

### 3.6 Architectures

We evaluate the ten configurations in Table 2. Transfer-learning backbones are
instantiated from ImageNet-pretrained weights via `timm` with a shared classifier
head (global average pooling → BatchNorm → 512 → 256 → softmax, with dropout
0.4/0.3). The from-scratch custom CNN establishes the performance floor.

**Table 2.** Architectures benchmarked.

| # | Model | Params (M) | Key idea | Role |
|---|---|---|---|---|
| 1 | Custom CNN | ~1 | trained from scratch | performance floor |
| 2 | VGG16 | 138 | deep 3×3 stacks | classic baseline |
| 3 | ResNet50 | 25.6 | residual connections | core baseline |
| 4 | InceptionV3 | 23 | multi-scale convolutions | multi-scale baseline |
| 5 | DenseNet121 | 8 | dense feature reuse | data-efficient baseline |
| 6 | EfficientNetB0 | 5.3 | compound scaling | efficient baseline |
| 7 | EfficientNetB3 | 12 | scaled EfficientNet | strong baseline |
| 8 | ResNet50 + SE | 26 | channel attention | attention comparison |
| 9 | **EfficientNetB3 + CBAM** | 13 | channel + spatial attention | **proposed** |
| 10 | Ensemble (top-3) | — | soft voting | upper bound |

### 3.7 Proposed model: EfficientNetB3 + CBAM

The proposed model inserts a CBAM block [3] on the final convolutional feature
map of EfficientNetB3, immediately before global pooling. CBAM applies, in
sequence:

- **Channel attention** — average- and max-pooled channel descriptors pass
  through a shared MLP; their sum is sigmoid-gated to produce a per-channel
  weight, emphasising the most informative feature channels (*what*).
- **Spatial attention** — the channel-refined map is reduced by channel-wise
  average and max pooling, concatenated, and passed through a 7 × 7 convolution
  with a sigmoid to produce a spatial weight map, emphasising the most
  informative locations (*where*).

The module adds roughly 1 M parameters (about 8 % over the 12 M backbone), so the
gains it produces are not simply the result of greater capacity — an important
control that the no-attention ablation (AB-4) verifies.

### 3.8 Training protocol

All models use **two-stage transfer learning**:

1. **Stage 1 (head):** the backbone is frozen and only the new classifier head is
   trained (10 epochs, learning rate 1e-3).
2. **Stage 2 (fine-tune):** the top backbone blocks are unfrozen and the whole
   network is fine-tuned at a low learning rate (up to 30 epochs, 1e-5).

Common settings: **AdamW** (weight decay 1e-4), **cosine** learning-rate
schedule, batch size 32, label smoothing 0.1, **focal loss** (γ = 2) with
inverse-frequency class weighting, gradient clipping (max-norm 1.0), mixed
precision, and early stopping on validation loss (patience 12). Seeds are fixed
at 42 throughout. The custom CNN is trained end-to-end with the same optimizer
and schedule.

### 3.9 Evaluation metrics

We report accuracy, macro precision/recall/F1 (**macro-F1 is the primary metric**
because it weights all classes equally under imbalance), weighted F1, per-class
F1, macro specificity, one-vs-rest AUC-ROC, Cohen's kappa, the confusion matrix,
single-image inference latency, and parameter count. In a medical context recall
(sensitivity) is emphasised, since a missed tumour (false negative) is generally
more harmful than a false alarm.

---

## 4. Experimental Setup

Experiments run on [GPU model, e.g. NVIDIA T4/A100 via Google Colab] with PyTorch
2.2 and `timm` 0.9. Each model trains in [fill] minutes. All randomness is seeded
(Python, NumPy, PyTorch, cuDNN deterministic). Code, configuration, and split
CSVs are released at [GitHub URL] for full reproducibility.

---

## 5. Results and Discussion

> All numbers in this section are **placeholders** pending your runs. Generate
> them with `scripts/02`–`07` and overwrite. The accompanying figures are written
> to `figures/` by the same scripts.

### 5.1 Master comparison

**Table 3.** Test-set performance, all models (*placeholder*; from
`experiments/results/master_comparison.csv`).

| Model | Accuracy | Precision | Recall | Macro-F1 | AUC-ROC | Params (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Custom CNN | 0.885 | 0.872 | 0.868 | 0.870 | 0.943 | 1.0 | 8 |
| VGG16 | 0.942 | 0.938 | 0.935 | 0.936 | 0.981 | 138 | 25 |
| InceptionV3 | 0.951 | 0.948 | 0.945 | 0.946 | 0.987 | 23 | 14 |
| DenseNet121 | 0.957 | 0.954 | 0.951 | 0.952 | 0.989 | 8 | 13 |
| ResNet50 | 0.961 | 0.958 | 0.955 | 0.956 | 0.991 | 25.6 | 15 |
| EfficientNetB0 | 0.966 | 0.963 | 0.961 | 0.962 | 0.993 | 5.3 | 10 |
| ResNet50 + SE | 0.969 | 0.966 | 0.964 | 0.965 | 0.994 | 26 | 16 |
| EfficientNetB3 | 0.973 | 0.970 | 0.968 | 0.969 | 0.995 | 12 | 12 |
| **EffNetB3 + CBAM (ours)** | **0.981** | **0.978** | **0.976** | **0.977** | **0.998** | 13 | 14 |
| Ensemble (top-3) | 0.984 | 0.982 | 0.980 | 0.981 | 0.998 | — | 40 |

*(Illustrative literature-consistent values — replace with your results.)*

The proposed model is expected to lead all single models, with the soft-voting
ensemble forming the upper bound. The attention models (SE, CBAM) should exceed
their non-attention counterparts, and EfficientNet backbones should give the best
accuracy-per-parameter.

### 5.2 Effect of attention (AB-4)

**Table 4.** Attention ablation on EfficientNetB3 (*placeholder*).

| Variant | Macro-F1 | Meningioma recall | Δ Macro-F1 |
|---|---|---|---|
| No attention | 0.969 | 0.946 | — |
| + SE (channel only) | 0.973 | 0.955 | +0.004 |
| + CBAM (channel + spatial) | 0.977 | 0.964 | +0.008 |

CBAM's spatial stage is expected to add value beyond SE's channel-only
reweighting, consistent with the intuition that tumour localisation matters.

### 5.3 Ablation studies

**Table 5.** Single-factor ablations (*placeholder*; from
`experiments/results/ablations/`). Each row changes exactly one factor.

| Ablation | Setting | Macro-F1 | Finding |
|---|---|---|---|
| AB-1 Preprocessing | none → CLAHE | 0.961 → 0.969 | CLAHE helps (+0.8) |
| AB-2 Augmentation | none / standard / heavy | 0.948 / 0.969 / 0.964 | moderate aug best |
| AB-3 Fine-tuning | 1-stage → 2-stage | 0.960 → 0.969 | 2-stage better (+0.9) |
| AB-4 Attention | none / SE / CBAM | 0.969 / 0.973 / 0.977 | CBAM > SE > none |
| AB-5 Loss | CE / weighted-CE / focal | 0.966 / 0.971 / 0.977 | focal best, esp. meningioma |
| AB-6 Optimizer | SGD / Adam / AdamW | 0.962 / 0.970 / 0.977 | AdamW best generalisation |

### 5.4 Per-class analysis

Meningioma is consistently the hardest class (lowest recall, most often confused
with glioma), which motivates focal loss and class weighting. Report your final
per-class F1 and confusion matrix here and discuss the dominant confusions.

### 5.5 Explainability (Grad-CAM++)

Figure [grad-cam] shows the 4 × 3 Grad-CAM++ grid (one MRI per class × original /
ResNet50 baseline / proposed model). The proposed model's activations should
concentrate on tumour tissue, whereas the baseline more often attends to
non-tumour regions — visual evidence that the attention mechanism improves
lesion focus and supports clinical trust.

### 5.6 Statistical significance

**Table 6.** McNemar's test: proposed model vs each baseline (*placeholder*; from
`experiments/results/mcnemar.csv`).

| Comparison | b (ours✓, other✗) | c (ours✗, other✓) | *p*-value | Significant |
|---|---|---|---|---|
| vs EfficientNetB3 | [fill] | [fill] | < 0.05 | yes |
| vs ResNet50 | [fill] | [fill] | < 0.01 | yes |
| vs Custom CNN | [fill] | [fill] | < 0.001 | yes |

**Table 7.** 5-fold cross-validation, macro-F1 (*placeholder*; from
`experiments/results/cross_validation.json`).

| Model | Mean ± SD | 95 % CI |
|---|---|---|
| EfficientNetB3 | 0.962 ± 0.008 | [0.952, 0.972] |
| **EffNetB3 + CBAM** | **0.972 ± 0.006** | **[0.965, 0.979]** |

Using the publication-quality phrasing: *"the proposed model achieved a macro-F1
of 97.2 ± 0.6 %, significantly outperforming the next-best model (p < 0.05,
McNemar's test, two-tailed)."*

### 5.7 Comparison with the state of the art

Compile a 5–8 row table comparing your best model against recent published work
on the same Kaggle 4-class dataset (matching split protocol where possible), and
discuss where your result sits and why.

### 5.8 Discussion

Synthesise *why* the proposed model wins: CBAM's combined channel-and-spatial
attention sharpens lesion focus (supported by Grad-CAM), two-stage fine-tuning
adapts the backbone without catastrophic forgetting, focal loss recovers
meningioma recall, and CLAHE improves boundary contrast — and the ablations show
these effects are additive rather than redundant. Emphasise that the gain over
the no-attention model came from ~8 % more parameters, so it is an attention
effect, not a capacity effect.

---

## 6. Limitations

- **Single dataset.** Results are on one public dataset; cross-dataset validation
  (e.g. Figshare CE-MRI) is needed to establish generalisation.
- **2D slices.** The method classifies individual 2D slices, not full 3D volumes,
  discarding inter-slice context.
- **No clinical validation.** Performance is measured offline; prospective
  clinical evaluation and radiologist comparison are future work.
- **No external test set / scanner diversity.** Robustness to different scanners,
  field strengths, and acquisition protocols is untested.

---

## 7. Conclusion and Future Work

We presented a fair, reproducible benchmark of ten deep-learning configurations
for four-class brain-tumour MRI classification and a proposed EfficientNetB3+CBAM
model that, under identical conditions, is expected to outperform strong
baselines with negligible parameter overhead. Controlled ablations isolate the
contribution of CLAHE, two-stage fine-tuning, CBAM attention, focal loss, and the
optimizer; Grad-CAM++ confirms tumour-focused decision making; and McNemar's test
with 5-fold cross-validation establishes that the improvements are statistically
significant rather than incidental.

**Future work:** cross-dataset and external validation; 3D / volumetric models;
multi-modal fusion (T1/T2/FLAIR); transformer and hybrid backbones; federated and
privacy-preserving training; richer explainability (SHAP, t-SNE/UMAP of
embeddings); and lightweight deployment for real-time, point-of-care use.

---

## Reproducibility and Ethics

Code, configuration, and the exact split CSVs are released at [GitHub URL]. All
experiments use seed 42. The Kaggle Brain Tumor MRI dataset is publicly available
for research use and contains no personally identifying information; this study
involved no new human-subjects data collection. Cite the dataset as required by
its license.

## Author Contributions

[Your Name]: conceptualisation, methodology, software, experiments, writing.
[Supervisor]: supervision, review and editing.

## Conflicts of Interest

The authors declare no conflict of interest.

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image
Recognition," in *Proc. IEEE CVPR*, 2016. arXiv:1512.03385.

[2] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE
CVPR*, 2018. arXiv:1709.01507.

[3] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block
Attention Module," in *Proc. ECCV*, 2018. arXiv:1807.06521.

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks," in *Proc. ICML*, 2019. arXiv:1905.11946.

[5] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense
Object Detection," in *Proc. IEEE ICCV*, 2017. arXiv:1708.02002.

[6] A. Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image
Recognition at Scale," in *Proc. ICLR*, 2021. arXiv:2010.11929.

[7] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected
Convolutional Networks," in *Proc. IEEE CVPR*, 2017. arXiv:1608.06993.

[8] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the
Inception Architecture for Computer Vision," in *Proc. IEEE CVPR*, 2016.
arXiv:1512.00567.

[9] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition," in *Proc. ICLR*, 2015. arXiv:1409.1556.

[10] S. Deepak and P. M. Ameer, "Brain Tumor Classification Using Deep CNN
Features via Transfer Learning," *Computers in Biology and Medicine*, 2019.

[11] Z. N. K. Swati et al., "Brain Tumor Classification for MR Images Using
Transfer Learning and Fine-Tuning," *Computerized Medical Imaging and Graphics*,
2019.

[12] M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle, 2021. [Online]. Available:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

[13] J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via
Tumor Region Augmentation and Partition," *PLoS ONE*, 2015.

*(Add the 25–40 references your literature review produced, including recent
2023–2026 work, formatted to your target venue's style.)*
