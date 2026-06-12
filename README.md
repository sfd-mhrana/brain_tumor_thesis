# Brain Tumor MRI Classification — Thesis Project

Comparative benchmarking of 10 deep-learning architectures for 4-class brain
tumour MRI classification (glioma · meningioma · pituitary · no-tumour), with a
proposed **EfficientNetB3 + CBAM** model, full ablations, Grad-CAM explainability
and statistical significance testing.

This repository is the complete, runnable implementation of the
[thesis master guide](../Brain_Tumor_MRI_Thesis_Master_Guide.md). It is built on
**PyTorch + [`timm`](https://github.com/huggingface/pytorch-image-models)**, which
exposes every backbone behind one uniform API so the comparison is fair by
construction.

> **Important — results are not yet real.** No training has been run in this
> repository (it ships with code only, no dataset/GPU). The tables in
> [`paper/paper.md`](paper/paper.md) marked *(placeholder)* contain reference
> values from the literature so the manuscript is complete and readable. Replace
> them with the numbers produced by your own runs (`experiments/results/`)
> **before submitting anything**. Reporting unrun numbers as real is research
> misconduct.

## 1. Setup

This project uses a **project-local virtual environment** (`venv`). All packages
install into `./.venv/` inside this folder — nothing touches your system Python,
and deleting the folder removes everything.

**Python version: 3.10 or 3.11** (3.11 recommended). The pinned dependencies do
*not* support 3.9 (`scipy` requires ≥3.10) or 3.12+ (`torch==2.2.2` has no
wheels). On macOS, install 3.11 side-by-side without disturbing the system Python:

```bash
brew install python@3.11                 # one-time; lives in /opt/homebrew
```

Create and activate the environment, then install dependencies:

```bash
cd brain_tumor_thesis
python3.11 -m venv .venv                  # creates ./.venv/  (your "node_modules")
source .venv/bin/activate                 # prompt shows (.venv); use deactivate to exit
pip install --upgrade pip
pip install -r requirements.txt           # installs into ./.venv/, ~2-3 GB
```

> On Linux/Windows, replace `python3.11` with your 3.10/3.11 interpreter
> (`py -3.11` on Windows) and activate with `.venv\Scripts\activate`.

Re-activate `source .venv/bin/activate` in every new terminal before running any
script. To rebuild from scratch: `rm -rf .venv` and repeat the steps above.

### Apple Silicon (M1–M4) note

Training runs on the Mac GPU via PyTorch's **MPS** backend, which the code
auto-detects (`src/utils.py`). The CPU-fallback flag for any op Metal doesn't yet
support is **set automatically in `src/__init__.py`** (`PYTORCH_ENABLE_MPS_FALLBACK`),
so there is nothing to export — it's scoped to this project, not your shell. If
you ever run torch outside this package, set it yourself with
`export PYTORCH_ENABLE_MPS_FALLBACK=1`.

Local runs are fine for the smoke test, EDA, and single-model trials; for the
full 10-model sweep + ablations + 5-fold CV, a free cloud GPU (Colab / Kaggle) is
much faster.

## 2. Get the data

```bash
pip install kaggle              # put kaggle.json in ~/.kaggle/
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/raw/
# Expected layout: data/raw/Training/<class>/*.jpg  and  data/raw/Testing/<class>/*.jpg
```

## 3. Verify the pipeline (no data needed)

```bash
python tests/test_smoke.py      # synthetic data, ~30 s on CPU -> "SMOKE TEST PASSED"
```

## 4. Run the full study

```bash
python scripts/00_make_splits.py        # one shared stratified 70/15/15 split
python scripts/01_eda.py                # EDA figures for Chapter 3
python scripts/02_train_all.py          # train + evaluate all 10 models
python scripts/03_evaluate_all.py       # master table, confusion matrices, ROC, ensemble
python scripts/04_ablations.py --study all   # AB1..AB6
python scripts/05_gradcam.py            # headline Grad-CAM++ grid
python scripts/06_stats.py              # McNemar's test vs proposed model
python scripts/07_cross_validation.py   # 5-fold CV (mean ± std, 95% CI)
```

Train a single model: `python scripts/02_train_all.py --models efficientnet_b3_cbam`.
Disable Weights & Biases logging with `--no-wandb`.

Everything is driven by [`config.yaml`](config.yaml) — change hyperparameters
there once and every script picks them up.

## 5. Repository layout

```
brain_tumor_thesis/
├── .venv/                      # local virtual environment (created by you; git-ignore it)
├── config.yaml                 # central hyperparameters
├── requirements.txt
├── src/
│   ├── utils.py                # seeding, config, device
│   ├── losses.py               # focal loss / weighted CE
│   ├── train.py                # two-stage fine-tuning engine
│   ├── data/
│   │   ├── preprocessing.py    # CLAHE
│   │   └── dataset.py          # Dataset, transforms, shared splits
│   ├── models/
│   │   ├── attention.py        # SE block, CBAM
│   │   ├── custom_cnn.py       # from-scratch baseline
│   │   └── build.py            # model factory (10 architectures)
│   └── evaluation/
│       ├── metrics.py          # full metric suite
│       ├── plots.py            # 300-DPI figures
│       ├── gradcam.py          # Grad-CAM++ grid
│       ├── ensemble.py         # top-k soft voting
│       └── stats_tests.py      # McNemar, paired t-test, CI
├── scripts/                    # 00..07 ordered pipeline
├── tests/test_smoke.py         # synthetic end-to-end check
└── paper/paper.md              # the manuscript (Markdown)
```

## 6. Models benchmarked

| Model | Role |
|---|---|
| Custom CNN | performance floor (from scratch) |
| VGG16 · ResNet50 · InceptionV3 · DenseNet121 | classic transfer-learning baselines |
| EfficientNetB0 · EfficientNetB3 | efficient backbones |
| ResNet50 + SE | channel attention |
| **EfficientNetB3 + CBAM** | **proposed model** (channel + spatial attention) |
| Soft-voting ensemble (top-3) | upper bound |

## 7. Reproducibility

Seed 42 is set in every script via `src.utils.set_seed` (Python, NumPy, PyTorch,
cuDNN deterministic). The split CSVs are written once and shared by all models.
Cite the dataset and include an ethical-use statement in the paper.
