"""Phase 7 — generate the headline Grad-CAM++ grid: one MRI per class, comparing
the original image, a baseline (ResNet50) and the proposed EfficientNetB3+CBAM.

    python scripts/05_gradcam.py
"""
import _bootstrap  # noqa: F401
from pathlib import Path

import torch

from src.data.dataset import make_loaders
from src.evaluation.gradcam import gradcam_grid
from src.models.build import build_model
from src.utils import get_device, load_config


def _load(name, cfg, device):
    model = build_model(name, cfg, pretrained=False).to(device)
    ckpt = Path(cfg["paths"]["weights_dir"]) / f"{name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def main():
    cfg = load_config()
    device = get_device()
    classes = cfg["data"]["classes"]
    _, _, test_loader = make_loaders(cfg)

    # one correctly-labelled sample per class
    samples, seen = [], set()
    for x, y in test_loader:
        for i in range(x.size(0)):
            c = int(y[i])
            if c not in seen:
                samples.append((x[i:i + 1], c))
                seen.add(c)
        if len(seen) == len(classes):
            break
    samples.sort(key=lambda s: s[1])

    baseline = cfg.get("gradcam_baseline", "resnet50")
    models = {"ResNet50 (baseline)": _load(baseline, cfg, device),
              "EffNetB3+CBAM (ours)": _load(cfg["proposed_model"], cfg, device)}

    out = Path(cfg["paths"]["figures_dir"]) / "gradcam_grid.png"
    gradcam_grid(models, samples, classes, device, out)
    print(f"Grad-CAM grid -> {out}")


if __name__ == "__main__":
    main()
