"""Grad-CAM++ explainability. For a medical-imaging paper this is mandatory: it
shows the model attends to the tumour region rather than scanner artefacts, and
visually demonstrates the benefit of the attention module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _target_layer(model):
    """Best convolutional layer for Grad-CAM: the last conv of the backbone."""
    backbone = getattr(model, "backbone", model)
    convs = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return convs[-1]


def gradcam_overlay(model, input_tensor, class_idx, device):
    """Return an (H, W, 3) RGB overlay using grad-cam library (Grad-CAM++)."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAMPlusPlus(model=model, target_layers=[_target_layer(model)])
    grayscale = cam(input_tensor=input_tensor.to(device),
                    targets=[ClassifierOutputTarget(class_idx)])[0]

    # De-normalise the input image for display.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = input_tensor[0].cpu().numpy().transpose(1, 2, 0) * std + mean
    img = np.clip(img, 0, 1)

    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(img.astype(np.float32), grayscale, use_rgb=True)


def gradcam_grid(models_dict, samples, classes, device, path):
    """Build the headline 4x(N+1) figure.

    models_dict : {display_name: model}
    samples     : list of (input_tensor[1,C,H,W], true_class_idx), one per class
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(len(samples), n_models + 1,
                             figsize=(3 * (n_models + 1), 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]

    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    for r, (x, cls_idx) in enumerate(samples):
        orig = x[0].cpu().numpy().transpose(1, 2, 0) * std + mean
        axes[r, 0].imshow(np.clip(orig, 0, 1))
        axes[r, 0].set_ylabel(classes[cls_idx], fontsize=12)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Original")
        for c, (mname, model) in enumerate(models_dict.items(), start=1):
            overlay = gradcam_overlay(model, x, cls_idx, device)
            axes[r, c].imshow(overlay)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(mname)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
