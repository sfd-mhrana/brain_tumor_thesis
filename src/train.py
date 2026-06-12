"""Training engine: two-stage fine-tuning, AdamW, cosine/plateau scheduling,
mixed precision, gradient clipping, early stopping. Used identically for every
model so that only the architecture varies.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .data.dataset import compute_class_weights, make_loaders
from .losses import build_loss
from .models.build import build_model, count_params
from .utils import ensure_dir, get_device, set_seed


def _make_optimizer(cfg, params, lr):
    name = cfg["train"]["optimizer"]
    wd = cfg["train"]["weight_decay"]
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'")


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def _run_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
               epochs, device, cfg, history, best, patience_left, ckpt_path,
               wandb_run=None, stage=""):
    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = cfg["train"]["grad_clip"]

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        history.append({"stage": stage, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{stage}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if wandb_run is not None:
            wandb_run.log({"train_loss": train_loss, "val_loss": val_loss,
                           "val_acc": val_acc})

        if val_loss < best["val_loss"] - 1e-4:
            best["val_loss"] = val_loss
            best["val_acc"] = val_acc
            best["state"] = copy.deepcopy(model.state_dict())
            patience_left[0] = cfg["train"]["early_stopping_patience"]
            torch.save(best["state"], ckpt_path)
        else:
            patience_left[0] -= 1
            if patience_left[0] <= 0:
                print(f"[{stage}] early stopping triggered")
                break
    return history, best


def train_model(name, cfg, loaders=None, wandb_run=None, pretrained=True):
    """Train one model with the two-stage protocol and return a result dict."""
    device = get_device()
    set_seed(cfg["seed"])

    if loaders is None:
        train_loader, val_loader, test_loader = make_loaders(cfg)
    else:
        train_loader, val_loader, test_loader = loaders

    model = build_model(name, cfg, pretrained=pretrained).to(device)
    class_weights = compute_class_weights(cfg).to(device)
    criterion = build_loss(cfg, class_weights=class_weights)

    weights_dir = ensure_dir(cfg["paths"]["weights_dir"])
    ckpt_path = Path(weights_dir) / f"{name}.pt"

    history = []
    best = {"val_loss": float("inf"), "val_acc": 0.0, "state": None}
    patience_left = [cfg["train"]["early_stopping_patience"]]
    t0 = time.time()

    is_transfer = name != "custom_cnn"

    # ---- Stage 1: frozen backbone, train head ----
    if is_transfer:
        model.freeze_backbone()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = _make_optimizer(cfg, params, cfg["train"]["lr"]["head"])
        sched = (CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"]["head"])
                 if cfg["train"]["scheduler"] == "cosine"
                 else ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-7))
        history, best = _run_stage(
            model, train_loader, val_loader, criterion, opt, sched,
            cfg["train"]["epochs"]["head"], device, cfg, history, best,
            patience_left, ckpt_path, wandb_run, stage="head")

    # ---- Stage 2: unfreeze top blocks, fine-tune (or full training for custom CNN) ----
    if is_transfer:
        model.unfreeze_last(cfg["train"]["finetune_unfreeze_blocks"])
        lr = cfg["train"]["lr"]["finetune"]
        epochs = cfg["train"]["epochs"]["finetune"]
        stage = "finetune"
    else:
        lr = cfg["train"]["lr"]["head"]
        epochs = cfg["train"]["epochs"]["head"] + cfg["train"]["epochs"]["finetune"]
        stage = "scratch"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = _make_optimizer(cfg, params, lr)
    sched = (CosineAnnealingLR(opt, T_max=epochs)
             if cfg["train"]["scheduler"] == "cosine"
             else ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-7))
    patience_left[0] = cfg["train"]["early_stopping_patience"]
    history, best = _run_stage(
        model, train_loader, val_loader, criterion, opt, sched,
        epochs, device, cfg, history, best, patience_left, ckpt_path,
        wandb_run, stage=stage)

    train_time = time.time() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return {
        "name": name,
        "model": model,
        "history": history,
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "train_time_s": round(train_time, 1),
        "params_M": round(count_params(model), 2),
        "ckpt": str(ckpt_path),
        "loaders": (train_loader, val_loader, test_loader),
    }
