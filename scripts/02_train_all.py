"""Phases 5-6 — train every model in the registry with the identical two-stage
protocol, then evaluate each on the shared test set.

    python scripts/02_train_all.py                  # all models
    python scripts/02_train_all.py --models resnet50 efficientnet_b3_cbam
    python scripts/02_train_all.py --no-wandb
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from src.data.dataset import make_loaders
from src.evaluation.metrics import evaluate
from src.evaluation.plots import plot_training_curves
from src.models.build import MODEL_REGISTRY
from src.train import train_model
from src.utils import ensure_dir, get_device, load_config, set_seed


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    classes = cfg["data"]["classes"]

    # Build the shared loaders ONCE so every model sees identical data.
    loaders = make_loaders(cfg)

    summary = []
    for name in args.models:
        print(f"\n{'='*70}\nTraining {name}\n{'='*70}")
        wandb_run = None
        if not args.no_wandb:
            try:
                import wandb
                wandb_run = wandb.init(project="brain-tumor-thesis", name=name,
                                       config=cfg, reinit=True)
            except Exception as e:  # noqa: BLE001
                print(f"  wandb disabled ({e})")

        out = train_model(name, cfg, loaders=loaders, wandb_run=wandb_run)
        res = evaluate(out["model"], loaders[2], device, name, classes,
                       params_M=out["params_M"], train_time_s=out["train_time_s"],
                       tta=cfg["eval"]["tta"], results_dir=results_dir)
        plot_training_curves(out["history"], Path(fig_dir) / f"curves_{name}.png")
        with open(Path(results_dir) / f"{name}_history.json", "w") as f:
            json.dump(out["history"], f, indent=2)
        summary.append(res)
        print(f"  {name}: acc={res['accuracy']}  macroF1={res['f1_macro']}  "
              f"AUC={res['auc_roc_ovr']}")
        if wandb_run is not None:
            wandb_run.finish()

    with open(Path(results_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll done. Summary -> {results_dir}/summary.json")


if __name__ == "__main__":
    main()
