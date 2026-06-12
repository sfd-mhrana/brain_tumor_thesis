"""Phase 7 — assemble the master comparison table, confusion matrices, ROC
curves and the comparison bar chart from saved results. Also builds the top-k
soft-voting ensemble.

    python scripts/03_evaluate_all.py
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.ensemble import pick_topk, soft_voting
from src.evaluation.plots import (plot_confusion_matrix, plot_model_comparison,
                                  plot_multiclass_roc)
from src.utils import load_config


def main():
    cfg = load_config()
    classes = cfg["data"]["classes"]
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"])

    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit("Run scripts/02_train_all.py first (no summary.json).")
    results = json.loads(summary_path.read_text())

    # Master comparison table
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "auc_roc_ovr", "kappa", "params_M", "inference_ms"]
    df = pd.DataFrame(results)[cols].sort_values("f1_macro", ascending=False)
    df.to_csv(results_dir / "master_comparison.csv", index=False)
    print(df.to_string(index=False))

    # Markdown table for the paper
    md = df.to_markdown(index=False, floatfmt=".4f")
    (results_dir / "master_comparison.md").write_text(md)

    # Comparison bar chart
    plot_model_comparison(results, fig_dir / "model_comparison_f1.png", "f1_macro")

    # Confusion matrices + ROC for the top 4 models
    top4 = df["model"].head(4).tolist()
    for name in top4:
        res = next(r for r in results if r["model"] == name)
        plot_confusion_matrix(res["confusion_matrix"], classes,
                              fig_dir / f"cm_{name}.png",
                              title=f"Confusion Matrix — {name}")
        preds = np.load(results_dir / f"{name}_preds.npz")
        plot_multiclass_roc(preds["y_true"], preds["y_prob"], classes,
                            fig_dir / f"roc_{name}.png",
                            title=f"ROC (OvR) — {name}")

    # Top-k soft-voting ensemble (AB-7)
    topk = pick_topk(results, k=cfg["ensemble_topk"])
    ens = soft_voting(results_dir, topk)
    with open(results_dir / "ensemble.json", "w") as f:
        json.dump(ens, f, indent=2)
    print(f"\nEnsemble of {topk}: acc={ens['accuracy']}  macroF1={ens['f1_macro']}")
    print(f"Figures + tables written to {fig_dir}/ and {results_dir}/")


if __name__ == "__main__":
    main()
