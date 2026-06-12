"""Phase 9 — statistical significance testing. McNemar's test between the
proposed model and every other model, tabulated with p-values.

    python scripts/06_stats.py
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import pandas as pd

from src.evaluation.stats_tests import mcnemar_matrix
from src.utils import load_config


def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    summary = json.loads((results_dir / "summary.json").read_text())
    model_names = [r["model"] for r in summary]
    reference = cfg["proposed_model"]

    rows = mcnemar_matrix(results_dir, model_names, reference)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "mcnemar.csv", index=False)
    (results_dir / "mcnemar.md").write_text(df.to_markdown(index=False, floatfmt=".4g"))
    print(df.to_string(index=False))
    print(f"\nMcNemar results -> {results_dir}/mcnemar.csv")


if __name__ == "__main__":
    main()
