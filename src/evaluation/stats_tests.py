"""Statistical significance testing: McNemar's test between model pairs and
paired comparisons across cross-validation folds. Turns "model A scored higher"
into "model A is significantly better (p<0.05)".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact McNemar's test on the discordant pairs of two classifiers."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))    # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))    # B right, A wrong
    table = [[0, b], [c, 0]]
    res = mcnemar(table, exact=(b + c) < 25)
    return {"b": b, "c": c, "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
            "significant": bool(res.pvalue < 0.05)}


def mcnemar_matrix(results_dir, model_names, reference):
    """McNemar p-values comparing `reference` against every other model."""
    results_dir = Path(results_dir)
    ref = np.load(results_dir / f"{reference}_preds.npz")
    y_true, ref_pred = ref["y_true"], ref["y_pred"]
    rows = []
    for name in model_names:
        if name == reference:
            continue
        other = np.load(results_dir / f"{name}_preds.npz")
        r = mcnemar_test(y_true, ref_pred, other["y_pred"])
        rows.append({"reference": reference, "vs": name, **r})
    return rows


def paired_ttest(scores_a, scores_b):
    """Paired t-test + non-parametric Wilcoxon on K-fold scores."""
    t_stat, t_p = stats.ttest_rel(scores_a, scores_b)
    try:
        w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {"t_stat": float(t_stat), "t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p)}


def mean_ci(scores, confidence=0.95):
    """Mean and 95% confidence interval of cross-validation scores."""
    scores = np.asarray(scores)
    m = scores.mean()
    sem = stats.sem(scores)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    return {"mean": float(m), "std": float(scores.std(ddof=1)),
            "ci_low": float(m - h), "ci_high": float(m + h)}
