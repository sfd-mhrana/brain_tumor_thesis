"""Phase 8 — controlled ablation studies. Each study changes exactly ONE factor
relative to the proposed configuration and reports the macro-F1 delta.

    python scripts/04_ablations.py --study AB4        # attention: none/SE/CBAM
    python scripts/04_ablations.py --study all

Studies:
  AB1 preprocessing : CLAHE on/off
  AB2 augmentation  : none / standard / heavy
  AB3 finetuning    : 1-stage vs 2-stage
  AB4 attention     : none / SE / CBAM
  AB5 loss          : ce / weighted_ce / focal
  AB6 optimizer     : sgd / adam / adamw

AB7 (ensemble: best single vs top-3 soft voting) is produced by
scripts/03_evaluate_all.py, which already builds the soft-voting ensemble.
"""
import _bootstrap  # noqa: F401
import argparse
import copy
import json
from pathlib import Path

from src.data.dataset import make_loaders
from src.evaluation.metrics import evaluate
from src.train import train_model
from src.utils import ensure_dir, get_device, load_config, set_seed

PROPOSED = "efficientnet_b3_cbam"


def _run(name, cfg, tag, results_dir, device, classes, clahe=None, augment=True):
    set_seed(cfg["seed"])
    loaders = make_loaders(cfg, clahe_override=clahe, augment=augment)
    out = train_model(name, cfg, loaders=loaders)
    res = evaluate(out["model"], loaders[2], device, tag, classes,
                   params_M=out["params_M"], results_dir=results_dir)
    print(f"  [{tag}] macroF1={res['f1_macro']}  per-class={res['per_class_f1']}")
    return res


def study_ab1(cfg, *a):  # CLAHE on/off
    return [_run(PROPOSED, cfg, "AB1_no_clahe", *a, clahe=False),
            _run(PROPOSED, cfg, "AB1_clahe", *a, clahe=True)]


def study_ab2(cfg, results_dir, device, classes):  # augmentation intensity
    out = []
    base = copy.deepcopy(cfg)
    out.append(_run(PROPOSED, base, "AB2_no_aug", results_dir, device, classes,
                    augment=False))
    out.append(_run(PROPOSED, base, "AB2_standard", results_dir, device, classes))
    heavy = copy.deepcopy(cfg)
    heavy["augment"].update(rotation_deg=30, zoom=0.2, brightness=0.3,
                            random_erasing=0.25)
    out.append(_run(PROPOSED, heavy, "AB2_heavy", results_dir, device, classes))
    return out


def study_ab3(cfg, results_dir, device, classes):  # fine-tuning strategy
    one_stage = copy.deepcopy(cfg)
    one_stage["train"]["epochs"]["head"] = 0   # skip frozen stage
    return [_run(PROPOSED, one_stage, "AB3_one_stage", results_dir, device, classes),
            _run(PROPOSED, cfg, "AB3_two_stage", results_dir, device, classes)]


def study_ab4(cfg, results_dir, device, classes):  # attention module
    return [_run("efficientnet_b3", cfg, "AB4_none", results_dir, device, classes),
            _run("resnet50_se", cfg, "AB4_se_on_resnet", results_dir, device, classes),
            _run(PROPOSED, cfg, "AB4_cbam", results_dir, device, classes)]


def study_ab5(cfg, results_dir, device, classes):  # loss function
    out = []
    for loss in ("ce", "weighted_ce", "focal"):
        c = copy.deepcopy(cfg); c["train"]["loss"] = loss
        out.append(_run(PROPOSED, c, f"AB5_{loss}", results_dir, device, classes))
    return out


def study_ab6(cfg, results_dir, device, classes):  # optimizer
    out = []
    for opt in ("sgd", "adam", "adamw"):
        c = copy.deepcopy(cfg); c["train"]["optimizer"] = opt
        out.append(_run(PROPOSED, c, f"AB6_{opt}", results_dir, device, classes))
    return out


STUDIES = {"AB1": study_ab1, "AB2": study_ab2, "AB3": study_ab3,
           "AB4": study_ab4, "AB5": study_ab5, "AB6": study_ab6}


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="all")
    args = parser.parse_args()
    device = get_device()
    classes = cfg["data"]["classes"]
    results_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "ablations")

    todo = STUDIES if args.study == "all" else {args.study.upper(): STUDIES[args.study.upper()]}
    all_results = {}
    for key, fn in todo.items():
        print(f"\n{'='*70}\nAblation {key}\n{'='*70}")
        all_results[key] = fn(cfg, results_dir, device, classes)

    with open(results_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results -> {results_dir}/ablation_summary.json")


if __name__ == "__main__":
    main()
