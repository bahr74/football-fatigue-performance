"""Historical Step 6/7 results, read from the saved checksummed scoring summaries (never recomputed, never hard-coded).

These numbers describe the frozen model on the validation and locked-test matches. They are not computed from the encounter
selected in the dashboard, and they are provider-reference agreement, not verified identity accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_fatigue.matching import validate
from football_fatigue.matching.locked_test import TEST_DIR

from .data import DashboardError, frozen_spec, processed_root

HI = "high_confidence_reference"
BASELINES = (("nearest", "Nearest position"), ("constant_velocity", "Constant velocity"), ("hungarian_cv", "Hungarian (CV cost)"),
             ("chain_during", "Chaining"), ("e0_adapted", "E0-Adapted (FOOTBALLTrace-inspired)"), ("step4_rf@frozen", "Step 4 Random Forest"),
             ("frozen@relaxed", "Frozen model, relaxed"), ("frozen@conservative", "Frozen model, conservative"))


def _summary(d: Path) -> dict | None:
    import json

    if not (d / "manifest.json").exists():
        return None
    try:
        validate.verify_dir(d)
    except Exception as e:
        raise DashboardError(f"{d}: {e}") from None
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


def _counts(incoming: int, correct: int, wrong: int) -> dict[str, int]:
    """Saved outcome counts behind the rates: coverage = matched / incoming, false association = wrong / matched."""
    return dict(incoming=int(incoming), matched=int(correct) + int(wrong), correct=int(correct), wrong=int(wrong))


def _pair_counts(pw: dict) -> dict[str, int] | None:
    """Saved candidate-pair counts behind the pairwise ROC-AUC (same-player / different-player pairs per provider reference)."""
    return dict(pairs=pw["pairs"], positives=pw["positives"], negatives=pw["negatives"]) if "pairs" in pw else None


def _split_row(s: dict) -> dict[str, Any]:
    enc, pw = s["encounter"][HI], s["pairwise"][HI]["frozen"]
    out = dict(matches=len(s["matches"]), incoming=enc["frozen@relaxed"]["all"]["incoming"], roc_auc=pw["roc_auc"],
               roc_auc_ci95=pw.get("roc_auc_match_bootstrap_ci95"), pairs=_pair_counts(pw))
    for pt, crit in (("relaxed", "primary"), ("conservative", "secondary")):
        a = enc[f"frozen@{pt}"]["all"]
        b = s["criteria"][crit]["bootstrap"]
        out[pt] = dict(coverage=a["coverage"], false_association_rate=a["false_association_rate"], coverage_ci95=b["coverage_ci95"],
                       false_association_ci95=b["false_association_ci95"], criterion_met=s["criteria"][crit]["met"],
                       counts=_counts(a["incoming"], a["counts"]["correct_match"], a["counts"]["wrong_match"]))
    out["baselines"] = {name: dict(label=label, coverage=enc[name]["all"]["coverage"], false_association_rate=enc[name]["all"]["false_association_rate"])
                        for name, label in BASELINES if name in enc}
    return out


def research_results(raw_root) -> dict[str, Any]:
    """Development (leave-one-match-out), validation and locked-test rows; a split whose artifacts are missing is ``None``."""
    proc = processed_root(raw_root)
    spec = frozen_spec(raw_root)
    dev = spec["development_reference"]
    rows: dict[str, Any] = dict(development=dict(matches=len(spec["splits"]["development"]), roc_auc=dev["pairwise"]["roc_auc"], roc_auc_ci95=None,
                                                 incoming=dev["operating"]["relaxed"]["incoming"], pairs=_pair_counts(dev["pairwise"]),
                                                 **{pt: dict(coverage=dev["operating"][pt]["coverage"],
                                                             false_association_rate=dev["operating"][pt]["false_association_rate"],
                                                             coverage_ci95=dev["operating"][pt]["bootstrap"]["coverage_ci95"],
                                                             false_association_ci95=dev["operating"][pt]["bootstrap"]["false_association_ci95"],
                                                             criterion_met=None,
                                                             counts=_counts(dev["operating"][pt]["incoming"], dev["operating"][pt]["correct"],
                                                                            dev["operating"][pt]["wrong"])) for pt in ("relaxed", "conservative")}))
    sources = {"development": str(Path(validate.FROZEN_DIR) / "spec.json")}
    for name, d in (("validation", proc / validate.VALIDATION_DIR / "scoring"), ("test", proc / TEST_DIR / "scoring")):
        s = _summary(d)
        rows[name] = None if s is None else _split_row(s)
        sources[name] = str(d.relative_to(proc)).replace("\\", "/") + "/summary.json"
    return dict(rows=rows, sources=sources, operating_points=spec["operating_points"],
                note="all rates are agreement with the provisional provider identity reference, not verified identity accuracy")
