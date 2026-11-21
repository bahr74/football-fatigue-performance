"""M11 step 6: freeze the selected classifier, then one preregistered evaluation on the frozen validation matches.

    python -m football_fatigue.matching freeze   --matches <10 development ids>
    python -m football_fatigue.matching valbuild --match <validation id>       # Step 2-3 benchmark of one validation match
    python -m football_fatigue.matching predict
    python -m football_fatigue.matching score

The stages are separate and ordered, and each writes its own directory once:

1. ``freeze`` (development matches only) refits the Step 5 configuration without the provider confidence codes, fits its
   Platt calibrator on inner grouped out-of-fold predictions, checks the two preselected operating points on those inner
   predictions and writes ``m11-frozen-v1`` (model, specification, preregistered criteria, SHA-256 manifest).
2. ``predict`` reads only the matcher-visible task B instances of the validation matches (never a reference file), applies
   the frozen models with no fitting of any kind, and writes ``m11-validation-v1/predictions`` with checksums.
3. ``score`` verifies those checksums, then reads the evaluation-only references and baseline predictions and writes
   ``m11-validation-v1/scoring``.

Test matches are refused everywhere. Every rate is agreement with the provider identity reference, not verified identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.splits import MANIFEST_NAME
from football_fatigue.ingestion import IngestionError, approved_output_root

from . import ml, ml2
from .baselines import BASELINES
from .benchmark import OUTPUT_DIR as SOURCE_DIR, SCHEMA_VERSION as SOURCE_SCHEMA, TASK_B, _sha256, code_digest, output_paths
from .evaluation import evaluate, outcome

FROZEN_DIR = "m11-frozen-v1"
VALIDATION_DIR = "m11-validation-v1"
LABEL_TIER, EXPLORATORY_TIER, UNVERIFIED_TIER = ml.LABEL_TIER, ml.EXPLORATORY_TIER, ml.UNVERIFIED_TIER
EXCLUDED_FEATURES = ("conf_in", "conf_out")               # provider confidence codes: weak identity clue (Step 5, section 5.4)
FROZEN_FEATURES = tuple(f for f in ml2.FEATURE_SETS["core_g"] if f not in EXCLUDED_FEATURES)
FROZEN_KIND = "rf_reg"
FROZEN_PARAMS = dict(min_samples_leaf=5, max_features=0.33)  # chosen in every Step 5 fold, with and without conf_*; not searched again
OPERATING = {"conservative": dict(threshold=0.87, margin=1.0, target=0.05), "relaxed": dict(threshold=0.75, margin=1.0, target=0.10)}
STEP4_MODELS = ("rf", "rf_ft", "lr", "lr_ft", "lr_distance")
PRIMARY_BASELINE = "step4_rf"
BOOTSTRAP = 2000
CRITERIA = dict(
    primary=("at the relaxed point (0.75, margin 1.0) the pooled high-tier validation false-association rate is <= 0.10, AND its coverage is "
             "higher than the frozen Step 4 RF (m11-ml-v1 final rf, threshold 0.86, margin 0.5) with the lower bound of the 95% match-bootstrap "
             "interval of the paired coverage difference > 0"),
    secondary=("at the conservative point (0.87, margin 1.0) the pooled high-tier validation false-association rate is <= 0.05; reported "
               "separately, and its coverage difference to the frozen Step 4 RF with the same bootstrap"),
    reported_not_gated=("95% match-bootstrap intervals of coverage and false association; per-match rates; pairwise ROC-AUC, top-1, Brier, ECE; "
                        "strata by gap, sharp turn, crossing, player count and tier; lower tier exploratory; natural suspicious cases as decision "
                        "counts only"),
    failure="any primary condition not met; the secondary criterion is reported but does not decide the primary outcome",
    no_adjustment="thresholds, margins, calibration, features and model are fixed by this file; nothing is changed after validation is opened")
METRICS = dict(
    A_pairwise=("ROC-AUC, average precision, Brier, ECE (10 bins) and log-loss of calibrated pair probabilities over all incoming x outgoing "
                "pairs of high-tier instances; top-1: share of incoming tracklets with a reference outgoing tracklet whose highest-probability "
                "candidate is that tracklet"),
    B_encounter=("per incoming tracklet: correct_match (MATCHED to the reference), wrong_match (MATCHED to anything else), correct_unmatched, "
                 "missed_unmatched, uncertain; correct and wrong match rates are shares of incoming tracklets"),
    C_selective=("coverage = MATCHED / incoming; false-association rate = wrong_match / MATCHED"),
    D_complete="share of instances in which every incoming outcome is correct_match or correct_unmatched",
    E_strata="the Step 2 evaluator strata: gap bins <1.5, 1.5-2.5, 2.5-4, >=4 s; sharp turn; crossing; incoming count 2 / 3+; tier",
    uncertainty="match-level bootstrap, 2000 resamples of validation matches, seed 0; paired for differences",
    reference="all metrics are agreement with the provisional provider identity reference")


# ---------------------------------------------------------------- shared helpers

def _dump(obj: Any) -> bytes:
    return ml2._dump(obj)


def _splits(raw_root) -> tuple[dict, Path]:
    path = approved_output_root(raw_root) / ml.SPLITS_DIR / MANIFEST_NAME
    if not path.exists():
        raise IngestionError(f"the frozen split manifest {path} is missing")
    return json.loads(path.read_text(encoding="utf-8")), path


def check_split(split: dict, ids: list[str], expected: str) -> None:
    """Every ID must belong to ``expected``; a test match is refused whatever is expected."""
    test = [m for m in ids if split["matches"].get(m, {}).get("split") == "test"]
    if test:
        raise IngestionError(f"locked test matches are never accessed: {test}")
    bad = [m for m in ids if split["matches"].get(m, {}).get("split") != expected]
    if bad:
        raise IngestionError(f"only {expected} matches are allowed here; refused: {bad}")


def raw_snapshot(raw_root) -> dict[str, Any]:
    """Size and modification time of every file below the raw root, with one digest: evidence that raw data was not written."""
    root = Path(raw_root)
    rows = sorted((str(p.relative_to(root)).replace("\\", "/"), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())
    return dict(files=len(rows), digest=hashlib.sha256(json.dumps(rows).encode()).hexdigest())


def _write_dir(out: Path, blobs: dict[str, bytes], models: dict[str, Any], manifest: dict) -> Path:
    """Write a whole directory at once (temporary name, then rename); an existing directory is never replaced."""
    import joblib

    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    tmp = out.parent / f".{out.name}.partial-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        for name, b in blobs.items():
            with open(tmp / name, "xb") as f:
                f.write(b)
        for name, obj in models.items():
            joblib.dump(obj, tmp / name)
        manifest = dict(manifest, outputs_sha256={str(p.relative_to(tmp)).replace("\\", "/"): _sha256(p) for p in sorted(tmp.rglob("*")) if p.is_file()})
        with open(tmp / "manifest.json", "xb") as f:
            f.write(_dump(manifest))
        if out.exists():
            raise IngestionError(f"{out} appeared while writing; not overwritten")
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return out


def verify_dir(d: Path) -> dict:
    """The manifest of a written directory, after checking every listed file's SHA-256."""
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    bad = [k for k, v in m["outputs_sha256"].items() if _sha256(d / k) != v]
    if bad:
        raise IngestionError(f"{d}: checksum mismatch for {bad}")
    return m


def features_only(instances: list[dict], cfg: ml.MLConfig = ml.MLConfig()) -> tuple[list[dict], np.ndarray]:
    """Pair rows and the ``ml2.FEATURES2`` matrix from matcher-visible instances alone (no reference is read)."""
    rows, X = ml.build_features(instances, cfg)
    G = [ml2.football_features(i, cfg) for i in instances]
    G = np.vstack(G) if G else np.zeros((0, len(ml2.G_FEATURES)))
    return rows, np.hstack([X, G])


def decide(rows: list[dict], p: np.ndarray, X: np.ndarray, threshold: float, margin: float, cfg: ml.MLConfig = ml.MLConfig()) -> dict[str, list[dict]]:
    """Frozen one-to-one log-odds assignment and MATCHED / UNMATCHED / UNCERTAIN, with evidence for the candidate pair."""
    assigned = ml.instance_assignments(rows, p, cfg)
    ds = ml.decisions(assigned, threshold, replace(cfg, margin_logit=margin))
    for iid, dl in ds.items():
        a = assigned[iid]
        for i, d in enumerate(dl):
            k = a["rows"][i, a["outgoing"].index(d["candidate"])] if d["candidate"] is not None else None
            d["evidence"] = None if k is None else {f: ml._r(X[k, ml2.FEATURES2.index(f)], 3) for f in ml2.EVIDENCE_KEYS}
    return ds


# ---------------------------------------------------------------- 1. freeze (development only)

def _oof_platt(tables: list[dict], cfg: ml2.ML2Config) -> tuple[dict[str, np.ndarray], ml2.Platt, list[list[str]]]:
    """Inner grouped out-of-fold raw probabilities of the frozen configuration and the Platt calibrator fitted on them."""
    groups = ml2.match_groups([t["match_id"] for t in tables], cfg.inner_groups)
    c = ml2.cols(FROZEN_FEATURES)
    oof = {}
    for g in groups:
        model = ml2.fit_model(FROZEN_KIND, FROZEN_PARAMS, *ml2._xy([t for t in tables if t["match_id"] not in g], FROZEN_FEATURES), cfg)
        for t in tables:
            if t["match_id"] in g:
                oof[t["match_id"]] = ml2.proba(model, t["X"][:, c])
    ys = np.concatenate([np.array([lb["label"] for lb, k in zip(t["labels"], ml._train_mask(t)) if k], int) for t in tables])
    ps = np.concatenate([oof[t["match_id"]][ml._train_mask(t)] for t in tables])
    return oof, ml2.fit_platt(ps, ys), groups


def freeze_core(tables: list[dict], cfg: ml2.ML2Config = ml2.ML2Config(), dev_reference: bool = True) -> dict[str, Any]:
    """Fit the frozen model on all development tables; check the preselected operating points on inner out-of-fold predictions."""
    oof, cal, groups = _oof_platt(tables, cfg)
    assigned, refs = {}, {}
    for t in tables:
        assigned |= ml2.assigned_for(t, cal(oof[t["match_id"]]), ml2._hi(t), cfg)
        refs |= t["refs"]
    T = ml2.decision_table(assigned, refs)
    check, operating = {}, {}
    for name, op in OPERATING.items():
        r = ml2.table_rates(T, op["threshold"], op["margin"])
        alt = ml2.select_operating(T, op["target"], cfg)
        ok = r["false_association_rate"] is not None and r["false_association_rate"] <= op["target"]
        check[name] = dict(preselected=op, inner_oof_at_preselected=r, compatible=ok,
                           inner_selection_for_reference=dict(threshold=alt["threshold"], margin=alt["margin"], inner_oof=alt["inner_oof"]))
        # resolution rule fixed before validation: keep the preselected point if its inner false association meets its target
        operating[name] = dict(op) if ok else dict(threshold=alt["threshold"], margin=alt["margin"], target=op["target"])
    model = ml2.fit_model(FROZEN_KIND, FROZEN_PARAMS, *ml2._xy(tables, FROZEN_FEATURES), cfg)
    out = dict(model=model, calibrator=cal, inner_groups=groups, operating=operating, operating_check=check,
               inner_oof_pairwise={k: v for k, v in ml.pairwise_metrics(
                   np.concatenate([np.array([lb["label"] for lb, k in zip(t["labels"], ml2._hi(t)) if k], int) for t in tables]),
                   np.concatenate([cal(oof[t["match_id"]])[ml2._hi(t)] for t in tables]),
                   [r for t in tables for r, k in zip(t["rows"], ml2._hi(t)) if k]).items() if k != "reliability"})
    if dev_reference:
        out["development_reference"] = development_reference(tables, operating, cfg)
    return out


def development_reference(tables: list[dict], operating: dict, cfg: ml2.ML2Config) -> dict[str, Any]:
    """Leave-one-match-out over the development matches with the frozen configuration and the frozen operating points: the
    development numbers that validation is compared with (each fold's calibrator is fitted on its own training matches)."""
    c = ml2.cols(FROZEN_FEATURES)
    ys, ps, rows, per = [], [], [], {}
    Ts = {name: {} for name in operating}
    for h, t in enumerate(tables):
        train = [u for m, u in enumerate(tables) if m != h]
        _, cal, _ = _oof_platt(train, cfg)
        model = ml2.fit_model(FROZEN_KIND, FROZEN_PARAMS, *ml2._xy(train, FROZEN_FEATURES), cfg)
        p = cal(ml2.proba(model, t["X"][:, c]))
        hi = ml2._hi(t)
        y = np.array([lb["label"] for lb, k in zip(t["labels"], hi) if k], int)
        r = [x for x, k in zip(t["rows"], hi) if k]
        ys.append(y)
        ps.append(p[hi])
        rows += r
        per[t["match_id"]] = dict(roc_auc=ml.pairwise_metrics(y, p[hi], r)["roc_auc"])
        T = ml2.decision_table(ml2.assigned_for(t, p, hi, cfg), t["refs"])
        for name, op in operating.items():
            Ts[name][t["match_id"]] = ml2.table_rates(T, op["threshold"], op["margin"])
            per[t["match_id"]][name] = {k: Ts[name][t["match_id"]][k] for k in ("coverage", "false_association_rate", "matched", "incoming")}
    pw = ml.pairwise_metrics(np.concatenate(ys), np.concatenate(ps), rows)
    pooled = {}
    for name in operating:
        tot = {k: sum(v[k] for v in Ts[name].values()) for k in ("incoming", "matched", "correct", "wrong")}
        pooled[name] = dict(tot, coverage=tot["matched"] / tot["incoming"], false_association_rate=tot["wrong"] / tot["matched"] if tot["matched"] else None,
                            correct_match_rate=tot["correct"] / tot["incoming"], bootstrap=ml2.bootstrap(Ts[name], None, BOOTSTRAP, 0))
    return dict(pairwise={k: v for k, v in pw.items()}, operating=pooled, by_match=per,
                note="development leave-one-match-out at the frozen points; development data only")


def run_freeze(raw_root, dev_ids: list[str], *, source_root=None, out_root=None, cfg: ml2.ML2Config = ml2.ML2Config()) -> tuple[Path, dict]:
    t0 = time.perf_counter()
    out = Path(out_root) if out_root is not None else approved_output_root(raw_root) / FROZEN_DIR
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    split, split_path = _splits(raw_root)
    ids = [str(m) for m in dev_ids]
    check_split(split, ids, "development")
    snap = raw_snapshot(raw_root)
    data = [ml.load_match(raw_root, m, source_root) for m in ids]
    tables = [ml2.match_table2(d, cfg.base) for d in data]
    fz = freeze_core(tables, cfg)
    v1 = approved_output_root(raw_root) / "m11-ml-v1"
    v2 = approved_output_root(raw_root) / "m11-ml-v2"
    v1_thr = json.loads((v1 / "summary.json").read_text(encoding="utf-8"))["thresholds"]
    v2_final = json.loads((v2 / "summary.json").read_text(encoding="utf-8"))["final"]
    comparators = dict(
        step4={n: dict(path=str(v1 / "models" / "final" / f"{n}.joblib"), sha256=_sha256(v1 / "models" / "final" / f"{n}.joblib"),
                       features=list(ml.MODEL_FEATURES[n]), threshold=v1_thr[n]["final"]["threshold"], margin=ml.MLConfig().margin_logit, calibration="none")
               for n in STEP4_MODELS},
        step5_with_conf=dict(path=str(v2 / "models" / "final" / "rf_reg__core_g.joblib"), sha256=_sha256(v2 / "models" / "final" / "rf_reg__core_g.joblib"),
                             operating={"conservative": {k: v2_final["operating"]["0.05"][k] for k in ("threshold", "margin")},
                                        "relaxed": {k: v2_final["operating"]["0.1"][k] for k in ("threshold", "margin")}}, calibration="Platt (stored)"),
        baselines=list(BASELINES), published_footballtrace="E0-Literal protocol L (evaluation-side, needs provider-labelled trajectories; not a matcher)")
    spec = dict(
        schema=FROZEN_DIR, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"), task=TASK_B,
        splits=dict(manifest=str(split_path), manifest_sha256=_sha256(split_path), development=ids, validation=split["splits"]["validation"],
                    test_locked=split["splits"]["test"], test_access="never"),
        model=dict(kind=FROZEN_KIND, estimator="RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=0)",
                   params=FROZEN_PARAMS, features=list(FROZEN_FEATURES), n_features=len(FROZEN_FEATURES), excluded_features=list(EXCLUDED_FEATURES),
                   training=f"{LABEL_TIER} pairs without ESTIMATED endpoints of the {len(ids)} development matches",
                   calibration=dict(method="Platt (logistic on logit p) fitted on inner grouped out-of-fold predictions", groups=fz["inner_groups"],
                                    a=fz["calibrator"].a, b=fz["calibrator"].b),
                   decision=("one-to-one assignment maximising summed log-odds over reachable pairs; MATCHED if assigned, p >= threshold and margin >= "
                             "margin; UNMATCHED only when no outgoing tracklet is reachable; otherwise UNCERTAIN"),
                   provenance="Step 5 selection rf_reg/core_g (docs/research/m11_step5_model_improvement.md) with conf_in/conf_out removed"),
        operating_points=fz["operating"], operating_check=fz["operating_check"], inner_oof_pairwise=fz["inner_oof_pairwise"],
        development_reference=fz["development_reference"], comparators=comparators, metrics=METRICS, criteria=CRITERIA, bootstrap=dict(resamples=BOOTSTRAP, seed=0))
    bundle = dict(model=fz["model"], features=list(FROZEN_FEATURES), calibrator=asdict(fz["calibrator"]), operating=fz["operating"], params=FROZEN_PARAMS)
    manifest = dict(schema=FROZEN_DIR, development=ids, sources={d["match_id"]: d["provenance"] for d in data}, code_sha256=code_digest(),
                    raw_snapshot=snap, validation_or_test_matches_used=False, synthetic_swaps_injected=False,
                    processing=dict(seconds=round(time.perf_counter() - t0, 1)), python=platform.python_version(), numpy=np.__version__,
                    sklearn=__import__("sklearn").__version__)
    _write_dir(out, {"spec.json": _dump(spec)}, {"model.joblib": bundle}, manifest)
    m = verify_dir(out)
    return out, dict(spec=spec, model_sha256=m["outputs_sha256"]["model.joblib"])


# ---------------------------------------------------------------- 2. validation benchmark build (Steps 2-3 pipeline, unchanged)

def build_validation_match(raw_root, match_id, *, val_root=None) -> dict:
    """Run the unchanged Step 2-3 benchmark builder on one validation match's Step 1 catalogue, inside the validation directory."""
    from .benchmark import run_match

    split, _ = _splits(raw_root)
    mid = str(match_id)
    check_split(split, [mid], "validation")
    root = Path(val_root) if val_root is not None else approved_output_root(raw_root) / VALIDATION_DIR
    paths, summary = run_match(raw_root, mid, source_root=root / "encounters", out_root=root / "matching", split=split["matches"][mid])
    return dict(outputs={k: str(v) for k, v in paths.items()}, events=summary["audit"]["events"], accepted_by_tier=summary["audit"]["accepted_by_tier"],
                rejected=summary["audit"]["rejected"])


# ---------------------------------------------------------------- 3. predict (matcher-visible files only, no fitting)

VISIBLE = (".instances_b.jsonl",)


def load_visible(raw_root, match_id, split: dict, matching_root) -> tuple[list[dict], dict]:
    """Matcher-visible task B instances of one validation match, checksum-verified. No reference file is opened."""
    mid = str(match_id)
    check_split(split, [mid], "validation")
    paths = output_paths(raw_root, mid, matching_root)
    m = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
    if m.get("schema") != SOURCE_SCHEMA or str(m.get("match_id")) != mid or (m.get("split") or {}).get("split") != "validation":
        raise IngestionError(f"{paths['.manifest.json']} is not a {SOURCE_SCHEMA} validation output of match {mid}")
    if m.get("synthetic_swaps_injected") is not False or m.get("provider_smoothed_coordinates_used") is not False:
        raise IngestionError("the source is not a RAW, swap-free matching benchmark")
    p = paths[".instances_b.jsonl"]
    if m["outputs_sha256"].get(p.name) != _sha256(p):
        raise IngestionError(f"{p} does not match its manifest checksum")
    inst = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    return inst, dict(instances_b=_sha256(p), manifest=_sha256(paths[".manifest.json"]))


def load_frozen(frozen_dir: Path) -> tuple[dict, dict, str]:
    import joblib

    m = verify_dir(frozen_dir)
    spec = json.loads((frozen_dir / "spec.json").read_text(encoding="utf-8"))
    bundle = joblib.load(frozen_dir / "model.joblib")
    if list(bundle["features"]) != list(FROZEN_FEATURES) or spec["model"]["features"] != list(FROZEN_FEATURES):
        raise IngestionError("the frozen feature schema differs from FROZEN_FEATURES")
    if bundle["operating"] != spec["operating_points"]:
        raise IngestionError("the frozen model and specification disagree on the operating points")
    return spec, bundle, m["outputs_sha256"]["model.joblib"]


def predict_tables(tables: list[dict], spec: dict, bundle: dict, comparators: dict[str, Any], cfg: ml.MLConfig = ml.MLConfig()) -> tuple[list[dict], list[dict]]:
    """Frozen probabilities and decisions for every method at its frozen operating points. ``tables``: match_id, rows, X only."""
    cal = ml2.Platt(**bundle["calibrator"])
    pair_rows, preds = [], []
    for t in tables:
        X, rows = t["X"], t["rows"]
        probs: dict[str, np.ndarray] = {"frozen": cal(ml2.proba(bundle["model"], X[:, ml2.cols(tuple(bundle["features"]))]))}
        points: dict[str, dict] = {"frozen": spec["operating_points"]}
        for name, c in comparators.items():
            if name.startswith("step4_"):
                k = name[len("step4_"):]
                probs[name] = ml.predict(c["model"], k, X[:, :len(ml.ALL_FEATURES)])
                points[name] = {"frozen": dict(threshold=c["threshold"], margin=c["margin"])}
            else:
                probs[name] = ml2.Platt(**c["calibrator"])(ml2.proba(c["model"], X[:, ml2.cols(tuple(c["features"]))]))
                points[name] = c["operating"]
        for n, r in enumerate(rows):
            pair_rows.append(dict(match_id=t["match_id"], instance_id=r["instance_id"], incoming=r["incoming"], outgoing=r["outgoing"],
                                  reachable=r["reachable"], probability={k: round(float(v[n]), 6) for k, v in probs.items()}))
        for name, pts in points.items():
            for pt, op in pts.items():
                ds = decide(rows, probs[name], X, op["threshold"], op["margin"], cfg)
                preds += [dict(method=f"{name}@{pt}", match_id=t["match_id"], instance_id=iid, threshold=op["threshold"], margin=op["margin"], decisions=d)
                          for iid, d in ds.items()]
    return pair_rows, preds


def run_predict(raw_root, *, frozen_dir=None, val_root=None, match_ids: list[str] | None = None) -> tuple[Path, dict]:
    import joblib

    t0 = time.perf_counter()
    proc = approved_output_root(raw_root)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else proc / FROZEN_DIR
    root = Path(val_root) if val_root is not None else proc / VALIDATION_DIR
    out = root / "predictions"
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    spec, bundle, model_sha = load_frozen(frozen_dir)
    split, split_path = _splits(raw_root)
    if _sha256(split_path) != spec["splits"]["manifest_sha256"]:
        raise IngestionError("the split manifest changed after the freeze")
    ids = [str(m) for m in (match_ids or spec["splits"]["validation"])]
    check_split(split, ids, "validation")
    comps = {}
    for n, c in spec["comparators"]["step4"].items():
        if _sha256(Path(c["path"])) != c["sha256"]:
            raise IngestionError(f"frozen Step 4 model {n} changed")
        comps[f"step4_{n}"] = dict(c, model=joblib.load(c["path"]))
    c5 = spec["comparators"]["step5_with_conf"]
    if _sha256(Path(c5["path"])) != c5["sha256"]:
        raise IngestionError("frozen Step 5 model changed")
    b5 = joblib.load(c5["path"])
    comps["step5_with_conf"] = dict(model=b5["model"], features=b5["features"], calibrator=b5["calibrator"], operating=c5["operating"])
    tables, sources = [], {}
    for m in ids:
        inst, prov = load_visible(raw_root, m, split, root / "matching")
        rows, X = features_only(inst)
        for r in rows:
            r["match_id"] = m
        tables.append(dict(match_id=m, rows=rows, X=X))
        sources[m] = prov
    pair_rows, preds = predict_tables(tables, spec, bundle, comps)
    manifest = dict(schema=f"{VALIDATION_DIR}-predictions", matches=ids, frozen_model_sha256=model_sha, frozen_spec_sha256=_sha256(frozen_dir / "spec.json"),
                    sources=sources, files_read=["root.instances_b.jsonl", "root.manifest.json"], reference_files_read=False, model_fitting=False,
                    threshold_tuning=False, raw_snapshot=raw_snapshot(raw_root), code_sha256=code_digest(), processing=dict(seconds=round(time.perf_counter() - t0, 1)))
    _write_dir(out, {"pair_probabilities.jsonl": ml._jsonl(pair_rows), "predictions.jsonl": ml._jsonl(preds)}, {}, manifest)
    return out, verify_dir(out)


# ---------------------------------------------------------------- 4. score (after the predictions are written and checksummed)

def _labels_by_pair(instances: list[dict], references: list[dict]) -> dict[tuple, dict]:
    insts = {i["instance_id"]: i for i in instances}
    return {(r["instance_id"], r["incoming"], r["outgoing"]): r for ref in references for r in ml.pair_labels(ref, insts[ref["instance_id"]])}


def per_match_counts(preds: list[dict], refs: dict[str, dict], method: str, tier: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in preds:
        if p["method"] != method or refs[p["instance_id"]]["tier"] != tier:
            continue
        c = out.setdefault(p["match_id"], dict(incoming=0, matched=0, correct=0, wrong=0))
        for d in p["decisions"]:
            o = outcome(d, refs[p["instance_id"]]["expected"][d["incoming"]])
            c["incoming"] += 1
            c["matched"] += o in ("correct_match", "wrong_match")
            c["correct"] += o == "correct_match"
            c["wrong"] += o == "wrong_match"
    for c in out.values():
        c["coverage"] = c["matched"] / c["incoming"] if c["incoming"] else None
        c["false_association_rate"] = c["wrong"] / c["matched"] if c["matched"] else None
    return out


def pooled(pm: dict[str, dict]) -> dict:
    tot = {k: sum(v[k] for v in pm.values()) for k in ("incoming", "matched", "correct", "wrong")}
    return dict(tot, coverage=tot["matched"] / tot["incoming"] if tot["incoming"] else None,
                false_association_rate=tot["wrong"] / tot["matched"] if tot["matched"] else None,
                correct_match_rate=tot["correct"] / tot["incoming"] if tot["incoming"] else None)


def criteria_outcome(primary_pm: dict, base_pm: dict, secondary_pm: dict, base_secondary_pm: dict) -> dict[str, Any]:
    p, b = pooled(primary_pm), pooled(base_pm)
    bp = ml2.bootstrap(primary_pm, base_pm, BOOTSTRAP, 0)
    s, bs = pooled(secondary_pm), pooled(base_secondary_pm)
    bsec = ml2.bootstrap(secondary_pm, base_secondary_pm, BOOTSTRAP, 0)
    prim = dict(false_association_rate=p["false_association_rate"], coverage=p["coverage"], baseline_coverage=b["coverage"],
                baseline_false_association_rate=b["false_association_rate"], bootstrap=bp,
                fa_met=p["false_association_rate"] is not None and p["false_association_rate"] <= 0.10,
                coverage_improvement_met=bp["coverage_diff_ci95"][0] is not None and bp["coverage_diff_ci95"][0] > 0)
    prim["met"] = bool(prim["fa_met"] and prim["coverage_improvement_met"])
    sec = dict(false_association_rate=s["false_association_rate"], coverage=s["coverage"], baseline_coverage=bs["coverage"], bootstrap=bsec,
               met=s["false_association_rate"] is not None and s["false_association_rate"] <= 0.05)
    return dict(primary=prim, secondary=sec)


def run_score(raw_root, *, frozen_dir=None, val_root=None) -> tuple[Path, dict]:
    t0 = time.perf_counter()
    proc = approved_output_root(raw_root)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else proc / FROZEN_DIR
    root = Path(val_root) if val_root is not None else proc / VALIDATION_DIR
    out = root / "scoring"
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    spec, _, model_sha = load_frozen(frozen_dir)
    pm = verify_dir(root / "predictions")                          # predictions must be complete and intact before any reference is read
    if pm["frozen_model_sha256"] != model_sha:
        raise IngestionError("the predictions were not made by the frozen model")
    split, _ = _splits(raw_root)
    ids = pm["matches"]
    check_split(split, ids, "validation")
    preds = [json.loads(x) for x in (root / "predictions" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = [json.loads(x) for x in (root / "predictions" / "pair_probabilities.jsonl").read_text(encoding="utf-8").splitlines()]
    refs, labels, base_preds, audits, literal = {}, {}, [], {}, {}
    for m in ids:
        paths = output_paths(raw_root, m, root / "matching")
        man = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
        for sfx in (".instances_b.jsonl", ".references.jsonl", ".predictions_b.jsonl", ".summary.json"):
            if man["outputs_sha256"].get(paths[sfx].name) != _sha256(paths[sfx]):
                raise IngestionError(f"{paths[sfx]} does not match its manifest checksum")
        read = lambda sfx: [json.loads(x) for x in paths[sfx].read_text(encoding="utf-8").splitlines()]   # noqa: E731
        inst, rr = read(".instances_b.jsonl"), read(".references.jsonl")
        refs |= {r["instance_id"]: dict(r, match_id=m) for r in rr}
        labels |= _labels_by_pair(inst, rr)
        base_preds += [dict(p, match_id=m, method=p["baseline"]) for p in read(".predictions_b.jsonl")]
        summ = json.loads(paths[".summary.json"].read_text(encoding="utf-8"))
        audits[m] = {k: summ["audit"][k] for k in ("events", "accepted_by_tier", "rejected", "lower_tier_reasons")}
        literal[m] = summ["footballtrace_literal"]
    methods = sorted({p["method"] for p in preds})
    allp = [dict(p, baseline=p["method"]) for p in preds] + [dict(p, baseline=p["method"]) for p in base_preds]
    encounter = {}
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        rs = [r for r in refs.values() if r["tier"] == tier]
        idset = {r["instance_id"] for r in rs}
        encounter[tier] = evaluate([p for p in allp if p["instance_id"] in idset], rs, methods=tuple(methods) + tuple(BASELINES))
    by_match = {tier: {meth: per_match_counts(preds + base_preds, refs, meth, tier) for meth in list(methods) + list(BASELINES)}
                for tier in (LABEL_TIER, EXPLORATORY_TIER)}
    boot = {meth: ml2.bootstrap(by_match[LABEL_TIER][meth], None if meth == "step4_rf@frozen" else by_match[LABEL_TIER]["step4_rf@frozen"], BOOTSTRAP, 0)
            for meth in by_match[LABEL_TIER]}
    hi = by_match[LABEL_TIER]
    crit = criteria_outcome(hi["frozen@relaxed"], hi["step4_rf@frozen"], hi["frozen@conservative"], hi["step4_rf@frozen"])
    pairwise = {}
    names = sorted(pairs[0]["probability"]) if pairs else []
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        sel = [(r, labels[(r["instance_id"], r["incoming"], r["outgoing"])]) for r in pairs]
        sel = [(r, lb) for r, lb in sel if lb["tier"] == tier]
        y = np.array([lb["label"] for _, lb in sel], int)
        rows = [r for r, _ in sel]
        pairwise[tier] = {}
        for n in names:
            p = np.array([r["probability"][n] for r in rows])
            pairwise[tier][n] = ml.pairwise_metrics(y, p, rows) | dict(log_loss=ml._r(ml2.log_loss(y, p)))
            if tier == LABEL_TIER:
                pairwise[tier][n]["by_match"] = {}
                for m in ids:
                    k = np.array([r["match_id"] == m for r in rows])
                    pairwise[tier][n]["by_match"][m] = {kk: v for kk, v in ml.pairwise_metrics(y[k], p[k], [r for r, kk in zip(rows, k) if kk]).items() if kk != "reliability"}
    unverified = {}
    for p in preds:
        if refs[p["instance_id"]]["tier"] == UNVERIFIED_TIER:
            for d in p["decisions"]:
                unverified.setdefault(p["method"], {}).setdefault(d["decision"], 0)
                unverified[p["method"]][d["decision"]] += 1
    cases = case_examples(preds, pairs, refs, labels)
    summary = dict(schema=f"{VALIDATION_DIR}-scoring", matches=ids, frozen_model_sha256=model_sha, eligibility=audits, criteria=crit, encounter=encounter,
                   by_match=by_match, bootstrap=boot, pairwise=pairwise, unverified_decisions=unverified, footballtrace_literal=literal, cases=cases,
                   development_reference=spec["development_reference"],
                   note="all rates are agreement with the provisional provider identity reference, not verified identity accuracy")
    manifest = dict(schema=f"{VALIDATION_DIR}-scoring", matches=ids, predictions_manifest_sha256=_sha256(root / "predictions" / "manifest.json"),
                    frozen_model_sha256=model_sha, raw_snapshot=raw_snapshot(raw_root), code_sha256=code_digest(), processing=dict(seconds=round(time.perf_counter() - t0, 1)))
    _write_dir(out, {"summary.json": _dump(summary)}, {}, manifest)
    return out, summary


def case_examples(preds: list[dict], pairs: list[dict], refs: dict, labels: dict, k: int = 15) -> dict[str, list[dict]]:
    """Frozen model, relaxed point, high tier: most confident wrong matches; correct abstentions (the top candidate was not the
    reference); smooth-alternative-vs-strained-reference flags (possible provider-label errors, unverified)."""
    prob = {(r["instance_id"], r["incoming"], r["outgoing"]): r["probability"]["frozen"] for r in pairs}
    wrong, abstain = [], []
    for p in preds:
        if p["method"] != "frozen@relaxed" or refs[p["instance_id"]]["tier"] != LABEL_TIER:
            continue
        ref = refs[p["instance_id"]]
        for d in p["decisions"]:
            e = ref["expected"][d["incoming"]]
            base = dict(match_id=p["match_id"], instance_id=p["instance_id"], incoming=d["incoming"], candidate=d["candidate"], expected=e,
                        probability=d["probability"], margin=d["margin"], reference_pair_probability=None if e is None else ml._r(prob.get((p["instance_id"], d["incoming"], e))),
                        evidence=d["evidence"], n_incoming=len(ref["incoming"]), gap_s=ref["difficulty"]["ambiguity_s"], difficulty=ref["difficulty"]["reasons"],
                        crossing=ref["difficulty"]["crossing"])
            o = outcome(d, e)
            if o == "wrong_match":
                wrong.append(base)
            elif o == "uncertain" and d["candidate"] is not None and d["candidate"] != e:
                abstain.append(base)
    flagged = [w for w in wrong if w["evidence"] and w["evidence"]["cv_mid_err"] < 0.5 and (w["reference_pair_probability"] or 1) <= 0.2 and w["probability"] >= 0.8]
    return dict(wrong_most_confident=sorted(wrong, key=lambda x: -x["probability"])[:k],
                correct_abstentions_highest_candidate_probability=sorted(abstain, key=lambda x: -(x["probability"] or 0))[:k],
                correct_abstention_count=len(abstain), wrong_count=len(wrong),
                possible_reference_errors=dict(rule="wrong match with p >= 0.8, reference pair p <= 0.2 and chosen cv_mid_err < 0.5 m (unverified)",
                                               count=len(flagged), examples=flagged[:k]))
