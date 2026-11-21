"""M11 step 7: the one-time evaluation of the frozen Step 6 classifier on the 12 locked test matches.

    python -m football_fatigue.matching.locked_test audit                   # pre-test integrity gate; reads no test data
    python -m football_fatigue.matching.locked_test build --match <test id>  # unchanged Step 1 and Step 2-3 pipelines (resumable)
    python -m football_fatigue.matching.locked_test predict
    python -m football_fatigue.matching.locked_test score

Every stage writes inside ``m11-test-v1`` once and refuses to run again:

1. ``audit`` checks the split, the frozen model, specification, preregistration, code and raw data, and writes
   ``m11-test-v1/audit`` only if every check passes. It reads no test-match file.
2. ``build`` runs the unchanged Step 1 catalogue and Step 2-3 benchmark builder for one test match. An existing,
   checksum-verified stage output is kept (resume), never rebuilt.
3. ``predict`` reads only the matcher-visible task B instances, applies the frozen models with no fitting, checks that every
   incoming tracklet of every instance has a decision from every method, and writes ``predictions`` with checksums.
4. ``score`` verifies those checksums, then reads the evaluation-only references and writes ``scoring``.

Nothing here fits, calibrates or selects anything. Every rate is agreement with the provider identity reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root

from . import ml, ml2, validate
from .baselines import BASELINES
from .benchmark import SCHEMA_VERSION as SOURCE_SCHEMA, _sha256, code_digest, output_paths
from .evaluation import evaluate, gap_bin, metrics, outcome

TEST_DIR = "m11-test-v1"
EXPECTED_TEST = frozenset({"3813", "3845", "3848", "3850", "3853", "3854", "3857", "10504", "10505", "10509", "10515", "10516"})
FROZEN_MODEL_SHA256 = "629cca77c4692c3e37d3d6c6ce0c6eff7421c4ac2fbd70c063d7cdd3abeb9187"
FROZEN_SPEC_SHA256 = "6fa1df5c0067c1671aa35329dfcfc6c55794a3059dde3777b8ebfa423d032d49"
PREREG_DOC = "docs/research/m11_step6_preregistration.md"
PREREG_SHA256 = "a0c606b1480b792770a9ed3eb21afd8e3fd8d7de3cce22d0e0087b29d5b126c2"
STEP6_REPORT = "docs/research/m11_step6_frozen_validation.md"
FROZEN_POINTS = {"conservative": dict(threshold=0.87, margin=1.0, target=0.05), "relaxed": dict(threshold=0.76, margin=1.0, target=0.10)}
STEP6_SUMMARY = dict(development=dict(roc_auc=0.839, relaxed=(0.399, 0.086), conservative=(0.269, 0.043)),
                     validation=dict(roc_auc=0.848, relaxed=(0.428, 0.083), conservative=(0.294, 0.047)))
FROZEN_CODE = ("baselines.py", "benchmark.py", "evaluation.py", "footballtrace.py", "ml.py", "ml2.py", "validate.py")
REPRO_MATCH = "3832"                                     # validation match re-predicted by the audit as an implementation check
STRATA = dict(                                           # fixed in the audit, before any test file exists
    gap="the Step 2 evaluator gap bins <1.5, 1.5-2.5, 2.5-4, >=4 s (withheld interval)",
    duration="withheld encounter interval in finer bins <1, 1-2, 2-3, 3-4, 4-6, >=6 s (same quantity as the gap in task B)",
    players="incoming tracklets 2 / 3 / 4+",
    turn_severity="max heading change of the reference continuation: unmeasured, <=45 (not sharp), 45-90, 90-135, >135 deg",
    crossing="no crossing; crossing with minimum distance >=1 m, 0.5-1 m, <0.5 m",
    visibility_confidence=("matcher-visible samples of all incoming and outgoing tracklets: any ESTIMATED sample; else all HIGH confidence; "
                           "else minimum MEDIUM; else LOW or unknown"),
    tier="reference quality tier (high, lower, natural suspicious unverified)",
    team_relation="same team / opposite team / mixed")


# ---------------------------------------------------------------- split

def check_test(split: dict, ids: list[str]) -> None:
    """Every ID must be one of the 12 locked test matches of the frozen manifest."""
    bad = [m for m in ids if m not in EXPECTED_TEST or split["matches"].get(m, {}).get("split") != "test"]
    if bad:
        raise IngestionError(f"only the locked test matches are evaluated here; refused: {bad}")


def _root(raw_root, test_root=None) -> Path:
    return Path(test_root) if test_root is not None else approved_output_root(raw_root) / TEST_DIR


def _audit(root: Path) -> dict:
    if not (root / "audit" / "manifest.json").exists():
        raise IngestionError(f"{root / 'audit'} is missing: the pre-test integrity gate has not passed")
    validate.verify_dir(root / "audit")
    a = json.loads((root / "audit" / "audit.json").read_text(encoding="utf-8"))
    if not a["passed"]:
        raise IngestionError("the pre-test integrity audit did not pass")
    return a


# ---------------------------------------------------------------- 1. pre-test integrity gate (reads no test data)

def _close(a, b, tol=0.0006) -> bool:
    return a is not None and abs(a - b) <= tol


def integrity_checks(raw_root, *, frozen_dir=None, val_root=None, repo_root=None, test_root=None) -> tuple[list[dict], dict]:
    import joblib

    proc = approved_output_root(raw_root)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else proc / validate.FROZEN_DIR
    val = Path(val_root) if val_root is not None else proc / validate.VALIDATION_DIR
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    root = _root(raw_root, test_root)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    split, split_path = validate._splits(raw_root)
    test = split["splits"]["test"]
    check("split_manifest_unchanged_since_freeze", None)            # filled below once the spec is read
    check("test_ids_exact", set(test) == EXPECTED_TEST and len(test) == 12 and all(split["matches"][m]["split"] == "test" for m in test), sorted(test))
    dev_all, val_ids = set(split["splits"]["development"]), set(split["splits"]["validation"])
    check("test_disjoint_from_development_and_validation", not (set(test) & (dev_all | val_ids)) and len(dev_all) == 40 and len(val_ids) == 12,
          dict(development=len(dev_all), validation=len(val_ids), test=len(test)))
    frozen_ok = True
    try:
        fm = validate.verify_dir(frozen_dir)
    except (IngestionError, FileNotFoundError) as e:
        frozen_ok, fm = False, {}
        check("frozen_directory_checksums", False, str(e))
    if frozen_ok:
        check("frozen_directory_checksums", True, sorted(fm["outputs_sha256"]))
        model_sha, spec_sha = _sha256(frozen_dir / "model.joblib"), _sha256(frozen_dir / "spec.json")
        check("frozen_model_sha256", model_sha == FROZEN_MODEL_SHA256, model_sha)
        check("frozen_spec_sha256", spec_sha == FROZEN_SPEC_SHA256, spec_sha)
        spec, bundle, _ = validate.load_frozen(frozen_dir)
        checks[0] = dict(check="split_manifest_unchanged_since_freeze", passed=_sha256(split_path) == spec["splits"]["manifest_sha256"], detail=_sha256(split_path))
        check("spec_test_ids_match_manifest", set(spec["splits"]["test_locked"]) == EXPECTED_TEST and spec["splits"]["test_access"] == "never")
        rf = bundle["model"].steps[-1][1]                           # Pipeline([("clf", RandomForestClassifier)])
        check("frozen_feature_schema", list(bundle["features"]) == list(validate.FROZEN_FEATURES) and len(bundle["features"]) == 51
              and bundle["model"].n_features_in_ == 51 and len(bundle["model"].steps) == 1 and not set(validate.EXCLUDED_FEATURES) & set(bundle["features"]), len(bundle["features"]))
        check("frozen_hyperparameters", rf.n_estimators == 300 and rf.max_features == 0.33 and rf.min_samples_leaf == 5
              and rf.class_weight == "balanced_subsample" and rf.random_state == 0 and bundle["params"] == validate.FROZEN_PARAMS,
              dict(n_estimators=rf.n_estimators, max_features=rf.max_features, min_samples_leaf=rf.min_samples_leaf, class_weight=rf.class_weight))
        cal = spec["model"]["calibration"]
        check("frozen_calibration", bundle["calibrator"] == dict(a=cal["a"], b=cal["b"]), bundle["calibrator"])
        check("frozen_operating_points", bundle["operating"] == spec["operating_points"] == FROZEN_POINTS, spec["operating_points"])
        check("development_matches_disjoint_from_test", not set(spec["splits"]["development"]) & EXPECTED_TEST, spec["splits"]["development"])
        comp_ok = all(_sha256(Path(c["path"])) == c["sha256"] for c in spec["comparators"]["step4"].values()) and \
            _sha256(Path(spec["comparators"]["step5_with_conf"]["path"])) == spec["comparators"]["step5_with_conf"]["sha256"]
        check("frozen_comparator_checksums", comp_ok)
        dr = spec["development_reference"]
        check("step6_development_numbers", _close(dr["pairwise"]["roc_auc"], STEP6_SUMMARY["development"]["roc_auc"]) and all(
            _close(dr["operating"][p]["coverage"], STEP6_SUMMARY["development"][p][0]) and _close(dr["operating"][p]["false_association_rate"], STEP6_SUMMARY["development"][p][1])
            for p in FROZEN_POINTS), {p: (dr["operating"][p]["coverage"], dr["operating"][p]["false_association_rate"]) for p in FROZEN_POINTS} | dict(roc_auc=dr["pairwise"]["roc_auc"]))
        # implementation check: re-predict one validation match from its matcher-visible file and compare with Step 6
        inst, _ = validate.load_visible(raw_root, REPRO_MATCH, split, val / "matching")
        rows, X = validate.features_only(inst)
        for r in rows:
            r["match_id"] = REPRO_MATCH
        pr, pd = validate.predict_tables([dict(match_id=REPRO_MATCH, rows=rows, X=X)], spec, bundle, {})
        old_p = [json.loads(x) for x in (val / "predictions" / "pair_probabilities.jsonl").read_text(encoding="utf-8").splitlines()]
        old_d = [json.loads(x) for x in (val / "predictions" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        old_p = {(r["instance_id"], r["incoming"], r["outgoing"]): r["probability"]["frozen"] for r in old_p if r["match_id"] == REPRO_MATCH}
        old_d = {(p["method"], p["instance_id"]): [(d["incoming"], d["decision"], d["outgoing"]) for d in p["decisions"]]
                 for p in old_d if p["match_id"] == REPRO_MATCH and p["method"].startswith("frozen@")}
        new_p = {(r["instance_id"], r["incoming"], r["outgoing"]): r["probability"]["frozen"] for r in pr}
        new_d = {(p["method"], p["instance_id"]): [(d["incoming"], d["decision"], d["outgoing"]) for d in p["decisions"]] for p in pd}
        check("frozen_prediction_and_assignment_reproduce_step6", new_p == old_p and new_d == old_d and len(new_p) > 0,
              dict(match=REPRO_MATCH, pairs=len(new_p), instance_decisions=len(new_d)))
    vp = validate.verify_dir(val / "predictions")
    vs = validate.verify_dir(val / "scoring")
    check("step6_outputs_checksums", vp["frozen_model_sha256"] == FROZEN_MODEL_SHA256 and vs["frozen_model_sha256"] == FROZEN_MODEL_SHA256
          and set(vp["matches"]) == val_ids == set(vs["matches"]))
    s6 = json.loads((val / "scoring" / "summary.json").read_text(encoding="utf-8"))
    hi = s6["encounter"][validate.LABEL_TIER]
    v = STEP6_SUMMARY["validation"]
    got = {p: (hi[f"frozen@{p}"]["all"]["coverage"], hi[f"frozen@{p}"]["all"]["false_association_rate"]) for p in FROZEN_POINTS}
    auc = s6["pairwise"][validate.LABEL_TIER]["frozen"]["roc_auc"]
    check("step6_validation_numbers", _close(auc, v["roc_auc"]) and all(_close(got[p][0], v[p][0]) and _close(got[p][1], v[p][1]) for p in FROZEN_POINTS),
          dict(got, roc_auc=auc))
    here = code_digest()
    changed = [f for f in FROZEN_CODE if here.get(f) != vp["code_sha256"].get(f)]
    check("frozen_matching_code_unchanged_since_step6_predictions", not changed, changed)
    from football_fatigue.encounters.catalogue import code_digest as enc_digest
    bm = json.loads(output_paths(raw_root, REPRO_MATCH, val / "matching")[".manifest.json"].read_text(encoding="utf-8"))
    check("step1_code_unchanged_since_step6_build", enc_digest() == bm["code_sha256"]["encounters"],
          [f for f in enc_digest() if enc_digest()[f] != bm["code_sha256"]["encounters"].get(f)])
    pre = repo / PREREG_DOC
    check("step6_preregistration_sha256", pre.exists() and _sha256(pre) == PREREG_SHA256, _sha256(pre) if pre.exists() else None)
    rep = repo / STEP6_REPORT
    check("step6_report_present", rep.exists(), _sha256(rep) if rep.exists() else None)
    # no earlier test-match output or test-derived selection
    hits = sorted(str(p) for d in proc.glob("m11-*") if d.is_dir() for p in d.rglob("*") if p.name in EXPECTED_TEST)
    check("no_test_match_outputs_under_m11_dirs", not hits, hits[:20])
    check("test_directory_absent_before_audit", not root.exists(), str(root))
    used = {}
    for d in ("m11-ml-v1", "m11-ml-v2", validate.FROZEN_DIR):
        m = json.loads((proc / d / "manifest.json").read_text(encoding="utf-8"))
        used[d] = sorted(set(map(str, m.get("matches") or m.get("development") or [])))
    check("no_test_match_in_model_selection_or_fitting", not any(set(u) & EXPECTED_TEST for u in used.values()) and all(used.values()), used)
    snap = validate.raw_snapshot(raw_root)
    check("raw_data_unchanged_since_step6", snap == vs["raw_snapshot"] == vp["raw_snapshot"] and (not frozen_ok or snap == fm["raw_snapshot"]), snap)
    return checks, dict(split_path=str(split_path), test=sorted(test), raw_snapshot=snap)


def run_audit(raw_root, *, test_root=None, **kw) -> tuple[Path | None, dict]:
    root = _root(raw_root, test_root)
    checks, info = integrity_checks(raw_root, test_root=test_root, **kw)
    failed = [c for c in checks if not c["passed"]]
    audit = dict(schema=f"{TEST_DIR}-audit", audited_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), passed=not failed, checks=checks, **info,
                 frozen_model_sha256=FROZEN_MODEL_SHA256, frozen_spec_sha256=FROZEN_SPEC_SHA256, preregistration_sha256=PREREG_SHA256,
                 operating_points=FROZEN_POINTS, criteria=validate.CRITERIA, strata=STRATA, bootstrap=dict(resamples=validate.BOOTSTRAP, seed=0),
                 protocol=("task B as in Step 6: RAW coordinates, interval withheld, anonymous outgoing tracklets, unchanged eligibility, features, "
                           "assignment and abstention; the Step 6 primary and secondary criteria are applied unchanged to the test matches"),
                 test_files_read=False)
    if failed:
        return None, audit                                          # nothing is written; the test set stays unconsumed
    return validate._write_dir(root / "audit", {"audit.json": validate._dump(audit)}, {}, dict(schema=f"{TEST_DIR}-audit")), audit


# ---------------------------------------------------------------- 2. build (unchanged Step 1 and Step 2-3, resumable per stage)

def _verified(paths: dict[str, Path]) -> bool:
    """True if a stage output exists and every file matches its manifest; an incomplete or tampered output is an error."""
    man = paths[".manifest.json"]
    if not man.exists():
        if any(p.exists() for p in paths.values()):
            raise IngestionError(f"incomplete stage output next to {man}; not touched")
        return False
    m = json.loads(man.read_text(encoding="utf-8"))
    bad = [p.name for s, p in paths.items() if s != ".manifest.json" and m["outputs_sha256"].get(p.name) != _sha256(p)]
    if bad:
        raise IngestionError(f"{man.parent}: checksum mismatch for {bad}; not rebuilt")
    return True


def build_test_match(raw_root, match_id, *, test_root=None) -> dict:
    from football_fatigue.encounters import catalogue
    from football_fatigue.encounters.splits import default_quarantine

    from .benchmark import run_match

    root = _root(raw_root, test_root)
    _audit(root)
    split, _ = validate._splits(raw_root)
    mid = str(match_id)
    check_test(split, [mid])
    done = {}
    enc = catalogue.output_paths(raw_root, mid, root / "encounters")
    done["step1"] = "kept" if _verified(enc) else "built"
    if done["step1"] == "built":
        catalogue.run_match(raw_root, mid, out_root=root / "encounters", quarantine_periods=default_quarantine(mid), split=split["matches"][mid])
    bpaths = output_paths(raw_root, mid, root / "matching")
    done["benchmark"] = "kept" if _verified(bpaths) else "built"
    if done["benchmark"] == "built":
        run_match(raw_root, mid, source_root=root / "encounters", out_root=root / "matching", split=split["matches"][mid])
    summ = json.loads(bpaths[".summary.json"].read_text(encoding="utf-8"))
    return dict(match_id=mid, stages=done, events=summ["audit"]["events"], accepted_by_tier=summ["audit"]["accepted_by_tier"], rejected=summ["audit"]["rejected"])


# ---------------------------------------------------------------- 3. predict (matcher-visible files only, no fitting)

def load_visible_test(raw_root, match_id, split: dict, matching_root) -> tuple[list[dict], dict]:
    """Matcher-visible task B instances of one test match, checksum-verified. No reference file is opened."""
    mid = str(match_id)
    check_test(split, [mid])
    paths = output_paths(raw_root, mid, matching_root)
    m = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
    if m.get("schema") != SOURCE_SCHEMA or str(m.get("match_id")) != mid or (m.get("split") or {}).get("split") != "test":
        raise IngestionError(f"{paths['.manifest.json']} is not a {SOURCE_SCHEMA} test output of match {mid}")
    if m.get("synthetic_swaps_injected") is not False or m.get("provider_smoothed_coordinates_used") is not False:
        raise IngestionError("the source is not a RAW, swap-free matching benchmark")
    p = paths[".instances_b.jsonl"]
    if m["outputs_sha256"].get(p.name) != _sha256(p):
        raise IngestionError(f"{p} does not match its manifest checksum")
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()], dict(instances_b=_sha256(p), manifest=_sha256(paths[".manifest.json"]))


def completeness(tables: list[dict], instances: dict[str, list[dict]], preds: list[dict]) -> dict:
    """Every method decides every incoming tracklet of every instance exactly once; raises otherwise."""
    methods = sorted({p["method"] for p in preds})
    seen = Counter((p["method"], p["instance_id"]) for p in preds)
    by = {(p["method"], p["instance_id"]): p for p in preds}
    n_inst = n_inc = 0
    for m, insts in instances.items():
        for i in insts:
            n_inst += 1
            inc = sorted(t["tracklet_id"] for t in i["incoming"])
            n_inc += len(inc)
            for meth in methods:
                if seen[(meth, i["instance_id"])] != 1 or sorted(d["incoming"] for d in by[(meth, i["instance_id"])]["decisions"]) != inc:
                    raise IngestionError(f"{meth} did not decide every incoming tracklet of {m}/{i['instance_id']} exactly once")
    n_pairs = sum(len(t["rows"]) for t in tables)
    return dict(methods=methods, instances=n_inst, incoming=n_inc, pairs=n_pairs, predictions=len(preds), complete=True)


def run_predict(raw_root, *, frozen_dir=None, test_root=None) -> tuple[Path, dict]:
    import joblib

    t0 = time.perf_counter()
    proc = approved_output_root(raw_root)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else proc / validate.FROZEN_DIR
    root = _root(raw_root, test_root)
    a = _audit(root)
    out = root / "predictions"
    if out.exists():
        raise IngestionError(f"{out} already exists; test predictions are made once and never replaced")
    spec, bundle, model_sha = validate.load_frozen(frozen_dir)
    if model_sha != FROZEN_MODEL_SHA256 or _sha256(frozen_dir / "spec.json") != FROZEN_SPEC_SHA256 or bundle["operating"] != FROZEN_POINTS:
        raise IngestionError("the frozen model or specification differs from the audited one")
    split, _ = validate._splits(raw_root)
    ids = list(a["test"])
    check_test(split, ids)
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
    tables, sources, insts = [], {}, {}
    for m in ids:
        inst, prov = load_visible_test(raw_root, m, split, root / "matching")
        rows, X = validate.features_only(inst)
        for r in rows:
            r["match_id"] = m
        tables.append(dict(match_id=m, rows=rows, X=X))
        sources[m], insts[m] = prov, inst
    pair_rows, preds = validate.predict_tables(tables, spec, bundle, comps)
    comp = completeness(tables, insts, preds)
    manifest = dict(schema=f"{TEST_DIR}-predictions", matches=ids, frozen_model_sha256=model_sha, frozen_spec_sha256=_sha256(frozen_dir / "spec.json"),
                    audit_manifest_sha256=_sha256(root / "audit" / "manifest.json"), operating_points=bundle["operating"], completeness=comp,
                    sources=sources, files_read=["root.instances_b.jsonl", "root.manifest.json"], reference_files_read=False, model_fitting=False,
                    threshold_tuning=False, raw_snapshot=validate.raw_snapshot(raw_root), code_sha256=code_digest(),
                    processing=dict(seconds=round(time.perf_counter() - t0, 1)))
    validate._write_dir(out, {"pair_probabilities.jsonl": ml._jsonl(pair_rows), "predictions.jsonl": ml._jsonl(preds)}, {}, manifest)
    return out, validate.verify_dir(out)


# ---------------------------------------------------------------- 4. score (after the predictions are written and checksummed)

def _bin(v: float | None, edges: tuple[float, ...], labels: tuple[str, ...], none: str = "unmeasured") -> str:
    if v is None:
        return none
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def vis_conf(inst: dict) -> str:
    tr = inst["incoming"] + inst["outgoing"]
    vis = [v for t in tr for v in t["visibility"]]
    conf = [c for t in tr for c in t["confidence"]]
    if 2 in vis:
        return "estimated_sample_present"
    if conf and min(conf) == 3:
        return "all_high_confidence"
    return "min_medium_confidence" if conf and min(conf) == 2 else "low_or_unknown_confidence"


def strata(ref: dict, inst: dict) -> dict[str, str]:
    d = ref["difficulty"]
    n = len(ref["incoming"])
    turn = d.get("max_heading_change_deg")
    cross = "no_crossing" if not d["crossing"] else "crossing_" + _bin(d.get("min_distance_m"), (0.5, 1.0, float("inf")), ("<0.5m", "0.5-1m", ">=1m"))
    return dict(gap=gap_bin(d["ambiguity_s"]), duration=_bin(d["ambiguity_s"], (1, 2, 3, 4, 6, float("inf")), ("<1s", "1-2s", "2-3s", "3-4s", "4-6s", ">=6s")),
                players=str(n) if n < 4 else "4+", turn_severity="unmeasured" if turn is None else _bin(turn, (45.0001, 90, 135, float("inf")), ("<=45", "45-90", "90-135", ">135")),
                crossing=cross, visibility_confidence=vis_conf(inst), tier=ref["tier"], team_relation=ref["team_relation"])


def stratified(preds: list[dict], refs: dict, insts: dict, method: str) -> dict[str, dict]:
    """Encounter-level metrics of one method per fixed stratum, every tier (the tier is itself a stratum); sample sizes included."""
    rows = []
    for p in preds:
        if p["method"] != method:
            continue
        ref = refs[p["instance_id"]]
        outs = [outcome(d, ref["expected"][d["incoming"]]) for d in p["decisions"]]
        claimed = Counter(d["outgoing"] for d in p["decisions"] if d["decision"] == "MATCHED")
        rows.append((strata(ref, insts[p["instance_id"]]), ref["tier"], outs, any(v > 1 for v in claimed.values())))
    out = {}
    for key in STRATA:
        for tier in ((None,) if key == "tier" else (validate.LABEL_TIER,)):
            sel = [r for r in rows if tier is None or r[1] == tier]
            out[key] = {v: metrics([(o, c) for s, _, o, c in sel if s[key] == v]) for v in sorted({s[key] for s, _, _, _ in sel})}
    return out


def threshold_prf(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yh = p >= thr
    tp, fp, fn = int((yh & (y == 1)).sum()), int((yh & (y == 0)).sum()), int((~yh & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    return dict(threshold=thr, tp=tp, fp=fp, fn=fn, precision=ml._r(pr), recall=ml._r(rc), f1=ml._r(2 * pr * rc / (pr + rc)) if pr and rc else None)


def auc_bootstrap(y: np.ndarray, p: np.ndarray, match: np.ndarray, n: int = validate.BOOTSTRAP, seed: int = 0) -> list:
    """95% match-level bootstrap interval of the pooled pairwise ROC-AUC (matches resampled, pairs never treated as independent units)."""
    from sklearn.metrics import roc_auc_score

    ids = sorted(set(match.tolist()))
    idx = {m: np.flatnonzero(match == m) for m in ids}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        k = np.concatenate([idx[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
        if len(set(y[k].tolist())) == 2:
            v.append(roc_auc_score(y[k], p[k]))
    return [ml._r(np.percentile(v, 2.5)), ml._r(np.percentile(v, 97.5))]


def composition(refs: list[dict]) -> dict:
    """Shares of high-tier incoming tracklets per stratum (distribution-shift check; references of one split)."""
    hi = [r for r in refs if r["tier"] == validate.LABEL_TIER]
    n = sum(len(r["incoming"]) for r in hi)
    def share(f):
        c = Counter()
        for r in hi:
            c[f(r)] += len(r["incoming"])
        return {k: round(v / n, 4) for k, v in sorted(c.items())}
    d = lambda r: r["difficulty"]   # noqa: E731
    return dict(instances=len(hi), incoming=n, gap=share(lambda r: gap_bin(d(r)["ambiguity_s"])), players=share(lambda r: "2" if len(r["incoming"]) == 2 else "3+"),
                sharp_turn=share(lambda r: "yes" if "sharp_turn" in d(r)["reasons"] else "no"), crossing=share(lambda r: "yes" if d(r)["crossing"] else "no"),
                team_relation=share(lambda r: r["team_relation"]))


def _case(p: dict, d: dict, ref: dict, inst: dict, prob: dict, why: str) -> dict:
    e = ref["expected"][d["incoming"]]
    return dict(reason=why, match_id=p["match_id"], instance_id=p["instance_id"], period=ref["period"], start_time_s=ref["start_time_s"],
                end_time_s=ref["end_time_s"], incoming=d["incoming"], predicted_outgoing=d["outgoing"], decision=d["decision"], top_candidate=d["candidate"],
                reference_outgoing=e, probability=d["probability"], margin=d["margin"],
                reference_pair_probability=None if e is None else ml._r(prob.get((p["instance_id"], d["incoming"], e))),
                flags=dict(tier=ref["tier"], below_high_reasons=ref.get("below_high_reasons"), difficulty=ref["difficulty"]["reasons"],
                           gap_s=ref["difficulty"]["ambiguity_s"], n_incoming=len(ref["incoming"]), crossing=ref["difficulty"]["crossing"],
                           max_heading_change_deg=ref["difficulty"].get("max_heading_change_deg"), visibility_confidence=vis_conf(inst),
                           discontinuity_evidence=bool(ref.get("discontinuity_evidence"))),
                evidence=d.get("evidence"))


def failure_cases(preds: list[dict], pairs: list[dict], refs: dict, insts: dict, k: int = 12) -> dict:
    """Frozen model, relaxed point. Anonymous tracklet IDs only; no player identity is copied into the cases."""
    prob = {(r["instance_id"], r["incoming"], r["outgoing"]): r["probability"]["frozen"] for r in pairs}
    cat: dict[str, list[dict]] = {c: [] for c in ("correct_confident", "wrong_confident", "appropriate_abstention", "long_gap_failure",
                                                    "sharp_turn_failure", "multi_player_ambiguity", "possible_reference_inconsistency", "natural_suspicious")}
    for p in preds:
        if p["method"] != "frozen@relaxed":
            continue
        ref, inst = refs[p["instance_id"]], insts[p["instance_id"]]
        for d in p["decisions"]:
            e = ref["expected"][d["incoming"]]
            o = outcome(d, e)
            if ref["tier"] == validate.UNVERIFIED_TIER:
                cat["natural_suspicious"].append(_case(p, d, ref, inst, prob, f"natural suspicious (unverified) instance, decision {d['decision']}"))
                continue
            if ref["tier"] != validate.LABEL_TIER:
                continue
            c = lambda why: _case(p, d, ref, inst, prob, why)   # noqa: E731
            if o == "correct_match":
                cat["correct_confident"].append(c("MATCHED and agrees with the provider reference"))
            if o == "wrong_match":
                cat["wrong_confident"].append(c("MATCHED but disagrees with the provider reference"))
                if ref["difficulty"]["ambiguity_s"] >= 2.5:
                    cat["long_gap_failure"].append(c("wrong match with a withheld gap >= 2.5 s"))
                if "sharp_turn" in ref["difficulty"]["reasons"]:
                    cat["sharp_turn_failure"].append(c("wrong match where the reference continuation turns sharply"))
                rp = prob.get((p["instance_id"], d["incoming"], e)) if e else None
                if d["evidence"] and d["evidence"]["cv_mid_err"] < 0.5 and (rp if rp is not None else 1) <= 0.2 and d["probability"] >= 0.8:
                    cat["possible_reference_inconsistency"].append(c("Step 6 flag: p >= 0.8, reference pair p <= 0.2, chosen cv_mid_err < 0.5 m (unverified)"))
            if o == "uncertain" and d["candidate"] is not None and d["candidate"] != e:
                cat["appropriate_abstention"].append(c("abstained; the top candidate disagrees with the provider reference"))
            if len(ref["incoming"]) >= 3 and o in ("wrong_match", "uncertain") and d["candidate"] != e:
                cat["multi_player_ambiguity"].append(c("3+ player encounter; the top candidate disagrees with the reference"))
    key = {"appropriate_abstention": lambda x: -(x["probability"] or 0), "natural_suspicious": lambda x: (x["match_id"], x["instance_id"])}
    return {name: dict(count=len(v), examples=sorted(v, key=key.get(name, lambda x: -(x["probability"] or 0)))[:k]) for name, v in cat.items()}


def run_score(raw_root, *, frozen_dir=None, test_root=None) -> tuple[Path, dict]:
    t0 = time.perf_counter()
    proc = approved_output_root(raw_root)
    frozen_dir = Path(frozen_dir) if frozen_dir is not None else proc / validate.FROZEN_DIR
    root = _root(raw_root, test_root)
    _audit(root)
    out = root / "scoring"
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    spec, _, model_sha = validate.load_frozen(frozen_dir)
    pm = validate.verify_dir(root / "predictions")                 # complete and intact before any reference is read
    if pm["frozen_model_sha256"] != model_sha or model_sha != FROZEN_MODEL_SHA256 or not pm["completeness"]["complete"]:
        raise IngestionError("the predictions were not made by the frozen model or are incomplete")
    split, _ = validate._splits(raw_root)
    ids = pm["matches"]
    check_test(split, ids)
    preds = [json.loads(x) for x in (root / "predictions" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = [json.loads(x) for x in (root / "predictions" / "pair_probabilities.jsonl").read_text(encoding="utf-8").splitlines()]
    refs, labels, base_preds, audits, literal, insts = {}, {}, [], {}, {}, {}
    for m in ids:
        paths = output_paths(raw_root, m, root / "matching")
        man = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
        for sfx in (".instances_b.jsonl", ".references.jsonl", ".predictions_b.jsonl", ".summary.json"):
            if man["outputs_sha256"].get(paths[sfx].name) != _sha256(paths[sfx]):
                raise IngestionError(f"{paths[sfx]} does not match its manifest checksum")
        read = lambda sfx: [json.loads(x) for x in paths[sfx].read_text(encoding="utf-8").splitlines()]   # noqa: E731
        inst, rr = read(".instances_b.jsonl"), read(".references.jsonl")
        insts |= {i["instance_id"]: i for i in inst}
        refs |= {r["instance_id"]: dict(r, match_id=m) for r in rr}
        labels |= validate._labels_by_pair(inst, rr)
        base_preds += [dict(p, match_id=m, method=p["baseline"]) for p in read(".predictions_b.jsonl")]
        summ = json.loads(paths[".summary.json"].read_text(encoding="utf-8"))
        audits[m] = {k: summ["audit"][k] for k in ("events", "accepted_by_tier", "rejected", "lower_tier_reasons")}
        literal[m] = summ["footballtrace_literal"]
    methods = sorted({p["method"] for p in preds})
    allp = [dict(p, baseline=p["method"]) for p in preds + base_preds]
    encounter = {}
    for tier in (validate.LABEL_TIER, validate.EXPLORATORY_TIER):
        rs = [r for r in refs.values() if r["tier"] == tier]
        idset = {r["instance_id"] for r in rs}
        encounter[tier] = evaluate([p for p in allp if p["instance_id"] in idset], rs, methods=tuple(methods) + tuple(BASELINES))
    by_match = {tier: {meth: validate.per_match_counts(preds + base_preds, refs, meth, tier) for meth in list(methods) + list(BASELINES)}
                for tier in (validate.LABEL_TIER, validate.EXPLORATORY_TIER)}
    hi = by_match[validate.LABEL_TIER]
    boot = {meth: ml2.bootstrap(hi[meth], None if meth == "step4_rf@frozen" else hi["step4_rf@frozen"], validate.BOOTSTRAP, 0) for meth in hi}
    boot_vs_frozen = {pt: {meth: ml2.bootstrap(hi[f"frozen@{pt}"], hi[meth], validate.BOOTSTRAP, 0)
                           for meth in hi if meth != f"frozen@{pt}"} for pt in FROZEN_POINTS}
    crit = validate.criteria_outcome(hi["frozen@relaxed"], hi["step4_rf@frozen"], hi["frozen@conservative"], hi["step4_rf@frozen"])
    strat = {meth: stratified(preds + base_preds, refs, insts, meth)
             for meth in ("frozen@relaxed", "frozen@conservative", "step4_rf@frozen", "hungarian_cv", "chain_during", "e0_adapted")}
    pairwise = {}
    names = sorted(pairs[0]["probability"]) if pairs else []
    for tier in (validate.LABEL_TIER, validate.EXPLORATORY_TIER):
        sel = [(r, labels[(r["instance_id"], r["incoming"], r["outgoing"])]) for r in pairs]
        sel = [(r, lb) for r, lb in sel if lb["tier"] == tier]
        y = np.array([lb["label"] for _, lb in sel], int)
        rows = [r for r, _ in sel]
        mid = np.array([r["match_id"] for r in rows])
        pairwise[tier] = {}
        for n in names:
            p = np.array([r["probability"][n] for r in rows])
            pairwise[tier][n] = ml.pairwise_metrics(y, p, rows) | dict(log_loss=ml._r(ml2.log_loss(y, p)))
            if n == "frozen":
                pairwise[tier][n]["at_operating_thresholds"] = {pt: threshold_prf(y, p, op["threshold"]) for pt, op in FROZEN_POINTS.items()}
                pairwise[tier][n]["roc_auc_match_bootstrap_ci95"] = auc_bootstrap(y, p, mid)
            if tier == validate.LABEL_TIER:
                pairwise[tier][n]["by_match"] = {}
                for m in ids:
                    k = mid == m
                    pairwise[tier][n]["by_match"][m] = {kk: v for kk, v in ml.pairwise_metrics(y[k], p[k], [r for r, kk in zip(rows, k) if kk]).items() if kk != "reliability"}
    unverified = {}
    for p in preds:
        if refs[p["instance_id"]]["tier"] == validate.UNVERIFIED_TIER:
            for d in p["decisions"]:
                unverified.setdefault(p["method"], Counter())[d["decision"]] += 1
    dev_refs = []
    for m in spec["splits"]["development"]:
        rp = output_paths(raw_root, m, proc / SOURCE_SCHEMA)[".references.jsonl"]
        dev_refs += [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines()]
    val_refs = []
    for m in spec["splits"]["validation"]:
        rp = output_paths(raw_root, m, proc / validate.VALIDATION_DIR / "matching")[".references.jsonl"]
        val_refs += [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines()]
    shift = dict(development=composition(dev_refs), validation=composition(val_refs), test=composition(list(refs.values())))
    s6 = json.loads((proc / validate.VALIDATION_DIR / "scoring" / "summary.json").read_text(encoding="utf-8"))
    summary = dict(schema=f"{TEST_DIR}-scoring", matches=ids, frozen_model_sha256=model_sha, operating_points=FROZEN_POINTS, eligibility=audits,
                   criteria=crit, encounter=encounter, by_match=by_match, bootstrap=boot, bootstrap_vs_frozen=boot_vs_frozen, strata=strat,
                   pairwise=pairwise, unverified_decisions={k: dict(v) for k, v in unverified.items()}, footballtrace_literal=literal,
                   cases=failure_cases(preds, pairs, refs, insts), composition_shift=shift,
                   development_reference=spec["development_reference"],
                   validation_reference=dict(encounter={k: s6["encounter"][validate.LABEL_TIER][k]["all"] for k in ("frozen@relaxed", "frozen@conservative", "step4_rf@frozen")},
                                             pairwise={k: v for k, v in s6["pairwise"][validate.LABEL_TIER]["frozen"].items() if k not in ("reliability", "by_match")},
                                             bootstrap={k: s6["bootstrap"][k] for k in ("frozen@relaxed", "frozen@conservative")},
                                             by_match={k: s6["by_match"][validate.LABEL_TIER][k] for k in ("frozen@relaxed", "frozen@conservative")}),
                   note="all rates are agreement with the provisional provider identity reference, not verified identity accuracy")
    manifest = dict(schema=f"{TEST_DIR}-scoring", matches=ids, predictions_manifest_sha256=_sha256(root / "predictions" / "manifest.json"),
                    frozen_model_sha256=model_sha, raw_snapshot=validate.raw_snapshot(raw_root), code_sha256=code_digest(),
                    processing=dict(seconds=round(time.perf_counter() - t0, 1)))
    validate._write_dir(out, {"summary.json": validate._dump(summary)}, {}, manifest)
    return out, summary


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.matching.locked_test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "build", "predict", "score"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
        if name == "build":
            p.add_argument("--match", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "audit":
            out, audit = run_audit(a.root)
            print(json.dumps(dict(written=str(out) if out else None, passed=audit["passed"],
                                  failed=[c for c in audit["checks"] if not c["passed"]]), indent=1, default=str))
            return 0 if audit["passed"] else 3
        if a.cmd == "build":
            print(json.dumps(build_test_match(a.root, a.match), indent=1))
        elif a.cmd == "predict":
            out, m = run_predict(a.root)
            print(json.dumps(dict(written=str(out), completeness=m["completeness"], reference_files_read=m["reference_files_read"]), indent=1))
        else:
            out, _ = run_score(a.root)
            print(json.dumps(dict(written=str(out)), indent=1))
        return 0
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
