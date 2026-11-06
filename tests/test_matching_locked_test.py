"""Tests for football_fatigue.matching.locked_test (M11 step 7). Synthetic inputs only: they check the one-time test path is
isolated, ordered and counted correctly, not that the classifier is accurate."""

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from football_fatigue.ingestion import IngestionError  # noqa: E402
from football_fatigue.matching import locked_test as lt  # noqa: E402
from football_fatigue.matching.benchmark import SCHEMA_VERSION, _sha256, output_paths  # noqa: E402

SPLIT = dict(matches={"1": dict(split="development"), "2": dict(split="validation")} | {m: dict(split="test") for m in lt.EXPECTED_TEST})


def tracklet(tid, vis=1, conf=3):
    return dict(tracklet_id=tid, t=[0.0, 0.1], x=[0.0, 0.1], y=[0.0, 0.0], visibility=[vis, 1], confidence=[conf, 3])


def inst(iid, n=2, vis=1, conf=3):
    return dict(instance_id=iid, incoming=[tracklet(f"in-{k}", vis, conf) for k in range(n)], outgoing=[tracklet(f"out-{k}") for k in range(n)])


def ref(iid, n=2, gap=1.0, turn=None, crossing=False, tier=lt.validate.LABEL_TIER):
    return dict(instance_id=iid, tier=tier, team_relation="opposite_team", period=1, start_time_s=10.0, end_time_s=10.0 + gap,
                expected={f"in-{k}": f"out-{k}" for k in range(n)}, incoming={f"in-{k}": {} for k in range(n)},
                difficulty=dict(ambiguity_s=gap, crossing=crossing, min_distance_m=0.3, max_heading_change_deg=turn,
                                reasons=["sharp_turn"] if turn and turn > 45 else []))


def pred(method, iid, decisions):
    return dict(method=method, match_id="3813", instance_id=iid,
                decisions=[dict(incoming=i, outgoing=o if d == "MATCHED" else None, candidate=o, decision=d, probability=0.9, margin=2.0, evidence=None)
                           for i, o, d in decisions])


def test_check_test_accepts_only_locked_test_matches():
    lt.check_test(SPLIT, sorted(lt.EXPECTED_TEST))
    for bad in (["1"], ["2"], ["9999"], ["3813", "2"]):
        with pytest.raises(IngestionError, match="only the locked test"):
            lt.check_test(SPLIT, bad)


def test_stages_refuse_to_run_without_a_passed_audit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    root = tmp_path / "processed" / lt.TEST_DIR
    with pytest.raises(IngestionError, match="integrity gate"):
        lt.build_test_match(raw, "3813", test_root=root)
    with pytest.raises(IngestionError, match="integrity gate"):
        lt.run_predict(raw, test_root=root)
    with pytest.raises(IngestionError, match="integrity gate"):
        lt.run_score(raw, test_root=root)
    lt.validate._write_dir(root / "audit", {"audit.json": json.dumps(dict(passed=False)).encode()}, {}, {})
    with pytest.raises(IngestionError, match="did not pass"):
        lt.run_predict(raw, test_root=root)


def test_failed_audit_writes_nothing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sp = tmp_path / "processed" / "m11-encounters-v1"
    sp.mkdir(parents=True)
    (sp / "splits.json").write_text(json.dumps(dict(SPLIT, splits=dict(development=["1"], validation=["2"], test=sorted(lt.EXPECTED_TEST)))))
    root = tmp_path / "processed" / lt.TEST_DIR
    with pytest.raises((IngestionError, FileNotFoundError)):                # no frozen model or Step 6 outputs here
        lt.run_audit(raw, test_root=root, repo_root=tmp_path)
    assert not root.exists()


def test_load_visible_test_reads_no_reference_and_checks_split(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    mroot = tmp_path / "processed" / lt.TEST_DIR / "matching"
    paths = output_paths(raw, "3813", mroot)
    paths[".instances_b.jsonl"].parent.mkdir(parents=True)
    paths[".instances_b.jsonl"].write_text(json.dumps(inst("a")) + "\n", encoding="utf-8")
    man = dict(schema=SCHEMA_VERSION, match_id="3813", split=dict(split="test"), synthetic_swaps_injected=False, provider_smoothed_coordinates_used=False,
               outputs_sha256={paths[".instances_b.jsonl"].name: _sha256(paths[".instances_b.jsonl"])})
    paths[".manifest.json"].write_text(json.dumps(man), encoding="utf-8")
    assert not paths[".references.jsonl"].exists()
    got, _ = lt.load_visible_test(raw, "3813", SPLIT, mroot)
    assert [i["instance_id"] for i in got] == ["a"]
    paths[".manifest.json"].write_text(json.dumps(dict(man, split=dict(split="validation"))), encoding="utf-8")
    with pytest.raises(IngestionError, match="not a"):
        lt.load_visible_test(raw, "3813", SPLIT, mroot)


def test_completeness_requires_every_incoming_decided_once():
    insts = {"3813": [inst("a"), inst("b", n=3)]}
    tables = [dict(rows=[0] * 13)]
    ok = [pred("m", "a", [("in-0", "out-0", "MATCHED"), ("in-1", "out-1", "UNCERTAIN")]),
          pred("m", "b", [("in-0", "out-0", "MATCHED"), ("in-1", None, "UNMATCHED"), ("in-2", "out-2", "UNCERTAIN")])]
    c = lt.completeness(tables, insts, ok)
    assert (c["instances"], c["incoming"], c["pairs"], c["predictions"]) == (2, 5, 13, 2)
    with pytest.raises(IngestionError, match="exactly once"):
        lt.completeness(tables, insts, ok[:1])
    with pytest.raises(IngestionError, match="exactly once"):
        lt.completeness(tables, insts, [ok[0], pred("m", "b", [("in-0", "out-0", "MATCHED")])])


def test_stratified_denominators_and_bins():
    refs = {"a": ref("a", gap=1.0), "b": ref("b", n=3, gap=3.0, turn=100.0, crossing=True), "c": ref("c", tier=lt.validate.EXPLORATORY_TIER)}
    insts = {"a": inst("a"), "b": inst("b", n=3, vis=2), "c": inst("c")}
    preds = [pred("m", "a", [("in-0", "out-0", "MATCHED"), ("in-1", "out-0", "UNCERTAIN")]),
             pred("m", "b", [("in-0", "out-1", "MATCHED"), ("in-1", "out-0", "MATCHED"), ("in-2", "out-2", "UNCERTAIN")]),
             pred("m", "c", [("in-0", "out-0", "MATCHED"), ("in-1", "out-1", "MATCHED")])]
    s = lt.stratified(preds, refs, insts, "m")
    assert sum(v["incoming"] for v in s["gap"].values()) == 5                     # high tier only
    assert sum(v["incoming"] for v in s["tier"].values()) == 7                    # the tier stratum covers every tier
    b = s["players"]["3"]
    assert (b["incoming"], b["coverage"], b["false_association_rate"]) == (3, round(2 / 3, 4), 1.0)
    assert set(s["turn_severity"]) == {"unmeasured", "90-135"} and set(s["crossing"]) == {"no_crossing", "crossing_<0.5m"}
    assert set(s["visibility_confidence"]) == {"all_high_confidence", "estimated_sample_present"}
    assert (s["duration"]["1-2s"]["incoming"], s["duration"]["3-4s"]["incoming"]) == (2, 3)
    assert lt.strata(ref("x", gap=0.5, turn=45.0), inst("x", conf=2))["turn_severity"] == "<=45"
    assert lt.vis_conf(inst("x", conf=2)) == "min_medium_confidence"


def test_threshold_prf_and_match_bootstrap():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.95, 0.3])
    r = lt.threshold_prf(y, p, 0.75)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1) and r["precision"] == round(2 / 3, 4) and r["recall"] == round(2 / 3, 4)
    m = np.array(["a", "a", "b", "b", "c", "c"])
    lo, hi = lt.auc_bootstrap(y, p, m, n=200)
    assert lo <= hi and 0 <= lo and hi <= 1
    assert lt.auc_bootstrap(y, p, m, n=50) == lt.auc_bootstrap(y, p, m, n=50)     # deterministic
