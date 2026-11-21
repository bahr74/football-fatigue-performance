"""Tests for football_fatigue.m11_dashboard (M11 step 8).

Synthetic tests build a temporary processed tree. Tests marked by ``REAL`` read the local development artifacts (git-ignored
``data/``) and skip when they are absent; they check exact reproduction of the frozen Step 6 inference, the hidden-reference
behaviour and that nothing under ``data/`` is modified. One test reads the matcher-visible instances and the saved frozen
predictions of a validation match (no reference, no scoring) only to prove the per-encounter inference is the batch inference.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from football_fatigue.matching import ml, validate  # noqa: E402
from football_fatigue.matching.benchmark import SCHEMA_VERSION, TASK_B, _sha256, output_paths  # noqa: E402
from football_fatigue.m11_dashboard import data as D  # noqa: E402
from football_fatigue.m11_dashboard import inference as I  # noqa: E402
from football_fatigue.m11_dashboard import pitch as P  # noqa: E402
from football_fatigue.m11_dashboard import presentation as PR  # noqa: E402
from football_fatigue.m11_dashboard import research as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
REAL = pytest.mark.skipif(not (PROC / validate.FROZEN_DIR / "manifest.json").exists() or not (PROC / "m11-matching-v3" / "3829").exists(),
                          reason="local M11 development artifacts not present")
DEV_MATCH = "3829"
IDENTITY_WORDS = ("jersey", "player_id", "slot", "team", "side", "expected", "crossing", "heading")


# ---------------------------------------------------------------- synthetic tree

def track(tid, f, t0, t1, vis=1):
    k = np.arange(round(t0 * 30), round(t1 * 30) + 1)
    P_ = np.array([f(q / 30) for q in k])
    return dict(tracklet_id=tid, t=[round(q / 30, 4) for q in k], x=P_[:, 0].tolist(), y=P_[:, 1].tolist(), visibility=[vis] * len(k),
                confidence=[3] * len(k))


def synthetic_instance(iid="i1", gap=1.0, vis=1):
    fs = [lambda t: (t * 3.0, 0.0), lambda t: (0.0, 5.0 + t * 2.0)]
    return dict(instance_id=iid, schema=SCHEMA_VERSION, task=TASK_B, units={}, ambiguity_interval_s=[0.0, gap],
                incoming=[track(f"in-{k}", f, -2.0, -1 / 30, vis) for k, f in enumerate(fs)],
                outgoing=[track(f"out-{k}", f, gap, gap + 2.0) for k, f in enumerate(fs)], during=dict(t=[], points=[]))


def synthetic_reference(iid="i1", tier=D.LABEL_TIER):
    slot = lambda s: dict(slot=s, side="home", jersey=s, player_id=100 + s, team_id=1, core=True)  # noqa: E731
    return dict(instance_id=iid, schema=SCHEMA_VERSION, match_id=1, period=1, tier=tier, below_high_reasons=[], start_frame=10, end_frame=40,
                start_time_s=100.0, end_time_s=101.0, rows=dict(before=[0, 60], during=[60, 90], after=[90, 150]), encounter_ids=["1-t2-0"],
                step1_categories={"ordinary": 1}, discontinuity_evidence=[], incoming={"in-0": slot(3), "in-1": slot(7)},
                outgoing={"out-0": slot(3), "out-1": slot(7)}, expected={"in-0": "out-0", "in-1": "out-1"}, without_incoming=[], without_outgoing=[],
                n_participants=2, team_relation="same_team",
                difficulty=dict(label="challenging", reasons=["sharp_turn"], ambiguity_s=1.0, min_distance_m=0.4, crossing=True,
                                max_heading_change_deg=90.0, mean_speed_m_s=2.0))


def make_tree(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"
    sp = proc / ml.SPLITS_DIR / "splits.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(dict(matches={"1": dict(split="development"), "2": dict(split="validation"), "3": dict(split="test"),
                                           "4": dict(split="development")})), encoding="utf-8")
    fz = proc / validate.FROZEN_DIR
    fz.mkdir()
    (fz / "spec.json").write_text(json.dumps(dict(splits=dict(development=["1", "4"], manifest_sha256=_sha256(sp)))), encoding="utf-8")
    paths = output_paths(raw, "1")
    paths[".manifest.json"].parent.mkdir(parents=True)
    paths[".instances_b.jsonl"].write_text(json.dumps(synthetic_instance()) + "\n", encoding="utf-8")
    paths[".references.jsonl"].write_text(json.dumps(synthetic_reference()) + "\n", encoding="utf-8")
    man = dict(schema=SCHEMA_VERSION, match_id=1, split=dict(split="development"), synthetic_swaps_injected=False,
               provider_smoothed_coordinates_used=False, outputs_sha256={p.name: _sha256(p) for p in (paths[".instances_b.jsonl"], paths[".references.jsonl"])})
    paths[".manifest.json"].write_text(json.dumps(man), encoding="utf-8")
    return raw, proc, paths


# ---------------------------------------------------------------- selection and missing artifacts

def test_development_only_match_selection(tmp_path):
    raw, _, _ = make_tree(tmp_path)
    assert D.development_matches(raw) == ["1"]                    # "4" is development but has no benchmark output
    md = D.load_match(raw, "1")
    assert list(md.instances) == ["i1"]
    with pytest.raises(D.DashboardError, match="locked test"):
        D.load_match(raw, "3")
    with pytest.raises(D.DashboardError, match="only development"):
        D.load_match(raw, "2")
    with pytest.raises(D.DashboardError, match="only development"):
        D.load_cache(raw, "2")


def test_missing_artifacts_raise_clear_errors(tmp_path):
    raw, proc, paths = make_tree(tmp_path)
    with pytest.raises(D.DashboardError, match="frozen model artifacts are missing"):
        I.load_frozen_model(raw)
    with pytest.raises(D.DashboardError, match="no Step 2-3 benchmark output"):
        D.load_match(raw, "4")
    with pytest.raises(D.DashboardError, match="RAW cache"):
        D.load_cache(raw, "1")
    assert R._summary(proc / "m11-test-v1" / "scoring") is None
    paths[".references.jsonl"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(D.DashboardError, match="checksum"):
        D.load_match(raw, "1")
    (proc / validate.FROZEN_DIR / "spec.json").unlink()
    with pytest.raises(D.DashboardError, match="frozen specification"):
        D.development_matches(raw)


def test_split_manifest_change_is_refused(tmp_path):
    raw, proc, _ = make_tree(tmp_path)
    sp = proc / ml.SPLITS_DIR / "splits.json"
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(D.DashboardError, match="changed after the freeze"):
        D.development_matches(raw)


# ---------------------------------------------------------------- hidden reference

def test_public_context_holds_no_identity_or_mapping(tmp_path):
    raw, _, _ = make_tree(tmp_path)
    md = D.load_match(raw, "1")
    ctx = md.contexts["i1"]
    assert not set(D.HIDDEN_KEYS) & set(ctx)
    assert not [k for k in ctx if any(w in k for w in IDENTITY_WORDS)]
    assert ctx["filter_sharp_turn"] is True and ctx["n_incoming"] == 2 and ctx["gap_bin"] == "0-1.5s"
    assert "expected" not in json.dumps(ctx) and "jersey" not in json.dumps(ctx)
    assert md.provider_reference("i1")["expected"] == {"in-0": "out-0", "in-1": "out-1"}
    assert md.participant_slots("i1") == {3, 7}
    assert "_references" not in repr(md)


def test_inference_accepts_only_task_b_instances():
    inst = synthetic_instance()
    inst["during"] = dict(t=[0.1], points=[[[0.0, 0.0, 1, 3]]])
    with pytest.raises(D.DashboardError, match="withheld-interval"):
        I.infer_instance(inst, None)
    assert list(I.infer_instance.__code__.co_varnames[:I.infer_instance.__code__.co_argcount]) == ["inst", "fm", "cfg"]


# ---------------------------------------------------------------- assignment and thresholds (no model needed)

def test_check_assignment_detects_violations():
    pairs = [dict(incoming=i, outgoing=o, reachable=True, probability=p) for i, o, p in
             (("in-0", "out-0", 0.95), ("in-0", "out-1", 0.1), ("in-1", "out-0", 0.2), ("in-1", "out-1", 0.9))]
    ops = dict(relaxed=dict(threshold=0.76, margin=1.0), conservative=dict(threshold=0.87, margin=1.0))
    ok = [dict(incoming="in-0", decision="MATCHED", outgoing="out-0", candidate="out-0", probability=0.95, margin=3.0),
          dict(incoming="in-1", decision="MATCHED", outgoing="out-1", candidate="out-1", probability=0.9, margin=3.0)]
    res = dict(pairs=pairs, operating_points=ops, decisions=dict(relaxed=ok, conservative=ok))
    assert I.check_assignment(res) == []
    dup = [dict(ok[0]), dict(ok[1], outgoing="out-0", candidate="out-0", probability=0.2)]
    assert any("matched twice" in m for m in I.check_assignment(dict(res, decisions=dict(relaxed=dup, conservative=ok))))
    low = [dict(ok[0], probability=0.8), ok[1]]
    assert any("threshold" in m for m in I.check_assignment(dict(res, decisions=dict(relaxed=ok, conservative=low))))


# ---------------------------------------------------------------- pitch coordinates (synthetic)

def test_pitch_uses_source_coordinates_unchanged():
    shapes = P.pitch_shapes(105.0, 68.0)
    assert (shapes[0]["x0"], shapes[0]["y0"], shapes[0]["x1"], shapes[0]["y1"]) == (-52.5, -34.0, 52.5, 34.0)
    inst = synthetic_instance(gap=1.0)
    times = P.frame_times(inst, None, 1)
    k = int(np.argmin(np.abs(times - (-1.0))))
    fig = P.replay_figure(inst, None, step=1, start=k)
    a = inst["incoming"][0]
    j = int(np.searchsorted(a["t"], times[k] + 1e-6) - 1)
    tr = fig["data"][1]
    assert (tr["x"][-1], tr["y"][-1]) == (a["x"][j], a["y"][j])          # no flip, scale or offset
    assert fig["layout"]["yaxis"]["scaleanchor"] == "x"
    k_mid = int(np.argmin(np.abs(times - 0.5)))
    mid = P.replay_figure(inst, None, step=1, start=k_mid)
    assert all(len(t["x"]) == 0 for t in mid["data"][3:5])            # no outgoing marker inside the withheld interval
    assert "WITHHELD" in mid["layout"]["annotations"][0]["text"]
    assert len(fig["frames"]) == len(times)


# ---------------------------------------------------------------- real development data

@pytest.fixture(scope="module")
def fm():
    return I.load_frozen_model(RAW)


@pytest.fixture(scope="module")
def dev():
    return D.load_match(RAW, DEV_MATCH)


def _hashes(paths):
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in paths}


@REAL
def test_real_development_matches_come_from_the_frozen_split(fm):
    ids = D.development_matches(RAW)
    split = D.split_manifest(RAW)
    assert ids and set(ids) <= set(fm.spec["splits"]["development"])
    assert all(split["matches"][m]["split"] == "development" for m in ids)
    assert not set(ids) & set(fm.spec["splits"]["validation"] + fm.spec["splits"]["test_locked"])
    for m in fm.spec["splits"]["validation"][:1] + fm.spec["splits"]["test_locked"][:1]:
        with pytest.raises(D.DashboardError):
            D.load_match(RAW, m)


@REAL
def test_frozen_model_and_feature_schema(fm):
    from football_fatigue.matching.locked_test import FROZEN_MODEL_SHA256, FROZEN_SPEC_SHA256

    assert (fm.model_sha256, fm.spec_sha256) == (FROZEN_MODEL_SHA256, FROZEN_SPEC_SHA256)
    assert fm.features == validate.FROZEN_FEATURES and len(fm.features) == 51 and fm.bundle["model"].n_features_in_ == 51
    assert not {"conf_in", "conf_out"} & set(fm.features)
    with pytest.raises(D.DashboardError, match="differs from the recorded"):
        I.load_frozen_model(RAW, expected=("0" * 64, fm.spec_sha256))


@REAL
def test_operating_points_come_from_the_frozen_spec(fm):
    assert fm.operating_points == dict(relaxed=dict(threshold=0.76, margin=1.0), conservative=dict(threshold=0.87, margin=1.0))
    assert validate.OPERATING["relaxed"]["threshold"] == 0.75        # the code default was resolved to 0.76 at freeze time


@REAL
def test_exact_reproduction_of_batch_inference_on_development(fm, dev):
    res = I.compare_with_batch(list(dev.instances.values())[:60], fm)
    assert res["max_abs_probability_diff"] == 0.0 and res["decision_mismatches"] == [] and res["assignment_problems"] == []


@REAL
def test_exact_reproduction_of_saved_frozen_predictions(fm):
    """The saved Step 6 frozen predictions of a validation match (matcher-visible instances only; no reference is read)."""
    mid = fm.spec["splits"]["validation"][0]
    pred_dir = PROC / validate.VALIDATION_DIR / "predictions"
    if not pred_dir.exists():
        pytest.skip("saved Step 6 predictions not present")
    validate.verify_dir(pred_dir)
    insts, _ = validate.load_visible(RAW, mid, D.split_manifest(RAW), PROC / validate.VALIDATION_DIR / "matching")
    insts = insts[:40]
    want = {i["instance_id"] for i in insts}
    saved_p = {}
    for line in (pred_dir / "pair_probabilities.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["match_id"] == mid and r["instance_id"] in want:
            saved_p[(r["instance_id"], r["incoming"], r["outgoing"])] = r["probability"]["frozen"]
    saved_d = {}
    for line in (pred_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["match_id"] == mid and r["instance_id"] in want and r["method"].startswith("frozen@"):
            saved_d[(r["method"], r["instance_id"])] = r["decisions"]
    n = 0
    for inst in insts:
        res = I.infer_instance(inst, fm)
        for q in res["pairs"]:
            assert round(q["probability"], 6) == saved_p[(inst["instance_id"], q["incoming"], q["outgoing"])]
            n += 1
        for pt in I.POINTS:
            assert res["decisions"][pt] == saved_d[(f"frozen@{pt}", inst["instance_id"])]
    assert n == len(saved_p) > 0


@REAL
def test_threshold_rule_and_assignment_on_real_encounters(fm, dev):
    for inst in list(dev.instances.values())[:80]:
        res = I.infer_instance(inst, fm)
        assert I.check_assignment(res) == []
        for pt, op in fm.operating_points.items():
            for d in res["decisions"][pt]:
                clears = d["margin"] is not None and d["probability"] >= op["threshold"] and d["margin"] >= op["margin"]
                assert (d["decision"] == "MATCHED") == clears
        rel = {d["incoming"] for d in res["decisions"]["relaxed"] if d["decision"] == "MATCHED"}
        con = {d["incoming"] for d in res["decisions"]["conservative"] if d["decision"] == "MATCHED"}
        assert con <= rel


@REAL
def test_path_contributions_are_exact_and_leave_predictions_unchanged(fm, dev):
    inst = next(iter(dev.instances.values()))
    res = I.infer_instance(inst, fm)
    for k, q in enumerate(res["pairs"]):
        pc = I.path_contributions(fm, res["Xf"][k])
        assert pc["additive_error"] < 1e-9 and pc["raw_probability"] == pytest.approx(q["raw_probability"], abs=1e-12)
    assert I.infer_instance(inst, fm)["pairs"] == res["pairs"]          # bit-identical repeat (single-threaded forest)
    assert fm.forest.n_jobs == 1
    assert set(I.global_importance(fm)) == set(fm.features)
    assert all(I.feature_group(f) != "Other" for f in fm.features)


@REAL
def test_training_pairs_match_the_label_protocol(dev):
    n = 0
    for iid, inst in list(dev.instances.items())[:200]:
        ref = dev.provider_reference(iid)
        if ref["tier"] == D.UNVERIFIED_TIER:
            continue
        want = {(r["incoming"], r["outgoing"]) for r in ml.pair_labels(ref, inst) if r["status"] == "train"}
        assert D.training_pairs(inst, ref["tier"]) == want
        n += len(want)
    assert n > 0


@REAL
def test_real_contexts_hold_no_identity(dev):
    for ctx in dev.contexts.values():
        assert not set(D.HIDDEN_KEYS) & set(ctx)
        assert not [k for k in ctx if any(w in k for w in IDENTITY_WORDS)]


@REAL
def test_coordinate_mapping_from_raw_cache(dev):
    cache = D.load_cache(RAW, DEV_MATCH)
    iid = next(i for i, c in dev.contexts.items() if c["tier"] == D.LABEL_TIER)
    inst, ctx, ref = dev.instances[iid], dev.contexts[iid], dev.provider_reference(iid)
    w = D.window_rows(ctx)
    for side, rows, trs in (("incoming", w["before"], inst["incoming"]), ("outgoing", w["after"], inst["outgoing"])):
        for tr in trs:
            s = ref[side][tr["tracklet_id"]]["slot"]
            x, y = cache["x"][rows[0]:rows[1], s], cache["y"][rows[0]:rows[1], s]
            ok = np.isfinite(x)
            assert x[ok].tolist() == tr["x"] and y[ok].tolist() == tr["y"]
    win = D.raw_window(cache, ctx, dev.participant_slots(iid))
    assert not set(win["slots"]) & dev.participant_slots(iid)
    assert win["t"][w["during"][0] - w["before"][0]] == 0.0
    hid = D.hidden_interval(cache, ctx, ref)
    assert {h["slot"] for h in hid} == {v["slot"] for v in list(ref["incoming"].values()) + list(ref["outgoing"].values())}
    for h in hid:
        x = cache["x"][w["during"][0]:w["during"][1], h["slot"]]
        assert h["x"].tolist() == x[np.isfinite(x)].tolist()
    meta = D.match_metadata(RAW, DEV_MATCH)
    assert (meta.pitch_length_m, meta.pitch_width_m) == (105.0, 68.0)
    assert 0 <= D.period_clock(meta, ctx["period"], ctx["start_time_s"]) < 3600
    assert np.nanmax(np.abs(cache["x"])) < meta.pitch_length_m / 2 + 10 and np.nanmax(np.abs(cache["y"])) < meta.pitch_width_m / 2 + 10


@REAL
def test_research_results_read_from_saved_scoring():
    rr = R.research_results(RAW)
    t = rr["rows"]["test"]
    if t is None:
        pytest.skip("locked-test scoring not present")
    s = json.loads((PROC / "m11-test-v1" / "scoring" / "summary.json").read_text(encoding="utf-8"))
    hi = s["encounter"]["high_confidence_reference"]
    assert t["roc_auc"] == s["pairwise"]["high_confidence_reference"]["frozen"]["roc_auc"]
    assert t["relaxed"]["coverage"] == hi["frozen@relaxed"]["all"]["coverage"]
    assert t["conservative"]["false_association_rate"] == hi["frozen@conservative"]["all"]["false_association_rate"]
    assert "not verified identity accuracy" in rr["note"]


@REAL
def test_app_hides_reference_until_revealed_and_writes_nothing(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    watched = sorted((PROC / validate.FROZEN_DIR).glob("*")) + sorted((PROC / "m11-matching-v3" / DEV_MATCH).glob("*")) \
        + sorted((PROC / "m11-encounters-v2" / DEV_MATCH).glob("*"))
    before = _hashes(watched)
    monkeypatch.setenv("M11_DASHBOARD_ROOT", str(RAW))
    at = AppTest.from_file(str(REPO / "src" / "football_fatigue" / "m11_dashboard" / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value(DEV_MATCH).run()
    assert not at.exception and not at.error

    def shown() -> str:
        parts = [df.value.to_csv() for df in at.dataframe] + [m.value for m in at.markdown] + [w.value for w in at.warning]
        return "\n".join(parts)

    assert "Provider identity" not in shown() and "home #" not in shown() and "away #" not in shown()
    next(b for b in at.button if b.label == "Run Identity Recovery").click().run()
    assert not at.exception and not at.error
    assert "Provider identity" not in shown() and "home #" not in shown()
    next(b for b in at.button if b.label == "Reveal Provider Reference").click().run()
    assert not at.exception
    assert "Provider identity" in shown() and ("home #" in shown() or "away #" in shown())
    assert any("not** verified identity accuracy" in w.value for w in at.warning)
    assert _hashes(watched) == before


# ---------------------------------------------------------------- Step 8.1: presentation logic (synthetic)

def _decide(P_, R_=None, ops=(("relaxed", 0.76), ("conservative", 0.87))):
    """Records of the real frozen decision rule (``ml.instance_assignments`` + ``ml.decisions``) on a synthetic probability matrix."""
    from dataclasses import replace

    P_ = np.asarray(P_, float)
    R_ = np.ones_like(P_, bool) if R_ is None else np.asarray(R_, bool)
    rows = [dict(instance_id="i", incoming=f"in-{i}", outgoing=f"out-{j}", reachable=bool(R_[i, j])) for i in range(P_.shape[0]) for j in range(P_.shape[1])]
    p = np.array([P_[i, j] for i in range(P_.shape[0]) for j in range(P_.shape[1])])
    assigned = ml.instance_assignments(rows, p)
    pairs = [dict(incoming=r["incoming"], outgoing=r["outgoing"], reachable=r["reachable"], probability=float(v)) for r, v in zip(rows, p)]
    dec = {pt: ml.decisions(assigned, thr, replace(ml.MLConfig(), margin_logit=1.0))["i"] for pt, thr in ops}
    return dict(pairs=pairs, decisions=dec, operating_points={pt: dict(threshold=thr, margin=1.0) for pt, thr in ops})


def test_explanation_follows_assignment_threshold_and_margin():
    # in-0's best candidate (out-0, 0.95) is not its assigned link: the joint assignment gives out-0 to in-1
    res = _decide([[0.95, 0.90], [0.92, 0.10]])
    rows = {r["incoming"]: r for r in PR.assignment_rows(res)}
    e = rows["in-0"]["explanations"]["relaxed"]
    assert e["consistent"] and e["assigned_to"] == "out-1" and e["best"] == "out-0"
    assert "not to its highest-scoring candidate out-0" in e["reasons"][0] and "goes to in-1" in e["reasons"][0]
    # a high probability that is not committed: one outgoing tracklet, two strong incoming candidates
    res = _decide([[0.90], [0.80]])
    e = PR.assignment_rows(res)[1]["explanations"]["relaxed"]
    assert e["decision"] == "UNCERTAIN" and not e["assigned"] and e["consistent"]
    assert "linked to in-0 instead" in e["reasons"][0]
    # below the 0.5 assignment floor
    e = PR.assignment_rows(_decide([[0.30]]))[0]["explanations"]["relaxed"]
    assert e["decision"] == "UNCERTAIN" and "never links a pair below 0.5" in e["reasons"][0] and e["consistent"]
    # assigned, clears relaxed, not conservative
    e = PR.assignment_rows(_decide([[0.80]]))[0]["explanations"]
    assert e["relaxed"]["decision"] == "MATCHED" and e["relaxed"]["threshold_ok"] and e["relaxed"]["margin_ok"]
    assert e["conservative"]["decision"] == "UNCERTAIN" and e["conservative"]["threshold_ok"] is False and e["conservative"]["consistent"]
    # a close alternative: high p but small margin
    e = PR.assignment_rows(_decide([[0.90, 0.88], [0.88, 0.90]]))[0]["explanations"]["relaxed"]
    assert e["decision"] == "UNCERTAIN" and e["threshold_ok"] and e["margin_ok"] is False and "close alternative" in e["reasons"][2]
    # nothing reachable
    e = PR.assignment_rows(_decide([[0.9]], [[False]]))[0]["explanations"]["relaxed"]
    assert e["decision"] == "UNMATCHED" and not e["reachable"] and e["consistent"]


def test_tracklet_colours_never_suggest_a_pairing():
    inst = synthetic_instance()
    c = PR.tracklet_colors(inst)
    assert set(c) == {"in-0", "in-1", "out-0", "out-1"}
    assert not {c["in-0"], c["in-1"]} & {c["out-0"], c["out-1"]} and not set(PR.IN_PALETTE) & set(PR.OUT_PALETTE)


def test_demo_examples_use_public_metadata_only():
    def ctx(i, t, tier, n, gap):
        return dict(instance_id=i, period=1, start_time_s=t, tier=tier, n_incoming=n, ambiguity_s=gap)
    cs = {c["instance_id"]: c for c in (ctx("a", 5, D.LABEL_TIER, 2, 0.8), ctx("b", 9, D.LABEL_TIER, 2, 1.2), ctx("c", 3, D.LABEL_TIER, 3, 0.5),
                                         ctx("d", 1, D.EXPLORATORY_TIER, 3, 3.0))}
    ex = PR.demo_examples(cs)
    assert [(e["key"], e["instance_id"]) for e in ex] == [("A", "b"), ("B", "c"), ("C", None)]
    # extra (hidden) keys cannot influence the choice: the rules only read DEMO_KEYS
    noisy = {k: dict(v, expected={"in-0": "out-1"}, difficulty={"crossing": True}) for k, v in cs.items()}
    assert PR.demo_examples(noisy) == ex
    assert set(PR.DEMO_KEYS) <= set(D.PUBLIC_KEYS) | {"n_incoming", "ambiguity_s"}


def test_feature_categories_cover_the_frozen_schema_exactly_once():
    names = [f for c in PR.CATEGORIES.values() for f in c["features"]]
    assert len(names) == len(set(names)) == 51
    assert set(names) == set(validate.FROZEN_FEATURES)
    assert list(PR.CATEGORIES) == ["Position and distance", "Speed and acceleration", "Direction and turning",
                                   "Gap and trajectory continuity", "Assignment ambiguity"]


def test_candidate_figure_marks_assignment_and_frozen_thresholds():
    res = _decide([[0.95, 0.90], [0.92, 0.10]])
    fig = PR.candidate_figure("in-0", res["pairs"], "out-1", res["operating_points"], PR.tracklet_colors(synthetic_instance()))
    bar = fig["data"][0]
    assert bar["y"] == ["out-0", "out-1"] and bar["x"] == [0.95, 0.90]
    assert "assigned" in bar["text"][1] and "assigned" not in bar["text"][0]
    assert sorted(s["x0"] for s in fig["layout"]["shapes"]) == [0.5, 0.76, 0.87]


def test_board_hides_reference_until_reveal_and_labels_links():
    inst = synthetic_instance()
    times = P.frame_times(inst, None, 1)
    k_mid = int(np.argmin(np.abs(times - 0.5)))
    res = _decide([[0.95, 0.05], [0.05, 0.95]])
    for zoom in P.ZOOMS:
        for animate in (True, False):
            fig = P.replay_figure(inst, None, step=1, start=k_mid, zoom=zoom, animate=animate, decisions=res["decisions"]["relaxed"])
            s = json.dumps(fig, default=str).lower()
            assert "reference" not in s and "provider" not in s and "home #" not in s and "away #" not in s
            assert "p 0.95" in s                                                      # the model's links are labelled
    assert not P.replay_figure(inst, None, step=1, animate=False)["frames"]
    rv = dict(colors={"in-0": "home #3", "in-1": "home #7", "out-0": "home #3", "out-1": "home #7"},
              hidden=[dict(label="home #3", t=np.array([0.1, 0.2]), x=np.array([0.3, 0.6]), y=np.array([0.0, 0.0]))])
    s = json.dumps(P.replay_figure(inst, None, step=1, start=k_mid, reveal=rv), default=str)
    assert "home #3" in s and "provider reference" in s
    mid = P.replay_figure(inst, None, step=1, start=k_mid)
    assert mid["layout"]["annotations"][1]["visible"] is True and "WITHHELD" in mid["layout"]["annotations"][1]["text"]
    sb = P.storyboard_times(inst, None, 1)
    assert times[sb["before"]] < 0 <= times[sb["during"]] <= 1.0 < times[sb["after"]]


# ---------------------------------------------------------------- Step 8.1: real development data

@REAL
def test_explanations_agree_with_frozen_decisions_on_real_encounters(fm, dev):
    for inst in list(dev.instances.values())[:150]:
        res = I.infer_instance(inst, fm)
        for row in PR.assignment_rows(res):
            assert all(e["consistent"] for e in row["explanations"].values())
        # the assignment is threshold-free: the same candidate and margin at both points
        assert [(d["candidate"], d["margin"]) for d in res["decisions"]["relaxed"]] == \
            [(d["candidate"], d["margin"]) for d in res["decisions"]["conservative"]]


@REAL
def test_demo_examples_are_the_documented_development_encounters(dev):
    ex = PR.demo_examples(dev.contexts)
    assert [(e["key"], e["instance_id"][:8]) for e in ex] == [("A", "453e55e4"), ("B", "0c080ccb"), ("C", "82a4f751")]
    assert all(e["instance_id"] in dev.instances for e in ex)


@REAL
def test_research_counts_reproduce_the_saved_rates():
    rr = R.research_results(RAW)
    for name, r in rr["rows"].items():
        if r is None:
            continue
        for pt in I.POINTS:
            c = r[pt]["counts"]
            assert c["matched"] == c["correct"] + c["wrong"] and c["matched"] <= c["incoming"]
            assert c["matched"] / c["incoming"] == pytest.approx(r[pt]["coverage"], abs=1e-4)
            assert c["wrong"] / c["matched"] == pytest.approx(r[pt]["false_association_rate"], abs=1e-4)
    t = rr["rows"]["test"]
    if t is not None:
        assert t["relaxed"]["counts"] == dict(incoming=5555, matched=2240, correct=2042, wrong=198)
        assert t["pairs"]["positives"] + t["pairs"]["negatives"] == t["pairs"]["pairs"]


def _walk(node):
    yield node
    for c in getattr(node, "children", {}).values():
        yield from _walk(c)


def _visible(at) -> str:
    """Everything the page sent: every element's proto (text, labels, help, tooltips, plotly specs) plus dataframe contents."""
    return "\n".join([str(getattr(n, "proto", "")) for n in _walk(at._tree)] + [df.value.to_csv() for df in at.dataframe])


@REAL
def test_no_reference_leak_before_reveal_in_any_view(monkeypatch):
    """Advisor Demo examples A, B, C through the whole page: no provider-reference call and no identity text before Reveal."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    calls = []
    orig_ref, orig_hidden = D.MatchData.provider_reference, D.hidden_interval
    monkeypatch.setattr(D.MatchData, "provider_reference", lambda self, i: calls.append(("ref", i)) or orig_ref(self, i))
    monkeypatch.setattr(D, "hidden_interval", lambda *a, **k: calls.append(("hidden",)) or orig_hidden(*a, **k))
    monkeypatch.setenv("M11_DASHBOARD_ROOT", str(RAW))
    md = D.load_match(RAW, DEV_MATCH)
    at = AppTest.from_file(str(REPO / "src" / "football_fatigue" / "m11_dashboard" / "app.py"), default_timeout=300)
    at.run()
    for key in "ABC":
        next(b for b in at.button if b.label == f"Load example {key}").click().run()
        assert not at.exception and not at.error
        iid = at.session_state[f"enc_{DEV_MATCH}"]
        ref = orig_ref(md, iid)
        tokens = {D.identity_label(v) for v in list(ref["incoming"].values()) + list(ref["outgoing"].values())}
        calls.clear()
        for step in ("loaded", "run"):
            if step == "run":
                next(b for b in at.button if b.label == "Run Identity Recovery").click().run()
                assert not at.exception and not at.error
                assert any("Global assignment and decisions" in m.value for m in at.markdown)
            text = _visible(at)
            assert calls == [], f"example {key}: reference read before reveal ({step})"
            leaked = [t for t in tokens if t in text]
            assert not leaked, f"example {key}: identity {leaked} visible before reveal ({step})"
            for word in ("Provider identity", "Provider continuation", "provider reference, withheld path", "REFERENCE ONLY", "crossing = ",
                         "team relation", "player_id", "jersey"):
                assert word not in text, f"example {key}: '{word}' visible before reveal ({step})"
        next(b for b in at.button if b.label == "Reveal Provider Reference").click().run()
        assert not at.exception and not at.error
        text = _visible(at)
        assert calls and all(t in text for t in tokens) and "Provider identity" in text
        next(b for b in at.button if b.label == "Hide provider reference").click().run()
        assert not any(t in _visible(at) for t in tokens)
    # the in-sample / not-generalisation statement is on the demo page
    assert any("not independent evidence" in i.value for i in at.info)
