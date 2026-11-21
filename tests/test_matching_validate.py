"""Tests for football_fatigue.matching.validate (M11 step 6). All instances are SYNTHETIC hand-built task B instances; passing
these tests says the frozen-evaluation pipeline is wired correctly, not that it is accurate on real data."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from football_fatigue.ingestion import IngestionError  # noqa: E402
from football_fatigue.matching import ml, ml2, validate  # noqa: E402
from football_fatigue.matching.benchmark import SCHEMA_VERSION, TASK_B, output_paths  # noqa: E402

CFG = ml2.ML2Config(n_jobs_folds=1, n_jobs_model=1)


# ---------------------------------------------------------------- synthetic instances (as in test_matching_ml)

def track(tid, f, t0, t1, vis=1, step=2):
    k = np.arange(round(t0 * 30), round(t1 * 30) + 1)
    kq = k - (k - k[0]) % step
    P = np.array([f(q / 30) for q in kq])
    return dict(tracklet_id=tid, t=[round(q / 30, 4) for q in k], x=P[:, 0].tolist(), y=P[:, 1].tolist(),
                visibility=[vis] * len(k), confidence=[3] * len(k))


def instance(iid, players, gap=1.5):
    inc = [track(i, f, -2.0, -1 / 30) for i, _, f in players]
    out = [track(o, f, gap, gap + 2.0) for _, o, f in players]
    return dict(instance_id=iid, schema="test", task=TASK_B, units={}, ambiguity_interval_s=[0.0, gap], incoming=inc, outgoing=out,
                during=dict(t=[], points=[]))


def random_encounter(rng, iid, tier=ml.LABEL_TIER):
    gap = float(rng.uniform(1.0, 3.0))
    players, expected = [], {}
    outs = [f"out-{k}" for k in rng.permutation(2)]
    for k in range(2):
        p, v, w = rng.uniform(-3, 3, 2), rng.uniform(-5, 5, 2), rng.uniform(-0.3, 0.3)
        f = (lambda p, v, w: lambda t: (p[0] + v[0] * t + w * t * t, p[1] + v[1] * t - w * t * t))(p, v, w)
        players.append((f"in-{k}", outs[k], f))
        expected[f"in-{k}"] = outs[k]
    inst = instance(iid, players, gap=gap)
    inst["outgoing"].sort(key=lambda b: b["tracklet_id"])
    ref = dict(instance_id=iid, tier=tier, expected=expected, incoming={k: {} for k in expected}, outgoing={b["tracklet_id"]: {} for b in inst["outgoing"]},
               team_relation="opposite_team", without_incoming=[], without_outgoing=[], step1_categories={},
               difficulty=dict(label="standard", reasons=[], ambiguity_s=gap, crossing=False, max_heading_change_deg=None))
    return inst, ref


def data(match_id, n, seed):
    rng = np.random.default_rng(seed)
    pairs = [random_encounter(rng, f"{match_id}-{k:03d}") for k in range(n)]
    return dict(match_id=match_id, instances=[i for i, _ in pairs], references=[r for _, r in pairs])


@pytest.fixture(scope="module")
def dev():
    return [ml2.match_table2(data(m, 16, s)) for m, s in (("d1", 1), ("d2", 2), ("d3", 3), ("d4", 4))]


@pytest.fixture(scope="module")
def frozen(dev):
    fz = validate.freeze_core(dev, CFG, dev_reference=False)
    bundle = dict(model=fz["model"], features=list(validate.FROZEN_FEATURES), calibrator=vars(fz["calibrator"]), operating=fz["operating"])
    return fz, bundle, dict(operating_points=fz["operating"])


# ---------------------------------------------------------------- schema and leakage

def test_frozen_feature_schema():
    f = validate.FROZEN_FEATURES
    assert len(f) == len(ml2.FEATURE_SETS["core_g"]) - 2 == 51
    assert not set(validate.EXCLUDED_FEATURES) & set(f) and set(f) <= set(ml2.FEATURES2)
    assert not set(ml2.FAMILIES["F_footballtrace"]) & set(f)


def test_features_only_equals_training_features_without_references():
    d = data("v", 5, 9)
    rows, X = validate.features_only(d["instances"])
    t = ml2.match_table2(d)
    assert np.array_equal(X, t["X"]) and [(r["instance_id"], r["incoming"], r["outgoing"]) for r in rows] == \
        [(r["instance_id"], r["incoming"], r["outgoing"]) for r in t["rows"]]
    tagged = copy.deepcopy(d["instances"])
    for i in tagged:
        for tr in i["incoming"] + i["outgoing"]:
            tr.update(player_id=7, jersey=10)
    assert np.array_equal(validate.features_only(tagged)[1], X)


def test_split_membership_and_test_lock():
    split = dict(matches={"1": dict(split="development"), "2": dict(split="validation"), "3": dict(split="test")})
    validate.check_split(split, ["2"], "validation")
    with pytest.raises(IngestionError, match="test"):
        validate.check_split(split, ["2", "3"], "validation")
    with pytest.raises(IngestionError, match="test"):
        validate.check_split(split, ["3"], "test")
    with pytest.raises(IngestionError, match="only validation"):
        validate.check_split(split, ["1"], "validation")
    with pytest.raises(IngestionError, match="only development"):
        validate.check_split(split, ["2"], "development")


# ---------------------------------------------------------------- freeze, frozen prediction

def test_preselected_points_kept_or_resolved(dev, monkeypatch):
    fz = validate.freeze_core(dev, CFG, dev_reference=False)
    for name, c in fz["operating_check"].items():
        assert fz["operating"][name] == (c["preselected"] if c["compatible"] else fz["operating"][name])
    monkeypatch.setitem(validate.OPERATING, "relaxed", dict(threshold=0.5, margin=0.0, target=-1.0))   # impossible target
    fz2 = validate.freeze_core(dev, CFG, dev_reference=False)
    assert not fz2["operating_check"]["relaxed"]["compatible"]
    alt = fz2["operating_check"]["relaxed"]["inner_selection_for_reference"]
    assert fz2["operating"]["relaxed"] == dict(threshold=alt["threshold"], margin=alt["margin"], target=-1.0)


def test_frozen_prediction_fits_nothing_and_is_reproducible(frozen, monkeypatch):
    fz, bundle, spec = frozen
    d = data("v1", 8, 11)
    rows, X = validate.features_only(d["instances"])
    tables = [dict(match_id="v1", rows=[dict(r, match_id="v1") for r in rows], X=X)]

    def boom(*a, **k):
        raise AssertionError("no fitting on validation data")

    from sklearn.ensemble import RandomForestClassifier
    monkeypatch.setattr(ml2, "fit_model", boom)
    monkeypatch.setattr(ml2, "fit_platt", boom)
    monkeypatch.setattr(RandomForestClassifier, "fit", boom)
    a = validate.predict_tables(tables, spec, bundle, {})
    b = validate.predict_tables(tables, spec, bundle, {})
    assert a == b
    pairs, preds = a
    assert {p["method"] for p in preds} == {"frozen@conservative", "frozen@relaxed"}
    for p in preds:
        claimed = [d["outgoing"] for d in p["decisions"] if d["decision"] == "MATCHED"]
        assert len(claimed) == len(set(claimed))                     # one-to-one
        for d in p["decisions"]:
            if d["decision"] == "MATCHED":
                assert d["probability"] >= p["threshold"] and d["margin"] >= p["margin"]
            if d["decision"] == "UNMATCHED":
                assert d["candidate"] is None                         # only when nothing is reachable


def test_criteria_outcome():
    good = {m: dict(incoming=100, matched=40, correct=37, wrong=3) for m in "abcd"}
    base = {m: dict(incoming=100, matched=15, correct=14, wrong=1) for m in "abcd"}
    c = validate.criteria_outcome(good, base, good, base)
    assert c["primary"]["met"] and not c["secondary"]["met"]          # 7.5% false association: <= 10%, > 5%
    bad = {m: dict(incoming=100, matched=40, correct=34, wrong=6) for m in "abcd"}
    assert not validate.criteria_outcome(bad, base, bad, base)["primary"]["met"]


# ---------------------------------------------------------------- non-destructive I/O and checksums

def test_write_dir_checksums_and_no_overwrite(tmp_path):
    out = validate._write_dir(tmp_path / "p", {"a.jsonl": b"x\n"}, {}, dict(schema="t"))
    assert validate.verify_dir(out)["outputs_sha256"]["a.jsonl"]
    with pytest.raises(IngestionError, match="never overwritten"):
        validate._write_dir(tmp_path / "p", {"a.jsonl": b"y\n"}, {}, dict(schema="t"))
    (out / "a.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(IngestionError, match="checksum"):
        validate.verify_dir(out)


def test_raw_snapshot_detects_writes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "f.csv").write_text("1")
    a = validate.raw_snapshot(raw)
    assert validate.raw_snapshot(raw) == a
    (raw / "f.csv").write_text("12")
    assert validate.raw_snapshot(raw) != a


def test_load_visible_reads_no_reference(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    root = tmp_path / "processed" / "m11-validation-v1" / "matching"
    d = data("2", 3, 5)
    paths = output_paths(raw, "2", root)
    paths[".instances_b.jsonl"].parent.mkdir(parents=True, exist_ok=True)
    paths[".instances_b.jsonl"].write_text("".join(json.dumps(i) + "\n" for i in d["instances"]), encoding="utf-8")
    from football_fatigue.matching.benchmark import _sha256
    man = dict(schema=SCHEMA_VERSION, match_id="2", split=dict(split="validation"), synthetic_swaps_injected=False, provider_smoothed_coordinates_used=False,
               outputs_sha256={paths[".instances_b.jsonl"].name: _sha256(paths[".instances_b.jsonl"])})
    paths[".manifest.json"].write_text(json.dumps(man), encoding="utf-8")
    assert not paths[".references.jsonl"].exists()
    split = dict(matches={"2": dict(split="validation"), "3": dict(split="test")})
    inst, _ = validate.load_visible(raw, "2", split, root)
    assert [i["instance_id"] for i in inst] == [i["instance_id"] for i in d["instances"]]
    with pytest.raises(IngestionError, match="test"):
        validate.load_visible(raw, "3", split, root)
    paths[".instances_b.jsonl"].write_text("", encoding="utf-8")
    with pytest.raises(IngestionError, match="checksum"):
        validate.load_visible(raw, "2", split, root)


def test_run_freeze_refuses_non_development(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sp = tmp_path / "processed" / ml.SPLITS_DIR
    sp.mkdir(parents=True)
    (sp / "splits.json").write_text(json.dumps(dict(matches={"1": dict(split="development"), "2": dict(split="validation"), "3": dict(split="test")})))
    with pytest.raises(IngestionError, match="only development"):
        validate.run_freeze(raw, ["1", "2"], out_root=tmp_path / "f")
    with pytest.raises(IngestionError, match="test"):
        validate.run_freeze(raw, ["3"], out_root=tmp_path / "f")
    with pytest.raises(IngestionError, match="only validation"):
        validate.build_validation_match(raw, "1", val_root=tmp_path / "v")
    with pytest.raises(IngestionError, match="test"):
        validate.build_validation_match(raw, "3", val_root=tmp_path / "v")
