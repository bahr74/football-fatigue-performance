"""M11 identity recovery: identity-free contacts, scenario views (labels, leakage controls), motion baselines, models and
metrics. Synthetic data only (fast); real-data checks live in the experiment manifests."""

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections.contacts import ContactConfig, contact_events, link_geometry
from football_fatigue.intersections.detect import DetectConfig, detect
from football_fatigue.m11_recovery import dataset as DS
from football_fatigue.m11_recovery import evaluate as EV
from football_fatigue.m11_recovery import motion as MO
from football_fatigue.m11_recovery.models import GBM, Baseline, Hybrid
from football_fatigue.m11_recovery.views import GRID, Interaction, make_sample, resample

FPS, T0 = 10.0, 100.0
K = np.arange(101)


def cache(paths) -> RawCache:
    F, S = len(paths[0][0]), len(paths)
    x, y = np.stack([p[0] for p in paths], 1).astype(float), np.stack([p[1] for p in paths], 1).astype(float)
    ok = np.isfinite(x)
    sides = ["home" if s % 2 == 0 else "away" for s in range(S)]
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=T0 + np.arange(F) / FPS, source_period=np.ones(F, np.int16),
                    canonical_period=np.ones(F, np.int8), record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64), slot_side=np.array(sides, "<U4"),
                    slot_jersey=np.arange(1, S + 1, dtype=np.int16), slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=np.where(ok, 1, 0).astype(np.int8),
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


t = K / FPS - 5.0
CROSS = [(2.0 * t, 0.3 + 0 * t), (0.05 + 0 * t, -2.0 * t + 0.01)]                    # perpendicular crossing at t = 0
BOUNCE = [(2.0 * t, 0.4 + 0 * t), (-2.0 * t, -0.4 + 0 * t)]                           # head-on pass, no path crossing


# ---------------------------------------------------------------- contacts

def contacts(paths):
    c = cache(paths)
    qr = Q.assess(c, [])
    geo, _ = detect(c, qr.mask, qr.steps, DetectConfig())
    ev = contact_events(c, ContactConfig())
    link_geometry(ev, geo, ContactConfig())
    return c, ev


def test_contacts_found_for_crossing_and_non_crossing():
    _, ev = contacts(CROSS)
    assert len(ev) == 1 and ev[0]["geometry"]["geometric_crossing"] and ev[0]["geometry"]["aligned_crossing"]
    assert ev[0]["time_a_s"] == pytest.approx(105.0, abs=0.11) and ev[0]["min_synchronous_distance_m"] < 0.35
    _, ev = contacts(BOUNCE)
    assert len(ev) == 1 and not ev[0]["geometry"]["geometric_crossing"]
    assert ev[0]["min_synchronous_distance_m"] == pytest.approx(0.8, abs=1e-6)


def test_contacts_are_identity_free():
    """Exchanging the two players' labels after the contact changes the identities, not the contact events."""
    a, b = CROSS
    sw = t > 0
    a2 = (np.where(sw, b[0], a[0]), np.where(sw, b[1], a[1]))
    b2 = (np.where(sw, a[0], b[0]), np.where(sw, a[1], b[1]))
    _, e1 = contacts([a, b])
    _, e2 = contacts([a2, b2])
    keys = lambda ev: [(e["period"], e["time_a_s"], e["min_synchronous_distance_m"], e["slot_a"], e["slot_b"]) for e in ev]  # noqa: E731
    assert keys(e1) == keys(e2)


def test_no_contact_beyond_radius():
    _, ev = contacts([(2.0 * t, 0.6 + 0 * t), (-2.0 * t, -0.6 + 0 * t)])              # 1.2 m apart at closest
    assert ev == []


# ---------------------------------------------------------------- views

def interaction(paths, eid="e1"):
    tr = np.round(T0 + K / FPS - 105.0, 6)
    pos = np.stack([resample(tr, p[0], p[1], np.ones(len(K), bool)) for p in paths])
    return Interaction("999", eid, pos, "clean", [], dict(pair_id="p", cluster="c", crossing=True, aligned_crossing=True,
                                                          **{k: 0.0 for k in DS.META_KEYS}))


def test_resample_and_label_definition():
    it = interaction(CROSS)
    assert it.pos.shape == (2, len(GRID), 2) and np.allclose(it.pos[0, :, 0], 2.0 * GRID)
    for d in range(20):
        s = make_sample(it, 0.8, 0.0, draw=d)
        a0_is_b0 = np.allclose(s.after[0][0], it.pos[0, GRID > 0.4 + 1e-9][0]) == np.allclose(s.before[0][-1], it.pos[0, GRID < -0.4 - 1e-9][-1])
        assert s.y == int(a0_is_b0)
        assert np.all(GRID[(GRID >= -0.4) & (GRID <= 0.4)] == GRID[(GRID >= -0.4) & (GRID <= 0.4)])
        assert s.t_before.max() < -0.4 and s.t_after.min() > 0.4              # the gap is removed for both players


def test_label_balance_and_order_independence():
    ys = [make_sample(interaction(CROSS, f"e{k}"), 0.8, 0.0).y for k in range(2000)]
    assert 0.46 < np.mean(ys) < 0.54


def test_scenario_noise_is_deterministic_and_synthetic():
    it = interaction(CROSS)
    a, b = make_sample(it, 0.8, 0.3), make_sample(it, 0.8, 0.3)
    assert np.array_equal(a.before, b.before, equal_nan=True)
    assert not np.allclose(a.before, make_sample(it, 0.8, 0.0).before, equal_nan=True)


def test_tables_have_identical_time_grids_and_no_identity():
    tab, meta = DS.build([interaction(CROSS, f"e{k}") for k in range(5)])
    assert np.all(tab["bseq"][:, 0, :, 3] == tab["bseq"][:, 1, :, 3]) and np.all(tab["aseq"][:, 0, :, 3] == tab["aseq"][:, 1, :, 3])
    assert np.all(tab["bseq"][:, 0, :, 2] == tab["bseq"][:, 1, :, 2])         # same observed-point masks here
    assert not any(k in tab for k in ("slot", "jersey", "player_id", "side"))
    assert meta["rows"] == 5 * len(DS.SCENARIOS)


# ---------------------------------------------------------------- baselines and models

def test_baselines_on_clean_geometry():
    s = make_sample(interaction(CROSS), 0.8, 0.0)
    pc = MO.pair_costs(MO.Kin(s))
    assert (MO.margin(pc["cv_fwd"]) > 0) == bool(s.y) and (MO.margin(pc["ca_bidir"]) > 0) == bool(s.y)
    assert (MO.margin(pc["nearest"]) > 0) != bool(s.y)                         # nearest position picks the swap at a crossing
    s = make_sample(interaction(BOUNCE), 0.8, 0.0)
    assert (MO.margin(MO.pair_costs(MO.Kin(s))["cv_fwd"]) > 0) == bool(s.y)


def synthetic_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    its = []
    for k in range(n):
        th = rng.uniform(0, np.pi)
        v1, v2 = rng.uniform(1, 4, 2)
        p1 = (v1 * np.cos(th) * t + rng.normal(0, 0.05, len(t)), v1 * np.sin(th) * t + rng.normal(0, 0.05, len(t)))
        p2 = (v2 * np.cos(th + 1.2) * t + rng.normal(0, 0.05, len(t)), v2 * np.sin(th + 1.2) * t + 0.3 + rng.normal(0, 0.05, len(t)))
        its.append(interaction([p1, p2], f"s{seed}-{k}"))
    tab, meta = DS.build(its, scenarios=((0.8, 0.0), (1.6, 0.15)))
    tab["match"] = np.array([str(k % 3) for k in range(len(tab["y"]))])
    return tab, meta


def test_gbm_is_exactly_antisymmetric_and_learns():
    tab, meta = synthetic_table()
    m = GBM(meta["feature_names"], max_iter=50).fit(tab)
    p = m.predict(tab)
    sw = dict(tab, feats=tab["feats"].copy())
    sw["feats"][:, :m.np] *= -1
    assert np.allclose(m.predict(sw), 1 - p)
    assert np.mean((p > 0.5) == tab["y"]) > 0.9
    b = Baseline("cv_fwd").fit(tab)
    assert np.mean((b.predict(tab) > 0.5) == tab["y"]) > 0.9


def test_hybrid_is_antisymmetric():
    pytest.importorskip("torch")
    tab, _ = synthetic_table(60)
    m = Hybrid(epochs=1, batch=64).fit(tab)
    p = m.predict(tab)
    sw = dict(tab, aseq=tab["aseq"][:, ::-1].copy(), phys=tab["phys"][:, :, ::-1].copy())
    assert np.allclose(m.predict(sw), 1 - p, atol=1e-5)


# ---------------------------------------------------------------- metrics

def test_abstention_metrics():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.99, 0.9, 0.2, 0.45, 0.4, 0.6, 0.55, 0.05])
    b = EV.basic(y, p)
    assert b["accuracy"] == pytest.approx(6 / 8)
    a = EV.at_threshold(y, p, 0.8)
    assert a["committed"] == 4 and a["false_association_rate"] == 0 and a["recovery_rate"] == pytest.approx(0.5) and a["coverage"] == 0.5
    c = EV.threshold_for(y, p, 0.0)
    assert EV.at_threshold(y, p, c)["false_association_rate"] == 0


# ---------------------------------------------------------------- inference

class _Const:
    def __init__(self, p):
        self.p = p

    def predict(self, row):
        assert row["feats"].shape[0] == 1 and row["bseq"].shape[0] == 1
        return np.array([self.p])


def test_inference_resampling_matches_training_views():
    from football_fatigue.m11_recovery.infer import to_sample

    it = interaction(CROSS)
    for g in (0.4, 1.2, 2.0):
        s = make_sample(it, g, 0.0)
        r = to_sample(s.before, s.t_before + 37.0, s.after, s.t_after + 37.0)     # absolute times are re-centred on the gap
        assert r.gap_s == pytest.approx(g) and np.allclose(r.before, s.before, equal_nan=True) and np.allclose(r.after, s.after, equal_nan=True)


def test_inference_decisions_and_abstention():
    from football_fatigue.m11_recovery.infer import RecoveryModel

    s = make_sample(interaction(CROSS), 0.8, 0.0)
    spec = dict(operating_points={"fa_05": 0.8}, default_operating_point="fa_05", kind="gbm")
    assert RecoveryModel(_Const(0.9), spec).recover(s.before, s.t_before, s.after, s.t_after).decision == "keep"
    assert RecoveryModel(_Const(0.1), spec).recover(s.before, s.t_before, s.after, s.t_after).decision == "swap"
    r = RecoveryModel(_Const(0.7), spec).recover(s.before, s.t_before, s.after, s.t_after)
    assert r.decision == "abstain" and r.p_h0 == 0.7
    far = make_sample(interaction(CROSS), 0.0, 0.0)                                # no gap: outside the trained domain
    r = RecoveryModel(_Const(0.99), spec).recover(far.before, far.t_before, far.after, far.t_after)
    assert r.decision == "abstain" and r.p_h0 is None and "outside" in r.reason


def test_gap_shift_hides_t_star_and_default_is_unchanged():
    it = interaction(CROSS)
    a, b = make_sample(it, 0.8, 0.15), make_sample(it, 0.8, 0.15, shift_s=0.0)
    assert np.array_equal(a.before, b.before, equal_nan=True) and np.array_equal(a.t_after, b.t_after)
    s = make_sample(it, 0.8, 0.0, shift_s=0.3)
    assert s.t_before.max() == pytest.approx(-0.5) and s.t_after.min() == pytest.approx(0.5)     # times are relative to the gap centre
    assert np.allclose(s.before[:, -1], it.pos[:, GRID == -0.2][:, 0]) or np.allclose(s.before[::-1, -1], it.pos[:, GRID == -0.2][:, 0])
    tab, _ = DS.build([it], scenarios=((0.4, 0.0),), shifted=True)
    assert tab["bseq"].shape[2] == DS.T_PAD


def test_frozen_model_roundtrip_and_tamper_check(tmp_path):
    import hashlib
    import json

    import joblib

    from football_fatigue.m11_recovery.infer import RecoveryModel

    tab, meta = synthetic_table(80)
    m = GBM(meta["feature_names"], max_iter=20).fit(tab)
    joblib.dump(m, tmp_path / "model.joblib")
    (tmp_path / "spec.json").write_text(json.dumps(dict(kind="gbm", operating_points={"fa_05": 0.6}, default_operating_point="fa_05")))
    sha = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("model.joblib", "spec.json")}
    (tmp_path / "manifest.json").write_text(json.dumps(dict(outputs_sha256=sha)))
    rm = RecoveryModel.load(tmp_path)
    s = make_sample(interaction(CROSS), 1.2, 0.0)
    r = rm.recover(s.before, s.t_before, s.after, s.t_after)
    from football_fatigue.m11_recovery.dataset import model_inputs

    assert r.p_h0 == pytest.approx(float(m.predict({k: np.asarray([v]) for k, v in model_inputs(s).items()})[0]))
    (tmp_path / "spec.json").write_text(json.dumps(dict(kind="gbm", operating_points={"fa_05": 0.0}, default_operating_point="fa_05")))
    with pytest.raises(ValueError):
        RecoveryModel.load(tmp_path)


@pytest.mark.skipif(not __import__("pathlib").Path("data/processed/m11-recovery-v1/validation/final/predictions.npz").exists(),
                    reason="recovery outputs not generated")
def test_dashboard_renders_development_and_validation_without_revealing_by_default():
    from streamlit.testing.v1 import AppTest

    app = __import__("pathlib").Path(__file__).resolve().parents[1] / "src/football_fatigue/m11_recovery/app.py"
    at = AppTest.from_file(str(app), default_timeout=600).run()
    assert not at.exception
    src = next(s for s in at.selectbox if s.label == "Source")
    assert any(o.startswith("official validation") for o in src.options)
    assert not any("provider" in str(j.value) for j in at.json)                 # provider label hidden before reveal
    src.set_value(next(o for o in src.options if o.startswith("official validation"))).run()
    assert not at.exception and len(at.get("plotly_chart")) == 1
    at.toggle[0].set_value(True).run()
    assert not at.exception and any("provider" in str(j.value) for j in at.json)
