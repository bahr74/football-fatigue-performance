"""M11 Trajectory Recovery Lab (Milestone 3): truth separation, animation timeline, match-to-fold selection, live inference,
scenario regeneration, raw-data provenance, protected-match refusal and the Streamlit UI flows. Unit tests are synthetic;
integration tests use the train/CV development data and are skipped when it is absent."""

import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.m11_dashboard import lab as L
from football_fatigue.m11_dashboard import lab_board as LB
from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P
from football_fatigue.window_intersections import samples as S

DT = 1.0 / 15.0
APP = Path(__file__).resolve().parents[1] / "src" / "football_fatigue" / "m11_dashboard" / "lab_app.py"


# ---------------------------------------------------------------- synthetic scenario (no data needed)

def _synthetic(swapped: bool = True):
    """Two players crossing at the origin; g0 = 3.0, g1 = 3.8. Track X is the continuation of Player 2 when ``swapped``."""
    tin = np.arange(0.0, 3.0 - 1e-9, DT)
    tout = np.arange(3.8 + DT, 6.8, DT)
    p1 = lambda t: (2.0 * (t - 3.4), 0.0 * t)          # noqa: E731  moves along +x
    p2 = lambda t: (0.0 * t, 2.0 * (t - 3.4))          # noqa: E731  moves along +y
    inc = [A.Track(tin, *p1(tin)), A.Track(tin, *p2(tin))]
    out = [A.Track(tout, *(p2 if swapped else p1)(tout)), A.Track(tout, *(p1 if swapped else p2)(tout))]
    pub = L.Public("synthetic", "0", 0, "c", "baseline", "base", dict(gap=0.8, shift=0.0, noise=0.0, dropout="none", context=2.0), True, "",
                   1, "synthetic", 3.0, 3.8, inc, out)
    pairing = P.SWAPPED if swapped else P.MATCHED
    ht = 3.0 + np.arange(13) * DT
    truth = L.Truth(int(swapped), pairing, L.mapping_for(pairing), {}, ht, {"Player 1": np.c_[p1(ht)], "Player 2": np.c_[p2(ht)]})
    return pub, truth


def test_mapping_for():
    assert L.mapping_for(P.MATCHED) == {"Player 1": "Track X", "Player 2": "Track Y"}
    assert L.mapping_for(P.SWAPPED) == {"Player 1": "Track Y", "Player 2": "Track X"}


def test_public_carries_no_truth():
    pub, truth = _synthetic()
    names = set(vars(pub))
    assert not names & {"y", "pairing", "mapping", "who_in", "who_out", "hidden", "hidden_xy", "players"}
    js = json.dumps(pub.input_json())
    assert "pairing" not in js and "SWAPPED" not in js and "MATCHED" not in js
    for tr in pub.incoming:
        assert (tr.t < pub.g0).all()
    for tr in pub.outgoing:
        assert (tr.t > pub.g1).all()                    # nothing from the hidden interval


def test_board_before_reveal_does_not_leak():
    pub, truth = _synthetic()
    fig = LB.board(pub)
    js = json.dumps(fig)
    assert "= Player" not in js and "true hidden path" not in js and "TRUE path" not in js
    out_names = [d.get("name", "") for d in fig["data"] if "Track" in d.get("name", "")]
    assert out_names and all("identity unknown" in n for n in out_names)
    out_colors = {d["marker"]["color"] for d in fig["data"] if "Track" in d.get("name", "") and "marker" in d and isinstance(d["marker"].get("color"), str)}
    assert out_colors and not out_colors & set(LB.PLAYER_COLORS.values())
    hidden = np.concatenate(list(truth.hidden_xy.values()))
    xs = [x for d in fig["data"] + [t for f in fig["frames"] for t in f["data"]] for x in (d.get("x") or [])]
    assert not any(np.isclose(xs, h).any() for h in np.round(hidden[1:-1, 0], 3) if abs(h) > 0.05)


def test_board_after_reveal_shows_truth():
    pub, truth = _synthetic()
    js = json.dumps(LB.board(pub, truth))
    assert "Track X = Player 2" in js and "Track Y = Player 1" in js and "true hidden path" in js


def test_animation_timeline():
    pub, truth = _synthetic()
    t = LB.frame_times(pub)
    assert np.allclose(np.diff(t), 1 / L.FRAME_HZ) and t[0] == pytest.approx(0.0) and t[-1] >= pub.outgoing[0].t[-1] - 1 / L.FRAME_HZ
    assert [LB.phase(pub, v) for v in (2.9, 3.4, 3.9)] == ["before", "during", "after"]
    kf = LB.key_frames(pub)
    assert 0 <= kf["approach"] < kf["gap"] < kf["after"] <= kf["end"] == len(t) - 1
    assert t[kf["approach"]] < pub.g0 and pub.g0 <= t[kf["gap"]] <= pub.g1 and t[kf["after"]] > pub.g1
    res = dict(prediction=dict(predicted_pairing=P.SWAPPED, abstained=False, model="gru_c"), routed_mapping=L.mapping_for(P.SWAPPED))
    fig = LB.board(pub, truth, res)
    frames = fig["frames"]
    assert len(frames) == len(t) == len(fig["layout"]["sliders"][0]["steps"])
    assert len({len(f["data"]) for f in frames}) == 1 and len(frames[0]["data"]) == len(frames[0]["traces"])
    links = lambda f: [d for d in f["data"] if d.get("hovertext", "").startswith("GRU decision")]  # noqa: E731
    for k, f in enumerate(frames):                      # decision links appear only after the hidden interval
        assert all((len(d["x"]) > 0) == (t[k] > pub.g1) for d in links(f))
    for k, f in enumerate(frames):                      # outgoing tracks are empty before the hidden interval ends
        trk = [d for d in f["data"] if "Track" in d.get("name", "")]
        assert all((len(d["x"]) > 0) == (t[k] > pub.g1 + 1e-6) for d in trk)
    still = LB.board(pub, animate=False, start=kf["gap"])
    assert still["frames"] == [] and "DURING" not in json.dumps(still) and "SIMULATED OCCLUSION" in json.dumps(still)


def test_live_inference_api_with_injected_models():
    pub, _ = _synthetic()

    class Const:
        def __init__(self, p):
            self.p = p

        def predict(self, x):
            return np.full(len(x["gap_s"]), self.p)

    r = L.run_inference(pub, P.Predictor(0, models={"gru_c": Const(0.9), "hgb_c": Const(0.2)}))
    assert r["prediction"]["model"] == "gru_c" and r["prediction"]["decision"] == P.SWAPPED
    assert r["routed_mapping"] == L.mapping_for(P.SWAPPED) and r["models"]["hgb_c"]["pairing"] == P.MATCHED
    assert r["models"]["hgb_c"]["diagnostic_only"] and not r["models"]["gru_c"]["diagnostic_only"]
    empty = L.Public(**{**vars(pub), "outgoing": [pub.outgoing[0], A.Track(np.zeros(0), np.zeros(0), np.zeros(0))]})
    assert "error" in L.run_inference(empty, P.Predictor(0, models={"gru_c": Const(0.9), "hgb_c": Const(0.2)}))


# ---------------------------------------------------------------- development data

def _have():
    try:
        return (B.source_dir() / "neural" / "C" / "seed0" / "fold4_c160.pt").exists() and L.metrics_path().exists()
    except Exception:
        return False


data = pytest.mark.skipif(not _have(), reason="M11 development data / Milestone 2 results not present")


@pytest.fixture(scope="module")
def md():
    from football_fatigue.m11_benchmark import scenarios as SC
    return SC.MatchData("3812")


@pytest.fixture(scope="module")
def enc(md):
    return L.encounter_table(md)


@data
def test_protected_matches_refused():
    from football_fatigue.m11_benchmark import scenarios as SC
    for m in B.heldout_matches()[:2]:
        with pytest.raises(S.HeldOutRefused):
            SC.MatchData(m)
        with pytest.raises(S.HeldOutRefused):
            B.fold_for_match(m)
    assert not set(L.development_matches()) & set(B.heldout_matches()) and len(L.development_matches()) == 32


@data
def test_match_to_fold_and_validity(md, enc):
    pub, _ = L.make(md, enc["candidate_id"].iloc[0])
    assert pub.fold == B.fold_for_match("3812") == 4 and "3812" in B.fold_matches(pub.fold)
    pr = P.Predictor(pub.fold)
    v = L.validity(pub, pr)
    assert all(x["ok"] for x in v) and len(v) >= 8
    bad = L.Public(**{**vars(pub), "fold": 0})                 # a fold that trained on 3812 must fail the check
    assert not all(x["ok"] for x in L.validity(bad))


@data
def test_encounter_table_is_public(md, enc):
    assert len(enc) == 60 and enc["candidate_id"].is_unique
    assert not {"y", "p_gru", "p_hgb", "outcome", "decision"} & set(enc.columns)


@data
def test_regeneration_and_custom_seed(md, enc):
    cid = enc["candidate_id"].iloc[2]
    a, ta = L.make(md, cid, factor="noise", level="0.05")
    b, tb = L.make(md, cid, factor="noise", level="0.05")
    assert a.scenario_id == b.scenario_id and ta.pairing == tb.pairing
    assert all(np.array_equal(x.x, y.x) for x, y in zip(a.incoming + a.outgoing, b.incoming + b.outgoing))
    c1, _ = L.make(md, cid, custom=dict(noise=0.05), seed=1)
    c2, _ = L.make(md, cid, custom=dict(noise=0.05), seed=2)
    assert c1.scenario_id != c2.scenario_id and not np.array_equal(c1.incoming[0].x, c2.incoming[0].x)
    with pytest.raises(ValueError):
        L.make(md, cid, custom=dict(gap=9.0), seed=0)


@data
@pytest.mark.parametrize("factor, level", [("baseline", "base"), ("gap", "1.6"), ("noise", "0.05"), ("dropout", "random30"), ("gap", "2.4")])
def test_live_inference_matches_saved_milestone2(md, enc, factor, level):
    pr = P.Predictor(B.fold_for_match("3812"))
    for cid in enc["candidate_id"].iloc[:3]:
        pub, truth = L.make(md, cid, factor=factor, level=level)
        if not pub.valid:
            continue
        res = L.run_inference(pub, pr)
        cc = L.saved_crosscheck(pub, res)
        assert cc is not None
        for m in ("gru_c", "hgb_c"):
            assert cc[m]["abs_diff"] <= cc[m]["tolerance"]
        ev = L.evaluate(res, truth)
        assert ev["outcome"] == cc["outcome_saved"]          # same routed outcome as the saved run
        assert res["checkpoints"] == {m: B.recorded_sha256(m, pr.fold) for m in B.MODELS}


@data
def test_truth_never_reaches_inference(md, enc):
    pub, truth = L.make(md, enc["candidate_id"].iloc[1])
    pr = P.Predictor(pub.fold)
    r1 = L.run_inference(pub, pr)
    fake = L.Truth(1 - truth.y, P.SWAPPED if truth.pairing == P.MATCHED else P.MATCHED, {}, {}, truth.hidden_t, {})
    r2 = L.run_inference(pub, pr)                              # the truth object is not an input at all
    assert r1["prediction"]["probabilities"] == r2["prediction"]["probabilities"] and fake.y != truth.y
    hid = np.concatenate(list(truth.hidden_xy.values()))
    obs = np.concatenate([np.c_[t.x, t.y] for t in pub.incoming + pub.outgoing])
    inner = hid[1:-1] if len(hid) > 2 else hid
    assert not any(np.all(np.isclose(obs, h, atol=1e-9), axis=1).any() for h in inner)


@data
def test_raw_window_provenance(md, enc):
    pub, truth = L.make(md, enc["candidate_id"].iloc[0])
    rw = L.raw_window(md, pub, truth)
    assert set(rw["player"]) == {"Player 1", "Player 2"}
    h = rw[rw["hidden_from_model"]]
    assert len(h) and h["video_time_s"].between(pub.g0, pub.g1).all()
    assert rw["distance_between_players_m"].min() <= 0.55            # a real close encounter in the raw file
    p1 = rw[rw["player"] == "Player 1"].set_index("video_time_s")
    tr = pub.incoming[0]
    k = np.isin(np.round(tr.t, 3), p1.index)
    assert k.sum() > 10
    assert np.allclose(p1.loc[np.round(tr.t[k], 3), "x_m"].to_numpy(), np.round(tr.x[k], 3), atol=2e-3)   # incoming = raw file values


@data
def test_explanations(md, enc):
    pub, truth = L.make(md, enc["candidate_id"].iloc[0])
    ph = L.physics(pub)
    assert ph["pairing"] == (P.MATCHED if ph["matched_cost"] <= ph["swapped_cost"] else P.SWAPPED) and len(ph["cost"]) == 4
    mi = L.model_inputs(pub)
    assert mi["gru_in"].shape == (2, S.T, 4) and mi["gru_out"].shape == (2, S.T, 4) and len(mi["features"]) == 36
    assert len(L.training_history(pub.fold)) >= 1
    d = L.difficulty(pub)
    assert d and all(0 <= x["gru_accuracy"] <= 1 for x in d)


@data
def test_saved_milestone2_outputs_unchanged():
    from football_fatigue.m11_benchmark import scenario_eval as E
    out = E.out_dir()
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert B.sha256_file(out / "summary" / "metrics_v1.json") == man["metrics_sha256"]
    for m, info in man["per_match"].items():
        d = out / "matches" / m
        assert B.sha256_file(d / "scenarios.parquet") == info["scenarios_sha256"]
        assert B.sha256_file(d / "arrays.npz") == info["arrays_sha256"]


# ---------------------------------------------------------------- Streamlit UI flows

def _app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=600).run()


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(m.value for m in at.caption)


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


@data
def test_ui_demo_flow_truth_hidden_until_reveal():
    at = _app()
    _ok(at)
    assert at.radio(key="w_mode").value == "Demo"
    _btn(at, "🎲 Pick a random unseen-match scenario").click().run()
    _ok(at)
    assert at.session_state["demo_step"] == 2 and at.session_state["pub"].valid
    for _ in range(2):
        _btn(at, "Next ▶").click().run()
    assert at.session_state["result"] is None
    _btn(at, "▶ Run prediction").click().run()
    _ok(at)
    res = at.session_state["result"]
    assert "prediction" in res and set(res["prediction"]["probabilities"]) == {"gru_c", "hgb_c"}
    assert res["fold"] == B.fold_for_match(at.session_state["pub"].match_id)
    assert "Ground truth:" not in _text(at) and "Raw provider data" not in " ".join(e.label for e in at.expander)
    _btn(at, "👁 Reveal ground truth").click().run()
    _ok(at)
    assert "Ground truth:" in _text(at)
    assert any("Raw provider data" in e.label for e in at.expander) and any("Why could the models fail" in e.label for e in at.expander)


@data
def test_ui_lab_controls_regenerate_and_reset():
    at = _app()
    at.radio(key="w_mode").set_value("Research lab").run()
    _ok(at)
    at.selectbox(key="w_match").set_value("3830").run()
    assert at.session_state["pub"].match_id == "3830" and at.session_state["pub"].fold == B.fold_for_match("3830")
    _btn(at, "▶ Run prediction").click().run()
    assert at.session_state["result"] is not None
    at.selectbox(key="w_factor").set_value("gap").run()
    at.select_slider(key="w_level").set_value("2.4").run()
    _ok(at)
    pub = at.session_state["pub"]
    assert (pub.factor, pub.level) == ("gap", "2.4") and at.session_state["result"] is None and not at.session_state["revealed"]
    _btn(at, "▶ Run prediction").click().run()
    assert at.session_state["result"]["prediction"]["support"] == "unsupported"
    assert any("diagnostic only" in e.value for e in at.error)
    at.radio(key="lab_play").set_value("Step by step").run()
    _btn(at, "In gap").click().run()
    t = LB.frame_times(pub)
    assert pub.g0 <= t[at.session_state["lab_frame"]] <= pub.g1
    _btn(at, "+1 frame ▶").click().run()
    _ok(at)
    at.radio(key="w_vmode").set_value("custom").run()
    at.select_slider(key="w_gap").set_value(1.2).run()
    _ok(at)
    assert at.session_state["pub"].params["gap"] == 1.2 and at.session_state["pub"].params["shift"] == 0.0


@data
def test_ui_gallery_models_tab_and_json():
    at = _app()
    next(s for s in at.selectbox if s.label == "Category").set_value("gru_wrong_primary").run()
    _btn(at, "Load #1").click().run()
    _ok(at)
    row = L.gallery_fixed("gru_wrong_primary")[0]
    assert at.session_state["pub"].scenario_id == row["scenario_id"]
    _btn(at, "▶ Run prediction").click().run()
    _ok(at)
    assert at.session_state["result"]["prediction"]["decision"] == row["decision"]      # the saved example reproduces live
    assert any("GRU C input tensors" in m.value for m in at.markdown)                     # models tab rendered with live inputs
    next(r for r in at.radio if r.label == "Input").set_value("Use the current scenario (model-visible part)").run()
    _btn(at, "▶ Run prediction on this input").click().run()
    _ok(at)
    assert at.session_state["json_result"][1]["fold"] == B.fold_for_match(row["match_id"])
