"""M7.1 display-continuity layer: data-contract tests on small SYNTHETIC overlays (not a real source schema)."""

import numpy as np

from football_fatigue.demo.continuity import (
    ContinuityConfig,
    build_continuity,
    eligibility,
    flicker_metrics,
    hysteresis,
    prediction_eval,
    trust_level,
)

KEYS = ("local", "gid", "trk", "x1", "y1", "x2", "y2", "st", "conf", "conf_local", "link", "as", "team", "tconf", "cls", "seg", "est",
        "pxm", "pym", "pv", "kpd", "role")


def synth(tracks, n_frames):
    """tracks: list of dicts {local, trk, frames, x(f)->metres, st(f), team, px(f)} -> v3-like overlay (synthetic)."""
    rows = []
    for t in tracks:
        for f in t["frames"]:
            xm = t["x"](f)
            px = 100 + 20 * xm
            rows.append((f, {"local": t["local"], "gid": t.get("gid", t["local"]), "trk": t["trk"](f) if callable(t["trk"]) else t["trk"],
                             "x1": px - 10, "y1": 400.0, "x2": px + 10, "y2": 460.0, "st": t.get("st", lambda f: 0)(f), "conf": 0.9, "conf_local": 0.9,
                             "link": None, "as": 0, "team": t.get("team", 0), "tconf": 0.9, "cls": "player", "seg": 0, "est": 0,
                             "pxm": xm, "pym": 30.0, "pv": 0, "kpd": 3.0, "role": 0}))
    rows.sort(key=lambda r: r[0])
    cols = {k: [r[1][k] for r in rows] for k in KEYS}
    cnt = np.bincount([r[0] for r in rows], minlength=n_frames)
    return {"n_frames": n_frames, "cols": cols, "frame_offsets": [0] + np.cumsum(cnt).tolist(), "summary": {}}


def obs_rows(ov):
    c, off = ov["cols"], ov["frame_offsets"]
    return [[tuple(c[k][i] for k in KEYS) for i in range(off[f], off[f + 1]) if not c["est"][i]] for f in range(ov["n_frames"])]


def test_hysteresis_downgrades_immediately_and_upgrades_after_dwell():
    assert hysteresis([1, 1, 0, 0, 0, 1, 0, 0], 3) == [1, 1, 1, 1, 0, 1, 1, 1]
    assert hysteresis([0, 2, 0, 0, 0], 2) == [0, 2, 2, 0, 0]
    seq = list(np.random.default_rng(1).integers(0, 3, 500))
    assert all(d >= r for d, r in zip(hysteresis(seq, 12), seq))          # never more trusted than the data
    assert trust_level(0, 1) == 1 and trust_level(1, 0) == 1 and trust_level(3, 0) == 2 and trust_level(0, 0) == 0


def test_short_gap_bridged_inside_one_tracklet_and_observed_rows_unchanged():
    ov = synth([{"local": 1, "trk": 10, "frames": [f for f in range(40) if not 10 <= f < 16], "x": lambda f: 20 + 0.2 * f}], 40)
    v4 = build_continuity(ov, ContinuityConfig(tail_frames=0))
    c = v4["cols"]
    assert obs_rows(ov) == obs_rows(v4)
    est = [i for i in range(len(c["local"])) if c["src"][i] == 1]
    assert len(est) == 6 and v4["continuity"]["stats"]["bridged"] == 1
    for i in est:
        assert c["est"][i] == 1 and c["elig"][i] == 1 and c["conf"][i] is None and c["seg"][i] == -1 and 0.5 <= c["dconf"][i] < 1
    f = np.repeat(np.arange(40), np.diff(v4["frame_offsets"]))
    mid = [i for i in est if f[i] == 12][0]
    assert abs(c["pxm"][mid] - (20 + 0.2 * 12)) < 1e-6                     # linear interpolation between the observed ends


def test_bridge_refused_across_tracklets_long_gaps_and_implausible_jumps():
    trk = lambda f: 10 if f < 10 else 11
    a = synth([{"local": 1, "trk": trk, "frames": [f for f in range(30) if not 10 <= f < 13], "x": lambda f: 20.0}], 30)
    assert build_continuity(a, ContinuityConfig(tail_frames=0))["continuity"]["stats"]["refused_tracklet_change"] == 1
    b = synth([{"local": 1, "trk": 1, "frames": [f for f in range(80) if not 10 <= f < 50], "x": lambda f: 20.0}], 80)
    assert build_continuity(b, ContinuityConfig(tail_frames=0))["continuity"]["stats"]["refused_too_long"] == 1
    jump = synth([{"local": 1, "trk": 1, "frames": [f for f in range(30) if f != 10], "x": lambda f: 20.0 if f < 10 else 40.0}], 30)
    st = build_continuity(jump, ContinuityConfig(tail_frames=0))["continuity"]["stats"]
    assert st["refused_implausible"] == 1 and st["bridged"] == 0           # 20 m in 0.08 s is not one person walking


def test_long_gap_in_a_crowd_is_not_bridged():
    p1 = {"local": 1, "trk": 1, "frames": [f for f in range(60) if not 10 <= f < 30], "x": lambda f: 20.0}
    p2 = {"local": 2, "trk": 2, "frames": list(range(60)), "x": lambda f: 20.5}   # someone else right next to the gap end
    st = build_continuity(synth([p1, p2], 60), ContinuityConfig(tail_frames=0))["continuity"]["stats"]
    assert st["refused_crowded"] == 1 and st["bridged"] == 0


def test_lost_tail_is_short_fading_and_never_eligible():
    ov = synth([{"local": 1, "trk": 1, "frames": list(range(20)), "x": lambda f: 30.0}], 60)
    v4 = build_continuity(ov, ContinuityConfig(tail_frames=6))
    c = v4["cols"]
    tail = [i for i in range(len(c["local"])) if c["src"][i] == 2]
    assert len(tail) == 6 and all(c["elig"][i] == 1 for i in tail)
    d = [c["dconf"][i] for i in tail]
    assert d == sorted(d, reverse=True) and d[-1] < d[0] <= 0.6


def test_display_trust_and_eligibility_contract():
    ov = synth([{"local": 1, "trk": 1, "frames": list(range(50)), "x": lambda f: 30.0, "st": lambda f: 0 if f % 4 else 2}], 50)
    c = build_continuity(ov)["cols"]
    assert all(d >= r for d, r in zip(c["dlev"], c["lev"]))
    assert sum(1 for i in range(1, 50) if c["dlev"][i] != c["dlev"][i - 1]) < sum(1 for i in range(1, 50) if c["lev"][i] != c["lev"][i - 1])
    row = {k: [v] for k, v in dict(est=0, src=0, pv=0, role=0, team=0, tconf=0.9).items()}
    assert eligibility(row, 0) == 0
    for k, v, code in (("pv", 1, 2), ("role", 2, 3), ("team", 2, 4), ("tconf", 0.5, 4), ("src", 1, 1)):
        bad = {kk: list(vv) for kk, vv in row.items()}
        bad[k] = [v]
        assert eligibility(bad, 0) == code


def test_prediction_eval_and_flicker_metrics_on_known_motion():
    ov = synth([{"local": 1, "trk": 1, "frames": list(range(200)), "x": lambda f: 10 + 0.1 * f}], 200)
    r = prediction_eval(ov["cols"], ov["frame_offsets"], gaps=(4,))
    assert r[4]["interp"]["m"]["p90"] < 1e-6 and r[4]["const_vel"]["m"]["p90"] < 1e-6 and r[4]["hold"]["m"]["median"] > 0.1
    gappy = synth([{"local": 1, "trk": 1, "frames": [f for f in range(100) if f % 10], "x": lambda f: 10.0}], 100)
    assert flicker_metrics(gappy["cols"], gappy["frame_offsets"])["marker_flicker"] == 9
    v4 = build_continuity(gappy, ContinuityConfig(tail_frames=0))
    assert flicker_metrics(v4["cols"], v4["frame_offsets"])["marker_flicker"] == 0
