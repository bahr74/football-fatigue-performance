"""Encounter explorer (``m11_dashboard.encounters`` + ``encounters_app.py``) on the real ``m11-sync-proximity-v2`` development
events with their event-centred ``m11-sync-proximity-v3`` contexts: matches, event / trajectory agreement, the fixed nominal time
axis, unsupported times, incomplete contexts, filters, non-primary thresholds, figures, review persistence (datasets untouched)
and headless app runs. Skipped when the outputs are absent."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.intersections.proximity import ProximityConfig, verify_existing
from football_fatigue.m11_dashboard import encounters as E

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "data" / "raw")
BASE = REPO / "data" / "processed" / "m11-sync-proximity-v2"
V3 = REPO / "data" / "processed" / "m11-sync-proximity-v3"
pytestmark = pytest.mark.skipif(not (BASE / "development-index" / "summary.json").exists()
                                or not (V3 / "development-index" / "summary.json").exists(), reason="v2 / v3 outputs not written")
MID = "3812"
MAX_STEP_S = 0.5 + 1 / 29.97 + 1e-6


@pytest.fixture(scope="module")
def data():
    ev = {t: E.load_events(ROOT, MID, t) for t in E.THRESHOLDS}
    return dict(ev=ev, npz=E.load_npz(ROOT, MID), hq=E.high_quality_ids(ROOT))


def ctx_of(data, thr, rec):
    return E.context_for(ROOT, MID, thr, rec, npz=data["npz"], events=data["ev"])


def pick(data, f):
    return next(r for r in data["ev"]["0.25m"] if f(r))


def test_development_matches_only():
    split = json.loads((REPO / "data" / "processed" / "m11-encounters-v1" / "splits.json").read_text(encoding="utf-8"))
    ms = E.development_matches(ROOT)
    assert len(ms) == 40 and set(ms) <= set(split["splits"]["development"])
    with pytest.raises(E.ExplorerError):
        E.load_events(ROOT, split["splits"]["validation"][0], "0.25m")


def check_context(ctx, rec):
    """Event metadata and trajectory export agree at t*; every synchronized instant is both players at that same t."""
    a, b, s = ctx.a, ctx.b, ctx.sync
    for side, key in ((a, "player_a"), (b, "player_b")):
        assert np.allclose(E.position(side, rec["t_star_s"]), rec[key]["position_at_t_star"], atol=1e-6)
        assert np.all(np.diff(side["t"]) > 0)
        dt, held = np.diff(side["t"]), (np.diff(side["x"]) == 0) & (np.diff(side["y"]) == 0)
        assert not np.any((dt > MAX_STEP_S) & ~held)                      # one valid run: never joined across a gap
    for t, xa, ya, xb, yb, d in zip(s["t"], s["xa"], s["ya"], s["xb"], s["yb"], s["d"]):
        assert np.allclose(E.position(a, t), (xa, ya), atol=1e-9) and np.allclose(E.position(b, t), (xb, yb), atol=1e-9)
        assert d == pytest.approx(np.hypot(xa - xb, ya - yb), abs=1e-12)


def test_primary_event_matches_jsonl_and_npz(data):
    for k in (0, 100, 500):
        rec = data["ev"]["0.25m"][k]
        ctx = ctx_of(data, "0.25m", rec)
        assert ctx.kind == "exported" and ctx.source_event_id == rec["event_id"]
        check_context(ctx, rec)
        assert E.state_at(ctx, rec["t_star_s"])["d"] == pytest.approx(rec["d_min_m"], abs=1e-6)
        assert rec["context"]["nominal_start_s"] == rec["t_star_s"] - 2.5 and rec["context_v2"] is not None
        seg = rec["player_a"]["segment_at_t_star"]
        assert seg["row0"] in ctx.a["row"] and seg["row1"] in ctx.a["row"]


@pytest.mark.parametrize("name,f", [
    ("observed/observed high quality", lambda r, hq: (str(r["match_id"]), r["event_id"]) in hq),
    ("estimated/estimated", lambda r, hq: r["pair_quality"] == "estimated/estimated" and not r["coincident_interval"]),
    ("coincident provider interval", lambda r, hq: r["coincident_interval"]),
    ("incomplete context", lambda r, hq: not r["context"]["complete"] and r["context"]["player_a"]["cut_before"]),
])
def test_categories_and_figures(data, name, f):
    rec = pick(data, lambda r: f(r, data["hq"]))
    ctx = ctx_of(data, "0.25m", rec)
    check_context(ctx, rec)
    if name == "incomplete context":                                     # the drawn run starts after the window start: nothing joined
        assert ctx.a["t"][0] > rec["context"]["window_start_s"] + 1e-9
    if name == "coincident provider interval":
        assert rec["d_min_m"] == 0
    fig = E.pitch_figure(ctx, rec, rec["t_star_s"], r_m=0.25)
    marks = {t.name: (t.x[0], t.y[0]) for t in fig.data if t.name in ("A(t*)", "B(t*)")}
    assert np.allclose(marks["A(t*)"], rec["player_a"]["position_at_t_star"]) and np.allclose(marks["B(t*)"], rec["player_b"]["position_at_t_star"])
    star = [t for t in fig.data if t.name.startswith("P*")][0]
    assert np.allclose((star.x[0], star.y[0]), rec["midpoint"])
    assert ("midpoint" in star.name) == (rec["d_min_m"] > 0)
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.yaxis.scaleratio == 1
    E.distance_figure(ctx, rec, rec["t_star_s"], r_m=0.25)


def test_non_primary_thresholds_never_load_a_wrong_trajectory(data):
    for thr in ("0m", "0.1m"):
        rec = data["ev"][thr][0]
        ctx = ctx_of(data, thr, rec)
        assert ctx.kind == "related" and np.allclose(E.position(ctx.a, rec["t_star_s"]), rec["player_a"]["position_at_t_star"], atol=1e-6)
    far = next(r for r in data["ev"]["0.5m"] if r["d_min_m"] > 0.25)
    assert ctx_of(data, "0.5m", far).kind == "none"
    near = next(r for r in data["ev"]["0.5m"] if r["d_min_m"] <= 0.25)
    assert ctx_of(data, "0.5m", near).kind == "related"
    forged = dict(data["ev"]["0.1m"][0])
    forged["player_a"] = dict(forged["player_a"], position_at_t_star=[forged["player_a"]["position_at_t_star"][0] + 0.5,
                                                                      forged["player_a"]["position_at_t_star"][1]])
    assert ctx_of(data, "0.1m", forged).kind == "none"                   # positions do not reproduce: refused, not shown


def test_filters(data):
    table = E.rows(data["ev"]["0.25m"], data["hq"], {}, "0.25m")
    hq_rows = E.filter_rows(table, high_quality_only=True)
    assert {r["event_id"] for r in hq_rows} == {e for m, e in data["hq"] if m == MID}
    assert all(r["quality"] == "observed/observed" and not r["coincident"] for r in hq_rows)
    est = E.filter_rows(table, qualities={"estimated/estimated"}, coincident="exclude", relation="opposite")
    assert est and all(r["quality"] == "estimated/estimated" and not r["coincident"] and r["team_relation"] == "opposite" for r in est)
    assert len(E.filter_rows(table, coincident="only")) == sum(r["coincident_interval"] for r in data["ev"]["0.25m"])


def _hashes():
    files = sorted((BASE / MID).glob("root.*")) + sorted((BASE / "development-index").glob("*")) + sorted((V3 / MID).glob("root.*"))
    return {p.name + str(p.parent.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_reviews_persist_and_dataset_untouched(tmp_path, data):
    before = _hashes()
    p = tmp_path / "reviews.csv"
    eid = data["ev"]["0.25m"][0]["event_id"]
    E.save_review(p, MID, "0.25m", eid, "uncertain", "first look")
    E.save_review(p, MID, "0.25m", eid, "accept", "  crossing,\n clear ")
    E.save_review(p, MID, "0.1m", data["ev"]["0.1m"][0]["event_id"], "reject")
    r = E.load_reviews(p)
    assert len(r) == 2 and r[(MID, "0.25m", eid)]["status"] == "accept" and r[(MID, "0.25m", eid)]["note"] == "crossing, clear"
    assert E.reviews_csv(r).decode().splitlines()[0] == ",".join(E.REVIEW_FIELDS)
    with pytest.raises(E.ExplorerError):
        E.save_review(p, MID, "0.25m", eid, "swap")
    assert _hashes() == before and verify_existing(BASE / MID, ProximityConfig()) is None


def test_navigation_and_phase_helpers():
    ids = ["a", "b", "c"]
    assert E.neighbour(ids, "b", 1) == "c" and E.neighbour(ids, "c", 1) == "c" and E.neighbour(ids, "a", -1) == "a"
    assert E.neighbour(ids, "zz", 1) == "a" and E.neighbour([], "a", 1) is None
    assert [E.phase_offset(p) for p in ("before", "at", "after")] == [-1.0, 0.0, 1.0]      # the same for every event
    assert E.NOMINAL_OFFSETS[0] == -2.5 and E.NOMINAL_OFFSETS[-1] == 2.5 and 0.0 in E.NOMINAL_OFFSETS and len(E.NOMINAL_OFFSETS) == 51
    assert all(o in E.NOMINAL_OFFSETS for o in (-1.0, 1.0))


def test_direction_markers_follow_travel():
    east = E._arrows(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    north = E._arrows(np.array([[0.0, 0.0], [0.0, 3.0]]), k=1)
    assert len(east) == 2 and all(abs(a - 90) < 1e-9 for _, _, a in east) and north[0][2] == pytest.approx(0) and north[0][1] == 1.5
    assert E._arrows(np.array([[0.0, 0.0], [0.05, 0.0]])) == []          # standing still: no direction shown


def test_terminology_and_notes(data):
    hq_rec = pick(data, lambda r: (str(r["match_id"]), r["event_id"]) in data["hq"] and r["d_min_m"] > 0)
    assert E.encounter_kind(hq_rec) == "Closest synchronized approach"
    text = " ".join(t for _, t in E.notes(hq_rec, ctx_of(data, "0.25m", hq_rec)))
    assert "ambiguity" not in text and "estimated" not in text and "incomplete" not in text
    cloned = pick(data, lambda r: r["coincident_interval"])
    assert E.encounter_kind(cloned).startswith("Potential tracking ambiguity")
    assert any(lv == "warning" and "Potential tracking ambiguity" in t for lv, t in E.notes(cloned, ctx_of(data, "0.25m", cloned)))
    est = pick(data, lambda r: r["pair_quality"] == "estimated/estimated" and not r["coincident_interval"])
    assert any("are estimated by the provider" in t for _, t in E.notes(est, ctx_of(data, "0.25m", est)))
    cut = pick(data, lambda r: not r["context"]["complete"] and r["context"]["player_a"]["cut_before"] == "implausible_jump")
    assert "A's track starts late (a tracking jump" in E.context_summary(cut["context"])


def test_pitch_zoom_equal_scale_and_off_pitch_unclipped(data):
    for rec in (pick(data, lambda r: (str(r["match_id"]), r["event_id"]) in data["hq"]), pick(data, lambda r: r["coincident_interval"])):
        ctx = ctx_of(data, "0.25m", rec)
        for full in (False, True):
            fig = E.pitch_figure(ctx, rec, rec["t_star_s"], r_m=0.25, full_pitch=full)
            ax, ay = fig.layout.xaxis, fig.layout.yaxis
            assert ay.scaleanchor == "x" and ay.scaleratio == 1 and ax.constrain in (None, "range")   # one metre = one metre
            xs = np.r_[ctx.a["x"], ctx.b["x"]]
            ys = np.r_[ctx.a["y"], ctx.b["y"]]
            assert ax.range[0] <= xs.min() and ax.range[1] >= xs.max() and ay.range[0] <= ys.min() and ay.range[1] >= ys.max()
            if not full:
                assert ax.range[1] - ax.range[0] >= 8 - 1e-9 and ax.range[1] - ax.range[0] < 110
    off = pick(data, lambda r: E.out_of_pitch(ctx_of(data, "0.25m", r)) > 0)
    ctx = ctx_of(data, "0.25m", off)
    fig = E.pitch_figure(ctx, off, off["t_star_s"], r_m=0.25, full_pitch=True)
    tr = next(t for t in fig.data if t.name == "outside the pitch lines")
    assert len(tr.x) == E.out_of_pitch(ctx) and max(np.abs(tr.x).max() - 52.5, np.abs(tr.y).max() - 34) > 0
    assert fig.layout.xaxis.range[1] >= np.r_[ctx.a["x"], ctx.b["x"]].max()       # range widened, data not clipped
    assert any("outside the 105 × 68 m pitch" in t for _, t in E.notes(off, ctx))


def test_app_runs_headless(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("M11_REVIEWS_PATH", str(tmp_path / "reviews.csv"))
    monkeypatch.setenv("M11_DASHBOARD_ROOT", ROOT)
    at = AppTest.from_file(str(REPO / "src" / "football_fatigue" / "m11_dashboard" / "encounters_app.py"), default_timeout=120)
    at.run()
    assert not at.exception and at.title[0].value == "Football Trajectory Explorer"
    s = json.loads((BASE / "development-index" / "summary.json").read_text(encoding="utf-8"))
    vals = {m.label: m.value for m in at.metric}
    assert vals["Development matches"] == str(s["matches"]) and vals["Encounters (R = 0.25 m)"] == f"{s['episodes']:,}"
    at.button(key="save_review").click().run()                            # save the default review of the first high-quality event
    assert not at.exception and len(E.load_reviews(tmp_path / "reviews.csv")) == 1
    at.radio(key="thr").set_value("0.5m").run()
    assert not at.exception


def test_app_navigation_state_and_presentation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("M11_REVIEWS_PATH", str(tmp_path / "reviews.csv"))
    monkeypatch.setenv("M11_DASHBOARD_ROOT", ROOT)
    at = AppTest.from_file(str(REPO / "src" / "football_fatigue" / "m11_dashboard" / "encounters_app.py"), default_timeout=120)
    at.run()
    mid = at.selectbox(key="mid").value
    ev = E.load_events(ROOT, mid, "0.25m")
    val = E.load_validation(ROOT)
    hq_ids = [r["event_id"] for r in ev if val[(mid, r["event_id"])]["presentation_ready"]]   # default list: audited, ready
    assert at.selectbox(key="event").value == hq_ids[0] and at.button(key="prev").disabled
    at.button(key="next").click().run()
    assert at.selectbox(key="event").value == hq_ids[1]
    at.button(key="prev").click().run()
    at.button(key="next").click().run()
    eid = hq_ids[1]
    rec = next(r for r in ev if r["event_id"] == eid)
    key = f"t-{mid}-0.25m-{eid}"
    assert at.select_slider(key=key).value == 0.0                        # opens at t*
    at.button(key=f"after-{key}").click().run()
    assert at.select_slider(key=key).value == 1.0 and not at.exception   # t* + 1 s: inside the complete context of a ready event
    state = next(m.value for m in at.markdown if m.value.startswith("**P") and "t* +1.0 s" in m.value)
    assert "unavailable" not in state and "synchronized distance **" in state
    at.selectbox(key="rel").set_value("opposite").run()                  # advanced filter, then hide it in presentation mode
    at.toggle(key="present").set_value(True).run()
    assert not at.exception and not [r for r in at.radio if r.key == "thr"]
    labels = {m.label for m in at.metric}
    assert {"Event time", "Min. synchronized distance", "Quality", "Context t* ± 2.5 s"} <= labels
    number = {r["event_id"]: k + 1 for k, r in enumerate(ev)}
    assert [o.split(" ·")[0] for o in at.selectbox(key="event").options] == [f"#{number[i]}" for i in hq_ids]   # ready list, in order
    at.button(key="next").click().run()
    at.toggle(key="present").set_value(False).run()
    assert not at.exception and at.selectbox(key="rel").value == "opposite"   # hidden filter state survived the reruns
    assert at.selectbox(key="event").value in hq_ids
    at.selectbox(key="quality").set_value("All encounters (incl. incomplete / problematic)").run()
    at.selectbox(key="rel").set_value("all").run()
    at.selectbox(key="coin").set_value("only").run()
    cloned = at.selectbox(key="event").value
    assert next(r for r in ev if r["event_id"] == cloned)["coincident_interval"]
    assert any("Potential tracking ambiguity" in w.value for w in at.warning)


CASES = dict(A="7d69f374b095e733", C="548ad35e88e51867", D="22092c0d35089442")   # 3812: the reported screenshots


def test_screenshot_cases_render_honestly(data):
    ev = {r["event_id"]: r for r in data["ev"]["0.25m"]}
    a = ev[CASES["A"]]                                                   # outgoing tracks identical: drawn layered, marked, explained
    ctx = ctx_of(data, "0.25m", a)
    fig = E.pitch_figure(ctx, a, a["t_star_s"], r_m=0.25)
    same = next(t for t in fig.data if t.name.startswith("A and B identical"))
    assert "2.20 s" in same.name and len(same.x) > 0                     # t* ... end of period 1 (v2 window: 2.00 s)
    widths = {t.name.split(" ")[0]: t.line.width for t in fig.data if t.name.endswith("outgoing")}
    assert widths["A"] > widths["B"]                                     # A stays visible under B where they coincide
    assert any(t.legendgroup == "cut" and t.showlegend is not False for t in fig.data)
    assert E.encounter_kind(a, ctx).startswith("Potential tracking ambiguity")
    txt = " ".join(t for _, t in E.notes(a, ctx))
    assert "Player B's incoming track covers 0.47 s of the 2.5 s window" in txt and "identical coordinates for 2.20 s" in txt
    dist = E.distance_figure(ctx, a, a["t_star_s"], r_m=0.25)
    assert dist.layout.xaxis.range[0] <= -2.5 and any(t.name == "no shared valid track (not drawn)" for t in dist.data)
    shade = [s for s in dist.layout.shapes if s.fillcolor == "#9e9e9e"]
    assert shade and shade[0].x0 == pytest.approx(-2.5) and shade[0].x1 == pytest.approx(-0.467134, abs=1e-6)
    assert shade[-1].x1 == pytest.approx(2.5)                            # P1 ends 2.24 s after t*: the rest of the window is grey
    assert a["context"]["window_clipped_by_period"]
    c = ev[CASES["C"]]                                                   # no incoming track at all: said so, nothing drawn before t*
    ctx = ctx_of(data, "0.25m", c)
    assert ctx.a["t"][0] == ctx.b["t"][0] == c["t_star_s"]
    before = E.state_at(ctx, c["t_star_s"] - 1.0)                        # an unsupported time: reported, never invented
    assert before["a"] is None and before["b"] is None and before["d"] is None and before["why_a"] == "implausible_jump"
    txt = " ".join(t for _, t in E.notes(c, ctx))
    assert "No incoming track for Player A" in txt and "No incoming track for Player B" in txt
    d = ev[CASES["D"]]                                                   # exact synchronized coincidence at one instant, not a clone
    ctx = ctx_of(data, "0.25m", d)
    assert E.encounter_kind(d, ctx) == "Exact synchronized coincidence" and E.identical_seconds(ctx) == 0
    star = next(t for t in E.pitch_figure(ctx, d, d["t_star_s"], r_m=0.25).data if t.name.startswith("P*"))
    assert "exact synchronized coincidence" in star.name and "midpoint" not in star.name


def test_layering_and_zoom_keep_paths_visible(data):
    rec = pick(data, lambda r: (str(r["match_id"]), r["event_id"]) in data["hq"])
    fig = E.pitch_figure(E.context_for(ROOT, MID, "0.25m", rec, npz=data["npz"], events=data["ev"]), rec, rec["t_star_s"], r_m=0.25)
    names = [t.name for t in fig.data]
    assert names.index("A(t*)") > names.index(next(n for n in names if n.endswith("incoming")))   # markers above the paths
    assert fig.layout.legend.orientation == "h" and fig.layout.legend.y < 0                       # legend below, not on the pitch
    now = next(t for t in fig.data if t.name.startswith("selected time"))
    assert now.marker.symbol == "circle-open"                                                     # hollow: never hides a path


def test_app_case_slider_bounds_and_presentation_list(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("M11_REVIEWS_PATH", str(tmp_path / "reviews.csv"))
    monkeypatch.setenv("M11_DASHBOARD_ROOT", ROOT)
    at = AppTest.from_file(str(REPO / "src" / "football_fatigue" / "m11_dashboard" / "encounters_app.py"), default_timeout=120)
    at.run()
    at.selectbox(key="mid").set_value(MID).run()
    at.selectbox(key="quality").set_value("All encounters (incl. incomplete / problematic)").run()
    ev = {r["event_id"]: r for r in E.load_events(ROOT, MID, "0.25m")}
    at.selectbox(key="event").set_value(CASES["A"]).run()
    key = f"t-{MID}-0.25m-{CASES['A']}"
    opts = at.select_slider(key=key).options
    assert not at.exception and opts[0].startswith("t* -2.5 s") and opts[-1].startswith("t* +2.5 s")   # the fixed nominal axis
    assert any("Potential tracking ambiguity" in w.value for w in at.warning)
    at.selectbox(key="event").set_value(CASES["C"]).run()
    key = f"t-{MID}-0.25m-{CASES['C']}"
    assert not at.button(key=f"before-{key}").disabled                   # same controls for every event
    at.button(key=f"before-{key}").click().run()
    assert at.select_slider(key=key).value == -1.0 and not at.exception
    txt = " ".join(m.value for m in at.markdown)
    assert "A **unavailable** (a tracking jump" in txt and "no synchronized distance" in txt
    at.toggle(key="present").set_value(True).run()
    val = E.load_validation(ROOT)
    ready = [e for e in ev if val[(MID, e)]["presentation_ready"]]
    assert len(at.selectbox(key="event").options) == len(ready) and at.selectbox(key="event").value in ready
    assert at.selectbox(key="quality").disabled


def test_fixed_nominal_axis_for_every_event(data):
    clipped = next(r for r in data["ev"]["0.25m"] if r["context"]["window_clipped_by_period"])
    for rec in (data["ev"]["0.25m"][0], data["ev"]["0.25m"][300], clipped):
        ctx = ctx_of(data, "0.25m", rec)
        for fig in (E.distance_figure(ctx, rec, rec["t_star_s"], r_m=0.25), E.coverage_figure(ctx, rec, rec["t_star_s"])):
            assert tuple(fig.layout.xaxis.range) == tuple(E.AXIS)       # -2.55 ... +2.55 s whatever the coverage
        cx = rec["context"]
        assert cx["nominal_end_s"] - cx["nominal_start_s"] == pytest.approx(5.0) and cx["nominal_start_s"] == rec["t_star_s"] - 2.5


def test_plotly_paths_are_the_valid_runs_only(data):
    for rec in (pick(data, lambda r: not r["context"]["complete"] and r["context"]["player_b"]["cut_before"]),
                pick(data, lambda r: r["context"]["complete"] and r["context"]["player_a"]["n_window_edge_points"] == 2)):
        ctx = ctx_of(data, "0.25m", rec)
        fig = E.pitch_figure(ctx, rec, rec["t_star_s"], r_m=0.25)
        for key, side in (("A", ctx.a), ("B", ctx.b)):
            path = np.c_[side["x"], side["y"]]
            pos = np.array(E.position(side, rec["t_star_s"]))
            for t in fig.data:
                if t.name.startswith(key + " ·") and t.name.endswith(("incoming", "outgoing")):
                    pts = np.c_[t.x, t.y]                                # every drawn vertex is a run vertex or the position at t*
                    assert all(np.any(np.all(np.isclose(path, p, atol=1e-12), axis=1)) or np.allclose(p, pos) for p in pts)
                if t.name.startswith(key + " ·") and t.name.endswith("samples"):
                    assert len(t.x) == int((~side["edge"]).sum())       # window-edge interpolations are not drawn as samples
        cov = E.coverage_figure(ctx, rec, rec["t_star_s"])
        a_bar = next(t for t in cov.data if t.name == "A valid")
        assert a_bar.x[0] == pytest.approx(ctx.a["t"][0] - rec["t_star_s"]) and a_bar.x[1] == pytest.approx(ctx.a["t"][-1] - rec["t_star_s"])
