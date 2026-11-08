"""Tests for the M1 answer-key labeller. All events and frames are SYNTHETIC (made up here), not source data."""
import ast
from pathlib import Path

import numpy as np

from m1_labels import CONTROL, DEAD, LOOSE, TRAVELLING, UNKNOWN, Row, label_match

FPS = 30.0
SRC = Path(__file__).resolve().parents[1] / "src"


def frames(seconds=20.0):
    n = int(seconds * FPS)
    return np.arange(n) / FPS, np.ones(n, int)


def f(ts):
    return int(round(ts * FPS))


def row(idx, ge, get, pet, st, et, en, team=1, player=10, **kw):
    return Row(idx=idx, ge=ge, pe=(idx + 100 if pet else None), get=get, pet=pet, period=1, st=st, et=et, en=en,
               team=team, player=player, **kw)


def run(rows, seconds=20.0):
    t, per = frames(seconds)
    spans = {}
    for r in rows:
        if r.get in ("OTB", "FIRSTKICKOFF") and r.st is not None:
            a, b = spans.get(r.ge, (f(r.st), f(r.en)))
            spans[r.ge] = (min(a, f(r.st)), max(b, f(r.en)))
    row_frames = {r.idx: {"start": f(r.st) if r.st is not None else None, "event": f(r.et) if r.et is not None else None,
                          "end": f(r.en) if r.en is not None else None} for r in rows}
    return t, label_match(t, per, rows, spans, row_frames)


def test_kickoff_pass_control_out_restart():
    rows = [
        row(0, 1, "FIRSTKICKOFF", "PA", 1.0, 1.0, 1.0, setpiece="K"),
        row(1, 2, "OTB", "IT", 2.0, 2.0, 3.0, player=11, setpiece="O"),
        row(2, 2, "OTB", "PA", 2.0, 3.0, 3.0, player=11, setpiece="O"),
        row(3, 3, "OUT", None, 4.0, 4.0, None, out_type="T"),
        row(4, 4, "OTB", "PA", 8.0, 9.0, 9.0, team=2, player=20, setpiece="T"),
        row(5, 5, "OTB", "IT", 10.0, 10.0, 11.0, team=2, player=21, setpiece="O"),
    ]
    t, L = run(rows)
    s = L.state
    assert (s[:f(1.0)] == DEAD).all()                                  # before kick-off
    assert (s[f(1.5)] == TRAVELLING) and L.team[f(1.5)] == 1           # kick-off pass in flight
    assert s[f(2.5)] == CONTROL and L.player[f(2.5)] == 11
    assert s[f(3.5)] == TRAVELLING                                     # pass after the spell
    assert (s[f(4.0):f(9.0)] == DEAD).all()                            # OUT -> throw-in taken (restart_setup until release)
    assert L.rule_names[L.rule[f(8.5)]] == "restart_setup_T"
    assert s[f(9.0)] == CONTROL and s[f(9.5)] == TRAVELLING and s[f(10.5)] == CONTROL
    assert L.log["restart_setup_s"][-1][0] == "T"


def test_rebound_duel_and_null_player_are_loose():
    rows = [
        row(0, 1, "OTB", "SH", 1.0, 1.5, 1.5, setpiece="O"),
        row(1, 2, "OTB", "RE", 2.0, 2.0, 2.0, team=2, player=20, setpiece="O"),
        row(2, 3, "OTB", "CH", 3.0, 3.0, 3.5, team=1, player=12, challenge_type="FIFTY", setpiece="O"),
        row(3, 4, "OTB", "CH", 5.0, 5.0, 5.5, team=None, player=None, setpiece="O"),
        row(4, 5, "OTB", "IT", 7.0, 7.0, 8.0, player=13, setpiece="O"),
    ]
    _, L = run(rows)
    assert L.state[f(1.8)] == TRAVELLING
    assert L.state[f(2.0)] == LOOSE and L.rule_names[L.rule[f(2.0)]] == "rebound_spell"
    assert L.state[f(2.5)] == LOOSE                                    # after a rebound
    assert L.state[f(3.2)] == LOOSE and L.rule_names[L.rule[f(3.2)]] == "duel_spell"
    assert L.state[f(5.2)] == LOOSE and L.rule_names[L.rule[f(5.2)]] == "ch_null_player"
    assert L.state[f(7.5)] == CONTROL


def test_unclear_gap_is_unknown_and_masked_and_video_missing_masked():
    rows = [
        row(0, 1, "OTB", "IT", 1.0, 1.0, 2.0, setpiece="O"),            # touch only: gap after it is unclear
        row(1, 2, "OTB", "PA", 4.0, 5.0, 5.0, player=11, setpiece="O", video_missing=True),
        row(2, 3, "OTB", "IT", 6.0, 6.0, 7.0, player=12, setpiece="O"),
    ]
    _, L = run(rows)
    assert L.state[f(3.0)] == UNKNOWN and L.rule_names[L.rule[f(3.0)]] == "gap_after_IT"
    assert not L.score_mask[f(3.0)]
    assert L.state[f(4.5)] == CONTROL and not L.score_mask[f(4.5)]       # labelled normally, excluded from scoring
    assert L.score_mask[f(6.5)]


def test_touch_and_carry_outcomes_and_drop_ball():
    rows = [
        row(0, 1, "OTB", "TC", 1.0, 1.0, 1.5, player=10, setpiece="O", touch_outcome="C"),   # touch lost (challenge)
        row(1, 2, "OTB", "TC", 3.0, 3.0, 3.5, player=11, setpiece="O", touch_outcome="R"),   # touch retained
        row(2, 3, "OTB", "BC", 5.0, 5.5, 6.0, player=12, setpiece="O", carry_outcome="R"),   # carry retained
        row(3, 4, "OTB", "BC", 8.0, 8.5, 9.0, player=13, setpiece="O", carry_outcome="L"),   # carry lost
        row(4, 5, "OTB", "PA", 11.0, 12.0, 12.0, player=14, setpiece="D"),                    # drop ball
        row(5, 6, "OTB", "IT", 13.0, 13.0, 14.0, player=15, setpiece="O"),
    ]
    _, L = run(rows)
    assert L.state[f(2.0)] == LOOSE and L.rule_names[L.rule[f(2.0)]] == "loose_after_TC"
    assert L.state[f(4.0)] == CONTROL and L.player[f(4.0)] == 11
    assert L.state[f(7.0)] == CONTROL and L.player[f(7.0)] == 12 and L.score_mask[f(7.0)]
    assert L.state[f(10.0)] == LOOSE
    assert L.state[f(11.5)] == CONTROL and L.player[f(11.5)] == 14       # drop ball live from the spell start
    assert L.log["drop_ball_live"] == [5]


def test_overlap_later_spell_wins_and_end_is_dead():
    rows = [
        row(0, 1, "OTB", "CH", 1.0, 1.0, 3.0, player=10, challenge_type="T", setpiece="O"),
        row(1, 2, "OTB", "IT", 2.5, 2.5, 4.0, team=2, player=20, setpiece="O"),
        row(2, 3, "END", None, 6.0, 6.0, 6.0),
    ]
    _, L = run(rows, seconds=10)
    assert L.player[f(2.8)] == 20
    assert len(L.log["overlaps"]) == 1
    assert (L.state[f(6.0):] == DEAD).all()


def test_tag_free_package_never_imports_labels():
    for p in (SRC / "m1_possession").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "m1_labels" for n in names), f"{p} imports m1_labels"
        assert "m1_labels" not in p.read_text(encoding="utf-8"), f"{p} mentions m1_labels"


def test_locked_split_is_reproducible_and_disjoint():
    from m1_possession.splits import (ALL_MATCHES, DEV_MATCHES, FOLDS, LEARNING_MATCHES, STRESS_MATCHES, TEST_MATCHES,
                                      make_split)
    assert make_split() == (TEST_MATCHES, FOLDS)
    assert len(TEST_MATCHES) == 16 and len(LEARNING_MATCHES) == 30 and all(len(f) == 6 for f in FOLDS)
    parts = [set(TEST_MATCHES), set(LEARNING_MATCHES), set(STRESS_MATCHES)]
    assert all(a.isdisjoint(b) for i, a in enumerate(parts) for b in parts[i + 1:])
    assert set().union(*parts) == set(ALL_MATCHES)
    assert set(DEV_MATCHES) <= set(LEARNING_MATCHES)
