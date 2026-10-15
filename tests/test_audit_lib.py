"""Tests for scripts/audit/audit_lib.py verification logic.

All data below is SYNTHETIC and only mimics the shape of the audit logic's inputs; it is not
real source data and does not describe the real schemas.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_LIB = Path(__file__).resolve().parents[1] / "scripts" / "audit" / "audit_lib.py"
_spec = importlib.util.spec_from_file_location("audit_lib", _LIB)
L = importlib.util.module_from_spec(_spec)
sys.modules["audit_lib"] = L
_spec.loader.exec_module(L)

FPS = 30.0
FRAME = 1 / FPS


def entry(j, x, y, **kw):
    return dict(jerseyNum=j, x=x, y=y, visibility="VISIBLE", confidence="HIGH", **kw)


def test_duplicate_jersey_classification_and_order_independence():
    assert L.classify_duplicate_jerseys(None)["status"] == "missing"
    assert L.classify_duplicate_jerseys([entry("1", 0, 0), entry("2", 1, 1)])["status"] == "unique"
    ident = [entry("1", 0, 0), entry("1", 0, 0)]
    assert L.classify_duplicate_jerseys(ident)["status"] == "duplicate_identical"
    conf = [entry("1", 0, 0), entry("1", 0.1, 0), entry("2", 5, 5)]
    r = L.classify_duplicate_jerseys(conf)
    assert r["status"] == "duplicate_conflicting"
    assert r["max_spread_m"] == pytest.approx(0.1)
    assert len(r["groups"]["1"]) == 2  # both candidates preserved; none selected
    assert L.classify_duplicate_jerseys(list(reversed(conf)))["status"] == "duplicate_conflicting"


def ev(ge, pe, t, gt="OTB", pt="PA", **extra):
    return dict(gameEventId=ge, possessionEventId=pe, startTime=t, gameEvents=dict(gameEventType=gt, period=1),
                possessionEvents=dict(possessionEventType=pt), **extra)


def test_repeated_pair_rows_counts_every_row():
    rows = [ev(1, 10, 1.0), ev(1, 10, 2.0, gt="FOUL", pt="FO"), ev(2, 11, 3.0)]
    assert L.repeated_pair_rows(rows) == 2


def test_collision_safe_matching_preserves_rows_and_reports_unmatched():
    a = [ev(1, 10, 1.0), ev(1, 10, 2.0, gt="FOUL", pt="FO"), ev(2, 11, 3.0), ev(2, 11, 3.0), ev(9, None, 4.0, pt="IT")]
    b = [ev(2, 11, 3.0), ev(1, 10, 2.0, gt="FOUL", pt="FO"), ev(1, 10, 1.0)]
    m = L.match_rows(a, b)
    matched_a = sorted(i for i, _ in m["pairs"])
    assert matched_a == [0, 1, 2]                       # the colliding pair is matched row by row
    assert sorted(m["unmatched_a"]) == [3, 4]           # surplus duplicate and the IT row are reported
    assert m["unmatched_b"] == []
    assert len(m["pairs"]) + len(m["unmatched_a"]) == len(a)   # nothing dropped
    assert len(m["ambiguous_signatures"]) == 1          # (2, 11, 3.0, ...) has multiplicity 2 vs 1


def test_matching_is_one_to_one_for_identical_rows():
    a = [ev(1, 1, 1.0), ev(1, 1, 1.0)]
    m = L.match_rows(a, [ev(1, 1, 1.0), ev(1, 1, 1.0)])
    assert sorted(j for _, j in m["pairs"]) == [0, 1]


def test_field_comparison_scope_separates_changes_from_schema_differences():
    a = ev(1, 1, 1.0, sequence=1.0, homePlayers=[entry(1, 0, 0)])
    b = ev(1, 1, 1.0, homePlayers=[entry(1, 5, 5)])
    b["possessionEvents"]["touchPlayerId"] = None
    a["possessionEvents"]["touchPlayerId"] = 7
    c = L.compare_fields(a, b)
    assert c["changed"] == ["possessionEvents.touchPlayerId"]   # null vs value is a change
    assert c["only_a"] == ["sequence"] and c["only_b"] == []   # schema difference, not a value change
    assert not any(p.startswith("homePlayers") for p in c["changed"])  # positions compared separately
    q = L.compare_positions(a, b)
    assert q["xy_changed"] == 1 and q["player_set_differs"] is False


def frames(times, period=1):
    return [(n, t, period) for n, t in enumerate(times)]


def run_aligner(events, frame_list):
    al = L.Aligner(events, FPS)
    for fn, t, p in frame_list:
        al.offer(fn, t, p, lambda fn=fn: f"snap{fn}")
    return al


def test_alignment_nearest_frame_and_tolerance_is_one_frame():
    fl = frames([0, FRAME, 2 * FRAME])
    al = run_aligner([(0, 1.4 * FRAME, 1), (1, 2 * FRAME + 1.01 * FRAME, 1), (2, 2 * FRAME + 0.99 * FRAME, 1)], fl)
    assert al.best[0][2] == 1                    # nearer to frame 1 than to frame 2
    assert al.outcome(0, 1.4 * FRAME, 1) == ("matched", None)
    assert al.outcome(1, 3.01 * FRAME, 1)[0] == "unmatched"    # more than one frame beyond the last frame
    assert al.outcome(2, 2.99 * FRAME, 1)[0] == "matched"      # within one frame of the last frame


def test_alignment_exact_tie_chooses_earlier_frame():
    al = run_aligner([(0, 0.5 * FRAME, 1)], frames([0, FRAME]))
    assert al.best[0][2] == 0


def test_alignment_requires_same_period_and_reports_reasons():
    fl = frames([0, FRAME], period=1) + frames([100, 100 + FRAME], period=2)
    ev_ = [(0, 0.0, 2), (1, 0.0, 0), (2, 50.0, 1), (3, 500.0, 3), (4, None, 1), (5, FRAME / 2, 1)]
    al = run_aligner(ev_, [(n, t, p) for n, (_, t, p) in enumerate(fl)])
    assert al.outcome(0, 0.0, 2) == ("unmatched", "period_label_differs_from_nearest_frame")
    assert al.outcome(1, 0.0, 0) == ("unmatched", "invalid_source_period")
    assert al.outcome(2, 50.0, 1) == ("unmatched", "outside_period_frame_range")
    assert al.outcome(3, 500.0, 3) == ("unmatched", "no_tracking_for_period")
    assert al.outcome(4, None, 1) == ("unmatched", "missing_event_time")
    assert al.outcome(5, FRAME / 2, 1)[0] == "matched"


def test_coordinate_status_thresholds_and_not_assessable():
    ev_players = {("homePlayers", j): dict(x=float(j), y=0.0) for j in range(11)}
    same = {k: (p["x"], p["y"]) for k, p in ev_players.items()}
    assert L.coordinate_status(ev_players, same)["status"] == "coordinate_agreeing"
    moved = {k: (v[0] + 5, v[1]) for k, v in same.items()}
    assert L.coordinate_status(ev_players, moved)["status"] == "coordinate_disagreeing"
    assert L.coordinate_status(ev_players, {})["status"] == "coordinate_not_assessable"


def test_run_matches_refuses_to_write_inside_raw_root(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(L.AuditError):
        L.run_matches(lambda r, m: {}, str(raw), ["1"], raw / "out", 1, False)


# ---------------------------------------------------------------- output-path safety (synthetic, tmp_path only)

import os
import subprocess

RAW_CONTENT = '{"synthetic": "raw metadata, must never change"}'


def make_raw(tmp_path):
    raw = tmp_path / "raw"
    meta = raw / "Metadata-20000101T000000Z-1-001" / "Metadata"
    meta.mkdir(parents=True)
    f = meta / "3812.json"
    f.write_text(RAW_CONTENT)
    return raw, f


def make_link(link, target):
    """Directory symlink, or a Windows junction when symlinks need a privilege; skip if neither works."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if r.returncode == 0:
            return
    pytest.skip("symlinks/junctions unavailable")


@pytest.mark.parametrize("good", ["3812", "10517", "0", "123456789"])
def test_valid_match_ids_are_accepted(good):
    assert L.validate_match_id(good) == good


# The exact ID of the original report was not available to this test; these are representative traversal,
# separator, absolute-path, and malformed forms (including a path that reaches a raw metadata file).
@pytest.mark.parametrize("bad", [
    "../3812", r"..\3812", "../../raw/Metadata-20000101T000000Z-1-001/Metadata/3812",
    r"..\..\raw\Metadata-20000101T000000Z-1-001\Metadata\3812", "3812/../../x", "/etc/passwd", r"C:\Windows\x",
    "C:/x", r"\\server\share\3812", "", " 3812", "3812 ", "38 12", "3812.json", "３８１２", "1234567890", None, 3812,
])
def test_malformed_match_ids_are_rejected(bad):
    with pytest.raises(L.AuditError):
        L.validate_match_id(bad)


def test_traversal_id_is_rejected_before_any_work_and_raw_file_is_untouched(tmp_path):
    raw, raw_file = make_raw(tmp_path)
    calls = []
    out = tmp_path / "out"
    evil = "../../raw/Metadata-20000101T000000Z-1-001/Metadata/3812"
    with pytest.raises(L.AuditError):
        L.run_matches(lambda r, m: calls.append(m) or {}, str(raw), ["3812", evil], out, 1, True)
    assert calls == []                                   # nothing was submitted, not even the valid id
    assert raw_file.read_text() == RAW_CONTENT
    assert not out.exists() or not any(out.iterdir())


def test_destination_must_stay_inside_output_dir_and_outside_raw(tmp_path):
    raw, raw_file = make_raw(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert L.safe_destination(raw, out, "3812") == out.resolve() / "3812.json"
    assert L.safe_destination(raw, out, "3812", suffix=".json.tmp").name == "3812.json.tmp"
    with pytest.raises(L.AuditError):
        L.safe_destination(raw, out, "../3812")
    with pytest.raises(L.AuditError):
        L.safe_destination(raw, out, str(raw_file))      # absolute path
    with pytest.raises(L.AuditError):
        L.safe_destination(raw, raw_file.parent, "3812")   # output directory inside raw


def test_output_dir_linked_into_raw_is_rejected(tmp_path):
    raw, raw_file = make_raw(tmp_path)
    link = tmp_path / "out_link"
    make_link(link, raw_file.parent)
    with pytest.raises(L.AuditError):
        L.run_matches(lambda r, m: {}, str(raw), ["3812"], link, 1, True)
    assert raw_file.read_text() == RAW_CONTENT


def test_destination_file_symlinked_to_raw_is_rejected(tmp_path):
    raw, raw_file = make_raw(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    try:
        os.symlink(raw_file, out / "3812.json")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable")
    with pytest.raises(L.AuditError):
        L.run_matches(lambda r, m: {"x": 1}, str(raw), ["3812"], out, 1, True)
    assert raw_file.read_text() == RAW_CONTENT


def test_valid_run_writes_only_inside_output_dir_and_leaves_no_temp_file(tmp_path):
    raw, raw_file = make_raw(tmp_path)
    out = tmp_path / "out"
    summary = L.run_matches(lambda r, m: {"value": int(m)}, str(raw), ["3812", "10517"], out, 1, False)
    assert summary == dict(attempted=2, failed={})
    assert sorted(p.name for p in out.iterdir()) == ["10517.json", "3812.json"]
    assert raw_file.read_text() == RAW_CONTENT


# ---------------------------------------------------------------- batch exit status (no real audits are run)

def ok_func(root, mid):
    return {"v": 1}


def bad_func(root, mid):
    raise RuntimeError("synthetic failure")


def mixed_func(root, mid):
    if mid == "2":
        raise RuntimeError("synthetic failure")
    return {"v": 1}


def cli(tmp_path, func, mids, extra=()):
    raw, _ = make_raw(tmp_path)
    seen = []
    code = L.cli_main("synthetic", str(tmp_path / "default_out"), func, lambda res: seen.append(sorted(res)),
                      ["--root", str(raw), "--out", str(tmp_path / "out"), "--workers", "1", "--matches", *mids, *extra])
    return code, seen


def test_exit_status_zero_when_all_succeed(tmp_path, capsys):
    code, _ = cli(tmp_path, ok_func, ["1", "2"])
    assert code == 0
    assert "complete, successful output" in capsys.readouterr().out


def test_exit_status_nonzero_when_one_fails_and_diagnostics_are_kept(tmp_path, capsys):
    code, seen = cli(tmp_path, mixed_func, ["1", "2"])
    cap = capsys.readouterr()
    assert code == 1
    assert "synthetic failure" in cap.out + cap.err and "FAILED" in cap.err
    assert "complete, successful output" not in cap.out    # no misleading success summary


def test_exit_status_nonzero_when_all_fail(tmp_path, capsys):
    code, _ = cli(tmp_path, bad_func, ["1", "2"])
    assert code == 1
    assert "complete, successful output" not in capsys.readouterr().out


def test_exit_status_nonzero_for_missing_requested_results(tmp_path, capsys):
    code, _ = cli(tmp_path, ok_func, ["1", "2"], extra=["--aggregate"])    # nothing was ever computed
    cap = capsys.readouterr()
    assert code == 1 and "missing output" in cap.err


def test_invalid_match_id_is_a_controlled_error_with_nonzero_status(tmp_path, capsys):
    code, _ = cli(tmp_path, ok_func, ["../3812"])
    assert code == 2 and "invalid match id" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_check_outputs_reports_failed_missing_and_incomplete():
    res = {"1": {"ok": True}, "2": {"ok": False, "error": "boom"}, "3": {"ok": False}}
    assert L.check_outputs(res, ["1", "2", "3", "4"]) == {"2": "boom", "3": "incomplete output", "4": "missing output"}
