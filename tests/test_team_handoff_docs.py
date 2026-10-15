"""The Python examples in docs/team_handoff.md are executed against SYNTHETIC data so they cannot drift from the API."""

import importlib.util
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "team_handoff.md"

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)


def python_blocks(text: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)


def test_handoff_examples_execute_against_the_real_api(tmp_path, monkeypatch, capsys):
    records = [H.rec(n) for n in range(10, 40)] + [H.rec(n, period=2) for n in range(60, 70)]
    events = [H.event(1, 101, 12 / 30), H.event(2, 102, 65 / 30, period=2), H.event(3, 103, 9000.0, period=3)]
    raw = H.build(tmp_path, records, events=events, versions=("root", "march", "may"))
    before = H.digest(raw)
    monkeypatch.setenv("FF_RAW_ROOT", str(raw))
    monkeypatch.setenv("FF_MATCH", "100")
    blocks = python_blocks(DOC.read_text(encoding="utf-8"))
    assert len(blocks) >= 8
    ns: dict = {}
    for i, code in enumerate(blocks):
        exec(compile(code, f"team_handoff.md block {i}", "exec"), ns)       # one shared namespace, in document order
    out = capsys.readouterr().out
    assert "complete" in out or "partial" in out
    assert "no_tracking_for_period" in out                                   # the event-only period 3 is reported, not hidden
    assert ns["q"].scan.status == "complete" and ns["q"].event_periods_without_tracking == (3,)
    assert len(ns["attached"]) == 2 and len(ns["md"].unmatched_events()) == 1
    assert H.digest(raw) == before


def test_documented_metric_names_exist():
    text = DOC.read_text(encoding="utf-8")
    for name in ("ball_raw_missing", "player_raw_missing", "player_raw_ambiguous", "player_smoothed_ambiguous", "player_estimated",
                 "events_temporally_matched", "events_unmatched", "unmatched_reasons", "players_per_side_per_frame"):
        assert name in text
