"""Stage S1.1 audit v3: the near-crossing speed-spike rule on synthetic RAW caches (through the real Step 1 quality pass), and the
recorded v1 regression / v2 explanation of the written development outputs (skipped when absent)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections import audit as AU
from football_fatigue.intersections import audit_v3 as V3
from football_fatigue.intersections.detect import DetectConfig, detect

FPS, T0 = 10.0, 100.0
K = np.arange(101)
V3_DIR = Path("data/processed/m11-intersections-v3")


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


def crossing(spike_at_s=None, jump_m=1.3):
    """Perpendicular crossing at t = 105 s; optionally player B's position jumps ``jump_m`` sideways for one frame."""
    t = K / FPS - 5.0
    a = (0.23 + 2.0 * t, np.full(len(K), 0.37))
    bx = 0.2313 + 0 * t
    if spike_at_s is not None:
        bx = bx.copy()
        bx[int(round((spike_at_s + 5.0) * FPS))] += jump_m
    b = (bx, 0.37 + 2.0 * t)
    return cache([a, b])


def audited(c):
    qr = Q.assess(c, [])
    events, _ = detect(c, qr.mask, qr.steps, DetectConfig())
    cx = AU.Context(c, qr.mask, V3.AuditV3Config())
    return V3.audit_events(cx, events), AU.audit_events(AU.Context(c, qr.mask, AU.AuditConfig()), events)


def test_clean_crossing_is_unchanged_from_v2():
    v3, v2 = audited(crossing())
    assert len(v3) == 1
    assert v3[0]["status"] == v2[0]["status"] and v3[0]["primary_category"] == "clean_candidate"
    assert not v3[0]["checks"][V3.SPIKE_REASON] and v3[0]["measurements"]["spikes_b"]["spikes"] == 0


def test_spike_near_crossing_makes_clean_uncertain():
    v3, v2 = audited(crossing(spike_at_s=0.2))
    r = v3[0]
    assert r["measurements"]["spikes_b"]["spikes_near_crossing"] >= 1 and r["checks"][V3.SPIKE_REASON]
    for tau, s in r["status"].items():
        assert s["reasons"] == v2[0]["status"][tau]["reasons"] + [V3.SPIKE_REASON]
        assert s["category"] != "clean_candidate"


def test_spike_away_from_crossing_is_recorded_only():
    v3, v2 = audited(crossing(spike_at_s=-2.0))
    r = v3[0]
    assert r["measurements"]["spikes_b"]["spikes"] >= 1 and r["measurements"]["spikes_b"]["spikes_near_crossing"] == 0
    assert not r["checks"][V3.SPIKE_REASON]
    assert r["status"] == v2[0]["status"]


def test_spike_threshold_is_strict():
    assert V3.AuditV3Config().spike_speed_m_s == 12.0
    v3, _ = audited(crossing(spike_at_s=0.2, jump_m=0.5))                    # 5 m/s lateral step + 2 m/s motion: below 12 m/s
    assert v3[0]["measurements"]["spikes_b"]["spikes"] == 0


@pytest.mark.skipif(not (V3_DIR / "cross-match" / "summary.json").exists(), reason="v3 outputs not generated")
def test_written_v3_outputs_reproduce_v1_and_explain_v2():
    s = json.loads((V3_DIR / "cross-match" / "summary.json").read_text(encoding="utf-8"))
    assert s["schema"] == V3.SCHEMA
    for mid in s["matches"]:
        summ = json.loads((V3_DIR / mid / "root.summary.json").read_text(encoding="utf-8"))
        if summ["v1_regression"] is not None:
            assert summ["v1_regression"]["identical"]
        if summ["v2_comparison"] is not None:
            assert summ["v2_comparison"]["unexplained"] == 0 and summ["v2_comparison"]["same_event_ids"]
        assert math.isclose(sum(v for k, v in s["per_match"][mid].items() if k.endswith("@0.20") and not k.startswith("clean_")
                                and k.split("@")[0] in ("uncertain_candidate", "invalid_tracking_artifact", "non_time_aligned_crossing"))
                            + s["per_match"][mid]["clean_candidate@0.20"], s["per_match"][mid]["events"])
