import numpy as np

from football_fatigue.multicam_lab.analytics import heatmap, player_metrics, trajectory_segments
from football_fatigue.multicam_lab.fusion import Uncertainty, associate, state_of


def test_trajectory_breaks_on_gap_jump_and_identity_flag():
    pts = [{"t": 0.0, "p": (0, 0)}, {"t": 0.2, "p": (1, 0)}, {"t": 1.0, "p": (2, 0)},          # gap 0.8 s
           {"t": 1.2, "p": (30, 0)},                                                         # 28 m in 0.2 s
           {"t": 1.4, "p": (30.5, 0)}, {"t": 1.6, "p": (31, 0), "break_before": True}]
    segs = trajectory_segments(pts)
    assert [len(s) for s in segs] == [2, 1, 2, 1]


def test_heatmap_counts_and_outside():
    h = heatmap([(0.5, 0.5), (1, 1), (10, 10), (100, 0)], (0, 12), (0, 12), cell_m=3)
    assert h["counts"][0][0] == 2 and h["counts"][3][3] == 1 and h["outside"] == 1 and h["observations"] == 3


def test_player_metrics_never_report_physical_values():
    obs = [{"t": i * 0.2, "p": (i * 0.5, 0), "conf": 0.8, "state": "matched" if i % 2 else "moving-only"} for i in range(10)]
    m = player_metrics(obs, interval_s=3.8, step_s=0.2)
    e = m["experimental"]
    assert e["observation_count"] == 10 and e["instants_supported_by_both_cameras"] == 5
    assert e["visibility_ratio"] == 0.5
    assert all(v["status"] == "unavailable" for v in m["conditional"].values())
    assert "distance_covered_m" in m["conditional"]


def test_association_gates_and_flags_ambiguity():
    cov = (0.05 * np.eye(2)).tolist()
    mov = [{"k": 0, "pitch": (0, 0), "cov": cov}, {"k": 1, "pitch": (20, 0), "cov": cov}]
    fix = [{"k": 0, "pitch": (0.3, 0), "cov": cov}, {"k": 1, "pitch": (20.2, 0.1), "cov": cov}, {"k": 2, "pitch": (20.6, 0.0), "cov": cov}, {"k": 3, "pitch": (50, 0), "cov": cov}]
    links = associate(mov, fix, Uncertainty(0.3, 0.05))
    assert [(l["m"], l["f"]) for l in links] == [(0, 0), (1, 1)]
    assert links[0]["ambiguous"] is False and links[1]["ambiguous"] is True     # two fixed candidates near moving #1
    assert all(l["f"] != 3 for l in links)                                       # 30 m away: outside the gate


def test_state_labels():
    assert state_of(False, None, "moving", True) == "off-pitch"
    assert state_of(True, {"ambiguous": True}, "fixed", True) == "ambiguous"
    assert state_of(True, None, "fixed", False) == "fixed-only (outside other camera)"
