"""M1 step 1: run baselines C0 (naive) and C1 (Vidal-Codina / databallpy) on the dev matches and score them against
labels v1 with ``m1_labels.scoring``.

Inputs per match (all tag-free for the baselines): raw ball and raw players from the event-detection positions cache,
raw ball visibility from data/processed/m1_possession/frames/, pitch size from the metadata; smoothed with
``m1_possession.smooth_ball`` (default). The labels are read only for scoring.

    python scripts/m1/run_baselines.py --workers 2

Writes data/processed/m1_possession/baselines/scores.json and prints the report.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from football_fatigue.ingestion import Dataset
from m1_labels.scoring import STATES4, merge, score
from m1_possession import SmootherConfig, smooth_ball
from m1_possession.baselines import c0_naive, c1_paper
from m1_possession.splits import DEV_MATCHES

ROOT = "data/raw/FIFA World Cup 2022"
POS_CACHE = Path("data/processed/event_detection/cache")
LABELS = Path("data/processed/m1_possession/labels/v1")
FRAMES = Path("data/processed/m1_possession/frames")
OUT = Path("data/processed/m1_possession/baselines")
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)


def run_match(mid: int) -> dict:
    meta = Dataset(ROOT).metadata(str(mid))
    cpath = next(iter(sorted(POS_CACHE.glob(f"{mid}*.positions.npz"))))
    with np.load(cpath, allow_pickle=False) as z:
        cfn, ct, cper, ball, px, py = z["frame_num"], z["t"], z["period"].astype(int), z["ball"], z["px"], z["py"]
        pid, side = z["player_id"], z["player_side"]
    lab = dict(np.load(LABELS / f"{mid}.npz", allow_pickle=False))
    fr = dict(np.load(FRAMES / f"{mid}.npz", allow_pickle=False))
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    idx = np.array([li[int(f)] for f in cfn])
    vis = VIS_NAMES[fr["ball_visibility"][idx]]
    b = smooth_ball(ct, ball[:, 0], ball[:, 1], ball[:, 2], visibility=vis, period=cper, cfg=SmootherConfig())
    team = np.where(side == 0, meta.home_team_id, meta.away_team_id).astype(np.int64)
    sub = {k: lab[k][idx] for k in ("state", "player_id", "team_id", "score_mask", "period", "rule")}
    out = dict(match=mid)
    for name, res in (("C0", c0_naive(ct, cper, b, px, py, pid, team)),
                      ("C1", c1_paper(ct, cper, b, px, py, pid, team, meta.pitch_length_m, meta.pitch_width_m))):
        out[name] = score(sub, res.state, res.player, res.team, meta.fps, rule_names=lab["rule_names"])
        out[name]["pred_rules"] = {res.rule_names[k]: int(v) for k, v in zip(*np.unique(res.rule, return_counts=True))}
    return out


def report(res: list[dict]) -> dict:
    summary = {}
    for name in ("C0", "C1"):
        m = merge([r[name] for r in res])
        summary[name] = m
        conf = np.asarray(m["confusion"], float)
        rowp = 100 * conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
        print(f"\n=== {name}: {m['frames']} scored frames, frame accuracy {100 * m['accuracy']:.1f}% ===")
        print("| answer key \\ predicted | " + " | ".join(STATES4) + " | frames |")
        print("|---|" + "---|" * 5)
        for i, s in enumerate(STATES4):
            print(f"| {s} | " + " | ".join(f"{v:.1f}" for v in rowp[i]) + f" | {int(conf[i].sum())} |")
        print(f"player accuracy in CONTROL {100 * m['player_acc']:.1f}%, team accuracy {100 * m['team_acc']:.1f}% ({m['control_frames']} frames)")
        for k in ("possession_change", "dead_start_0.3s", "dead_end_0.3s", "dead_start_2s", "dead_end_2s"):
            v = m[k]
            print(f"{k}: P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (true {v['n_true']}, predicted {v['n_pred']})")
        worst = sorted(((r, v) for r, v in m["by_rule"].items() if v["frames"] > 2000), key=lambda kv: kv[1]["acc"])[:8]
        print("lowest state accuracy by answer-key rule:", ", ".join(f"{r} {100 * v['acc']:.0f}% ({v['frames']})" for r, v in worst))
        pr = {}
        for r in res:
            for k, v in r[name]["pred_rules"].items():
                pr[k] = pr.get(k, 0) + v
        print("prediction rules (frames):", pr)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with multiprocessing.Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(run_match, list(DEV_MATCHES))
    s = report(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(dict(summary=s, matches=res), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
