"""M1: per-match input cache (tag-free) and evaluation cache (labels + PFF passes) for the learning matches.

inputs/<match>.npz  – model input only: t, period, frame_num, our smoothed raw ball (x, y, z, vx, vy, flag), raw players
                      (px, py float32 (n, P), player_id, team_id per column), pitch size, fps. No tags, no *Smoothed fields.
evalset/<match>.npz – labels v1 on the same frames (state, player, team, score_mask, rule, rule names) and the PFF PA events
                      (passer, eventTime frame index) for pass-detection scoring. Read only by scoring code.

    python scripts/m1/build_inputs.py --workers 2            # learning matches (default)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from football_fatigue.ingestion import Dataset, EventVersion
from m1_possession import SmootherConfig, smooth_ball
from m1_possession.splits import LEARNING_MATCHES

ROOT = "data/raw/FIFA World Cup 2022"
POS = Path("data/processed/event_detection/cache")
LAB = Path("data/processed/m1_possession/labels/v1")
FR = Path("data/processed/m1_possession/frames")
INP = Path("data/processed/m1_possession/inputs")
EVS = Path("data/processed/m1_possession/evalset")
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)


def run(mid: int) -> str:
    ds = Dataset(ROOT)
    meta = ds.metadata(str(mid))
    z = dict(np.load(sorted(POS.glob(f"{mid}*.positions.npz"))[0]))
    lab = dict(np.load(LAB / f"{mid}.npz"))
    fr = dict(np.load(FR / f"{mid}.npz"))
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    idx = np.array([li[int(f)] for f in z["frame_num"]])
    per = z["period"].astype(np.int8)
    b = smooth_ball(z["t"], z["ball"][:, 0], z["ball"][:, 1], z["ball"][:, 2], visibility=VIS_NAMES[fr["ball_visibility"][idx]],
                    period=per, cfg=SmootherConfig())
    team = np.where(z["player_side"] == 0, meta.home_team_id, meta.away_team_id).astype(np.int64)
    INP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INP / f"{mid}.npz", t=z["t"], period=per, frame_num=z["frame_num"], bx=b.x, by=b.y, bz=b.z, bvx=b.vx,
                        bvy=b.vy, bflag=b.flag, px=z["px"], py=z["py"], player_id=z["player_id"], team_id=team,
                        pitch=np.array([meta.pitch_length_m, meta.pitch_width_m]), fps=meta.fps)
    events = ds.load_events(str(mid), EventVersion.ROOT)
    t = z["t"]
    pa_t, pa_p, pa_per = [], [], []
    for r in events:
        if r.possession_event_type == "PA" and r.event_time_s is not None:
            pe = r.attributes.get("possessionEvents") or {}
            passer = pe.get("passerPlayerId") or r.source_player_id
            if passer is not None:
                pa_t.append(r.event_time_s); pa_p.append(int(passer)); pa_per.append(r.canonical_period or 0)
    k = np.clip(np.searchsorted(t, np.asarray(pa_t)), 0, len(t) - 1)
    EVS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVS / f"{mid}.npz", state=lab["state"][idx], player_id=lab["player_id"][idx], team_id=lab["team_id"][idx],
                        score_mask=lab["score_mask"][idx], period=lab["period"][idx], rule=lab["rule"][idx], rule_names=lab["rule_names"],
                        pass_frame=k, pass_time=np.asarray(pa_t), pass_player=np.asarray(pa_p, np.int64), pass_period=np.asarray(pa_per))
    return f"{mid} ok ({len(t)} frames, {len(pa_t)} passes)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--matches", nargs="*", type=int)
    a = ap.parse_args()
    with multiprocessing.Pool(a.workers, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(run, a.matches or list(LEARNING_MATCHES)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
