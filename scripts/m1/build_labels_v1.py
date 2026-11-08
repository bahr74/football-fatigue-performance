"""M1 step 1: answer key v1 for all 64 matches (PFF tags; evaluation only).

Per match:
- one fast pass over the raw tracking records (regular expressions, no full JSON parse): frameNum, videoTimeMs, period,
  raw ball visibility and the ``game_event`` overlay id per frame; the overlay's start_frame / end_frame / video_url are
  read from the first record of each game event;
- event times aligned to frames with ``synchronization.align_events`` (frames passed as light records);
- labels from ``m1_labels.label_match`` (rules v1);
- far-control check: CONTROL frames where the labelled player is > 10 m from our smoothed raw ball for > TOLERANCE_S
  keep their label but get score_mask False (raw players and raw ball from the event-detection positions cache);
- periods excluded for a suspected side/identity anomaly (event_detection.config.EXCLUDED_PERIODS) are not scored.

Writes data/processed/m1_possession/labels/v1/<match>.npz (labels, all tracking frames), .../frames/<match>.npz (tag-free
frame index with raw ball visibility), .../labels/v1/summary.json; prints the summary and the 10 worst far-control cases.

    python scripts/m1/build_labels_v1.py --workers 8
"""
from __future__ import annotations

import argparse
import bz2
import json
import multiprocessing
import re
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from football_fatigue.event_detection.config import EXCLUDED_PERIODS
from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.synchronization import MatchStatus, align_events
from m1_labels import CONTROL, DEAD, STATE_NAMES, TOLERANCE_S, UNKNOWN, Row, label_match
from m1_possession import SmootherConfig, smooth_ball
from m1_possession.splits import ALL_MATCHES

ROOT = "data/raw/FIFA World Cup 2022"
POS_CACHE = Path("data/processed/event_detection/cache")
OUT = Path("data/processed/m1_possession/labels/v1")
FRAMES = Path("data/processed/m1_possession/frames")
FAR_M = 10.0
VIS = {"VISIBLE": 1, "ESTIMATED": 2}
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)
RX = {k: re.compile(p) for k, p in dict(
    fn=r'"frameNum":(\d+)', vt=r'"videoTimeMs":([-0-9.eE]+)', per=r'"period":(\d+|null)',
    bvis=r'"balls":\[(?:\{"visibility":"(\w+)")?', ge=r'"game_event_id":([0-9.]+|null)').items()}


def raw_pass(path: Path) -> dict:
    fn, vt, per, bvis, ge = [], [], [], [], []
    ov: dict = {}
    last_fn = None
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = RX["fn"].search(line)
            if not m:
                continue
            f = int(m.group(1))
            if f == last_fn:                                     # repeated record of the same frame: keep the first
                continue
            last_fn = f
            fn.append(f)
            vt.append(float(RX["vt"].search(line).group(1)) / 1000.0)
            p = RX["per"].search(line).group(1)
            per.append(int(p) if p != "null" else 0)
            b = RX["bvis"].search(line)
            bvis.append(VIS.get(b.group(1), 3) if b and b.group(1) else 0)
            g = RX["ge"].search(line)
            gid = int(float(g.group(1))) if g and g.group(1) != "null" else -1
            ge.append(gid)
            if gid >= 0 and gid not in ov:
                d = json.loads(line).get("game_event") or {}
                ov[gid] = (d.get("start_frame"), d.get("end_frame"), d.get("game_event_type"), d.get("video_url"))
    return dict(fn=np.asarray(fn, np.int64), t=np.asarray(vt), per=np.asarray(per, np.int8), bvis=np.asarray(bvis, np.int8),
                ge=np.asarray(ge, np.int64), ov=ov)


def to_row(r) -> Row:
    ge, pe = r.attributes.get("gameEvents") or {}, r.attributes.get("possessionEvents") or {}
    return Row(idx=r.source_row_index, ge=r.game_event_id, pe=r.possession_event_id, get=r.game_event_type,
               pet=r.possession_event_type, period=r.canonical_period, st=r.start_time_s, et=r.event_time_s, en=r.end_time_s,
               team=r.source_team_id, player=r.source_player_id, setpiece=ge.get("setpieceType"), out_type=ge.get("outType"),
               video_missing=bool(ge.get("videoMissing")), non_event=bool(pe.get("nonEvent")),
               initial_non_event=bool(ge.get("initialNonEvent")), challenge_type=pe.get("challengeType"),
               carry_outcome=pe.get("ballCarryOutcome"), touch_outcome=pe.get("touchOutcomeType"))


def run_match(mid: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run_match(int(mid))


def _run_match(mid: int) -> dict:
    ds = Dataset(ROOT)
    meta = ds.metadata(str(mid))
    events = ds.load_events(str(mid), EventVersion.ROOT)
    rows = [to_row(r) for r in events]
    path, _ = ds.tracking_source(str(mid))
    rp = raw_pass(path)
    fn, t, per, n = rp["fn"], rp["t"], rp["per"], len(rp["fn"])
    if n > 1 and np.any(np.diff(t) <= 0):
        raise RuntimeError(f"{mid}: tracking times not strictly increasing")
    fidx = {int(f): i for i, f in enumerate(fn)}
    canon = np.where((per >= 1) & (per <= 4), per, 0)
    frames = (SimpleNamespace(match_id=mid, frame_num=int(fn[i]), video_time_s=float(t[i]),
                              canonical_period=int(canon[i]) or None, source=None, source_record_indices=(),
                              quality_flags=frozenset()) for i in range(n))   # streamed: align_events reads them once
    copies = [replace(r, event_row_id=f"{r.event_row_id}|{tag}", event_time_s=tv)
              for r in events for tag, tv in (("start", r.start_time_s), ("event", r.event_time_s), ("end", r.end_time_s))]
    sync = align_events(copies, frames, match_id=mid, event_version=EventVersion.ROOT, fps=meta.fps)
    row_frames: dict = {}
    for a in sync.alignments:
        tag = a.event_row_id.rsplit("|", 1)[1]
        row_frames.setdefault(int(a.source_row_index), {})[tag] = fidx.get(a.frame_num) if a.status is MatchStatus.MATCHED else None
    spans, ov_bad = {}, 0
    for g, (sf, ef, _typ, _url) in rp["ov"].items():
        if sf is None or ef is None or int(sf) not in fidx or int(ef) not in fidx:
            ov_bad += 1
            continue
        spans[g] = (fidx[int(sf)], fidx[int(ef)])
    L = label_match(t, canon, rows, spans, row_frames)
    mask = L.score_mask.copy()
    excl = EXCLUDED_PERIODS.get(mid, ())
    mask[np.isin(canon, list(excl))] = False

    # far-control check with raw positions (event-detection cache) and our smoothed raw ball
    cpath = next(iter(sorted(POS_CACHE.glob(f"{mid}*.positions.npz"))))
    with np.load(cpath, allow_pickle=False) as z:
        cfn, cball, cpx, cpy, cpid = z["frame_num"], z["ball"], z["px"], z["py"], z["player_id"]
    ci = np.array([fidx.get(int(f), -1) for f in cfn])
    if (ci < 0).any():
        raise RuntimeError(f"{mid}: {int((ci < 0).sum())} cached frames not in the raw pass")
    ours = smooth_ball(t[ci], cball[:, 0], cball[:, 1], cball[:, 2], visibility=VIS_NAMES[rp["bvis"][ci]], period=canon[ci],
                       cfg=SmootherConfig())
    col_of = {int(p): j for j, p in enumerate(cpid) if p >= 0}
    lp = L.player[ci]
    cols = np.array([col_of.get(int(p), -1) for p in lp])
    ok = (L.state[ci] == CONTROL) & (cols >= 0) & np.isfinite(ours.x)
    dist = np.full(len(ci), np.nan)
    k = np.flatnonzero(ok)
    dist[k] = np.hypot(cpx[k, cols[k]] - ours.x[k], cpy[k, cols[k]] - ours.y[k])
    far = np.isfinite(dist) & (dist > FAR_M)
    tolf = int(round(TOLERANCE_S * meta.fps))
    far_frames = np.zeros(n, bool)
    runs = []
    src = L.src_ge[ci]
    i = 0
    while i < len(ci):
        if far[i]:
            j = i
            while j + 1 < len(ci) and far[j + 1] and src[j + 1] == src[i] and ci[j + 1] == ci[j] + 1:
                j += 1
            if j - i + 1 > tolf:
                far_frames[ci[i:j + 1]] = True
                g = int(src[i])
                runs.append(dict(match=mid, t0=round(float(t[ci[i]]), 3), dur_s=round(float(t[ci[j]] - t[ci[i]]), 3),
                                 med_dist_m=round(float(np.median(dist[i:j + 1])), 1), game_event_id=g,
                                 video_url=(rp["ov"].get(g) or (None, None, None, None))[3]))
            i = j + 1
        else:
            i += 1
    mask[far_frames] = False

    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{mid}.npz", frame_num=fn, t=t, period=canon, state=L.state, player_id=L.player, team_id=L.team,
                        score_mask=mask, far_control=far_frames, src_game_event_id=L.src_ge, src_possession_event_id=L.src_pe,
                        rule=L.rule, rule_names=np.array(L.rule_names), state_names=np.array(STATE_NAMES), tolerance_s=TOLERANCE_S,
                        version="v1")
    np.savez_compressed(FRAMES / f"{mid}.npz", frame_num=fn, t=t, period=canon, ball_visibility=rp["bvis"],
                        visibility_codes=np.array(["none", "VISIBLE", "ESTIMATED", "other"]))

    inp = canon > 0
    st = L.state
    spell_ges = {r.ge for r in rows if r.get in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")}
    otb_ov = np.isin(rp["ge"], list(spell_ges))
    ctrl = st == CONTROL
    from_overlay = sum(1 for g in spell_ges if g in spans)
    return dict(match=mid, frames=int(inp.sum()), state_pct={STATE_NAMES[s]: 100 * float((st[inp] == s).mean()) for s in range(5)},
                in_play_pct=100 * float((st[inp] != DEAD).mean()),
                ctrl_with_overlay=100 * float(otb_ov[ctrl].mean()) if ctrl.any() else None,
                overlay_is_ctrl=100 * float(ctrl[otb_ov].mean()) if otb_ov.any() else None,
                mask_pct=100 * float(mask[inp].mean()), far_frames=int(far_frames.sum()), far_runs=runs,
                spells=len(spell_ges), spells_from_overlay=from_overlay, overlay_bad=ov_bad, excluded_periods=list(excl),
                periods=sorted(int(p) for p in np.unique(canon[inp])), unknown_rules=dict(Counter(
                    L.rule_names[r] for r in L.rule[inp & (st == UNKNOWN)].tolist())),
                drop_balls=len(L.log.get("drop_ball_live", [])), overlaps=len(L.log.get("overlaps", [])),
                rules=dict(Counter(L.rule_names[r] for r in L.rule[inp].tolist())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--matches", nargs="*", type=int)
    a = ap.parse_args()
    mids = a.matches or list(ALL_MATCHES)
    with multiprocessing.Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(run_match, mids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("| match | DEAD | CONTROL | TRAVEL | LOOSE | UNKNOWN | in play | CONTROL has OTB overlay | OTB overlay is CONTROL | scored | flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        s = r["state_pct"]
        flags = []
        if not 50 <= r["in_play_pct"] <= 66:
            flags.append(f"in play {r['in_play_pct']:.0f}%")
        if s["UNKNOWN"] > 0.5:
            flags.append("UNKNOWN > 0.5%")
        if (r["ctrl_with_overlay"] or 0) < 95 or (r["overlay_is_ctrl"] or 0) < 95:
            flags.append("overlay agreement < 95%")
        if r["spells_from_overlay"] < r["spells"] - 2:
            flags.append(f"{r['spells'] - r['spells_from_overlay']} spells without overlay")
        if r["far_frames"] > 0.02 * r["frames"] * s["CONTROL"] / 100:
            flags.append(f"far control {100 * r['far_frames'] / (r['frames'] * s['CONTROL'] / 100):.1f}% of CONTROL")
        if r["excluded_periods"]:
            flags.append(f"periods {r['excluded_periods']} not scored")
        print(f"| {r['match']} | {s['DEAD']:.1f} | {s['CONTROL']:.1f} | {s['TRAVELLING']:.1f} | {s['LOOSE']:.1f} | {s['UNKNOWN']:.2f} | "
              f"{r['in_play_pct']:.1f} | {r['ctrl_with_overlay']:.1f} | {r['overlay_is_ctrl']:.1f} | {r['mask_pct']:.1f} | {'; '.join(flags)} |")
    tot = sum(r["frames"] for r in res)
    pooled = {k: sum(r["state_pct"][k] * r["frames"] for r in res) / tot for k in STATE_NAMES}
    print("all:", {k: round(v, 2) for k, v in pooled.items()}, "in play", round(sum(r["in_play_pct"] * r["frames"] for r in res) / tot, 1),
          "scored", round(sum(r["mask_pct"] * r["frames"] for r in res) / tot, 1))
    runs = sorted((x for r in res for x in r["far_runs"]), key=lambda x: (-x["dur_s"], -x["med_dist_m"]))
    print(f"\nfar-control runs (> {FAR_M} m for > {TOLERANCE_S} s): {len(runs)}, frames masked {sum(r['far_frames'] for r in res)}")
    for x in runs[:10]:
        print("  ", x)
    uk = Counter()
    for r in res:
        uk.update(r["unknown_rules"])
    print("UNKNOWN frames by rule:", dict(uk), "| drop balls live:", sum(r["drop_balls"] for r in res),
          "| overlaps:", sum(r["overlaps"] for r in res), "| overlay records unusable:", sum(r["overlay_bad"] for r in res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
