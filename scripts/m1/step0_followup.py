"""M1 step 0 follow-up (evaluation only; dev matches): kick time offset, missed kicks, is ballsSmoothed built from the
tags, and a leakage test of ballsSmoothed positions.

The smoother (recommended default ``SmootherConfig()``) sees only the raw ball and its visibility. PFF tags, the tracking
``game_event_id`` overlay and ``ballsSmoothed`` are read here for scoring only.

    python scripts/m1/step0_followup.py --workers 4

Writes data/processed/m1_possession/step0_followup/results.json and prints the report.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_smoother import CACHE, KICK_TYPES, ROOT, build_cache, dv_series, peaks
from football_fatigue.event_detection.config import DEV_MATCHES
from football_fatigue.ingestion import Dataset, EventVersion
from m1_possession import FLAG_MISSING, SmootherConfig, smooth_ball

OUT = Path("data/processed/m1_possession/step0_followup")
OFFSET_WIN_S = 1.0
CONTROL_RADIUS_M = 1.5
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)


def window(arr, k, a, b):
    return arr[max(0, k + a):max(0, k + b + 1)]


def dead_mask(t, per, rows) -> tuple[np.ndarray, list]:
    """PFF dead ball: OUT eventTime -> startTime of the next event (another gameEventId, same period, time order)."""
    order = sorted((r for r in rows if r.event_time_s is not None and r.canonical_period is not None),
                   key=lambda r: (r.canonical_period, r.event_time_s, r.source_row_index))
    dead = np.zeros(len(t), bool)
    spans = []
    for i, r in enumerate(order):
        if r.game_event_type != "OUT":
            continue
        nxt = next((x for x in order[i + 1:] if x.game_event_id != r.game_event_id), None)
        if nxt is None or nxt.canonical_period != r.canonical_period or nxt.start_time_s is None:
            continue
        a, b = np.searchsorted(t, r.event_time_s, "left"), np.searchsorted(t, nxt.start_time_s, "left")
        sel = np.arange(a, b)
        dead[sel[per[sel] == r.canonical_period]] = True
        spans.append((r.event_time_s, nxt.start_time_s))
    return dead, spans


def tag_track(t, per, rows) -> tuple[np.ndarray, np.ndarray]:
    """Straight lines between the ball positions of all event rows at their eventTime, per period."""
    x, y = np.full(len(t), np.nan), np.full(len(t), np.nan)
    pts = defaultdict(list)
    for r in rows:
        b = (r.positions_snapshot or {}).get("ball") or []
        if r.event_time_s is None or r.canonical_period is None or not b or not isinstance(b[0], dict):
            continue
        bx, by = b[0].get("x"), b[0].get("y")
        if bx is None or by is None:
            continue
        pts[r.canonical_period].append((r.event_time_s, float(bx), float(by)))
    for p, v in pts.items():
        v.sort()
        tt = np.array([a for a, _, _ in v]); xx = np.array([b for _, b, _ in v]); yy = np.array([c for _, _, c in v])
        sel = (per == p) & (t >= tt[0]) & (t <= tt[-1])
        x[sel], y[sel] = np.interp(t[sel], tt, xx), np.interp(t[sel], tt, yy)
    return x, y


def carrier_track(t, per, cx, cy, in_spell, dead) -> tuple[np.ndarray, np.ndarray]:
    """Tag-based reconstruction 4b: ball = the tagged carrier's smoothed position inside OTB spells, straight lines between
    the end of one spell and the start of the next (same period, not across dead ball), missing in dead ball."""
    x, y = np.full(len(t), np.nan), np.full(len(t), np.nan)
    known = in_spell & np.isfinite(cx)
    x[known], y[known] = cx[known], cy[known]
    for p in np.unique(per):
        sel = np.flatnonzero((per == p) & ~dead)
        k = sel[known[sel]]
        if len(k) < 2:
            continue
        gap = sel[~known[sel] & (sel > k[0]) & (sel < k[-1])]
        x[gap], y[gap] = np.interp(t[gap], t[k], cx[k]), np.interp(t[gap], t[k], cy[k])
    x[dead], y[dead] = np.nan, np.nan
    return x, y


def run_match(mid: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # empty +-10 frame windows at segment edges
        return _run_match(mid)


def _run_match(mid: str) -> dict:
    c = dict(np.load(build_cache(mid)))
    t, per, fn = c["t"], c["per"], c["fn"]
    n = len(t)
    fps = float(c["fps"])
    ds = Dataset(ROOT)
    meta = ds.metadata(mid)
    roster = ds.roster(mid, meta)
    rows = ds.load_events(mid, EventVersion.ROOT)

    ours = smooth_ball(t, c["rx"], c["ry"], c["rz"], visibility=VIS_NAMES[c["rvis"]], period=per, cfg=SmootherConfig())
    sp = np.hypot(ours.vx, ours.vy)
    dv = {"ours": dv_series(t, ours.x, ours.y, per), "raw": dv_series(t, c["rx"], c["ry"], per),
          "pff": dv_series(t, c["sx"], c["sy"], per)}
    pk = peaks(dv["ours"])
    w = int(round(OFFSET_WIN_S * fps))

    # ---- 1 + 2: per kick
    kicks = []
    for k, ty in zip(c["kick_idx"], c["kick_type"]):
        same = per[max(0, k - w):k + w + 1] == per[k]
        item = dict(match=mid, type=KICK_TYPES[ty])
        for src, d in dv.items():
            seg = np.where(same, window(d, k, -w, w), np.nan)
            if np.isfinite(seg).any():
                j = int(np.nanargmax(seg)) + max(0, k - w)
                item[f"dt_{src}"] = float(t[j] - t[k])
                item[f"dvmax_{src}"] = float(np.nanmax(seg))
        for h in (3, 10, 15):
            item[f"hit{h}"] = bool(window(pk, k, -h, h).any())
        vb = np.array([np.nanmean(window(ours.vx, k, -10, -3)), np.nanmean(window(ours.vy, k, -10, -3))])
        va = np.array([np.nanmean(window(ours.vx, k, 3, 10)), np.nanmean(window(ours.vy, k, 3, 10))])
        item["speed_before"], item["speed_after"] = float(np.hypot(*vb)), float(np.hypot(*va))
        if np.isfinite(vb).all() and np.isfinite(va).all() and np.hypot(*vb) > 0.5 and np.hypot(*va) > 0.5:
            cosang = float(np.dot(vb, va) / (np.hypot(*vb) * np.hypot(*va)))
            item["angle"] = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        item["ours_missing_pm3"] = bool((window(ours.flag, k, -3, 3) == FLAG_MISSING).any())
        rawwin = window(c["rx"], k, -3, 3)
        item["raw_missing_pm3"] = bool(np.isnan(rawwin).any())
        vis = window(c["rvis"], k, -3, 3)
        item["raw_est_share_pm3"] = float((vis == 2).sum() / max((vis > 0).sum(), 1)) if (vis > 0).any() else None
        kicks.append(item)

    # ---- 4: is ballsSmoothed built from the tags?
    dead, spans = dead_mask(t, per, rows)
    tx, ty = tag_track(t, per, rows)
    pff_ok = np.isfinite(c["sx"])
    both = pff_ok & np.isfinite(tx)
    d_tag = np.hypot(tx - c["sx"], ty - c["sy"])[both]
    both_raw = np.isfinite(c["rx"]) & pff_ok
    d_raw = np.hypot(c["rx"] - c["sx"], c["ry"] - c["sy"])[both_raw]
    ours_ok = ours.flag != FLAG_MISSING
    both_ours = ours_ok & pff_ok
    d_ours = np.hypot(ours.x - c["sx"], ours.y - c["sy"])[both_ours]
    proof = dict(frames=n, both=int(both.sum()), d_tag=d_tag.astype(np.float32).tolist(),
                 d_raw_med=float(np.median(d_raw)), d_raw_within05=float((d_raw <= 0.5).mean()),
                 d_ours_med=float(np.median(d_ours)), d_ours_within05=float((d_ours <= 0.5).mean()),
                 pff_missing=int((~pff_ok).sum()), pff_missing_in_dead=int((~pff_ok & dead).sum()),
                 dead_frames=int(dead.sum()), dead_pff_missing=int((dead & ~pff_ok).sum()),
                 dead_ours_missing=int((dead & ~ours_ok).sum()), dead_raw_missing=int((dead & ~np.isfinite(c["rx"])).sum()),
                 ours_missing=int((~ours_ok).sum()), ours_missing_in_dead=int((~ours_ok & dead).sum()),
                 raw_missing=int((~np.isfinite(c["rx"])).sum()), raw_missing_in_dead=int((~np.isfinite(c["rx"]) & dead).sum()),
                 n_dead_spans=len(spans))

    # ---- 5: naive control rule with each ball source vs PFF per-frame OTB spells
    spell = {}
    for r in rows:
        if r.game_event_type == "OTB" and r.game_event_id is not None and r.game_event_id not in spell:
            spell[r.game_event_id] = (r.source_player_id, r.source_team_id)
    balls = {"pff_smoothed": (c["sx"], c["sy"]), "ours": (np.where(ours_ok, ours.x, np.nan), np.where(ours_ok, ours.y, np.nan)),
             "raw": (c["rx"], c["ry"])}
    cnt = {s: defaultdict(int) for s in balls}
    cx, cy, in_spell = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    i = 0
    for fr in ds.iter_tracking_frames(mid, roster=roster, metadata=meta):
        if fr.frame_num != fn[i]:
            raise RuntimeError(f"{mid}: frame order differs from cache at {i}")
        truth = spell.get(fr.game_event_id)
        if truth is not None:
            in_spell[i] = True
            cp = next((p for p in fr.players if p.player_id == truth[0] and p.x_smooth_m is not None), None)
            if cp is not None:
                cx[i], cy[i] = cp.x_smooth_m, cp.y_smooth_m
        pl = [(p.x_raw_m, p.y_raw_m, p.player_id, p.team_id) for p in fr.players
              if p.x_raw_m is not None and p.y_raw_m is not None]
        for s, (bx, by) in balls.items():
            k = cnt[s]
            pred = None
            if np.isfinite(bx[i]) and pl:
                d, pid, tid = min((math.hypot(px - bx[i], py - by[i]), pid, tid) for px, py, pid, tid in pl)
                if d <= CONTROL_RADIUS_M:
                    pred = (pid, tid)
            if truth is not None:
                k["spell"] += 1
                k["spell_none"] += pred is None
                k["spell_player_ok"] += pred is not None and pred[0] == truth[0]
                k["spell_team_ok"] += pred is not None and pred[1] == truth[1]
            else:
                k["nospell"] += 1
                k["nospell_pred"] += pred is not None
            k["frame_player_ok"] += (truth is None and pred is None) or (truth is not None and pred is not None and pred[0] == truth[0])
            k["frame_team_ok"] += (truth is None and pred is None) or (truth is not None and pred is not None and pred[1] == truth[1])
        i += 1
    if i != n:
        raise RuntimeError(f"{mid}: {i} frames streamed, cache has {n}")
    kx, ky = carrier_track(t, per, cx, cy, in_spell, dead)
    bk = np.isfinite(kx) & pff_ok
    dk = np.hypot(kx - c["sx"], ky - c["sy"])
    proof.update(d_carrier=dk[bk].astype(np.float32).tolist(), d_carrier_in_spell=dk[bk & in_spell].astype(np.float32).tolist(),
                 d_carrier_between=dk[bk & ~in_spell].astype(np.float32).tolist(),
                 carrier_missing=int((~np.isfinite(kx)).sum()), carrier_missing_pff_missing=int((~np.isfinite(kx) & ~pff_ok).sum()),
                 carrier_present_pff_missing=int((np.isfinite(kx) & ~pff_ok).sum()))
    deadball = {}
    for s, ok in (("pff_smoothed", pff_ok), ("ours", ours_ok), ("raw", np.isfinite(c["rx"]))):
        pred = ~ok
        deadball[s] = dict(tp=int((pred & dead).sum()), fp=int((pred & ~dead).sum()), fn=int((~pred & dead).sum()),
                           tn=int((~pred & ~dead).sum()))
    return dict(match=mid, kicks=kicks, proof=proof, control={s: dict(v) for s, v in cnt.items()}, deadball=deadball,
                counts=ours.counts)


# ---------------------------------------------------------------- report

def q(v, p):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(v, p)) if len(v) else float("nan")


def d3(v, unit="", f=".3f"):
    return f"med {q(v, 50):{f}}{unit}, p5 {q(v, 5):{f}}, p95 {q(v, 95):{f}} (n={sum(1 for x in v if x is not None and np.isfinite(x))})"


def report(res: list[dict]) -> dict:
    K = [k for r in res for k in r["kicks"]]
    out = {}
    print(f"kicks (PA/CR/SH/CL, aligned): {len(K)}")
    print("\n=== 1. time offset: frame of max velocity change within +-1 s, minus PFF eventTime (s) ===")
    for src in ("ours", "raw", "pff"):
        print(f"  {src:5}: {d3([k.get('dt_' + src) for k in K])}")
    print("  ours by type:")
    for ty in KICK_TYPES:
        print(f"    {ty}: {d3([k.get('dt_ours') for k in K if k['type'] == ty])}")
    print("  ours by match:")
    for r in res:
        print(f"    {r['match']}: {d3([k.get('dt_ours') for k in r['kicks']])}")
    dts = np.array([k["dt_ours"] for k in K if "dt_ours" in k])
    print(f"  share within +-0.1 s: {np.mean(np.abs(dts) <= 0.1):.3f}; in [-0.4, 0]: {np.mean((dts >= -0.4) & (dts <= 0)):.3f}; > 0.2 s late: {np.mean(dts > 0.2):.3f}")
    print(f"  max dv near kick (m/s): ours {d3([k.get('dvmax_ours') for k in K], f='.2f')}; pff {d3([k.get('dvmax_pff') for k in K], f='.2f')}")

    print("\n=== 2. missed kicks (no peak within +-3 frames) ===")
    for h in (3, 10, 15):
        print(f"  hit rate +-{h} frames: {np.mean([k[f'hit{h}'] for k in K]):.3f}")
    miss = [k for k in K if not k["hit3"]]
    hit = [k for k in K if k["hit3"]]
    print(f"  missed {len(miss)} of {len(K)}")
    for ty in KICK_TYPES:
        a = [k for k in K if k["type"] == ty]
        print(f"    {ty}: missed {sum(not k['hit3'] for k in a)}/{len(a)} ({100 * np.mean([not k['hit3'] for k in a]):.1f}%)")
    for lab, grp in (("missed", miss), ("hit", hit)):
        print(f"  {lab}: speed before {d3([k['speed_before'] for k in grp], f='.2f')}; after {d3([k['speed_after'] for k in grp], f='.2f')}; "
              f"|after-before| {d3([abs(k['speed_after'] - k['speed_before']) for k in grp], f='.2f')}")

    def cat(k):
        sb, sa = k["speed_before"], k["speed_after"]
        if k["ours_missing_pm3"]:
            return "our ball missing in +-3 frames"
        if not (np.isfinite(sb) and np.isfinite(sa)):
            return "speed undefined (segment edge)"
        if k.get("angle") is not None and k["angle"] > 20 and abs(sa - sb) < 3:
            return "direction change only (>20 deg, |dspeed|<3)"
        if abs(sa - sb) < 3:
            return "small change (|dspeed|<3, angle<=20)"
        return "speed change >= 3 m/s"
    for lab, grp in (("missed", miss), ("hit", hit)):
        cc = defaultdict(int)
        for k in grp:
            cc[cat(k)] += 1
        print(f"  {lab} by category: " + "; ".join(f"{a}: {b} ({100 * b / len(grp):.1f}%)" for a, b in sorted(cc.items(), key=lambda kv: -kv[1])))
    for lab, grp in (("missed", miss), ("hit", hit)):
        est = [k["raw_est_share_pm3"] for k in grp if k["raw_est_share_pm3"] is not None]
        print(f"  {lab}: raw ball missing in >=1 frame of +-3: {np.mean([k['raw_missing_pm3'] for k in grp]):.3f}; "
              f"raw samples ESTIMATED >= 50% of +-3: {np.mean([e >= 0.5 for e in est]):.3f}; any ESTIMATED: {np.mean([e > 0 for e in est]):.3f}")
    print(f"  missed with |dspeed|>=3 and no missing: those found only at +-10: "
          f"{sum(1 for k in miss if k['hit10'] and cat(k) == 'speed change >= 3 m/s')}")

    print("\n=== 4. proof check: straight lines between event ball positions vs ballsSmoothed ===")
    dt_all = np.concatenate([np.asarray(r["proof"]["d_tag"]) for r in res])
    P = {k: sum(r["proof"][k] for r in res) for k in ("pff_missing", "pff_missing_in_dead", "dead_frames", "dead_pff_missing",
                                                      "dead_ours_missing", "dead_raw_missing", "ours_missing", "ours_missing_in_dead",
                                                      "raw_missing", "raw_missing_in_dead", "frames")}
    print(f"  tag track vs ballsSmoothed: frames {len(dt_all)}; distance med {np.median(dt_all):.3f} m, p95 {np.percentile(dt_all, 95):.3f} m, "
          f"within 0.5 m {100 * np.mean(dt_all <= 0.5):.1f}%, within 0.1 m {100 * np.mean(dt_all <= 0.1):.1f}%")
    for r in res:
        d = np.asarray(r["proof"]["d_tag"])
        print(f"    {r['match']}: med {np.median(d):.3f}, p95 {np.percentile(d, 95):.3f}, <=0.5 m {100 * np.mean(d <= 0.5):.1f}%  "
              f"(raw vs smoothed: med {r['proof']['d_raw_med']:.2f}, <=0.5 m {100 * r['proof']['d_raw_within05']:.1f}%; "
              f"ours vs smoothed: med {r['proof']['d_ours_med']:.2f}, <=0.5 m {100 * r['proof']['d_ours_within05']:.1f}%)")
    print(f"  ballsSmoothed missing frames inside PFF dead-ball periods: {P['pff_missing_in_dead']}/{P['pff_missing']} "
          f"({100 * P['pff_missing_in_dead'] / P['pff_missing']:.1f}%)")
    print(f"  dead-ball frames with ball missing: smoothed {100 * P['dead_pff_missing'] / P['dead_frames']:.1f}%, ours "
          f"{100 * P['dead_ours_missing'] / P['dead_frames']:.1f}%, raw {100 * P['dead_raw_missing'] / P['dead_frames']:.1f}% (dead frames {P['dead_frames']})")
    print(f"  missing frames that are dead ball: ours {100 * P['ours_missing_in_dead'] / P['ours_missing']:.1f}%, raw {100 * P['raw_missing_in_dead'] / P['raw_missing']:.1f}%")
    for lab in ("d_carrier", "d_carrier_in_spell", "d_carrier_between"):
        d = np.concatenate([np.asarray(r["proof"][lab]) for r in res])
        print(f"  4b {lab}: frames {len(d)}, med {np.median(d):.3f} m, p95 {np.percentile(d, 95):.3f} m, within 0.5 m {100 * np.mean(d <= 0.5):.1f}%, "
              f"within 0.1 m {100 * np.mean(d <= 0.1):.1f}%")
    for r in res:
        d = np.asarray(r["proof"]["d_carrier"])
        print(f"    {r['match']}: carrier track med {np.median(d):.3f}, <=0.5 m {100 * np.mean(d <= 0.5):.1f}%")
    cm = sum(r["proof"]["carrier_missing"] for r in res); cmm = sum(r["proof"]["carrier_missing_pff_missing"] for r in res)
    cpm = sum(r["proof"]["carrier_present_pff_missing"] for r in res)
    print(f"  4b presence: carrier track missing {cm} frames, of which ballsSmoothed also missing {cmm} ({100 * cmm / cm:.1f}%); "
          f"carrier track present but ballsSmoothed missing {cpm}")
    out["proof"] = dict(P, tag_med=float(np.median(dt_all)), tag_p95=float(np.percentile(dt_all, 95)), tag_within05=float(np.mean(dt_all <= 0.5)))

    print("\n=== 5. leakage test: nearest player within 1.5 m = in control (raw players), vs PFF OTB spells ===")
    out["control"], out["deadball"] = {}, {}
    for s in ("pff_smoothed", "ours", "raw"):
        k = defaultdict(int)
        for r in res:
            for a, b in r["control"][s].items():
                k[a] += b
        tot = k["spell"] + k["nospell"]
        row = dict(spell_frames=k["spell"], player_acc_in_spells=k["spell_player_ok"] / k["spell"], team_acc_in_spells=k["spell_team_ok"] / k["spell"],
                   none_in_spells=k["spell_none"] / k["spell"], control_claimed_outside=k["nospell_pred"] / k["nospell"],
                   frame_player_acc=k["frame_player_ok"] / tot, frame_team_acc=k["frame_team_ok"] / tot)
        out["control"][s] = row
        print(f"  {s:13}: in OTB spells ({k['spell']} frames) player {100 * row['player_acc_in_spells']:.1f}%, team {100 * row['team_acc_in_spells']:.1f}%, "
              f"nobody {100 * row['none_in_spells']:.1f}%; outside spells control claimed {100 * row['control_claimed_outside']:.1f}%; "
              f"all frames: player {100 * row['frame_player_acc']:.1f}%, team {100 * row['frame_team_acc']:.1f}%")
    for s in ("pff_smoothed", "ours", "raw"):
        k = defaultdict(int)
        for r in res:
            for a, b in r["deadball"][s].items():
                k[a] += b
        prec = k["tp"] / max(k["tp"] + k["fp"], 1); rec = k["tp"] / max(k["tp"] + k["fn"], 1)
        acc = (k["tp"] + k["tn"]) / sum(k.values())
        out["deadball"][s] = dict(k, precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9), accuracy=acc)
        print(f"  dead ball = ball missing, {s:13}: precision {prec:.3f}, recall {rec:.3f}, F1 {out['deadball'][s]['f1']:.3f}, frame accuracy {acc:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    mids = [str(m) for m in DEV_MATCHES]
    if a.workers > 1:
        with multiprocessing.Pool(a.workers) as pool:
            res = pool.map(run_match, mids)
    else:
        res = [run_match(m) for m in mids]
    summary = report(res)
    OUT.mkdir(parents=True, exist_ok=True)
    slim = [dict(r, proof={k: v for k, v in r["proof"].items() if not k.startswith("d_tag") and not k.startswith("d_carrier")}) for r in res]
    (OUT / "results.json").write_text(json.dumps(dict(summary=summary, matches=slim, config=SmootherConfig().as_dict(),
                                                      control_radius_m=CONTROL_RADIUS_M), indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
