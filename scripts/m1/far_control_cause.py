"""M1 step 1 follow-up, Part 1: why is the labelled CONTROL player far from the ball? (evaluation only, read-only)

Stretches: CONTROL frames of labels v1 flagged far_control (> 10 m from our smoothed ball for > 0.3 s), grouped into runs of
one game event. Sample: 5 named stretches + 200 random (seed 7) from the flagged matches (far control > 10 % of CONTROL or in
the coordinate-disagreement list). Per stretch:
 a. median distance of PFF's player (tracking column of that playerId via the roster join) to the raw and our smoothed ball;
 b. the player nearest the raw ball (mode over frames): label, same jersey on the other team / other jersey same team;
 c. mirrored position (-x, -y) of PFF's player to the ball;
 d. roster jersey vs event-file jersey for that playerId (all matches);
 e. ball visibility (frames index) and PFF player's visibility (raw records, targeted parse);
 f. best time shift within +-5 s of PFF's player track vs the raw ball.
Classification order: time offset, mirrored, identity mix-up, ball tracking error, player tracking error, unclear.
Also: block-by-block (5 min) stability of every jersey-swap signature (is a fixed relabelling possible?).

    python scripts/m1/far_control_cause.py
"""
from __future__ import annotations

import bz2
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from football_fatigue.event_detection.config import COORDINATE_FLAGGED_MATCHES
from m1_possession import SmootherConfig, smooth_ball
from m1_possession.splits import ALL_MATCHES

R = Path("data/raw/FIFA World Cup 2022")
LAB = Path("data/processed/m1_possession/labels/v1")
FR = Path("data/processed/m1_possession/frames")
POS = Path("data/processed/event_detection/cache")
OUT = Path("data/processed/m1_possession/far_control")
NAMED = [(10502, 5492.5), (3846, 934.3), (3837, 5850.2), (10515, 5995.3), (3852, 4466.0)]
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)
FN_RE = re.compile(r'"frameNum":(\d+)')


def load(mid):
    lab = dict(np.load(LAB / f"{mid}.npz"))
    fr = dict(np.load(FR / f"{mid}.npz"))
    z = dict(np.load(sorted(POS.glob(f"{mid}*.positions.npz"))[0]))
    keys = json.loads(str(z["meta"]))["player_key"]
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    idx = np.array([li[int(f)] for f in z["frame_num"]])
    b = smooth_ball(z["t"], z["ball"][:, 0], z["ball"][:, 1], z["ball"][:, 2], visibility=VIS_NAMES[fr["ball_visibility"][idx]],
                    period=z["period"].astype(int), cfg=SmootherConfig())
    return lab, fr, z, keys, idx, b


def runs_of(lab, idx):
    far = lab["far_control"][idx]
    src = lab["src_game_event_id"][idx]
    out, i, n = [], 0, len(idx)
    while i < n:
        if far[i]:
            j = i
            while j + 1 < n and far[j + 1] and src[j + 1] == src[i]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def player_visibility(mid, frames_needed: dict) -> dict:
    """frameNum -> {side:jersey: visibility} for the needed frames only."""
    out = {}
    path = R / "Tracking Data" / f"{mid}.jsonl.bz2"
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = FN_RE.search(line)
            if not m or int(m.group(1)) not in frames_needed or int(m.group(1)) in out:
                continue
            rec = json.loads(line)
            out[int(m.group(1))] = {f"{s}:{p.get('jerseyNum')}": p.get("visibility") for s, k in (("home", "homePlayers"), ("away", "awayPlayers"))
                                    for p in rec.get(k) or []}
    return out


def analyse(mid, lab, fr, z, keys, idx, b, i, j, pvis=None):
    sl = np.arange(i, j + 1)
    pid = int(lab["player_id"][idx[i]])
    col = {int(p): c for c, p in enumerate(z["player_id"]) if p >= 0}.get(pid)
    if col is None:
        return None
    bx, by = z["ball"][sl, 0], z["ball"][sl, 1]
    px, py = z["px"][sl, col], z["py"][sl, col]
    d_raw = np.hypot(px - bx, py - by)
    d_sm = np.hypot(px - b.x[sl], py - b.y[sl])
    d_mir = np.hypot(-px - bx, -py - by)
    d_all = np.hypot(z["px"][sl] - bx[:, None], z["py"][sl] - by[:, None])
    d_all = np.where(np.isfinite(d_all), d_all, np.inf)
    near = np.argmin(d_all, axis=1)
    dn = d_all[np.arange(len(sl)), near]
    ok = np.isfinite(bx)
    nk = Counter(keys[c] for c, d in zip(near[ok], dn[ok]) if d <= 1.5)
    own = keys[col]
    top, topn = (nk.most_common(1)[0] if nk else (None, 0))
    side, jer = own.split(":")
    rel = None
    if top:
        ts, tj = top.split(":")
        rel = "same jersey, other team" if tj == jer and ts != side else "other jersey, same team" if ts == side else "other jersey, other team"
    # time shift: PFF player at t vs raw ball at t + s
    t = z["t"]
    best = (0.0, float(np.nanmedian(d_raw)) if ok.any() else np.inf)
    for s in np.arange(-5.0, 5.01, 0.2):
        k = np.clip(np.searchsorted(t, t[sl] + s), 0, len(t) - 1)
        dd = np.hypot(px - z["ball"][k, 0], py - z["ball"][k, 1])
        if np.isfinite(dd).any():
            m = float(np.nanmedian(dd))
            if m < best[1] - 1e-9:
                best = (round(float(s), 1), m)
    bv = fr["ball_visibility"][idx[sl]]
    pv = [pvis.get(int(z["frame_num"][k]), {}).get(own) for k in sl] if pvis else []
    rec = dict(match=mid, t0=round(float(t[i]), 2), dur_s=round(float(t[j] - t[i]), 2), game_event_id=int(lab["src_game_event_id"][idx[i]]),
               player_id=pid, label=own, d_raw_med=float(np.nanmedian(d_raw)) if ok.any() else None, d_sm_med=float(np.nanmedian(d_sm)),
               d_mirror_med=float(np.nanmedian(d_mir)) if ok.any() else None, nearest=top, nearest_share=topn / max(ok.sum(), 1),
               nearest_relation=rel, ball_missing=float(1 - ok.mean()), ball_estimated=float((bv == 2).mean()),
               any_player_within2=float((dn[ok] <= 2.0).mean()) if ok.any() else 0.0,
               player_estimated=(float(np.mean([v == "ESTIMATED" for v in pv])) if pv else None),
               player_missing=float(np.isnan(px).mean()), best_shift_s=best[0], best_shift_dist=best[1])
    rec["cls"] = classify(rec)
    return rec


def classify(r) -> str:
    if r["best_shift_s"] != 0 and abs(r["best_shift_s"]) >= 0.4 and r["best_shift_dist"] < 2.0:
        return "time offset"
    if r["d_mirror_med"] is not None and r["d_mirror_med"] < 2.0:
        return "mirrored coordinates"
    if r["nearest"] and r["nearest_share"] >= 0.5:
        return "identity mix-up"
    if r["ball_missing"] >= 0.3 or (r["ball_estimated"] >= 0.5 and r["any_player_within2"] < 0.5):
        return "ball tracking error"
    if (r["player_estimated"] or 0) >= 0.5 or r["player_missing"] >= 0.3:
        return "player tracking error"
    return "unclear"


def swap_stability(mid, lab, z, keys, idx):
    """For each (period, player) whose CONTROL frames mostly sit at another track: share per 5-minute block."""
    st, pl, per = lab["state"][idx], lab["player_id"][idx], lab["period"][idx]
    b = z["ball"]
    col = {int(p): c for c, p in enumerate(z["player_id"]) if p >= 0}
    d = np.hypot(z["px"] - b[:, [0]], z["py"] - b[:, [1]])
    d = np.where(np.isfinite(d), d, np.inf)
    near = np.argmin(d, axis=1)
    dn = d[np.arange(len(d)), near]
    ok = (st == 1) & (pl >= 0) & np.isfinite(b[:, 0]) & (dn < 1.5)
    t = z["t"]
    res = []
    for p in np.unique(per[ok]):
        for pid in np.unique(pl[ok & (per == p)]):
            c = col.get(int(pid))
            if c is None:
                continue
            m = ok & (per == p) & (pl == pid)
            cnt = Counter(keys[k] for k in near[m])
            n = sum(cnt.values())
            top, k = cnt.most_common(1)[0]
            if n < 60 or top == keys[c] or k / n <= 0.5:
                continue
            t0 = t[m].min()
            blocks = defaultdict(lambda: [0, 0])
            for ti, nk in zip(t[m], near[m]):
                bk = int((ti - t0) // 300)
                blocks[bk][0] += keys[nk] == top
                blocks[bk][1] += 1
            shares = [round(a / b_, 2) for _, (a, b_) in sorted(blocks.items()) if b_ >= 20]
            res.append(dict(match=mid, period=int(p), player_id=int(pid), own=keys[c], at=top, share=round(k / n, 3), frames=int(n),
                            own_frames=cnt.get(keys[c], 0), block_shares=shares,
                            stable=bool(shares) and min(shares) >= 0.8))
    return res


def main() -> int:
    rng = np.random.default_rng(7)
    summ = json.loads((LAB / "summary.json").read_text(encoding="utf-8"))
    farpct = {r["match"]: 100 * r["far_frames"] / max(r["frames"] * r["state_pct"]["CONTROL"] / 100, 1) for r in summ}
    flagged = sorted(m for m in ALL_MATCHES if farpct[m] > 10 or m in COORDINATE_FLAGGED_MATCHES)
    data, pool = {}, []
    for m in sorted(set(flagged) | {n for n, _ in NAMED}):
        data[m] = load(m)
        lab, fr, z, keys, idx, b = data[m]
        for i, j in runs_of(lab, idx):
            pool.append((m, i, j))
    named = []
    for m, t0 in NAMED:
        lab, fr, z, keys, idx, b = data[m]
        k = int(np.searchsorted(z["t"], t0 - 0.05))
        named.append(next((m, i, j) for mm, i, j in pool if mm == m and i <= k + 3 <= j + 3 and i >= k - 3))
    others = [p for p in pool if p[0] in flagged and p not in named]
    pick = [others[i] for i in rng.choice(len(others), 200, replace=False)]
    sample = named + pick
    need = defaultdict(set)
    for m, i, j in sample:
        z = data[m][2]
        need[m].update(int(f) for f in z["frame_num"][i:j + 1])
    recs = []
    for m in sorted(need):
        pv = player_visibility(m, need[m])
        for mm, i, j in sample:
            if mm == m:
                r = analyse(m, *data[m], i, j, pvis=pv)
                if r:
                    r["named"] = (mm, i, j) in named
                    recs.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"flagged matches ({len(flagged)}): {flagged}")
    print(f"far-control stretches in flagged + named matches: {len(pool)}; analysed {len(recs)} (5 named + 200 random)")
    print("\n5 named stretches:")
    for r in recs:
        if r["named"]:
            print(f"  {r['match']} @{r['t0']} ({r['dur_s']} s) {r['label']}: raw {r['d_raw_med']:.1f} m, ours {r['d_sm_med']:.1f} m, mirror "
                  f"{r['d_mirror_med']:.1f} m; nearest {r['nearest']} ({r['nearest_relation']}, {100 * r['nearest_share']:.0f}% of frames <=1.5 m); "
                  f"ball ESTIMATED {100 * r['ball_estimated']:.0f}%, missing {100 * r['ball_missing']:.0f}%; player ESTIMATED "
                  f"{100 * (r['player_estimated'] or 0):.0f}%; best shift {r['best_shift_s']} s -> {r['best_shift_dist']:.1f} m => {r['cls']}")
    print("\nclass shares (all 205):", dict(Counter(r["cls"] for r in recs)))
    print("identity relation:", dict(Counter(r["nearest_relation"] for r in recs if r["cls"] == "identity mix-up")))
    print("medians: raw dist", round(float(np.median([r['d_raw_med'] for r in recs if r['d_raw_med'] is not None])), 1),
          "| mirror", round(float(np.median([r['d_mirror_med'] for r in recs if r['d_mirror_med'] is not None])), 1),
          "| best-shift dist", round(float(np.median([r['best_shift_dist'] for r in recs])), 1),
          "| ball ESTIMATED", round(float(np.mean([r['ball_estimated'] for r in recs])), 2),
          "| player ESTIMATED", round(float(np.mean([r['player_estimated'] or 0 for r in recs])), 2))
    print("\nper match (class counts):")
    for m in sorted({r["match"] for r in recs}):
        c = Counter(r["cls"] for r in recs if r["match"] == m)
        n = sum(c.values())
        print(f"  {m} (far {farpct[m]:.1f}% of CONTROL{', coord-flagged' if m in COORDINATE_FLAGGED_MATCHES else ''}): n={n} "
              + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in c.most_common()))
    sw = []
    for m in ALL_MATCHES:
        if m not in data:
            data[m] = load(m)
        sw.extend(swap_stability(m, data[m][0], data[m][2], data[m][3], data[m][4]))
        if m not in flagged and m not in {n for n, _ in NAMED}:
            del data[m]
    stable = [s for s in sw if s["stable"]]
    print(f"\nswap signatures (player's CONTROL frames mostly at another track, >= 60 frames): {len(sw)} in {len({s['match'] for s in sw})} matches; "
          f"stable over every 5-min block (>= 80 %): {len(stable)} in {len({s['match'] for s in stable})} matches, frames {sum(s['frames'] for s in stable)}")
    for s in sorted(sw, key=lambda s: -s["frames"])[:12]:
        print("  ", s)
    (OUT / "stretches.json").write_text(json.dumps(dict(records=recs, swaps=sw, flagged=flagged), indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
