"""Read-only investigation of how PFF encodes possession (M1 step 1). No possession logic, no labels.

Per match (default: the 8 event-detection DEV_MATCHES), two streaming passes over the tracking file:

1. raw records (``audit_lib.iter_tracking``): key inventory of the tracking record, its ball/player entries and the
   embedded ``game_event`` / ``possession_event`` overlays (step 0a, 0d);
2. ingestion frames (``Dataset.iter_tracking_frames``) fed into ``synchronization.align_events`` with three copies of
   every event row whose time is replaced by startTime, eventTime and endTime (same nearest-frame rule, one-frame
   tolerance, same period). While frames stream, raw and smoothed ball arrays and per-frame player counts are recorded,
   and player positions are kept only for frames near an OTB start/event/end time.

Per-match JSON goes to data/processed/audit_m1/possession_investigation/; the pooled report is printed.

    python scripts/audit/m1_possession_investigation.py --workers 4
    python scripts/audit/m1_possession_investigation.py --aggregate
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

from football_fatigue.event_detection.config import DEV_MATCHES  # noqa: E402
from football_fatigue.ingestion import Dataset  # noqa: E402
from football_fatigue.ingestion.models import EventVersion  # noqa: E402
from football_fatigue.synchronization import MatchStatus, align_events  # noqa: E402

DEFAULT_ROOT = "data/raw/FIFA World Cup 2022"
OUT_DIR = "data/processed/audit_m1/possession_investigation"
EPS = 1e-6
MOMENTS = ("start", "event", "end")
STOP_TYPES = {"OUT", "SUB", "END", "OFF", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}


# ---------------------------------------------------------------- pass 1: raw record keys and overlays

def raw_key_pass(path: Path) -> dict:
    keys = {k: Counter() for k in ("top", "balls", "ballsSmoothed", "player", "playerSmoothed", "game_event", "possession_event")}
    types: Counter = Counter()
    n = n_rec = 0
    prev = None
    ge: dict = {}   # game_event_id -> [first_fn, last_fn, n_frames, start_frame, end_frame, type, start_time, end_time]
    pe: dict = {}   # possession_event_id -> [first_fn, n_frames, start_frame, type, game_event_id]
    for r in L.iter_tracking(path):
        n_rec += 1
        keys["top"].update(r.keys())
        if r.get("frameNum") == prev:
            continue
        prev = r.get("frameNum")
        n += 1
        for k, v in r.items():
            types[f"{k}:{type(v).__name__}"] += 1
        for b in r.get("balls") or []:
            keys["balls"].update(b.keys())
        if isinstance(r.get("ballsSmoothed"), dict):
            keys["ballsSmoothed"].update(r["ballsSmoothed"].keys())
        for side in ("homePlayers", "awayPlayers"):
            for p in r.get(side) or []:
                keys["player"].update(p.keys())
            for p in r.get(side + "Smoothed") or []:
                keys["playerSmoothed"].update(p.keys())
        fn = r.get("frameNum")
        g = r.get("game_event")
        if isinstance(g, dict):
            keys["game_event"].update(g.keys())
            gid = r.get("game_event_id")
            if gid is not None:
                gid = int(gid)
                if gid not in ge:
                    ge[gid] = [fn, fn, 0, g.get("start_frame"), g.get("end_frame"), g.get("game_event_type"), g.get("start_time"), g.get("end_time")]
                ge[gid][1] = fn
                ge[gid][2] += 1
        p = r.get("possession_event")
        if isinstance(p, dict):
            keys["possession_event"].update(p.keys())
            pid = r.get("possession_event_id")
            if pid is not None:
                pid = int(pid)
                if pid not in pe:
                    pe[pid] = [fn, 0, p.get("start_frame"), p.get("possession_event_type"), p.get("game_event_id")]
                pe[pid][1] += 1
    return dict(records=n_rec, frames=n, keys={k: dict(v) for k, v in keys.items()}, value_types=dict(types),
                overlay_game_events={str(k): v for k, v in ge.items()}, overlay_possession_events={str(k): v for k, v in pe.items()})


# ---------------------------------------------------------------- pass 2: frames through align_events

class Recorder:
    def __init__(self, need_times: np.ndarray, tol: float):
        self.need = need_times
        self.tol = tol
        self.t, self.per, self.fn = [], [], []
        self.br, self.bs = [], []          # (x, y, z) raw / smoothed, nan when missing
        self.n_raw, self.n_raw_est, self.n_sm, self.n_sm_est = [], [], [], []
        self.pos: dict[int, dict[int, tuple]] = {}    # frame_num -> player_id -> (xr, yr, xs, ys)

    def _near(self, t: float) -> bool:
        if len(self.need) == 0:
            return False
        i = np.searchsorted(self.need, t)
        return any(0 <= j < len(self.need) and abs(self.need[j] - t) <= 2 * self.tol for j in (i - 1, i))

    def wrap(self, frames):
        nan = math.nan
        for fr in frames:
            self.t.append(fr.video_time_s)
            self.per.append(fr.canonical_period or 0)
            self.fn.append(fr.frame_num)
            b = fr.ball
            f = lambda v: nan if v is None else v  # noqa: E731
            self.br.append((f(b.x_raw_m), f(b.y_raw_m), f(b.z_raw_m)) if b and b.x_raw_m is not None and b.y_raw_m is not None else (nan, nan, nan))
            self.bs.append((f(b.x_smooth_m), f(b.y_smooth_m), f(b.z_smooth_m)) if b and b.x_smooth_m is not None and b.y_smooth_m is not None else (nan, nan, nan))
            nr = nre = ns = nse = 0
            for p in fr.players:
                if p.x_raw_m is not None and p.y_raw_m is not None:
                    nr += 1
                    nre += p.visibility == "ESTIMATED"
                c = next((c for c in p.smoothed_candidates if c.x_m is not None and c.y_m is not None), None)
                if c is not None:
                    ns += 1
                    nse += c.visibility == "ESTIMATED"
            for _side, c, _j in fr.unmatched_smoothed:
                if c.x_m is not None and c.y_m is not None:
                    ns += 1
                    nse += c.visibility == "ESTIMATED"
            self.n_raw.append(nr); self.n_raw_est.append(nre); self.n_sm.append(ns); self.n_sm_est.append(nse)
            if self._near(fr.video_time_s):
                self.pos[fr.frame_num] = {p.player_id: (p.x_raw_m, p.y_raw_m, p.x_smooth_m, p.y_smooth_m)
                                          for p in fr.players if p.player_id is not None}
            yield fr

    def arrays(self):
        return (np.array(self.t), np.array(self.per), np.array(self.fn), np.array(self.br, dtype=float).reshape(-1, 3),
                np.array(self.bs, dtype=float).reshape(-1, 3))


def ball_stats(t, per, fn, b, fps) -> dict:
    present = ~np.isnan(b[:, 0])
    n = len(t)
    prev_ok = np.zeros(n, bool)
    prev_ok[1:] = (per[1:] == per[:-1]) & (fn[1:] - fn[:-1] == 1) & present[1:] & present[:-1]
    same = np.zeros(n, bool)
    z0, z1 = b[:-1, 2], b[1:, 2]
    same[1:] = (b[1:, 0] == b[:-1, 0]) & (b[1:, 1] == b[:-1, 1]) & ((z1 == z0) | (np.isnan(z1) & np.isnan(z0)))
    rep = prev_ok & same
    fresh = present & ~rep
    secs = present.sum() / fps
    return dict(frames=int(n), pct_ball=round(100 * present.mean(), 2),
                pct_repeat_of_present_pairs=round(100 * rep.sum() / max(prev_ok.sum(), 1), 2),
                pct_repeat_of_frames=round(100 * rep.mean(), 2),
                effective_hz=round(fresh.sum() / secs, 2) if secs else None,
                pct_ball_z=round(100 * (present & ~np.isnan(b[:, 2])).mean(), 2),
                pct_ball_z_nonzero=round(100 * (present & ~np.isnan(b[:, 2]) & (b[:, 2] != 0)).mean(), 2))


def gap_ball(t, per, b, t0, t1, period) -> dict:
    """Ball over frames with t0 < t < t1 of one period: missing share, path speed on >=0.2 s steps, height."""
    i0, i1 = np.searchsorted(t, t0, "right"), np.searchsorted(t, t1, "left")
    sel = np.arange(i0, i1)
    sel = sel[per[sel] == period] if len(sel) else sel
    if len(sel) == 0:
        return dict(n=0)
    bb = b[sel]
    ok = ~np.isnan(bb[:, 0])
    out = dict(n=int(len(sel)), missing=round(float(1 - ok.mean()), 4))
    ts, xy, z = t[sel][ok], bb[ok, :2], bb[ok, 2]
    if len(ts) >= 2:
        keep = [0]
        for k in range(1, len(ts)):
            if ts[k] - ts[keep[-1]] >= 0.2:
                keep.append(k)
        if len(keep) >= 2 and ts[keep[-1]] > ts[keep[0]]:
            path = float(np.sum(np.hypot(*np.diff(xy[keep], axis=0).T)))
            out["speed"] = round(path / (ts[keep[-1]] - ts[keep[0]]), 3)
    zz = z[~np.isnan(z)]
    if len(zz):
        out["z_med"], out["z_max"] = round(float(np.median(zz)), 3), round(float(zz.max()), 3)
    return out


def run_match(root: str, mid: str) -> dict:
    t_start = time.time()
    ds = Dataset(root)
    meta = ds.metadata(mid)
    roster = ds.roster(mid, meta)
    pteam = {e.player_id: e.team_id for e in roster.entries}
    path, _ = ds.tracking_source(mid)
    raw = raw_key_pass(path)

    events = ds.load_events(mid, EventVersion.ROOT)
    tol = 1.0 / meta.fps
    copies = []
    for r in events:
        for tag, tv in zip(MOMENTS, (r.start_time_s, r.event_time_s, r.end_time_s)):
            copies.append(replace(r, event_row_id=f"{r.event_row_id}|{tag}", event_time_s=tv))
    need = np.array(sorted(tv for r in events if r.game_event_type == "OTB"
                           for tv in (r.start_time_s, r.event_time_s, r.end_time_s) if tv is not None))
    rec = Recorder(need, tol)
    frames = ds.iter_tracking_frames(mid, roster=roster, metadata=meta)
    sync = align_events(copies, rec.wrap(frames), match_id=int(mid), event_version=EventVersion.ROOT, fps=meta.fps)
    t, per, fn, br, bs = rec.arrays()
    amap = {a.event_row_id: a for a in sync.alignments}
    fidx = {int(f): i for i, f in enumerate(fn)}

    both = ~np.isnan(br[:, 0]) & ~np.isnan(bs[:, 0])
    d_rs = np.hypot(br[both, 0] - bs[both, 0], br[both, 1] - bs[both, 1])
    step0 = dict(
        raw=dict(ball_stats(t, per, fn, br, meta.fps), players_per_frame=round(float(np.mean(rec.n_raw)), 3),
                 pct_players_estimated=round(100 * sum(rec.n_raw_est) / max(sum(rec.n_raw), 1), 2)),
        smoothed=dict(ball_stats(t, per, fn, bs, meta.fps), players_per_frame=round(float(np.mean(rec.n_sm)), 3),
                      pct_players_estimated=round(100 * sum(rec.n_sm_est) / max(sum(rec.n_sm), 1), 2)),
        ball_presence=dict(both=int(both.sum()), raw_only=int((~np.isnan(br[:, 0]) & np.isnan(bs[:, 0])).sum()),
                           smoothed_only=int((np.isnan(br[:, 0]) & ~np.isnan(bs[:, 0])).sum()),
                           neither=int((np.isnan(br[:, 0]) & np.isnan(bs[:, 0])).sum())),
        raw_vs_smoothed_ball_m=dict(p50=float(np.median(d_rs)) if len(d_rs) else None,
                                    p95=float(np.percentile(d_rs, 95)) if len(d_rs) else None,
                                    pct_identical=round(100 * float((d_rs == 0).mean()), 2) if len(d_rs) else None),
    )

    def frame_of(r, tag):
        a = amap.get(f"{r.event_row_id}|{tag}")
        return (a.frame_num, a.residual_s) if a is not None and a.status is MatchStatus.MATCHED else (None, None)

    # per-row table (source order)
    rows = []
    for r in events:
        ge, pe = r.attributes.get("gameEvents") or {}, r.attributes.get("possessionEvents") or {}
        rows.append(dict(i=r.source_row_index, ge=r.game_event_id, pe=r.possession_event_id, get=r.game_event_type,
                         pet=r.possession_event_type, per=r.canonical_period, st=r.start_time_s, et=r.event_time_s,
                         en=r.end_time_s, dur=r.duration_s, team=r.source_team_id, player=r.source_player_id, seq=r.sequence,
                         sp=ge.get("setpieceType"), out=ge.get("outType"), vm=ge.get("videoMissing"), ine=ge.get("initialNonEvent"),
                         ne=pe.get("nonEvent"), chw=pe.get("challengeWinnerPlayerId"), cht=pe.get("challengeType"),
                         cho=pe.get("challengeOutcomeType"), reb=pe.get("rebounderPlayerId"), rbo=pe.get("reboundOutcomeType"),
                         org=pe.get("originateType"), chw_team=pteam.get(pe.get("challengeWinnerPlayerId")),
                         frames={tag: frame_of(r, tag) for tag in MOMENTS}))
    unmatched = Counter(f"{a.event_row_id.split('|')[1]}:{a.reason.value}" for a in sync.alignments if a.status is not MatchStatus.MATCHED)

    # game-event level OTB records (rows sharing a gameEventId collapsed; first row's player/team)
    order = sorted(range(len(rows)), key=lambda k: (rows[k]["per"] or 0, rows[k]["et"] if rows[k]["et"] is not None else 1e12, rows[k]["i"]))
    otb: dict = {}
    for k in order:
        r = rows[k]
        if r["get"] != "OTB" or r["ge"] is None:
            continue
        g = otb.setdefault(r["ge"], dict(ge=r["ge"], pe=r["pe"], per=r["per"], st=r["st"], et=r["et"], en=r["en"], team=r["team"],
                                         player=r["player"], pets=[], frames=r["frames"], n_rows=0, times_differ=False))
        g["pets"].append(r["pet"]); g["n_rows"] += 1
        if (r["st"], r["en"]) != (g["st"], g["en"]):
            g["times_differ"] = True
    otb_list = sorted(otb.values(), key=lambda g: (g["per"] or 0, g["st"] if g["st"] is not None else 1e12))

    # Q2: carrier-to-ball distance at the three moments
    q2 = []
    for g in otb_list:
        for tag in MOMENTS:
            f, res = g["frames"][tag]
            item = dict(ge=g["ge"], pe=g["pe"], pet=g["pets"][0], tag=tag, t=g[{"start": "st", "event": "et", "end": "en"}[tag]], matched=f is not None)
            if f is not None and g["player"] is not None:
                i = fidx[f]
                pp = rec.pos.get(f, {}).get(g["player"])
                item["player_found"] = pp is not None
                if pp is not None:
                    for lab, bb, (px, py) in (("raw", br[i], pp[:2]), ("sm", bs[i], pp[2:])):
                        if px is not None and py is not None and not np.isnan(bb[0]):
                            item[f"d_{lab}"] = round(float(math.hypot(bb[0] - px, bb[1] - py)), 3)
                    if not np.isnan(br[i, 0]) and pp[2] is not None:
                        item["d_rawball_smplayer"] = round(float(math.hypot(br[i, 0] - pp[2], br[i, 1] - pp[3])), 3)
            q2.append(item)

    # Q4: OUT -> next event (next row with another gameEventId, in time order)
    q4 = []
    for pos, k in enumerate(order):
        r = rows[k]
        if r["get"] != "OUT":
            continue
        nxt = next((rows[j] for j in order[pos + 1:] if rows[j]["ge"] != r["ge"]), None)
        item = dict(ge=r["ge"], pe=r["pe"], st=r["st"], et=r["et"], en=r["en"], out=r["out"], team=r["team"])
        if nxt is not None:
            item.update(next_ge=nxt["ge"], next_pe=nxt["pe"], next_get=nxt["get"], next_pet=nxt["pet"], next_sp=nxt["sp"],
                        next_st=nxt["st"], next_et=nxt["et"], same_period=nxt["per"] == r["per"])
            if nxt["per"] == r["per"] and r["et"] is not None and nxt["st"] is not None:
                item["gap_s"] = round(nxt["st"] - r["et"], 3)
                item["ball_raw"] = gap_ball(t, per, br, r["et"], nxt["st"], r["per"])
                item["ball_sm"] = gap_ball(t, per, bs, r["et"], nxt["st"], r["per"])
        q4.append(item)

    # Q6: consecutive OTB game events, same period
    q6 = []
    for a, b in zip(otb_list, otb_list[1:]):
        if a["per"] != b["per"] or a["en"] is None or b["st"] is None:
            continue
        item = dict(ge=a["ge"], pe=a["pe"], next_ge=b["ge"], next_pe=b["pe"], pet=a["pets"][-1], next_pet=b["pets"][0],
                    en=a["en"], next_st=b["st"], gap_s=round(b["st"] - a["en"], 3),
                    same_team=(a["team"] == b["team"]) if a["team"] is not None and b["team"] is not None else None)
        if item["gap_s"] > 0:
            item["ball_raw"] = gap_ball(t, per, br, a["en"], b["st"], a["per"])
            item["ball_sm"] = gap_ball(t, per, bs, a["en"], b["st"], a["per"])
        q6.append(item)

    # overlay vs synchronization: aligned startTime/endTime frames against the overlay's own start_frame/end_frame
    ov = []
    first_row = {}
    for r in rows:
        first_row.setdefault(r["ge"], r)
    for gid, (first_fn, last_fn, nfr, sf, ef, typ, ost, oen) in raw["overlay_game_events"].items():
        r = first_row.get(int(gid))
        if r is None:
            ov.append(dict(ge=int(gid), in_events=False)); continue
        fs, fe = r["frames"]["start"][0], r["frames"]["end"][0]
        ov.append(dict(ge=int(gid), type=typ, first_seen_minus_start_frame=None if sf is None else first_fn - sf,
                       last_seen_minus_end_frame=None if ef is None else last_fn - ef,
                       n_frames=nfr, span=None if sf is None or ef is None else ef - sf + 1,
                       aligned_start_minus_start_frame=None if fs is None or sf is None else fs - sf,
                       aligned_end_minus_end_frame=None if fe is None or ef is None else fe - ef,
                       start_time_diff=None if ost is None or r["st"] is None else round(float(ost) - r["st"], 4)))
    overlay_frames = dict(game_event=int(raw["value_types"].get("game_event:dict", 0)),
                          possession_event=int(raw["value_types"].get("possession_event:dict", 0)), frames=raw["frames"])
    for k in ("overlay_game_events", "overlay_possession_events"):
        raw[k + "_n"] = len(raw[k])
    pe_overlay = Counter(v[3] for v in raw["overlay_possession_events"].values())
    del raw["overlay_game_events"]
    raw["overlay_possession_types"] = dict(pe_overlay)
    raw["overlay_possession_frames_per_event"] = dict(Counter(v[1] for v in raw.pop("overlay_possession_events").values()))

    return dict(ok=True, match_id=mid, seconds=round(time.time() - t_start, 1), fps=meta.fps, scan_complete=sync.scan_complete,
                fatal_error=sync.fatal_error, unmatched=dict(unmatched), step0=step0, raw_pass=raw, overlay_frames=overlay_frames,
                overlay_vs_sync=ov, rows=rows, q2=q2, q4=q4, q6=q6)


# ---------------------------------------------------------------- pooled report

def dist(v) -> str:
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "n=0"
    a = np.asarray(v, float)
    return f"n={len(a)} p5={np.percentile(a, 5):.3g} med={np.median(a):.3g} p95={np.percentile(a, 95):.3g} min={a.min():.3g} max={a.max():.3g}"


def ex(items, keys) -> str:
    return "\n".join("      e.g. " + ", ".join(f"{k}={it.get(k)}" for k in keys if k in it) for it in items[:3])


def ordered(rows):
    return sorted(rows, key=lambda r: (r["per"] or 0, r["et"] if r["et"] is not None else 1e12, r["i"]))


def aggregate(res: dict) -> None:
    ok = {m: r for m, r in sorted(res.items()) if r.get("ok")}
    print(f"matches: {sorted(ok)}  (failed: {[m for m in res if m not in ok]})")
    for m, r in ok.items():
        if not r["scan_complete"]:
            print(f"  WARNING {m}: tracking scan incomplete: {r['fatal_error']}")

    # ---- step 0
    print("\n=== STEP 0a/0d: tracking record keys (pooled frame counts, repeats of a frameNum skipped) ===")
    keys = defaultdict(Counter)
    vt = Counter()
    for r in ok.values():
        for k, c in r["raw_pass"]["keys"].items():
            keys[k].update(c)
        vt.update(r["raw_pass"]["value_types"])
    for k, c in keys.items():
        print(f"  {k}: {dict(c)}")
    print(f"  value types: {dict(vt)}")
    print(f"  records {sum(r['raw_pass']['records'] for r in ok.values())}, distinct frames {sum(r['raw_pass']['frames'] for r in ok.values())}")
    print("\n=== STEP 0c: per match (frames = ingestion frames, repeats collapsed) ===")
    hdr = "match  kind      frames  %ball  %repeat(pairs) %repeat(frames)  Hz     %z    %z!=0  players/frame %ESTIMATED"
    print(hdr)
    for m, r in ok.items():
        for kind in ("raw", "smoothed"):
            s = r["step0"][kind]
            print(f"{m:6} {kind:8} {s['frames']:7} {s['pct_ball']:6} {s['pct_repeat_of_present_pairs']:14} {s['pct_repeat_of_frames']:15} "
                  f"{s['effective_hz']:6} {s['pct_ball_z']:6} {s['pct_ball_z_nonzero']:6} {s['players_per_frame']:13} {s['pct_players_estimated']}")
        print(f"       presence {r['step0']['ball_presence']}  raw-vs-smoothed ball distance {r['step0']['raw_vs_smoothed_ball_m']}")
    print("\n=== STEP 0d: overlays ===")
    for m, r in ok.items():
        rp = r["raw_pass"]
        print(f"  {m}: frames with game_event {r['overlay_frames']['game_event']}/{r['overlay_frames']['frames']}, with possession_event "
              f"{r['overlay_frames']['possession_event']}; distinct game events {rp['overlay_game_events_n']}, possession events "
              f"{rp['overlay_possession_events_n']} {rp['overlay_possession_types']}; frames per possession_event {rp['overlay_possession_frames_per_event']}")
    ov = [o for r in ok.values() for o in r["overlay_vs_sync"] if o.get("type")]
    print(f"  overlay game events compared: {len(ov)}; types {dict(Counter(o['type'] for o in ov))}")
    print(f"   first frame seen - overlay start_frame: {dict(Counter(o['first_seen_minus_start_frame'] for o in ov).most_common(6))}")
    print(f"   last frame seen - overlay end_frame:   {dict(Counter(o['last_seen_minus_end_frame'] for o in ov).most_common(6))}")
    print(f"   frames seen == end-start+1: {sum(o['n_frames'] == o['span'] for o in ov)}/{len(ov)}")
    print(f"   aligned(startTime) frame - start_frame: {dict(Counter(o['aligned_start_minus_start_frame'] for o in ov).most_common(6))}")
    print(f"   aligned(endTime) frame - end_frame:     {dict(Counter(o['aligned_end_minus_end_frame'] for o in ov).most_common(6))}")
    print(f"   overlay start_time - event startTime (s): {dist([o['start_time_diff'] for o in ov])}")
    um = Counter()
    for r in ok.values():
        um.update(r["unmatched"])
    print(f"  unmatched alignments by moment:reason: {dict(um)}")

    rows = [dict(rr, m=m) for m, r in ok.items() for rr in r["rows"]]
    print(f"\nevent rows (root version): {len(rows)}")

    # ---- Q1
    print("\n=== Q1: startTime <= eventTime <= endTime ===")
    def q1(group_key):
        by = defaultdict(list)
        for r in rows:
            by[r[group_key]].append(r)
        for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            v = [r for r in rs if None not in (r["st"], r["et"], r["en"])]
            inside = sum(r["st"] - EPS <= r["et"] <= r["en"] + EPS for r in v)
            zero = sum(abs(r["st"] - r["et"]) < EPS and abs(r["en"] - r["et"]) < EPS for r in v)
            print(f"  {group_key}={k}: n={len(rs)} with3times={len(v)} inside={inside} ({100 * inside / max(len(v), 1):.1f}%) all-equal={zero}")
            print(f"     event-start {dist([r['et'] - r['st'] for r in v])}")
            print(f"     end-event   {dist([r['en'] - r['et'] for r in v])}")
    q1("get")
    q1("pet")
    v = [r for r in rows if None not in (r["st"], r["et"], r["en"])]
    viol = [r for r in v if not (r["st"] - EPS <= r["et"] <= r["en"] + EPS)]
    print(f"  violations: {len(viol)} by (get,pet) {dict(Counter((r['get'], r['pet']) for r in viol))}")
    print(f"   eventTime < startTime: {sum(r['et'] < r['st'] - EPS for r in viol)}, eventTime > endTime: {sum(r['et'] > r['en'] + EPS for r in viol)}")
    big = sorted(v, key=lambda r: -max(abs(r["et"] - r["st"]), abs(r["en"] - r["et"])))
    print("  largest |eventTime-startTime| or |endTime-eventTime| (top 12):")
    for r in big[:12]:
        print(f"    m={r['m']} ge={r['ge']} pe={r['pe']} {r['get']}/{r['pet']} st={r['st']} et={r['et']} en={r['en']} dur={r['dur']} sp={r['sp']} vm={r['vm']}")
    for thr in (1, 5, 10, 30):
        rs = [r for r in v if max(abs(r["et"] - r["st"]), abs(r["en"] - r["et"])) > thr]
        print(f"   > {thr:2} s: {len(rs)} rows, by (get,pet) {dict(Counter((r['get'], r['pet']) for r in rs).most_common(8))}")
    # rows that share a gameEventId: do they share start/end?
    byge = defaultdict(list)
    for r in rows:
        byge[(r["m"], r["ge"])].append(r)
    multi = [g for g in byge.values() if len(g) > 1]
    print(f"  gameEventIds with >1 row: {len(multi)} of {len(byge)}; share identical start/end: "
          f"{sum(len({(r['st'], r['en']) for r in g}) == 1 for g in multi)}; row-count dist {dict(Counter(len(g) for g in byge.values()))}")
    print(f"  type combos in multi-row game events (top): {dict(Counter(tuple(r['pet'] for r in g) for g in multi).most_common(10))}")
    print(ex([r for r in v if r["get"] == "OTB" and r["pet"] == "PA"], ("m", "ge", "pe", "get", "pet", "st", "et", "en")))

    # ---- Q2
    print("\n=== Q2: distance carrier (gameEvents.playerId) to ball at OTB startTime / eventTime / endTime (game-event level) ===")
    q2 = [dict(it, m=m) for m, r in ok.items() for it in r["q2"]]
    for tag in MOMENTS:
        it = [x for x in q2 if x["tag"] == tag]
        print(f"  {tag}: game events {len(it)}, frame matched {sum(x['matched'] for x in it)}, player found {sum(bool(x.get('player_found')) for x in it)}")
        for lab, name in (("d_sm", "smoothed ball vs smoothed player"), ("d_raw", "raw ball vs raw player"), ("d_rawball_smplayer", "raw ball vs smoothed player")):
            d = [x[lab] for x in it if lab in x]
            if d:
                a = np.asarray(d)
                print(f"    {name}: {dist(d)}  <=1m {100 * (a <= 1).mean():.1f}%  <=2m {100 * (a <= 2).mean():.1f}%  >5m {100 * (a > 5).mean():.1f}%")
    for pet in ("PA", "IT", "CH", "RE", "SH", "CR", "CL", "BC", "TC"):
        print(f"   eventTime d_sm for first pet={pet}: {dist([x.get('d_sm') for x in q2 if x['tag'] == 'event' and x['pet'] == pet])}")
        print(f"   eventTime d_raw for first pet={pet}: {dist([x.get('d_raw') for x in q2 if x['tag'] == 'event' and x['pet'] == pet])}")
    far = [x for x in q2 if x["tag"] == "event" and x.get("d_sm", 0) > 5]
    print(ex(far, ("m", "ge", "pe", "pet", "tag", "t", "d_sm", "d_raw")))

    # ---- Q3
    print("\n=== Q3: rebounds (RE) and challenges (CH) ===")
    for pet in ("RE", "CH"):
        rs = [r for r in rows if r["pet"] == pet]
        shared = [r for r in rs if len(byge[(r["m"], r["ge"])]) > 1]
        print(f"  {pet}: rows {len(rs)}; gameEventType {dict(Counter(r['get'] for r in rs))}; teamId null {sum(r['team'] is None for r in rs)}; "
              f"playerId null {sum(r['player'] is None for r in rs)}")
        print(f"     gameEventId shared with another row: {len(shared)}; sibling types {dict(Counter(tuple(x['pet'] for x in byge[(r['m'], r['ge'])]) for r in shared).most_common(6))}")
        print(f"     duration {dist([r['dur'] for r in rs])}")
        if pet == "CH":
            print(f"     challengeType {dict(Counter(r['cht'] for r in rs))}; outcome {dict(Counter(r['cho'] for r in rs))}; winner present {sum(r['chw'] is not None for r in rs)}")
            print(f"     winner team == row teamId: {sum(r['chw_team'] == r['team'] for r in rs if r['chw_team'] is not None)}/{sum(r['chw_team'] is not None for r in rs)}")
        else:
            print(f"     reboundOutcomeType {dict(Counter(r['rbo'] for r in rs))}; originateType {dict(Counter(r['org'] for r in rs))}; rebounder==playerId "
                  f"{sum(r['reb'] == r['player'] for r in rs)}/{len(rs)}")
        # next OTB game event (different gameEventId) in time order
        cnt = Counter()
        exs = []
        for m in ok:
            o = ordered([r for r in rows if r["m"] == m])
            for p, r in enumerate(o):
                if r["pet"] != pet:
                    continue
                nxt = next((x for x in o[p + 1:] if x["get"] == "OTB" and x["ge"] != r["ge"]), None)
                if nxt is None or nxt["per"] != r["per"] or r["team"] is None or nxt["team"] is None:
                    cnt["undetermined"] += 1; continue
                k = "same" if nxt["team"] == r["team"] else "other"
                cnt[k] += 1
                if pet == "CH" and r["chw_team"] is not None:
                    cnt["winner_team==next_team" if r["chw_team"] == nxt["team"] else "winner_team!=next_team"] += 1
                if len(exs) < 3:
                    exs.append(dict(m=m, ge=r["ge"], pe=r["pe"], st=r["st"], et=r["et"], en=r["en"], team=r["team"], next_ge=nxt["ge"], next_team=nxt["team"], next_pet=nxt["pet"]))
        print(f"     next OTB game event team vs this row's team: {dict(cnt)}")
        print(ex(exs, ("m", "ge", "pe", "st", "et", "en", "team", "next_ge", "next_pet", "next_team")))

    # ---- Q4
    print("\n=== Q4: OUT events ===")
    q4 = [dict(it, m=m) for m, r in ok.items() for it in r["q4"]]
    print(f"  OUT rows {len(q4)}; outType {dict(Counter(x['out'] for x in q4))}")
    print(f"  OUT own duration (en-st) {dist([x['en'] - x['st'] for x in q4 if x['en'] is not None and x['st'] is not None])}")
    print(f"  next event gameEventType {dict(Counter(x.get('next_get') for x in q4))}; setpieceType {dict(Counter(x.get('next_sp') for x in q4))}")
    for o in sorted({x["out"] for x in q4}, key=str):
        xs = [x for x in q4 if x["out"] == o]
        print(f"   outType={o}: n={len(xs)} next setpiece {dict(Counter(x.get('next_sp') for x in xs))}; gap OUT eventTime->next startTime {dist([x.get('gap_s') for x in xs])}")
    g = [x for x in q4 if "gap_s" in x]
    print(f"  gap all {dist([x['gap_s'] for x in g])}; next in another period: {sum(not x.get('same_period', True) for x in q4)}")
    print(f"  ball missing share in gap: raw {dist([x['ball_raw'].get('missing') for x in g])}; smoothed {dist([x['ball_sm'].get('missing') for x in g])}")
    print(f"  gaps fully missing (>=99%): raw {sum((x['ball_raw'].get('missing') or 0) >= .99 for x in g)}/{len(g)}, smoothed {sum((x['ball_sm'].get('missing') or 0) >= .99 for x in g)}/{len(g)}; "
          f"no missing at all: raw {sum(x['ball_raw'].get('missing') == 0 for x in g)}, smoothed {sum(x['ball_sm'].get('missing') == 0 for x in g)}")
    pooled_raw = sum(x["ball_raw"]["n"] * x["ball_raw"].get("missing", 0) for x in g) / max(sum(x["ball_raw"]["n"] for x in g), 1)
    pooled_sm = sum(x["ball_sm"]["n"] * x["ball_sm"].get("missing", 0) for x in g) / max(sum(x["ball_sm"]["n"] for x in g), 1)
    print(f"  frame-weighted missing share in OUT gaps: raw {100 * pooled_raw:.1f}%, smoothed {100 * pooled_sm:.1f}%")
    print(ex(g, ("m", "ge", "pe", "out", "st", "et", "en", "next_ge", "next_sp", "next_st", "gap_s")))

    # ---- Q5
    print("\n=== Q5: sequence changes ===")
    cats = Counter()
    team_change_no_seq = Counter()
    exs = defaultdict(list)
    seq_null = Counter((r["get"]) for r in rows if r["seq"] is None)
    for m in ok:
        o = ordered([r for r in rows if r["m"] == m])
        prev_seq = None; last_team = None; stop_between = False; prev_per = None
        for r in o:
            if r["per"] != prev_per:
                stop_between = True; prev_per = r["per"]
            if r["get"] in STOP_TYPES:
                stop_between = True
            if r["seq"] is None:
                continue
            team_now = r["team"] if r["get"] == "OTB" else None
            if prev_seq is not None and r["seq"] != prev_seq:
                tchg = team_now is not None and last_team is not None and team_now != last_team
                stop = stop_between or (r["sp"] not in (None, "O"))
                k = ("team_change" if tchg else "no_team_change") + "+" + ("after_stop_or_setpiece" if stop else "open_play")
                cats[k] += 1
                cats["seq_decreases"] += r["seq"] < prev_seq
                if len(exs[k]) < 3:
                    exs[k].append(dict(m=m, ge=r["ge"], pe=r["pe"], get=r["get"], pet=r["pet"], st=r["st"], et=r["et"], seq=f"{prev_seq}->{r['seq']}", sp=r["sp"]))
            elif prev_seq is not None and team_now is not None and last_team is not None and team_now != last_team:
                team_change_no_seq[r["pet"]] += 1
                if len(exs["team_change_same_seq"]) < 3:
                    exs["team_change_same_seq"].append(dict(m=m, ge=r["ge"], pe=r["pe"], pet=r["pet"], st=r["st"], et=r["et"], seq=r["seq"]))
            if r["seq"] != prev_seq:
                stop_between = False
            prev_seq = r["seq"]
            if team_now is not None:
                last_team = team_now
    print(f"  rows with null sequence by gameEventType: {dict(seq_null)}")
    print(f"  sequence changes: {dict(cats)}")
    print(f"  team changes (consecutive OTB rows) WITHOUT sequence change, by possessionEventType: {dict(team_change_no_seq)} total {sum(team_change_no_seq.values())}")
    for k, v in exs.items():
        print(f"   [{k}]")
        print(ex(v, ("m", "ge", "pe", "get", "pet", "st", "et", "seq", "sp")))

    # ---- Q6
    print("\n=== Q6: gap between one OTB game event endTime and the next OTB startTime (same period) ===")
    q6 = [dict(it, m=m) for m, r in ok.items() for it in r["q6"]]
    gaps = [x["gap_s"] for x in q6]
    a = np.asarray(gaps)
    print(f"  pairs {len(q6)}; gap {dist(gaps)}; <0 (overlap) {(a < -EPS).sum()}; ==0 {(np.abs(a) <= EPS).sum()}; >0 {(a > EPS).sum()}; >2s {(a > 2).sum()}; >10s {(a > 10).sum()}")
    for st in (True, False, None):
        xs = [x for x in q6 if x["same_team"] is st]
        print(f"  same_team={st}: n={len(xs)} gap {dist([x['gap_s'] for x in xs])}")
    pos = [x for x in q6 if x["gap_s"] > 0]
    for lab in ("ball_sm", "ball_raw"):
        print(f"  {lab} (gaps > 0): missing {dist([x[lab].get('missing') for x in pos])}; speed m/s {dist([x[lab].get('speed') for x in pos])}; "
              f"median z {dist([x[lab].get('z_med') for x in pos])}; max z {dist([x[lab].get('z_max') for x in pos])}")
    for st in (True, False):
        xs = [x for x in pos if x["same_team"] is st]
        print(f"   same_team={st}: smoothed speed {dist([x['ball_sm'].get('speed') for x in xs])}; max z {dist([x['ball_sm'].get('z_max') for x in xs])}")
    print(f"  by last possession type of the first game event (gap s):")
    for pet, n in Counter(x["pet"] for x in q6).most_common(9):
        xs = [x for x in q6 if x["pet"] == pet]
        print(f"   {pet}: n={n} gap {dist([x['gap_s'] for x in xs])}; same team {sum(x['same_team'] is True for x in xs)}; smoothed speed {dist([x.get('ball_sm', {}).get('speed') for x in xs])}")
    print(ex([x for x in q6 if x["gap_s"] > 0.5], ("m", "ge", "pe", "pet", "en", "next_ge", "next_pe", "next_pet", "next_st", "gap_s", "same_team")))
    print(ex([x for x in q6 if x["gap_s"] < -EPS], ("m", "ge", "pe", "pet", "en", "next_ge", "next_pe", "next_pet", "next_st", "gap_s", "same_team")))

    # ---- Q7
    print("\n=== Q7: flags ===")
    for f, name in (("vm", "gameEvents.videoMissing"), ("ne", "possessionEvents.nonEvent"), ("ine", "gameEvents.initialNonEvent")):
        tr = [r for r in rows if r[f] is True]
        print(f"  {name}: True {len(tr)}, False {sum(r[f] is False for r in rows)}, null/absent {sum(r[f] is None for r in rows)}; True by (get,pet) {dict(Counter((r['get'], r['pet']) for r in tr))}")
        print(ex(tr, ("m", "ge", "pe", "get", "pet", "st", "et", "en")))


def main() -> int:
    argv = sys.argv[1:]
    if "--root" not in argv:
        argv += ["--root", DEFAULT_ROOT]
    if "--matches" not in argv:
        argv += ["--matches", *map(str, DEV_MATCHES)]
    if "--out" not in argv:
        argv += ["--out", OUT_DIR]
    return L.cli_main(__doc__, OUT_DIR, run_match, aggregate, argv)


if __name__ == "__main__":
    raise SystemExit(main())
