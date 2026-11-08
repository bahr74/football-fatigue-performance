"""M1 step 1 (A): build the per-frame answer key from PFF tags for the dev matches and run the checks.

Frames per game event come from the tracking ``game_event`` start_frame/end_frame (read from the raw records); rows
without one are aligned with ``synchronization.align_events`` (startTime/eventTime/endTime, one-frame tolerance). The
checks compare labels with our tag-free smoothed raw ball (``m1_possession.smooth_ball``, default) using the scoring
tolerance ``m1_labels.TOLERANCE_S`` (+-0.3 s).

    python scripts/m1/build_labels.py --workers 4

Labels: data/processed/m1_possession/labels/<match>.npz; checks: .../labels/checks.json; plots: reports/m1_possession/labels/.
"""
from __future__ import annotations

import argparse
import bz2
import json
import multiprocessing
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_smoother import ROOT, build_cache, dv_series, peaks
from football_fatigue.event_detection.config import DEV_MATCHES
from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.synchronization import MatchStatus, align_events
from m1_labels import CONTROL, DEAD, LOOSE, STATE_NAMES, TOLERANCE_S, TRAVELLING, UNKNOWN, Row, label_match
from m1_possession import FLAG_MISSING, SmootherConfig, smooth_ball

OUT = Path("data/processed/m1_possession/labels")
PLOTS = Path("reports/m1_possession/labels")
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)
MAXP = 30
GE_RE = re.compile(r'"game_event_id":\s*([0-9.]+)')


def overlay_spans(path: Path) -> dict:
    """game_event_id -> (start_frame, end_frame) from the tracking overlay (first record of each id is parsed)."""
    spans, last = {}, None
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = GE_RE.search(line)
            if not m:
                last = None
                continue
            gid = int(float(m.group(1)))
            if gid == last or gid in spans:
                last = gid
                continue
            last = gid
            g = json.loads(line).get("game_event") or {}
            spans[gid] = (g.get("start_frame"), g.get("end_frame"), g.get("game_event_type"))
    return spans


def to_row(r) -> Row:
    ge, pe = r.attributes.get("gameEvents") or {}, r.attributes.get("possessionEvents") or {}
    return Row(idx=r.source_row_index, ge=r.game_event_id, pe=r.possession_event_id, get=r.game_event_type,
               pet=r.possession_event_type, period=r.canonical_period, st=r.start_time_s, et=r.event_time_s, en=r.end_time_s,
               team=r.source_team_id, player=r.source_player_id, setpiece=ge.get("setpieceType"), out_type=ge.get("outType"),
               video_missing=bool(ge.get("videoMissing")), non_event=bool(pe.get("nonEvent")),
               initial_non_event=bool(ge.get("initialNonEvent")), challenge_type=pe.get("challengeType"),
               carry_outcome=pe.get("ballCarryOutcome"))


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else None


def run_match(mid: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run_match(mid)


def _run_match(mid: str) -> dict:
    c = dict(np.load(build_cache(mid)))
    t, per, fn = c["t"], c["per"], c["fn"]
    n = len(t)
    fidx = {int(f): i for i, f in enumerate(fn)}
    ds = Dataset(ROOT)
    meta = ds.metadata(mid)
    roster = ds.roster(mid, meta)
    events = ds.load_events(mid, EventVersion.ROOT)
    rows = [to_row(r) for r in events]
    by_idx = {r.idx: r for r in rows}

    # frames: overlay spans (raw records) and aligned times (synchronization)
    path, _ = ds.tracking_source(mid)
    ov = overlay_spans(path)
    copies = [replace(r, event_row_id=f"{r.event_row_id}|{tag}", event_time_s=tv)
              for r in events for tag, tv in (("start", r.start_time_s), ("event", r.event_time_s), ("end", r.end_time_s))]
    geid = np.full(n, -1, np.int64)
    pid = np.full((n, MAXP), -1, np.int64)
    px, py = np.full((n, MAXP), np.nan, np.float32), np.full((n, MAXP), np.nan, np.float32)
    counter = [0]

    def wrap(frames):
        for fr in frames:
            i = counter[0]
            if fr.frame_num != fn[i]:
                raise RuntimeError(f"{mid}: frame order differs from cache at {i}")
            geid[i] = fr.game_event_id if fr.game_event_id is not None else -1
            k = 0
            for p in fr.players:
                if p.player_id is not None and p.x_raw_m is not None and p.y_raw_m is not None and k < MAXP:
                    pid[i, k], px[i, k], py[i, k] = p.player_id, p.x_raw_m, p.y_raw_m
                    k += 1
            counter[0] += 1
            yield fr

    sync = align_events(copies, wrap(ds.iter_tracking_frames(mid, roster=roster, metadata=meta)), match_id=int(mid),
                        event_version=EventVersion.ROOT, fps=meta.fps)
    if not sync.scan_complete or counter[0] != n:
        raise RuntimeError(f"{mid}: incomplete scan ({sync.fatal_error}), {counter[0]} of {n} frames")
    row_frames: dict = defaultdict(dict)
    for a in sync.alignments:
        rid, tag = a.event_row_id.rsplit("|", 1)
        idx = int(a.source_row_index)
        row_frames[idx][tag] = fidx.get(a.frame_num) if a.status is MatchStatus.MATCHED else None
    spans, agree = {}, Counter()
    span_rows = defaultdict(list)
    for r in rows:
        span_rows[r.ge].append(r)
    for g, (sf, ef, typ) in ov.items():
        if sf is None or ef is None or int(sf) not in fidx or int(ef) not in fidx:
            agree["overlay_frame_not_in_tracking"] += 1
            continue
        i0, i1 = fidx[int(sf)], fidx[int(ef)]
        spans[g] = (i0, i1)
        rs = span_rows.get(g)
        if not rs:
            agree["overlay_without_event_row"] += 1
            continue
        a0 = [row_frames[r.idx].get("start") for r in rs if row_frames[r.idx].get("start") is not None]
        a1 = [row_frames[r.idx].get("end") for r in rs if row_frames[r.idx].get("end") is not None]
        if a0:
            d0 = min(a0) - i0
            agree["start_equal" if d0 == 0 else "start_within1" if abs(d0) == 1 else "start_differs"] += 1
        if a1:
            d1 = max(a1) - i1
            agree["end_equal" if d1 == 0 else "end_within1" if abs(d1) == 1 else "end_differs"] += 1
        else:
            agree["end_time_missing"] += 1
    spell_ges = {r.ge for r in rows if r.get in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF")}
    agree["spells_from_overlay"] = sum(1 for g in spell_ges if g in spans)
    agree["spells_aligned_fallback"] = sum(1 for g in spell_ges if g not in spans)

    L = label_match(t, per, rows, spans, row_frames)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{mid}.npz", frame_num=fn, t=t, period=per, state=L.state, player_id=L.player, team_id=L.team,
                        score_mask=L.score_mask, src_game_event_id=L.src_ge, src_possession_event_id=L.src_pe, rule=L.rule,
                        rule_names=np.array(L.rule_names), state_names=np.array(STATE_NAMES), tolerance_s=TOLERANCE_S)

    # ---- checks
    inp = per > 0
    st = L.state
    res = dict(match=mid, home=meta.home_team_id, away=meta.away_team_id, home_name=meta.home_team_name, away_name=meta.away_team_name,
               frames=int(inp.sum()), fps=meta.fps, agree=dict(agree))
    res["state_pct"] = {STATE_NAMES[s]: 100 * float((st[inp] == s).mean()) for s in range(5)}
    res["one_state_per_frame"] = bool(np.isin(st[inp], [0, 1, 2, 3, 4]).all())
    res["mask_pct"] = 100 * float(L.score_mask[inp].mean())
    res["in_play_pct"] = 100 * float((st[inp] != DEAD).mean())
    res["rule_frames"] = {L.rule_names[k]: int(v) for k, v in Counter(L.rule[inp].tolist()).items()}
    # 3. agreement with the overlay
    otb_ov = np.isin(geid, list(spell_ges))
    ctrl = st == CONTROL
    res["ctrl_with_otb_overlay"] = 100 * float(otb_ov[ctrl].mean())
    res["otb_overlay_labelled_control"] = 100 * float(ctrl[otb_ov].mean())
    res["otb_overlay_not_control_by_rule"] = {L.rule_names[k]: int(v) for k, v in Counter(L.rule[otb_ov & ~ctrl].tolist()).most_common(8)}
    other = np.where(geid < 0, "no overlay", "other game event")
    res["control_without_otb_overlay"] = dict(Counter(other[ctrl & ~otb_ov].tolist()))
    res["control_other_spell_overlay"] = int((ctrl & otb_ov & (geid != L.src_ge)).sum())
    # 4. our smoothed raw ball
    ours = smooth_ball(t, c["rx"], c["ry"], c["rz"], visibility=VIS_NAMES[c["rvis"]], period=per, cfg=SmootherConfig())
    ball = ours.flag != FLAG_MISSING
    speed = np.hypot(ours.vx, ours.vy)
    res["ball_present_pct_by_state"] = {STATE_NAMES[s]: 100 * float(ball[inp & (st == s)].mean()) if (inp & (st == s)).any() else None for s in range(5)}
    res["speed_by_state"] = {STATE_NAMES[s]: (q(speed[ball & (st == s)], 50), q(speed[ball & (st == s)], 95)) for s in range(5)}
    cf = np.flatnonzero(ctrl & ball & (L.player >= 0))
    col = (pid[cf] == L.player[cf][:, None])
    has = col.any(axis=1)
    j = col.argmax(axis=1)
    dist = np.full(n, np.nan)
    dist[cf[has]] = np.hypot(px[cf[has], j[has]] - ours.x[cf[has]], py[cf[has], j[has]] - ours.y[cf[has]])
    d = dist[np.isfinite(dist)]
    res["control_dist"] = dict(n=int(len(d)), med=q(d, 50), p95=q(d, 95), within_1m=100 * float((d <= 1).mean()), within_2m=100 * float((d <= 2).mean()),
                               player_not_found=int((~has).sum()))
    tolf = int(round(TOLERANCE_S * meta.fps))
    far = np.isfinite(dist) & (dist > 10)
    runs, i = [], 0
    while i < n:
        if far[i]:
            k = i
            while k + 1 < n and far[k + 1] and L.src_ge[k + 1] == L.src_ge[i]:
                k += 1
            if k - i + 1 > tolf:
                runs.append(dict(ge=int(L.src_ge[i]), t0=round(float(t[i]), 3), dur_s=round(float(t[k] - t[i]), 3),
                                 med_dist=round(float(np.median(dist[i:k + 1])), 2)))
            i = k + 1
        else:
            i += 1
    res["far_control_runs"] = runs
    pk = peaks(dv_series(t, ours.x, ours.y, per))
    kicks = Counter()
    for r in rows:
        if r.pet in ("PA", "CR", "SH", "CL"):
            f = row_frames[r.idx].get("event")
            if f is None:
                kicks[(r.pet, "unaligned")] += 1
                continue
            hit = bool(pk[max(0, f - tolf):f + tolf + 1].any())
            kicks[(r.pet, hit)] += 1
    res["kick_at_release"] = {f"{a}|{b}": v for (a, b), v in kicks.items()}
    # 5. team possession
    tp = Counter()
    for s in (CONTROL, TRAVELLING):
        for tm, v in Counter(L.team[inp & (st == s)].tolist()).items():
            tp[tm] += v
    res["team_possession_frames"] = {str(k): v for k, v in tp.items()}
    # 6. odd cases
    log = L.log
    ex = lambda g: [dict(ge=r.ge, pe=r.pe, get=r.get, pet=r.pet, st=r.st, et=r.et, en=r.en) for r in span_rows.get(g, [])][:2]  # noqa: E731
    odd = {}
    odd["stoppage_next_no_setpiece"] = [dict(out_ge=a, next_ge=b, next_setpiece=sp, rows=ex(b)) for a, b, sp in log.get("stoppage_next_no_setpiece", [])]
    odd["null_player_spells"] = [dict(ge=s.ge, rule=s.rule, pets=[r.pet for r in s.rows], rows=ex(s.ge)) for s in L.spells if s.player is None]
    odd["long_spells_10s"] = [dict(ge=s.ge, dur=round(max(r.en or 0 for r in s.rows) - min(r.st or 0 for r in s.rows), 2),
                                   vm=any(r.video_missing for r in s.rows), pets=[r.pet for r in s.rows], rows=ex(s.ge))
                              for s in L.spells if max(r.en or 0 for r in s.rows) - min(r.st or 0 for r in s.rows) > 10]
    odd["overlaps"] = [dict(ge=g, frames=k, rows=ex(g)) for g, k in log.get("overlaps", [])]
    odd["restart_without_release"] = [dict(ge=g, setpiece=sp, last=lp, rows=ex(g)) for g, sp, lp in log.get("restart_without_release", [])]
    odd["out_without_restart"] = log.get("out_without_restart", [])
    odd["spell_no_frames"] = log.get("spell_no_frames", [])
    unk = inp & (st == UNKNOWN)
    odd["unknown_frames_by_rule"] = {L.rule_names[k]: int(v) for k, v in Counter(L.rule[unk].tolist()).items()}
    uex = defaultdict(list)
    for g, r_ in zip(L.src_ge[unk], L.rule[unk]):
        nm = L.rule_names[r_]
        if len(uex[nm]) < 3 and g not in [e["ge"] for e in uex[nm]]:
            uex[nm].append(dict(ge=int(g), rows=ex(int(g))))
    odd["unknown_examples"] = dict(uex)
    res["odd"] = odd
    res["restart_setup_s"] = log.get("restart_setup_s", [])
    res["spell_rules"] = dict(Counter(s.rule for s in L.spells))
    res["tc_outcomes"] = dict(Counter(str((r.pet, r.carry_outcome)) for r in rows if r.pet in ("TC", "BC")))
    return res


# ---------------------------------------------------------------- report and plots

def report(res: list[dict]) -> None:
    f1 = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    print("CHECK 1 – state % of in-period frames (one state per frame: "
          f"{all(r['one_state_per_frame'] for r in res)})")
    print("| match | frames | DEAD | CONTROL | TRAVELLING | LOOSE | UNKNOWN | score_mask % |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res:
        s = r["state_pct"]
        print(f"| {r['match']} | {r['frames']} | {f1(s['DEAD'])} | {f1(s['CONTROL'])} | {f1(s['TRAVELLING'])} | {f1(s['LOOSE'])} | {f1(s['UNKNOWN'])} | {f1(r['mask_pct'])} |")
    tot = sum(r["frames"] for r in res)
    pooled = {k: sum(r["state_pct"][k] * r["frames"] for r in res) / tot for k in STATE_NAMES}
    print(f"| all | {tot} | " + " | ".join(f1(pooled[k]) for k in STATE_NAMES) + f" | {f1(sum(r['mask_pct'] * r['frames'] for r in res) / tot)} |")
    print("\nCHECK 2 – ball in play % (non-DEAD):", {r["match"]: round(r["in_play_pct"], 1) for r in res},
          "flag (outside 55-65):", [r["match"] for r in res if not 55 <= r["in_play_pct"] <= 65])
    print("\nframe mapping (overlay start/end vs aligned startTime/endTime):")
    ag = Counter()
    for r in res:
        ag.update(r["agree"])
    print(" ", dict(ag))
    print("\nCHECK 3 – agreement with tracking game_event overlay")
    for r in res:
        print(f"  {r['match']}: CONTROL with OTB overlay {r['ctrl_with_otb_overlay']:.1f}%; OTB overlay labelled CONTROL "
              f"{r['otb_overlay_labelled_control']:.1f}%; CONTROL in another spell's overlay {r['control_other_spell_overlay']}")
    nc, cw = Counter(), Counter()
    for r in res:
        nc.update(r["otb_overlay_not_control_by_rule"]); cw.update(r["control_without_otb_overlay"])
    print("  OTB overlay frames not CONTROL, by rule:", dict(nc.most_common(10)))
    print("  CONTROL frames without OTB overlay:", dict(cw))
    print("\nCHECK 4 – sanity with our smoothed raw ball")
    for r in res:
        d = r["control_dist"]
        print(f"  {r['match']}: CONTROL player-ball distance med {d['med']:.2f} m, p95 {d['p95']:.2f} m, <=1 m {d['within_1m']:.1f}%, <=2 m {d['within_2m']:.1f}% "
              f"(n={d['n']}, player not in frame {d['player_not_found']}); far runs (>10 m for >0.3 s): {len(r['far_control_runs'])}")
    for s in STATE_NAMES:
        meds = [r["speed_by_state"][s][0] for r in res if r["speed_by_state"][s][0] is not None]
        p95 = [r["speed_by_state"][s][1] for r in res if r["speed_by_state"][s][1] is not None]
        bp = [r["ball_present_pct_by_state"][s] for r in res if r["ball_present_pct_by_state"][s] is not None]
        print(f"  speed {s}: median of match medians {np.median(meds) if meds else float('nan'):.2f} m/s, median p95 {np.median(p95) if p95 else float('nan'):.2f}; "
              f"ball present {np.mean(bp) if bp else float('nan'):.1f}% of frames")
    kk = Counter()
    for r in res:
        kk.update(r["kick_at_release"])
    for pet in ("PA", "CR", "SH", "CL"):
        h, m_ = kk.get(f"{pet}|True", 0), kk.get(f"{pet}|False", 0)
        print(f"  kick within +-{TOLERANCE_S} s of {pet} release: {h}/{h + m_} ({100 * h / max(h + m_, 1):.1f}%), unaligned {kk.get(f'{pet}|unaligned', 0)}")
    h = sum(v for k, v in kk.items() if k.endswith("True")); m_ = sum(v for k, v in kk.items() if k.endswith("False"))
    print(f"  all releases: {h}/{h + m_} ({100 * h / max(h + m_, 1):.1f}%)")
    runs = [dict(x, match=r["match"]) for r in res for x in r["far_control_runs"]]
    print(f"  far CONTROL runs total {len(runs)}; durations {q([x['dur_s'] for x in runs], 50)} s median; examples: {runs[:3]}")
    print("\nCHECK 5 – team possession % (CONTROL + TRAVELLING frames by team)")
    for r in res:
        tp = r["team_possession_frames"]
        tot = sum(v for k, v in tp.items() if k != "-1")
        print(f"  {r['match']}: {r['home_name']} {100 * tp.get(str(r['home']), 0) / tot:.1f}% – {r['away_name']} {100 * tp.get(str(r['away']), 0) / tot:.1f}%"
              f"  (frames without team: {tp.get('-1', 0)})")
    print("\nrestart_setup duration (release eventTime - spell startTime, s) by setpieceType:")
    rs = defaultdict(list)
    for r in res:
        for sp, d, _g in r["restart_setup_s"]:
            rs[sp].append(d)
    for sp, v in sorted(rs.items()):
        print(f"  {sp}: n={len(v)} med {q(v, 50):.2f}, p5 {q(v, 5):.2f}, p95 {q(v, 95):.2f}, max {max(v):.2f}")
    print("\nspell rules:", dict(sum((Counter(r["spell_rules"]) for r in res), Counter())))
    print("TC/BC (type, ballCarryOutcome):", dict(sum((Counter(r["tc_outcomes"]) for r in res), Counter())))
    rf = Counter()
    for r in res:
        rf.update(r["rule_frames"])
    print("frames by rule:", dict(rf.most_common()))
    print("\nCHECK 6 – odd cases")
    for k in ("stoppage_next_no_setpiece", "null_player_spells", "long_spells_10s", "overlaps", "restart_without_release",
              "out_without_restart", "spell_no_frames"):
        items = [dict(x, match=r["match"]) if isinstance(x, dict) else dict(ge=x, match=r["match"]) for r in res for x in r["odd"][k]]
        print(f"  {k}: {len(items)}")
        if k == "null_player_spells":
            print(f"     by rule {dict(Counter(x['rule'] for x in items))}; pets {dict(Counter(tuple(x['pets']) for x in items).most_common(5))}")
        if k == "long_spells_10s":
            print(f"     videoMissing {sum(x['vm'] for x in items)}; durations med {q([x['dur'] for x in items], 50)} max {max([x['dur'] for x in items], default=None)}")
        if k == "stoppage_next_no_setpiece":
            print(f"     next setpiece {dict(Counter(str(x['next_setpiece']) for x in items))}")
        for x in items[:3]:
            print(f"     e.g. {x}")
    uk = Counter()
    for r in res:
        uk.update(r["odd"]["unknown_frames_by_rule"])
    print(f"  UNKNOWN frames by rule: {dict(uk)}")
    for nm in uk:
        exs = [dict(e, match=r["match"]) for r in res for e in r["odd"]["unknown_examples"].get(nm, [])][:3]
        for e in exs:
            print(f"     {nm} e.g. {e}")


COLORS = {0: "#9e9e9e", 1: "#2e7d32", 2: "#1565c0", 3: "#ef6c00", 4: "#d81b60"}


def plot(mid: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lab = dict(np.load(OUT / f"{mid}.npz"))
    c = dict(np.load(build_cache(mid)))
    t, per, st = lab["t"], lab["period"], lab["state"]
    ours = smooth_ball(t, c["rx"], c["ry"], c["rz"], visibility=VIS_NAMES[c["rvis"]], period=per, cfg=SmootherConfig())
    sp = np.hypot(ours.vx, ours.vy)
    events = Dataset(ROOT).load_events(mid, EventVersion.ROOT)
    outs = [r.event_time_s for r in events if r.game_event_type == "OUT" and r.canonical_period == 1 and r.event_time_s and r.event_time_s > t[0] + 900]
    t0 = outs[0] - 30 if outs else t[per == 1][0] + 900
    sel = (t >= t0) & (t <= t0 + 60)
    idx = np.flatnonzero(sel)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    a = idx[0]
    for i in range(idx[0] + 1, idx[-1] + 2):
        if i > idx[-1] or st[i] != st[a]:
            ax.axvspan(t[a] - t0, t[min(i, idx[-1])] - t0, color=COLORS[int(st[a])], alpha=0.35, lw=0)
            a = i
    ax.plot(t[sel] - t0, sp[sel], color="k", lw=1.2, label="ball speed (our smoothed raw ball)")
    for r in events:
        if r.event_time_s is not None and t0 <= r.event_time_s <= t0 + 60:
            lbl = r.possession_event_type or r.game_event_type
            ax.axvline(r.event_time_s - t0, color="k", ls=":", lw=0.7)
            ax.text(r.event_time_s - t0, 31, lbl, rotation=90, fontsize=7, va="bottom", ha="center")
    ax.set_ylim(0, 36); ax.set_xlim(0, 60)
    ax.set_xlabel(f"seconds from video time {t0:.1f} s"); ax.set_ylabel("speed (m/s)")
    handles = [Patch(color=COLORS[k], alpha=.35, label=STATE_NAMES[k]) for k in range(5)] + [ax.lines[0]]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=6, bbox_to_anchor=(1, 1.22))
    ax.set_title(f"Match {mid}: answer-key states (bands), ball speed, PFF event marks (dotted)", pad=26)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"timeline_{mid}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    mids = [str(m) for m in DEV_MATCHES]
    if not a.plots_only:
        if a.workers > 1:
            with multiprocessing.Pool(a.workers) as pool:
                res = pool.map(run_match, mids)
        else:
            res = [run_match(m) for m in mids]
        (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        report(res)
    for m in ("3812", "3815", "3820"):
        print("plot:", plot(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
