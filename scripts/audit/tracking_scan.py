"""Read-only streaming scan of tracking files (Milestone 1B audit): coverage, missing data,
estimated positions, and smoothed-list duplicate/conflict classification.

Denominators are explicit: 'records' are source lines; 'frames' are unique frameNum values
(a repeated frameNum collapses to its first record; whether the repeats are identical is
counted). Per-match results are stored separately from pooled totals. Smoothed-list duplicates
are classified (identical vs conflicting) and never resolved: no candidate is selected.

Scan:        python scripts/audit/tracking_scan.py --matches 3812 --workers 1
All matches: python scripts/audit/tracking_scan.py --workers 7        (about 3-4 min CPU per match)
Pooled:      python scripts/audit/tracking_scan.py --aggregate
Show a frame (raw + smoothed candidates, early exit):
             python scripts/audit/tracking_scan.py show-frame MATCH FRAMENUM
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

SIDES = (("home", "homePlayers", "homePlayersSmoothed"), ("away", "awayPlayers", "awayPlayersSmoothed"))
SPREAD_BUCKETS = (("<=0.05m", 0.05), ("<=0.10m", 0.10), ("<=0.50m", 0.50), (">0.50m", float("inf")))


def bucket(d: float) -> str:
    for name, hi in SPREAD_BUCKETS:
        if d <= hi:
            return name
    return SPREAD_BUCKETS[-1][0]


def scan_match(root: str, mid: str) -> dict:
    f = L.locate(root, mid)
    meta = L.load_metadata(f["metadata"])
    fps = float(meta["fps"])
    c: Counter = Counter()
    periods: dict = {}
    sm_status: Counter = Counter()          # per smoothed list (frame x side)
    sm_len: Counter = Counter()
    sm_groups: Counter = Counter()          # duplicate jersey groups
    sm_group_sizes: Counter = Counter()
    sm_spread: Counter = Counter()          # conflicting groups by spread bucket
    sm_vs_raw: Counter = Counter()          # conflicting groups where raw same-jersey entry exists
    sm_raw_dist: list[float] = []
    raw_status: Counter = Counter()
    repeat_sm_status: Counter = Counter()   # smoothed lists of repeated-frameNum records (not part of 'frames')
    examples: list[dict] = []
    visibility: Counter = Counter(); confidence: Counter = Counter()
    prev = None
    cur_dup = False; cur_rep = 0            # per-frame cross-tab: smoothed-list duplication vs repeated records
    xtab: Counter = Counter()
    max_time_formula_err_ms = 0.0
    for r in L.iter_tracking(f["tracking"]):
        c["records"] += 1
        fn, vt, per = r["frameNum"], r["videoTimeMs"], r["period"]
        if prev is not None and fn == prev["fn"]:
            c["repeat_records"] += 1
            cur_rep += 1
            same = r["homePlayers"] == prev["home"] and r["awayPlayers"] == prev["away"]
            c["repeat_records_identical_raw_players"] += same
            c["repeat_records_identical_smoothed_players"] += (r.get("homePlayersSmoothed") == prev["home_sm"] and r.get("awayPlayersSmoothed") == prev["away_sm"])
            for _side, _kr, ks in SIDES:
                repeat_sm_status[L.classify_duplicate_jerseys(r.get(ks))["status"]] += 1
            continue
        c["frames"] += 1
        if prev is not None:
            xtab[("smoothed_duplicated" if cur_dup else "smoothed_unique", "has_repeat_records" if cur_rep else "no_repeat_records")] += 1
        cur_dup, cur_rep = False, 0
        if prev is not None:
            if per == prev["per"]:
                step = fn - prev["fn"]
                if step < 0 or vt < prev["vt"]:
                    c["nonmonotonic_steps"] += 1
                elif step != 1:
                    c["frame_gaps_within_period"] += 1
            else:
                c["period_transitions"] += 1
        prev = dict(fn=fn, vt=vt, per=per, home=r["homePlayers"], away=r["awayPlayers"], home_sm=r.get("homePlayersSmoothed"), away_sm=r.get("awayPlayersSmoothed"))
        max_time_formula_err_ms = max(max_time_formula_err_ms, abs(vt - fn / fps * 1000.0))
        p = periods.setdefault(str(per), dict(frames=0, first_frame=fn, last_frame=fn, first_video_s=vt / 1000, last_video_s=vt / 1000))
        p["frames"] += 1
        p["last_frame"], p["last_video_s"] = fn, vt / 1000
        # ball
        balls = r.get("balls")
        c["ball_list_empty"] += (not balls)
        bs = r.get("ballsSmoothed")
        c["ball_smoothed_null"] += bs is None
        if bs is not None:
            c["ball_smoothed_present"] += 1
            c["ball_smoothed_xy_null"] += (bs.get("x") is None or bs.get("y") is None)
        # players
        for side, kr, ks in SIDES:
            raw, sm = r.get(kr), r.get(ks)
            for e in raw or []:
                c["raw_entries"] += 1
                visibility[e.get("visibility")] += 1
                confidence[e.get("confidence")] += 1
                c["raw_entries_with_speed"] += "speed" in e
                c["raw_entries_null_xy"] += (e.get("x") is None or e.get("y") is None)
            rc = L.classify_duplicate_jerseys(raw)
            raw_status[rc["status"]] += 1
            sc = L.classify_duplicate_jerseys(sm)
            sm_status[sc["status"]] += 1
            cur_dup = cur_dup or sc["status"] != "unique" and sc["status"] != "missing"
            if sm is not None:
                sm_len[len(sm)] += 1
                c["smoothed_entries_null_xy"] += sum(1 for e in sm if e.get("x") is None or e.get("y") is None)
            rawmap = {e.get("jerseyNum"): e for e in (raw or [])}
            for j, g in sc["groups"].items():
                sm_groups["groups"] += 1
                sm_group_sizes[len(g)] += 1
                if L.classify_duplicate_jerseys(g)["status"] == "duplicate_conflicting":
                    sm_groups["conflicting_groups"] += 1
                    spread = L.classify_duplicate_jerseys(g)["max_spread_m"]
                    sm_spread[bucket(spread)] += 1
                    re_ = rawmap.get(j)
                    if re_ is not None:
                        sm_vs_raw["conflicting_groups_with_raw_entry"] += 1
                        ds = [L.dist_xy((e["x"], e["y"]), (re_["x"], re_["y"])) for e in g]
                        sm_raw_dist.extend(d for d in ds if d is not None)
                    if len(examples) < 3:
                        examples.append(dict(frameNum=fn, side=side, jerseyNum=j, candidates=[(e["x"], e["y"], e["visibility"], e["confidence"]) for e in g],
                                             raw=(re_["x"], re_["y"]) if re_ else None))
                else:
                    sm_groups["identical_groups"] += 1
    xtab[("smoothed_duplicated" if cur_dup else "smoothed_unique", "has_repeat_records" if cur_rep else "no_repeat_records")] += 1
    return dict(
        frame_crosstab={f"{a}|{b}": n for (a, b), n in xtab.items()},
        fps=fps, counts=dict(c), periods=periods, max_abs_videoTimeMs_minus_frame_formula_ms=max_time_formula_err_ms,
        visibility=dict(visibility), confidence=dict(confidence),
        raw_list_status=dict(raw_status), repeat_record_smoothed_list_status=dict(repeat_sm_status), smoothed_list_status=dict(sm_status), smoothed_list_lengths={str(k): v for k, v in sm_len.items()},
        smoothed_duplicate_groups=dict(sm_groups), smoothed_group_sizes={str(k): v for k, v in sm_group_sizes.items()},
        smoothed_conflict_spread=dict(sm_spread), smoothed_conflict_vs_raw=dict(sm_vs_raw),
        smoothed_candidate_to_raw_distance_m=L.dist_stats(sm_raw_dist), examples=examples,
    )


def aggregate(res: dict[str, dict]) -> None:
    ok = {m: r for m, r in res.items() if r.get("ok")}
    bad = {m: r.get("error") for m, r in res.items() if not r.get("ok")}
    print(f"matches with output: {len(res)}; ok: {len(ok)}; failed/incomplete: {len(bad)} {bad if bad else ''}")
    if not ok:
        return
    S = lambda k: sum(r["counts"].get(k, 0) for r in ok.values())  # noqa: E731
    rec, fr = S("records"), S("frames")
    print(f"\nrecords {rec}; unique frames {fr}; repeated-frameNum records {S('repeat_records')} (identical raw players in {S('repeat_records_identical_raw_players')}); "
          f"frames per match {min(r['counts']['frames'] for r in ok.values())}-{max(r['counts']['frames'] for r in ok.values())}")
    print(f"non-monotonic steps {S('nonmonotonic_steps')}; frame gaps within a period {S('frame_gaps_within_period')}; period transitions {S('period_transitions')}; "
          f"max |videoTimeMs - frameNum/fps*1000| = {max(r['max_abs_videoTimeMs_minus_frame_formula_ms'] for r in ok.values()):.3f} ms; raw entries with null x/y {S('raw_entries_null_xy')}")
    pc: Counter = Counter()
    for r in ok.values():
        pc[tuple(sorted(r["periods"]))] += 1
    print(f"period sets across matches: {dict(pc)}")
    print(f"matches without periods 3/4 in tracking: {sorted(m for m, r in ok.items() if '3' not in r['periods'])[:70]}")
    # ball and positions (frame denominators)
    bm = [r["counts"]["ball_list_empty"] / r["counts"]["frames"] for r in ok.values()]
    print(f"\nball list empty: {S('ball_list_empty')} of {fr} frames ({100 * S('ball_list_empty') / fr:.2f}% pooled; per match {100 * min(bm):.1f}-{100 * max(bm):.1f}%)")
    print(f"ballsSmoothed null: {S('ball_smoothed_null')} of {fr} frames; present {S('ball_smoothed_present')}, of which x or y null {S('ball_smoothed_xy_null')} "
          f"({100 * S('ball_smoothed_xy_null') / max(1, S('ball_smoothed_present')):.2f}% of present)")
    vis: Counter = Counter(); conf: Counter = Counter()
    for r in ok.values():
        vis.update(r["visibility"]); conf.update(r["confidence"])
    re_ = S("raw_entries")
    em = [r["visibility"].get("ESTIMATED", 0) / sum(r["visibility"].values()) for r in ok.values()]
    print(f"raw player entries {re_}; ESTIMATED {vis['ESTIMATED']} ({100 * vis['ESTIMATED'] / re_:.2f}% pooled; per match {100 * min(em):.1f}-{100 * max(em):.1f}%); confidence {dict(conf)}")
    sp = [m for m, r in ok.items() if r["counts"].get("raw_entries_with_speed", 0)]
    full = [m for m, r in ok.items() if r["counts"].get("raw_entries_with_speed", 0) == r["counts"]["raw_entries"]]
    print(f"raw entries with speed: {S('raw_entries_with_speed')}; matches with any speed {len(sp)}, with speed on every entry {len(full)}")
    # smoothed lists
    rs: Counter = Counter(); ss: Counter = Counter(); sl: Counter = Counter(); sg: Counter = Counter(); sz: Counter = Counter(); spd: Counter = Counter(); vr: Counter = Counter()
    dist_all = 0
    for r in ok.values():
        rs.update(r["raw_list_status"]); ss.update(r["smoothed_list_status"]); sl.update(r["smoothed_list_lengths"])
        sg.update(r["smoothed_duplicate_groups"]); sz.update(r["smoothed_group_sizes"]); spd.update(r["smoothed_conflict_spread"]); vr.update(r["smoothed_conflict_vs_raw"])
    lists = sum(ss.values())
    print(f"\nraw lists (frame x side) by status: {dict(rs)}")
    print(f"smoothed lists (frame x side) by status: {dict(ss)} (denominator {lists} lists = {fr} frames x 2 sides)")
    rp: Counter = Counter()
    for r in ok.values():
        rp.update(r.get("repeat_record_smoothed_list_status", {}))
    xt: Counter = Counter()
    for r in ok.values():
        xt.update(r.get("frame_crosstab", {}))
    print(f"frames cross-tab (smoothed list duplicated in either side vs frame has repeated records): {dict(xt)}")
    print(f"smoothed lists of the {S('repeat_records')} repeated-frameNum records (denominator {2 * S('repeat_records')} lists) by status: {dict(rp)}; repeats whose smoothed lists equal the first record's: {S('repeat_records_identical_smoothed_players')}")
    print(f"smoothed list lengths: {dict(sl)}")
    print(f"smoothed duplicate jersey groups: {dict(sg)}; group sizes {dict(sz)}")
    print(f"conflicting groups by candidate spread: {dict(spd)}; with a raw same-jersey entry at the frame: {vr.get('conflicting_groups_with_raw_entry', 0)}")
    dm = [r["smoothed_candidate_to_raw_distance_m"] for r in ok.values() if r["smoothed_candidate_to_raw_distance_m"]["n"]]
    if dm:
        print(f"candidate-to-raw distance (per-match p50 range {min(x['p50'] for x in dm):.2f}-{max(x['p50'] for x in dm):.2f} m; max {max(x['max'] for x in dm):.2f} m)")
    mm = {m: r["smoothed_list_status"].get("duplicate_conflicting", 0) for m, r in ok.items()}
    print(f"matches with >=1 conflicting smoothed list: {sum(1 for v in mm.values() if v)}; matches with identical-only duplicates: "
          f"{sum(1 for m, r in ok.items() if r['smoothed_list_status'].get('duplicate_identical', 0) and not mm[m])}; no duplicates: {sum(1 for m, r in ok.items() if not r['smoothed_list_status'].get('duplicate_identical', 0) and not mm[m])}")


def show_frame(root: str, mid: str, frame: int) -> int:
    f = L.locate(root, mid)
    for r in L.iter_tracking(f["tracking"]):
        if r["frameNum"] != frame:
            continue
        print(f"match {mid} frameNum {frame} period {r['period']} videoTimeMs {r['videoTimeMs']}")
        for side, kr, ks in SIDES:
            raw, sm = r[kr], r[ks]
            cl = L.classify_duplicate_jerseys(sm)
            print(f"{side}: raw entries {len(raw or [])} status={L.classify_duplicate_jerseys(raw)['status']}; smoothed entries {len(sm or [])} status={cl['status']} max candidate spread {cl['max_spread_m']}")
            rawmap = {e['jerseyNum']: e for e in raw or []}
            for j, g in sorted(cl["groups"].items(), key=lambda kv: int(kv[0]))[:4]:
                print(f"  jersey {j}: raw=({rawmap[j]['x']}, {rawmap[j]['y']}) candidates=" + "; ".join(f"({e['x']}, {e['y']}) {e['visibility']}/{e['confidence']}" for e in g))
        return 0
    print("frame not found")
    return 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "show-frame":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("match"); ap.add_argument("frame", type=int); ap.add_argument("--root", default="data/raw")
        a = ap.parse_args()
        try:
            return show_frame(a.root, L.validate_match_id(a.match), a.frame)
        except L.AuditError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    return L.cli_main(__doc__, "data/processed/audit_1b/tracking_scan", scan_match, aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
