"""Read-only check for suspected side/identity anomalies and for extra-time coverage (Milestone 1B audit).

Test: the player named by an event as the ball carrier (team + player or shirt number) should be
near the ball in the position list of that team. If the carrier's own-side entry is far from the
ball while the opposite list has a player at the ball, the side labels of the position lists are
suspect for that period. The result is a flag, `suspected_side_identity_anomaly`; nothing is swapped
or corrected, and the cause is not established by this test.

Population (events, root version): game event OTB, possession event present and not IT, teamId and
playerId non-null, ball list non-empty. Population (tracking, only for matches that have extra-time
events): first frame of each possession_event_id whose embedded game_event is OTB with team_id and
shirt_number, ball non-empty. Heuristic flag (provisional): n >= 30, median own-side distance > 8 m
and median opposite-side distance < 4 m.

All matches (events; tracking for extra-time matches only):
    python scripts/audit/extra_time_identity.py --workers 4
Pooled: python scripts/audit/extra_time_identity.py --aggregate
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

MIN_N, OWN_FAR_M, OPP_NEAR_M = 30, 8.0, 4.0


def med(v):
    return round(statistics.median(v), 2) if v else None


def flag(n, own, opp):
    if n >= MIN_N and own is not None and opp is not None and own > OWN_FAR_M and opp < OPP_NEAR_M:
        return "suspected_side_identity_anomaly"
    return "no_anomaly_detected" if n >= MIN_N else "insufficient_events"


def run_match(root: str, mid: str) -> dict:
    f = L.locate(root, mid)
    meta = L.load_metadata(f["metadata"])
    home = int(meta["homeTeam"]["id"])
    rows = L.load_json(f["events"]["root"])
    per_rows = Counter((r.get("gameEvents") or {}).get("period") for r in rows)
    acc: dict = {}
    for r in rows:
        ge, pe = r.get("gameEvents") or {}, r.get("possessionEvents") or {}
        if ge.get("gameEventType") != "OTB" or not pe.get("possessionEventType") or pe.get("possessionEventType") == "IT":
            continue
        if ge.get("teamId") is None or ge.get("playerId") is None or not r.get("ball"):
            continue
        b = (r["ball"][0].get("x"), r["ball"][0].get("y"))
        own_key, opp_key = ("homePlayers", "awayPlayers") if ge["teamId"] == home else ("awayPlayers", "homePlayers")
        own = [p for p in r[own_key] if p.get("playerId") == ge["playerId"]]
        if not own:
            continue
        d_own = L.dist_xy(b, (own[0]["x"], own[0]["y"]))
        d_opp = [L.dist_xy(b, (p["x"], p["y"])) for p in r[opp_key]]
        d_opp = [x for x in d_opp if x is not None]
        if d_own is None or not d_opp:
            continue
        a = acc.setdefault(str(ge.get("period")), dict(own=[], opp=[]))
        a["own"].append(d_own); a["opp"].append(min(d_opp))
    events = {}
    for k, a in sorted(acc.items()):
        n, mo, mp = len(a["own"]), med(a["own"]), med(a["opp"])
        events[k] = dict(n=n, median_own_side_m=mo, median_nearest_opposite_m=mp, share_own_over_10m=round(sum(1 for x in a["own"] if x > 10) / n, 3), flag=flag(n, mo, mp))
    out = dict(event_rows_by_period={str(k): v for k, v in per_rows.items()}, has_extra_time_events=any((k or 0) >= 3 for k in per_rows), events=events)
    if out["has_extra_time_events"]:
        t_acc: dict = {}
        seen = set(); tracking_periods: Counter = Counter(); prev = None
        for r in L.iter_tracking(f["tracking"]):
            if r["frameNum"] == prev:
                continue
            prev = r["frameNum"]
            tracking_periods[str(r["period"])] += 1
            ge = r.get("game_event")
            pid = r.get("possession_event_id")
            if not ge or pid is None or pid in seen or ge.get("game_event_type") != "OTB" or ge.get("team_id") is None or not r["balls"]:
                continue
            seen.add(pid)
            b = (r["balls"][0].get("x"), r["balls"][0].get("y"))
            own_key, opp_key = ("homePlayers", "awayPlayers") if int(ge["team_id"]) == home else ("awayPlayers", "homePlayers")
            own = [e for e in r[own_key] if e.get("jerseyNum") == ge.get("shirt_number")]
            opp = [e for e in r[opp_key] if e.get("jerseyNum") == ge.get("shirt_number")]
            if not own or not opp:
                continue
            d1 = L.dist_xy(b, (own[0]["x"], own[0]["y"])); d2 = L.dist_xy(b, (opp[0]["x"], opp[0]["y"]))
            if d1 is None or d2 is None:
                continue
            a = t_acc.setdefault(str(r["period"]), dict(own=[], opp=[]))
            a["own"].append(d1); a["opp"].append(d2)
        tr = {}
        for k, a in sorted(t_acc.items()):
            n, mo, mp = len(a["own"]), med(a["own"]), med(a["opp"])
            tr[k] = dict(n=n, median_own_side_m=mo, median_same_shirt_opposite_side_m=mp, flag=flag(n, mo, mp))
        out["tracking_frames_by_period"] = dict(tracking_periods)
        out["tracking"] = tr
    return out


def aggregate(res: dict[str, dict]) -> None:
    ok = {m: r for m, r in res.items() if r.get("ok")}
    bad = {m: r.get("error") for m, r in res.items() if not r.get("ok")}
    print(f"matches with output: {len(res)}; ok: {len(ok)}; failed/incomplete: {len(bad)} {bad if bad else ''}")
    reg = [(m, k, v) for m, r in ok.items() for k, v in r["events"].items() if k in ("1", "2")]
    print(f"\nEVENTS, regular time (periods 1-2) over {len(ok)} matches: {len(reg)} match-periods; "
          f"max median own-side distance {max(v['median_own_side_m'] for _, _, v in reg)} m; flagged: {[(m, k) for m, k, v in reg if v['flag'] == 'suspected_side_identity_anomaly']}; insufficient: {[(m, k) for m, k, v in reg if v['flag'] == 'insufficient_events']}")
    et = {m: r for m, r in ok.items() if r["has_extra_time_events"]}
    print(f"\nmatches with extra-time EVENTS ({len(et)}): {sorted(et)}")
    for m, r in sorted(et.items()):
        print(f"  {m} event rows by period {r['event_rows_by_period']}")
        for k, v in r["events"].items():
            if k in ("3", "4"):
                print(f"     events p{k}: {v}")
        print(f"     tracking frames by period {r.get('tracking_frames_by_period')}")
        for k, v in (r.get("tracking") or {}).items():
            if k in ("3", "4"):
                print(f"     tracking p{k}: {v}")
        if "3" not in (r.get("tracking_frames_by_period") or {}):
            print("     tracking: NO extra-time frames (event-only extra time)")
    other = Counter()
    for m, r in ok.items():
        if any(k not in ("1", "2", "3", "4") for k in r["event_rows_by_period"]):
            other[m] = {k: v for k, v in r["event_rows_by_period"].items() if k not in ("1", "2", "3", "4")}
    print(f"\nmatches with event rows whose source period is not 1-4: {dict(other)}")


def main() -> int:
    return L.cli_main(__doc__, "data/processed/audit_1b/extra_time_identity", run_match, aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
