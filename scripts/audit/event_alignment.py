"""Read-only temporal-coverage and coordinate-agreement check between events and tracking
(Milestone 1B audit). This is NOT a synchronization engine and validates nothing physical.

Three separate statuses are reported, never merged:
  1. temporal coverage      - a same-period tracking frame exists within one frame interval of
                              the event time (policy in audit_lib.Aligner);
  2. coordinate agreement   - the event's player coordinates equal the raw tracking positions at
                              the selected frame (audit_lib.coordinate_status; tolerance 0.01 m,
                              >= 90 % of >= 11 compared players);
  3. validated correspondence - NOT assessed here (needs independent ground truth such as video).
Event rows of each version are aligned independently, every row preserved (no key collapse).

One match:   python scripts/audit/event_alignment.py --matches 3812 --workers 1
All matches: python scripts/audit/event_alignment.py --workers 7     (about 4 min CPU per match)
Pooled:      python scripts/audit/event_alignment.py --aggregate
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

RESIDUAL_BUCKETS_MS = (("<=1ms", 1.0), ("<=16.7ms(half frame)", 1000 / 29.97 / 2), ("<=33.4ms(1 frame)", 1000 / 29.97), (">1 frame", float("inf")))


def jersey(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def frame_snapshot(r: dict) -> dict:
    players = {}
    for side in ("homePlayers", "awayPlayers"):
        for e in r.get(side) or []:
            players[(side, jersey(e.get("jerseyNum")))] = (e.get("x"), e.get("y"))
    b = (r.get("balls") or [None])[0]
    return dict(players=players, ball=((b.get("x"), b.get("y"), b.get("z")) if b else None))


def run_match(root: str, mid: str) -> dict:
    f = L.locate(root, mid)
    fps = float(L.load_metadata(f["metadata"])["fps"])
    rows = {v: L.load_json(p) for v, p in f["events"].items()}
    al = {}
    for v, rs in rows.items():
        al[v] = L.Aligner([(i, r.get("eventTime"), (r.get("gameEvents") or {}).get("period")) for i, r in enumerate(rs)], fps)
    prev_fn = None
    c: Counter = Counter()
    for r in L.iter_tracking(f["tracking"]):
        c["records"] += 1
        fn = r["frameNum"]
        if fn == prev_fn:
            c["repeat_records_skipped"] += 1
            continue
        prev_fn = fn
        c["frames"] += 1
        memo: list = []

        def payload(r=r, memo=memo):
            if not memo:
                memo.append(frame_snapshot(r))
            return memo[0]

        t = r["videoTimeMs"] / 1000.0
        for a in al.values():
            a.offer(fn, t, r["period"], payload)
    out = dict(fps=fps, tracking=dict(c), tolerance_s=1.0 / fps, versions={})
    for v, rs in rows.items():
        a = al[v]
        outcome: Counter = Counter(); reasons: Counter = Counter(); by_period: dict = {}
        resid_ms: list[float] = []; resid_signed: list[float] = []
        bucket: Counter = Counter()
        status: Counter = Counter(); players: Counter = Counter(); ball: Counter = Counter()
        for i, r in enumerate(rs):
            per = (r.get("gameEvents") or {}).get("period")
            o, reason = a.outcome(i, r.get("eventTime"), per)
            pp = by_period.setdefault(str(per), Counter())
            pp[o if o == "matched" else f"unmatched:{reason}"] += 1
            outcome[o] += 1
            if o == "unmatched":
                reasons[reason] += 1
                continue
            ad, dt, fn, snap = a.best[i]
            resid_ms.append(ad * 1000); resid_signed.append(dt * 1000)
            for name, hi in RESIDUAL_BUCKETS_MS:
                if ad * 1000 <= hi + 1e-9:
                    bucket[name] += 1
                    break
            cs = L.coordinate_status(L.player_map(r), snap["players"])
            status[cs["status"]] += 1
            players["compared"] += cs["compared"]; players["exact"] += cs["exact"]; players["within_0.01m"] += cs["near"]
            by_period[str(per)][cs["status"]] += 1
            eb = (r.get("ball") or [None])[0]
            tb = snap["ball"]
            if eb and tb:
                d = L.dist_xy((eb.get("x"), eb.get("y")), tb)
                ball["both_present"] += 1
                ball["exact" if d is not None and d <= L.COORD_EXACT_M else ("within_0.01m" if d is not None and d <= L.COORD_NEAR_M else ("null_coordinate" if d is None else "farther"))] += 1
            elif eb and not tb:
                ball["event_only"] += 1
            elif tb and not eb:
                ball["tracking_only"] += 1
            else:
                ball["neither"] += 1
        out["versions"][v] = dict(
            event_rows=len(rs), temporal=dict(outcome), unmatched_reasons=dict(reasons), by_period={k: dict(x) for k, x in by_period.items()},
            residual_abs_ms=L.dist_stats(resid_ms), residual_signed_median_ms=L.pct(resid_signed, 0.5), residual_buckets=dict(bucket),
            coordinate_status=dict(status), players=dict(players), ball=dict(ball), validated_correspondence="not_assessed",
        )
    return out


def aggregate(res: dict[str, dict]) -> None:
    ok = {m: r for m, r in res.items() if r.get("ok")}
    bad = {m: r.get("error") for m, r in res.items() if not r.get("ok")}
    print(f"matches with output: {len(res)}; ok: {len(ok)}; failed/incomplete: {len(bad)} {bad if bad else ''}")
    if not ok:
        return
    print(f"tracking frames {sum(r['tracking']['frames'] for r in ok.values())}; repeat records skipped {sum(r['tracking'].get('repeat_records_skipped', 0) for r in ok.values())}; tolerance 1 frame = {1000 / 29.97:.2f} ms")
    for v in L.VERSIONS:
        s = [r["versions"][v] for r in ok.values()]
        n = sum(x["event_rows"] for x in s)
        m = sum(x["temporal"].get("matched", 0) for x in s)
        print(f"\n[{v}] event rows {n}; temporally matched {m} ({100 * m / n:.3f}%); unmatched {n - m}")
        rs: Counter = Counter(); bk: Counter = Counter(); cs: Counter = Counter(); pl: Counter = Counter(); bl: Counter = Counter()
        for x in s:
            rs.update(x["unmatched_reasons"]); bk.update(x["residual_buckets"]); cs.update(x["coordinate_status"]); pl.update(x["players"]); bl.update(x["ball"])
        print(f"  unmatched reasons: {dict(rs)}")
        print(f"  per-match residual p50 range {min(x['residual_abs_ms']['p50'] for x in s):.3f}-{max(x['residual_abs_ms']['p50'] for x in s):.3f} ms; "
              f"worst per-match p99 {max(x['residual_abs_ms']['p99'] for x in s):.2f} ms; max {max(x['residual_abs_ms']['max'] for x in s):.2f} ms; pooled buckets {dict(bk)}")
        print(f"  coordinate status over {m} temporally matched events: {dict(cs)}; players compared {pl['compared']}: exact {pl['exact']} ({100 * pl['exact'] / max(1, pl['compared']):.2f}%), within 0.01 m {pl['within_0.01m']} ({100 * pl['within_0.01m'] / max(1, pl['compared']):.2f}%)")
        print(f"  ball: {dict(bl)}")
        dis = sorted(mm for mm, r in ok.items() if r["versions"][v]["coordinate_status"].get("coordinate_disagreeing", 0) > 0.05 * r["versions"][v]["temporal"].get("matched", 1))
        print(f"  matches where >5% of matched events are coordinate_disagreeing ({len(dis)}): {dis}")
        unm = {mm: r["versions"][v]["event_rows"] - r["versions"][v]["temporal"].get("matched", 0) for mm, r in ok.items() if r["versions"][v]["event_rows"] - r["versions"][v]["temporal"].get("matched", 0)}
        print(f"  matches with unmatched events ({len(unm)}), top: {sorted(unm.items(), key=lambda kv: -kv[1])[:8]}")


def main() -> int:
    return L.cli_main(__doc__, "data/processed/audit_1b/event_alignment", run_match, aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
