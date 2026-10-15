"""Read-only, collision-safe comparison of the three Event Data versions (Milestone 1B audit).

Every source row is preserved. Rows are matched between versions on (signature, ordinal), where
the signature is (gameEventId, possessionEventId, startTime, gameEventType, possessionEventType)
and the ordinal numbers repeats of an identical signature in file order. Unmatched and
ambiguous rows are reported, never dropped. Field comparison covers paths present in BOTH rows
(common scope); paths present in one row only are schema differences. Player and ball position
lists are compared separately. Changed rows and changed field values are counted separately.

Per match:   python scripts/audit/event_versions.py --matches 3812
All matches: python scripts/audit/event_versions.py --workers 4
Pooled view: python scripts/audit/event_versions.py --aggregate
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as L  # noqa: E402

PAIRS = (("may_1_2025", "march_14_2025"), ("root", "may_1_2025"), ("root", "march_14_2025"))


def version_summary(rows: list[dict]) -> dict:
    ge = Counter((r.get("gameEvents") or {}).get("gameEventType") for r in rows)
    pe = Counter((r.get("possessionEvents") or {}).get("possessionEventType") for r in rows)
    periods = Counter((r.get("gameEvents") or {}).get("period") for r in rows)
    keys = Counter(L.pair_key(r) for r in rows)
    return dict(
        rows=len(rows), distinct_key_pairs=len(keys), rows_in_repeated_key_pairs=L.repeated_pair_rows(rows),
        repeated_key_pairs=sum(1 for n in keys.values() if n > 1),
        distinct_signatures=len({L.signature(r) for r in rows}),
        game_event_types={str(k): v for k, v in ge.items()}, possession_event_types={str(k): v for k, v in pe.items()},
        periods={str(k): v for k, v in periods.items()},
        rows_with_null_possession_id=sum(1 for r in rows if r.get("possessionEventId") is None),
        top_level_keys=sorted({k for r in rows for k in r}),
    )


def compare_versions(a: list[dict], b: list[dict]) -> dict:
    m = L.match_rows(a, b)
    changed_rows = 0
    changed_values = 0
    per_field: Counter = Counter()
    rows_with_field: Counter = Counter()
    only_a: Counter = Counter()
    only_b: Counter = Counter()
    pos = Counter()
    max_xy = 0.0
    for i, j in m["pairs"]:
        c = L.compare_fields(a[i], b[j])
        if c["changed"]:
            changed_rows += 1
        changed_values += len(c["changed"])
        for p in c["changed"]:
            per_field[p] += 1
        for p in c["only_a"]:
            only_a[p] += 1
        for p in c["only_b"]:
            only_b[p] += 1
        q = L.compare_positions(a[i], b[j])
        pos["rows"] += 1
        pos["players_compared"] += q["players_compared"]
        pos["rows_player_set_differs"] += q["player_set_differs"]
        pos["player_xy_changed"] += q["xy_changed"]
        pos["rows_with_xy_change"] += q["xy_changed"] > 0
        pos["rows_ball_differs"] += q["ball_differs"]
        max_xy = max(max_xy, q["max_xy_diff_m"])
    type_a = Counter((a[i].get("gameEvents") or {}).get("gameEventType") for i in m["unmatched_a"])
    ptype_a = Counter((a[i].get("possessionEvents") or {}).get("possessionEventType") for i in m["unmatched_a"])
    type_b = Counter((b[j].get("gameEvents") or {}).get("gameEventType") for j in m["unmatched_b"])
    ptype_b = Counter((b[j].get("possessionEvents") or {}).get("possessionEventType") for j in m["unmatched_b"])
    return dict(
        rows_a=len(a), rows_b=len(b), matched_pairs=len(m["pairs"]), unmatched_a=len(m["unmatched_a"]), unmatched_b=len(m["unmatched_b"]),
        ambiguous_signatures=len(m["ambiguous_signatures"]),
        unmatched_a_possession_types={str(k): v for k, v in ptype_a.items()}, unmatched_b_possession_types={str(k): v for k, v in ptype_b.items()},
        unmatched_a_game_types={str(k): v for k, v in type_a.items()}, unmatched_b_game_types={str(k): v for k, v in type_b.items()},
        rows_with_changed_common_field=changed_rows, changed_field_values=changed_values,
        changed_by_field=dict(per_field), paths_only_in_a=dict(only_a), paths_only_in_b=dict(only_b),
        positions={**dict(pos), "max_xy_diff_m": max_xy},
    )


def run_match(root: str, mid: str) -> dict:
    f = L.locate(root, mid)
    rows = {label: L.load_json(path) for label, path in f["events"].items()}
    out = dict(versions={k: version_summary(v) for k, v in rows.items()}, compare={})
    for a, b in PAIRS:
        out["compare"][f"{a}__vs__{b}"] = compare_versions(rows[a], rows[b])
    return out


def aggregate(res: dict[str, dict]) -> None:
    ok = {m: r for m, r in res.items() if r.get("ok")}
    bad = {m: r.get("error") for m, r in res.items() if not r.get("ok")}
    print(f"matches with output: {len(res)}; ok: {len(ok)}; failed: {len(bad)} {bad if bad else ''}")
    for v in L.VERSIONS:
        s = [r["versions"][v] for r in ok.values()]
        print(f"\n[{v}] rows {sum(x['rows'] for x in s)} (per match {min(x['rows'] for x in s)}-{max(x['rows'] for x in s)}); "
              f"distinct key pairs {sum(x['distinct_key_pairs'] for x in s)}; repeated key pairs {sum(x['repeated_key_pairs'] for x in s)} "
              f"covering {sum(x['rows_in_repeated_key_pairs'] for x in s)} rows in {sum(1 for x in s if x['repeated_key_pairs'])} matches; "
              f"distinct signatures {sum(x['distinct_signatures'] for x in s)}")
        for k in ("game_event_types", "possession_event_types", "periods"):
            c: Counter = Counter()
            for x in s:
                c.update(x[k])
            print(f"  {k}: {dict(c)}")
    for a, b in PAIRS:
        key = f"{a}__vs__{b}"
        s = [r["compare"][key] for r in ok.values()]
        t = lambda f: sum(x[f] for x in s)  # noqa: E731
        print(f"\n[{a} vs {b}] rows {t('rows_a')} vs {t('rows_b')}; matched {t('matched_pairs')} "
              f"({100 * t('matched_pairs') / t('rows_a'):.3f}% of {a}, {100 * t('matched_pairs') / t('rows_b'):.3f}% of {b}); "
              f"unmatched {a}: {t('unmatched_a')}, {b}: {t('unmatched_b')}; ambiguous signatures: {t('ambiguous_signatures')}")
        pa: Counter = Counter(); pb: Counter = Counter()
        for x in s:
            pa.update(x["unmatched_a_possession_types"]); pb.update(x["unmatched_b_possession_types"])
        print(f"  unmatched possession types {a}: {dict(pa)}; {b}: {dict(pb)}")
        print(f"  common-field comparison over matched rows: rows with >=1 changed value {t('rows_with_changed_common_field')}; "
              f"changed field values {t('changed_field_values')}")
        ch: Counter = Counter(); oa: Counter = Counter(); ob: Counter = Counter()
        for x in s:
            ch.update(x["changed_by_field"]); oa.update(x["paths_only_in_a"]); ob.update(x["paths_only_in_b"])
        print(f"  top changed fields: {ch.most_common(8)}; distinct changed paths {len(ch)}")
        print(f"  schema-only paths (row counts): only in {a}: {len(oa)} paths, e.g. {oa.most_common(5)}; only in {b}: {len(ob)} paths, e.g. {ob.most_common(3)}")
        p: Counter = Counter()
        for x in s:
            p.update({k: v for k, v in x["positions"].items() if k != "max_xy_diff_m"})
        print(f"  positions: {dict(p)}; matches with any xy change {sum(1 for x in s if x['positions'].get('player_xy_changed'))}; "
              f"max xy diff {max(x['positions']['max_xy_diff_m'] for x in s):.3f} m")


def main() -> int:
    return L.cli_main(__doc__, "data/processed/audit_1b/event_versions", run_match, aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
