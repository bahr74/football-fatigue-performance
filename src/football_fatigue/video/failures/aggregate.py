"""Cross-match tables of the pipeline decomposition (``_diag/pipeline-v1/aggregate.json`` + ``report.md``)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from football_fatigue.video.failures.pipeline import CODES, LOST, PROFILES, VERSION, load_episodes, load_summary

DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 2) if b else None


def aggregate(out_root: Path, sources: tuple[str, ...] = DEV) -> dict[str, Any]:
    out: dict[str, Any] = {"version": VERSION, "profiles": {}}
    for prof in PROFILES:
        per = {}
        for sid in sources:
            s = load_summary(out_root, sid, prof)
            if s is None:
                continue
            ev = s["observations"]["evaluable"]
            c = s["observations"]["codes"]
            lost = sum(c.get(k, 0) for k in LOST)
            per[sid] = {
                "role": s["role"], "evaluable": ev, "tracked": c.get("ok", 0) + c.get("id_changed", 0),
                "lost": lost, "codes": c,
                "lost_share_of_evaluable_pct": _pct(lost, ev),
                "lost_composition_pct": {k: _pct(c.get(k, 0), lost) for k in LOST},
                "id_changes": s["id_changes"], "episodes": s["episodes"], "coverage": s["per_frame_coverage"],
                "by_size": s["by_size"], "by_visibility": s["by_visibility"], "miss_hints": s["miss_hints"],
                "miss_hints_by_visibility": s["miss_hints_by_visibility"],
                "tracked_under_id_mostly_on_other_player": s["tracked_under_id_mostly_on_other_player"],
            }
        pooled: dict[str, Any] = {}
        for role in ("tune", "check", "all"):
            rs = [v for v in per.values() if role == "all" or v["role"] == role]
            if not rs:
                continue
            c = Counter()
            for v in rs:
                c.update(v["codes"])
            ev = sum(v["evaluable"] for v in rs)
            lost = sum(c[k] for k in LOST)
            eps = Counter()
            for v in rs:
                eps.update(v["episodes"]["by_group"])
            pooled[role] = {"evaluable": ev, "lost": lost, "lost_share_pct": _pct(lost, ev), "codes": dict(c),
                            "lost_composition_pct": {k: _pct(c[k], lost) for k in LOST},
                            "id_changes": sum(v["id_changes"]["total"] for v in rs),
                            "episodes_by_group": dict(eps), "episodes": sum(eps.values())}
        out["profiles"][prof] = {"per_match": per, "pooled": pooled}
    return out


def _episode_examples(out_root: Path, sid: str, prof: str, n: int = 3) -> list[dict]:
    e = load_episodes(out_root, sid, prof)
    if not e:
        return []
    eps = sorted(e["episodes"], key=lambda x: (-x["frames"], x["start"]))
    return eps[:n]


def write(out_root: Path) -> Path:
    agg = aggregate(out_root)
    base = Path(out_root) / "_diag" / VERSION
    (base / "aggregate.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    L = ["# Pipeline decomposition (generated from _diag/pipeline-v1)", "",
         "Each SoccerTrack GT player observation in each frame gets one outcome per profile. "
         "`miss` = no box after the tile merge (detector or merge NMS: not separable from the cache).", ""]
    for prof in PROFILES:
        P = agg["profiles"][prof]
        L += [f"## {prof}", "", "| match | role | evaluable | lost % | miss | rej_conf | rej_turf | rej_class | untracked | ID changes | episodes | mean missing players/frame |",
              "|" + "---|" * 12]
        for sid, v in P["per_match"].items():
            c = v["codes"]
            L.append(f"| {sid[12:]} | {v['role']} | {v['evaluable']} | {v['lost_share_of_evaluable_pct']} | {c.get('miss', 0)} | "
                     f"{c.get('rej_conf', 0)} | {c.get('rej_turf', 0)} | {c.get('rej_class', 0)} | {c.get('untracked', 0)} | "
                     f"{v['id_changes']['total']} | {v['episodes']['total']} | {v['coverage']['mean_missing_players']} |")
        L += ["", "Composition of lost observations (% of lost):", "", "| group | " + " | ".join(LOST) + " |", "|" + "---|" * (len(LOST) + 1)]
        for role, v in P["pooled"].items():
            L.append(f"| {role} | " + " | ".join(str(v['lost_composition_pct'][k]) for k in LOST) + " |")
        L.append("")
    codes_vis: dict[str, Counter] = defaultdict(Counter)
    for v in agg["profiles"]["T0"]["per_match"].values():
        for vis, c in v["by_visibility"].items():
            codes_vis[vis].update(c)
    L += ["## T0 outcome by GT-derived visibility (all matches)", "", "| visibility | n | " + " | ".join(c for c in CODES if c != "not_evaluable") + " |",
          "|" + "---|" * (len(CODES) + 1)]
    for vis, c in codes_vis.items():
        n = sum(c.values())
        L.append(f"| {vis} | {n} | " + " | ".join(f"{_pct(c.get(k, 0), n)}" for k in CODES if k != "not_evaluable") + " |")
    codes_size: dict[str, Counter] = defaultdict(Counter)
    for v in agg["profiles"]["T0"]["per_match"].values():
        for sz, c in v["by_size"].items():
            codes_size[sz].update(c)
    L += ["", "## T0 outcome by GT height (all matches)", "", "| height | n | " + " | ".join(c for c in CODES if c != "not_evaluable") + " |",
          "|" + "---|" * (len(CODES) + 1)]
    for sz in ("<35 px", "35-50 px", "50-80 px", ">=80 px"):
        c = codes_size.get(sz, Counter())
        n = sum(c.values())
        L.append(f"| {sz} | {n} | " + " | ".join(f"{_pct(c.get(k, 0), n)}" for k in CODES if k != "not_evaluable") + " |")
    L += ["", "## Miss hints (T0; GT-geometry hints, not measured causes)", "", "| match | none | merge_suspect | poor_localisation |", "|---|---|---|---|"]
    for sid, v in agg["profiles"]["T0"]["per_match"].items():
        h = v["miss_hints"]
        L.append(f"| {sid[12:]} | {h.get('none', 0)} | {h.get('merge_suspect', 0)} | {h.get('poor_localisation', 0)} |")
    (base / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return base / "report.md"
