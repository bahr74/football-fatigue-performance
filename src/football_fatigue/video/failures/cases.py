"""M13 failure case studies: a reproducible table from the saved pipeline-v1 decomposition (no new inference).

Six categories, each defined by a fixed rule on saved observations; two cases per category drawn with ``SEED`` from all
qualifying cases (pool sizes recorded), plus the fixed reference case 117093 G4 f31-37 used in earlier handoffs.
"Visibility" is GT-derived (cover by other GT boxes), never a confirmed visual occlusion.

For each case: previous / next temporary tracker ID (T##, not an identity), confidence before / after, box height,
GT-derived visibility and overlap, the pipeline stage, whether the same GT player was tracked again (recovered), and
whether the previous temporary ID was output on a *different* GT player during or within 2 s after the gap (inherited).

Output (write-once): ``<out_root>/_diag/cases-m13-v1/{cases.json, cases.md}``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.failures.pipeline import CODES, load_episodes, out_dir

VERSION = "cases-m13-v1"
SEED = 17
PER = 2
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
AFTER = 50                       # frames after the gap searched for the old ID on another player (2 s at 25 fps)

RULES = {
    "heavy_overlap_disappearance": ("T0", "no-box episode >= 5 frames, GT-derived visibility heavy/hidden, not 117092"),
    "turf_filter_false_rejection": ("B0", "B0 episode caused by the turf filter, >= 5 frames (box matched to the GT player)"),
    "low_confidence_interruption": ("T0", "episode caused by conf <= 0.4, >= 3 frames, GT-derived visibility visible/partial"),
    "confirmed_identity_switch": ("T0", "temporary-ID change of a GT player while the previous ID was output elsewhere"),
    "small_player_missed_detection": ("T0", "no-box episode >= 5 frames, median GT height < 35 px, visible, not 117092"),
    "low_light_failure": ("T0", "no-box episode >= 5 frames in 117092 (low light), GT-derived visibility visible"),
}


def _pred(cat: str, sid: str, e: dict) -> bool:
    if cat == "heavy_overlap_disappearance":
        return e["cause"] == "miss" and e["frames"] >= 5 and e["visibility"] in ("heavy", "hidden") and not e["low_light"]
    if cat == "turf_filter_false_rejection":
        return e["cause"] == "rej_turf" and e["frames"] >= 5
    if cat == "low_confidence_interruption":
        return e["cause"] == "rej_conf" and e["frames"] >= 3 and e["visibility"] in ("visible", "partial")
    if cat == "small_player_missed_detection":
        return e["cause"] == "miss" and e["frames"] >= 5 and e["size"] == "<35 px" and e["visibility"] == "visible" \
            and not e["low_light"]
    if cat == "low_light_failure":
        return e["cause"] == "miss" and e["frames"] >= 5 and e["low_light"] and e["visibility"] == "visible"
    return False


class _Rows:
    """Per-frame outcomes of one profile: frame -> {gid: (code, tid, conf)} (loaded once per match/profile)."""

    def __init__(self, out_root: Path, diag: str):
        self.out_root, self.diag, self._c = out_root, diag, {}

    def get(self, sid: str, prof: str) -> list[dict]:
        k = (sid, prof)
        if k not in self._c:
            rows = []
            with (out_dir(self.out_root, sid, self.diag) / prof / "frames.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    rows.append({g: (CODES[c], t, x) for g, c, t, x in r["o"]})
            self._c = {k: rows}                     # keep one match in memory
        return self._c[k]


def _inherited(rows: list[dict], gid: int, tid: int | None, a: int, b: int) -> dict[str, Any] | None:
    if tid is None:
        return None
    for f in range(a, min(len(rows), b + AFTER + 1)):
        for g, (code, t, _c) in rows[f].items():
            if g != gid and t == tid and code in ("ok", "id_changed"):
                return {"frame": f, "gid": g}
    return None


def build(data_root: Path, out_root: Path, diag: str = "pipeline-v1") -> Path:
    out_root = Path(out_root)
    dest = out_root / "_diag" / VERSION
    if (dest / "cases.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    rng = np.random.default_rng(SEED)
    pools: dict[str, list] = {c: [] for c in RULES}
    for sid in DEV:
        for prof in ("B0", "T0"):
            ep = load_episodes(out_root, sid, prof) or {}
            for e in ep.get("episodes", []):
                for cat, (p, _r) in RULES.items():
                    if p == prof and _pred(cat, sid, e):
                        pools[cat].append(("ep", sid, prof, e))
            if prof == "T0":
                for c in ep.get("id_changes", []):
                    if c["previous_id_still_output"]:
                        pools["confirmed_identity_switch"].append(("id", sid, prof, c))
    for cat in pools:
        pools[cat].sort(key=lambda x: (x[1], x[3].get("start", x[3].get("frame")), x[3]["gid"]))
    picks = []
    ref = next(e for e in load_episodes(out_root, "soccertrack-117093", "T0")["episodes"] if e["gid"] == 4 and e["start"] == 31)
    picks.append(("reference (earlier handoffs)", ("ep", "soccertrack-117093", "T0", ref)))
    for cat, pool in pools.items():
        for i in sorted(rng.choice(len(pool), size=min(PER, len(pool)), replace=False)):
            picks.append((cat, pool[int(i)]))
    rows_cache = _Rows(out_root, diag)
    cases = []
    for cat, (kind, sid, prof, e) in picks:
        rows = rows_cache.get(sid, prof)
        if kind == "ep":
            a, b = e["start"], e["end"]
            prev_tid = (e["before"] or {}).get("tid")
            nxt = e["after"] or {}
            t0_during = None
            if prof == "B0":                               # what T0 did with the same frames
                t0 = rows_cache.get(sid, "T0")
                t0_during = sorted({t0[f].get(e["gid"], ("absent",))[0] for f in range(a, b + 1)})
                rows = rows_cache.get(sid, prof)
            cases.append({
                "category": cat, "source_id": sid, "profile": prof, "gid": e["gid"], "start": a, "end": b,
                "frames": e["frames"], "previous_tid": prev_tid, "next_tid": nxt.get("tid"),
                "conf_before": (e["before"] or {}).get("conf"), "conf_after": nxt.get("conf"),
                "median_height_px": e["median_height"], "size": e["size"], "gt_visibility": e["visibility"],
                "max_gt_cover": e["max_cover"], "max_iou_with_other_gt": e["max_iou_with_other_gt"],
                "nearest_gt": e["nearest_gt"], "stage": e["cause"], "stage_codes": e["codes"],
                "miss_hints": e.get("miss_hints"), "recovered": e["after"] is not None,
                "same_tid_after": e["id_after_gap"], "old_tid_on_other_player": _inherited(rows, e["gid"], prev_tid, a, b),
                "t0_codes_during": t0_during, "low_light": e["low_light"],
                "open": f"?source={sid}&ti={prof}&debug=1&gid={e['gid']}&frame={max(0, a - 1)}"})
        else:
            f = e["frame"]
            cur = rows[f].get(e["gid"], (None, None, None))
            prv = next(((rows[k][e["gid"]][2], k) for k in range(f - 1, max(-1, f - 300), -1)
                        if e["gid"] in rows[k] and rows[k][e["gid"]][1] == e["from"]), (None, None))
            cases.append({
                "category": cat, "source_id": sid, "profile": prof, "gid": e["gid"], "start": f, "end": f,
                "frames": e["frames_since_last_tracked"], "previous_tid": e["from"], "next_tid": e["to"],
                "conf_before": prv[0], "conf_after": cur[2], "stage": "id_changed",
                "gap_frames_before_change": e["frames_since_last_tracked"], "recovered": True, "same_tid_after": "changed",
                "old_tid_on_other_player": _inherited(rows, e["gid"], e["from"], max(0, f - e["frames_since_last_tracked"]), f),
                "open": f"?source={sid}&ti={prof}&debug=1&gid={e['gid']}&frame={max(0, f - e['frames_since_last_tracked'] - 5)}"})
    out = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
           "rules": {k: {"profile": p, "rule": r} for k, (p, r) in RULES.items()},
           "pool_sizes": {k: len(v) for k, v in pools.items()}, "cases": cases,
           "notes": ["T## are temporary ByteTrack IDs of the named profile, G## SoccerTrack GT players",
                     "visibility/overlap are GT-derived (GT-GT box cover), not confirmed visual occlusion",
                     f"old_tid_on_other_player: previous ID output on another GT player during the gap or <= {AFTER} frames after"]}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cases.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (dest / "cases.md").write_text(_markdown(out), encoding="utf-8")
    return dest


def _markdown(out: dict) -> str:
    lines = ["| category | match | profile | GT | frames | prev T | next T | conf before → after | height px | "
             "GT visibility, typical (max cover) | stage | recovered | old T on other player | open |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in out["cases"]:
        inh = c.get("old_tid_on_other_player")
        lines.append(
            f"| {c['category']} | {c['source_id'][12:]} | {c['profile']} | G{c['gid']} | {c['start']}–{c['end']} | "
            f"{'T' + str(c['previous_tid']) if c['previous_tid'] is not None else '–'} | "
            f"{'T' + str(c['next_tid']) if c['next_tid'] is not None else '–'} | {c.get('conf_before')} → {c.get('conf_after')} | "
            f"{c.get('median_height_px', '–')} | {c.get('gt_visibility', '–')} ({c.get('max_gt_cover', '–')}) | {c['stage']}"
            f"{' (T0: ' + ', '.join(c['t0_codes_during']) + ')' if c.get('t0_codes_during') else ''} | "
            f"{'yes, ' + str(c['same_tid_after']) + ' ID' if c['recovered'] else 'no'} | "
            f"{'G' + str(inh['gid']) + ' @f' + str(inh['frame']) if inh else 'no'} | `{c['open']}` |")
    return "\n".join(lines) + "\n"
