"""Reproducible failure gallery (``_diag/gallery-v2``): before / during / after strips around one GT player.

Every item comes from saved, evaluated outputs (pipeline-v1 decomposition or the engine's eval-v1), is chosen by a
fixed rule and a seeded draw (``SEED``) from **all** qualifying cases (never by eye), and records the source, frame,
GT player and profile so the player's Debugger opens exactly that case (``openItem``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from football_fatigue.video.failures.pipeline import CODES, load_episodes, out_dir
from football_fatigue.video.player.debug import GALLERY_VERSION, Debug

SEED = 11
PER_CATEGORY = 2
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")


def _frames_rows(out_root: Path, sid: str, prof: str):
    with (out_dir(out_root, sid) / prof / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _engine_eval(out_root: Path, sid: str, prof: str) -> dict | None:
    suffix = f"-v2-{prof}-ff8815ed" if prof != "B0" else "-v2-ff8815ed"
    ev = sorted((Path(out_root) / sid / "occlusion" / "evaluation").glob(f"occl-{sid}-0-*{suffix}.eval-v1.json"))
    ev = [e for e in ev if int(e.name.split("-")[4]) > 1000]
    return json.loads(ev[0].read_text(encoding="utf-8")) if ev else None


def build(data_root: Path, out_root: Path) -> Path:
    data_root, out_root = Path(data_root), Path(out_root)
    dest = out_root / "_diag" / GALLERY_VERSION
    if (dest / "index.json").exists():
        raise SystemExit(f"gallery already exists (write-once): {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dbg = Debug(data_root, out_root)
    eps = {(s, p): load_episodes(out_root, s, p) for s in DEV for p in ("B0", "T0", "T0L")}
    cands: dict[str, list[dict[str, Any]]] = {}
    rules: dict[str, str] = {}

    def add(cat: str, rule: str, items: list[dict]):
        rules[cat] = rule
        cands[cat] = sorted(items, key=lambda x: (x["source_id"], x["frame"], x.get("gid") or -1))

    def ep_items(prof: str, pred: Callable[[str, dict], bool]) -> list[dict]:
        out = []
        for s in DEV:
            for e in (eps[(s, prof)] or {}).get("episodes", []):
                if pred(s, e):
                    out.append({"source_id": s, "profile": prof, "kind": "ep", "gid": e["gid"], "frame": e["start"],
                                "start": e["start"], "end": e["end"], "code": e["cause"], "ref": e})
        return out

    add("detector_miss", "T0 episode, cause 'no box after merge', >= 5 frames, GT-visible (cover < 0.1 for most frames)",
        ep_items("T0", lambda s, e: e["cause"] == "miss" and e["frames"] >= 5 and e["visibility"] == "visible"))
    add("turf_filter_deletion", "B0 episode caused by the turf filter, >= 5 frames",
        ep_items("B0", lambda s, e: e["cause"] == "rej_turf" and e["frames"] >= 5))
    add("tracking_loss_despite_detection", "T0 episode, detection passed to ByteTrack but no track output, >= 3 frames",
        ep_items("T0", lambda s, e: e["cause"] == "untracked" and e["frames"] >= 3))
    add("small_player_failure", "T0 'no box' episode, median GT height < 35 px, >= 5 frames",
        ep_items("T0", lambda s, e: e["cause"] == "miss" and e["size"] == "<35 px" and e["frames"] >= 5))
    add("low_light_failure", "T0 'no box' episode in 117092 (low light), >= 5 frames",
        ep_items("T0", lambda s, e: e["cause"] == "miss" and e["low_light"] and e["frames"] >= 5))
    add("heavy_overlap", "T0 episode with GT-derived heavy/hidden visibility (another GT box covers >= 50 %), >= 5 frames",
        ep_items("T0", lambda s, e: e["visibility"] in ("heavy", "hidden") and e["frames"] >= 5))
    # verified ID switch: the previous temporary ID continued on another box (swap) - GT-verified
    sw = []
    for s in DEV:
        for c in (eps[(s, "T0")] or {}).get("id_changes", []):
            if c["previous_id_still_output"]:
                sw.append({"source_id": s, "profile": "T0", "kind": "id", "gid": c["gid"], "frame": c["frame"],
                           "start": c["frame"], "end": c["frame"], "code": "id_changed", "ref": c})
    add("verified_id_switch", "T0 temporary-ID change of one GT player while the previous ID continued on another box", sw)
    # low-confidence detection kept by T0L: T0 'rej_conf' episode whose frames are tracked in T0L
    kept = []
    for s in DEV:
        t0 = [e for e in (eps[(s, "T0")] or {}).get("episodes", []) if e["cause"] == "rej_conf" and e["frames"] >= 3]
        if not t0:
            continue
        want = {(e["gid"], f): e for e in t0 for f in range(e["start"], e["end"] + 1)}
        frames = {f for _g, f in want}
        hit: dict[tuple, int] = {}
        for r in _frames_rows(out_root, s, "T0L"):
            if r["frame"] not in frames:
                continue
            for gid, code, _tid, _c in r["o"]:
                e = want.get((gid, r["frame"]))
                if e is not None and CODES[code] in ("ok", "id_changed"):
                    hit[(e["gid"], e["start"])] = hit.get((e["gid"], e["start"]), 0) + 1
        for e in t0:
            if hit.get((e["gid"], e["start"]), 0) >= max(1, e["frames"] // 2):
                kept.append({"source_id": s, "profile": "T0L", "kind": "ep_t0", "gid": e["gid"], "frame": e["start"],
                             "start": e["start"], "end": e["end"], "code": "rej_conf", "ref": e})
    add("low_conf_kept_by_T0L", "T0 episode 'below conf threshold' (>= 3 frames) whose frames are tracked in T0L (>= half)", kept)
    # success: frames where every evaluable GT player is tracked with an unchanged ID (T0)
    ok = []
    for s in DEV:
        for r in _frames_rows(out_root, s, "T0"):
            codes = [CODES[c] for _g, c, _t, _x in r["o"] if CODES[c] != "not_evaluable"]
            if len(codes) >= 18 and r["frame"] % 25 == 0 and all(c == "ok" for c in codes):
                g = sorted(r["o"])[len(r["o"]) // 2][0]
                ok.append({"source_id": s, "profile": "T0", "kind": "frame", "gid": g, "frame": r["frame"],
                           "start": r["frame"], "end": r["frame"], "code": "ok"})
    add("successful_detection", "T0 frame (every 25th) where >= 18 evaluable GT players are all tracked with unchanged IDs", ok)
    # engine: genuine overlap (PO matched to a GT occlusion episode) and false occlusion warnings (fp_* outcome)
    tp, fp = [], []
    for s in DEV:
        ev = _engine_eval(out_root, s, "T0")
        if not ev:
            continue
        for e in ev["events"]:
            gts = [int(v) for v in (e.get("member_gt") or {}).values() if v is not None]   # track id -> GT id
            if e["event_class"] != "probable_occlusion" or not gts:
                continue
            gid = min(gts)
            item = {"source_id": s, "profile": "T0", "kind": "event", "gid": gid, "frame": e["start_frame"],
                    "start": e["start_frame"], "end": e["end_frame"], "code": "ok", "event_id": e["event_id"],
                    "outcome": e["outcome"]}
            if e["outcome"] == "tp":
                tp.append(item)
            elif e["outcome"].startswith("fp_"):
                fp.append(item)
    add("genuine_overlap", "T0 engine 'probable occlusion' matched to a GT-derived occlusion episode (eval-v1 tp)", tp)
    add("false_occlusion_warning", "T0 engine 'probable occlusion' with an fp_* eval-v1 outcome (GT players were not occluding)", fp)

    items = []
    for cat, pool in cands.items():
        if not pool:
            continue
        idx = sorted(rng.choice(len(pool), size=min(PER_CATEGORY, len(pool)), replace=False))
        for i in idx:
            it = pool[int(i)]
            a, b = it["start"], it["end"]
            ref = it.get("ref") or {}
            if it["kind"] in ("ep", "ep_t0"):
                bf = (ref.get("before") or {}).get("frame", max(0, a - 1))
                af = (ref.get("after") or {}).get("frame", b + 1)
                frames, labels = [bf, a, (a + b) // 2, b, af], ["before", "start", "middle", "end", "after"]
            elif it["kind"] == "id":
                gap = ref.get("frames_since_last_tracked") or 1
                frames, labels = [max(0, a - gap - 5), a - 1, a, a + 5, a + 25], ["before", "prev", "switch", "+5", "+25"]
            elif it["kind"] == "event":
                frames, labels = [max(0, a - 10), a, (a + b) // 2, b, b + 10], ["before", "start", "middle", "end", "after"]
            else:
                frames, labels = [max(0, a - 10), a, a + 10], ["-10", "frame", "+10"]
            png = dbg.strip(it["source_id"], it["gid"], frames, it["profile"], labels)
            name = f"{cat}_{it['source_id'][12:]}_f{it['frame']}_G{it['gid']}_{it['profile']}.png"
            (dest / name).write_bytes(png)
            title = {"ep": f"G{it['gid']} missing f{a}–{b} ({b - a + 1} fr): {ref.get('cause', '')}",
                     "ep_t0": f"G{it['gid']} below 0.4 in T0, f{a}–{b}; shown in T0L",
                     "id": f"G{it['gid']} T{ref.get('from')} → T{ref.get('to')} at f{a}",
                     "event": f"engine probable occlusion f{a}–{b} ({it.get('outcome')}), G{it['gid']}",
                     "frame": f"all players tracked at f{a}"}[it["kind"]]
            items.append({"category": cat, "file": name, "title": title, "caption": f"rule: {rules[cat]}; "
                          f"{len(pool)} qualifying, {min(PER_CATEGORY, len(pool))} drawn (seed {SEED})",
                          "source_id": it["source_id"], "frame": it["frame"], "start": a, "end": b, "gid": it["gid"],
                          "profile": it["profile"] if it["kind"] != "ep_t0" else "T0L",
                          "kind": "ep" if it["kind"] == "ep" else "id" if it["kind"] == "id" else "frame",
                          "code": it["code"], "strip_frames": frames,
                          **({"event_id": it["event_id"], "eval_outcome": it["outcome"]} if it["kind"] == "event" else {})})
    index = {"version": GALLERY_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
             "selection_rule": f"per category, {PER_CATEGORY} items drawn with seed {SEED} from all qualifying cases "
                               "(rules in each caption); development matches only",
             "pool_sizes": {k: len(v) for k, v in cands.items()}, "rules": rules, "items": items}
    (dest / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return dest
