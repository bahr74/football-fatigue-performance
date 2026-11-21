"""Pipeline decomposition of every ground-truth player observation (development matches, CPU, cached outputs only).

For each SoccerTrack GT box in each frame, and for each tracking profile (B0, T0, T0L), the observation is followed
through the pipeline **in order** and given exactly one outcome code:

====================  =====  ==============================================================================
code                  group  meaning (what was measured)
====================  =====  ==============================================================================
``ok``                ok     a tracker output box (= a measured detection) matches the player, same temporary ID
                             as the player's previous tracked frame (or its first one)
``id_changed``        D      tracked, but the temporary ID differs from the player's previous tracked frame
``untracked``         C      a detection that this profile passes to ByteTrack matches the player, but ByteTrack
                             output no box for it (track not activated / not matched / not confirmed: the
                             production tracker does not export which)
``rej_conf``          B      the matching detection has confidence <= the profile's tracker-input threshold
``rej_turf``          B      the matching detection was removed by the turf (grass-colour) filter (B0 only)
``rej_class``         B      the matching cached box carries a non-person class (ball / unmapped)
``miss``              E      no cached box matches (IoU >= 0.5) after the tile merge
``not_evaluable``     F      GT box off the annotated pitch (> 0.5 box heights) or touching the frame border
====================  =====  ==============================================================================

``miss`` is "absent from the post-merge cache". The cache stores only boxes after tile inference and merge NMS, so a
miss cannot be split into "the model never produced a box" and "the box was suppressed by the merge"; each miss
carries a *hint* (``poor_localisation``: an unclaimed box with IoU 0.2-0.5; ``merge_suspect``: a surviving box of
another player overlaps it with IoU >= 0.3 or covers >= 70 % of it, so the merge could have removed it; ``none``)
and the hint is never reported as a measured cause. Separating the two needs the raw per-tile capture
(``video.detection.capture``, GPU, not approved).

Matching is one-to-one (Hungarian, IoU >= 0.5) between the frame's GT boxes and **all** cached boxes (conf > 0.1),
so the same detection is followed through every profile. Visibility (visible / partial / heavy / hidden) is derived
from GT boxes overlapping each other: it is a GT-geometry estimate, not a human-confirmed occlusion.

Outputs (write-once): ``<out_root>/_diag/pipeline-v1/<source>/<profile>/{frames.jsonl, episodes.json, summary.json}``
and ``<out_root>/_diag/pipeline-v1/<source>/manifest.json``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.diagnose import frame_size, iter_cache
from football_fatigue.video.detection.stages import (Params, gt_cover, iou_matrix, cover_matrix, match,
                                                       outside_distance, size_bin, visibility)
from football_fatigue.video.sources import SOURCES, guard_not_reserved, play_area_polygon
from football_fatigue.video.tracking import TRACKER_INPUTS, tracks_dir

VERSION = "pipeline-v1"
PROFILES = ("B0", "T0", "T0L")
CODES = ("ok", "id_changed", "untracked", "rej_conf", "rej_turf", "rej_class", "miss", "not_evaluable")
GROUP = {"ok": "ok", "id_changed": "D", "untracked": "C", "rej_conf": "B", "rej_turf": "B", "rej_class": "B",
         "miss": "E", "not_evaluable": "F"}
TRACKED = ("ok", "id_changed")
LOST = ("untracked", "rej_conf", "rej_turf", "rej_class", "miss")
CODE_TEXT = {
    "ok": "tracked (same temporary ID)",
    "id_changed": "tracked, temporary ID changed",
    "untracked": "detected and passed to ByteTrack, but no track output",
    "rej_conf": "detected below the tracker confidence threshold",
    "rej_turf": "detected, removed by the turf filter",
    "rej_class": "detected with a non-person class",
    "miss": "no box after the tile merge (detector or merge; not separable)",
    "not_evaluable": "outside the evaluation assumptions (off pitch / frame border)",
}
LOW_LIGHT = {"soccertrack-117092"}
DURATION_BINS = ((1, 1, "1"), (2, 5, "2-5"), (6, 25, "6-25"), (26, 75, "26-75"), (76, 250, "76-250"),
                 (251, 10 ** 9, ">250"))
P = Params()


def out_dir(out_root: Path, source_id: str, version: str = VERSION) -> Path:
    return Path(out_root) / "_diag" / version / source_id


def _load_tracks(data_root: Path, out_root: Path, sid: str, profile: str) -> dict[int, list[tuple[int, tuple]]]:
    path = tracks_dir(data_root, out_root, sid, profile) / "frames.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no {profile} tracks for {sid}: {path} (build with python -m football_fatigue.video.tracking)")
    out: dict[int, list] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[int(r["frame_num"])] = [(int(p["track_id"]), tuple(float(v) for v in p["bbox_xyxy_px"]))
                                        for p in r["players"]]
    return out


def _passes(d, profile: str) -> str | None:
    """None if the profile passes the detection to ByteTrack, else the rejection code (pipeline order)."""
    f = TRACKER_INPUTS[profile]
    if not d.person:
        return "rej_class"
    if not d.conf > f.threshold:
        return "rej_conf"
    if f.turf == "cached" and d.turf is False:
        return "rej_turf"
    return None


def _duration_bin(n: int) -> str:
    return next(name for lo, hi, name in DURATION_BINS if lo <= n <= hi)


def decompose(data_root: Path, out_root: Path, source_id: str, profiles: tuple[str, ...] = PROFILES,
              max_frames: int | None = None, overwrite: bool = False, version: str = VERSION) -> Path:
    """``version`` names the output folder only (the method stays pipeline-v1); new profiles go to a new folder."""
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"], spec["gt"])
    from football_fatigue.vision.mot_eval import load_reference

    dest = out_dir(out_root, source_id, version)
    if (dest / "manifest.json").exists() and not overwrite:
        raise SystemExit(f"output already exists (write-once): {dest}")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    t0 = time.perf_counter()
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    size = frame_size(data_root, source_id)
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    tracks = {p: _load_tracks(data_root, out_root, source_id, p) for p in profiles}
    fh = {}
    for p in profiles:
        (tmp / p).mkdir(parents=True, exist_ok=True)
        fh[p] = (tmp / p / "frames.jsonl").open("w", encoding="utf-8", newline="\n")
    # per-profile state
    last_tid: dict[str, dict[int, int]] = {p: {} for p in profiles}
    obs: dict[str, dict[int, list]] = {p: defaultdict(list) for p in profiles}   # gid -> [(frame, code, tid, conf)]
    idsw: dict[str, list] = {p: [] for p in profiles}
    tid_gt: dict[str, dict[int, Counter]] = {p: defaultdict(Counter) for p in profiles}
    attrs: dict[int, dict[int, tuple]] = defaultdict(dict)      # gid -> frame -> (cover, height, visibility, hint, near)
    frames_done = 0
    for n, (fr, dets) in enumerate(iter_cache(data_root, source_id)):
        if max_frames is not None and n >= max_frames:
            break
        g = gt.get(fr, [])
        gids = [gid for gid, _b in g]
        gboxes = [tuple(float(v) for v in b) for _gid, b in g]
        cover = gt_cover(gboxes, P.cover_grid) if gboxes else np.zeros(0)
        dboxes = [d.box for d in dets]
        g_d = iou_matrix(gboxes, dboxes)
        gc_d = cover_matrix(gboxes, dboxes)
        m = dict(match(g_d, list(range(len(g))), list(range(len(dets))), P.match_iou)) if g and dets else {}
        claimed = set(m.values())
        g_g = iou_matrix(gboxes, gboxes)
        rows_common = []
        for gi, (gid, box) in enumerate(zip(gids, gboxes)):
            vis = visibility(box, float(cover[gi]), outside_distance(box, poly), size, P)
            hint = None
            if gi not in m and vis not in ("off_pitch", "uncertain"):
                free = [d for d in range(len(dets)) if d not in claimed and dets[d].person and P.poor_iou <= g_d[gi, d] < P.match_iou]
                near = [d for d in claimed if dets[d].person and (g_d[gi, d] >= P.suspect_iou or gc_d[gi, d] >= P.suspect_cover)]
                hint = "poor_localisation" if free else "merge_suspect" if near else "none"
            others = [j for j in range(len(g)) if j != gi]
            nb = max(others, key=lambda j: g_g[gi, j]) if others else None
            near_gid = gids[nb] if nb is not None and g_g[gi, nb] > 0 else None
            attrs[gid][fr] = (round(float(cover[gi]), 3), round(box[3] - box[1], 1), vis, hint, near_gid,
                              round(float(g_g[gi, nb]), 3) if near_gid is not None else 0.0)
            rows_common.append((gi, gid, box, vis))
        for p in profiles:
            tr = tracks[p].get(fr, [])
            tboxes = [b for _t, b in tr]
            d_t = iou_matrix(dboxes, tboxes) if dboxes and tboxes else np.zeros((len(dboxes), len(tboxes)))
            g_t = iou_matrix(gboxes, tboxes) if gboxes and tboxes else np.zeros((len(gboxes), len(tboxes)))
            out_rows = []
            for gi, gid, box, vis in rows_common:
                tid = conf = None
                di = m.get(gi)
                if di is not None:
                    conf = round(dets[di].conf, 3)
                if vis in ("off_pitch", "uncertain"):
                    code = "not_evaluable"
                    # still record a track if one is on the player (for continuity of temporary IDs)
                    if tboxes and g_t[gi].max() >= P.match_iou:
                        tid = tr[int(np.argmax(g_t[gi]))][0]
                elif di is None:
                    code = "miss"
                else:
                    code = _passes(dets[di], p)
                    if code is None:
                        hit = np.nonzero(d_t[di] > 0.99)[0] if tboxes else []
                        if len(hit):
                            tid = tr[int(hit[0])][0]
                            prev = last_tid[p].get(gid)
                            code = "ok" if prev is None or prev == tid else "id_changed"
                            if code == "id_changed":
                                alive = any(t == prev for t, _b in tr)
                                prev_frame = max((o[0] for o in obs[p][gid] if o[1] in TRACKED), default=None)
                                idsw[p].append({"frame": fr, "gid": gid, "from": prev, "to": tid,
                                                "frames_since_last_tracked": fr - prev_frame if prev_frame is not None else None,
                                                "previous_id_still_output": alive})
                        else:
                            code = "untracked"
                if tid is not None:
                    last_tid[p][gid] = tid
                    if code in TRACKED:
                        tid_gt[p][tid][gid] += 1
                obs[p][gid].append((fr, code, tid, conf))
                out_rows.append([gid, CODES.index(code), tid, conf])
            fh[p].write(json.dumps({"frame": fr, "o": out_rows}, separators=(",", ":")) + "\n")
        frames_done += 1
    for f in fh.values():
        f.close()
    for p in profiles:
        eps, summary = _summarise(source_id, p, obs[p], attrs, idsw[p], tid_gt[p], frames_done)
        (tmp / p / "episodes.json").write_text(json.dumps({**eps, "id_changes": idsw[p]}, separators=(",", ":")),
                                               encoding="utf-8")
        (tmp / p / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    manifest = {"version": VERSION, "status": "complete" if max_frames is None else "subset", "source_id": source_id,
                "role": spec["role"], "created_at_utc": datetime.now(timezone.utc).isoformat(), "profiles": list(profiles),
                "frames": frames_done, "seconds": round(time.perf_counter() - t0, 1), "codes": list(CODES),
                "code_text": CODE_TEXT, "groups": GROUP,
                "inputs": {"detection_cache_run": spec["cache_run"], "gt": spec["gt"],
                           "tracks": {p: str(tracks_dir(data_root, out_root, source_id, p)) for p in profiles}},
                "params": {"match_iou": P.match_iou, "poor_iou": P.poor_iou, "suspect_iou": P.suspect_iou,
                           "suspect_cover": P.suspect_cover, "off_pitch_heights": P.off_pitch_heights,
                           "border_px": P.border_px, "cover_grid": list(P.cover_grid)},
                "frames_jsonl_row": "[gt id, code index into codes, temporary track id or null, matched cached box confidence or null]",
                "limitations": ["the cache holds post-merge boxes only: 'miss' includes possible merge-NMS suppression",
                                "ByteTrack internals (activation, lost/removed state) are not exported: 'untracked' is not split",
                                "visibility is derived from overlapping GT boxes, not human-confirmed occlusion",
                                "SoccerTrack GT annotates the 22 players only (referees/staff are absent)"]}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def _summarise(sid: str, profile: str, obs: dict[int, list], attrs: dict[int, dict[int, tuple]], idsw: list,
               tid_gt: dict[int, Counter], frames_done: int) -> tuple[dict[str, list], dict[str, Any]]:
    codes = Counter()
    by_size: dict[str, Counter] = defaultdict(Counter)
    by_vis: dict[str, Counter] = defaultdict(Counter)
    hints = Counter()
    hint_by_vis: dict[str, Counter] = defaultdict(Counter)
    per_frame_eval: Counter = Counter()
    per_frame_tracked: Counter = Counter()
    dominant = {tid: c.most_common(1)[0][0] for tid, c in tid_gt.items() if c}
    wrong_owner = 0
    episodes = []
    exits = []
    for gid, rows in obs.items():
        rows.sort()
        run: list = []
        ne: list = []

        def close_ne():
            if ne:
                vis = Counter(attrs[gid][f][2] for f in ne).most_common(1)[0][0]
                exits.append({"gid": gid, "start": ne[0], "end": ne[-1], "frames": len(ne),
                              "reason": "off_pitch" if vis == "off_pitch" else "frame_border"})
                ne.clear()
        last_tracked: tuple | None = None

        def close(after: tuple | None):
            if not run:
                return
            a = [attrs[gid][r[0]] for r in run]
            cs = Counter(r[1] for r in run)
            hs = Counter(x[3] for x in a if x[3])
            cov = [x[0] for x in a]
            hts = [x[1] for x in a]
            near = Counter(x[4] for x in a if x[4] is not None)
            first_code = run[0][1]
            ep = {"gid": gid, "start": run[0][0], "end": run[-1][0], "frames": len(run),
                  "duration_bin": _duration_bin(len(run)), "cause": cs.most_common(1)[0][0], "first_code": first_code,
                  "codes": dict(cs), "group": GROUP[cs.most_common(1)[0][0]],
                  "miss_hints": dict(hs),
                  "before": None if last_tracked is None else {"frame": last_tracked[0], "tid": last_tracked[2], "conf": last_tracked[3]},
                  "after": None if after is None else {"frame": after[0], "tid": after[2], "conf": after[3]},
                  "max_cover": max(cov), "median_height": float(np.median(hts)), "size": size_bin(float(np.median(hts))),
                  "visibility": Counter(x[2] for x in a).most_common(1)[0][0],
                  "nearest_gt": near.most_common(1)[0][0] if near else None,
                  "max_iou_with_other_gt": max(x[5] for x in a),
                  "low_light": sid in LOW_LIGHT}
            ep["id_after_gap"] = (None if ep["before"] is None or ep["after"] is None
                                  else "same" if ep["before"]["tid"] == ep["after"]["tid"] else "changed")
            episodes.append(ep)
            run.clear()

        prev_frame = None
        for r in rows:
            fr, code, tid, conf = r
            a = attrs[gid][fr]
            codes[code] += 1
            if code != "not_evaluable":
                by_size[size_bin(a[1])][code] += 1
                by_vis[a[2]][code] += 1
                per_frame_eval[fr] += 1
                if code in TRACKED:
                    per_frame_tracked[fr] += 1
                    if dominant.get(tid) not in (None, gid):
                        wrong_owner += 1
            if code == "miss":
                hints[a[3]] += 1
                hint_by_vis[a[2]][a[3]] += 1
            if prev_frame is not None and fr != prev_frame + 1:   # GT absent in between: nothing continues across it
                close(None)
                close_ne()
                last_tracked = None
            if code == "not_evaluable":
                ne.append(fr)
            else:
                close_ne()
            if code in LOST:
                run.append(r)
            else:
                close(r if code in TRACKED else None)
                if code in TRACKED:
                    last_tracked = r
                elif code == "not_evaluable":
                    last_tracked = None
            prev_frame = fr
        close(None)
        close_ne()
    n_eval = sum(v for k, v in codes.items() if k != "not_evaluable")
    cov = np.array([per_frame_tracked[f] / per_frame_eval[f] for f in sorted(per_frame_eval)]) if per_frame_eval else np.zeros(0)
    missing = np.array([per_frame_eval[f] - per_frame_tracked[f] for f in sorted(per_frame_eval)]) if per_frame_eval else np.zeros(0)
    ep_cause = Counter(e["cause"] for e in episodes)
    ep_group = Counter(e["group"] for e in episodes)
    dur = defaultdict(Counter)
    for e in episodes:
        dur[e["group"]][e["duration_bin"]] += 1
    frames_by_group = Counter()
    for e in episodes:
        frames_by_group[e["group"]] += e["frames"]
    summary = {
        "version": VERSION, "source_id": sid, "role": SOURCES[sid]["role"], "profile": profile, "frames": frames_done,
        "low_light": sid in LOW_LIGHT,
        "observations": {"total": sum(codes.values()), "evaluable": n_eval, "codes": dict(codes),
                         "share_of_evaluable": {k: round(codes[k] / n_eval, 4) for k in CODES if k != "not_evaluable" and n_eval}},
        "lost_observations_by_group": {g: sum(codes[c] for c in LOST if GROUP[c] == g) for g in ("B", "C", "E")},
        "by_size": {k: dict(v) for k, v in sorted(by_size.items())},
        "by_visibility": {k: dict(v) for k, v in by_visibility_order(by_vis)},
        "miss_hints": dict(hints), "miss_hints_by_visibility": {k: dict(v) for k, v in hint_by_vis.items()},
        "id_changes": {"total": len(idsw), "direct": sum(1 for x in idsw if x["frames_since_last_tracked"] == 1),
                       "after_gap": sum(1 for x in idsw if (x["frames_since_last_tracked"] or 0) > 1),
                       "previous_id_still_output_elsewhere": sum(1 for x in idsw if x["previous_id_still_output"])},
        "tracked_under_id_mostly_on_other_player": wrong_owner,
        "episodes": {"total": len(episodes), "by_cause": dict(ep_cause), "by_group": dict(ep_group),
                     "frames_by_group": dict(frames_by_group),
                     "duration_by_group": {g: dict(v) for g, v in dur.items()},
                     "reappeared_same_id": sum(1 for e in episodes if e["id_after_gap"] == "same"),
                     "reappeared_new_id": sum(1 for e in episodes if e["id_after_gap"] == "changed")},
        "per_frame_coverage": {"frames": int(len(cov)), "mean": round(float(cov.mean()), 4) if len(cov) else None,
                               "p10": round(float(np.quantile(cov, 0.1)), 4) if len(cov) else None,
                               "median": round(float(np.median(cov)), 4) if len(cov) else None,
                               "frames_all_tracked": int((missing == 0).sum()),
                               "frames_ge3_missing": int((missing >= 3).sum()),
                               "mean_missing_players": round(float(missing.mean()), 3) if len(missing) else None},
    }
    summary["view_exits"] = {"runs": len(exits), "frames": sum(x["frames"] for x in exits)}
    return {"episodes": episodes, "exits": exits}, summary


def by_visibility_order(d: dict) -> list:
    order = ("visible", "partial", "heavy", "hidden", "off_pitch", "uncertain")
    return sorted(d.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)


def load_summary(out_root: Path, source_id: str, profile: str) -> dict[str, Any] | None:
    p = out_dir(out_root, source_id) / profile / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_episodes(out_root: Path, source_id: str, profile: str) -> dict[str, Any] | None:
    p = out_dir(out_root, source_id) / profile / "episodes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
