"""Team-shape analytics from a saved CV run (image space) with reference validation.

Per frame and per colour group (blue, white) the foot points of labelled detections give:

* ``length``  - 10th-90th percentile span along image x (the pitch's long axis in this camera view);
* ``width``   - 10th-90th percentile span along image y (across the pitch, strongly foreshortened);
* ``spread``  - RMS distance of players from the group's median position, ignoring the single farthest
  player when 8 or more are labelled (smaller = more compact);

each also divided by the group's median box height ("player heights", ``*_bh``), which removes most of the
effect of the group being nearer to or farther from the camera. These are *image-space* quantities, not
metres: perspective and lens distortion remain. Groups with fewer than ``MIN_PLAYERS`` labelled detections in a
frame are skipped, never zero-filled. Notable moments are found from each group's own distribution, so a constant
perspective offset does not create an "event". Referees, goalkeepers and unknown-colour boxes are excluded.

``validate`` compares the same metrics with (a) the SoccerTrack reference tracks (evaluation only) and (b) the
prototype pitch projection (``pitch_prototype``), and flags frames with too few players, doubtful labels or low
confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEAMS = ("blue", "white")
MIN_PLAYERS = 6
FPS_DEFAULT = 25.0
LOW_CONFIDENCE = 0.55
UNKNOWN_DOUBT = 3  # >= this many unknown-colour boxes in a frame makes its colour split doubtful
MIN_FULL_FRAME = 14  # fewer labelled blue+white boxes than this (of 20 outfield players) means many are missing
METRICS = ("spread_bh", "length_bh", "width_bh", "cx")
MIN_DEVIATION = 0.2  # a moment must differ from the group's own median by at least 20%


def robust_spread(x: np.ndarray, y: np.ndarray) -> float:
    """RMS distance from the median position, ignoring the farthest point when 8+ are given."""
    d = np.hypot(x - np.median(x), y - np.median(y))
    if len(d) >= 8:
        d = np.sort(d)[:-1]  # a single stray detection must not define a team's spread
    return float(np.sqrt(np.mean(d ** 2)))


def _group_metrics(x: np.ndarray, y: np.ndarray, h: np.ndarray) -> dict | None:
    if len(x) < MIN_PLAYERS:
        return None
    mh = float(np.median(h))
    if mh <= 1:
        return None
    cx, cy = float(np.median(x)), float(np.median(y))
    p = lambda a, q: float(np.percentile(a, q))
    spread = robust_spread(x, y)
    return {
        "n": int(len(x)), "cx": cx, "cy": cy, "mh": mh,
        "length_px": p(x, 90) - p(x, 10), "width_px": p(y, 90) - p(y, 10), "spread_px": spread,
        "length_bh": (p(x, 90) - p(x, 10)) / mh, "width_bh": (p(y, 90) - p(y, 10)) / mh, "spread_bh": spread / mh,
    }


def read_frames(path: Path) -> list[dict]:
    """Per-frame inputs: foot points, box heights and label counts for the two groups."""
    out = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            groups = {t: ([], [], []) for t in TEAMS}
            unknown = other = 0
            confs = []
            for p in rec.get("players") or []:
                kit = p.get("kit_color")
                box = p.get("bbox_xyxy_px")
                if p.get("confidence") is not None:
                    confs.append(float(p["confidence"]))
                if kit in groups and box and p.get("pixel_x") is not None:
                    g = groups[kit]
                    g[0].append(float(p["pixel_x"]))
                    g[1].append(float(p["pixel_y"]))
                    g[2].append(float(box[3] - box[1]))
                elif kit == "other":
                    other += 1
                else:
                    unknown += 1
            out.append({
                "frame": int(rec["frame_num"]), "fps": float(rec.get("fps") or FPS_DEFAULT),
                "groups": {t: tuple(np.array(a, float) for a in g) for t, g in groups.items()},
                "unknown": unknown, "other": other, "mean_conf": float(np.mean(confs)) if confs else None,
            })
    return out


def reference_frames(mot: dict, kit_reference: dict) -> list[dict]:
    """The same per-frame inputs from reference tracks (evaluation only; never drawn)."""
    ids = {t: set(kit_reference.get(t, [])) for t in TEAMS}
    out = []
    for frame in sorted(mot):
        groups = {}
        for t in TEAMS:
            boxes = [b for i, b in mot[frame] if i in ids[t]]
            a = np.array(boxes, float).reshape(-1, 4)
            groups[t] = ((a[:, 0] + a[:, 2]) / 2, a[:, 3], a[:, 3] - a[:, 1])
        out.append({"frame": frame, "fps": FPS_DEFAULT, "groups": groups, "unknown": 0, "other": 0, "mean_conf": None})
    return out


def per_frame(frames: list[dict]) -> dict[str, list[dict | None]]:
    return {t: [_group_metrics(*f["groups"][t]) for f in frames] for t in TEAMS}


def bin_series(frames: list[dict], metrics: dict[str, list[dict | None]], fps: float, keys=METRICS) -> dict:
    """1-second median of each metric; a bin needs >= half its frames valid, otherwise it is null."""
    per = max(1, int(round(fps)))
    n_bins = (frames[-1]["frame"] - frames[0]["frame"]) // per + 1 if frames else 0
    first = frames[0]["frame"] if frames else 0
    out = {"t": [round((first + b * per) / fps, 2) for b in range(n_bins)]}
    for t in TEAMS:
        buckets: dict[int, list[dict]] = {}
        totals: dict[int, int] = {}
        for f, m in zip(frames, metrics[t]):
            b = (f["frame"] - first) // per
            totals[b] = totals.get(b, 0) + 1
            if m:
                buckets.setdefault(b, []).append(m)
        series = {k: [] for k in (*keys, "n")}
        for b in range(n_bins):
            rows = buckets.get(b)
            ok = rows and len(rows) >= 0.5 * totals.get(b, 1)
            for k in (*keys, "n"):
                series[k].append(round(float(np.median([r[k] for r in rows])), 3) if ok else None)
        out[t] = series
    return out


def _arr(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], float)


def _rolling_median(a: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.full_like(a, np.nan)
    half = k // 2
    for i in range(len(a)):
        w = a[max(0, i - half): i + half + 1]
        w = w[~np.isnan(w)]
        if len(w) >= (k + 1) // 2:
            out[i] = np.median(w)
    return out


def _windows(mask: np.ndarray, min_len: int, merge_gap: int = 2) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a + 1 >= min_len]


def _percent_vs_median(value: float, median: float) -> int:
    return int(round(100 * value / median)) if median else 0


def _win_n(series: dict, teams, i0: int, i1: int) -> float | None:
    """Median labelled players per group over a window (lowest across the groups involved)."""
    vals = [np.nanmedian(_arr(series[t]["n"])[i0:i1 + 1]) for t in teams]
    vals = [v for v in vals if not np.isnan(v)]
    return round(float(min(vals)), 1) if vals else None


def find_events(series: dict, extent_x: tuple[float, float]) -> list[dict]:
    """Notable spatial moments from each group's own distribution (all image-space)."""
    t = series["t"]
    events: list[dict] = []
    metric_of = {"spread": "spread_bh", "compact": "spread_bh", "stretch": "length_bh", "squeeze": "length_bh", "gap": "spread_bh", "shift": "cx"}
    sm = {team: {k: _rolling_median(_arr(series[team][k])) for k in ("spread_bh", "length_bh", "cx")} for team in TEAMS}

    def add(kind, team, i0, i1, values, text, strength):
        peak_i = i0 + int(np.nanargmax(values[i0:i1 + 1]) if kind in ("spread", "stretch", "gap") else np.nanargmin(values[i0:i1 + 1]))
        cx = np.nanmedian(_arr(series[team]["cx"])[i0:i1 + 1]) if team in TEAMS else np.nan
        frac = (cx - extent_x[0]) / max(1e-9, extent_x[1] - extent_x[0]) if not np.isnan(cx) else None
        events.append({
            "type": kind, "metric": metric_of[kind], "team": team, "start_s": t[i0], "end_s": t[i1] + 1.0, "peak_s": t[peak_i],
            "description": text, "strength": round(float(strength), 2), "mode": "image-space",
            "near_image_edge": bool(frac is not None and (frac < 0.12 or frac > 0.88)),
            "players_per_group": _win_n(series, (team,), i0, i1),
        })

    for team in TEAMS:
        s = sm[team]["spread_bh"]
        valid = s[~np.isnan(s)]
        if len(valid) < 20:
            continue
        med, hi, lo = np.median(valid), np.percentile(valid, 90), np.percentile(valid, 10)
        for a, b in sorted(_windows(s >= max(hi, med * (1 + MIN_DEVIATION)), 3), key=lambda w: -np.nanmax(s[w[0]:w[1] + 1]))[:3]:
            peak = float(np.nanmax(s[a:b + 1]))
            add("spread", team, a, b, s,
                f"{team.title()} were unusually spread out (spread index {_percent_vs_median(peak, med)} where 100 is their match median).",
                peak / med)
        for a, b in sorted(_windows(s <= min(lo, med * (1 - MIN_DEVIATION)), 3), key=lambda w: np.nanmin(s[w[0]:w[1] + 1]))[:3]:
            low = float(np.nanmin(s[a:b + 1]))
            add("compact", team, a, b, s,
                f"{team.title()} were unusually compact (spread index {_percent_vs_median(low, med)} where 100 is their match median).",
                med / max(low, 1e-9))
        s = sm[team]["length_bh"]
        valid = s[~np.isnan(s)]
        if len(valid) >= 20:
            med, hi, lo = np.median(valid), np.percentile(valid, 90), np.percentile(valid, 10)
            for a, b in sorted(_windows(s >= max(hi, med * (1 + MIN_DEVIATION)), 3), key=lambda w: -np.nanmax(s[w[0]:w[1] + 1]))[:2]:
                peak = float(np.nanmax(s[a:b + 1]))
                add("stretch", team, a, b, s,
                    f"{team.title()} were stretched along the pitch (length index {_percent_vs_median(peak, med)}, 100 = their match median).", peak / med)
            for a, b in sorted(_windows(s <= min(lo, med * (1 - MIN_DEVIATION)), 3), key=lambda w: np.nanmin(s[w[0]:w[1] + 1]))[:2]:
                low = float(np.nanmin(s[a:b + 1]))
                add("squeeze", team, a, b, s,
                    f"{team.title()} were squeezed together along the pitch (length index {_percent_vs_median(low, med)}, 100 = their match median).", med / max(low, 1e-9))
        # fast shift of the whole block along image x over 4 s, relative to the group's own typical spread
        step = 4
        cx = _arr(series[team]["cx"])
        base = np.nanmedian(_arr(series[team]["spread_px"]))
        if not np.isnan(base) and base > 0:
            shift = np.full(len(cx), np.nan)
            for i in range(len(cx) - step):
                if not np.isnan(cx[i]) and not np.isnan(cx[i + step]):
                    shift[i] = (cx[i + step] - cx[i]) / base
            mag = np.abs(shift)
            ok = mag[~np.isnan(mag)]
            if len(ok) >= 20:
                thr = np.percentile(ok, 95)
                for a, b in sorted(_windows(mag >= thr, 2), key=lambda w: -np.nanmax(mag[w[0]:w[1] + 1]))[:2]:
                    i = a + int(np.nanargmax(mag[a:b + 1]))
                    direction = "right" if shift[i] > 0 else "left"
                    events.append({
                        "type": "shift", "metric": "cx", "team": team, "start_s": t[a], "end_s": t[min(len(t) - 1, b + step)] + 1.0, "peak_s": t[i],
                        "description": f"{team.title()} moved as a block to the {direction} of the picture ({abs(shift[i]):.1f}x their own spread in 4 s).",
                        "strength": round(float(abs(shift[i])), 2), "mode": "image-space", "near_image_edge": False,
                        "players_per_group": _win_n(series, (team,), a, min(len(t) - 1, b + step)),
                    })
    # relative compactness: the teams differ markedly in the same moment (same camera, same perspective)
    sb, sw = sm["blue"]["spread_bh"], sm["white"]["spread_bh"]
    diff = sb - sw
    ok = np.abs(diff[~np.isnan(diff)])
    if len(ok) >= 20:
        thr = max(np.percentile(ok, 90), MIN_DEVIATION * np.nanmedian(np.concatenate([sb, sw])))
        for a, b in sorted(_windows(np.abs(diff) >= thr, 4), key=lambda w: -np.nanmax(np.abs(diff[w[0]:w[1] + 1])))[:3]:
            i = a + int(np.nanargmax(np.abs(diff[a:b + 1])))
            wider = "blue" if diff[i] > 0 else "white"
            events.append({
                "type": "gap", "metric": "spread_bh", "team": wider, "start_s": t[a], "end_s": t[b] + 1.0, "peak_s": t[i],
                "description": f"{wider.title()} were much more spread than {'white' if wider == 'blue' else 'blue'} "
                               f"({sb[i]:.1f} vs {sw[i]:.1f} player heights).",
                "strength": round(float(abs(diff[i])), 2), "mode": "image-space", "near_image_edge": False,
                "players_per_group": _win_n(series, TEAMS, a, b),
            })
    return events


def frame_flags(frames: list[dict], fps: float) -> dict:
    """Per-frame quality flags, merged into segments of at least one second."""
    flags = {"missing_players": [], "label_doubt": [], "low_confidence": []}
    for f in frames:
        nb, nw = (len(f["groups"][t][0]) for t in TEAMS)
        if nb + nw < MIN_FULL_FRAME or min(nb, nw) < MIN_PLAYERS:
            flags["missing_players"].append(f["frame"])
        if f["unknown"] >= UNKNOWN_DOUBT:
            flags["label_doubt"].append(f["frame"])
        if f["mean_conf"] is not None and f["mean_conf"] < LOW_CONFIDENCE:
            flags["low_confidence"].append(f["frame"])
    out = {}
    for name, fr in flags.items():
        segs, start, prev = [], None, None
        for x in fr:
            if start is None:
                start = prev = x
            elif x - prev <= 5:
                prev = x
            else:
                segs.append((start, prev))
                start = prev = x
        if start is not None:
            segs.append((start, prev))
        segs = [(a, b) for a, b in segs if b - a + 1 >= fps]
        out[name] = {"frames": len(fr), "share": round(len(fr) / max(1, len(frames)), 4),
                     "segments": [{"start_s": round(a / fps, 1), "end_s": round((b + 1) / fps, 1)} for a, b in segs[:40]],
                     "segments_total": len(segs)}
    return out


def summarize(series: dict, extent_x: tuple[float, float]) -> dict:
    out: dict = {"teams": {}, "comparison": {}}
    sp = {t: _arr(series[t]["spread_bh"]) for t in TEAMS}
    for t in TEAMS:
        row = {}
        for k in ("spread_bh", "length_bh", "width_bh"):
            a = _arr(series[t][k])
            a = a[~np.isnan(a)]
            row[k] = {"median": round(float(np.median(a)), 2), "p25": round(float(np.percentile(a, 25)), 2),
                      "p75": round(float(np.percentile(a, 75)), 2)} if len(a) else None
        cx = _arr(series[t]["cx"])
        cx = cx[~np.isnan(cx)]
        if len(cx):
            frac = (cx - extent_x[0]) / max(1e-9, extent_x[1] - extent_x[0])
            row["block_position_share"] = {"left": round(float(np.mean(frac < 1 / 3)), 3),
                                           "centre": round(float(np.mean((frac >= 1 / 3) & (frac < 2 / 3))), 3),
                                           "right": round(float(np.mean(frac >= 2 / 3)), 3)}
        row["valid_seconds"] = int(np.sum(~np.isnan(_arr(series[t]["spread_bh"]))))
        out["teams"][t] = row
    both = ~np.isnan(sp["blue"]) & ~np.isnan(sp["white"])
    if both.sum() >= 10:
        out["comparison"] = {
            "seconds_compared": int(both.sum()),
            "blue_more_compact_share": round(float(np.mean(sp["blue"][both] < sp["white"][both])), 3),
            "median_spread_ratio_blue_over_white": round(float(np.median(sp["blue"][both] / sp["white"][both])), 2),
        }
    return out


def compute(frames_path: Path, pitch_map=None) -> dict:
    frames = read_frames(frames_path)
    fps = frames[0]["fps"] if frames else FPS_DEFAULT
    metrics = per_frame(frames)
    xs = np.concatenate([f["groups"][t][0] for f in frames for t in TEAMS] or [np.array([])])
    extent = (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))) if len(xs) else (0.0, 1.0)
    # extra pixel-spread series feeds the block-shift normalisation
    series = bin_series(frames, metrics, fps, keys=(*METRICS, "spread_px"))
    result = {
        "mode": "image-space",
        "definition": {
            "units": "player heights (box height) - approximate, image-space, not metres",
            "min_players_per_group": MIN_PLAYERS,
            "length": "10-90 percentile span along image x (the pitch's long axis in this view)",
            "width": "10-90 percentile span along image y (across the pitch; strongly foreshortened)",
            "spread": "RMS distance from the group's median position, ignoring the farthest player when >= 8 are labelled; lower = more compact",
            "excluded": "other-kit (goalkeeper/referee/staff) and unknown-colour boxes",
        },
        "fps": fps, "frames": len(frames), "extent_x_px": [round(extent[0]), round(extent[1])],
        "series": series,
    }
    result.update(summarize(series, extent))
    result["events"] = sorted(find_events(series, extent), key=lambda e: e["start_s"])
    result["quality"] = frame_flags(frames, fps)
    return result


def _corr(a, b) -> float | None:
    a, b = _arr(a), _arr(b)
    ok = ~np.isnan(a) & ~np.isnan(b)
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) if ok.sum() >= 10 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else None


def validate(frames_path: Path, mot_path: Path, kit_reference_path: Path, calibration_json: Path | None = None,
             insights: dict | None = None) -> dict:
    """Compare shape metrics with the reference tracks and (optionally) the prototype pitch projection."""
    from football_fatigue.vision.benchmark import load_mot_ids

    frames = read_frames(frames_path)
    fps = frames[0]["fps"]
    insights = insights or compute(frames_path)
    kit = json.loads(kit_reference_path.read_text(encoding="utf-8"))
    ref = reference_frames(load_mot_ids(mot_path), kit)
    ref = [r for r in ref if frames[0]["frame"] <= r["frame"] <= frames[-1]["frame"]]
    ref_series = bin_series(ref, per_frame(ref), fps, keys=(*METRICS, "spread_px"))
    out: dict = {"reference": {"note": "SoccerTrack reference tracks with derived colour labels; evaluation only, 10 players per group.",
                               "teams": {}}, "events": [], "pitch_prototype_crosscheck": None}
    for t in TEAMS:
        row = {}
        for k in METRICS:
            a, b = _arr(insights["series"][t][k]), _arr(ref_series[t][k])
            m = min(len(a), len(b))
            a, b = a[:m], b[:m]
            ok = ~np.isnan(a) & ~np.isnan(b)
            row[k] = {"pearson_r": _corr(a, b), "seconds": int(ok.sum()),
                      "median_relative_difference": round(float(np.median((a[ok] - b[ok]) / np.where(b[ok] == 0, np.nan, b[ok]))), 3) if ok.sum() else None}
        out["reference"]["teams"][t] = row
    # do detected events agree with the reference series on the same side of its own median, and is the
    # underlying metric reliable (Pearson r vs the reference series)?
    above = {"spread", "stretch"}
    below = {"compact", "squeeze"}
    for e in insights["events"]:
        r = out["reference"]["teams"].get(e["team"], {}).get(e["metric"], {}).get("pearson_r") if e["type"] != "gap" else None
        if e["type"] == "gap":
            rs = [out["reference"]["teams"][t]["spread_bh"]["pearson_r"] for t in TEAMS]
            r = min(x for x in rs if x is not None) if any(x is not None for x in rs) else None
        e["reference_r"] = r
        e["reliability"] = "unknown" if r is None else ("good" if r >= 0.8 else "fair" if r >= 0.6 else "poor")
        agrees = None
        if e["type"] in above | below:
            b = _arr(ref_series[e["team"]][e["metric"]])
            seg = b[int(e["start_s"]):min(len(b) - 1, int(e["end_s"]) - 1) + 1]
            med = np.nanmedian(b)
            if len(seg) and not np.all(np.isnan(seg)) and not np.isnan(med):
                agrees = bool(np.nanmedian(seg) > med) if e["type"] in above else bool(np.nanmedian(seg) < med)
                out["events"].append({"start_s": e["start_s"], "team": e["team"], "type": e["type"], "reference_agrees": agrees,
                                      "reference_index": _percent_vs_median(float(np.nanmedian(seg)), float(med))})
        e["reference_agrees"] = agrees
        e["default_visible"] = e["reliability"] in ("good", "fair") and agrees is not False
    if calibration_json and calibration_json.is_file():
        from football_fatigue.vision.pitch_prototype import PitchMap

        mapper = PitchMap(np.array(json.loads(calibration_json.read_text(encoding="utf-8"))["parameters"]), step=0.4)
        cc = {}
        for t in TEAMS:
            img, met = [], []
            for f in frames[::5]:
                x, y, h = f["groups"][t]
                m = _group_metrics(x, y, h)
                if not m:
                    continue
                p = mapper.to_pitch(np.stack([x, y], 1))
                img.append(m["spread_bh"])
                met.append(robust_spread(p[:, 0], p[:, 1]))
            cc[t] = {"frames": len(img), "pearson_r_spread": _corr(img, met),
                     "note": "prototype projection has ~2 m median / ~4 m p90 held-out error: a plausibility check only"}
        out["pitch_prototype_crosscheck"] = cc
    out["event_agreement_with_reference"] = {
        "events_checked": len(out["events"]), "agree": sum(1 for e in out["events"] if e["reference_agrees"]),
        "shown_by_default": sum(1 for e in insights["events"] if e.get("default_visible")), "events_total": len(insights["events"]),
    }
    return out


def ball_probe_summary(probe: list[dict], static_min_share: float = 0.5) -> dict:
    """From a sampled ball probe: how often is there a candidate that is not a static spare ball?"""
    from collections import Counter

    cells = Counter()
    for fr in probe:
        for _, b in fr["top"]:
            cells[(round((b[0] + b[2]) / 24), round((b[1] + b[3]) / 24))] += 1
    static = {k for k, v in cells.items() if v >= static_min_share * len(probe)}
    moving = sum(1 for fr in probe if any(c >= 0.25 and (round((b[0] + b[2]) / 24), round((b[1] + b[3]) / 24)) not in static
                                          for c, b in fr["top"]))
    return {"frames_sampled": len(probe), "static_candidate_positions": len(static),
            "frames_with_non_static_candidate": moving, "share": round(moving / max(1, len(probe)), 3)}


def advanced_metrics(calibration: dict | None, tracking_eval: dict | None, ball: dict | None) -> list[dict]:
    """Acceptance criteria for each advanced metric, evaluated against measured evidence. Status stays
    'unavailable' unless every criterion is met."""
    cv = (calibration or {}).get("held_out_cross_validation") or {}
    med, p90 = cv.get("median_m"), cv.get("p90_m")
    te = tracking_eval or {}
    idf1 = te.get("idf1")
    minutes = (te.get("frames") or 0) / FPS_DEFAULT / 60
    ids = te.get("reference_identities") or 22
    sw_pm = round(te["id_switches"] / ids / minutes, 1) if te.get("id_switches") is not None and minutes else None
    ball_share = (ball or {}).get("share")

    def req(name, criterion, measured, met):
        return {"name": name, "criterion": criterion, "measured": measured, "met": bool(met) if met is not None else None}

    calib_shape = req("Pitch calibration for team shape in metres", "held-out keypoint error: median <= 1.0 m and p90 <= 2.0 m",
                      None if med is None else f"median {med} m, p90 {p90} m (prototype)", None if med is None else (med <= 1.0 and p90 <= 2.0))
    calib_speed = req("Pitch calibration for distance and speed", "held-out keypoint error: median <= 0.5 m and p90 <= 1.0 m",
                      None if med is None else f"median {med} m, p90 {p90} m (prototype)", None if med is None else (med <= 0.5 and p90 <= 1.0))
    identity = req("Player identity continuity", "IDF1 >= 0.80 and <= 1 identity switch per player-minute against reference tracks",
                   None if idf1 is None else f"IDF1 {idf1}, {sw_pm} switches per player-minute", None if idf1 is None else (idf1 >= 0.8 and sw_pm is not None and sw_pm <= 1))
    ball_req = req("Match-ball detection", ">= 80% of frames with the match ball found (needs ball labels to measure) and spare balls rejected",
                   None if ball_share is None else f"non-static ball candidate in {ball_share:.0%} of {ball['frames_sampled']} sampled frames; no ball labels to measure recall",
                   False if ball_share is not None and ball_share < 0.8 else None)
    poss = req("Possession attribution accuracy", ">= 85% against manually labelled possession", "no labelled possession data", None)
    items = [
        ("formation", "Formation and positional shape", [calib_shape, identity], "Needs stable roles per player and metric positions; the image-space shape curves shown on this page are the supported substitute."),
        ("defensive_line", "Defensive-line positioning", [calib_shape, req("Attacking direction per team", "known and validated for every period", "not available (goalkeepers are often undetected)", False)], "Needs which way each team attacks and across-pitch depth in metres."),
        ("possession", "Ball possession %", [ball_req, poss, calib_shape], "A spare ball on the touchline outranks the match ball in detections; possession cannot be attributed yet."),
        ("pressing", "Pressing and defensive pressure", [calib_shape, ball_req, identity], "Needs the ball carrier, metric distances and identity."),
        ("distance", "Player total distance", [calib_speed, identity], "Needs metric positions and one identity per player."),
        ("sprints", "Sprints and high-speed running", [calib_speed, identity], "Needs metric speed per player."),
        ("pitch_control", "Pitch Control", [calib_speed, identity, ball_req], "Needs metric positions, velocities and the ball."),
        ("epv_obso_das", "EPV, OBSO and Dangerous Accessible Space", [calib_speed, identity, ball_req, poss], "Also needs a trained and validated value model."),
    ]
    out = []
    for key, title, reqs, note in items:
        out.append({"key": key, "title": title, "status": "available" if all(r["met"] for r in reqs) else "unavailable",
                    "requirements": reqs, "note": note})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mot", type=Path)
    ap.add_argument("--kit-reference", type=Path)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--ball-probe", type=Path, help="JSON written by the 40-frame ball probe")
    args = ap.parse_args(argv)
    frames = args.run_dir / "frames.jsonl"
    insights = compute(frames)
    validation = None
    if args.mot and args.kit_reference:
        validation = validate(frames, args.mot, args.kit_reference, args.calibration, insights)
        insights["validation"] = validation
    load = lambda p: json.loads(p.read_text(encoding="utf-8")) if p and p.is_file() else None
    insights["advanced_metrics"] = advanced_metrics(load(args.calibration), load(args.run_dir / "tracking_eval.json"),
                                                    ball_probe_summary(load(args.ball_probe)) if load(args.ball_probe) else None)
    if args.calibration and load(args.calibration):
        c = load(args.calibration)
        insights["calibration_prototype"] = {k: c[k] for k in ("model", "keypoints", "held_out_cross_validation", "train_reprojection_px", "status")}
    for name, payload in (("coach_insights.json", insights),):
        target = args.run_dir / name
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(insights["events"]), "summary": insights["comparison"],
                      "validation": validation and {k: validation[k] for k in ("reference", "event_agreement_with_reference", "pitch_prototype_crosscheck")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
