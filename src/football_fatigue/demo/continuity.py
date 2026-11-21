"""M7.1: display-continuity layer for the Football Vision Demo (overlay v4).

The saved tracker output has many short gaps: a player's box is missing for a few frames and then comes back with the same local tracker ID.
On a board redrawn every frame, such a player disappears and reappears ("marker flicker"). The identity status also toggles between
confirmed / provisional / unknown, which changes the label ("label flicker").

This module adds display-only continuity on top of the frozen observed rows. It never changes or removes an observed row.

* Gap bridging. A gap of at most ``max_gap`` frames between two observations of the same local tracker ID inside the same identity tracklet
  gets ``src=1`` rows, linearly interpolated in image and pitch space. Bridging is refused when the implied movement is implausible.
  ``max_gap`` is justified by :func:`prediction_eval` on masked observed runs.
* Lost-track tail. When a track ends without being bridged, up to ``tail_frames`` ``src=2`` rows are added. They hold the last observed
  position and are drawn fading as "lost". The tail never claims the player is still there.
* Smoothed display positions ``sx, sy`` (centred moving average inside one display run). The raw ``pxm, pym`` stay unchanged.
* Display trust level with conservative hysteresis ``dlev``. A downgrade (towards unknown) is shown immediately. An upgrade is shown only
  after the raw level has held for ``upgrade_dwell`` frames, so the display is never more confident than the data.
* Analytics eligibility ``elig`` with a reason code. Only ``elig == 0`` rows (observed, on the pitch, active role, confident team) may feed
  spatial analytics. Estimated rows can never be eligible.

Estimated rows (``est=1``) carry no confidence and are excluded from every count, segment and metric.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from football_fatigue.demo.overlay import pitch_validity

SRC = ["observed", "interpolated_display", "lost_hold_display"]
ELIG = ["eligible", "estimated_display_only", "off_or_near_pitch", "role_not_active_player", "team_uncertain"]
LEVELS = ["trusted", "hypothesis", "untrusted"]   # 0 confirmed + no unvalidated link | 1 provisional or linked | 2 unknown/ambiguous


@dataclass(frozen=True)
class ContinuityConfig:
    max_gap: int = 25              # frames (1 s): largest tested gap with interpolation p90 <= 1.0 m and <= 0.25 box heights (M7.1 E1)
    max_speed_mps: float = 10.0    # implied pitch speed across a gap above this refuses the bridge ...
    noise_m: float = 1.0           # ... after allowing this much position noise (calibration held-out p90 ~1.07 m; 1-frame jitter p90 0.46 m)
    max_px_per_frame_h: float = 0.25   # off-pitch fallback: foot displacement per gap frame, in box heights
    crowd_gap: int = 12            # gaps longer than this are not bridged when another person stands within crowd_radius_h box heights
    crowd_radius_h: float = 0.6    # of either end (crowded scenes are where a visual continuation could join two different people)
    tail_frames: int = 6           # lost-track tail (frames): longest horizon where holding the last position stays <= 1.0 m p90 (M7.1 E1)
    smooth_half: int = 2           # centred moving-average half width (frames) for display positions
    upgrade_dwell: int = 12        # frames a higher trust level must hold before the label shows it
    team_reliable: float = 0.7
    fps: float = 25.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def trust_level(st: int, assoc: int) -> int:
    if st in (2, 3) or st < 0:
        return 2
    if st == 1 or assoc == 1:
        return 1
    return 0


def hysteresis(levels: list[int], dwell: int) -> list[int]:
    """Displayed level: a downgrade (larger value) is immediate; an upgrade only after the raw level has held for ``dwell`` consecutive steps."""
    out, shown, run_level, run = [], None, None, 0
    for lv in levels:
        run = run + 1 if lv == run_level else 1
        run_level = lv
        if shown is None or lv > shown:
            shown = lv
        elif lv < shown and run >= dwell:
            shown = lv
        out.append(shown)
    return out


def frames_of(offsets: list[int]) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def observed_runs(cols: dict[str, list], offsets: list[int]) -> dict[int, list[int]]:
    """Observed row indices per local tracker ID, sorted by frame (estimated rows dropped)."""
    fr = frames_of(offsets)
    by: dict[int, list[int]] = defaultdict(list)
    for i in np.argsort(fr, kind="stable"):
        if not cols["est"][i]:
            by[cols["local"][i]].append(int(i))
    return by


def _foot(cols, i):
    return (cols["x1"][i] + cols["x2"][i]) / 2, cols["y2"][i]


# ------------------------------------------------------------------------------------------------------------ evaluation
def prediction_eval(cols: dict[str, list], offsets: list[int], gaps=(1, 2, 4, 6, 8, 12, 16, 25, 50), context: int = 6, fps: float = 25.0) -> dict:
    """Mask ``g`` frames inside continuous observed runs (same local ID and tracklet) and predict them.

    Methods: linear interpolation between the gap ends (offline replay knows both ends), constant velocity from the last ``context`` frames,
    an alpha-beta (steady-state Kalman) filter extrapolation, and holding the last position. Errors are in pitch metres (only windows fully
    on the pitch) and in box heights (image space). The 'truth' is itself tracker output, so these are display-consistency errors, not
    ground-truth position errors.
    """
    fr = frames_of(offsets)
    runs = observed_runs(cols, offsets)
    res: dict = {}
    for g in gaps:
        e = {m: {"m": [], "h": []} for m in ("interp", "const_vel", "alpha_beta", "hold")}
        for idx in runs.values():
            f = fr[idx]
            # split into strictly consecutive runs inside one tracklet
            cut = np.flatnonzero((np.diff(f) != 1) | (np.diff([cols["trk"][i] for i in idx]) != 0)) + 1
            for part in np.split(np.array(idx), cut):
                n = len(part)
                step = g + 2 * context
                for a in range(context, n - g - 1, step):   # masked indices a .. a+g-1; context before, at least one after
                    win = part[a - context:a + g + 1]
                    P = np.array([[cols["pxm"][i], cols["pym"][i]] for i in win])
                    on = all(cols["pv"][i] == 0 for i in win)
                    F = np.array([_foot(cols, i) for i in win])
                    hgt = np.array([cols["y2"][i] - cols["y1"][i] for i in win])
                    for space, Z in (("m", P), ("h", F)):
                        if space == "m" and not on:
                            continue
                        before, truth, after = Z[:context], Z[context:context + g], Z[context + g]
                        k = np.arange(1, g + 1)[:, None]
                        preds = {
                            "interp": before[-1] + (after - before[-1]) * k / (g + 1),
                            "const_vel": before[-1] + (before[-1] - before[0]) / (context - 1) * k,
                            "alpha_beta": (lambda xv: xv[0] + xv[1] * k)(_alpha_beta(before)),
                            "hold": np.repeat(before[-1:], g, 0),
                        }
                        scale = 1.0 if space == "m" else hgt[context:context + g, None].mean()
                        for m, pr in preds.items():
                            e[m][space] += list(np.linalg.norm(pr - truth, axis=1) / scale)
        res[g] = {m: {sp: _summ(v) for sp, v in d.items()} for m, d in e.items()}
    return res


def _alpha_beta(z: np.ndarray, alpha: float = 0.5, beta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    x, v = z[0].astype(float), np.zeros(z.shape[1])
    for obs in z[1:]:
        xp = x + v
        r = obs - xp
        x, v = xp + alpha * r, v + beta * r
    return x, v


def _summ(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.array(v)
    return {"n": int(len(a)), "median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3)}


def gap_histogram(cols, offsets) -> dict:
    """Missing-frame gaps between consecutive observations of the same local tracker ID (all gaps, any tracklet)."""
    fr = frames_of(offsets)
    g = []
    for idx in observed_runs(cols, offsets).values():
        d = np.diff(fr[idx]) - 1
        g += [int(x) for x in d if x > 0]
    g = np.array(g)
    bins = [1, 2, 3, 5, 8, 13, 25, 50, 100, 10**9]
    return {"n_gaps": int(len(g)), "hist": {f"{a}-{b - 1}": int(((g >= a) & (g < b)).sum()) for a, b in zip(bins, bins[1:])}}


def flicker_metrics(cols: dict[str, list], offsets: list[int], fps: float = 25.0, short: int = 50, window: int = 50,
                    label_of=None) -> dict:
    """Marker flicker: a displayed local track vanishes for 1..``short`` frames and the same local ID is displayed again.
    Label flicker: the label of one local track changes A -> B and returns to A within ``window`` displayed frames.
    ``label_of(i)`` gives the label of row i. Only rows present in ``cols`` count as displayed."""
    fr = frames_of(offsets)
    minutes = (len(offsets) - 1) / fps / 60
    by: dict[int, list[int]] = defaultdict(list)
    for i in np.argsort(fr, kind="stable"):
        by[cols["local"][i]].append(int(i))
    marker = label_ch = label_flick = 0
    for idx in by.values():
        f = fr[idx]
        d = np.diff(f) - 1
        marker += int(((d >= 1) & (d <= short)).sum())
        if label_of is not None:
            labs = [label_of(i) for i in idx]
            changes = [k for k in range(1, len(labs)) if labs[k] != labs[k - 1]]
            label_ch += len(changes)
            for a, b in zip(changes, changes[1:]):
                if labs[b] == labs[a - 1] and f[b] - f[a] <= window:
                    label_flick += 1
    out = {"minutes": round(minutes, 3), "marker_flicker": marker, "marker_flicker_per_min": round(marker / minutes, 1)}
    if label_of is not None:
        out.update({"label_changes": label_ch, "label_changes_per_min": round(label_ch / minutes, 1),
                    "label_flicker": label_flick, "label_flicker_per_min": round(label_flick / minutes, 1)})
    return out


# ------------------------------------------------------------------------------------------------------------ build
def build_continuity(ov: dict[str, Any], cfg: ContinuityConfig = ContinuityConfig()) -> dict[str, Any]:
    """Return a new overlay (v4): observed rows unchanged, previous estimated rows replaced by the continuity layer, extra columns."""
    cols, offsets = ov["cols"], ov["frame_offsets"]
    n_frames = ov["n_frames"]
    fr = frames_of(offsets)
    keys = list(cols.keys())
    obs = [i for i in range(len(fr)) if not cols["est"][i]]
    rows: list[tuple[int, int, dict]] = []          # (frame, order, row)
    for i in obs:
        rows.append((int(fr[i]), len(rows), {k: cols[k][i] for k in keys} | {"src": 0, "dconf": 1.0}))
    stats = {"gaps_seen": 0, "bridged": 0, "bridged_frames": 0, "refused_tracklet_change": 0, "refused_implausible": 0,
             "refused_too_long": 0, "refused_crowded": 0, "tails": 0, "tail_frames": 0}
    runs = observed_runs(cols, offsets)
    dt = 1.0 / cfg.fps
    for local, idx in runs.items():
        for p, (a, b) in enumerate(zip(idx, idx[1:] + [None])):
            fa = int(fr[a])
            gap = (int(fr[b]) - fa - 1) if b is not None else None
            if gap == 0:
                continue
            bridged = False
            if gap is not None:
                stats["gaps_seen"] += 1
                if gap > cfg.max_gap:
                    stats["refused_too_long"] += 1
                elif cols["trk"][a] != cols["trk"][b]:
                    stats["refused_tracklet_change"] += 1
                elif not _plausible(cols, a, b, gap, dt, cfg):
                    stats["refused_implausible"] += 1
                elif gap > cfg.crowd_gap and (_crowded(cols, offsets, a, int(fr[a]), cfg) or _crowded(cols, offsets, b, int(fr[b]), cfg)):
                    stats["refused_crowded"] += 1
                else:
                    bridged = True
                    stats["bridged"] += 1
                    stats["bridged_frames"] += gap
                    for k in range(1, gap + 1):
                        w = k / (gap + 1)
                        # display confidence: lowest mid-gap, lower for longer gaps (1.0 at the observed ends, 0.5 mid-way through a max-length gap)
                        rows.append((fa + k, len(rows), _est_row(cols, a, b, w, src=1, dconf=round(1 - 0.5 * min(1.0, gap / cfg.max_gap) * (1 - abs(2 * w - 1)), 2))))
            if not bridged and cfg.tail_frames > 0:
                tail = cfg.tail_frames if gap is None else min(cfg.tail_frames, gap)
                tail = min(tail, n_frames - 1 - fa)
                if tail > 0:
                    stats["tails"] += 1
                    stats["tail_frames"] += tail
                    for k in range(1, tail + 1):
                        rows.append((fa + k, len(rows), _est_row(cols, a, a, 0.0, src=2, dconf=round(0.6 * (1 - k / (tail + 1)), 2))))
    rows.sort(key=lambda t: (t[0], t[2]["src"], t[1]))
    new = {k: [] for k in keys + ["src", "dconf"]}
    counts = np.zeros(n_frames, int)
    for f, _, r in rows:
        counts[f] += 1
        for k in new:
            new[k].append(r[k])
    new_off = [0] + np.cumsum(counts).tolist()
    _add_display_columns(new, new_off, ov, cfg)
    out = dict(ov)
    out.update({"schema": "m7-demo-overlay-v4", "cols": new, "frame_offsets": new_off,
                "continuity": {"config": cfg.as_dict(), "stats": stats, "src": SRC, "elig": ELIG, "levels": LEVELS}})
    out["summary"] = dict(ov["summary"], boxes_estimated_display_only=int(sum(1 for s in new["src"] if s)))
    return out


def _plausible(cols, a, b, gap, dt, cfg) -> bool:
    if cols["pv"][a] != 2 and cols["pv"][b] != 2:
        d = float(np.hypot(cols["pxm"][b] - cols["pxm"][a], cols["pym"][b] - cols["pym"][a]))
        return d <= cfg.max_speed_mps * (gap + 1) * dt + cfg.noise_m
    (xa, ya), (xb, yb) = _foot(cols, a), _foot(cols, b)
    h = max(1.0, (cols["y2"][a] - cols["y1"][a] + cols["y2"][b] - cols["y1"][b]) / 2)
    return float(np.hypot(xb - xa, yb - ya)) / h <= cfg.max_px_per_frame_h * (gap + 1)


def _crowded(cols, offsets, i, f, cfg) -> bool:
    (x, y), h = _foot(cols, i), max(1.0, cols["y2"][i] - cols["y1"][i])
    for j in range(offsets[f], offsets[f + 1]):
        if j != i and not cols["est"][j] and float(np.hypot(*(np.subtract(_foot(cols, j), (x, y))))) < cfg.crowd_radius_h * h:
            return True
    return False


def _est_row(cols, a, b, w, *, src, dconf) -> dict:
    lerp = lambda k: (1 - w) * cols[k][a] + w * cols[k][b]
    worse = lambda k: max(cols[k][a], cols[k][b])
    same = lambda k, d: cols[k][a] if cols[k][a] == cols[k][b] else d
    row = {k: cols[k][a] for k in cols}
    row.update({"x1": lerp("x1"), "y1": lerp("y1"), "x2": lerp("x2"), "y2": lerp("y2"), "pxm": round(lerp("pxm"), 2), "pym": round(lerp("pym"), 2),
                "st": worse("st"), "as": worse("as"), "team": same("team", -1), "tconf": None if cols["tconf"][a] is None or cols["tconf"][b] is None else min(cols["tconf"][a], cols["tconf"][b]),
                "role": same("role", 3), "kpd": max(cols["kpd"][a], cols["kpd"][b]), "conf": None, "conf_local": None, "link": None,
                "seg": -1, "est": 1, "src": src, "dconf": dconf})
    for k in ("x1", "y1", "x2", "y2"):
        row[k] = round(row[k], 1)
    row["pv"] = pitch_validity(row["pxm"], row["pym"])
    return row


def _add_display_columns(c: dict[str, list], off: list[int], ov: dict, cfg: ContinuityConfig) -> None:
    n = len(c["local"])
    fr = frames_of(off)
    c["lev"] = [trust_level(c["st"][i], c["as"][i]) for i in range(n)]
    c["dlev"] = list(c["lev"])
    c["sx"] = list(c["pxm"])
    c["sy"] = list(c["pym"])
    by: dict[int, list[int]] = defaultdict(list)
    for i in np.argsort(fr, kind="stable"):
        by[c["local"][i]].append(int(i))
    h = cfg.smooth_half
    for idx in by.values():
        # hysteresis over the displayed sequence; restart after a display break
        f = fr[idx]
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        for part in np.split(np.array(idx), breaks):
            for i, lv in zip(part, hysteresis([c["lev"][i] for i in part], cfg.upgrade_dwell)):
                c["dlev"][i] = lv
            # smoothing inside a run of one tracklet, excluding the lost-hold tail
            sub = [i for i in part if c["src"][i] != 2]
            for seg in np.split(np.array(sub, int), np.flatnonzero(np.diff([c["trk"][i] for i in sub]) != 0) + 1) if sub else []:
                x = np.array([c["pxm"][i] for i in seg], float)
                y = np.array([c["pym"][i] for i in seg], float)
                for k, i in enumerate(seg):
                    a, b = max(0, k - h), min(len(seg), k + h + 1)
                    c["sx"][i], c["sy"][i] = round(float(x[a:b].mean()), 2), round(float(y[a:b].mean()), 2)
    c["elig"] = [eligibility(c, i, cfg.team_reliable) for i in range(n)]


def eligibility(c: dict[str, list], i: int, team_reliable: float = 0.7) -> int:
    """Index into ELIG: 0 only for observed, on-pitch, active-role (player/goalkeeper), confident team_0/team_1 rows."""
    if c["est"][i] or c.get("src", [0] * (i + 1))[i]:
        return 1
    if c["pv"][i] != 0:
        return 2
    if c["role"][i] not in (0, 1):
        return 3
    if c["team"][i] not in (0, 1) or (c["tconf"][i] or 0) < team_reliable:
        return 4
    return 0
