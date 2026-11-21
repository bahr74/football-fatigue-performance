"""Visual inspection figures for the window-intersection datasets. Every figure uses actual FIFA samples; nothing is synthesised."""

from __future__ import annotations

import io
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from football_fatigue.encounters import quality as Q  # noqa: E402
from football_fatigue.encounters.trajectories import STATE_AVAILABLE  # noqa: E402

from . import core as W  # noqa: E402

# reference categorical palette (fixed order), text inks and surface
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _curve(seg, cf, slot: int, period: int, lo: float, hi: float, K: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Points of one player's variant curve in [lo, hi]; NaN separates chains (no line is drawn across a break)."""
    cx, cy = cf
    g = W.slot_slice(seg, slot, period)
    g = g[(seg["t1"][g] >= lo) & (seg["t0"][g] <= hi)]
    xs, ys, starts = [], [], []
    prev_chain = None
    for i in g:
        if prev_chain is not None and seg["chain"][i] != prev_chain:
            xs.append(np.nan), ys.append(np.nan)
        s = np.linspace(max(lo, seg["t0"][i]), min(hi, seg["t1"][i]), K + 1) - seg["t0"][i]
        xs += list(W.evalc(cx[i], s))
        ys += list(W.evalc(cy[i], s))
        prev_chain = seg["chain"][i]
    return np.array(xs, float), np.array(ys, float), starts


def event_figure(c, mask, seg, coefs: dict[str, tuple], ev: dict[str, Any], cfg: W.WindowConfig, title: str, condition: str,
                 anchor: int | None = None, show_pchip: bool = False, pchip_ev: dict[str, Any] | None = None) -> bytes:
    """Left: both trajectories in one 5 s window (raw samples, linear path, optional PCHIP path), the selected intersection,
    each player's arrival time and the anchor-time positions. Right: synchronized raw distance over the window."""
    p = ev["period"]
    pstart = ev["t_event_s"] - ev["t_event_rel_s"]
    wins = json.loads(ev["windows"])
    if anchor is None:
        anchor = min(wins, key=lambda k: (abs(k - ev["t_event_rel_s"]), k))
    lo, hi = pstart + anchor - cfg.half_s, pstart + anchor + cfg.half_s
    rows = np.flatnonzero((c.canonical_period == p) & (c.video_time_s >= lo) & (c.video_time_s <= hi))
    est = (mask & Q.ESTIMATED) != 0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for tag, s, col in (("A", ev["slot_a"], BLUE), ("B", ev["slot_b"], ORANGE)):
        x, y = _curve(seg, coefs["linear"], s, p, lo, hi, 1)[:2]
        ax.plot(x, y, color=col, lw=1.6, label=f"player {tag} (slot {s}) linear path", zorder=2)
        if show_pchip:
            xp, yp = _curve(seg, coefs["pchip"], s, p, lo, hi, cfg.pchip_substeps)[:2]
            ax.plot(xp, yp, color=col, lw=1.0, ls=(0, (3, 2)), alpha=0.9, label=f"player {tag} PCHIP path", zorder=3)
        r = rows[c.state[rows, s] == STATE_AVAILABLE]
        vis, es = r[~est[r, s]], r[est[r, s]]
        ax.scatter(c.x[vis, s], c.y[vis, s], s=9, color=col, zorder=4, label=f"{tag} raw samples (VISIBLE)" if len(vis) else None)
        ax.scatter(c.x[es, s], c.y[es, s], s=11, facecolors="none", edgecolors=col, lw=0.8, zorder=4,
                   label=f"{tag} raw samples (ESTIMATED)" if len(es) else None)
        if len(r):
            ax.annotate("start", (c.x[r[0], s], c.y[r[0], s]), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
        px, py = W.positions_at(seg, *coefs["linear"], s, p, np.array([pstart + anchor], float))
        ax.scatter(px, py, marker="s", s=46, color=col, edgecolors=INK, lw=0.8, zorder=6, label=f"{tag} at anchor t={anchor} s")
    pa = W.positions_at(seg, *coefs["linear"], ev["slot_a"], p, np.array([pstart + anchor], float))
    pb = W.positions_at(seg, *coefs["linear"], ev["slot_b"], p, np.array([pstart + anchor], float))
    if np.isfinite(pa[0][0]) and np.isfinite(pb[0][0]):
        ax.plot([pa[0][0], pb[0][0]], [pa[1][0], pb[1][0]], color=INK2, lw=0.8, ls=":", zorder=5)
        dA = float(np.hypot(pa[0][0] - pb[0][0], pa[1][0] - pb[1][0]))
    else:
        dA = float("nan")
    ax.scatter([ev["x_m"]], [ev["y_m"]], marker="*", s=170, color=INK, zorder=7, label="selected (first) intersection")
    if pchip_ev is not None:
        ax.scatter([pchip_ev["x_m"]], [pchip_ev["y_m"]], marker="X", s=70, color=AQUA, edgecolors=INK, lw=0.5, zorder=7, label="PCHIP intersection")
    ax.annotate(f"A arrives {ev['t_a_rel_s'] - anchor:+.3f} s\nB arrives {ev['t_b_rel_s'] - anchor:+.3f} s\n|dt| = {ev['abs_dt_s']:.3f} s",
                (ev["x_m"], ev["y_m"]), xytext=(10, -34), textcoords="offset points", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=MUTED, lw=0.5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.legend(fontsize=6.5, loc="best", frameon=True, framealpha=0.9)
    # distance panel
    tt = c.video_time_s[rows] - pstart - anchor
    d = np.hypot(c.x[rows, ev["slot_a"]] - c.x[rows, ev["slot_b"]], c.y[rows, ev["slot_a"]] - c.y[rows, ev["slot_b"]])
    ax2.plot(tt, d, color=INK2, lw=1.4, label="synchronized raw distance")
    ax2.axhline(cfg.radius_m, color=RED, lw=0.9, ls="--", label=f"R = {cfg.radius_m} m (methods B, C)")
    ax2.axvline(0, color=INK, lw=1.0, label="anchor t (whole second)")
    ax2.axvline(ev["t_a_rel_s"] - anchor, color=BLUE, lw=1.2, ls="-.", label="A arrival at intersection")
    ax2.axvline(ev["t_b_rel_s"] - anchor, color=ORANGE, lw=1.2, ls="-.", label="B arrival at intersection")
    if np.isfinite(dA):
        ax2.scatter([0], [dA], color=INK, s=30, zorder=5)
        ax2.annotate(f"d(anchor) = {dA:.3f} m", (0, dA), xytext=(6, 8), textcoords="offset points", fontsize=8)
    ax2.set_xlim(-cfg.half_s, cfg.half_s)
    ax2.set_ylim(0, max(1.0, float(np.nanmax(d)) * 1.05) if np.isfinite(d).any() else 5)
    ax2.set_xlabel("time relative to anchor (s); window = anchor +- 2.5 s"), ax2.set_ylabel("distance (m)")
    ax2.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.text(0.01, -0.02, condition, fontsize=8.5, color=INK2, ha="left", va="top")
    return _png(fig)


def dedup_figure(ev: dict[str, Any], cfg: W.WindowConfig, title: str) -> bytes:
    """Overlapping windows around one event. Green: windows whose primary intersection is this event; orange: windows that contain
    it but whose first intersection is another; grey: windows that do not contain it."""
    wins = set(json.loads(ev["windows"]))
    ta, tb = ev["t_a_rel_s"], ev["t_b_rel_s"]
    k0, k1 = int(np.floor(min(ta, tb) - cfg.half_s)) - 1, int(np.ceil(max(ta, tb) + cfg.half_s)) + 1
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * (k1 - k0 + 1) + 1.4))
    for i, k in enumerate(range(k0, k1 + 1)):
        contains = (k - cfg.half_s <= min(ta, tb)) and (max(ta, tb) <= k + cfg.half_s)
        col = GREEN if k in wins else ORANGE if contains else GRID
        ax.barh(i, 2 * cfg.half_s, left=k - cfg.half_s, height=0.6, color=col, edgecolor=SURFACE, lw=2)
        ax.text(k + cfg.half_s + 0.08, i, f"anchor {k} s" + ("  -> merged into event" if k in wins else "  (contains, not first)" if contains else ""),
                va="center", fontsize=7, color=INK2)
        ax.plot([k], [i], marker="|", color=INK, ms=8)
    ax.axvline(ta, color=BLUE, lw=1.3, ls="-.", label="A arrival")
    ax.axvline(tb, color=ORANGE, lw=1.3, ls="-.", label="B arrival")
    ax.set_yticks([])
    ax.set_xlabel("period time (s)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=9.5, loc="left")
    return _png(fig)


def bars(groups: list[str], series: dict[str, list[float]], title: str, ylabel: str, log: bool = False, stacked: bool = False,
         note: str | None = None) -> bytes:
    """Grouped (or stacked) bars with the categorical palette in fixed order and value labels on top."""
    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(groups) * max(1, len(series)) * (0.45 if not stacked else 0.9)), 4.0))
    x = np.arange(len(groups))
    n = len(series)
    w = 0.8 / (1 if stacked else n)
    bottom = np.zeros(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        v = np.asarray(vals, float)
        if stacked:
            ax.bar(x, v, w, bottom=bottom, color=SERIES[i % len(SERIES)], edgecolor=SURFACE, lw=2, label=name)
            bottom = bottom + v
        else:
            pos = x - 0.4 + w * (i + 0.5)
            ax.bar(pos, v, w, color=SERIES[i % len(SERIES)], edgecolor=SURFACE, lw=2, label=name)
            for px_, pv in zip(pos, v):
                ax.text(px_, pv, f"{int(pv):,}" if pv == int(pv) else f"{pv:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    if stacked:
        for px_, tot in zip(x, bottom):
            ax.text(px_, tot, f"{int(tot):,}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, groups, fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", visible=False)
    if n > 1 or stacked:
        ax.legend(fontsize=7)
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=INK2, ha="left", va="top")
    return _png(fig)


def hist(values: dict[str, np.ndarray], title: str, xlabel: str, bins: np.ndarray, note: str | None = None) -> bytes:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, (name, v) in enumerate(values.items()):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=2, color=SERIES[i % len(SERIES)], label=f"{name} (n={len(v):,})")
    ax.set_xlabel(xlabel), ax.set_ylabel("count")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=7)
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=INK2, ha="left", va="top")
    return _png(fig)
