"""Inspectable PNG figures of intersection events (evaluation / audit view: identities are shown).

Each event figure shows both players' RAW positions within +-2.5 s of the event midpoint, split into the incoming span (solid),
the withheld interval (dotted) and the outgoing span (dashed), with the intersection point and the source references (match,
period, frames, video times, slots). Coordinates are RAW metres with equal axis scales; nothing is smoothed.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from football_fatigue.encounters.trajectories import RawCache

from .detect import Timeline

COLORS = ("#1f77b4", "#ff7f0e")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    _plt().close(fig)
    return buf.getvalue()


def event_figure(c: RawCache, tl: Timeline, ev: dict, ref: dict | None, title: str, half_span_s: float = 2.5) -> bytes:
    plt = _plt()
    p = ev["period"]
    t_lo, t_hi = min(ev["time_a_s"], ev["time_b_s"]), max(ev["time_a_s"], ev["time_b_s"])
    t_mid = (t_lo + t_hi) / 2
    rows = tl.between(p, t_mid - half_span_s, t_mid + half_span_s)
    t = c.video_time_s[rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for k, (slot, side, jersey) in enumerate(((ev["slot_a"], ev["side_a"], ev["jersey_a"]), (ev["slot_b"], ev["side_b"], ev["jersey_b"]))):
        x, y = c.x[rows, slot], c.y[rows, slot]
        for sel, ls, lab in ((t < t_lo, "-", "incoming"), ((t >= t_lo) & (t <= t_hi), ":", "withheld"), (t > t_hi, "--", "outgoing")):
            if sel.any():
                ax.plot(x[sel], y[sel], ls, color=COLORS[k], lw=2.2 if ls != ":" else 1.6,
                        label=f"{side} #{jersey} (slot {slot}) {lab}")
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok):
            ax.plot(x[ok[0]], y[ok[0]], "o", color=COLORS[k], ms=7)
            ax.annotate("", xy=(x[ok[-1]], y[ok[-1]]), xytext=(x[ok[-2]], y[ok[-2]]) if len(ok) > 1 else (x[ok[-1]], y[ok[-1]]),
                        arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=2))
    ax.plot(ev["x"], ev["y"], "*", color="crimson", ms=18, mec="black", label="intersection point", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m, RAW pitch coordinates)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="best")
    sa, sb = ev["segment_a"], ev["segment_b"]
    lines = [f"match {ev['match_id']} · period {p} · type {ev['type']} · event {ev['event_id']}",
             f"A crosses at video t = {ev['time_a_s']:.3f} s (frames {sa['frame0']}-{sa['frame1']}); "
             f"B at {ev['time_b_s']:.3f} s (frames {sb['frame0']}-{sb['frame1']})",
             f"time difference {ev['time_diff_s']:+.3f} s · min synchronous distance "
             + (f"{ev['min_synchronous_distance_m']:.2f} m" if ev["min_synchronous_distance_m"] is not None else "–")
             + f" · windows {ev['n_windows']} · pair cluster size {ev.get('cluster_size', '–')}",
             "sample: " + (f"{ref['instance_id']} target out-{ref['target']} (query in-0 = slot {ref['incoming']['in-0']['slot']})" if ref else "none (excluded)")]
    ax.set_title(title, fontsize=11)
    fig.text(0.01, -0.02, "\n".join(lines), fontsize=7.5, va="top", family="monospace")
    return _png(fig)


def overview_figure(events: list[dict], refs: list[dict]) -> bytes:
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    d = np.array([e["abs_time_diff_s"] for e in events])
    axs[0].hist(d, bins=np.arange(0, 5.25, 0.25), color="#4c72b0")
    axs[0].set_xlabel("|time difference| at the intersection (s)")
    axs[0].set_ylabel("unique events")
    axs[0].set_title("Crossing-time difference (metadata, not a filter)")
    axs[1].scatter([e["x"] for e in events], [e["y"] for e in events], s=2, alpha=0.25, color="#55a868")
    axs[1].add_patch(plt.Rectangle((-52.5, -34), 105, 68, fill=False))
    axs[1].set_aspect("equal")
    axs[1].set_title("Intersection points (RAW pitch coordinates)")
    cs = np.array(sorted({e["cluster_id"]: e["cluster_size"] for e in events}.values()))
    axs[2].hist(np.clip(cs, 1, 30), bins=np.arange(0.5, 31.5, 1), color="#c44e52")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("events per pair cluster (30 = 30 or more)")
    axs[2].set_title("Repeated crossings of the same pair")
    fig.tight_layout()
    return _png(fig)


def duplicate_figure(c: RawCache, tl: Timeline, dup: np.ndarray) -> tuple[bytes, dict[str, Any]] | None:
    """The slot pair with the most bit-identical positions, over 10 s of its longest duplicate stretch."""
    plt = _plt()
    best = None
    for a in range(c.n_slots):
        for b in range(a + 1, c.n_slots):
            eq = dup[:, a] & dup[:, b] & (c.x[:, a] == c.x[:, b]) & (c.y[:, a] == c.y[:, b])
            n = int(eq.sum())
            if n and (best is None or n > best[0]):
                best = (n, a, b, eq)
    if best is None:
        return None
    n, a, b, eq = best
    r0 = int(np.flatnonzero(eq)[0])
    rows = np.arange(r0, min(r0 + 300, c.n_frames))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(c.x[rows, a], c.y[rows, a], "-", color=COLORS[0], lw=4, alpha=0.6, label=f"{c.slot_side[a]} #{c.slot_jersey[a]} (slot {a})")
    ax.plot(c.x[rows, b], c.y[rows, b], "--", color=COLORS[1], lw=1.5, label=f"{c.slot_side[b]} #{c.slot_jersey[b]} (slot {b})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    info = dict(slot_a=a, slot_b=b, identical_frames=n, first_frame=int(c.frame_num[r0]), first_video_time_s=float(c.video_time_s[r0]),
                period=int(c.canonical_period[r0]))
    ax.set_title(f"Provider artifact: two slots with bit-identical positions ({n} frames)", fontsize=10)
    fig.text(0.01, -0.02, f"match {c.match_id} · period {info['period']} · frames {c.frame_num[rows[0]]}-{c.frame_num[rows[-1]]} · "
             f"identical positions in {int(eq[rows].sum())} of {len(rows)} frames shown", fontsize=8, family="monospace")
    return _png(fig), info
