"""Stage S1 visual inspection: audit figures by category and a blinded manual-review worksheet (v2).

*Audit figures* (``root.fig-<category>-<k>.png``) show identities, times, measurements and the classification. *Review figures*
(``root.review-<id>.png``) are blinded: tracks are "track 1/2" in a seeded order and no identity, time difference, threshold,
measurement or category is shown. The mapping is only in ``root.review_key.jsonl``; the reviewer works from
``root.review_worksheet.csv``. Every figure shows both players' complete RAW trajectories within the +-2.5 s window (nothing
hidden): filled markers = VISIBLE update instants, hollow = ESTIMATED, red x = Step 1 implausible-jump landing, square = position
at the window centre; right: same-time distance and speeds. Examples and review items are chosen by fixed rules and a seeded hash,
never by any model output. No figure is evidence of visual occlusion.
"""

from __future__ import annotations

import csv
import hashlib
import io
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q

from .figures import _plt, _png

COLORS = ("#1f77b4", "#ff7f0e")
EXAMPLES_PER_CATEGORY = 3
REVIEW_STRATA = (("clean_candidate", 15), ("uncertain_aligned", 15), ("timing_boundary", 10), ("non_aligned_near_threshold", 10),
                 ("invalid_aligned", 5))
WORKSHEET_COLUMNS = ("review_id", "figure", "same_time_passage_yes_no_unclear", "tracking_plausible_yes_no_unclear",
                     "identity_continuity_plausible_yes_no_unclear", "notes")


def _h(seed: str, *parts: str) -> str:
    return hashlib.sha256("|".join((seed, *parts)).encode()).hexdigest()


def interaction_figure(cx, ev: dict, rec: dict, cfg, title: str, blinded: bool, order: tuple[int, int] | None = None) -> bytes:
    plt = _plt()
    c, tl = cx.c, cx.tl
    m = rec["measurements"]
    p, t_star = ev["period"], m["t_star_s"]
    rows = tl.between(p, *m["window_s"])
    t = c.video_time_s[rows] - t_star
    slots = (ev["slot_a"], ev["slot_b"]) if order is None else order
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.25, 1))
    ax, ad, av = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for k, s in enumerate(slots):
        x, y = c.x[rows, s], c.y[rows, s]
        lab = f"track {k + 1}" if blinded else f"{c.slot_side[s]} #{c.slot_jersey[s]} (slot {s})"
        ax.plot(x, y, "-", color=COLORS[k], lw=1.3, alpha=0.8, label=lab)
        u = cx.upd[rows, s]
        vis = c.visibility[rows, s]
        ax.scatter(x[u & (vis == 1)], y[u & (vis == 1)], s=14, color=COLORS[k], zorder=3)
        ax.scatter(x[u & (vis == 2)], y[u & (vis == 2)], s=14, facecolors="none", edgecolors=COLORS[k], zorder=3)
        j = (cx.mask[rows, s] & Q.IMPLAUSIBLE_JUMP) != 0
        ax.plot(x[j], y[j], "x", color="red", ms=9, mew=2, zorder=4)
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok):
            ax.plot(x[ok[0]], y[ok[0]], "o", color=COLORS[k], ms=8, mec="black", zorder=4)
            ax.annotate("", xy=(x[ok[-1]], y[ok[-1]]), xytext=(x[ok[max(len(ok) - 3, 0)]], y[ok[max(len(ok) - 3, 0)]]),
                        arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=2))
        r0 = int(np.argmin(np.abs(t)))
        ax.plot(x[r0], y[r0], "s", color=COLORS[k], ms=8, mec="black", zorder=5)
        ur = rows[u]
        if len(ur) > 1:
            sp = np.hypot(np.diff(c.x[ur, s]), np.diff(c.y[ur, s])) / np.diff(c.video_time_s[ur])
            av.plot((c.video_time_s[ur][1:] + c.video_time_s[ur][:-1]) / 2 - t_star, sp, color=COLORS[k], lw=1.2)
        if not blinded:
            tc = ev["time_a_s"] if s == ev["slot_a"] else ev["time_b_s"]
            for a in (ad, av):
                a.axvline(tc - t_star, color=COLORS[k], ls="--", lw=1)
    ax.plot(ev["x"], ev["y"], "*", color="crimson", ms=16, mec="black", zorder=6, label="path intersection")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m, RAW)")
    ax.set_ylabel("y (m, RAW)")
    ax.legend(fontsize=7, loc="best")
    d = np.hypot(c.x[rows, slots[0]] - c.x[rows, slots[1]], c.y[rows, slots[0]] - c.y[rows, slots[1]])
    ad.plot(t, d, color="black", lw=1.3)
    ad.set_ylabel("same-time distance (m)")
    av.set_ylabel("speed (m/s)")
    av.set_xlabel("time relative to window centre (s)")
    for a in (ad, av):
        a.grid(alpha=0.3)
        a.set_xlim(-cfg.window_half_s, cfg.window_half_s)
    ax.set_title(title, fontsize=10)
    if not blinded:
        st = rec["status"]
        lines = [f"match {ev['match_id']} · period {p} · event {ev['event_id']} · frames {int(c.frame_num[rows[0]])}-{int(c.frame_num[rows[-1]])} "
                 f"· window centre video t = {t_star:.3f} s",
                 f"|dt| {m['abs_time_diff_s']:.3f} s · timing sigma {m['timing_sigma_s']} s · angle {m['crossing_angle_deg']} deg · speeds "
                 f"{m['motion_a']['speed_m_s']} / {m['motion_b']['speed_m_s']} m/s · min same-time distance {m['min_same_time_distance_m']} m",
                 f"ESTIMATED share {m['window_a']['estimated_share']:.2f} / {m['window_b']['estimated_share']:.2f} · jumps "
                 f"{m['window_a']['implausible_jumps']} / {m['window_b']['implausible_jumps']} · straightness "
                 f"{m['motion_a']['straightness']} / {m['motion_b']['straightness']} · v1 pair cluster size {rec['v1_cluster_size']}",
                 *[f"tau {k}: {s['category']}  {', '.join(s['reasons']) or '-'}" for k, s in st.items()]]
        fig.text(0.01, -0.01, "\n".join(lines), fontsize=7.3, va="top", family="monospace")
    else:
        fig.text(0.01, -0.01, "Filled = VISIBLE, hollow = ESTIMATED (provider flag), red x = flagged jump, square = position at the "
                 "window centre, star = path intersection. Nothing is hidden.", fontsize=7.5, va="top")
    fig.tight_layout()
    return _png(fig)


def example_sets(recs: list[dict], key: str) -> dict[str, list[dict]]:
    def st(r):
        return r["status"][key]

    def aligned_unc(r, reason):
        return st(r)["category"] == "uncertain_candidate" and st(r)["timing"] == "aligned" and reason in st(r)["reasons"]

    return {
        "clean_near_simultaneous": [r for r in recs if st(r)["category"] == "clean_candidate"],
        "temporal_mismatch": [r for r in recs if st(r)["category"] == "non_time_aligned_crossing" and r["measurements"]["abs_time_diff_s"] <= 1.0],
        "tracking_jitter": [r for r in recs if aligned_unc(r, "jitter")],
        "implausible_jump": [r for r in recs if aligned_unc(r, "implausible_jump")],
        "estimated_heavy": [r for r in recs if aligned_unc(r, "all_estimated_player")],
        "repeated_pair_crossings": [r for r in recs if st(r)["timing"] == "aligned" and r["v1_cluster_size"] >= 10],
        "uncertain_crossing_time": [r for r in recs if st(r)["category"] == "uncertain_candidate"
                                    and ("timing_boundary" in st(r)["reasons"] or "timing_ill_conditioned" in st(r)["reasons"])],
        "invalid_artifact": [r for r in recs if st(r)["category"] == "invalid_tracking_artifact" and st(r)["timing"] == "aligned"],
    }


def review_strata(recs: list[dict], key: str, tau: float) -> dict[str, list[dict]]:
    def st(r):
        return r["status"][key]

    return {
        "clean_candidate": [r for r in recs if st(r)["category"] == "clean_candidate"],
        "uncertain_aligned": [r for r in recs if st(r)["category"] == "uncertain_candidate" and st(r)["timing"] == "aligned"],
        "timing_boundary": [r for r in recs if st(r)["category"] == "uncertain_candidate" and st(r)["timing"] == "boundary"],
        "non_aligned_near_threshold": [r for r in recs if st(r)["category"] == "non_time_aligned_crossing"
                                       and r["measurements"]["abs_time_diff_s"] <= 0.5],
        "invalid_aligned": [r for r in recs if st(r)["category"] == "invalid_tracking_artifact" and st(r)["timing"] == "aligned"],
    }


def build(cx, events: list[dict], recs: list[dict], cfg) -> tuple[dict[str, bytes], list[dict], bytes, list[dict]]:
    key = f"{cfg.primary_s:.2f}"
    by_id = {e["event_id"]: e for e in events}
    figs: dict[str, bytes] = {}
    examples = []
    for name, sub in example_sets(recs, key).items():
        pick = sorted(sub, key=lambda r: _h(cfg.seed, "example", name, r["event_id"]))[:EXAMPLES_PER_CATEGORY]
        for k, r in enumerate(pick, 1):
            fn = f"root.fig-{name}-{k}.png"
            figs[fn] = interaction_figure(cx, by_id[r["event_id"]], r, cfg, f"{name.replace('_', ' ')} ({k}) — audit view", blinded=False)
            examples.append(dict(category=name, file=fn, event_id=r["event_id"], available=len(sub)))
    chosen: list[tuple[str, dict]] = []
    seen = set()
    for stratum, n in REVIEW_STRATA:
        sub = [r for r in review_strata(recs, key, cfg.primary_s)[stratum] if r["event_id"] not in seen]
        for r in sorted(sub, key=lambda r: _h(cfg.seed, "review", stratum, r["event_id"]))[:n]:
            chosen.append((stratum, r))
            seen.add(r["event_id"])
    chosen.sort(key=lambda sr: _h(cfg.seed, "review-order", sr[1]["event_id"]))
    rows, key_rows = [], []
    for stratum, r in chosen:
        rid = "R" + _h(cfg.seed, "review-id", r["event_id"])[:8]
        ev = by_id[r["event_id"]]
        flip = int(_h(cfg.seed, "review-track", r["event_id"])[0], 16) & 1
        order = (ev["slot_b"], ev["slot_a"]) if flip else (ev["slot_a"], ev["slot_b"])
        fn = f"root.review-{rid}.png"
        figs[fn] = interaction_figure(cx, ev, r, cfg, f"review item {rid}", blinded=True, order=order)
        rows.append(dict(review_id=rid, figure=fn))
        key_rows.append(dict(review_id=rid, event_id=r["event_id"], stratum=stratum, track_1_slot=int(order[0]), track_2_slot=int(order[1]),
                             abs_time_diff_s=r["measurements"]["abs_time_diff_s"],
                             categories={k: s["category"] for k, s in r["status"].items()}))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=WORKSHEET_COLUMNS, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in WORKSHEET_COLUMNS})
    return figs, examples, buf.getvalue().encode(), key_rows
