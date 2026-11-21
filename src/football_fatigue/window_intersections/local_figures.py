"""Figures for the four-method local-descriptor pilot. Every event figure shows actual FIFA samples; gallery events are a seeded
random sample per category (not hand-picked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from football_fatigue.encounters import quality as Q  # noqa: E402
from football_fatigue.encounters.trajectories import STATE_AVAILABLE  # noqa: E402

from . import local as L  # noqa: E402
from .figures import BLUE, INK, INK2, MUTED, ORANGE, RED, YELLOW, _png, bars, hist  # noqa: E402


def _short(lab: str) -> str:
    return lab.replace(" +guard", "\n+guard").replace(" tau ", "\ntau ").replace(" arr", "\narr").replace(" pt", "\npt")


def _pass_text(mrow: pd.Series, cols: dict[str, str]) -> str:
    return "; ".join(f"{lab}: {'PASS' if bool(mrow[c]) else 'fail'}" for lab, c in cols.items())


def event_figure(cache: dict, e: pd.Series, d: pd.Series, mrow: pd.Series, cfg: L.LocalConfig, cols: dict[str, str], title: str) -> bytes:
    c, mask, seg, ch = cache["c"], cache["mask"], cache["seg"], cache["ch"]
    cx, cy = cache["coefs"]["linear"]
    lk = L.segment_lookup(seg)
    ta, tb, P = float(e.t_a_s), float(e.t_b_s), (float(e.x_m), float(e.y_m))
    te, tmin, tmax, h = (ta + tb) / 2, min(ta, tb), max(ta, tb), cfg.primary_h_s
    lo, hi = te - 2.5, te + 2.5
    tracks = {}
    for tag, s, row0, t0 in (("A", int(e.slot_a), int(e.seg_a_row0), ta), ("B", int(e.slot_b), int(e.seg_b_row0), tb)):
        g = L.find_segment(lk, seg, s, row0, t0)
        k = int(seg["chain"][g])
        tracks[tag] = (s, t0, L.chain_knots(seg, cx, cy, int(ch["first"][k]), int(ch["last"][k]), lo - 0.01, hi + 0.01, 1),
                       L.chain_knots(seg, cx, cy, int(ch["first"][k]), int(ch["last"][k]), t0 - cfg.occupancy_cap_s, t0 + cfg.occupancy_cap_s, 1))
    est = (mask & Q.ESTIMATED) != 0
    rows = np.flatnonzero((c.canonical_period == int(e.period)) & (c.video_time_s >= lo) & (c.video_time_s <= hi))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ts_ = te + float(d[f"arr_h{L.key(h)}_t_rel_s"]) if np.isfinite(d[f"arr_h{L.key(h)}_t_rel_s"]) else np.nan
    for tag, col in (("A", BLUE), ("B", ORANGE)):
        s, t0, K, _ = tracks[tag]
        ax.plot(K[1], K[2], color=col, lw=1.6, label=f"player {tag} (slot {s}) linear path, chain at the intersection", zorder=2)
        r = rows[c.state[rows, s] == STATE_AVAILABLE]
        vis, es = r[~est[r, s]], r[est[r, s]]
        ax.scatter(c.x[vis, s], c.y[vis, s], s=10, color=col, zorder=4, label=f"{tag} raw VISIBLE" if len(vis) else None)
        ax.scatter(c.x[es, s], c.y[es, s], s=12, facecolors="none", edgecolors=col, lw=0.8, zorder=4, label=f"{tag} raw ESTIMATED" if len(es) else None)
        ax.annotate(f"{tag} start", (K[1][0], K[2][0]), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
        if np.isfinite(ts_):
            x_, y_ = L.at(K, ts_)
            ax.scatter([x_], [y_], marker="s", s=48, color=col, edgecolors=INK, lw=0.8, zorder=6, label=f"{tag} at t* (closest, arr h={h:.2f})")
    if np.isfinite(ts_):
        a_, b_ = L.at(tracks["A"][2], ts_), L.at(tracks["B"][2], ts_)
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=INK, lw=0.9, ls=":", zorder=5)
    for eps, ls in zip(cfg.eps_m, [(0, (1, 1)), (0, (3, 2)), (0, (6, 3))]):
        ax.add_patch(plt.Circle(P, eps, fill=False, ec=MUTED, lw=0.8, ls=ls, zorder=3))
    ax.add_patch(plt.Circle(P, cfg.rho_m, fill=False, ec=YELLOW, lw=0.9, ls="--", zorder=3, label=f"guard rho = {cfg.rho_m:.1f} m"))
    ax.scatter([P[0]], [P[1]], marker="*", s=170, color=INK, zorder=7, label="selected intersection P (circles: eps 0.05/0.10/0.25 m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.legend(fontsize=6.3, loc="best", framealpha=0.9)
    # synchronized distance on the chains (variant curve), time relative to the event time
    KA, KB = tracks["A"][2], tracks["B"][2]
    clo, chi = max(KA[0][0], KB[0][0]), min(KA[0][-1], KB[0][-1])
    if chi > clo:
        tt = np.linspace(clo, chi, 2500)
        dd = np.hypot(np.interp(tt, KA[0], KA[1]) - np.interp(tt, KB[0], KB[1]), np.interp(tt, KA[0], KA[2]) - np.interp(tt, KB[0], KB[2]))
        ax2.plot(tt - te, dd, color=INK2, lw=1.5, label="synchronized distance d(t) (linear, both chains)")
        ymax = max(1.2, float(np.percentile(dd, 90)) * 1.1)
    else:
        ymax = 1.2
    ax2.axvspan(tmin - h - te, tmax + h - te, color=BLUE, alpha=0.10, lw=0, label=f"arr neighbourhood [t- - {h:.2f}, t+ + {h:.2f}]")
    for i, t0 in enumerate((ta, tb)):
        ax2.axvspan(t0 - h - te, t0 + h - te, ymin=0, ymax=0.06, color=ORANGE, alpha=0.35, lw=0, label=f"pt neighbourhood [t_i +- {h:.2f}]" if i == 0 else None)
    ax2.axvline(ta - te, color=BLUE, lw=1.2, ls="-.", label="A arrival at P")
    ax2.axvline(tb - te, color=ORANGE, lw=1.2, ls="-.", label="B arrival at P")
    ax2.axhline(cfg.primary_r_m, color=RED, lw=0.9, ls="--", label=f"R = {cfg.primary_r_m:.2f} m")
    for R in cfg.radii_m[1:]:
        ax2.axhline(R, color=MUTED, lw=0.7, ls=":")
    if np.isfinite(ts_):
        hk = L.key(cfg.primary_h_s)
        ax2.scatter([ts_ - te], [d[f"arr_h{hk}_d_m"]], color=INK, s=34, zorder=6)
        ax2.annotate(f"d_loc = {d[f'arr_h{hk}_d_m']:.3f} m", (ts_ - te, d[f"arr_h{hk}_d_m"]), xytext=(6, 8), textcoords="offset points", fontsize=8)
    for tag, col, y in (("A", BLUE, -0.06), ("B", ORANGE, -0.12)):     # occupancy within eps = 0.25 m of P
        _, t0, _, O = tracks[tag]
        lo_o, hi_o, _ = L.occupancy(O, P, t0, 0.25)
        ax2.plot([lo_o - te, hi_o - te], [y, y], color=col, lw=4, solid_capstyle="butt", label=f"{tag} within 0.25 m of P" if tag == "A" else "B within 0.25 m of P")
    ax2.set_xlim(-2.5, 2.5), ax2.set_ylim(-0.18, ymax)
    ax2.set_xlabel("time relative to the event time (tA + tB) / 2 (s)"), ax2.set_ylabel("distance (m)")
    ax2.legend(fontsize=6.3, loc="upper right", framealpha=0.9)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    hs = ", ".join(f"h{x:.2f}: arr {d[f'arr_h{L.key(x)}_d_m']:.3f} / pt {d[f'pt_h{L.key(x)}_d_m']:.3f}" for x in cfg.hs_s)
    txt = (f"match {int(e.match_id)} period {int(e.period)} slots {int(e.slot_a)}/{int(e.slot_b)} ({e.team_relation}); |dt| = {e.abs_dt_s:.3f} s; "
           f"speeds {d.speed_a_m_s:.2f} / {d.speed_b_m_s:.2f} m/s; crossing angle {d.angle_deg:.0f} deg; tier {e.tier}; status {e.status}\n"
           f"d at tA / tB: {d.d_at_ta_m:.3f} / {d.d_at_tb_m:.3f} m; local minimum (m) {hs}; midpoint at t* is {d[f'arr_h{L.key(cfg.primary_h_s)}_mid_to_p_m']:.2f} m from P\n"
           f"occupancy gap (s): eps0.05 {d.occ_e0p05_gap_s:.3f}, eps0.10 {d.occ_e0p10_gap_s:.3f}, eps0.25 {d.occ_e0p25_gap_s:.3f}, tier {d.occ_tier_gap_s:.3f}\n"
           + _pass_text(mrow, cols))
    fig.text(0.01, -0.01, txt, fontsize=7.6, color=INK2, ha="left", va="top", wrap=True)
    return _png(fig)


def heatmap(ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame, panels: dict[str, str], cfg: L.LocalConfig) -> bytes:
    """Pass rate over (|dt|, slower player's speed). Sequential single hue; cells with < 5 events are left blank."""
    xb, yb = np.arange(0, 1.0001, 0.05), np.arange(0, 8.0001, 0.5)
    x, y = ev["abs_dt_s"].to_numpy(), D["speed_min_m_s"].to_numpy()
    n, _, _ = np.histogram2d(x, y, [xb, yb])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    for ax, (lab, c) in zip(np.atleast_1d(axes), panels.items()):
        k, _, _ = np.histogram2d(x[M[c].to_numpy(bool)], y[M[c].to_numpy(bool)], [xb, yb])
        rate = np.where(n >= 5, k / np.maximum(n, 1), np.nan)
        ax.set_facecolor("#c9c8c2")                          # cells with < 5 events stay grey (never confused with a 0 % rate)
        im = ax.pcolormesh(xb, yb, rate.T, cmap="Blues", vmin=0, vmax=1, shading="flat")
        vv = np.linspace(0.05, 8, 200)
        ax.plot(cfg.primary_r_m / vv, vv, color=ORANGE, lw=1.6, label=f"|dt| = R / v_min (R = {cfg.primary_r_m} m)")
        ax.axvline(cfg.primary_tau_s, color=INK, lw=1.0, ls="--", label=f"tau = {cfg.primary_tau_s:.2f} s")
        ax.set_xlim(0, 1), ax.set_ylim(0, 8)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_xlabel("|tA - tB| (s)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("slower player's speed at P (m/s)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=axes, label="share of primary events passing", shrink=0.85)
    fig.suptitle("Pass rate by arrival-time difference and speed (linear; grey: < 5 events in the cell)", fontsize=10, x=0.01, ha="left")
    return _png(fig)


def scatter_d_dt(ev: pd.DataFrame, D: pd.DataFrame, cfg: L.LocalConfig, seed: int) -> bytes:
    sel = (ev["abs_dt_s"] <= 1.0).to_numpy() & D[f"arr_h{L.key(cfg.primary_h_s)}_d_m"].notna().to_numpy()
    idx = np.flatnonzero(sel)
    idx = np.random.default_rng(seed).choice(idx, min(len(idx), 25000), replace=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sc = ax.scatter(ev["abs_dt_s"].to_numpy()[idx], np.clip(D[f"arr_h{L.key(cfg.primary_h_s)}_d_m"].to_numpy()[idx], 1e-3, None), c=np.clip(D["speed_min_m_s"].to_numpy()[idx], 0, 8),
                    cmap="Blues", vmin=0, vmax=8, s=5, lw=0, alpha=0.8)
    ax.axhline(cfg.primary_r_m, color=RED, lw=1.0, ls="--", label=f"R = {cfg.primary_r_m} m")
    ax.axvline(cfg.primary_tau_s, color=INK, lw=1.0, ls="--", label=f"tau = {cfg.primary_tau_s} s")
    ax.set_yscale("log"), ax.set_xlim(0, 1)
    ax.set_xlabel("|tA - tB| (s)"), ax.set_ylabel(f"local minimum distance, arr h = {cfg.primary_h_s} s (m, log)")
    fig.colorbar(sc, ax=ax, label="slower player's speed at P (m/s)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Local synchronized distance vs arrival-time difference (linear, {len(idx):,} events with |dt| <= 1 s)", fontsize=9.5, loc="left")
    return _png(fig)


def make_all(fig_dir: Path, EV: dict, DD: dict, MM: dict, cfg: L.LocalConfig, cols: dict[str, str], caches: dict, n_gallery: int, seed: int
             ) -> list[tuple[str, list[dict[str, Any]]]]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev, D, M = EV["linear"], DD["linear"], MM["linear"]
    k = L.key
    h, R, t, g = k(cfg.primary_h_s), k(cfg.primary_r_m), k(cfg.primary_tau_s), f"_g{k(cfg.rho_m)}"
    summary: list[dict[str, Any]] = []

    def save(name, png, cap):
        (fig_dir / name).write_bytes(png)
        summary.append(dict(file=name, caption=cap))

    groups = [f"h = {x:.2f} s" for x in cfg.hs_s]
    ser = {lab: [int(M[f"F2_{nb}_h{k(x)}_R{R}{gg}"].sum()) for x in cfg.hs_s]
           for lab, nb, gg in (("arr", "arr", ""), ("arr + guard", "arr", g), ("pt", "pt", ""), ("pt + guard", "pt", g))}
    save("counts_f2_neighbourhoods.png", bars(groups, ser, f"F2 events by neighbourhood and margin h (R = {cfg.primary_r_m} m, linear)", "events",
                                              note=f"arr: [t- - h, t+ + h]; pt: [tA +- h] U [tB +- h]; guard: pair midpoint at t* within {cfg.rho_m} m of P"),
         "F2 counts by neighbourhood definition, margin h and guard")
    taus =[f"tau = {x:.2f} s" for x in cfg.taus_s]
    ser = {"F1 (eps -> 0)": [int(M[f"F1_tau{k(x)}"].sum()) for x in cfg.taus_s]}
    ser |= {f"F4 eps {e:.2f} m": [int(M[f"F4_e{k(e)}_tau{k(x)}"].sum()) for x in cfg.taus_s] for e in cfg.eps_m}
    ser |= {"F4 tier (VIS 0.05 / EST 0.10 m)": [int(M[f"F4_tier_tau{k(x)}"].sum()) for x in cfg.taus_s]}
    save("counts_f1_f4.png", bars(taus, ser, "F1 and F4 events by tau and occupancy radius eps (linear)", "events",
                                  note="F4: the two players' time spans within eps of P come within tau of each other; eps -> 0 recovers F1"),
         "F1 vs F4 counts by tau and eps")
    ser = {lab: [int(M[c.format(R=k(r))].sum()) for r in cfg.radii_m] for lab, c in
           (("F2 arr + guard", f"F2_arr_h{h}_R{{R}}{g}"), ("F2 pt", f"F2_pt_h{h}_R{{R}}"), (f"F3 tau {cfg.primary_tau_s:.2f} arr + guard", f"F3_tau{t}_arr_h{h}_R{{R}}{g}"))}
    save("counts_r_sensitivity.png", bars([f"R = {r:.2f} m" for r in cfg.radii_m], ser, f"Distance-tolerance sensitivity (h = {cfg.primary_h_s} s, linear)", "events",
                                          note="R = 0.50 and 1.00 m are sensitivity checks only"), "R sensitivity")
    ov = L.overlap_table(M, {kk: cols[kk] for kk in cols if kk in (f"F1 tau {cfg.primary_tau_s:.2f}", f"F2 arr h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f} +guard",
                                                                     f"F4 eps 0.25 tau {cfg.primary_tau_s:.2f}")})
    ov = sorted(ov, key=lambda r: -r["events"])
    lab = [" & ".join(x.split(" h")[0].split(" eps")[0].split(" tau")[0] for x in r["members"]) for r in ov]
    save("overlap_f1_f2_f4.png", bars(lab, {"events": [r["events"] for r in ov]}, "Exclusive overlap of F1 (tau 0.30), F2 (arr h0.25 R0.25 + guard) and F4 (eps 0.25, tau 0.30)",
                                      "events", note="each bar counts events in exactly that combination; F3 = F1 & F2 by construction"), "method overlap")
    tiers, sts = ["T1_visible_visible", "T2_mixed", "T3_estimated_estimated"], ["accepted", "flagged", "rejected"]
    gl = [_short(x) for x in cols]
    save("tier_by_method.png", bars(gl, {tt: [int((ev.loc[M[c].to_numpy(bool), "tier"] == tt).sum()) for c in cols.values()] for tt in tiers},
                                    "Quality tier at the intersection, primary configurations (linear)", "events", stacked=True), "tier by method")
    save("status_by_method.png", bars(gl, {s: [int((ev.loc[M[c].to_numpy(bool), "status"] == s).sum()) for c in cols.values()] for s in sts},
                                      "Validation status, primary configurations (linear)", "events", stacked=True), "status by method")
    save("pass_rate_heatmap.png", heatmap(ev, D, M, {f"F1 tau {cfg.primary_tau_s}": f"F1_tau{t}", f"F2 arr h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f} + guard": f"F2_arr_h{h}_R{R}{g}",
                                                     "F4 eps 0.25 tau 0.30": f"F4_e0p25_tau{t}"}, cfg),
         "pass rate over |dt| and the slower player's speed; orange curve = straight-line F2 boundary")
    save("dloc_vs_dt.png", scatter_d_dt(ev, D, cfg, seed), "local distance vs |dt|, coloured by speed")
    nog = M[f"F2_arr_h{h}_R{R}"].to_numpy(bool)
    mid = D[f"arr_h{h}_mid_to_p_m"]
    save("guard_distance.png", hist({"|dt| <= 0.30 s": mid[nog & (ev["abs_dt_s"] <= 0.3).to_numpy()], "|dt| > 0.30 s": mid[nog & (ev["abs_dt_s"] > 0.3).to_numpy()]},
                                    "F2 (arr h0.25, R0.25, no guard): distance of the closest approach from P", "m", np.linspace(0, 6, 61),
                                    note=f"guard rho = {cfg.rho_m} m; events to the right of it are rejected by the guard"), "guard: where the closest approach happens")
    vmin = D["speed_min_m_s"].to_numpy()
    bands = [("<1", vmin < 1), ("1-2", (vmin >= 1) & (vmin < 2)), ("2-4", (vmin >= 2) & (vmin < 4)), ("4-7", (vmin >= 4) & (vmin < 7)), (">=7", vmin >= 7)]
    ser = {}
    for x in cfg.taus_s:
        f1 = M[f"F1_tau{k(x)}"].to_numpy(bool)
        f3 = M[f"F3_tau{k(x)}_arr_h{h}_R{R}{g}"].to_numpy(bool)
        ser[f"tau {x:.2f}"] = [round(float((f3 & s).sum() / max(1, (f1 & s).sum())), 3) for _, s in bands]
    save("f3_share_of_f1_by_speed.png", bars([f"v_min {b} m/s" for b, _ in bands], ser, "Share of F1 events that also pass F2 (= F3 / F1), by the slower player's speed",
                                             "share", note="straight-line motion: F1(tau) implies F2(R) when v_min <= R / tau (0.83 m/s at tau 0.30)"),
         "how much the spatial condition removes from F1, by speed")
    lv = L.variant_agreement(EV["linear"], MM["linear"], EV["pchip"], MM["pchip"], cols)
    save("linear_vs_pchip.png", bars(gl, {"linear": [lv[x]["linear_pass"] for x in cols], "PCHIP": [lv[x]["pchip_pass"] for x in cols]},
                                     "Linear (reference) vs PCHIP (sensitivity): events per primary configuration", "events",
                                     note="membership flips per configuration are in analysis.json -> linear_vs_pchip"), "linear vs PCHIP")
    # galleries
    rng = np.random.default_rng(seed)
    f1, f2g, f2n, f2pt = (M[c].to_numpy(bool) for c in (f"F1_tau{t}", f"F2_arr_h{h}_R{R}{g}", f"F2_arr_h{h}_R{R}", f"F2_pt_h{h}_R{R}"))
    f4e, f4t = M[f"F4_e0p25_tau{t}"].to_numpy(bool), M[f"F4_tier_tau{t}"].to_numpy(bool)
    cats = [("F3: passes F1 (tau 0.30) and F2 (arr h0.25 R0.25 + guard)", "f3", f1 & f2g),
            ("F1 only: |dt| <= 0.30 s but never within 0.25 m locally", "f1_only", f1 & ~f2g),
            ("F2 only: within 0.25 m near P but |dt| > 0.30 s", "f2_only", f2g & ~f1),
            ("Removed by the 1.0 m guard (F2 arr without guard passes)", "guard_removed", f2n & ~f2g),
            ("Neighbourhood definitions disagree: arr + guard passes, pt fails", "arr_not_pt", f2g & ~f2pt),
            ("F4 eps 0.25 m adds over F1 (fixed physical region)", "f4_fixed_adds", f4e & ~f1),
            ("F4 tier eps adds over F1 (measurement uncertainty)", "f4_tier_adds", f4t & ~f1)]
    sections = [("Summary figures (linear reference)", summary)]
    for title, tag, sel in cats:
        idx = np.flatnonzero(sel)
        pick = sorted(rng.choice(idx, min(n_gallery, len(idx)), replace=False).tolist()) if len(idx) else []
        items = []
        for i in pick:
            e, d, mrow = ev.iloc[i], D.iloc[i], M.iloc[i]
            png = event_figure(caches[str(int(e.match_id))], e, d, mrow, cfg, cols, f"{title}\n(random example {len(items) + 1} of {len(pick)}; {len(idx):,} events in category)")
            name = f"gallery_{tag}_{e.event_id}.png"
            (fig_dir / name).write_bytes(png)
            items.append(dict(file=name, caption=f"{title}: match {int(e.match_id)}, event {e.event_id}"))
        sections.append((f"{title} ({len(idx):,} events; {len(pick)} random examples)", items))
    (fig_dir / "figures.json").write_text(json.dumps([dict(section=s, items=i) for s, i in sections], indent=1), encoding="utf-8")
    return sections
