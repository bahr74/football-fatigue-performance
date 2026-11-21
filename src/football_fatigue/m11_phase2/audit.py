"""Task A: scientific validation of Interaction Catalogue v1 (read-only; 32 train/CV matches).

A1 structure and distributions, counting-inflation invariants; A2 provider-artifact evidence in four separate families; A3
sensitivity (radius, group-link tolerance, chain merge gap, crossing synchrony tolerance, boundary jitter, clone-like exclusion).
The Phase 1 catalogue is only read; every derived quantity is recomputed here from its tables and written to a new folder.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.interaction_catalogue import core as K

from . import common as C

OUT = "m11-phase2-quality-audit-v1"
RADII = (0.25, 0.5, 1.0, 1.5, 2.0)
STATUS = ("both_visible", "mixed", "both_estimated")
LINK_DELTAS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
MERGE_GAPS = (0.0, 0.5, 1.0, 2.0, 5.0)
SYNC_TOLS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
JITTER_M = (0.02, 0.05, 0.10)
CLONE_MIN_EDGES = 10          # Phase 1 definition of a clone-like repeated pair-period (at 0.5 m) ...
CLONE_EST = 0.95              # ... with mean ESTIMATED fraction of its edges >= this
EDGE_COLS = ["match_id", "period", "radius_m", "slot_a", "slot_b", "start", "end", "duration_s", "t_star", "d_min", "trunc_before",
             "trunc_after", "speed_a_m_s", "speed_b_m_s", "rel_speed_m_s", "max_speed_m_s", "max_accel_m_s2", "cv_margin_m", "cv_tier",
             "boundary_uncertain", "est_frac_a", "est_frac_b", "status_composition", "n_jump_flags", "n_overspeed_flags",
             "n_frame_shift_flags", "n_identity_risk_flags", "n_duplicate_samples", "n_oob_samples", "estimated_comotion",
             "reliability_reasons", "reliability", "label_risk_reasons", "label_risk", "team_relation", "edge_id", "node_a", "node_b",
             "n_sync_crossings_in_edge", "n_windows", "edge_chain_id", "n_concurrent_other", "n_sequential_other", "group_d0p00",
             "group_d1p00", "time_within_comotion_s"]


def _vc(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def _pct(a: float, b: float) -> float:
    return float(100.0 * a / b) if b else float("nan")


# ---------------------------------------------------------------- invariants (counting inflation)

def invariants(E: pd.DataFrame, G: pd.DataFrame, EC: pd.DataFrame, WS: pd.DataFrame, CR: pd.DataFrame, PW: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in RADII:
        e = E[E.radius_m == r]
        ws = WS[WS.radius_m == r]
        g = G[G.radius_m == r]
        ec = EC[EC.radius_m == r]
        k: dict[str, Any] = {}
        k["edges"] = int(len(e))
        k["window_stats_edges_equal"] = bool(int(ws["edges"].sum()) == len(e))
        k["edge_id_unique"] = bool(e["edge_id"].is_unique)
        k["window_inflation"] = float(ws["window_detections"].sum() / len(e))
        k["edges_with_zero_windows"] = int((e["n_windows"] == 0).sum())
        k["window_detections_equal_sum_n_windows"] = bool(int(e["n_windows"].sum()) == int(ws["window_detections"].sum()))
        # every edge in exactly one group per link tolerance; group n_edges agree with the edge table
        for d, col in ((0.0, "group_d0p00"), (1.0, "group_d1p00")):
            gg = g[g.link_s == d]
            cnt = e.groupby(["match_id", "period", col]).size().rename("n").reset_index().rename(columns={col: "group_id"})
            m = gg.merge(cnt, on=["match_id", "period", "group_id"], how="outer")
            k[f"groups_partition_edges_d{d:g}"] = bool(m["n"].notna().all() and m["n_edges"].notna().all() and (m["n"] == m["n_edges"]).all()
                                                       and int(gg["n_edges"].sum()) == len(e))
        # simultaneous groups refine connected groups (delta 0 -> delta 1 is a coarsening)
        ref = e.groupby(["match_id", "period", "group_d0p00"])["group_d1p00"].nunique()
        k["d0_groups_refine_d1_groups"] = bool((ref == 1).all())
        cnt = e.groupby(["match_id", "period", "edge_chain_id"]).size()
        k["edge_chains_count_equal"] = bool(len(cnt) == len(ec))
        # transitive inflation: players in a group vs the largest instantaneous component (stored for delta-0 groups only; a
        # connected group's value is the max over its member delta-0 groups)
        g0 = g[g.link_s == 0.0][["match_id", "period", "group_id", "max_simultaneous_players"]]
        m0 = e[["match_id", "period", "group_d0p00", "group_d1p00"]].drop_duplicates().merge(
            g0.rename(columns={"group_id": "group_d0p00"}), on=["match_id", "period", "group_d0p00"], how="left")
        ms1 = m0.groupby(["match_id", "period", "group_d1p00"])["max_simultaneous_players"].max().rename("max_sim").reset_index()
        g1 = g[g.link_s == 1.0].merge(ms1.rename(columns={"group_d1p00": "group_id"}), on=["match_id", "period", "group_id"], how="left")
        for tag, gg, col in (("simultaneous", g[g.link_s == 0.0], "max_simultaneous_players"), ("connected", g1, "max_sim")):
            multi = gg[gg.n_players >= 3]
            k[f"{tag}_groups_3plus"] = int(len(multi))
            k[f"{tag}_3plus_players_gt_max_simultaneous"] = int((multi.n_players > multi[col]).sum())
            k[f"{tag}_3plus_never_3_at_once"] = int((multi[col] < 3).sum())
            k[f"{tag}_3plus_players_over_max_simultaneous_p50_p90_max"] = [float(v) for v in np.quantile(multi.n_players / multi[col], [0.5, 0.9, 1.0])] if len(multi) else []
        out[str(r)] = k
    out["crossing_id_unique"] = bool(CR["crossing_id"].is_unique)
    out["pair_window_sum_n_intersections_over_crossings"] = float(PW["n_intersections"].sum() / len(CR))
    out["all_passed"] = all(v for r in RADII for kk, v in out[str(r)].items() if isinstance(v, bool)) and out["crossing_id_unique"]
    return out


# ---------------------------------------------------------------- A1 distributions

def per_match(E: pd.DataFrame, G: pd.DataFrame, CR: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for (m, r), e in E.groupby(["match_id", "radius_m"]):
        g1 = G[(G.match_id == m) & (G.radius_m == r) & (G.link_s == 1.0)]
        rows.append(dict(match_id=int(m), radius_m=r, edges=len(e), edge_chains=e.groupby(["period", "edge_chain_id"]).ngroups,
                         connected_groups=len(g1), groups_3plus=int((g1.n_players >= 3).sum()),
                         both_visible_share=float((e.status_composition == "both_visible").mean()),
                         cv_confusable_share=float((e.cv_tier == "cv_confusable").mean()),
                         label_risk_share=float(e.label_risk.mean()), reduced_share=float((e.reliability == "reduced").mean()),
                         comotion=int(e.estimated_comotion.sum())))
    T = pd.DataFrame(rows)
    cr = CR.groupby("match_id").agg(crossings=("crossing_id", "size"), sync=("synchronous", "sum")).reset_index()
    T = T.merge(cr, on="match_id", how="left")
    info: dict[str, Any] = {}
    for r, t in T.groupby("radius_m"):
        d: dict[str, Any] = {}
        for col in ("edges", "edge_chains", "connected_groups", "both_visible_share", "cv_confusable_share", "label_risk_share",
                    "reduced_share", "comotion"):
            v = t[col].to_numpy(float)
            med = np.median(v)
            mad = np.median(np.abs(v - med)) * 1.4826
            rz = (v - med) / mad if mad > 0 else np.zeros_like(v)
            d[col] = dict(min=float(v.min()), median=float(med), max=float(v.max()), cv=float(v.std(ddof=1) / v.mean()) if v.mean() else None,
                          outliers_abs_robust_z_gt3={str(int(m)): round(float(z), 2) for m, z in zip(t.match_id, rz) if abs(z) > 3})
        info[str(r)] = d
    v = cr["crossings"].to_numpy(float)
    info["crossings"] = dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()), cv=float(v.std(ddof=1) / v.mean()))
    return T, info


def per_player_pair(E: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        pl = pd.concat([e[["match_id", "slot_a"]].rename(columns={"slot_a": "slot"}), e[["match_id", "slot_b"]].rename(columns={"slot_b": "slot"})])
        cnt = pl.groupby(["match_id", "slot"]).size().to_numpy()
        srt = np.sort(cnt)[::-1]
        top = max(1, int(round(0.1 * len(srt))))
        pairs = e.groupby(["match_id", "slot_a", "slot_b"]).size().to_numpy()
        ps = np.sort(pairs)[::-1]
        ptop = max(1, int(round(0.1 * len(ps))))
        out[str(r)] = dict(player_matches=int(len(cnt)), edges_per_player=C.q(cnt), gini_player=C.gini(cnt),
                           top10pct_players_edge_end_share=float(srt[:top].sum() / srt.sum()),
                           match_pairs=int(len(pairs)), edges_per_pair=C.q(pairs), gini_pair=C.gini(pairs),
                           top10pct_pairs_edge_share=float(ps[:ptop].sum() / ps.sum()),
                           same_team_share=float((e.team_relation == "same_team").mean()))
    return out


def repeat_gaps(E: pd.DataFrame) -> dict[str, Any]:
    """Time between consecutive edges of the same pair (next start - running max end)."""
    out: dict[str, Any] = {}
    bins = [-np.inf, 1.0, 5.0, 30.0, 120.0, np.inf]
    lab = ["<=1s", "1-5s", "5-30s", "30-120s", ">120s"]
    for r, e in E.groupby("radius_m"):
        e = e.sort_values(["match_id", "period", "slot_a", "slot_b", "start", "end"])
        key = ["match_id", "period", "slot_a", "slot_b"]
        prev_end = e.groupby(key)["end"].cummax().groupby([e[k] for k in key]).shift(1)
        gap = (e["start"] - prev_end).dropna()
        out[str(r)] = dict(n_repeat_gaps=int(len(gap)), quantiles_s=C.q(gap, (0.1, 0.25, 0.5, 0.75, 0.9)),
                           bands=_vc(pd.cut(gap, bins, labels=lab)))
    return out


def distributions(E: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        out[str(r)] = dict(duration_s=C.q(e.duration_s), d_min_m=C.q(e.d_min), d_min_over_r=C.q(e.d_min / r, (0.1, 0.25, 0.5, 0.75, 0.9)),
                           rel_speed_m_s=C.q(e.rel_speed_m_s), max_accel_m_s2=C.q(e.max_accel_m_s2), cv_margin_m=C.q(e.cv_margin_m),
                           duration_lt_0p1s=int((e.duration_s < 0.1).sum()), duration_ge_2s=int((e.duration_s >= 2).sum()),
                           single_sample_like_lt_1_15s=int((e.duration_s < 1 / 15).sum()))
    return out


def crossings_dist(CR: pd.DataFrame) -> dict[str, Any]:
    s = CR[CR.synchronous]
    a = CR[~CR.synchronous]
    return dict(n=int(len(CR)), synchronous=int(len(s)), asynchronous=int(len(a)), abs_dt_s=C.q(CR.abs_dt_s, (0.1, 0.25, 0.5, 0.75, 0.9)),
                d_sync_sync_m=C.q(s.d_sync_m), d_sync_async_m=C.q(a.d_sync_m), kind=_vc(CR["kind"]),
                sync_within_r={c: int(s[c].sum()) for c in CR.columns if c.startswith("within_r")},
                async_within_r={c: int(a[c].sum()) for c in CR.columns if c.startswith("within_r")})


def groups_dist(G: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (r, d), g in G.groupby(["radius_m", "link_s"]):
        out[f"{r}|{d:g}"] = dict(groups=int(len(g)), n_players=_vc(g.n_players.clip(upper=8)), duration_s=C.q(g.duration_s),
                                 max_simultaneous=_vc(g.max_simultaneous_players.clip(upper=8)),
                                 mixed_team_3plus=int(((g.n_players >= 3) & (g.n_home > 0) & (g.n_away > 0)).sum()),
                                 groups_3plus=int((g.n_players >= 3).sum()))
    return out


def context(E: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        ctx = np.where(e.n_concurrent_other > 0, "simultaneous", np.where(e.n_sequential_other > 0, "sequential", "isolated"))
        out[str(r)] = _vc(pd.Series(ctx))
    return out


def extremes(E: pd.DataFrame, G: pd.DataFrame, PW: pd.DataFrame, k: int = 8) -> dict[str, Any]:
    cols = ["match_id", "period", "radius_m", "slot_a", "slot_b", "start", "duration_s", "d_min", "status_composition", "cv_margin_m",
            "max_accel_m_s2", "reliability_reasons", "label_risk_reasons"]
    e5 = E[E.radius_m == 0.5]
    return dict(
        longest_edges_0p5=e5.nlargest(k, "duration_s")[cols].to_dict("records"),
        most_negative_cv_margin_0p5=e5.nsmallest(k, "cv_margin_m")[cols].to_dict("records"),
        highest_accel_0p5=e5.nlargest(k, "max_accel_m_s2")[cols].to_dict("records"),
        largest_simultaneous_groups_1m=G[(G.radius_m == 1.0) & (G.link_s == 0.0)].nlargest(k, "max_simultaneous_players")[
            ["match_id", "period", "group_id", "n_players", "max_simultaneous_players", "duration_s", "start", "frac_edges_both_estimated"]].to_dict("records"),
        most_crossings_pair_windows=PW.nlargest(k, "n_intersections").to_dict("records"))


# ---------------------------------------------------------------- A2 provider artifacts (four evidence families)

def clone_pairs(E: pd.DataFrame) -> pd.DataFrame:
    """Clone-like pair-periods (Phase 1 definition, at 0.5 m): >= 10 edges and mean ESTIMATED fraction >= 0.95."""
    e = E[E.radius_m == 0.5].assign(est=lambda d: (d.est_frac_a + d.est_frac_b) / 2)
    pp = e.groupby(["match_id", "period", "slot_a", "slot_b"]).agg(n=("est", "size"), est=("est", "mean"), team=("team_relation", "first")).reset_index()
    return pp[(pp.n >= CLONE_MIN_EDGES) & (pp.est >= CLONE_EST)]


def artifacts(E: pd.DataFrame, N: pd.DataFrame, clones: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key = ["match_id", "period", "slot_a", "slot_b"]
    ck = set(map(tuple, clones[key].to_numpy()))
    ndur = N.set_index(["match_id", "node_id"])["duration_s"]
    for r, e in E.groupby("radius_m"):
        cl = np.fromiter((tuple(v) in ck for v in e[key].to_numpy()), bool, len(e))
        fam: dict[str, Any] = {}
        by = {}
        for s in STATUS:
            x = e[e.status_composition == s]
            n = len(x)
            if not n:
                continue
            by[s] = dict(
                n=n,
                # 1. measurement quality
                overspeed_step=_pct((x.n_overspeed_flags > 0).sum(), n), speed_gt12=_pct((x.max_speed_m_s > 12).sum(), n),
                accel_gt25=_pct((x.max_accel_m_s2 > 25).sum(), n), accel_p50=float(x.max_accel_m_s2.median()),
                accel_p90=float(x.max_accel_m_s2.quantile(0.9)), out_of_pitch=_pct((x.n_oob_samples > 0).sum(), n),
                # 2. identity-label reliability
                label_risk=_pct(x.label_risk.sum(), n), jump=_pct((x.n_jump_flags > 0).sum(), n),
                identity_flag=_pct((x.n_identity_risk_flags > 0).sum(), n), frame_shift=_pct((x.n_frame_shift_flags > 0).sum(), n),
                chain_truncated=_pct((x.trunc_before | x.trunc_after).sum(), n),
                # 3. event-boundary uncertainty
                boundary_uncertain=_pct(x.boundary_uncertain.sum(), n), duration_lt_0p1s=_pct((x.duration_s < 0.1).sum(), n),
                cv_unavailable=_pct((x.cv_tier == "unavailable").sum(), n),
                # 4. interpolation artifacts
                estimated_comotion=_pct(x.estimated_comotion.sum(), n), duplicate_sample=_pct((x.n_duplicate_samples > 0).sum(), n),
                d_min_lt_0p05=_pct((x.d_min < 0.05).sum(), n), clone_pair=_pct(cl[(e.status_composition == s).to_numpy()].sum(), n),
                cv_confusable=_pct((x.cv_tier == "cv_confusable").sum(), n),
                time_within_comotion_p90=float(x.time_within_comotion_s.quantile(0.9)))
        fam["by_status"] = by
        lr = e.label_risk_reasons[e.label_risk].str.split(";").explode()
        fam["label_risk_reasons"] = _vc(lr)
        rr = e.reliability_reasons[e.reliability == "reduced"].str.split(";").explode()
        fam["reliability_reasons"] = _vc(rr)
        # chain (node) duration of the two players' provider chains: short chains = frequent identity breaks
        da = ndur.reindex(pd.MultiIndex.from_arrays([e.match_id, e.node_a])).to_numpy()
        db = ndur.reindex(pd.MultiIndex.from_arrays([e.match_id, e.node_b])).to_numpy()
        mn = np.fmin(da, db)
        fam["shorter_chain_duration_s"] = C.q(mn, (0.1, 0.25, 0.5))
        fam["shorter_chain_lt_5s"] = int((mn < 5).sum())
        fam["clone_pair_edges"] = int(cl.sum())
        fam["comotion_edges"] = int(e.estimated_comotion.sum())
        fam["comotion_in_clone_pairs"] = int((e.estimated_comotion.to_numpy() & cl).sum())
        fam["label_risk_and_reduced"] = int((e.label_risk & (e.reliability == "reduced")).sum())
        fam["any_flag"] = int((e.label_risk | (e.reliability == "reduced") | e.boundary_uncertain).sum())
        fam["no_flag"] = int(len(e) - fam["any_flag"])
        out[str(r)] = fam
    out["clone_pairs"] = dict(n=int(len(clones)), same_team=int((clones.team == "same_team").sum()),
                              matches_with_any=int(clones.match_id.nunique()), per_match=_vc(clones.match_id))
    return out


# ---------------------------------------------------------------- A3 sensitivity

def link_sensitivity(E: pd.DataFrame, G: pd.DataFrame) -> dict[str, Any]:
    """Groups re-derived with ``core.link_components`` for several tolerances; delta 0 and 1 must equal the stored groups."""
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        res = {}
        parts = list(e.groupby(["match_id", "period"]))
        for d in LINK_DELTAS:
            n_groups = n3 = biggest = 0
            for _, x in parts:
                lab = K.link_components(x.start.to_numpy(), x.end.to_numpy(), x.slot_a.to_numpy(), x.slot_b.to_numpy(), d)
                pl = pd.DataFrame(dict(g=np.r_[lab, lab], p=np.r_[x.slot_a.to_numpy(), x.slot_b.to_numpy()])).drop_duplicates()
                sz = pl.groupby("g").size()
                n_groups += int(lab.max() + 1)
                n3 += int((sz >= 3).sum())
                biggest = max(biggest, int(sz.max()))
            res[f"{d:g}"] = dict(groups=n_groups, groups_3plus=n3, largest=biggest)
        for d in (0.0, 1.0):
            res[f"stored_equal_d{d:g}"] = bool(res[f"{d:g}"]["groups"] == int(((G.radius_m == r) & (G.link_s == d)).sum()))
        out[str(r)] = res
    return out


def merge_sensitivity(E: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        e = e.sort_values(["match_id", "period", "slot_a", "slot_b", "start", "end"])
        key = [e[k] for k in ("match_id", "period", "slot_a", "slot_b")]
        prev = e.groupby(key)["end"].cummax().groupby(key).shift(1)
        gap = (e["start"] - prev).to_numpy()
        out[str(r)] = {f"{g:g}": int((~(gap <= g + 1e-12)).sum()) for g in MERGE_GAPS}
    return out


def sync_sensitivity(CR: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for t in SYNC_TOLS:
        s = CR[CR.abs_dt_s <= t + 1e-12]
        out[f"{t:g}"] = dict(crossings=int(len(s)), share=float(len(s) / len(CR)), distinct_pairs=int(s.groupby(["match_id", "slot_a", "slot_b"]).ngroups),
                             d_sync_p50=float(s.d_sync_m.median()), d_sync_le_0p5=int((s.d_sync_m <= 0.5).sum()))
    return out


def jitter_sensitivity(E: pd.DataFrame) -> dict[str, Any]:
    """Uniform separation bias of +eps / -eps. Lost: edges whose minimum would exceed R (exact for a uniform bias, ignoring splits).
    Gained [EST]: edges of the next catalogue radius whose minimum lies in (R, R + eps] (each is at least one new contact)."""
    out: dict[str, Any] = {}
    for i, r in enumerate(RADII[:-1]):
        e = E[E.radius_m == r]
        nxt = E[E.radius_m == RADII[i + 1]]
        res = {}
        for eps in JITTER_M:
            lost = int((e.d_min > r - eps).sum())
            gained = int(((nxt.d_min > r) & (nxt.d_min <= r + eps)).sum())
            res[f"{eps:g}"] = dict(lost_plus_bias=lost, lost_share=float(lost / len(e)), gained_minus_bias_est=gained,
                                   gained_share_est=float(gained / len(e)))
        out[str(r)] = res
    return out


def clone_sensitivity(E: pd.DataFrame, clones: pd.DataFrame) -> dict[str, Any]:
    key = ["match_id", "period", "slot_a", "slot_b"]
    ck = set(map(tuple, clones[key].to_numpy()))
    out: dict[str, Any] = {}
    for r, e in E.groupby("radius_m"):
        cl = np.fromiter((tuple(v) in ck for v in e[key].to_numpy()), bool, len(e))
        com = e.estimated_comotion.to_numpy(bool)
        res = {}
        for name, drop in (("all", np.zeros(len(e), bool)), ("excl_comotion_edges", com), ("excl_clone_pairs", cl), ("excl_both", com | cl)):
            x = e[~drop]
            res[name] = dict(edges=int(len(x)), edge_chains=int(x.groupby(["match_id", "period", "edge_chain_id"]).ngroups),
                             distinct_pairs=int(x.groupby(["match_id", "slot_a", "slot_b"]).ngroups),
                             status=_vc(x.status_composition), cv_tier=_vc(x.cv_tier),
                             cv_confusable_share=float((x.cv_tier == "cv_confusable").mean()),
                             both_visible_share=float((x.status_composition == "both_visible").mean()),
                             label_risk_share=float(x.label_risk.mean()), d_min_lt_0p05=int((x.d_min < 0.05).sum()))
        out[str(r)] = res
    return out


# ---------------------------------------------------------------- driver

def tables_md(R: dict[str, Any]) -> str:
    L: list[str] = ["# Phase 2 quality audit: tables (generated)", ""]
    L += ["## Invariants", "", "| R | edges | window inflation | partition δ0 / δ1 | δ0 refines δ1 | 3+ simultaneous groups (never 3 at once) | 3+ connected groups (never 3 at once) |", "|---|---|---|---|---|---|---|"]
    for r in RADII:
        k = R["invariants"][str(r)]
        L.append(f"| {r} | {k['edges']:,} | {k['window_inflation']:.2f} | {k['groups_partition_edges_d0']} / {k['groups_partition_edges_d1']} | "
                 f"{k['d0_groups_refine_d1_groups']} | {k['simultaneous_groups_3plus']:,} ({k['simultaneous_3plus_never_3_at_once']:,}) | "
                 f"{k['connected_groups_3plus']:,} ({k['connected_3plus_never_3_at_once']:,}) |")
    L += ["", "## Provider-artifact families by status (percent of edges)", ""]
    cols = ["n", "overspeed_step", "accel_gt25", "accel_p50", "label_risk", "jump", "chain_truncated", "boundary_uncertain",
            "duration_lt_0p1s", "estimated_comotion", "clone_pair", "d_min_lt_0p05", "cv_confusable"]
    for r in RADII:
        L += [f"### R = {r} m", "", "| status | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for s, v in R["artifacts"][str(r)]["by_status"].items():
            L.append(f"| {s} | " + " | ".join(f"{v[c]:,.1f}" if isinstance(v[c], float) else f"{v[c]:,}" for c in cols) + " |")
        L.append("")
    L += ["## Sensitivity: group link tolerance (groups / 3+ / largest)", "", "| R | " + " | ".join(f"δ={d:g}" for d in LINK_DELTAS) + " |", "|---" * (len(LINK_DELTAS) + 1) + "|"]
    for r in RADII:
        v = R["sensitivity"]["link"][str(r)]
        L.append(f"| {r} | " + " | ".join(f"{v[f'{d:g}']['groups']:,} / {v[f'{d:g}']['groups_3plus']:,} / {v[f'{d:g}']['largest']}" for d in LINK_DELTAS) + " |")
    L += ["", "## Sensitivity: chain merge gap (edge chains)", "", "| R | " + " | ".join(f"{g:g} s" for g in MERGE_GAPS) + " |", "|---" * (len(MERGE_GAPS) + 1) + "|"]
    for r in RADII:
        v = R["sensitivity"]["merge"][str(r)]
        L.append(f"| {r} | " + " | ".join(f"{v[f'{g:g}']:,}" for g in MERGE_GAPS) + " |")
    L += ["", "## Sensitivity: crossing synchrony tolerance", "", "| τ (s) | crossings | share | distinct pairs | d_sync p50 | d_sync ≤ 0.5 m |", "|---|---|---|---|---|---|"]
    for t, v in R["sensitivity"]["sync"].items():
        L.append(f"| {t} | {v['crossings']:,} | {100 * v['share']:.1f} % | {v['distinct_pairs']:,} | {v['d_sync_p50']:.2f} | {v['d_sync_le_0p5']:,} |")
    L += ["", "## Sensitivity: boundary jitter (uniform separation bias)", "", "| R | ε | lost (+ε) | lost share | gained (−ε) [EST] | gained share |", "|---|---|---|---|---|---|"]
    for r, v in R["sensitivity"]["jitter"].items():
        for eps, w in v.items():
            L.append(f"| {r} | {eps} | {w['lost_plus_bias']:,} | {100 * w['lost_share']:.1f} % | {w['gained_minus_bias_est']:,} | {100 * w['gained_share_est']:.1f} % |")
    L += ["", "## Sensitivity: clone-like exclusion", "", "| R | variant | edges | edge chains | pairs | CV-confusable | both VISIBLE | label risk | d_min < 5 cm |", "|---|---|---|---|---|---|---|---|---|"]
    for r in RADII:
        for name, w in R["sensitivity"]["clone"][str(r)].items():
            L.append(f"| {r} | {name} | {w['edges']:,} | {w['edge_chains']:,} | {w['distinct_pairs']:,} | {100 * w['cv_confusable_share']:.1f} % | "
                     f"{100 * w['both_visible_share']:.1f} % | {100 * w['label_risk_share']:.1f} % | {w['d_min_lt_0p05']:,} |")
    return "\n".join(L) + "\n"


def main(matches: list[str] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    ms = C.checked(matches)                      # boundary check before anything is read
    out = C.Out(OUT)
    E = C.load_table("edges", ms, EDGE_COLS)
    G = C.load_table("groups", ms)
    EC = C.load_table("edge_chains", ms)
    WS = C.load_table("window_stats", ms)
    CR = C.load_table("crossings", ms)
    PW = C.load_table("pair_windows", ms)
    N = C.load_table("nodes", ms, ["match_id", "node_id", "duration_s"])
    R: dict[str, Any] = dict(n_matches=len(ms), totals=dict(edges={str(r): int((E.radius_m == r).sum()) for r in RADII},
                                                             edge_chains={str(r): int((EC.radius_m == r).sum()) for r in RADII},
                                                             crossings=int(len(CR))))
    R["invariants"] = invariants(E, G, EC, WS, CR, PW)
    PM, R["per_match"] = per_match(E, G, CR)
    R["players_pairs"] = per_player_pair(E)
    R["repeat_gaps"] = repeat_gaps(E)
    R["distributions"] = distributions(E)
    R["crossings"] = crossings_dist(CR)
    R["groups"] = groups_dist(G)
    R["context"] = context(E)
    R["extremes"] = extremes(E, G, PW)
    clones = clone_pairs(E)
    R["artifacts"] = artifacts(E, N, clones)
    R["sensitivity"] = dict(link=link_sensitivity(E, G), merge=merge_sensitivity(E), sync=sync_sensitivity(CR),
                            jitter=jitter_sensitivity(E), clone=clone_sensitivity(E, clones))
    R["sensitivity"]["checks"] = dict(
        merge_gap_1s_equals_stored_chains=all(R["sensitivity"]["merge"][str(r)]["1"] == R["totals"]["edge_chains"][str(r)] for r in RADII),
        link_d0_d1_equal_stored_groups=all(R["sensitivity"]["link"][str(r)][f"stored_equal_d{d:g}"] for r in RADII for d in (0.0, 1.0)))
    R["runtime_s"] = round(time.perf_counter() - t0, 1)
    out.json("audit_results.json", R)
    out.parquet("per_match_radius.parquet", PM)
    out.parquet("clone_like_pair_periods.parquet", clones.reset_index(drop=True))
    out.text("tables.md", tables_md(R))
    out.manifest(task="Phase 2 Task A1-A3 quality audit", inputs=C.catalogue_inputs(ms),
                 code=C.code_sha256(["audit.py", "common.py"]), command="python -m football_fatigue.m11_phase2 audit")
    return R


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["invariants"], indent=1)[:3000])
