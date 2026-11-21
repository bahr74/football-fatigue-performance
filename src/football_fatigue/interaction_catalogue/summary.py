"""Development-only scientific summary of Interaction Catalogue v1 (all counts exact aggregations of the per-match outputs).

Writes ``<catalogue>/summary/catalogue_summary.json`` and ``summary/tables.md``; refuses to overwrite either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EDGE_COLS = ["match_id", "period", "radius_m", "slot_a", "slot_b", "player_id_a", "player_id_b", "start", "end", "duration_s", "d_min",
             "team_relation", "rel_speed_m_s", "rel_heading_deg", "n_crossings_in_edge", "n_sync_crossings_in_edge", "n_windows",
             "edge_chain_id", "n_concurrent_other", "n_sequential_other", "cv_tier", "cv_margin_m", "boundary_uncertain",
             "status_composition", "est_frac_a", "est_frac_b", "estimated_comotion", "reliability", "reliability_reasons", "label_risk",
             "label_risk_reasons", "max_speed_m_s", "max_accel_m_s2", "trunc_before", "trunc_after", "time_within_comotion_s"]
BINS = ([0, 1, 2, 4, 9, np.inf], ["1", "2", "3-4", "5-9", "10+"])


def _q(s: pd.Series, qs=(0.1, 0.5, 0.9, 0.99)) -> dict[str, float]:
    s = s.dropna()
    return {f"p{int(q * 100)}": round(float(s.quantile(q)), 4) for q in qs} | {"n": int(len(s))} if len(s) else {"n": 0}


def _vc(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def _bin(s: pd.Series) -> dict[str, int]:
    return _vc(pd.cut(s, BINS[0], labels=BINS[1]).astype(str))


def _size(s: pd.Series) -> dict[str, int]:
    return _vc(s.clip(upper=5).map(lambda v: "5+" if v >= 5 else str(int(v))))


def _explode(s: pd.Series) -> dict[str, int]:
    x = s[s != ""].str.split(";").explode()
    return {str(k): int(v) for k, v in x.value_counts().items()}


def load(root: Path, matches: list[str], name: str, cols: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for m in matches:
        p = root / "matches" / m / f"{name}.parquet"
        if p.exists() and p.stat().st_size:
            df = pd.read_parquet(p, columns=cols) if cols else pd.read_parquet(p)
            if len(df):
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize(root: Path, allowed: list[str]) -> int:
    out_dir = root / "summary"
    if (out_dir / "catalogue_summary.json").exists():
        raise SystemExit(f"{out_dir} already summarised; never overwritten")
    matches = sorted((p.name for p in (root / "matches").iterdir() if p.is_dir()), key=int)
    bad = [m for m in matches if m not in set(allowed)]
    if bad:
        raise SystemExit(f"non-approved matches in the catalogue: {bad}")
    ms = {m: json.loads((root / "matches" / m / "match_summary.json").read_text(encoding="utf-8")) for m in matches}
    E = load(root, matches, "edges", EDGE_COLS)
    CH = load(root, matches, "edge_chains")
    G = load(root, matches, "groups")
    CR = load(root, matches, "crossings")
    PW = load(root, matches, "pair_windows")
    WS = load(root, matches, "window_stats")
    N = load(root, matches, "nodes")
    radii = sorted(E["radius_m"].unique())
    S: dict[str, Any] = dict(matches=matches, n_matches=len(matches), radii_m=[float(r) for r in radii])

    # checks
    S["checks"] = dict(
        all_structure_passed=all(v["checks"]["structure"]["passed"] for v in ms.values()),
        all_stage2_reproduced=all(v["checks"]["stage2_reproduction"].get("passed") for v in ms.values()),
        all_v1_reproduced=all(v["checks"]["window_intersections_v1_reproduction"].get("passed") for v in ms.values()),
        stage2_candidates=sum(v["checks"]["stage2_reproduction"]["stage2_candidates"] for v in ms.values()),
        stage2_episodes=sum(v["checks"]["stage2_reproduction"]["stage2_episodes"] for v in ms.values()),
        v1_intersections=sum(v["checks"]["window_intersections_v1_reproduction"]["v1_intersections"] for v in ms.values()),
        v1_primary_events=sum(v["checks"]["window_intersections_v1_reproduction"]["v1_events_primary"] for v in ms.values()),
        v1_never_primary=sum(v["checks"]["window_intersections_v1_reproduction"]["v1_never_primary"] for v in ms.values()))

    # data volume and quality at match level
    S["data"] = dict(
        frames=int(sum(sum(p["rows"] for p in v["timestamps"].values()) for v in ms.values())),
        player_chain_nodes=int(len(N)), chain_duration_s=_q(N["duration_s"]),
        segments={k: int(sum(v["segments"].get(k, 0) for v in ms.values())) for k in next(iter(ms.values()))["segments"]},
        sample_quality={k: int(sum(v["sample_quality"][k] for v in ms.values())) for k in next(iter(ms.values()))["sample_quality"]},
        coordinates=dict(centred_origin_supported_all=all(v["coordinates"]["centred_origin_supported"] for v in ms.values()),
                         oob_samples=int(sum(v["coordinates"]["oob_samples"] for v in ms.values())),
                         available_samples=int(sum(v["coordinates"]["available_samples"] for v in ms.values())),
                         available_non_finite=int(sum(v["coordinates"]["available_non_finite"] for v in ms.values())),
                         invalid_coordinate_flagged=int(sum(v["coordinates"]["invalid_coordinate_flagged_samples"] for v in ms.values())),
                         pitch_sizes=_vc(pd.Series([f"{v['coordinates']['pitch_length_m']}x{v['coordinates']['pitch_width_m']}" for v in ms.values()]))),
        timestamps=dict(non_increasing=int(sum(p["non_increasing_steps"] for v in ms.values() for p in v["timestamps"].values())),
                        frame_gaps=int(sum(p["frame_gaps"] for v in ms.values() for p in v["timestamps"].values())),
                        max_step_s=max(p["max_step_s"] for v in ms.values() for p in v["timestamps"].values())),
        seconds=dict(total=round(sum(v["seconds"] for v in ms.values()), 1), per_match=_q(pd.Series([v["seconds"] for v in ms.values()]))))

    # crossings and windows (radius independent)
    S["crossings"] = dict(
        total=int(len(CR)), synchronous=int(CR["synchronous"].sum()), kind=_vc(CR["kind"]), abs_dt_s=_q(CR["abs_dt_s"]),
        d_sync_m=_q(CR["d_sync_m"]), d_sync_unavailable=int(CR["d_sync_m"].isna().sum()),
        within_radius={f"{r}": int((CR["d_sync_m"] <= r + 1e-12).sum()) for r in radii},
        distinct_pairs=int(CR.groupby(["match_id", "slot_a", "slot_b"]).ngroups),
        pair_windows=int(len(PW)), pair_windows_by_n_intersections=_size(PW["n_intersections"]),
        pair_windows_multi=int((PW["n_intersections"] > 1).sum()), max_intersections_in_one_pair_window=int(PW["n_intersections"].max()),
        pair_windows_with_sync=int((PW["n_synchronous"] > 0).sum()),
        pair_windows_multi_sync=int((PW["n_synchronous"] > 1).sum()))

    per_r: dict[str, Any] = {}
    for R in radii:
        e = E[E["radius_m"] == R]
        ch = CH[CH["radius_m"] == R]
        g0 = G[(G["radius_m"] == R) & (G["link_s"] == 0.0)]
        g1 = G[(G["radius_m"] == R) & (G["link_s"] == 1.0)]
        ws = WS[WS["radius_m"] == R]
        pm = e.groupby("match_id").size()
        pair_n = e.groupby(["match_id", "period", "slot_a", "slot_b"]).size()
        pair_c = ch.groupby(["match_id", "period", "slot_a", "slot_b"]).size()
        pl = pd.concat([e[["match_id", "slot_a"]].rename(columns={"slot_a": "slot"}), e[["match_id", "slot_b"]].rename(columns={"slot_b": "slot"})])
        per_pm = pl.groupby(["match_id", "slot"]).size()
        pid = pd.concat([e["player_id_a"], e["player_id_b"]])
        upairs = pd.DataFrame(dict(a=np.minimum(e["player_id_a"], e["player_id_b"]), b=np.maximum(e["player_id_a"], e["player_id_b"])))
        conc = e["n_concurrent_other"] > 0
        seq = ~conc & (e["n_sequential_other"] > 0)
        pr: dict[str, Any] = dict(
            edges=int(len(e)), edge_chains=int(len(ch)), groups_simultaneous=int(len(g0)), groups_connected=int(len(g1)),
            edges_per_match=dict(min=int(pm.min()), median=float(pm.median()), max=int(pm.max())),
            window_detections=int(ws["window_detections"].sum()), unique_pair_windows=int(ws["unique_pair_windows"].sum()),
            distinct_pairs_match=int(len(pair_n.groupby(level=[0, 2, 3]).size())), distinct_player_id_pairs=int(len(upairs.drop_duplicates())),
            players_with_edges=int(pid.nunique()), player_match_identities=int(len(per_pm)),
            edges_per_pair_period=_bin(pair_n), chains_per_pair_period=_bin(pair_c), pairs_with_10plus_edges=int((pair_n >= 10).sum()),
            edges_in_pairs_10plus=int(pair_n[pair_n >= 10].sum()),
            edges_per_player_match=_q(per_pm), edges_per_chain=_vc(ch["n_edges"].clip(upper=5)),
            team_relation=_vc(e["team_relation"]),
            interaction_context=dict(simultaneous_with_other_partner=int(conc.sum()), sequential_within_1s_only=int(seq.sum()),
                                     isolated=int((~conc & ~seq).sum())),
            groups_simultaneous_by_players=_size(g0["n_players"]), groups_connected_by_players=_size(g1["n_players"]),
            groups_simultaneous_by_max_simultaneous=_size(g0["max_simultaneous_players"]),
            edges_in_simultaneous_groups_by_players={k: int(g0.loc[g0["n_players"].clip(upper=5).map(lambda v: "5+" if v >= 5 else str(int(v))) == k, "n_edges"].sum())
                                                     for k in ("2", "3", "4", "5+")},
            group_lifecycle=dict(merges=int(g0["n_merges"].sum()), splits=int(g0["n_splits"].sum()), joins=int(g0["n_joins"].sum()),
                                 leaves=int(g0["n_leaves"].sum()), groups_with_merge=int((g0["n_merges"] > 0).sum()),
                                 groups_with_split=int((g0["n_splits"] > 0).sum())),
            group_duration_s=dict(simultaneous=_q(g0["duration_s"]), connected=_q(g1["duration_s"])),
            max_group_players=dict(simultaneous=int(g0["n_players"].max()), connected=int(g1["n_players"].max())),
            duration_s=_q(e["duration_s"]), d_min_m=_q(e["d_min"]), rel_speed_m_s=_q(e["rel_speed_m_s"]), rel_heading_deg=_q(e["rel_heading_deg"]),
            edges_with_crossing=int((e["n_crossings_in_edge"] > 0).sum()), edges_with_sync_crossing=int((e["n_sync_crossings_in_edge"] > 0).sum()),
            status_composition=_vc(e["status_composition"]), est_frac_mean=round(float(((e["est_frac_a"] + e["est_frac_b"]) / 2).mean()), 4),
            estimated_comotion=int(e["estimated_comotion"].sum()),
            cv_tier=_vc(e["cv_tier"]), cv_margin_m=_q(e["cv_margin_m"]), boundary_uncertain=int(e["boundary_uncertain"].sum()),
            cv_tier_by_status={k: _vc(v) for k, v in e.groupby("status_composition")["cv_tier"]},
            cv_tier_by_team={k: _vc(v) for k, v in e.groupby("team_relation")["cv_tier"]},
            reliability=_vc(e["reliability"]), reliability_reasons=_explode(e["reliability_reasons"]),
            label_risk=int(e["label_risk"].sum()), label_risk_reasons=_explode(e["label_risk_reasons"]),
            max_speed_m_s=_q(e["max_speed_m_s"]), max_accel_m_s2=_q(e["max_accel_m_s2"], (0.5, 0.9, 0.99, 0.999)))
        # clone-like repeated pairs: >= 10 edges in a period, mostly estimated
        ee = e.assign(est=(e["est_frac_a"] + e["est_frac_b"]) / 2)
        pp = ee.groupby(["match_id", "period", "slot_a", "slot_b"]).agg(n=("est", "size"), est=("est", "mean"), team=("team_relation", "first"))
        rep = pp[(pp["n"] >= 10)]
        pr["repeated_pairs_10plus"] = dict(pairs=int(len(rep)), mostly_estimated=int((rep["est"] >= 0.95).sum()),
                                           mostly_estimated_same_team=int(((rep["est"] >= 0.95) & (rep["team"] == "same_team")).sum()),
                                           edges=int(rep["n"].sum()))
        per_r[f"{R}"] = pr
    S["per_radius"] = per_r
    base = per_r["0.5"]
    S["sensitivity_vs_0p5"] = {r: {k: round(v[k] / base[k], 3) for k in ("edges", "edge_chains", "groups_simultaneous", "groups_connected",
                                                                         "distinct_pairs_match")} for r, v in per_r.items()}
    S["per_match"] = {m: {f"{R}": int(((E["match_id"] == int(m)) & (E["radius_m"] == R)).sum()) for R in radii}
                      | dict(crossings=int((CR["match_id"] == int(m)).sum())) for m in matches}
    out_dir.mkdir(exist_ok=True)
    (out_dir / "catalogue_summary.json").write_text(json.dumps(S, indent=2, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    (out_dir / "tables.md").write_text(tables(S), encoding="utf-8")
    print(json.dumps(S["checks"], indent=1))
    return 0 if all(S["checks"][k] for k in ("all_structure_passed", "all_stage2_reproduced", "all_v1_reproduced")) else 1


def tables(S: dict[str, Any]) -> str:
    P = S["per_radius"]
    rs = list(P)
    L = ["# Interaction Catalogue v1: generated tables", "", f"{S['n_matches']} train/CV development matches. All values exact.", ""]

    def row(name: str, f) -> str:
        return f"| {name} | " + " | ".join(str(f(P[r])) for r in rs) + " |"

    L += ["| Quantity | " + " | ".join(f"R = {r} m" for r in rs) + " |", "|---|" + "---|" * len(rs)]
    for name, f in [("Proximity edges", lambda p: f"{p['edges']:,}"), ("Edge chains (same pair, <= 1 s)", lambda p: f"{p['edge_chains']:,}"),
                    ("Simultaneous groups", lambda p: f"{p['groups_simultaneous']:,}"), ("Connected groups (1 s)", lambda p: f"{p['groups_connected']:,}"),
                    ("Window detections (5 s, 1 s stride)", lambda p: f"{p['window_detections']:,}"),
                    ("Unique pair-windows", lambda p: f"{p['unique_pair_windows']:,}"),
                    ("Distinct pairs (match level)", lambda p: f"{p['distinct_pairs_match']:,}"),
                    ("Edges per match min / median / max", lambda p: f"{p['edges_per_match']['min']:,} / {p['edges_per_match']['median']:,.0f} / {p['edges_per_match']['max']:,}"),
                    ("Same team", lambda p: f"{p['team_relation'].get('same_team', 0):,}"),
                    ("Simultaneous with another partner", lambda p: f"{p['interaction_context']['simultaneous_with_other_partner']:,}"),
                    ("Sequential within 1 s only", lambda p: f"{p['interaction_context']['sequential_within_1s_only']:,}"),
                    ("Isolated", lambda p: f"{p['interaction_context']['isolated']:,}"),
                    ("Simultaneous groups 2 / 3 / 4 / 5+ players", lambda p: " / ".join(f"{p['groups_simultaneous_by_players'].get(k, 0):,}" for k in ("2", "3", "4", "5+"))),
                    ("Connected groups 2 / 3 / 4 / 5+ players", lambda p: " / ".join(f"{p['groups_connected_by_players'].get(k, 0):,}" for k in ("2", "3", "4", "5+"))),
                    ("Groups with merge / split", lambda p: f"{p['group_lifecycle']['groups_with_merge']:,} / {p['group_lifecycle']['groups_with_split']:,}"),
                    ("Duration p50 / p90 (s)", lambda p: f"{p['duration_s'].get('p50')} / {p['duration_s'].get('p90')}"),
                    ("d_min p50 (m)", lambda p: p["d_min_m"].get("p50")),
                    ("Edges with a synchronous crossing", lambda p: f"{p['edges_with_sync_crossing']:,}"),
                    ("Both VISIBLE / mixed / both ESTIMATED", lambda p: " / ".join(f"{p['status_composition'].get(k, 0):,}" for k in ("both_visible", "mixed", "both_estimated"))),
                    ("ESTIMATED co-motion (clone-like)", lambda p: f"{p['estimated_comotion']:,}"),
                    ("CV confusable / marginal / separable / unavailable", lambda p: " / ".join(f"{p['cv_tier'].get(k, 0):,}" for k in ("cv_confusable", "marginal", "cv_separable", "unavailable"))),
                    ("Boundary uncertain (d_min within 5 cm of R)", lambda p: f"{p['boundary_uncertain']:,}"),
                    ("Reliability reduced", lambda p: f"{p['reliability'].get('reduced', 0):,}"),
                    ("Possible label corruption", lambda p: f"{p['label_risk']:,}"),
                    ("Pairs with >= 10 edges in a period (mostly ESTIMATED)", lambda p: f"{p['repeated_pairs_10plus']['pairs']:,} ({p['repeated_pairs_10plus']['mostly_estimated']:,})")]:
        L.append(row(name, f))
    c = S["crossings"]
    L += ["", "## Crossings", "", f"- total {c['total']:,}; synchronous (|tA - tB| <= 0.3 s) {c['synchronous']:,}; distinct pairs {c['distinct_pairs']:,}",
          f"- pair-windows with >= 1 crossing {c['pair_windows']:,}; with > 1 {c['pair_windows_multi']:,}; by count {c['pair_windows_by_n_intersections']}",
          f"- max crossings in one pair-window {c['max_intersections_in_one_pair_window']}; crossings with separation <= R at crossing: {c['within_radius']}", ""]
    return "\n".join(L) + "\n"
