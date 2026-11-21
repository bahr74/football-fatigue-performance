"""Task A4: deterministic stratified inspection sample (~100 development events) for later human review.

The sample is drawn from Interaction Catalogue v1 only (32 train/CV matches). Selection is a pure function of the catalogue and the
seed string: within each stratum, candidates are ordered by sha256(seed | stratum | event key) and taken greedily with at most one
event per match per stratum (spreading the sample over matches). An event already chosen for an earlier stratum is skipped, and
every chosen event lists *all* strata it satisfies. Nothing here is a human judgement: the manifest has empty reviewer fields.

Optional figures (``--figures``) read the RAW Step 1 cache of the sampled matches (train/CV only, boundary-checked first) and draw
the two players' raw positions around t* with ESTIMATED samples hollow. They are diagnostics, not evidence of visual occlusion.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import common as C

OUT = "m11-phase2-review-sample-v1"
SEED = "m11-phase2-review-v1"
FIG_HALF_S = 3.0
REVIEW_FIELDS = ("reviewer", "review_date", "positions_plausible", "identity_continuity_plausible", "players_actually_close",
                 "provider_artifact_suspected", "visual_occlusion_checkable", "notes")


def _h(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def strata(E: pd.DataFrame, G: pd.DataFrame, clone_keys: set) -> list[tuple[str, int, str, Callable[[pd.DataFrame], pd.Series]]]:
    """(name, quota, radius-or-'group', predicate on the edge table at that radius)."""
    def in_clone(e):
        k = list(zip(e.match_id, e.period, e.slot_a, e.slot_b))
        return pd.Series([t in clone_keys for t in k], index=e.index)
    return [
        ("clean_visible", 10, "0.5", lambda e: (e.status_composition == "both_visible") & ~e.label_risk & (e.reliability == "nominal") & ~e.boundary_uncertain),
        ("mixed_status", 8, "0.5", lambda e: e.status_composition == "mixed"),
        ("estimated_dominated", 8, "0.5", lambda e: (e.status_composition == "both_estimated") & ~e.estimated_comotion & ~in_clone(e)),
        ("clone_like", 10, "0.5", lambda e: e.estimated_comotion | in_clone(e)),
        ("repeated_sync_crossings", 8, "0.5", lambda e: e.n_sync_crossings_in_edge >= 2),
        ("group_3_simultaneous", 8, "group", None),
        ("group_4plus_simultaneous", 8, "group", None),
        ("short_event", 6, "0.5", lambda e: e.duration_s < 0.1),
        ("long_event", 6, "0.5", lambda e: e.duration_s >= 2.0),
        ("high_ambiguity_cv", 8, "0.5", lambda e: e.cv_tier == "cv_confusable"),
        ("low_ambiguity_cv", 6, "0.5", lambda e: (e.cv_tier == "cv_separable") & (e.cv_margin_m >= 1.5)),
        ("boundary_sensitive", 6, "0.5", lambda e: e.boundary_uncertain),
        ("broad_radius_ambiguous", 8, "2.0", lambda e: (e.cv_tier == "cv_confusable") & (e.d_min > 1.0)),
    ]


def select(E: pd.DataFrame, G: pd.DataFrame, clone_keys: set) -> pd.DataFrame:
    rows, taken = [], set()
    e05 = E[E.radius_m == 0.5]
    st = strata(E, G, clone_keys)
    for name, quota, where, pred in st:
        if where == "group":
            g = G[(G.radius_m == 0.5) & (G.link_s == 0.0)]
            g = g[(g.max_simultaneous_players == 3)] if name == "group_3_simultaneous" else g[g.max_simultaneous_players >= 4]
            # represent a group by its longest edge (anchor) and keep the group id
            ee = e05.merge(g[["match_id", "period", "group_id", "n_players", "max_simultaneous_players"]].rename(columns={"group_id": "group_d0p00"}),
                           on=["match_id", "period", "group_d0p00"])
            cand = ee.sort_values("duration_s", ascending=False).drop_duplicates(["match_id", "period", "group_d0p00"])
        else:
            e = E[E.radius_m == float(where)]
            cand = e[pred(e)]
        cand = cand.assign(_h=[_h(SEED, name, k) for k in cand.edge_id]).sort_values("_h")
        used_m, n = set(), 0
        for _, r in cand.iterrows():
            if n >= quota:
                break
            if r.edge_id in taken or r.match_id in used_m:
                continue
            taken.add(r.edge_id)
            used_m.add(r.match_id)
            rows.append(r.drop(labels="_h").to_dict() | dict(stratum=name))
            n += 1
    S = pd.DataFrame(rows)
    # every stratum each selected edge satisfies (radius-specific predicates evaluated on the edge's own radius)
    tags = {i: [] for i in S.index}
    for name, _q, where, pred in st:
        if where == "group":
            continue
        sub = S[S.radius_m == float(where)]
        if len(sub):
            ok = pred(sub)
            for i in sub.index[ok.to_numpy()]:
                tags[i].append(name)
    S["all_strata"] = [";".join(sorted(set(tags[i]) | {S.at[i, "stratum"]})) for i in S.index]
    S.insert(0, "review_id", [f"P2R-{i + 1:03d}" for i in range(len(S))])
    for f in REVIEW_FIELDS:
        S[f] = ""
    return S


def figures(S: pd.DataFrame, out: C.Out) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from football_fatigue.encounters import quality as Q
    from football_fatigue.interaction_catalogue.run import RAW
    from football_fatigue.matching.benchmark import load_source

    done = {}
    for m in sorted({str(int(v)) for v in S.match_id}, key=int):
        C.checked([m])                                   # boundary check before loading tracking
        c, mask, _enc, _man, _prov = load_source(str(RAW), m)
        for _, r in S[S.match_id == int(m)].iterrows():
            rows = np.flatnonzero((c.canonical_period == r.period) & (np.abs(c.video_time_s - r.t_star) <= FIG_HALF_S))
            t = c.video_time_s[rows]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), dpi=110)
            sep = None
            for slot, col, mk, lab in ((int(r.slot_a), "#1f6fb4", "o", "A"), (int(r.slot_b), "#d4661f", "s", "B")):
                x, y = c.x[rows, slot], c.y[rows, slot]
                est = (mask[rows, slot] & Q.ESTIMATED) != 0
                ax[0].plot(x, y, "-", color=col, lw=1, alpha=0.7)
                ax[0].scatter(x[~est], y[~est], s=10, marker=mk, color=col, label=f"{lab} visible")
                ax[0].scatter(x[est], y[est], s=10, marker=mk, facecolors="none", edgecolors=col, label=f"{lab} estimated")
                k = np.argmin(np.abs(t - r.t_star))
                ax[0].annotate(lab, (x[k], y[k]), fontsize=8)
                sep = (x, y) if sep is None else (np.hypot(x - sep[0], y - sep[1]), None)
            ax[0].set_aspect("equal", adjustable="datalim")
            ax[0].set_xlabel("x (m)")
            ax[0].set_ylabel("y (m)")
            ax[0].legend(fontsize=7, loc="best", frameon=False)
            ax[1].plot(t - r.t_star, sep[0], color="#444444", lw=1.2)
            ax[1].axhline(r.radius_m, color="#888888", ls="--", lw=0.8)
            ax[1].axvspan(r.start - r.t_star, r.end - r.t_star, color="#bbbbbb", alpha=0.35, lw=0)
            ax[1].set_xlabel("time from t* (s)")
            ax[1].set_ylabel("separation (m)")
            fig.suptitle(f"{r.review_id} · {r.stratum} · match {m} p{int(r.period)} · R={r.radius_m} m · {r.status_composition} · cv {r.cv_tier}",
                         fontsize=8)
            fig.tight_layout()
            rel = f"figures/{r.review_id}.png"
            p = out.path(rel)
            fig.savefig(p)
            plt.close(fig)
            out.register(rel)
            done[r.review_id] = rel
    return done


def main(matches: list[str] | None = None, figures_on: bool = False, figures: bool | None = None) -> pd.DataFrame:
    figures_on = figures_on or bool(figures)
    ms = C.checked(matches)
    out = C.Out(OUT)
    cols = ["match_id", "period", "radius_m", "slot_a", "slot_b", "player_id_a", "player_id_b", "start", "end", "duration_s", "t_star", "d_min",
            "status_composition", "est_frac_a", "est_frac_b", "cv_margin_m", "cv_tier", "boundary_uncertain", "estimated_comotion",
            "reliability", "reliability_reasons", "label_risk", "label_risk_reasons", "team_relation", "edge_id", "n_sync_crossings_in_edge",
            "n_concurrent_other", "n_sequential_other", "group_d0p00", "group_d1p00", "max_accel_m_s2"]
    E = C.load_table("edges", ms, cols)
    G = C.load_table("groups", ms)
    clones = pd.read_parquet(C.PROC / "m11-phase2-quality-audit-v1" / "clone_like_pair_periods.parquet")
    clone_keys = set(zip(clones.match_id, clones.period, clones.slot_a, clones.slot_b))
    S = select(E, G, clone_keys)
    fig = _figures(S, out) if figures_on else {}
    S["figure"] = S.review_id.map(fig).fillna("")
    out.parquet("review_manifest.parquet", S)
    out.text("review_manifest.csv", S.to_csv(index=False))
    summ = dict(n_events=int(len(S)), by_stratum=S.stratum.value_counts().to_dict(), matches=int(S.match_id.nunique()),
                by_status=S.status_composition.value_counts().to_dict(), by_radius=S.radius_m.value_counts().to_dict(),
                multi_stratum_events=int((S.all_strata.str.count(";") > 0).sum()), figures=len(fig), seed=SEED,
                note="no human review has been performed; reviewer fields are empty")
    out.json("review_summary.json", summ)
    out.manifest(task="Phase 2 Task A4 stratified review sample", inputs=C.catalogue_inputs(ms) | dict(clone_pairs="m11-phase2-quality-audit-v1/clone_like_pair_periods.parquet"),
                 code=C.code_sha256(["review.py", "common.py"]), command="python -m football_fatigue.m11_phase2 review --figures")
    return S


_figures = figures
