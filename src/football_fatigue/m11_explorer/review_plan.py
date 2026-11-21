"""Prioritised human-review queue PROPOSAL (``m11-review-priority-v1``). Nothing here is a review: every item is ``unreviewed`` and
only a named reviewer's saved decision (``m11_explorer.review.save_decision``) can change that.

Tiers, in priority order (an item appears once, in its highest tier; at most ``PER_MATCH`` items per match and tier, chosen by a
hash of the candidate id so the selection does not depend on any model's correctness except where the tier is defined by it):

1. severe motion contradictions (Phase 2 top 25: physics confidently against the provider label);
2. clone-like pairs (duplicated / co-moving traces);
3. ESTIMATED-dominated events (both players ESTIMATED in the gap and > 50 % of context);
4. position jumps near the event (jump-transfer risk);
5. ambiguous crossings (local crossing, the GRU near 0.5 or GRU and HGB disagree);
6. multi-player events (exploratory N x M prototype, N >= 3, every solver wrong on the complete variant).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

OUT = "m11-review-priority-v1"
PER_TIER = {1: 25, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30}
PER_MATCH = 3
TIERS = {1: "severe_contradiction", 2: "clone_like", 3: "estimated_dominated", 4: "position_jump", 5: "ambiguous_crossing", 6: "multi_player"}


def _h(s: str) -> int:
    return int(hashlib.sha256(f"m11-review-priority-v1|{s}".encode()).hexdigest()[:12], 16)


def _pick(df: pd.DataFrame, n: int, key: str = "candidate_id") -> pd.DataFrame:
    df = df.assign(_h=df[key].astype(str).map(_h)).sort_values("_h")
    df = df.groupby("match_id", group_keys=False).head(PER_MATCH)
    return df.head(n).drop(columns="_h")


def build() -> tuple[pd.DataFrame, dict]:
    from football_fatigue.m11_audit import validity as V

    from . import nxm_compare as NC
    from . import nxm_dataset as ND
    L = V.load()
    m = L["meta"].copy()
    ph = L["phase2"].set_index("candidate_id")
    m = m[m["independent"]].copy()
    m["clone_pair"] = ph.loc[m["candidate_id"], "clone_pair"].to_numpy()
    pg, phg = L["preds"]["gru_gaplast"]["p"], L["preds"]["hgb"]["p"]
    idx = m.index.to_numpy()
    m["p_gru"], m["p_hgb"] = pg[idx], phg[idx]
    est_ctx = (m["est_frac_used_in0"] + m["est_frac_used_in1"]) / 2
    sev = pd.read_parquet(C.PROC / "m11-phase2-ambiguity-benchmark-v1" / "severe_failures_top25.parquet")
    pools = {
        1: sev.assign(match_id=sev["match_id"].astype(int)),
        2: m[m["clone_pair"]],
        3: m[(m["status_in_gap"] == "both_fully_estimated") & (est_ctx > 0.5)],
        4: m[m["jump_transfer_risk"].astype(bool)],
        5: m[(m["crossing_class"] == "local_crossing") & ((np.abs(m["p_gru"] - 0.5) < 0.15) | ((m["p_gru"] > 0.5) != (m["p_hgb"] > 0.5)))],
    }
    rows, seen, counts = [], set(), {}
    for tier in range(1, 6):
        pool = pools[tier]
        pool = pool[~pool["candidate_id"].isin(seen)]
        counts[TIERS[tier]] = dict(eligible=int(len(pool)))
        pick = pool if tier == 1 else _pick(pool, PER_TIER[tier])
        for r in pick.itertuples():
            seen.add(r.candidate_id)
            rows.append(dict(tier=tier, tier_name=TIERS[tier], item_kind="two_player_candidate", item_id=r.candidate_id, match_id=int(r.match_id),
                             period=int(r.period), t=float(r.t_star), review_status="unreviewed", provenance="provider-derived"))
    per = pd.read_parquet(C.PROC / NC.OUT / "per_scenario.parquet")
    idx6 = pd.read_parquet(C.PROC / ND.OUT / "index.parquet")[["scenario_id", "event_id", "variant", "N", "period", "t_centre"]]
    per = per.merge(idx6, on="scenario_id")
    ex_cols = [c for c in per.columns if c.endswith("_exact")]
    hard = per[(per["variant"] == "complete") & (per["N"] >= 3) & (per[ex_cols].sum(axis=1) == 0)]
    counts[TIERS[6]] = dict(eligible=int(len(hard)))
    for r in _pick(hard.rename(columns={"scenario_id": "candidate_id"}), PER_TIER[6]).itertuples():
        rows.append(dict(tier=6, tier_name=TIERS[6], item_kind="nxm_exploratory_event", item_id=r.candidate_id, match_id=int(r.match_id),
                         period=int(r.period), t=float(r.t_centre), review_status="unreviewed", provenance="provider-derived (exploratory N x M)"))
    q = pd.DataFrame(rows)
    for t, n in q["tier_name"].value_counts().items():
        counts[t]["selected"] = int(n)
    return q, counts


def main() -> int:
    out = C.Out(OUT)
    q, counts = build()
    out.parquet("queue.parquet", q)
    out.json("plan.json", dict(schema=OUT, status="PROPOSAL: nothing reviewed", tiers=TIERS, per_tier=PER_TIER, per_match=PER_MATCH, counts=counts,
                               total=int(len(q)), matches=int(q["match_id"].nunique())))
    out.manifest(inputs={}, parameters=dict(per_tier=PER_TIER, per_match=PER_MATCH), code={"m11_explorer/review_plan.py": C.sha256_file(__file__)})
    print(json.dumps(counts, indent=1), len(q))
    return 0
