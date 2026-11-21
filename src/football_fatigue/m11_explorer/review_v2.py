"""Human-review queue v2 and export (``m11-review-priority-v2``). A PROPOSAL: nothing here is a review.

* Tiers 1-5 are carried over unchanged from ``m11-review-priority-v1`` (same item ids, same order).
* M12 tier 6 (N x M events from the v1 prototype, which contained duplicate encounters and the timing shortcut) is replaced by:
  - tier 6 ``multi_player_hard``: v2 canonical encounters, N >= 3, ``async`` variant, both the best learned (L2 HGB) and the tuned
    physics (S2) solver wrong on the untouched test fold; one item per encounter;
  - tier 7 ``multi_player_null_new``: v2 ``null_new`` scenarios (N >= 3), L2 HGB wrong; one item per encounter.
  Inside a tier: at most ``PER_MATCH`` items per match, order by a hash of the item id.
* A deterministic double-review subset (``DOUBLE_PER_TIER`` per tier) for reviewer agreement.
* N x M items are exported BLINDED: ``nxm_items/<item>.json`` holds the scenario without truth (incoming identities, anonymous
  outgoing tracks); the provider answer key is in a separate ``nxm_key.json`` the reviewer must not open.

Every row is ``review_status = "unreviewed"``. Only a decision saved by a named reviewer counts as reviewed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

OUT = "m11-review-priority-v2"
V1_QUEUE = "m11-review-priority-v1"
PER_MATCH = 3
PER_TIER = {6: 30, 7: 15}
DOUBLE_PER_TIER = 5
TIERS = {1: "severe_contradiction", 2: "clone_like", 3: "estimated_dominated", 4: "position_jump", 5: "ambiguous_crossing",
         6: "multi_player_hard", 7: "multi_player_null_new"}
NXM_QUESTION = dict(
    per_incoming="Which outgoing track is the same player as this incoming track?",
    options=["<one outgoing id>", "none - the player did not reappear in the window", "unsure"],
    scenario=dict(positions_plausible=["unsure", "yes", "no"], provider_identity_continuous=["unsure", "yes", "no - swap suspected"],
                  visual_occlusion_checkable=["no (tracking data only)"]))


# ---------------------------------------------------------------- statistics for label-error estimation and agreement

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def n_required(max_upper: float, k_errors: int, alpha: float = 0.05) -> int:
    """Smallest n whose exact (Clopper-Pearson) upper bound with ``k_errors`` errors is <= ``max_upper``."""
    n = k_errors + 1
    while clopper_pearson(k_errors, n, alpha)[1] > max_upper:
        n += 1
    return n


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _h(s: str) -> int:
    return int(hashlib.sha256(f"{OUT}|{s}".encode()).hexdigest()[:12], 16)


def _pick(df: pd.DataFrame, n: int, key: str = "item_id") -> pd.DataFrame:
    df = df.assign(_h=df[key].astype(str).map(_h)).sort_values("_h")
    df = df.groupby("match_id", group_keys=False).head(PER_MATCH)
    return df.sort_values("_h").head(n).drop(columns="_h")


# ---------------------------------------------------------------- build

def build() -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    from . import nxm_assoc_v2 as A
    from . import nxm_dataset_v2 as N2
    q1 = pd.read_parquet(C.PROC / V1_QUEUE / "queue.parquet")
    keep = q1[q1["tier"] <= 5].copy()
    keep["source_queue"] = V1_QUEUE
    counts = {TIERS[t]: dict(selected=int((keep["tier"] == t).sum()), carried_from_v1=True) for t in range(1, 6)}
    counts["v1_tier6_dropped"] = dict(items=int((q1["tier"] == 6).sum()),
                                      reason="built on m11-nxm-exploratory-v1 (duplicate encounters, timing shortcut); replaced by tiers 6-7")
    per = pd.read_parquet(C.PROC / A.OUT / "v2" / "per_scenario.parquet")
    idx = pd.read_parquet(C.PROC / N2.OUT / "index.parquet")
    per = per.merge(idx[["scenario_id", "encounter_id", "variant", "N", "M", "n_null", "n_new", "period", "t_centre", "est_mean"]], on="scenario_id")
    pools = {6: per[(per["variant"] == "async") & (per["N"] >= 3) & (per["L2_hgb_exact"] == 0) & (per["S2_exact"] == 0)],
             7: per[(per["variant"] == "null_new") & (per["N"] >= 3) & (per["L2_hgb_exact"] == 0)]}
    rows = []
    for t, pool in pools.items():
        pool = pool.drop_duplicates("encounter_id").rename(columns={"scenario_id": "item_id"})
        pick = _pick(pool, PER_TIER[t])
        counts[TIERS[t]] = dict(eligible_encounters=int(len(pool)), selected=int(len(pick)))
        for r in pick.itertuples():
            rows.append(dict(tier=t, tier_name=TIERS[t], item_kind="nxm_exploratory_v2_scenario", item_id=r.item_id, match_id=int(r.match_id),
                             period=int(r.period), t=float(r.t_centre), review_status="unreviewed",
                             provenance="provider-derived (exploratory N x M v2)", source_queue=OUT, N=int(r.N), M=int(r.M)))
    q = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    q["double_review"] = False
    for t in sorted(q["tier"].unique()):
        ids = q.loc[q["tier"] == t, "item_id"].astype(str)
        dbl = sorted(ids, key=_h)[:DOUBLE_PER_TIER]
        q.loc[(q["tier"] == t) & q["item_id"].astype(str).isin(dbl), "double_review"] = True
    q["review_status"] = "unreviewed"
    C.checked(sorted({str(v) for v in q["match_id"]}))
    # blinded N x M items and the separate answer key
    want = set(q.loc[q["item_kind"] == "nxm_exploratory_v2_scenario", "item_id"])
    items, key = {}, {}
    for d in N2.load():
        if d["scenario_id"] in want:
            pub = {k: v for k, v in d.items() if k not in ("truth",)}
            pub["meta"] = {k: d["meta"][k] for k in ("match", "period", "t_centre", "variant")}
            pub["meta"]["blinded"] = "provider answer key removed; see nxm_key.json (coordinator only)"
            pub["review_question"] = NXM_QUESTION
            items[d["scenario_id"]] = pub
            key[d["scenario_id"]] = dict(truth=d["truth"], new_outgoing=[n["id"] for n in d["meta"]["new_outgoing"]])
    stats = dict(
        n_required_upper95_le_5pct={f"{k}_errors": n_required(0.05, k) for k in (0, 1, 2, 3)},
        n_required_upper95_le_10pct={f"{k}_errors": n_required(0.10, k) for k in (0, 1, 2)},
        examples_clopper_pearson={f"{k}/{n}": [round(v, 4) for v in clopper_pearson(k, n)] for k, n in ((0, 30), (1, 30), (0, 60), (2, 120), (5, 120))},
        agreement="Cohen's kappa on identity continuity over the double-review subset (target >= 0.6); also raw agreement and the "
                  "confusion table; disagreements adjudicated by a third reviewer, adjudication recorded separately")
    return q, counts, items, dict(stats=stats, key=key)


def existing_decisions() -> dict[str, Any]:
    from . import review as R
    d = R.decisions()
    return dict(decisions=int(len(d)), reviewers=sorted(d["reviewer"].unique().tolist()) if len(d) else [],
                store=str(R.OUT.relative_to(C.ROOT)) if hasattr(C, "ROOT") else str(R.OUT))


def main() -> int:
    from . import review as R
    out = C.Out(OUT)
    q, counts, items, extra = build()
    out.parquet("queue.parquet", q)
    out.text("queue.csv", q.to_csv(index=False))
    tmpl = q[["tier", "tier_name", "item_kind", "item_id", "match_id", "period", "t", "double_review"]].copy()
    tmpl.insert(0, "reviewer", "")
    for k in R.QUESTIONS:
        tmpl[k] = ""
    tmpl["nxm_links"] = ""
    tmpl["notes"] = ""
    out.text("review_template.csv", tmpl.to_csv(index=False))
    for sid, d in items.items():
        out.json(f"nxm_items/{sid}.json", d)
    out.json("nxm_key.json", dict(warning="COORDINATOR ONLY: provider-derived answer key, not ground truth; do not show to reviewers",
                                  key=extra["key"]))
    status = existing_decisions()
    plan = dict(schema=OUT, status="PROPOSAL: nothing reviewed", human_decisions_existing=status, tiers=TIERS, per_match=PER_MATCH,
                per_tier_new=PER_TIER, double_per_tier=DOUBLE_PER_TIER, counts=counts, total=int(len(q)), matches=int(q["match_id"].nunique()),
                double_review_items=int(q["double_review"].sum()), questions_two_player=R.QUESTIONS, question_nxm=NXM_QUESTION,
                statistics=extra["stats"])
    out.json("plan.json", plan)
    out.manifest(inputs=dict(v1_queue=f"data/processed/{V1_QUEUE}", association="data/processed/m11-nxm-association-v2",
                             dataset="data/processed/m11-nxm-exploratory-v2"),
                 parameters=dict(per_match=PER_MATCH, per_tier=PER_TIER, double_per_tier=DOUBLE_PER_TIER),
                 code={"m11_explorer/review_v2.py": C.sha256_file(__file__)})
    print(json.dumps(dict(counts=counts, total=len(q), double=int(q["double_review"].sum()), decisions=status), indent=1))
    return 0
