"""EXPLORATORY multi-player (N x M) association dataset, corrected version (``m11-nxm-exploratory-v2``).

NOT frozen, NOT validated, NOT dataset v2 of the project. Truth = provider identities (``provider-derived``; nothing is
human-reviewed). Hidden intervals are synthetic observation gaps cut into provider tracks; proximity is geometry, not occlusion.
v1 (``nxm_dataset.py`` / ``m11-nxm-exploratory-v1``) is left untouched; this module only reuses its helpers.

Corrections relative to v1 (root causes in ``docs/research/m13_nxm_dataset_quality_v1.md``):

1. **Repeated interaction.** v1 used ``max(order_in_chain) > 0``; ``order_in_chain`` is a 1-based rank, so the flag was always
   true. v2: ``repeated_encounter`` is true when a member pair had a *distinct* earlier edge chain (separated by more than the
   catalogue chain-merge gap, 1.0 s) that ended within ``REPEAT_LOOKBACK_S`` before the encounter started and is not part of the
   same encounter.
2. **Duplicate catalogue rows.** One continuous encounter is fragmented into several link-0 groups (one edge chain can span more
   than 20 groups). v2 forms *canonical encounters*: link-0 groups are united when they share an edge chain, and candidate events of
   one period are united when their scenario windows overlap and they share >= 2 members. Exactly one representative (largest N,
   then a seeded hash) is used per encounter; the number of merged catalogue rows is recorded.
3. **Timing shortcut.** v1 drew one gap length per player and a common centre offset, so the true link had ``t_first - t_end``
   equal to that player's gap while wrong links had a wider spread (a label-free timing oracle reached 0.45 exact). v2 draws the
   disappearance and reappearance times **independently** (``t_end = tc - U(0.3, 0.9)``, ``t_first = tc + U(0.3, 0.9)``), so the
   time difference has the same distribution for true and wrong pairs; distractors use the same reappearance distribution.
4. **Null coverage.** v1 removed at most one outgoing track and almost always added a distractor, so N > M with N >= 4 had 7
   scenarios. v2 variants remove 1 or 2 outgoing tracks with or without 1 or 2 new outgoing tracks, with the number of nulls drawn
   independently of the number of new tracks (so M - N does not determine the null count when M >= N).
5. **Opaque outgoing ids.** Outgoing ids are seeded hashes (no number sequence that could reveal a missing track).

Variants per encounter are augmentations of ONE real event (never independent observations; every CI is match-level):

* ``complete``  N x N, synchronous 0.8 s gap, no dropout (continuity with v1);
* ``async``     independent disappearance / reappearance times, 20 % context dropout;
* ``null1``     ``async`` with one member's outgoing removed: N x (N - 1);
* ``null2``     ``async`` with two removed (N >= 3): N x (N - 2);
* ``new1``      ``async`` + the nearest eligible non-member's outgoing: N x (N + 1);
* ``new2``      ``async`` + the two nearest: N x (N + 2) (when two are eligible);
* ``null_new``  ``async`` with k in {1, 2 (N >= 3)} removed and d in {1, 2} new, k and d drawn independently.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

from . import multiplayer as MP
from . import nxm_dataset as V1

OUT = "m11-nxm-exploratory-v2"
RADIUS = V1.RADIUS
CONTEXT_S, HZ, MAX_N = V1.CONTEXT_S, V1.HZ, V1.MAX_N
HALF = (0.3, 0.9)                      # disappearance / reappearance offsets from the centre (independent draws)
GAP_HALF_MAX = HALF[1]
COMPLETE_HALF = 0.4
DROP_RATE, DISTRACTOR_M = 0.2, 6.0
CHAIN_MERGE_GAP_S = 1.0                # catalogue chain_merge_gap_s: same-pair contacts closer than this are one edge chain
REPEAT_LOOKBACK_S = 60.0
PER_MATCH_CAP = {2: 25}                # encounters per match for N = 2; N >= 3: all eligible encounters
WINDOW_S = 2 * (GAP_HALF_MAX + CONTEXT_S)
VARIANTS = ("complete", "async", "null1", "null2", "new1", "new2", "null_new")


def _seed(*parts) -> int:
    return V1._seed("v2", *parts)


def _oid(*parts) -> str:
    return "out-" + hashlib.sha256("|".join(map(str, ("v2-out",) + parts)).encode()).hexdigest()[:8]


# ---------------------------------------------------------------- canonical encounters and repeated contact

class _UF:
    def __init__(self):
        self.p: dict[Any, Any] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def canonical_encounters(cands: pd.DataFrame, edges: pd.DataFrame, groups: pd.DataFrame, window_s: float = WINDOW_S) -> pd.DataFrame:
    """Assign every candidate event (one per link-0 group) to a canonical encounter.

    Groups are united when they share an edge chain (one continuous same-pair contact fragmented into several groups); candidate
    events of the same period are united when their scenario windows overlap (|dt| < ``window_s``) and they share >= 2 members.
    Adds ``encounter_id``, ``n_catalogue_groups`` (link-0 groups merged), ``n_candidates``, ``encounter_start`` and
    ``encounter_chains`` (edge chain ids of the encounter)."""
    uf = _UF()
    for g in groups["group_id"]:
        uf.find(g)
    for _cid, gs in edges.groupby("edge_chain_id")["group_d0p00"]:
        gs = list(dict.fromkeys(gs))
        for g in gs[1:]:
            uf.union(gs[0], g)
    c = cands.sort_values(["period", "t_centre"]).reset_index(drop=True)
    mem = [frozenset(m) for m in c["members"]]
    for per, idx in c.groupby("period").indices.items():
        idx = list(idx)
        for a in range(len(idx)):
            i = idx[a]
            for b in range(a + 1, len(idx)):
                j = idx[b]
                if c.at[j, "t_centre"] - c.at[i, "t_centre"] >= window_s:
                    break
                if len(mem[i] & mem[j]) >= 2:
                    uf.union(c.at[i, "group_id"], c.at[j, "group_id"])
    groups = groups.assign(_enc=groups["group_id"].map(uf.find))
    c["encounter_id"] = c["group_id"].map(uf.find)
    size = groups.groupby("_enc").size()
    start = groups.groupby("_enc")["start"].min()
    ech = edges.assign(_enc=edges["group_d0p00"].map(uf.find)).groupby("_enc")["edge_chain_id"].agg(lambda s: frozenset(s))
    c["n_catalogue_groups"] = c["encounter_id"].map(size).fillna(1).astype(int)
    c["n_candidates"] = c.groupby("encounter_id")["group_id"].transform("size")
    c["encounter_start"] = c["encounter_id"].map(start).astype(float)
    c["encounter_chains"] = c["encounter_id"].map(ech).apply(lambda v: v if isinstance(v, frozenset) else frozenset())
    return c


def repeat_info(row: dict[str, Any], chains: pd.DataFrame, lookback_s: float = REPEAT_LOOKBACK_S,
                min_sep_s: float = CHAIN_MERGE_GAP_S) -> dict[str, Any]:
    """Genuine repeated contact: some member pair has an edge chain that is NOT part of this encounter and ended between
    ``lookback_s`` and ``min_sep_s`` seconds before the encounter started. Fragments of the same continuous contact (same edge chain
    or same encounter) never count."""
    mem = sorted(int(s) for s in row["members"])
    ref = float(row["encounter_start"])
    ch = chains[(chains["period"] == row["period"]) & ~chains["edge_chain_id"].isin(row["encounter_chains"])]
    lo, hi = np.minimum(ch["slot_a"], ch["slot_b"]), np.maximum(ch["slot_a"], ch["slot_b"])
    ms = set(mem)
    pair_ok = lo.isin(ms) & hi.isin(ms)
    prior = ch[pair_ok & (ch["end"] <= ref - min_sep_s) & (ch["end"] >= ref - lookback_s)]
    gaps = ref - prior["end"]
    return dict(repeated_encounter=bool(len(prior) > 0), n_prior_encounters=int(len(prior)),
                prior_gap_s=round(float(gaps.min()), 3) if len(prior) else None,
                n_pairs_repeated=int(len({(int(a), int(b)) for a, b in zip(lo[prior.index], hi[prior.index])})))


def encounters(match: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Candidate events of a match with canonical encounter ids, plus the match's R = 1.0 m edge chains (None when no candidates)."""
    cands = V1.events(match)
    if cands.empty:
        return None
    cands = cands.drop(columns=["repeated_interaction"])
    E = C.load_table("edges", [match], ["period", "radius_m", "edge_chain_id", "group_d0p00", "slot_a", "slot_b"])
    E = E[np.isclose(E["radius_m"], RADIUS)]
    G = C.load_table("groups", [match])
    G = G[(G["link_s"] == 0) & np.isclose(G["radius_m"], RADIUS)]
    CH = C.load_table("edge_chains", [match])
    CH = CH[np.isclose(CH["radius_m"], RADIUS)]
    c = canonical_encounters(cands, E, G)
    c["_h"] = [_seed(match, g) for g in c["group_id"]]
    return c, CH


def select_representatives(c: pd.DataFrame, eligible: np.ndarray) -> pd.DataFrame:
    """Largest eligible N per encounter, ties broken by a seeded hash (never by any model or label)."""
    e = c[eligible].sort_values(["encounter_id", "n", "_h"], ascending=[True, False, True])
    return e.drop_duplicates("encounter_id").reset_index(drop=True)


# ---------------------------------------------------------------- scenarios

def _track(M, slot: int, per: int, t_cut: float, incoming: bool):
    if incoming:
        p = M.path(int(slot), per, t_cut - CONTEXT_S - 0.2, t_cut)
        return V1._resample(p[p["t"] < t_cut], np.arange(t_cut - CONTEXT_S, t_cut, 1 / HZ))
    p = M.path(int(slot), per, t_cut, t_cut + CONTEXT_S + 0.2)
    return V1._resample(p[p["t"] > t_cut], t_cut + np.arange(1, int(CONTEXT_S * HZ) + 1) / HZ)


def scenario(M, spans, ev: dict[str, Any], variant: str, positions_at, pid: dict[int, int] | None = None) -> dict[str, Any] | None:
    """One scenario of a canonical encounter (deterministic for the encounter and variant). Returns None when not constructible."""
    rng = np.random.default_rng(_seed(ev["match_id"], ev["encounter_id"], variant))
    mem, per, tc = [int(s) for s in ev["members"]], int(ev["period"]), float(ev["t_centre"])
    N = len(mem)
    a, b = tc - GAP_HALF_MAX - CONTEXT_S, tc + GAP_HALF_MAX + CONTEXT_S
    if not all(V1.covered(spans, s, per, a, b) for s in mem):
        return None
    sync = variant == "complete"
    k_null = {"null1": 1, "null2": 2}.get(variant, 0)
    d_new = {"new1": 1, "new2": 2}.get(variant, 0)
    if variant == "null_new":
        k_null = int(rng.integers(1, min(2, N - 1) + 1))
        d_new = int(rng.integers(1, 3))
    if k_null > N - 1:
        return None
    incoming, outgoing, truth, hidden = [], [], {}, {}
    for k, s in enumerate(mem):
        h0 = tc - (COMPLETE_HALF if sync else float(rng.uniform(*HALF)))
        h1 = tc + (COMPLETE_HALF if sync else float(rng.uniform(*HALF)))
        ia, ob = _track(M, s, per, h0, True), _track(M, s, per, h1, False)
        if ia is None or ob is None:
            return None
        if not sync:
            ia, ob = V1._drop(ia, DROP_RATE, rng, True), V1._drop(ob, DROP_RATE, rng, False)
        oid = _oid(ev["match_id"], ev["encounter_id"], variant, "m", s)
        incoming.append(dict(id=f"in-{k}", label=f"slot-{s}", **{q: ia[q] for q in ("t", "x", "y")}))
        outgoing.append(dict(id=oid, **{q: ob[q] for q in ("t", "x", "y")}))
        truth[f"in-{k}"] = oid
        hidden[f"in-{k}"] = dict(h0=round(h0, 3), h1=round(h1, 3), est_in=ia["est"], est_out=ob["est"], slot=s,
                                 player_id=(pid or {}).get(s))
    nulls = sorted(int(v) for v in rng.choice(N, size=k_null, replace=False)) if k_null else []
    for k in nulls:
        gone = truth[f"in-{k}"]
        outgoing = [o for o in outgoing if o["id"] != gone]
        truth[f"in-{k}"] = None
    new = []
    if d_new:
        others = positions_at(per, tc)
        others = others[~others["slot"].isin(mem)]
        cen = np.mean([[t["x"][-1], t["y"][-1]] for t in incoming], axis=0)
        dist = np.hypot(others["x"].to_numpy() - cen[0], others["y"].to_numpy() - cen[1])
        for j in np.argsort(dist, kind="stable"):
            if dist[j] > DISTRACTOR_M or len(new) == d_new:
                break
            s = int(others["slot"].iloc[j])
            if not V1.covered(spans, s, per, a, b):
                continue
            h1 = tc + float(rng.uniform(*HALF))                 # same reappearance distribution as members
            ob = _track(M, s, per, h1, False)
            if ob is None:
                continue
            ob = V1._drop(ob, DROP_RATE, rng, False)
            oid = _oid(ev["match_id"], ev["encounter_id"], variant, "d", s)
            outgoing.append(dict(id=oid, **{q: ob[q] for q in ("t", "x", "y")}))
            new.append(dict(id=oid, slot=s, player_id=(pid or {}).get(s), distance_m=round(float(dist[j]), 3), h1=round(h1, 3),
                            est_out=ob["est"]))
        if len(new) < d_new and variant in ("new1", "new2"):
            return None
        if not new and variant == "null_new":
            return None
    outgoing.sort(key=lambda o: o["id"])
    sid = f"{ev['match_id']}-{ev['encounter_id']}-{variant}"
    d = dict(schema=MP.SCHEMA, exploratory=True, synthetic_example=False, provenance="provider-derived", human_reviewed=False,
             dataset=OUT, scenario_id=sid, event_id=f"{ev['match_id']}-{ev['encounter_id']}", incoming=incoming, outgoing=outgoing,
             truth=truth,
             meta=dict(match=int(ev["match_id"]), period=per, t_centre=round(tc, 3), variant=variant, members=mem, hidden=hidden,
                       new_outgoing=new, null_incoming=[f"in-{k}" for k in nulls], radius_m=RADIUS, group_id=ev["group_id"],
                       encounter_id=ev["encounter_id"], n_catalogue_groups=int(ev["n_catalogue_groups"]),
                       n_candidates=int(ev["n_candidates"]),
                       **{k: ev[k] for k in ("max_simultaneous", "group_players", "n_sync_crossings", "frac_edges_both_estimated",
                                             "estimated_comotion", "label_risk", "repeated_encounter", "n_prior_encounters",
                                             "prior_gap_s")}))
    if MP.validate(d):
        raise RuntimeError(f"{sid}: invalid scenario {MP.validate(d)}")
    return d


def build_match(match: str) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    from . import tracks as TR
    m = C.checked([match])[0]
    M = TR.load(m)
    spans = V1.chain_spans(M)
    r = encounters(m)
    if r is None:
        return [], dict(candidates=0), pd.DataFrame()
    c, CH = r
    pid = dict(zip(M.slots()["slot"].astype(int), M.slots()["player_id"].astype(np.int64).tolist()))
    a = c["t_centre"] - GAP_HALF_MAX - CONTEXT_S
    b = c["t_centre"] + GAP_HALF_MAX + CONTEXT_S
    elig = np.array([all(V1.covered(spans, int(s), int(p), x, y) for s in mem) for mem, p, x, y in zip(c["members"], c["period"], a, b)])
    rep = select_representatives(c, elig)
    rep = pd.concat([rep, pd.DataFrame([repeat_info(r_, CH) for r_ in rep.to_dict("records")])], axis=1)
    stats: dict[str, Any] = dict(candidates=int(len(c)), candidate_encounters=int(c["encounter_id"].nunique()),
                                 candidates_merged_as_duplicates=int(len(c) - c["encounter_id"].nunique()),
                                 eligible_encounters=int(len(rep)), by_n_eligible=rep["n"].value_counts().sort_index().to_dict())
    cache: dict[tuple[int, float], pd.DataFrame] = {}

    def positions_at(per, t):
        k = (per, round(t, 2))
        if k not in cache:
            f = M.frames(per, t - 0.05, t + 0.05, 15)
            cache[k] = f.groupby("slot")[["x", "y"]].mean().reset_index()
        return cache[k]
    out, rejected, used = [], {}, []
    for n, grp in rep.groupby("n"):
        cap = PER_MATCH_CAP.get(int(n))
        grp = grp.sort_values("_h")
        if cap:
            grp = grp.head(cap)
        for ev in grp.to_dict("records"):
            made = 0
            for v in VARIANTS:
                d = scenario(M, spans, ev, v, positions_at, pid)
                if d is None:
                    rejected[v] = rejected.get(v, 0) + 1
                    continue
                out.append(d)
                made += 1
            if made:
                used.append(dict(match_id=int(m), encounter_id=ev["encounter_id"], period=int(ev["period"]), t_centre=float(ev["t_centre"]),
                                 n=int(ev["n"]), members=list(map(int, ev["members"])), n_catalogue_groups=int(ev["n_catalogue_groups"]),
                                 n_candidates=int(ev["n_candidates"]), repeated_encounter=bool(ev["repeated_encounter"]),
                                 n_prior_encounters=int(ev["n_prior_encounters"]), prior_gap_s=ev["prior_gap_s"], scenarios=made))
    stats.update(scenarios=len(out), encounters_used=len(used), variants_not_constructible=rejected)
    TR.load.cache_clear()
    return out, stats, pd.DataFrame(used)


def index_row(d: dict[str, Any]) -> dict[str, Any]:
    m = d["meta"]
    hid = list(m["hidden"].values())
    est = [h["est_in"] for h in hid] + [h["est_out"] for h in hid] + [n["est_out"] for n in m["new_outgoing"]]
    gaps = [h["h1"] - h["h0"] for h in hid]
    dists = [n["distance_m"] for n in m["new_outgoing"]]
    return dict(scenario_id=d["scenario_id"], event_id=d["event_id"], encounter_id=m["encounter_id"], match_id=m["match"], period=m["period"],
                t_centre=m["t_centre"], variant=m["variant"], N=len(d["incoming"]), M=len(d["outgoing"]),
                n_null=sum(v is None for v in d["truth"].values()), n_new=len(m["new_outgoing"]), distractor_min_m=min(dists) if dists else None,
                gap_mean_s=float(np.mean(gaps)), gap_max_s=float(np.max(gaps)), est_mean=float(np.mean(est)), est_max=float(np.max(est)),
                provenance=d["provenance"], human_reviewed=d["human_reviewed"],
                max_speed_m_s=max(V1.max_speed(tr) for tr in d["incoming"] + d["outgoing"]),
                **{k: m[k] for k in ("max_simultaneous", "group_players", "n_sync_crossings", "frac_edges_both_estimated", "estimated_comotion",
                                     "label_risk", "repeated_encounter", "n_prior_encounters", "prior_gap_s", "n_catalogue_groups",
                                     "n_candidates")})


def main() -> int:
    out = C.Out(OUT)
    t0 = time.perf_counter()
    per, rows, encs = {}, [], []
    path = out.path("scenarios.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for m in C.allowed():
            t1 = time.perf_counter()
            sc, st, used = build_match(m)
            for d in sc:
                fh.write(json.dumps(d) + "\n")
                rows.append(index_row(d))
            encs.append(used)
            st["seconds"] = round(time.perf_counter() - t1, 1)
            per[m] = st
            print(m, {k: st[k] for k in ("candidates", "eligible_encounters", "encounters_used", "scenarios", "seconds")}, flush=True)
    out.files["scenarios.jsonl.gz"] = C.sha256_file(path)
    idx = pd.DataFrame(rows)
    out.parquet("index.parquet", idx)
    enc = pd.concat(encs, ignore_index=True)
    out.parquet("encounters.parquet", enc.assign(members=enc["members"].map(lambda v: ";".join(map(str, v)))))
    out.json("build.json", dict(schema=OUT, radius_m=RADIUS, variants=VARIANTS, per_match=per, total=len(idx), encounters=int(len(enc)),
                                runtime_s=round(time.perf_counter() - t0, 1),
                                labels="provider-derived (not human-reviewed); exploratory; not frozen; variants are augmentations"))
    here = Path(__file__).resolve().parent
    out.manifest(inputs=C.catalogue_inputs(C.allowed()),
                 parameters=dict(radius_m=RADIUS, context_s=CONTEXT_S, hz=HZ, max_n=MAX_N, caps=PER_MATCH_CAP, half=HALF,
                                 complete_half=COMPLETE_HALF, drop_rate=DROP_RATE, distractor_m=DISTRACTOR_M,
                                 repeat_lookback_s=REPEAT_LOOKBACK_S, chain_merge_gap_s=CHAIN_MERGE_GAP_S, window_s=WINDOW_S),
                 code={p: C.sha256_file(here / p) for p in ("nxm_dataset_v2.py", "nxm_dataset.py", "multiplayer.py")})
    return 0


def load(path=None) -> list[dict[str, Any]]:
    p = path or C.PROC / OUT / "scenarios.jsonl.gz"
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh]
    C.checked(sorted({str(d["meta"]["match"]) for d in rows}))
    return rows
