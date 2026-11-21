"""EXPLORATORY multi-player (N x M) association prototype from real development tracking (``m11-nxm-exploratory-v1``).

NOT frozen, NOT validated, NOT dataset v2. Truth = provider identities (``provider-derived``; no example is human-reviewed).
Hidden intervals are synthetic observation gaps cut into provider tracks; proximity is geometry, not verified occlusion.

Source events: catalogue link-0 (simultaneous) groups at R = 1.0 m in the 32 train/CV matches. Each event is centred in the group
state with the most players in one connected component (all members close at that instant); members = that component (2 to 5
players; larger components keep the 5 players with the most proximity edges).

Eligibility (as Stage 3): every member has ONE continuous provider chain covering [centre - gap_max/2 - context, centre + gap_max/2 +
context] (no chain break, so provider identity is continuous); contexts use only samples strictly outside each player's hidden
interval, resampled to 15 Hz inside the observed span (no extrapolation).

Variants per event (seeded by event id; never pooled as independent observations):

* ``complete``  N x N, synchronous 0.8 s gap;
* ``async_var`` per-player gap length 0.6-1.6 s with +-25 % jitter and +-0.4 s centre offsets, 20 % random context dropout;
* ``unequal``   as ``async_var`` plus one member's outgoing track removed (unmatched incoming -> null truth) and, when an eligible
  non-member is within 6 m, that player's outgoing track added (new outgoing / distractor);
* ``extra_out`` as ``async_var`` plus the distractor's outgoing track only (N x (N + 1): a new player appears, nobody leaves).

Per-player gap jitter matters: with identical gap lengths, ``t_start(out) - t_end(in) = gap`` would identify asynchronous links
trivially (timing shortcut; measured in the quality report).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

from . import multiplayer as MP

OUT = "m11-nxm-exploratory-v1"
RADIUS = 1.0
CONTEXT_S = 2.0
HZ = 15.0
MAX_N = 5
PER_MATCH_CAP = {2: 25, 3: 40, 4: 60, 5: 60}          # events per match and size (sizes 4-5 are rare: effectively all)
VARIANTS = ("complete", "async_var", "unequal", "extra_out")
GAP_RANGE, JITTER, ASYNC_S, DROP_RATE, DISTRACTOR_M = (0.6, 1.6), 0.25, 0.4, 0.2, 6.0
GAP_MAX = GAP_RANGE[1] * (1 + JITTER) + 2 * ASYNC_S


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % 2**63


def chain_spans(M) -> pd.DataFrame:
    s = M.seg
    df = pd.DataFrame(dict(slot=s["slot"], period=s["period"], chain=s["chain"], t0=s["t0"], t1=s["t1"]))
    return df.groupby(["slot", "period", "chain"]).agg(start=("t0", "min"), end=("t1", "max")).reset_index()


def covered(spans: pd.DataFrame, slot: int, period: int, a: float, b: float) -> bool:
    x = spans[(spans["slot"] == slot) & (spans["period"] == period)]
    return bool(((x["start"] <= a) & (x["end"] >= b)).any())


def _components(s: str) -> list[list[int]]:
    return [EV_players(c) for c in str(s).split("|") if c.strip()]


def EV_players(c: str) -> list[int]:
    return [int(v) for v in c.replace(",", ";").split(";") if v.strip()]


def events(match: str) -> pd.DataFrame:
    """One row per link-0 group at RADIUS: centre time and members of its largest simultaneous component."""
    g = C.load_table("groups", [match])
    g = g[(g["link_s"] == 0) & np.isclose(g["radius_m"], RADIUS)]
    tl = C.load_table("group_timeline", [match])
    tl = tl[np.isclose(tl["radius_m"], RADIUS) & tl["group_id"].isin(g["group_id"])].sort_values(["group_id", "t"])
    E = C.load_table("edges", [match])
    E = E[np.isclose(E["radius_m"], RADIUS)]
    deg = pd.concat([E[["group_d0p00", "slot_a"]].rename(columns={"slot_a": "slot"}), E[["group_d0p00", "slot_b"]].rename(columns={"slot_b": "slot"})])
    deg = deg.groupby(["group_d0p00", "slot"]).size()
    rep = E.groupby("group_d0p00")["order_in_chain"].max()
    rows = []
    in_tl = set(tl["group_id"])
    pairs = g[~g["group_id"].isin(in_tl) & (g["n_players"] == 2)]
    tstar = E.sort_values("d_min").drop_duplicates("group_d0p00").set_index("group_d0p00")["t_star"]
    for r in pairs.itertuples():                       # two-player groups have no timeline: centre on the edge's closest approach
        if r.group_id not in tstar.index:
            continue
        rows.append(dict(match_id=int(match), period=int(r.period), group_id=r.group_id, t_centre=float(tstar[r.group_id]),
                         members=sorted(EV_players(str(r.players))), n=2, max_simultaneous=2, group_players=2,
                         n_sync_crossings=int(r.n_sync_crossings), frac_edges_both_estimated=float(r.frac_edges_both_estimated),
                         estimated_comotion=bool(r.any_estimated_comotion), label_risk=bool(r.any_label_risk),
                         repeated_interaction=bool(rep.get(r.group_id, 0) > 0)))
    for gid, st in tl.groupby("group_id", sort=False):
        best = None
        ts = st["t"].to_numpy()
        for k, (comp, t) in enumerate(zip(st["components"], ts)):
            for c in _components(comp):
                if best is None or len(c) > len(best[0]):
                    t_next = ts[k + 1] if k + 1 < len(ts) else t + 0.2
                    best = (c, 0.5 * (t + min(t_next, t + 1.0)))
        if best is None or len(best[0]) < 2:
            continue
        mem = best[0]
        if len(mem) > MAX_N:
            mem = sorted(mem, key=lambda s: -deg.get((gid, s), 0))[:MAX_N]
        r = g[g["group_id"] == gid].iloc[0]
        rows.append(dict(match_id=int(match), period=int(r["period"]), group_id=gid, t_centre=float(best[1]), members=sorted(mem),
                         n=len(mem), max_simultaneous=int(r["max_simultaneous_players"]), group_players=int(r["n_players"]),
                         n_sync_crossings=int(r["n_sync_crossings"]), frac_edges_both_estimated=float(r["frac_edges_both_estimated"]),
                         estimated_comotion=bool(r["any_estimated_comotion"]), label_risk=bool(r["any_label_risk"]),
                         repeated_interaction=bool(rep.get(gid, 0) > 0)))
    return pd.DataFrame(rows)


def _resample(p: pd.DataFrame, grid: np.ndarray) -> dict[str, list[float]] | None:
    if len(p) < 2:
        return None
    ok = (grid >= p["t"].iloc[0]) & (grid <= p["t"].iloc[-1])
    if ok.sum() < 3:
        return None
    t = grid[ok]
    est = np.interp(t, p["t"], p["estimated"].astype(float))
    return dict(t=np.round(t, 4).tolist(), x=np.round(np.interp(t, p["t"], p["x"]), 4).tolist(), y=np.round(np.interp(t, p["t"], p["y"]), 4).tolist(),
                est=float(est.mean()))


def _drop(tr: dict[str, Any], rate: float, rng, keep_end: bool) -> dict[str, Any]:
    n = len(tr["t"])
    keep = rng.random(n) >= rate
    keep[-1 if keep_end else 0] = True
    return {k: (list(np.asarray(v)[keep]) if isinstance(v, list) else v) for k, v in tr.items()}


def scenario(M, spans, ev: dict[str, Any], variant: str, positions_at) -> dict[str, Any] | None:
    rng = np.random.default_rng(_seed(ev["match_id"], ev["group_id"], variant))
    mem, per, tc = list(ev["members"]), ev["period"], ev["t_centre"]
    a, b = tc - GAP_MAX / 2 - CONTEXT_S, tc + GAP_MAX / 2 + CONTEXT_S
    if not all(covered(spans, s, per, a, b) for s in mem):
        return None
    base_gap = 0.8 if variant == "complete" else float(rng.uniform(*GAP_RANGE))
    incoming, outgoing, truth, hidden = [], [], {}, {}
    perm = rng.permutation(10)
    for k, s in enumerate(mem):
        g = base_gap if variant == "complete" else base_gap * float(rng.uniform(1 - JITTER, 1 + JITTER))
        c0 = tc if variant == "complete" else tc + float(rng.uniform(-ASYNC_S, ASYNC_S))
        h0, h1 = c0 - g / 2, c0 + g / 2
        pa, pb = M.path(int(s), per, h0 - CONTEXT_S - 0.2, h0), M.path(int(s), per, h1, h1 + CONTEXT_S + 0.2)
        ia = _resample(pa[pa["t"] < h0], np.arange(h0 - CONTEXT_S, h0, 1 / HZ))
        ob = _resample(pb[pb["t"] > h1], h1 + np.arange(1, int(CONTEXT_S * HZ) + 1) / HZ)
        if ia is None or ob is None:
            return None
        if variant != "complete":
            ia, ob = _drop(ia, DROP_RATE, rng, True), _drop(ob, DROP_RATE, rng, False)
        oid = f"out-{perm[k]}"
        incoming.append(dict(id=f"in-{k}", label=f"slot-{s}", **{q: ia[q] for q in ("t", "x", "y")}))
        outgoing.append(dict(id=oid, **{q: ob[q] for q in ("t", "x", "y")}))
        truth[f"in-{k}"] = oid
        hidden[f"in-{k}"] = dict(h0=round(h0, 3), h1=round(h1, 3), est_in=ia["est"], est_out=ob["est"], slot=int(s))
    distractor = None
    if variant in ("unequal", "extra_out"):
        if variant == "unequal":
            drop_k = int(rng.integers(0, len(mem)))
            gone = truth[f"in-{drop_k}"]
            outgoing = [o for o in outgoing if o["id"] != gone]
            truth[f"in-{drop_k}"] = None
        others = positions_at(per, tc)
        others = others[~others["slot"].isin(mem)]
        cx, cy = others["x"].to_numpy(), others["y"].to_numpy()
        cen = np.mean([[incoming[k]["x"][-1], incoming[k]["y"][-1]] for k in range(len(mem))], axis=0)
        dist = np.hypot(cx - cen[0], cy - cen[1])
        for j in np.argsort(dist):
            if dist[j] > DISTRACTOR_M:
                break
            s = int(others["slot"].iloc[j])
            if not covered(spans, s, per, a, b):
                continue
            h1 = tc + base_gap / 2 + float(rng.uniform(-ASYNC_S, ASYNC_S))
            pb = M.path(s, per, h1, h1 + CONTEXT_S + 0.2)
            ob = _resample(pb[pb["t"] > h1], h1 + np.arange(1, int(CONTEXT_S * HZ) + 1) / HZ)
            if ob is None:
                continue
            ob = _drop(ob, DROP_RATE, rng, False)
            outgoing.append(dict(id=f"out-{perm[len(mem)]}", **{q: ob[q] for q in ("t", "x", "y")}))
            distractor = dict(slot=s, distance_m=round(float(dist[j]), 3), est_out=ob["est"])
            break
        if variant == "extra_out" and distractor is None:
            return None
    outgoing.sort(key=lambda o: o["id"])
    sid = f"{ev['match_id']}-{ev['group_id']}-{variant}"
    d = dict(schema=MP.SCHEMA, exploratory=True, synthetic_example=False, provenance="provider-derived", human_reviewed=False,
             scenario_id=sid, event_id=f"{ev['match_id']}-{ev['group_id']}", incoming=incoming, outgoing=outgoing, truth=truth,
             meta=dict(match=int(ev["match_id"]), period=per, t_centre=round(tc, 3), variant=variant, base_gap_s=round(base_gap, 3),
                       members=mem, hidden=hidden, distractor=distractor, radius_m=RADIUS,
                       **{k: ev[k] for k in ("max_simultaneous", "group_players", "n_sync_crossings", "frac_edges_both_estimated",
                                             "estimated_comotion", "label_risk", "repeated_interaction")}))
    if MP.validate(d):
        raise RuntimeError(f"{sid}: invalid scenario {MP.validate(d)}")
    return d


def build_match(match: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from . import tracks as TR
    m = C.checked([match])[0]
    M = TR.load(m)
    spans = chain_spans(M)
    E = events(m)
    stats: dict[str, Any] = dict(events_total=int(len(E)), by_n_available=E["n"].value_counts().sort_index().to_dict() if len(E) else {})
    out, rejected = [], 0
    cache: dict[tuple[int, float], pd.DataFrame] = {}

    def positions_at(per, t):
        k = (per, round(t, 2))
        if k not in cache:
            f = M.frames(per, t - 0.05, t + 0.05, 15)
            cache[k] = f.groupby("slot")[["x", "y"]].mean().reset_index()
        return cache[k]
    for n, grp in E.groupby("n"):
        grp = grp.sample(frac=1.0, random_state=_seed(m, n) % 2**32).head(PER_MATCH_CAP.get(int(n), 60))
        for ev in grp.to_dict("records"):
            for v in VARIANTS:
                d = scenario(M, spans, ev, v, positions_at)
                if d is None:
                    rejected += 1
                    continue
                out.append(d)
    stats.update(scenarios=len(out), rejected_variants_chain_or_samples=rejected)
    TR.load.cache_clear()
    return out, stats


def index_row(d: dict[str, Any]) -> dict[str, Any]:
    m = d["meta"]
    est = [h["est_in"] for h in m["hidden"].values()] + [h["est_out"] for h in m["hidden"].values()]
    return dict(scenario_id=d["scenario_id"], event_id=d["event_id"], match_id=m["match"], period=m["period"], t_centre=m["t_centre"],
                variant=m["variant"], N=len(d["incoming"]), M=len(d["outgoing"]), n_null=sum(v is None for v in d["truth"].values()),
                n_new_out=int(m["distractor"] is not None), distractor_m=(m["distractor"] or {}).get("distance_m"), base_gap_s=m["base_gap_s"],
                est_mean=float(np.mean(est)), est_max=float(np.max(est)), provenance=d["provenance"], human_reviewed=d["human_reviewed"],
                **{k: m[k] for k in ("max_simultaneous", "group_players", "n_sync_crossings", "frac_edges_both_estimated", "estimated_comotion",
                                     "label_risk", "repeated_interaction")})


def max_speed(tr: dict[str, Any]) -> float:
    t, x, y = map(np.asarray, (tr["t"], tr["x"], tr["y"]))
    if len(t) < 2:
        return 0.0
    return float(np.max(np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-6)))


def main() -> int:
    out = C.Out(OUT)
    t0 = time.perf_counter()
    per, rows = {}, []
    path = out.path("scenarios.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for m in C.allowed():
            t1 = time.perf_counter()
            sc, st = build_match(m)
            for d in sc:
                fh.write(json.dumps(d) + "\n")
                r = index_row(d)
                r["max_speed_m_s"] = max(max_speed(tr) for tr in d["incoming"] + d["outgoing"])
                rows.append(r)
            st["seconds"] = round(time.perf_counter() - t1, 1)
            per[m] = st
            print(m, st, flush=True)
    out.files["scenarios.jsonl.gz"] = C.sha256_file(path)
    idx = pd.DataFrame(rows)
    out.parquet("index.parquet", idx)
    out.json("build.json", dict(schema=OUT, radius_m=RADIUS, variants=VARIANTS, per_match=per, total=len(idx), runtime_s=round(time.perf_counter() - t0, 1),
                                labels="provider-derived (not human-reviewed); exploratory; not frozen"))
    out.manifest(inputs=C.catalogue_inputs(C.allowed()), parameters=dict(radius_m=RADIUS, context_s=CONTEXT_S, hz=HZ, max_n=MAX_N, caps=PER_MATCH_CAP,
                                                                          gap_range=GAP_RANGE, jitter=JITTER, async_s=ASYNC_S, drop_rate=DROP_RATE,
                                                                          distractor_m=DISTRACTOR_M),
                 code={"m11_explorer/nxm_dataset.py": C.sha256_file(__file__), "m11_explorer/multiplayer.py": C.sha256_file(MP.__file__)})
    return 0


def load(path=None) -> list[dict[str, Any]]:
    p = path or C.PROC / OUT / "scenarios.jsonl.gz"
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh]
    C.checked(sorted({str(d["meta"]["match"]) for d in rows}))
    return rows
