"""N incoming x M outgoing identity association: schema, synthetic generator, deterministic Kalman / Hungarian-with-null reference
solver and set-level metrics.

Status: the solver and metrics are tested on SYNTHETIC examples; ``from_real_group`` builds EXPLORATORY, non-frozen scenarios from
real simultaneous groups of development matches (provider identities as truth, not human-verified). No learned model supports
N x M here; the binary MATCHED / SWAPPED models must not be applied to these scenarios.

Schema ``m11-multiplayer-association-v1`` (JSON-serialisable dict):

* ``incoming``: [{id, label, t, x, y}]: tracks with a known identity that end (disappear) at their last sample;
* ``outgoing``: [{id, t, x, y}]: anonymous tracks that start at their first sample (no identity field is allowed);
* ``dormant`` (optional): [{id, label, t_last, x_last, y_last}];
* ``truth`` (reveal only): {incoming id: outgoing id or null}; never read by the solver;
* ``meta``: free descriptive fields; ``synthetic_example`` / ``exploratory`` flags.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_fatigue.m11_phase2 import baselines as B

SCHEMA = "m11-multiplayer-association-v1"
HZ = 15.0
SIGMA_A, SIGMA_M = 8.0, 0.05            # label-free Phase 2 Kalman selection on development tracklets
GATE = B.GATE_CHI2                      # chi2(2, 0.99)
FORBIDDEN_OUTGOING_KEYS = {"label", "identity", "player_id", "slot", "jersey", "team"}


# ---------------------------------------------------------------- validation

def validate(d: dict[str, Any]) -> list[str]:
    """Structural problems of a scenario (empty list = valid)."""
    p = []
    if d.get("schema") != SCHEMA:
        p.append(f"schema must be {SCHEMA}")
    ins, outs = d.get("incoming", []), d.get("outgoing", [])
    if not ins:
        p.append("at least one incoming track is required")
    ids = [t.get("id") for t in ins + outs + d.get("dormant", [])]
    if len(ids) != len(set(ids)):
        p.append("track ids must be unique")
    for tr in outs:
        bad = FORBIDDEN_OUTGOING_KEYS & set(tr)
        if bad:
            p.append(f"outgoing {tr.get('id')} carries identity fields {sorted(bad)} (outgoing tracks must be anonymous)")
    for tr in ins + outs:
        t = np.asarray(tr.get("t", []), float)
        if len(t) < 2 or len(tr.get("x", [])) != len(t) or len(tr.get("y", [])) != len(t):
            p.append(f"track {tr.get('id')}: needs >= 2 samples with equal-length t / x / y")
        elif np.any(np.diff(t) <= 0):
            p.append(f"track {tr.get('id')}: times must be strictly increasing")
    tru = d.get("truth")
    if tru is not None:
        inn, outn = {t["id"] for t in ins}, {t["id"] for t in outs}
        if set(tru) - inn:
            p.append("truth references unknown incoming ids")
        tgt = [v for v in tru.values() if v is not None]
        if set(tgt) - outn:
            p.append("truth references unknown outgoing ids")
        if len(tgt) != len(set(tgt)):
            p.append("truth is not one-to-one")
    return p


def public(d: dict[str, Any]) -> dict[str, Any]:
    """The solver's view: everything except ``truth``."""
    return {k: v for k, v in d.items() if k != "truth"}


# ---------------------------------------------------------------- deterministic reference solver

def _pad(tracks: list[dict[str, Any]], last: bool, n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(K, n, 2) positions and (K, n) times of the n samples nearest the gap (incoming: last n, outgoing: first n), NaN-padded."""
    K = len(tracks)
    pos, tt = np.full((K, n, 2), np.nan), np.zeros((K, n))
    for k, tr in enumerate(tracks):
        t, x, y = (np.asarray(tr[c], float) for c in ("t", "x", "y"))
        sl = slice(-n, None) if last else slice(0, n)
        t, x, y = t[sl], x[sl], y[sl]
        o = n - len(t) if last else 0
        pos[k, o:o + len(t), 0], pos[k, o:o + len(t), 1], tt[k, o:o + len(t)] = x, y, t
        if last and o:                                   # fill leading times so filter steps are increasing
            tt[k, :o] = t[0] - (np.arange(o, 0, -1)) / HZ
        if not last and len(t) < n:
            tt[k, len(t):] = t[-1] + np.arange(1, n - len(t) + 1) / HZ
    return pos, tt


def kalman_cost(d: dict[str, Any], sa: float = SIGMA_A, sm: float = SIGMA_M) -> dict[str, np.ndarray]:
    """Forward CV-Kalman prediction of each incoming track to each outgoing track's first observation.

    Returns d2 (N, M) Mahalanobis distances, nll (N, M) = d2 + log det S, and ``feasible`` (N, M): False where the outgoing track
    starts before the incoming one ends (one player cannot be on two tracks at once) or d2 exceeds the 0.99 gate."""
    ins, outs = d["incoming"], d["outgoing"]
    N, M = len(ins), len(outs)
    pos, tt = _pad(ins, True)
    x, P, ts, _ = B.kf_filter(pos, tt, sa, sm)
    t_end = np.array([float(tr["t"][-1]) for tr in ins])
    first = np.array([[float(tr["x"][0]), float(tr["y"][0])] for tr in outs]).reshape(M, 2)
    t_first = np.array([float(tr["t"][0]) for tr in outs])
    d2, nll = np.full((N, M), np.inf), np.full((N, M), np.inf)
    order_ok = t_first[None, :] > t_end[:, None] + 1e-9
    R = np.eye(2) * sm ** 2
    for j in range(M):
        dt = np.maximum(t_first[j] - ts, 0.0)
        xp, Pp = B._predict(x, P, dt, sa)
        S = Pp[:, :2, :2] + R
        nu = first[j][None, :] - xp[:, :2]
        q = np.einsum("mi,mij,mj->m", nu, np.linalg.inv(S), nu)
        d2[:, j], nll[:, j] = q, q + np.log(np.linalg.det(S))
    feas = order_ok & (d2 <= GATE)
    return dict(d2=np.where(order_ok, d2, np.inf), nll=np.where(order_ok, nll, np.inf), feasible=feas)


@dataclass
class Decision:
    links: dict[str, str | None]     # incoming id -> outgoing id or None (unmatched)
    unmatched_outgoing: list[str]
    cost: np.ndarray
    feasible: np.ndarray
    runtime_ms: float


def solve(d: dict[str, Any], *, use_null: bool = True, sa: float = SIGMA_A, sm: float = SIGMA_M) -> Decision:
    """Global one-to-one assignment (Hungarian) on Mahalanobis d2, gated; with ``use_null`` each track may stay unmatched at the
    gate cost, so links outside the gate are never forced."""
    t0 = time.perf_counter()
    pub = public(d)
    c = kalman_cost(pub, sa, sm)
    cost = np.where(c["feasible"], c["d2"], np.inf) if use_null else np.where(np.isfinite(c["d2"]), c["d2"], 1e9)
    pairs = B.assign(cost, null_cost=GATE if use_null else None)
    ins, outs = pub["incoming"], pub["outgoing"]
    links = {tr["id"]: None for tr in ins}
    for i, j in pairs:
        links[ins[i]["id"]] = outs[j]["id"]
    used = {v for v in links.values() if v is not None}
    return Decision(links, [o["id"] for o in outs if o["id"] not in used], cost, c["feasible"], round((time.perf_counter() - t0) * 1000, 3))


def score(dec: Decision, truth: dict[str, str | None]) -> dict[str, Any]:
    """Set-level metrics of one scenario."""
    pred = {(i, o) for i, o in dec.links.items() if o is not None}
    true = {(i, o) for i, o in truth.items() if o is not None}
    tp = len(pred & true)
    null_pred = {i for i, o in dec.links.items() if o is None}
    null_true = {i for i, o in truth.items() if o is None}
    return dict(exact=dec.links == truth, links_tp=tp, links_pred=len(pred), links_true=len(true),
                null_tp=len(null_pred & null_true), null_pred=len(null_pred), null_true=len(null_true),
                id_switches=sum(1 for i, o in dec.links.items() if o is not None and truth.get(i) is not None and o != truth[i]))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = lambda k: sum(r[k] for r in rows)  # noqa: E731
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return dict(n=len(rows), exact_assignment=div(sum(r["exact"] for r in rows), len(rows)),
                link_precision=div(s("links_tp"), s("links_pred")), link_recall=div(s("links_tp"), s("links_true")),
                null_precision=div(s("null_tp"), s("null_pred")), null_recall=div(s("null_tp"), s("null_true")),
                id_switches_per_scenario=div(s("id_switches"), len(rows)))


# ---------------------------------------------------------------- synthetic generator (curriculum levels)

LEVELS = {
    1: dict(name="2 players, simultaneous 0.8 s gap", n=2, gap=(0.8, 0.8), async_s=0.0, noise=0.0, distractors=0, p_null=0.0),
    2: dict(name="2 players, variable gap 0.4-1.6 s, noise 5 cm", n=2, gap=(0.4, 1.6), async_s=0.0, noise=0.05, distractors=0, p_null=0.0),
    3: dict(name="2 players + 1 distractor, null with p 0.3", n=2, gap=(0.4, 1.6), async_s=0.0, noise=0.05, distractors=1, p_null=0.3),
    4: dict(name="3 players, overlapping gaps", n=3, gap=(0.6, 1.2), async_s=0.3, noise=0.05, distractors=0, p_null=0.0),
    5: dict(name="4-5 players, distractor and null", n=(4, 5), gap=(0.6, 1.2), async_s=0.3, noise=0.05, distractors=1, p_null=0.3),
    7: dict(name="3-5 players, asynchronous partial loss (one may stay visible)", n=(3, 5), gap=(0.2, 1.6), async_s=0.6, noise=0.05,
            distractors=1, p_null=0.3, p_visible=0.3),
}


def _motion(rng, T: np.ndarray, centre: np.ndarray, speed: float) -> np.ndarray:
    """A smooth player path passing near ``centre`` at t = 0 (random heading, gentle Ornstein-Uhlenbeck turning)."""
    h = rng.uniform(-np.pi, np.pi)
    w = np.zeros(len(T))
    dt = T[1] - T[0]
    for k in range(1, len(T)):
        w[k] = 0.9 * w[k - 1] + rng.normal(0, 0.15)
    i0 = int(np.argmin(np.abs(T)))
    head = h + np.cumsum(w) * dt - (np.cumsum(w) * dt)[i0]
    v = np.c_[np.cos(head), np.sin(head)] * speed
    p = np.cumsum(v * dt, 0)
    return p - p[i0] + centre


def generate(level: int, seed: int) -> dict[str, Any]:
    """One synthetic scenario of a curriculum level (deterministic for (level, seed))."""
    L = LEVELS[level]
    rng = np.random.default_rng([level, seed])
    n = L["n"] if isinstance(L["n"], int) else int(rng.integers(L["n"][0], L["n"][1] + 1))
    T = np.arange(-4.0, 4.0 + 1e-9, 1 / HZ)
    players = [_motion(rng, T, rng.normal(0, 0.4, 2), rng.uniform(1.0, 6.0)) for _ in range(n + L["distractors"])]
    incoming, outgoing, truth, hidden = [], [], {}, []
    out_ids = [f"out-{k}" for k in rng.permutation(len(players))]        # anonymous ids in random order
    null_player = int(rng.integers(n)) if rng.random() < L["p_null"] else None
    for k, p in enumerate(players):
        gap = rng.uniform(*L["gap"])
        if rng.random() < L.get("p_visible", 0.0) and k < n:
            gap = 1 / HZ                                                 # never really disappears (one-sample break)
        c0 = rng.uniform(-L["async_s"], L["async_s"])
        h0, h1 = c0 - gap / 2, c0 + gap / 2
        noise = rng.normal(0, L["noise"], p.shape) if L["noise"] else 0
        q = p + noise
        bi = (T < h0) & (T >= h0 - 2.0)
        ai = (T > h1) & (T <= h1 + 2.0)
        if k < n:
            incoming.append(dict(id=f"in-{k}", label=f"player-{k}", t=T[bi].round(4).tolist(), x=q[bi, 0].round(4).tolist(),
                                 y=q[bi, 1].round(4).tolist()))
            hidden.append([round(h0, 3), round(h1, 3)])
        if k == null_player:
            truth[f"in-{k}"] = None                                      # leaves the view: no outgoing track
            continue
        outgoing.append(dict(id=out_ids[k], t=T[ai].round(4).tolist(), x=q[ai, 0].round(4).tolist(), y=q[ai, 1].round(4).tolist()))
        if k < n:
            truth[f"in-{k}"] = out_ids[k]
    outgoing.sort(key=lambda o: o["id"])
    return dict(schema=SCHEMA, synthetic_example=True, event_id=f"syn-L{level}-{seed}", incoming=incoming, outgoing=outgoing, truth=truth,
                meta=dict(level=level, level_name=L["name"], n_incoming=n, n_outgoing=len(outgoing), distractors=L["distractors"],
                          null_player=null_player is not None, hidden_intervals_s=hidden, noise_sigma_m=L["noise"]))


def synthetic_benchmark(n_per_level: int = 300, levels=tuple(LEVELS)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lv in levels:
        rows, rt, rows_nonull = [], [], []
        for s in range(n_per_level):
            d = generate(lv, s)
            assert not validate(d), validate(d)
            dec = solve(d)
            rows.append(score(dec, d["truth"]))
            rows_nonull.append(score(solve(d, use_null=False), d["truth"]))
            rt.append(dec.runtime_ms)
        out[str(lv)] = dict(level=LEVELS[lv]["name"], kalman_hungarian_null=aggregate(rows), kalman_hungarian_forced=aggregate(rows_nonull),
                            runtime_ms_median=float(np.median(rt)))
    return out


# ---------------------------------------------------------------- exploratory scenarios from real simultaneous groups

def from_real_group(tracks, slots: list[int], period: int, t_centre: float, gap_s: float = 0.8, async_s: float = 0.0, seed: int = 0,
                    context_s: float = 2.0) -> dict[str, Any]:
    """EXPLORATORY (not frozen, not a dataset): hide each listed player for ``gap_s`` around ``t_centre`` (per-player offsets up to
    ``async_s``) in a development match. Truth = provider identities (not human-verified). Outgoing ids are anonymous and shuffled."""
    rng = np.random.default_rng([int(t_centre * 1000) % 2**31, seed, len(slots)])
    incoming, outgoing, truth, hidden = [], [], {}, []
    perm = rng.permutation(len(slots))
    for k, s in enumerate(slots):
        c0 = t_centre + rng.uniform(-async_s, async_s)
        h0, h1 = c0 - gap_s / 2, c0 + gap_s / 2
        a = tracks.path(int(s), period, h0 - context_s, h0)
        b = tracks.path(int(s), period, h1, h1 + context_s)
        a, b = a[a["t"] < h0], b[b["t"] > h1]
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"slot {s}: not enough observed samples around the hidden interval")
        oid = f"out-{perm[k]}"
        incoming.append(dict(id=f"in-{k}", label=f"slot-{s}", t=a["t"].round(4).tolist(), x=a["x"].round(4).tolist(), y=a["y"].round(4).tolist()))
        outgoing.append(dict(id=oid, t=b["t"].round(4).tolist(), x=b["x"].round(4).tolist(), y=b["y"].round(4).tolist()))
        truth[f"in-{k}"] = oid
        hidden.append([round(h0, 3), round(h1, 3)])
    outgoing.sort(key=lambda o: o["id"])
    return dict(schema=SCHEMA, exploratory=True, synthetic_example=False, event_id=f"real-{tracks.match}-p{period}-{t_centre:.2f}",
                incoming=incoming, outgoing=outgoing, truth=truth,
                meta=dict(match=tracks.match, period=period, slots=[int(s) for s in slots], gap_s=gap_s, async_s=async_s,
                          hidden_intervals_s=hidden, note="exploratory; provider identities as truth (not human-verified)"))
