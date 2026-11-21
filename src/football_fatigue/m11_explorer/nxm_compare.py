"""N x M association on the EXPLORATORY provider-derived prototype (``m11-nxm-exploratory-v1``): deterministic solver variants,
learned link scorers, dataset-quality / shortcut analysis (``python -m football_fatigue.m11_explorer nxm-compare``).

Systems (all enforce one-to-one assignment with the Hungarian algorithm):

* S0  original: Kalman forward Mahalanobis d2, 0.99 gate, null cost = gate (``multiplayer.solve``; untuned);
* S1  bidirectional motion cost: mean of forward d2 (incoming predicted to the outgoing start) and backward d2 (outgoing filtered in
      reverse time, predicted back to the incoming end); same gate / null;
* S2  S1 with sigma_a, the incoming null cost and the outgoing null cost tuned on the training folds (cross-fitted);
* L1  learned link scorer (logistic regression or HGB on motion pair features) + Hungarian, no explicit null (shape only);
* L2  learned link scorer + Hungarian with explicit nulls: null cost -log(tau), tau tuned on the training folds;
* L2t L2 with provider-texture features added (ablation that measures reliance on the provider-texture shortcut).

Cross-fitting: scenarios are split by match into the 8 frozen Stage 4 outer folds; every tuned parameter and every learned model is
fitted on the other 7 folds only. Labels (truth) are used only as training targets and for scoring; features come from
``multiplayer.public`` (truth removed). Confidence: per-link margin = increase of the optimal total cost when that link is
forbidden; abstention thresholds are tuned on training folds for a target link precision.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_fatigue.m11_phase2 import baselines as B
from football_fatigue.m11_phase2 import common as C

from . import multiplayer as MP

OUT = "m11-nxm-solver-comparison-v1"
SIGMAS = (2.0, 4.0, 8.0, 16.0, 32.0)
NULLS = (0.5, 1.0, 2.0, 4.0, 9.21, 16.0, 36.0)
TAUS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BIG = 1e12


# ---------------------------------------------------------------- assignment with separate null costs

def assign2(cost: np.ndarray, lam_in: float | None, lam_out: float | None) -> list[tuple[int, int]]:
    """One-to-one assignment; an incoming row may stay unmatched at ``lam_in``, an outgoing column at ``lam_out`` (None: no null,
    i.e. only the rectangular shape leaves tracks unmatched). Infinite costs are never chosen."""
    c = np.where(np.isfinite(cost), cost, BIG)
    n, m = c.shape
    if lam_in is None and lam_out is None:
        r, k = linear_sum_assignment(c)
        return [(int(a), int(b)) for a, b in zip(r, k) if c[a, b] < BIG]
    li = BIG if lam_in is None else lam_in
    lo = BIG if lam_out is None else lam_out
    A = np.full((n + m, m + n), BIG)
    A[:n, :m] = c
    A[:n, m:] = np.where(np.eye(n, dtype=bool), li, BIG)
    A[n:, :m] = np.where(np.eye(m, dtype=bool), lo, BIG)
    A[n:, m:] = 0.0
    if lam_in is None:                      # every incoming must link when possible: allow shape-forced nulls only if n > m
        A[:n, m:] = np.where(np.eye(n, dtype=bool), BIG / 10 if n > m else BIG, BIG)
    r, k = linear_sum_assignment(A)
    return [(int(a), int(b)) for a, b in zip(r, k) if a < n and b < m and A[a, b] < BIG / 10]


def link_margins(cost: np.ndarray, pairs, lam_in, lam_out) -> dict[tuple[int, int], float]:
    """Confidence of each chosen link: optimal total cost with the link forbidden minus the optimal total cost."""
    def total(c):
        p = assign2(c, lam_in, lam_out)
        used_i, used_j = {a for a, _ in p}, {b for _, b in p}
        n, m = c.shape
        return sum(c[a, b] for a, b in p) + (lam_in or 0) * (n - len(used_i)) + (lam_out or 0) * (m - len(used_j))
    base = total(cost)
    out = {}
    for a, b in pairs:
        c2 = cost.copy()
        c2[a, b] = np.inf
        out[(a, b)] = float(min(total(np.where(np.isfinite(c2), c2, BIG / 100)) - base, 1e6))
    return out


# ---------------------------------------------------------------- motion costs (public inputs only)

def _arrays(d: dict[str, Any]):
    pub = MP.public(d)
    return pub, MP._pad(pub["incoming"], True), MP._pad(pub["outgoing"], False)


def motion_costs(d: dict[str, Any], sa: float = MP.SIGMA_A, sm: float = MP.SIGMA_M) -> dict[str, np.ndarray]:
    pub, (pi, ti), (po, to) = _arrays(d)
    N, M = len(pub["incoming"]), len(pub["outgoing"])
    xi, Pi, tsi, _ = B.kf_filter(pi, ti, sa, sm)
    # backward: outgoing in reverse time
    pr, tr = po[:, ::-1], -to[:, ::-1]
    xo, Po, tso, _ = B.kf_filter(pr, tr, sa, sm)
    t_end = np.array([float(t["t"][-1]) for t in pub["incoming"]])
    end = np.array([[float(t["x"][-1]), float(t["y"][-1])] for t in pub["incoming"]]).reshape(N, 2)
    t_first = np.array([float(t["t"][0]) for t in pub["outgoing"]])
    first = np.array([[float(t["x"][0]), float(t["y"][0])] for t in pub["outgoing"]]).reshape(M, 2)
    R = np.eye(2) * sm ** 2
    fwd, bwd = np.full((N, M), np.inf), np.full((N, M), np.inf)
    for j in range(M):
        xp, Pp = B._predict(xi, Pi, np.maximum(t_first[j] - tsi, 0.0), sa)
        S = Pp[:, :2, :2] + R
        nu = first[j][None] - xp[:, :2]
        fwd[:, j] = np.einsum("mi,mij,mj->m", nu, np.linalg.inv(S), nu)
    for i in range(N):
        xp, Pp = B._predict(xo, Po, np.maximum(-t_end[i] - tso, 0.0), sa)
        S = Pp[:, :2, :2] + R
        nu = end[i][None] - xp[:, :2]
        bwd[i, :] = np.einsum("mi,mij,mj->m", nu, np.linalg.inv(S), nu)
    order = t_first[None, :] > t_end[:, None] + 1e-9
    return dict(fwd=np.where(order, fwd, np.inf), bwd=np.where(order, bwd, np.inf), bi=np.where(order, 0.5 * (fwd + bwd), np.inf),
                order=order, t_end=t_end, t_first=t_first, end=end, first=first)


def _vel(tr: dict[str, Any], last: bool, win: float = 0.5) -> np.ndarray:
    t, x, y = (np.asarray(tr[c], float) for c in ("t", "x", "y"))
    k = t >= t[-1] - win if last else t <= t[0] + win
    if k.sum() < 2:
        return np.zeros(2)
    A = np.c_[np.ones(k.sum()), t[k] - t[k].mean()]
    return np.array([np.linalg.lstsq(A, x[k], rcond=None)[0][1], np.linalg.lstsq(A, y[k], rcond=None)[0][1]])


def _rough(tr: dict[str, Any]) -> float:
    p = np.c_[tr["x"], tr["y"]].astype(float)
    if len(p) < 4:
        return 0.0
    return float(np.median(np.hypot(*np.diff(p, 2, axis=0).T)))


FEATURES = ["d2_fwd", "d2_bwd", "d2_bi", "dist", "dt", "req_speed", "speed_in", "speed_out", "dspeed", "cos_heading", "cv_miss_fwd",
            "cv_miss_bwd", "rel_row", "rel_col", "rank_row", "rank_col", "N", "M"]
TEXTURE = ["texture_diff", "texture_in"]


def pair_features(d: dict[str, Any], texture: bool = False) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(N*M, F) features of every incoming x outgoing pair from the public scenario (truth is not read)."""
    pub = MP.public(d)
    c = motion_costs(pub)
    ins, outs = pub["incoming"], pub["outgoing"]
    N, M = len(ins), len(outs)
    vi, vo = [_vel(t, True) for t in ins], [_vel(t, False) for t in outs]
    lc = lambda a: np.log1p(np.minimum(a, 1e6))  # noqa: E731
    bi = lc(np.where(np.isfinite(c["bi"]), c["bi"], 1e6))
    rows, idx = [], []
    for i in range(N):
        for j in range(M):
            dt = c["t_first"][j] - c["t_end"][i]
            dist = float(np.hypot(*(c["first"][j] - c["end"][i])))
            si_, so_ = float(np.hypot(*vi[i])), float(np.hypot(*vo[j]))
            cos = float(vi[i] @ vo[j] / (si_ * so_ + 1e-9))
            miss_f = float(np.hypot(*(c["end"][i] + vi[i] * max(dt, 0) - c["first"][j])))
            miss_b = float(np.hypot(*(c["first"][j] - vo[j] * max(dt, 0) - c["end"][i])))
            f = [lc(np.array(c["fwd"][i, j] if np.isfinite(c["fwd"][i, j]) else 1e6)), lc(np.array(c["bwd"][i, j] if np.isfinite(c["bwd"][i, j]) else 1e6)),
                 bi[i, j], dist, dt, dist / max(dt, 1e-3), si_, so_, abs(si_ - so_), cos, miss_f, miss_b, bi[i, j] - bi[i].min(), bi[i, j] - bi[:, j].min(),
                 float((bi[i] < bi[i, j]).sum()), float((bi[:, j] < bi[i, j]).sum()), N, M]
            if texture:
                ri, ro = _rough(ins[i]), _rough(outs[j])
                f += [abs(np.log1p(1e3 * ri) - np.log1p(1e3 * ro)), np.log1p(1e3 * ri)]
            rows.append(f)
            idx.append((i, j))
    return np.asarray(rows, float), idx


# ---------------------------------------------------------------- decisions and scoring

def decide(d: dict[str, Any], cost: np.ndarray, lam_in, lam_out, margins: bool = True) -> MP.Decision:
    pub = MP.public(d)
    pairs = assign2(cost, lam_in, lam_out)
    ins, outs = pub["incoming"], pub["outgoing"]
    links = {t["id"]: None for t in ins}
    for a, b in pairs:
        links[ins[a]["id"]] = outs[b]["id"]
    used = {v for v in links.values() if v is not None}
    dec = MP.Decision(links, [o["id"] for o in outs if o["id"] not in used], cost, np.isfinite(cost), 0.0)
    dec.margins = {(ins[a]["id"], outs[b]["id"]): m for (a, b), m in link_margins(cost, pairs, lam_in, lam_out).items()} if margins else {}
    return dec


def score_row(d: dict[str, Any], dec: MP.Decision) -> dict[str, Any]:
    s = MP.score(dec, d["truth"])
    s["false_links"] = s["links_pred"] - s["links_tp"]
    m = getattr(dec, "margins", None) or {}
    s["link_conf"] = [(float(v), int(d["truth"].get(i) == o)) for (i, o), v in m.items()]
    return s


def coverage_curve(rows: list[dict[str, Any]], coverages=(1.0, 0.9, 0.8, 0.6, 0.4, 0.2)) -> dict[str, Any]:
    """Descriptive link precision when only the most confident links are kept (margin ranking; no threshold is tuned on test)."""
    lc = np.array([x for r in rows for x in r.get("link_conf", [])], float)
    if not len(lc):
        return {}
    o = lc[np.argsort(-lc[:, 0])]
    return {str(c): round(float(o[: max(1, int(round(c * len(o))))][:, 1].mean()), 4) for c in coverages}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    tp, pred, true = df["links_tp"].sum(), df["links_pred"].sum(), df["links_true"].sum()
    p, r = tp / max(pred, 1), tp / max(true, 1)
    nt, npred, ntrue = df["null_tp"].sum(), df["null_pred"].sum(), df["null_true"].sum()
    return dict(n=int(len(df)), exact=round(float(df["exact"].mean()), 4), link_precision=round(float(p), 4), link_recall=round(float(r), 4),
                link_f1=round(float(2 * p * r / max(p + r, 1e-12)), 4), id_switches_per_scenario=round(float(df["id_switches"].mean()), 4),
                false_link_rate=round(float(df["false_links"].sum() / max(pred, 1)), 4),
                null_recall=round(float(nt / ntrue), 4) if ntrue else None, null_precision=round(float(nt / npred), 4) if npred else None)


def boot_exact(df: pd.DataFrame, col_a: str, col_b: str | None = None, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Match-level bootstrap of the exact-assignment rate (or of a paired difference of two systems)."""
    g = df.groupby("match_id")
    a = g[col_a].agg(["sum", "count"]).to_numpy(float)
    b = g[col_b].agg(["sum", "count"]).to_numpy(float) if col_b else None
    est = a[:, 0].sum() / a[:, 1].sum() - (b[:, 0].sum() / b[:, 1].sum() if b is not None else 0)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(a), len(a))
        bs.append(a[k, 0].sum() / a[k, 1].sum() - (b[k, 0].sum() / b[k, 1].sum() if b is not None else 0))
    return dict(value=round(float(est), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])


# ---------------------------------------------------------------- experiment

def folds_of(sc: list[dict[str, Any]]) -> np.ndarray:
    from football_fatigue.m11_benchmark import data as D
    fmap = json.loads((D.out_dir() / "folds.json").read_text(encoding="utf-8"))["outer"]
    return np.array([fmap[str(d["meta"]["match"])] for d in sc])


def _tune_motion(sc, costs_by_sigma, idx):
    best, bp = -1, None
    for sa in SIGMAS:
        for li in NULLS:
            for lo in NULLS:
                ex = np.mean([score_row(sc[k], decide(sc[k], costs_by_sigma[sa][k], li, lo, False))["exact"] for k in idx])
                if ex > best:
                    best, bp = ex, (sa, li, lo)
    return bp, best


def run(sc: list[dict[str, Any]], log=print) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fold = folds_of(sc)
    t0 = time.perf_counter()
    costs = {sa: [motion_costs(d, sa)["bi"] for d in sc] for sa in SIGMAS}
    log(f"motion costs {time.perf_counter() - t0:.0f} s")
    feats = [pair_features(d, texture=True) for d in sc]
    nf = len(FEATURES)
    labels = []
    for d, (_X, idx) in zip(sc, feats):
        ins, outs = [t["id"] for t in d["incoming"]], [t["id"] for t in d["outgoing"]]
        labels.append(np.array([int(d["truth"][ins[i]] == outs[j]) for i, j in idx]))
    log(f"features {time.perf_counter() - t0:.0f} s")
    rows = {k: [None] * len(sc) for k in ("S0", "S1", "S2", "L1_lr", "L1_hgb", "L2_lr", "L2_hgb", "L2t_hgb")}
    chosen = {}
    for f in np.unique(fold):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        (sa, li, lo), _ = _tune_motion(sc, costs, tr[:: max(1, len(tr) // 1500)])       # bounded tuning subset (training folds only)
        chosen[int(f)] = dict(S2=dict(sigma_a=sa, null_in=li, null_out=lo))
        Xtr = np.concatenate([feats[k][0] for k in tr])
        ytr = np.concatenate([labels[k] for k in tr])
        models = {"lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr[:, :nf], ytr),
                  "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr[:, :nf], ytr),
                  "hgbt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)}

        def pmat(k, name):
            X, idx = feats[k]
            p = models[name].predict_proba(X if name == "hgbt" else X[:, :nf])[:, 1]
            P = np.zeros((len(sc[k]["incoming"]), len(sc[k]["outgoing"])))
            for (i, j), v in zip(idx, p):
                P[i, j] = v
            return np.clip(P, 1e-6, 1 - 1e-6)
        # tau for explicit nulls on the training folds (subset)
        sub = tr[:: max(1, len(tr) // 1500)]
        for name in ("lr", "hgb", "hgbt"):
            Ps = {k: pmat(k, name) for k in sub}
            best = max(TAUS, key=lambda tau: np.mean([score_row(sc[k], decide(sc[k], -np.log(Ps[k]), -np.log(tau), -np.log(tau), False))["exact"] for k in sub]))
            chosen[int(f)][f"tau_{name}"] = best
        for k in te:
            d = sc[k]
            c0 = motion_costs(d)["fwd"]
            rows["S0"][k] = score_row(d, decide(d, np.where(c0 <= MP.GATE, c0, np.inf), MP.GATE, MP.GATE))
            c1 = costs[MP.SIGMA_A][k]
            rows["S1"][k] = score_row(d, decide(d, np.where(c1 <= MP.GATE, c1, np.inf), MP.GATE, MP.GATE))
            rows["S2"][k] = score_row(d, decide(d, costs[sa][k], li, lo))
            for name, key in (("lr", "lr"), ("hgb", "hgb")):
                P = pmat(k, name)
                rows[f"L1_{key}"][k] = score_row(d, decide(d, -np.log(P), None, None))
                tau = chosen[int(f)][f"tau_{name}"]
                dec = decide(d, -np.log(P), -np.log(tau), -np.log(tau))
                rows[f"L2_{key}"][k] = dict(score_row(d, dec), min_margin=min(dec.margins.values()) if dec.margins else None)
            P = pmat(k, "hgbt")
            tau = chosen[int(f)]["tau_hgbt"]
            rows["L2t_hgb"][k] = score_row(d, decide(d, -np.log(P), -np.log(tau), -np.log(tau)))
        log(f"fold {f}: {len(te)} scenarios, S2 {chosen[int(f)]['S2']} ({time.perf_counter() - t0:.0f} s)")
    return dict(rows=rows, chosen=chosen, runtime_s=round(time.perf_counter() - t0, 1))


def tables(sc: list[dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    meta = pd.DataFrame([dict(match_id=d["meta"]["match"], N=len(d["incoming"]), M=len(d["outgoing"]), variant=d["meta"]["variant"],
                              gap=d["meta"]["base_gap_s"], est=np.mean([h["est_in"] for h in d["meta"]["hidden"].values()]),
                              crossing=d["meta"]["n_sync_crossings"] > 0, max_sim=d["meta"]["max_simultaneous"],
                              has_null=any(v is None for v in d["truth"].values()), has_new=d["meta"]["distractor"] is not None) for d in sc])
    meta["gap_band"] = pd.cut(meta["gap"], [0, 0.7, 1.0, 1.3, 2.5], labels=["<=0.7", "0.7-1.0", "1.0-1.3", ">1.3"]).astype(str)
    meta["est_band"] = pd.cut(meta["est"], [-1e-9, 0.0, 0.5, 1.0], labels=["visible", "est<=50%", "est>50%"]).astype(str)
    meta["NxM"] = meta["N"].astype(str) + "x" + meta["M"].astype(str)
    out: dict[str, Any] = {"overall": {k: summarize(v) for k, v in rows.items()}}
    for by in ("N", "NxM", "variant", "gap_band", "est_band", "crossing", "has_null", "has_new"):
        out[f"by_{by}"] = {str(lv): {k: summarize([v[i] for i in np.flatnonzero(meta[by].to_numpy() == lv)]) for k, v in rows.items()}
                           for lv in sorted(meta[by].unique(), key=str)}
    ex = meta[["match_id"]].copy()
    for k, v in rows.items():
        ex[k] = [int(r["exact"]) for r in v]
    out["exact_ci"] = {k: boot_exact(ex, k) for k in rows}
    out["link_precision_by_coverage"] = {k: coverage_curve(v) for k, v in rows.items() if any(r.get("link_conf") for r in v)}
    out["paired_vs_S0"] = {k: boot_exact(ex, k, "S0") for k in rows if k != "S0"}
    out["paired_L2_hgb_vs_S2"] = boot_exact(ex, "L2_hgb", "S2")
    return out


def quality(sc: list[dict[str, Any]], idx: pd.DataFrame) -> dict[str, Any]:
    """Descriptive dataset quality and shortcut opportunities (no model fitting)."""
    q: dict[str, Any] = dict(scenarios=len(sc), events=int(idx["event_id"].nunique()), matches=int(idx["match_id"].nunique()))
    q["by_NxM"] = (idx["N"].astype(str) + "x" + idx["M"].astype(str)).value_counts().sort_index().to_dict()
    q["by_variant"] = idx["variant"].value_counts().to_dict()
    per = idx.groupby("match_id")["event_id"].nunique()
    q["events_per_match"] = dict(min=int(per.min()), median=float(per.median()), max=int(per.max()))
    q["events_by_N"] = idx.drop_duplicates("event_id")["N"].value_counts().sort_index().to_dict()
    q["null_frequency"] = dict(scenarios_with_null=round(float((idx["n_null"] > 0).mean()), 4), null_share_of_incoming=round(float(idx["n_null"].sum() / idx["N"].sum()), 4))
    q["distractor"] = dict(scenarios_with_new_outgoing=int(idx["n_new_out"].sum()), distance_m=C.q(idx["distractor_m"].dropna()) if idx["distractor_m"].notna().any() else None)
    q["estimated"] = dict(mean_context_est=round(float(idx["est_mean"].mean()), 4), share_any_track_mostly_est=round(float((idx["est_max"] > 0.5).mean()), 4),
                          share_all_visible=round(float((idx["est_max"] == 0).mean()), 4))
    q["label_ambiguity"] = dict(label_risk=round(float(idx["label_risk"].mean()), 4), estimated_comotion=round(float(idx["estimated_comotion"].mean()), 4),
                                repeated_interaction=round(float(idx["repeated_interaction"].mean()), 4), sync_crossing=round(float((idx["n_sync_crossings"] > 0).mean()), 4))
    q["motion_plausibility"] = dict(max_speed_over_12_m_s=round(float((idx["max_speed_m_s"] > 12).mean()), 4), max_speed_q=C.q(idx["max_speed_m_s"]))
    # independence: events of the same period sharing a player within 10 s are one dependence cluster
    ev = pd.DataFrame([dict(event_id=d["event_id"], match=d["meta"]["match"], period=d["meta"]["period"], t=d["meta"]["t_centre"], members=d["meta"]["members"]) for d in sc]).drop_duplicates("event_id")
    clusters = 0
    for (_m, _p), g in ev.sort_values("t").groupby(["match", "period"]):
        last: dict[int, float] = {}
        for r in g.itertuples():
            if not any(r.t - last.get(s, -1e9) < 10 for s in r.members):
                clusters += 1
            for s in r.members:
                last[s] = r.t
    q["dependence"] = dict(events=int(len(ev)), clusters_10s_shared_player=clusters,
                           note="variants of one event are augmentations; events sharing a player within 10 s are dependent")
    # shortcut probes
    tim, idord, tex, cnt = [], [], [], []
    for d in sc:
        ins, outs = d["incoming"], d["outgoing"]
        truth = d["truth"]
        g = d["meta"]["base_gap_s"]
        C_t = np.array([[abs((o["t"][0] - i["t"][-1]) - g) for o in outs] for i in ins])
        P_t = assign2(C_t, None, None)
        pred = {ins[a]["id"]: outs[b]["id"] for a, b in P_t}
        tim.append((d["meta"]["variant"], all(pred.get(k) == v for k, v in truth.items() if v is not None)))
        order = sorted(o["id"] for o in outs)
        idord.append(np.mean([order.index(v) == k for k, (kk, v) in enumerate(truth.items()) if v is not None and k < len(order)]))
        R = np.array([[abs(np.log1p(1e3 * _rough(i)) - np.log1p(1e3 * _rough(o))) for o in outs] for i in ins])
        P_r = assign2(R, None, None)
        pr = {ins[a]["id"]: outs[b]["id"] for a, b in P_r}
        tex.append((np.mean([h["est_in"] for h in d["meta"]["hidden"].values()]) > 0.5, len(ins),
                    all(pr.get(k) == v for k, v in truth.items() if v is not None)))
        cnt.append((len(outs) < len(ins), any(v is None for v in truth.values())))
    tim = pd.DataFrame(tim, columns=["variant", "ok"])
    q["shortcut_timing_oracle_exact"] = tim.groupby("variant")["ok"].mean().round(4).to_dict()
    q["shortcut_timing_note"] = ("assignment by |t_start(out) - t_end(in) - true base gap| (an oracle that knows the base gap); "
                                 "with per-player jitter this should be far from 1.0 in async variants")
    q["shortcut_outgoing_id_order_matches_incoming_order"] = round(float(np.nanmean(idord)), 4)
    tex = pd.DataFrame(tex, columns=["mostly_est", "N", "ok"])
    q["shortcut_texture_rule_exact"] = {f"N={n} {'est>50%' if e else 'est<=50%'}": round(float(g["ok"].mean()), 4) for (e, n), g in tex.groupby(["mostly_est", "N"])}
    import math
    q["chance_exact_by_N"] = {int(n): round(1 / math.factorial(int(n)), 4) for n in sorted(tex["N"].unique())}
    cnt = pd.DataFrame(cnt, columns=["fewer_out", "has_null"])
    q["shortcut_count"] = dict(p_null_given_fewer_outgoing=round(float(cnt.loc[cnt["fewer_out"], "has_null"].mean()), 4) if cnt["fewer_out"].any() else None,
                               p_null_given_not_fewer=round(float(cnt.loc[~cnt["fewer_out"], "has_null"].mean()), 4))
    return q


def synthetic_reference(n: int = 200) -> dict[str, Any]:
    out = {}
    for lv in sorted(MP.LEVELS):
        r0, r1 = [], []
        for s in range(n):
            d = MP.generate(lv, s)
            c = motion_costs(d)
            r0.append(score_row(d, decide(d, np.where(c["fwd"] <= MP.GATE, c["fwd"], np.inf), MP.GATE, MP.GATE)))
            r1.append(score_row(d, decide(d, np.where(c["bi"] <= MP.GATE, c["bi"], np.inf), MP.GATE, MP.GATE)))
        out[str(lv)] = dict(level=MP.LEVELS[lv]["name"], S0=summarize(r0), S1=summarize(r1))
    return out


def main() -> int:
    from . import nxm_dataset as ND
    out = C.Out(OUT)
    sc = ND.load()
    idx = pd.read_parquet(C.PROC / ND.OUT / "index.parquet")
    if list(idx["scenario_id"]) != [d["scenario_id"] for d in sc]:
        raise SystemExit("index and scenarios are not aligned")
    q = quality(sc, idx)
    out.json("quality.json", q)
    print(json.dumps({k: q[k] for k in ("scenarios", "by_NxM", "shortcut_timing_oracle_exact", "shortcut_texture_rule_exact")}, indent=1), flush=True)
    r = run(sc)
    t = tables(sc, r["rows"])
    out.json("comparison.json", dict(schema=OUT, label="EXPLORATORY (provider-derived truth, not human-reviewed, not frozen)", tables=t,
                                     chosen_parameters=r["chosen"], runtime_s=r["runtime_s"], systems=__doc__))
    out.json("synthetic_reference.json", synthetic_reference())
    pd.DataFrame([dict(scenario_id=d["scenario_id"], match_id=d["meta"]["match"], **{f"{k}_exact": int(v[i]["exact"]) for k, v in r["rows"].items()})
                  for i, d in enumerate(sc)]).to_parquet(out.path("per_scenario.parquet"), index=False)
    here = Path(__file__).resolve().parent
    out.manifest(inputs=dict(dataset=f"data/processed/{ND.OUT}"), parameters=dict(sigmas=SIGMAS, nulls=NULLS, taus=TAUS, features=FEATURES, texture=TEXTURE),
                 code={p.name: C.sha256_file(p) for p in (here / "nxm_compare.py", here / "nxm_dataset.py", here / "multiplayer.py")})
    print(json.dumps(t["overall"], indent=1))
    return 0
