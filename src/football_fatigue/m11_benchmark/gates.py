"""Stage 4 correctness and leakage gates (run before any model comparison). Each gate returns a dict with ``pass``."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.window_intersections import samples as S

from . import data as D
from . import folds as FO

TABLES = tuple(S.TABLES)


def g1_manifests() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in (S.STAGE2, D.STAGE3):
        d = D.proc_root() / name
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        bad = [k for k, h in man["outputs_sha256"].items() if hashlib.sha256((d / k).read_bytes()).hexdigest() != h]
        out[name] = dict(files=len(man["outputs_sha256"]), mismatched=bad, raw_unchanged=man.get("raw_unchanged"),
                         manifest_sha256=hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest())
    s3 = json.loads((D.stage3_dir() / "manifest.json").read_text(encoding="utf-8"))
    out["stage3_records_stage2_manifest"] = s3["stage2_manifest_sha256"] == out[S.STAGE2]["manifest_sha256"]
    ck = json.loads((D.stage3_dir() / "checks.json").read_text(encoding="utf-8"))
    out["stage3_checks_all_pass"] = ck["all_pass"]
    out["pass"] = all(not v["mismatched"] and v["raw_unchanged"] for k, v in out.items() if isinstance(v, dict)) \
        and out["stage3_records_stage2_manifest"] and ck["all_pass"] and not ck["heldout_built"]
    return out


def g2_split() -> dict[str, Any]:
    from football_fatigue.window_intersections.split import load_split
    sp = load_split(D.proc_root(), S.SPLIT)
    sha = S.split_sha256(D.proc_root())
    s3 = json.loads((D.stage3_dir() / "manifest.json").read_text(encoding="utf-8"))
    built = sorted(s3["matches"], key=int)
    r = dict(split_sha256=sha, approved=sha == S.SPLIT_SHA256, n_train_cv=len(sp["train_cv"]), n_heldout=len(sp["heldout"]),
             disjoint=not set(sp["train_cv"]) & set(sp["heldout"]), heldout=sp["heldout"],
             stage3_matches_equal_train_cv=set(built) == set(sp["train_cv"]))
    fo = FO.outer_folds(sp["train_cv"])
    sizes = np.bincount(list(fo.values()))
    r.update(folds=fo, fold_sizes=sizes.tolist())
    r["pass"] = r["approved"] and r["n_train_cv"] == 32 and r["n_heldout"] == 8 and r["disjoint"] and r["stage3_matches_equal_train_cv"] \
        and (sizes == 4).all()
    return r


def g3_loader_refuses_heldout() -> dict[str, Any]:
    from football_fatigue.window_intersections.split import load_split
    held = load_split(D.proc_root(), S.SPLIT)["heldout"]
    res = {}
    for m in held:
        for name, fn in (("inputs", lambda: D.load_inputs("g0p80", [m])), ("meta", lambda: D.load_meta("g0p80", [m]))):
            try:
                fn()
                res[f"{m}_{name}"] = "READ (FAIL)"
            except S.HeldOutRefused:
                res[f"{m}_{name}"] = "refused"
    return dict(results=res, **{"pass": all(v == "refused" for v in res.values())})


def g4_labels_and_order() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in TABLES:
        x, y, g = D.load_inputs(t)
        m = D.load_meta(t)
        bits = np.array([S.order_bits(c, t) for c in m["candidate_id"]])
        out[t] = dict(
            n=int(len(y)), labels_binary=bool(np.isin(y, (0, 1)).all()), meta_y_equals_array_y=bool((m["y"].to_numpy() == y).all()),
            groups_equal_meta_match=bool((m["match_id"].to_numpy() == g).all()),
            order_bits_recomputed=bool((bits[:, 0] == m["in_order"]).all() and (bits[:, 1] == m["out_order"]).all()),
            y_is_matched_iff_orders_equal=bool(((m["in_order"] != m["out_order"]).astype(int).to_numpy() == y).all()),
            swapped_share=float(y.mean()), in_order_share=float(bits[:, 0].mean()), out_order_share=float(bits[:, 1].mean()),
            balance_by={k: _chi2(m[k].astype(str).to_numpy(), y) for k in ("crossing_class", "status_in_gap", "d_band", "team_relation",
                                                                            "speed_band", "frame_mode")})
        o = out[t]
        o["pass"] = o["labels_binary"] and o["meta_y_equals_array_y"] and o["groups_equal_meta_match"] and o["order_bits_recomputed"] \
            and o["y_is_matched_iff_orders_equal"] and abs(o["swapped_share"] - 0.5) < 0.02
    out["recovery_conversion"] = dict(recovery_1_is_matched_to_stage3=int(S.recovery_y_to_stage3(1)) == S.MATCHED,
                                      recovery_0_is_crossed_to_stage3=int(S.recovery_y_to_stage3(0)) == S.SWAPPED)
    out["pass"] = all(out[t]["pass"] for t in TABLES) and all(out["recovery_conversion"].values())
    return out


def _chi2(col: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(col, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return dict(p_value=None)
    c, p, dof, _ = chi2_contingency(tab)
    return dict(p_value=round(float(p), 4), dof=int(dof), swapped_share={str(k): round(float(v), 4) for k, v in (tab[1] / tab.sum(1)).items()})


def g5_inputs_and_adaptation(feat_names: list[str]) -> dict[str, Any]:
    x, y, g = D.load_inputs("g0p80", None)
    fields_ok = tuple(x) == S.INPUT_FIELDS
    words = [w for w in S.FORBIDDEN_INPUT_WORDS if w not in ("match", "period")]        # "match" would flag nothing here either
    bad_feat = [n for n in feat_names if any(w in n for w in S.FORBIDDEN_INPUT_WORDS)]
    s = D.to_sample(x, 0)
    g0 = float(x["gap_s"][0])
    t_ok = np.allclose(s.t_before, -g0 / 2 + S.IN_REL, atol=1e-6) and np.allclose(s.t_after, g0 / 2 + S.OUT_REL, atol=1e-6)
    nan_ok = bool(np.array_equal(np.isnan(s.before[..., 0]), ~x["in_mask"][0]) and np.array_equal(np.isnan(s.after[..., 0]), ~x["out_mask"][0]))
    # synthetic: two players on parallel lanes 1 m apart at different speeds (no meeting point, so no tie); the straight
    # continuation must be favoured (positive margin = MATCHED) by every baseline. (A first version used two players meeting at
    # the same point at the gap centre: bidirectional costs tie exactly there, so that gate was ill-posed and was replaced.)
    from football_fatigue.m11_recovery import motion as MO
    from football_fatigue.m11_recovery.views import Sample
    tb, ta = -0.4 + S.IN_REL, 0.4 + S.OUT_REL
    A = lambda t: np.c_[2 * t, 0 * t]  # noqa: E731
    B = lambda t: np.c_[3 * t + 0.5, 0 * t + 1.0]  # noqa: E731
    syn = Sample(np.stack([A(tb), B(tb)]), np.stack([A(ta), B(ta)]), tb, ta, 0.8, 0.0, 0, ("", 0))
    pc = MO.pair_costs(MO.Kin(syn))
    margins = {b: MO.margin(pc[b]) for b in MO.BASELINES}
    sign_ok = all(v > 1e-6 for v in margins.values())
    r = dict(input_fields=list(x), input_fields_ok=fields_ok, forbidden_words_checked=list(S.FORBIDDEN_INPUT_WORDS), feature_names=feat_names,
             feature_names_with_forbidden_words=bad_feat, sample_times_ok=t_ok, nan_masks_ok=nan_ok,
             synthetic_straight_margins_favour_matched=sign_ok, synthetic_margins=margins, unused=words)
    r["pass"] = fields_ok and not bad_feat and t_ok and nan_ok and sign_ok
    return r


def g6_feature_symmetry(n: int = 400) -> dict[str, Any]:
    """Features recomputed from arrays with the outgoing (incoming) order exchanged equal the antisymmetric swap of the features."""
    x, y, g = D.load_inputs("g0p80", None)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), n, replace=False))
    sub = {k: v[idx] for k, v in x.items()}
    F, _, names = D.features(sub)
    k = D.n_perm(names)
    so = dict(sub, out_pos=sub["out_pos"][:, ::-1].copy(), out_mask=sub["out_mask"][:, ::-1].copy())
    si = dict(sub, in_pos=sub["in_pos"][:, ::-1].copy(), in_mask=sub["in_mask"][:, ::-1].copy())
    Fo, _, _ = D.features(so)
    Fi, _, _ = D.features(si)
    eo, ei = float(np.abs(Fo - D.swap_features(F, k)).max()), float(np.abs(Fi - D.swap_features(F, k)).max())
    return dict(n=n, max_err_outgoing_swap=eo, max_err_incoming_swap=ei, **{"pass": eo < 1e-8 and ei < 1e-8})


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation R, translation t with R P + t ~ Q (no reflection); RMS residual."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    t = mq - R @ mp
    return R, t, float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1))))


def g7_identity_audit(per_match: int = 20, n_matches: int = 8) -> dict[str, Any]:
    """Independent re-derivation of the label for a deterministic sample: the stored incoming tracklets are rigidly aligned to the
    raw-cache tracklets of the two provider slots (both assignments tried), the same transform is applied to the stored outgoing
    tracklets, and each is assigned to the slot whose raw tracklet it reproduces. Uses no order bit and no Stage 3 frame code."""
    from football_fatigue.intersections.catalogue import load_steps
    from football_fatigue.matching.benchmark import load_source
    from football_fatigue.window_intersections import core as W
    from football_fatigue.window_intersections.__main__ import RAW
    from football_fatigue.window_intersections.split import load_split

    sp = load_split(D.proc_root(), S.SPLIT)
    fo = FO.outer_folds(sp["train_cv"])
    pick = [sorted((m for m, f in fo.items() if f == k), key=lambda m: hashlib.sha256(f"audit|{m}".encode()).hexdigest())[0]
            for k in range(min(n_matches, FO.N_OUTER))]
    res = dict(matches=pick, checked=0, ok=0, mismatch=0, ambiguous=0, max_align_rms_m=0.0, examples=[])
    for m in pick:
        x, y, g = D.load_inputs("g0p80", [m])
        meta = D.load_meta("g0p80", [m])
        cand = pd.read_parquet(D.proc_root() / S.STAGE2 / "matches" / m / "candidates.parquet").set_index("candidate_id")
        c, mask, _e, _sm, _p = load_source(str(RAW), m)
        seg, _cnt, _dup = W.build_segments(c, mask, load_steps(str(RAW), m))
        ch = W.chain_table(seg)
        rng = np.random.default_rng(int(m))
        for i in np.sort(rng.choice(len(y), min(per_match, len(y)), replace=False)):
            r = cand.loc[meta["candidate_id"].iloc[i]]
            tc, gap = float(meta["gap_centre_s"].iloc[i]), float(meta["gap_s"].iloc[i])
            g0, g1 = tc - gap / 2, tc + gap / 2
            raw = {}
            for p_, slot, k in (("A", int(r["slot_a"]), int(r["chain_a"])), ("B", int(r["slot_b"]), int(r["chain_b"]))):
                rows = S.chain_vertices(seg, ch, k)[3]
                tv, xv, yv = c.video_time_s[rows], c.x[rows, slot], c.y[rows, slot]
                raw[p_] = {side: S.tracklet(tv, xv, yv, (g0 + S.IN_REL) if side == "in" else (g1 + S.OUT_REL), side, g0, g1)
                           for side in ("in", "out")}
            best = None
            for sig in (("A", "B"), ("B", "A")):
                P = np.concatenate([x["in_pos"][i, j][x["in_mask"][i, j]] for j in (0, 1)]).astype(np.float64)
                okm = all(np.array_equal(raw[sig[j]]["in"][1], x["in_mask"][i, j]) for j in (0, 1))
                if not okm:
                    continue
                Q = np.concatenate([raw[sig[j]]["in"][0][x["in_mask"][i, j]] for j in (0, 1)])
                R, t, rms = _kabsch(P, Q)
                if best is None or rms < best[3]:
                    best = (sig, R, t, rms)
            res["checked"] += 1
            if best is None or best[3] > 1e-3:
                res["ambiguous" if best is not None else "mismatch"] += 1
                continue
            sig, R, t, rms = best
            res["max_align_rms_m"] = max(res["max_align_rms_m"], rms)
            who_out = []
            for j in (0, 1):
                mo = x["out_mask"][i, j]
                P = x["out_pos"][i, j][mo].astype(np.float64) @ R.T + t
                err = {p_: float(np.abs(raw[p_]["out"][0][mo] - P).max()) if raw[p_]["out"][1][mo].all() else np.inf for p_ in ("A", "B")}
                good = [p_ for p_, e in err.items() if e < 1e-3]
                who_out.append(good[0] if len(good) == 1 else None)
            if None in who_out or who_out[0] == who_out[1]:
                res["ambiguous"] += 1
                continue
            y_raw = S.MATCHED if who_out[0] == sig[0] else S.SWAPPED
            if y_raw == int(y[i]):
                res["ok"] += 1
            else:
                res["mismatch"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append(dict(match=m, candidate_id=meta["candidate_id"].iloc[i], y=int(y[i]), y_raw=y_raw, align_rms_m=rms))
    res["pass"] = res["checked"] > 0 and res["ok"] == res["checked"]
    return res


def g9_lower_reconciliation() -> dict[str, Any]:
    from football_fatigue.window_intersections import candidates as C
    from football_fatigue.window_intersections.split import load_split
    sp = load_split(D.proc_root(), S.SPLIT)
    s2 = json.loads((D.proc_root() / S.STAGE2 / "candidates_summary.json").read_text(encoding="utf-8"))
    cols = ["candidate_id", "g0p40_outcome", "g0p80_outcome", "g1p60_outcome"]
    T = pd.concat([pd.read_parquet(D.proc_root() / S.STAGE2 / "matches" / m / "candidates.parquet", columns=cols) for m in sp["train_cv"]])
    lower = {g: int((T[f"g{g}_outcome"] == C.LOWER).sum()) for g in ("0p40", "0p80", "1p60")}
    union = int((T[[f"g{g}_outcome" for g in ("0p40", "0p80", "1p60")]] == C.LOWER).any(axis=1).sum())
    audit = sum(len(pd.read_parquet(D.stage3_dir() / "matches" / m / "lower_confidence_identity_audit.parquet")) for m in sp["train_cv"])
    s2_40 = {g: s2["per_gap"][g]["funnel"][C.LOWER] for g in ("0p40", "0p80", "1p60")}
    s2_train = {g: s2["per_gap"][g]["by_split"]["train_cv"]["outcomes"].get(C.LOWER, 0) for g in ("0p40", "0p80", "1p60")}
    r = dict(stage2_40_matches_per_gap=s2_40, stage2_summary_train_cv_per_gap=s2_train, recount_train_cv_per_gap=lower,
             recount_train_cv_any_gap=union, stage3_audit_rows=audit,
             explanation=("Stage 2 reported 23 LOWER_CONFIDENCE_IDENTITY at 0.8 s over 40 matches (17 train/CV + 6 held-out). "
                          "The Stage 3 audit table holds train/CV candidates that are LOWER at ANY of the three gaps (union), "
                          "hence a different number; none of them is ever built as a sample."))
    r["pass"] = lower == s2_train and union == audit
    return r
