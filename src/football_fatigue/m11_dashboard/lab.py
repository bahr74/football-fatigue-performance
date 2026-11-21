"""M11 Trajectory Recovery Lab: scenario loading, live inference and evaluation logic (no Streamlit; used by ``lab_app.py``).

Truth separation. :func:`make` returns two objects:

- :class:`Public`: everything the models (and the viewer, before reveal) may see. It holds the two incoming tracklets, labelled
  "Player 1" / "Player 2", and the two outgoing tracklets, labelled neutrally "Track X" / "Track Y". The outgoing order is drawn
  from the scenario id, so Track X is Player 1 in about half of the scenarios. It also holds the non-participating players, the
  hidden interval and the parameters.
- :class:`Truth`: the pairing, the provider identities of both players and their clean paths inside the hidden interval. It is
  shown only after "Reveal ground truth" and never passed to inference.

Inference (:func:`run_inference`) calls the Milestone 1 ``m11_infer.Predictor`` of the fold that never saw the match, on the
``Public`` tracklets, every time it is requested. Saved Milestone 2 probabilities are only a cross-check.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_benchmark import scenario_eval as E
from football_fatigue.m11_benchmark import scenarios as SC
from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P

IN_LABELS = ("Player 1", "Player 2")
OUT_LABELS = ("Track X", "Track Y")
FRAME_HZ = 15.0

EVIDENCE = {
    "checkpoints": "Existing trained checkpoints (Stage 4C, seed 0; GRU C and HGB C), loaded read-only and SHA-256 checked.",
    "unseen": "Unseen-match evaluation: each development match is scored by the outer-fold models trained on the other 28 matches.",
    "synthetic": "Synthetic interruptions of real FIFA provider trajectories: the occlusion is simulated by hiding an interval; "
                 "it is not a real video occlusion.",
    "unsupported": "Unsupported / out-of-distribution inputs (gap < 0.4 s or > 1.6 s, declared noise > 5 cm, too few points): "
                   "the policy abstains; probabilities are diagnostic only.",
    "heldout": "Frozen final held-out evaluation (8 matches): NOT run. No figure in this dashboard comes from held-out data.",
}


def development_matches() -> list[str]:
    return B.train_cv_matches()


# ---------------------------------------------------------------- public / truth

@dataclass
class Public:
    scenario_id: str
    match_id: str
    fold: int
    candidate_id: str
    factor: str
    level: str
    params: dict[str, Any]
    valid: bool
    invalid_reason: str
    period: int | None
    label_source: str
    g0: float | None = None
    g1: float | None = None
    incoming: list[A.Track] = field(default_factory=list)       # Player 1, Player 2
    outgoing: list[A.Track] = field(default_factory=list)       # Track X, Track Y
    context_t: np.ndarray | None = None                         # frame times (absolute s)
    context_xy: list[np.ndarray] = field(default_factory=list)  # per frame: (k, 2) other players
    source: str = "fifa"                                        # fifa | json

    @property
    def declared_noise_m(self) -> float:
        return float(self.params.get("noise", 0.0))

    def tracks(self) -> dict[str, A.Track]:
        return {**dict(zip(IN_LABELS, self.incoming)), **dict(zip(OUT_LABELS, self.outgoing))}

    def input_json(self) -> dict[str, Any]:
        """Model-visible content only (``m11-trajectory-input-v1``, readable by ``m11_infer predict`` and the JSON tab)."""
        return dict(schema="m11-trajectory-input-v1",
                    source=dict(dataset="FIFA WC 2022 (PFF), development match", match_id=self.match_id, candidate_id=self.candidate_id,
                                scenario_id=self.scenario_id, factor=self.factor, level=self.level, params=self.params, split="train_cv"),
                    units=dict(t="s (provider video time)", xy="m, provider pitch frame (origin centre)"),
                    incoming=[t.to_dict() for t in self.incoming], outgoing=[t.to_dict() for t in self.outgoing],
                    gap=dict(start_s=self.g0, end_s=self.g1), noise_sigma_m=self.declared_noise_m)


@dataclass
class Truth:
    y: int
    pairing: str                       # MATCHED / SWAPPED
    mapping: dict[str, str]            # {"Player 1": "Track ?", "Player 2": "Track ?"}
    players: dict[str, dict[str, Any]]  # Player k -> provider identity (jersey, team side, player id)
    hidden_t: np.ndarray                # absolute times inside [g0, g1]
    hidden_xy: dict[str, np.ndarray]    # Player k -> (n, 2) clean provider path in the hidden interval

    def to_json(self) -> dict[str, Any]:
        return dict(y=self.y, pairing=self.pairing, mapping=self.mapping, players=self.players)


def mapping_for(pairing: str) -> dict[str, str]:
    pr = P.PAIRS[pairing]
    return {IN_LABELS[i]: OUT_LABELS[int(pr[f"in{i}"][-1])] for i in range(2)}


# ---------------------------------------------------------------- encounters

@lru_cache(maxsize=4)
def m2_results() -> pd.DataFrame:
    return E.load_results()


def encounter_table(md: SC.MatchData) -> pd.DataFrame:
    """The Milestone 2 encounters of a match (seeded selection, independent of model output) with public descriptors only."""
    enc = SC.select_encounters(md.meta, md.match_id)
    fam = m2_results()
    fam = fam[(fam["match_id"] == md.match_id) & (fam["factor"] == "baseline")].set_index("candidate_id")
    rows = []
    for k, e in enc.iterrows():
        cid = str(e["candidate_id"])
        t = float(md.cand.loc[cid, "t_star"])
        f = fam.loc[cid] if cid in fam.index else None
        rows.append(dict(n=k + 1, candidate_id=cid, period=int(e["period"]), clock=_clock(md, int(e["period"]), t),
                         closest_m=round(float(e["d_min"]), 2), crossing=str(e["crossing_class"]).replace("_", " "),
                         teams=str(e["team_relation"]).replace("_", " "),
                         movement=None if f is None else str(f["fam_heading_relation"]).replace("_", " "),
                         in_gap=str(e["status_in_gap"]).replace("_", " ")))
    return pd.DataFrame(rows)


# Fields that may appear in a pre-reveal encounter label. crossing / movement / closest distance use the clean tracks inside or on
# both sides of the hidden interval and can disclose the pairing to a human tester (audit 2026-11-19, D1), so they are reveal-only.
PRE_REVEAL_LABEL_FIELDS = ("n", "clock", "teams")
REVEAL_ONLY_FIELDS = ("crossing", "movement", "closest_m", "in_gap")


def encounter_label(r: Any) -> str:
    """Pre-reveal dropdown label built from public fields only (``PRE_REVEAL_LABEL_FIELDS``)."""
    get = r.get if isinstance(r, dict) else (lambda k: getattr(r, k))
    return f"#{get('n')} · {get('clock')} · {get('teams')}"


def _clock(md: SC.MatchData, period: int, t: float) -> str:
    t0 = md.periods[period][2]
    s = max(0.0, t - t0) + (45 * 60 if period == 2 else 0)
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def pick(n: int, seed: int) -> int:
    """Reproducible random index (0..n-1) for a seed."""
    return int(np.random.default_rng(seed).integers(n))


def fresh_seed() -> int:
    """A truly random seed (OS entropy), shown to the user so the draw can be reproduced."""
    return int(np.random.SeedSequence().entropy % 1_000_000_000)


# ---------------------------------------------------------------- scenario

def make(md: SC.MatchData, candidate_id: str, *, factor: str = "baseline", level: Any = "base", custom: dict[str, Any] | None = None,
         seed: int = 0) -> tuple[Public, Truth | None]:
    """Build one scenario through the Milestone 2 generator (``custom``: any parameter combination with ``seed``)."""
    enc = md.meta.set_index("candidate_id", drop=False).loc[candidate_id]
    if custom is not None:
        sc = SC.build_custom(md, enc, custom, seed)
    else:
        lv = SC.parse_level(factor, level) if isinstance(level, str) and factor != "baseline" else level
        sc = SC.build(md, enc, factor, lv)
    r = md.cand.loc[candidate_id]
    pub = Public(sc.scenario_id, md.match_id, md.fold, candidate_id, sc.factor, sc.level, dict(sc.params), sc.valid, sc.invalid_reason,
                 int(r["period"]), sc.label_source)
    if not sc.valid:
        return pub, None
    pub.g0, pub.g1 = sc.g0, sc.g1
    pub.incoming, pub.outgoing = list(sc.incoming), list(sc.outgoing)
    _context(md, r, pub)
    pairing = P.SWAPPED if sc.y else P.MATCHED
    slot = {"A": int(r["slot_a"]), "B": int(r["slot_b"])}
    players = {}
    for lab, who in zip(IN_LABELS, sc.who_in):
        s = slot[who]
        players[lab] = dict(provider_label=who, jersey=_item(md.c.slot_jersey[s]), team_side=_item(md.c.slot_side[s]),
                            player_id=_item(md.c.slot_player_id[s]), slot=s)
    hm = sc.hidden_mask[0]
    truth = Truth(int(sc.y), pairing, mapping_for(pairing), players, sc.g0 + np.arange(SC.HIDDEN_LEN)[hm] * SC.DT,
                  {lab: np.asarray(sc.hidden[i][hm], float) for i, lab in enumerate(IN_LABELS)})
    return pub, truth


def _item(v):
    return v.item() if hasattr(v, "item") else v


def _context(md: SC.MatchData, r: pd.Series, pub: Public) -> None:
    """Positions of the players who are not in the encounter (both participants' slots are excluded in every frame)."""
    rows_p, tp, _a, _b = md.periods[int(r["period"])]
    tt = [tr.t for tr in (*pub.incoming, *pub.outgoing) if len(tr.t)]
    lo, hi = min(t[0] for t in tt), max(t[-1] for t in tt)
    times = np.arange(lo, hi + 1e-9, 1 / FRAME_HZ)
    k = np.clip(np.searchsorted(tp, times), 0, len(tp) - 1)
    rows = rows_p[k]
    keep = np.ones(md.c.x.shape[1], bool)
    keep[[int(r["slot_a"]), int(r["slot_b"])]] = False
    xy = []
    for row in rows:
        x, y = md.c.x[row, keep], md.c.y[row, keep]
        ok = np.isfinite(x) & np.isfinite(y)
        xy.append(np.c_[x[ok], y[ok]].astype(np.float32))
    pub.context_t, pub.context_xy = times, xy


def from_json(d: dict[str, Any]) -> Public:
    """External / manually prepared input (four timestamped metric tracklets). No context players, no truth."""
    inc, out, g0, g1 = A.tracks_from_json(d)
    for i, tr in enumerate((*inc, *out)):
        tr.validate(f"tracklet {i}")
    src = d.get("source") or {}
    noise = d.get("noise_sigma_m")
    params = dict(gap=round(g1 - g0, 6), shift=None, noise=float(noise) if noise is not None else 0.0, dropout="as given", context=None)
    return Public(str(src.get("scenario_id", "external")), str(src.get("match_id", "external")), -1, str(src.get("candidate_id", "-")),
                  "external", "json", params, True, "", None, "external (no verified label)", g0, g1, inc, out, source="json")


# ---------------------------------------------------------------- inference

def run_inference(pub: Public, predictor: P.Predictor, noise_sigma_m: float | None = None, operating_point: str = "error_05") -> dict[str, Any]:
    """Run both checkpoints and the routing policy on the public tracklets (never on the truth). Errors are returned, not hidden."""
    t0 = time.perf_counter()
    noise = pub.declared_noise_m if noise_sigma_m is None else noise_sigma_m
    out: dict[str, Any] = dict(ran_at=time.strftime("%H:%M:%S"), fold=predictor.fold, checkpoints=predictor.checkpoints(),
                               declared_noise_m=noise, operating_point=operating_point)
    try:
        pred = predictor.predict(pub.incoming, pub.outgoing, pub.g0, pub.g1, noise_sigma_m=noise, operating_point=operating_point)
    except A.InputError as e:
        out.update(error=str(e), runtime_ms=round(1000 * (time.perf_counter() - t0), 1))
        return out
    d = pred.to_dict()
    out.update(prediction=d, runtime_ms=round(1000 * (time.perf_counter() - t0), 1))
    out["models"] = {m: model_view(m, d["probabilities"].get(m), operating_point, d) for m in B.MODELS}
    out["routed_mapping"] = mapping_for(d["predicted_pairing"]) if d["predicted_pairing"] else None
    return out


def model_view(model: str, p: float | None, operating_point: str, pred: dict[str, Any]) -> dict[str, Any]:
    if p is None:
        return dict(model=model, available=False)
    pairing, dec, abst, conf = P.decide(p, B.THRESHOLDS[model][operating_point])
    return dict(model=model, available=True, p_swapped=p, p_matched=1 - p, pairing=pairing, mapping=mapping_for(pairing), confidence=conf,
                threshold=B.THRESHOLDS[model][operating_point], clears_threshold=not abst, routed=pred["model"] == model,
                diagnostic_only=pred["support"] == "unsupported" or pred["model"] != model)


def evaluate(result: dict[str, Any], truth: Truth) -> dict[str, Any]:
    """Compare each model and the routed decision with the truth (called only after reveal)."""
    if "prediction" not in result:
        return dict(error=result.get("error"))
    d = result["prediction"]
    ev = {m: dict(correct=v["pairing"] == truth.pairing) for m, v in result["models"].items() if v.get("available")}
    if d["model"] is None:
        outcome = "abstain_unsupported"
    elif d["abstained"]:
        outcome = "abstain_low_confidence"
    else:
        outcome = "accepted_correct" if d["predicted_pairing"] == truth.pairing else "accepted_wrong"
    lean = d["predicted_pairing"]
    return dict(models=ev, outcome=outcome,
                abstention_avoided_error=None if outcome != "abstain_low_confidence" else lean != truth.pairing)


def saved_crosscheck(pub: Public, result: dict[str, Any]) -> dict[str, Any] | None:
    """The Milestone 2 stored probabilities of the same scenario (declared variants only), with the difference to the live run."""
    if pub.source != "fifa" or pub.factor == "custom" or "prediction" not in result:
        return None
    df = m2_results()
    row = df[df["scenario_id"] == pub.scenario_id]
    if not len(row):
        return None
    r = row.iloc[0]
    live = result["prediction"]["probabilities"]
    out = dict(scenario_id=pub.scenario_id, outcome_saved=r["outcome"])
    for m, col in (("gru_c", "p_gru"), ("hgb_c", "p_hgb")):
        s = None if not np.isfinite(r[col]) else float(r[col])
        out[m] = dict(saved=s, live=live.get(m), abs_diff=None if s is None or live.get(m) is None else abs(s - live[m]),
                      tolerance=B.tolerance(m, "cpu"))
    return out


# ---------------------------------------------------------------- Milestone 2 evaluation and gallery

def metrics_path() -> Path:
    return E.out_dir() / "summary" / "metrics_v1.json"


@lru_cache(maxsize=1)
def metrics() -> dict[str, Any]:
    return json.loads(metrics_path().read_text(encoding="utf-8"))


FACTOR_UNITS = dict(gap="gap (s)", shift="gap centre shift (s)", noise="noise σ (m)", dropout="dropout", context="context (s)")


def factor_table(factor: str) -> pd.DataFrame:
    """Saved Milestone 2 results for one factor (baseline row first). ``supported`` marks the routing-policy range."""
    tab = metrics()["by_factor"][factor]
    rows = []
    for key, t in tab.items():
        g, h, r = t["gru"], t["hgb"], t["routed"]
        rows.append({FACTOR_UNITS[factor]: "baseline" if key == "base" else key, "valid": t["valid"],
                     "GRU BA": g.get("balanced_accuracy"), "GRU 95% CI": _ci(g.get("ba_ci")), "HGB BA": h.get("balanced_accuracy"),
                     "HGB 95% CI": _ci(h.get("ba_ci")), "GRU log loss": g.get("log_loss"), "HGB log loss": h.get("log_loss"),
                     "routed model": ", ".join(k for k in r["model_used"] if k != "none") or "none (unsupported)",
                     "coverage": r["coverage"], "realised error": r["realized_error"], "accepted": r["accepted_correct"] + r["accepted_wrong"],
                     "wrong": r["accepted_wrong"], "abstain low conf.": r["abstain_low_confidence"], "unsupported": r["abstain_unsupported"],
                     "invalid": t["encounters"] - t["valid"]})
    return pd.DataFrame(rows)


def _ci(ci):
    return None if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


GALLERY = (
    ("gru_correct_primary", "GRU correct (routed to GRU, accepted)"),
    ("gru_wrong_primary", "GRU wrong (routed to GRU, accepted, wrong)"),
    ("hgb_correct_fallback", "HGB correct (routed to HGB, accepted)"),
    ("hgb_wrong_fallback", "HGB wrong (routed to HGB, accepted, wrong)"),
    ("models_disagree", "GRU and HGB disagree"),
    ("successful_abstention", "Low-confidence abstention that avoided an error"),
    ("abstention_of_correct", "Low-confidence abstention of a correct answer"),
    ("long_gap_ge_1p6", "Longer gap (1.6 / 2.0 / 2.4 s)"),
    ("noisy_ge_10cm", "Noisy trajectories (10 / 25 / 50 cm)"),
)
# Neutral names shown by default: the real names reveal the outcome (audit 2026-11-19, D2). Category order is fixed, not result-based.
GALLERY_BLIND = {k: f"Example set {chr(65 + i)}" for i, (k, _) in enumerate(GALLERY)}


@lru_cache(maxsize=1)
def category_frames() -> dict[str, pd.DataFrame]:
    df = m2_results()
    _p, sets = E.example_sets(df[df["valid"]])
    return sets


def gallery_fixed(category: str) -> list[dict[str, Any]]:
    """The saved rule-based examples (scenario-id hash order, not chosen by result)."""
    return metrics()["examples"][category]["rows"]


def gallery_random(category: str, seed: int) -> dict[str, Any]:
    d = category_frames()[category].sort_values("scenario_id").reset_index(drop=True)
    r = d.iloc[pick(len(d), seed)]
    return dict(scenario_id=r["scenario_id"], match_id=r["match_id"], candidate_id=r["candidate_id"], factor=r["factor"], level=r["level"])


# ---------------------------------------------------------------- validity, raw data, explanations

VIS_TEXT = {1: "VISIBLE", 2: "ESTIMATED", 0: "unknown"}
CONF_TEXT = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 0: "unknown"}


def validity(pub: Public, predictor: P.Predictor | None = None) -> list[dict[str, Any]]:
    """Checks that the scenario is a legitimate unseen-match test. Uses no truth (safe before reveal)."""
    from football_fatigue.m11_benchmark import folds as FO

    dev, held = development_matches(), B.heldout_matches()
    out = [dict(check="Match is one of the 32 development (train/CV) matches", ok=pub.match_id in dev, detail=f"match {pub.match_id}"),
           dict(check="Match is not one of the 8 frozen held-out matches", ok=pub.match_id not in held,
                detail="held-out matches are refused before any data is read")]
    if pub.fold >= 0:
        unseen = B.fold_matches(pub.fold)
        train = [m for m in dev if m not in unseen]
        es = FO.early_stop_matches(train, pub.fold)
        out += [dict(check=f"Fold {pub.fold} models never saw this match in training", ok=pub.match_id in unseen and pub.match_id not in train,
                     detail=f"fold {pub.fold} test matches: {', '.join(unseen)}; trained on the other {len(train)} matches "
                            f"({len(train) - len(es)} for fitting, {len(es)} for early stopping: {', '.join(es)})"),
                dict(check="Match was not used for early stopping either", ok=pub.match_id not in es, detail="")]
    if predictor is not None:
        for m, sha in predictor.checkpoints().items():
            rec = B.recorded_sha256(m, predictor.fold)
            out.append(dict(check=f"{m} checkpoint is the verified Milestone 1 file (SHA-256)", ok=rec is not None and rec == sha,
                            detail=f"{sha[:16]}…"))
    if pub.source == "fifa":
        out += [dict(check="Pairing label accepted by the Stage 2 label-quality procedure", ok=pub.valid, detail=pub.label_source),
                dict(check="Label re-verified against the raw provider positions", ok=pub.valid, detail="samples.verify_label == ok")]
    return out


def raw_window(md: SC.MatchData, pub: Public, truth: Truth, pad: float = 2.0) -> pd.DataFrame:
    """Raw provider rows of the two players from ``pad`` s before the hidden interval to ``pad`` s after it (reveal only)."""
    rows_p, tp, _a, _b = md.periods[int(pub.period)]
    rows = rows_p[(tp >= pub.g0 - pad) & (tp <= pub.g1 + pad)]
    c = md.c
    sa, sb = truth.players["Player 1"]["slot"], truth.players["Player 2"]["slot"]
    d = np.hypot(c.x[rows, sa] - c.x[rows, sb], c.y[rows, sa] - c.y[rows, sb])
    recs = []
    for lab in IN_LABELS:
        s = truth.players[lab]["slot"]
        for i, r in enumerate(rows):
            t = float(c.video_time_s[r])
            recs.append(dict(player=lab, jersey=truth.players[lab]["jersey"], frame=int(c.frame_num[r]), video_time_s=round(t, 3),
                             t_rel_s=round(t - pub.g0, 3), x_m=round(float(c.x[r, s]), 3), y_m=round(float(c.y[r, s]), 3),
                             provider_visibility=VIS_TEXT.get(int(c.visibility[r, s]), "?"),
                             provider_confidence=CONF_TEXT.get(int(c.confidence[r, s]), "?"),
                             hidden_from_model=bool(pub.g0 <= t <= pub.g1), distance_between_players_m=round(float(d[i]), 3)))
    return pd.DataFrame(recs)


def raw_source(md: SC.MatchData) -> str:
    return f"data/raw/…/{md.c.tracking_path}  (PFF FIFA World Cup 2022 tracking, {md.c.fps:g} fps, read only)"


def model_inputs(pub: Public) -> dict[str, Any]:
    """What the two models receive for this scenario: the GRU tensors and the 36 HGB motion features (no truth involved)."""
    from football_fatigue.m11_benchmark import data as D

    mi = A.build_inputs(pub.incoming, pub.outgoing, pub.g0, pub.g1)
    si, so = D.seq_tensor(mi.arrays)
    F, _m, names = D.features(mi.arrays)
    return dict(mi=mi, gru_in=si[0], gru_out=so[0], features=dict(zip(names, F[0].tolist())))


def physics(pub: Public) -> dict[str, Any]:
    """Constant-velocity reference (not a deployed model): extrapolate each incoming player forward and each outgoing track
    backward across the gap and measure the mismatch (m) for the four Player -> Track links."""
    from football_fatigue.m11_benchmark import data as D
    from football_fatigue.m11_recovery import motion as MO

    mi = A.build_inputs(pub.incoming, pub.outgoing, pub.g0, pub.g1)
    smp = D.to_sample(mi.arrays, 0)
    pair, _ctx = MO.pair_matrices(smp, MO.Kin(smp))
    c = np.asarray(pair["d_cv_bidir"], float)
    straight, crossed = c[0, 0] + c[1, 1], c[0, 1] + c[1, 0]
    return dict(cost={f"{IN_LABELS[i]} → {OUT_LABELS[j]}": round(float(c[i, j]), 3) for i in (0, 1) for j in (0, 1)},
                matched_cost=round(float(straight), 3), swapped_cost=round(float(crossed), 3),
                pairing=P.MATCHED if straight <= crossed else P.SWAPPED)


def training_history(fold: int) -> list[dict[str, Any]]:
    """Per-epoch training loss and early-stopping log loss of the GRU C fold model (seed 0, 160-epoch budget)."""
    p = B.source_dir() / "neural" / "C" / "seed0" / f"fold{fold}.json"
    return json.loads(p.read_text(encoding="utf-8"))["history"] if p.exists() else []


FAMILY_LABELS = dict(fam_crossing_class="paths crossing", fam_heading_relation="relative direction", fam_distance_trend="distance trend",
                     fam_heading_change="largest heading change", fam_speed_change="speed change",
                     fam_status_in_gap="provider status in the gap", fam_team_relation="teams")


def difficulty(pub: Public) -> list[dict[str, Any]]:
    """This encounter's movement descriptors and the Milestone 2 baseline accuracy of encounters with the same descriptor (reveal
    only: the descriptors use the clean tracks on both sides of the gap)."""
    if pub.source != "fifa":
        return []
    df = m2_results()
    me = df[(df["candidate_id"] == pub.candidate_id) & (df["factor"] == "baseline")]
    if not len(me):
        return []
    me = me.iloc[0]
    base = df[(df["factor"] == "baseline") & df["valid"]]
    out = []
    for col, lab in FAMILY_LABELS.items():
        g = base[base[col] == me[col]]
        acc = g[g["outcome"].str.startswith("accepted")]
        out.append(dict(descriptor=lab, this_encounter=str(me[col]).replace("_", " "), encounters_like_it=int(len(g)),
                        gru_accuracy=round(float(((g["p_gru"] > 0.5).astype(int) == g["y"]).mean()), 3),
                        hgb_accuracy=round(float(((g["p_hgb"] > 0.5).astype(int) == g["y"]).mean()), 3),
                        routed_error=round(float((acc["outcome"] == "accepted_wrong").mean()), 3) if len(acc) else None))
    return out
