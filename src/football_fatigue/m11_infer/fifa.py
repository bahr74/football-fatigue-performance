"""Development-only source of real FIFA trajectories for the inference API (train/CV matches of the frozen split only).

For a Stage 3 sample it returns the two players' own provider chain observations (the native 15 Hz update instants, the Stage 2
geometry) cut to the context on each side of the hidden interval, in the Stage 3 incoming / outgoing order, plus the label kept
separately. Fed through ``adapter.build_inputs`` with the same ``[g0, g1]`` they reproduce the stored Stage 3 arrays exactly
(checked by ``check-adapter`` and the tests). Raw data are only read. Held-out and non-train/CV matches are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.window_intersections import samples as S

from . import adapter as A
from . import bundle as B

MARGIN_S = 1.0          # extra observations kept beyond the 2 s context so that grid interpolation is bracketed exactly


def _check_match(match_id: str | int) -> str:
    m = str(int(match_id))
    B.fold_for_match(m)             # raises for held-out and unknown matches
    return m


@dataclass
class FifaScenario:
    match_id: str
    candidate_id: str
    table: str
    incoming: list[A.Track]
    outgoing: list[A.Track]
    g0: float
    g1: float
    y: int                          # truth (0 MATCHED, 1 SWAPPED); never passed to the model
    meta: dict[str, Any]

    def input_json(self) -> dict[str, Any]:
        """Model-visible content only (no label, no player identities)."""
        return dict(schema="m11-trajectory-input-v1", source=dict(dataset="FIFA WC 2022 (PFF)", match_id=self.match_id,
                                                                  candidate_id=self.candidate_id, table=self.table, split="train_cv"),
                    units=dict(t="s (provider video time)", xy="m, provider pitch frame (origin centre)"),
                    incoming=[t.to_dict() for t in self.incoming], outgoing=[t.to_dict() for t in self.outgoing],
                    gap=dict(start_s=self.g0, end_s=self.g1), noise_sigma_m=None)

    def truth_json(self) -> dict[str, Any]:
        return dict(match_id=self.match_id, candidate_id=self.candidate_id, y=self.y, pairing="SWAPPED" if self.y else "MATCHED",
                    **{k: _py(v) for k, v in self.meta.items()})


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


META_KEEP = ("crossing_class", "team_relation", "d_min", "d_band", "speed_band", "status_in_gap", "independent", "shift_s",
             "gap_centre_s", "frame_mode", "in_order", "out_order", "period")


class MatchSource:
    """Raw provider data of one train/CV match, loaded once (read only)."""

    def __init__(self, match_id: str | int):
        from football_fatigue.intersections.catalogue import load_steps
        from football_fatigue.matching.benchmark import load_source
        from football_fatigue.window_intersections import core as W
        from football_fatigue.window_intersections.__main__ import RAW

        self.match_id = _check_match(match_id)
        proc = B.proc_root()
        c, mask, _e, _sm, _p = load_source(str(RAW), self.match_id)
        seg, _cnt, _dup = W.build_segments(c, mask, load_steps(str(RAW), self.match_id))
        self.seg, self.ch = seg, W.chain_table(seg)
        self.cand = pd.read_parquet(proc / S.STAGE2 / "matches" / self.match_id / "candidates.parquet").set_index("candidate_id")
        if set(self.cand["split"]) != {"train_cv"}:
            raise S.HeldOutRefused(f"{self.match_id}: Stage 2 rows are not all train_cv")
        self._meta: dict[str, pd.DataFrame] = {}

    def meta(self, table: str = S.PRIMARY) -> pd.DataFrame:
        if table not in S.TABLES:
            raise ValueError(table)
        if table not in self._meta:
            from football_fatigue.window_intersections import local as L

            d = B.proc_root() / "m11-stage3-samples-v1" / "matches" / self.match_id
            L.verify_match_dir(d)
            self._meta[table] = pd.read_parquet(d / f"meta_{table}.parquet")
        return self._meta[table]

    def scenario(self, row: int, table: str = S.PRIMARY) -> FifaScenario:
        """Stage 3 sample ``row`` of ``table`` as real trajectories (order and gap exactly as in Stage 3)."""
        m = self.meta(table).iloc[int(row)]
        r = self.cand.loc[m["candidate_id"]]
        g = float(S.TABLES[table][0])
        tc = float(m["gap_centre_s"])
        g0, g1 = tc - g / 2, tc + g / 2
        VA = S.chain_vertices(self.seg, self.ch, int(r["chain_a"]))[:3]
        VB = S.chain_vertices(self.seg, self.ch, int(r["chain_b"]))[:3]
        V = {"A": VA, "B": VB}
        who_in = ("B", "A") if int(m["in_order"]) else ("A", "B")
        who_out = ("B", "A") if int(m["out_order"]) else ("A", "B")
        lo, hi = g0 - A.CONTEXT_S - MARGIN_S, g1 + A.CONTEXT_S + MARGIN_S

        def cut(v, side):
            t, x, y = v
            k = ((t >= lo) & (t < g0)) if side == "in" else ((t > g1) & (t <= hi))
            return A.Track(t[k].astype(np.float64), x[k].astype(np.float64), y[k].astype(np.float64))

        inc = [cut(V[p], "in") for p in who_in]
        out = [cut(V[p], "out") for p in who_out]
        y = 0 if who_in[0] == who_out[0] else 1
        if y != int(m["y"]):
            raise RuntimeError(f"{m['candidate_id']}: rebuilt label differs from Stage 3")
        meta = {k: m[k] for k in META_KEEP if k in m.index}
        return FifaScenario(self.match_id, str(m["candidate_id"]), table, inc, out, g0, g1, y, meta)

    def stored_arrays(self, rows, table: str = S.PRIMARY) -> dict[str, np.ndarray]:
        """Stored Stage 3 model arrays of ``rows`` (through the Stage 3 adapter, which re-checks the split)."""
        x, _y, _g = S.load_model_inputs(B.proc_root() / "m11-stage3-samples-v1", table, [self.match_id])
        rows = np.asarray(rows)
        return {k: v[rows] for k, v in x.items()}


def check_adapter(match_id: str | int, limit: int | None = 50, predictor=None, seed: int = 0, log=print) -> dict[str, Any]:
    """Rebuild Stage 3 ``g0p80`` samples of one match from raw trajectories through the adapter: arrays must be identical to the
    stored ones; with a predictor, its probabilities must equal the stored out-of-fold predictions within tolerance."""
    src = MatchSource(match_id)
    meta = src.meta()
    n = len(meta)
    rows = np.arange(n) if limit is None or limit >= n else np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))
    ref = src.stored_arrays(rows)
    out = dict(match_id=src.match_id, n_checked=int(len(rows)), arrays_identical=0, arrays_different=[], rows=rows.tolist())
    ps = {k: [] for k in B.MODELS}
    for j, r in enumerate(rows):
        sc = src.scenario(int(r))
        mi = A.build_inputs(sc.incoming, sc.outgoing, sc.g0, sc.g1)
        same = mi.enough_points and all(np.array_equal(mi.arrays[k][0], ref[k][j]) for k in S.INPUT_FIELDS)
        if same:
            out["arrays_identical"] += 1
        else:
            out["arrays_different"].append(int(r))
        if predictor is not None and mi.enough_points:
            for k, v in predictor.predict_arrays(mi.arrays).items():
                ps[k].append(float(v[0]))
    if predictor is not None:
        fold = predictor.fold
        fm = B.fold_matches(fold)
        if src.match_id not in fm:
            raise ValueError(f"predictor fold {fold} has seen match {src.match_id}; use its unseen-match fold")
        offset = sum(len(S.load_model_inputs(B.proc_root() / "m11-stage3-samples-v1", S.PRIMARY, [m])[1]) for m in fm[:fm.index(src.match_id)])
        out["prediction_max_abs_diff"] = {}
        for k in B.MODELS:
            stored = B.stored_oof(k, fold)[offset + rows]
            ok = len(ps[k]) == len(rows)
            d = float(np.abs(np.asarray(ps[k]) - stored).max()) if ok else float("inf")
            flips = int(((np.asarray(ps[k]) > 0.5) != (stored > 0.5)).sum()) if ok else len(rows)
            out["prediction_max_abs_diff"][k] = d
            out.setdefault("decision_flips", {})[k] = flips
            out.setdefault("tolerance", {})[k] = B.tolerance(k, predictor.device)
        out["predictions_reproduced"] = all(out["prediction_max_abs_diff"][k] <= out["tolerance"][k] and out["decision_flips"][k] == 0
                                            for k in B.MODELS)
    out["all_identical"] = out["arrays_identical"] == len(rows)
    log(f"check-adapter {src.match_id}: {out['arrays_identical']}/{len(rows)} identical"
        + (f"; max|dp| {out['prediction_max_abs_diff']}" if predictor is not None else ""))
    return out
