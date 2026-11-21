"""Every catalogued interaction inside a time window (canonical event ids; nothing is discarded because another event came first).

Sources (read only): ``data/processed/m11-interaction-catalogue-v1/matches/<m>/{edges,crossings,groups,group_timeline}.parquet``.

Event semantics (``EVENT_TYPES``) are kept separate on purpose:

* a proximity edge is two players within R metres (geometry, not occlusion);
* a path crossing is two paths crossing at possibly different times; only |tA - tB| <= 0.3 s is *synchronous*;
* a simultaneous group (link 0 s) has members whose edges overlap in time, but its connected size can still exceed the number of
  players ever close at the same instant; both sizes are always reported (``n_players`` and ``max_simultaneous_players``);
* a temporally connected group (link 1 s) chains sequential contacts and is NOT a simultaneous N-player encounter;
* self-crossings (``selfcross``) involve one player and are never an association candidate;
* synthetic observation gaps exist only in the Recovery Lab; verified visual occlusion is not available in tracking data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

SYNC_S = 0.3              # catalogue ``sync_cross_s``
NEAR_SYNC_S = 1.0         # descriptive band: 0.3 < |dt| <= 1.0 s
RADII = (0.25, 0.5, 1.0, 1.5, 2.0)

EVENT_TYPES = {
    "proximity": "Two players within R metres for a contiguous interval (a catalogue edge). Geometry only, not occlusion.",
    "crossing_synchronous": "The two paths cross and both players pass the point within 0.3 s of each other.",
    "crossing_near_synchronous": "Paths cross with 0.3 s < |tA - tB| <= 1.0 s.",
    "crossing_asynchronous": "Paths cross at clearly different times (|tA - tB| > 1.0 s): the same place, not the same moment.",
    "repeated_pair": "The same two players have more than one proximity edge in the window (meet, separate, meet again).",
    "sequential_player": "One player has proximity edges with two or more different partners at different times.",
    "simultaneous_group": "Link-0 s group whose edges overlap in time; max_simultaneous_players counts players close at one instant.",
    "connected_group": "Link-1 s group: contacts chained in time. Its size is NOT a simultaneous N-player encounter.",
    "self_crossing": "One player's own path crosses an earlier part of itself (loop / return). Never a two-player event.",
    "synthetic_gap": "Observations hidden on purpose in the Recovery Lab (simulated loss, not a camera occlusion).",
    "verified_occlusion": "Real visual occlusion: NOT available in PFF tracking data (needs video and human annotation).",
}

EDGE_COLS = ["edge_id", "edge_chain_id", "period", "radius_m", "slot_a", "slot_b", "player_id_a", "player_id_b", "start", "end",
             "duration_s", "t_star", "d_min", "x_a", "y_a", "x_b", "y_b", "rel_speed_m_s", "rel_heading_deg", "team_relation",
             "status_composition", "est_frac_a", "est_frac_b", "cv_tier", "boundary_uncertain", "reliability", "reliability_reasons",
             "label_risk", "label_risk_reasons", "estimated_comotion", "n_crossings_in_edge", "n_sync_crossings_in_edge",
             "n_concurrent_other", "n_sequential_other", "group_d0p00", "group_d1p00", "order_in_chain"]


def group_players(v: Any) -> list[int]:
    """Group membership as a list of slots (the catalogue stores ``players`` as a ';'-separated string)."""
    if isinstance(v, str):
        return [int(x) for x in v.replace(",", ";").split(";") if x.strip()]
    return [int(x) for x in v]


def _table(match: str, table: str) -> pd.DataFrame:
    return C.load_table(table, [str(match)])          # guard: refuses non train/CV matches before reading


@lru_cache(maxsize=8)
def match_tables(match: str) -> dict[str, pd.DataFrame]:
    m = str(match)
    C.checked([m])
    return {k: _table(m, k) for k in ("edges", "crossings", "groups", "group_timeline", "nodes")}


def crossing_band(abs_dt: np.ndarray) -> np.ndarray:
    a = np.asarray(abs_dt, float)
    return np.where(a <= SYNC_S, "crossing_synchronous", np.where(a <= NEAR_SYNC_S, "crossing_near_synchronous", "crossing_asynchronous"))


@dataclass
class WindowEvents:
    match: str
    period: int
    t0: float
    t1: float
    radius_m: float
    edges: pd.DataFrame          # one row per canonical edge_id overlapping the window
    crossings: pd.DataFrame      # one row per crossing_id with either passage time in the window
    groups: pd.DataFrame         # link 0 s and 1 s groups overlapping the window (both sizes)
    players: pd.DataFrame        # per-slot involvement counts
    pairs: pd.DataFrame          # per-pair repeat counts
    query_ms: float

    def counts(self) -> dict[str, int]:
        g = self.groups
        return dict(proximity=len(self.edges),
                    **{k: int((self.crossings["event_type"] == k).sum()) if len(self.crossings) else 0
                       for k in ("crossing_synchronous", "crossing_near_synchronous", "crossing_asynchronous")},
                    repeated_pair=int((self.pairs["n_edges"] > 1).sum()) if len(self.pairs) else 0,
                    sequential_player=int((self.players["n_partners"] > 1).sum()) if len(self.players) else 0,
                    simultaneous_group=int(((g["link_s"] == 0) & (g["max_simultaneous_players"] >= 3)).sum()) if len(g) else 0,
                    connected_group=int(((g["link_s"] == 1) & (g["n_players"] >= 3)).sum()) if len(g) else 0)


def window_events(match: str, period: int, t0: float, t1: float, radius_m: float = 0.5, *, tables: dict[str, pd.DataFrame] | None = None,
                  slots: set[int] | None = None, teams: str | None = None, min_duration_s: float = 0.0,
                  crossing_types: tuple[str, ...] | None = None) -> WindowEvents:
    """All canonical events overlapping [t0, t1] in one period. ``tables`` lets tests pass synthetic catalogue tables."""
    if t1 < t0:
        raise ValueError("t1 < t0")
    q0 = time.perf_counter()
    T = match_tables(str(match)) if tables is None else tables
    e = T["edges"]
    e = e[(e["period"] == period) & np.isclose(e["radius_m"], radius_m) & (e["end"] >= t0) & (e["start"] <= t1)
          & (e["duration_s"] >= min_duration_s)]
    if teams in ("same", "opposite"):
        e = e[e["team_relation"].astype(str).str.startswith(teams)]
    if slots:
        e = e[e["slot_a"].isin(slots) | e["slot_b"].isin(slots)]
    e = e[[c for c in EDGE_COLS if c in e.columns]].drop_duplicates("edge_id").sort_values(["start", "slot_a", "slot_b"]).reset_index(drop=True)
    e.insert(0, "event_type", "proximity")
    e["context"] = np.where(e["n_concurrent_other"] > 0, "simultaneous", np.where(e["n_sequential_other"] > 0, "sequential", "isolated"))

    c = T["crossings"]
    c = c[(c["period"] == period) & (((c["ta"] >= t0) & (c["ta"] <= t1)) | ((c["tb"] >= t0) & (c["tb"] <= t1)))].copy()
    if slots:
        c = c[c["slot_a"].isin(slots) | c["slot_b"].isin(slots)]
    c["event_type"] = crossing_band(c["abs_dt_s"].to_numpy()) if len(c) else pd.Series(dtype=str)
    if crossing_types:
        c = c[c["event_type"].isin(crossing_types)]
    c = c.drop_duplicates("crossing_id").sort_values(["ta", "slot_a", "slot_b"]).reset_index(drop=True)
    rkey = f"within_r{radius_m:.2f}".replace(".", "p")
    if rkey in c.columns:
        c["within_radius"] = c[rkey]

    g = T["groups"]
    g = g[(g["period"] == period) & np.isclose(g["radius_m"], radius_m) & (g["end"] >= t0) & (g["start"] <= t1)].copy()
    g = _fill_connected_max(g, T, period, radius_m)
    if slots:
        g = g[g["players"].map(lambda ps: bool(set(group_players(ps)) & slots))]
    g["inflated"] = g["n_players"] > g["max_simultaneous_players"]
    g = g.sort_values(["link_s", "start"]).reset_index(drop=True)

    rows = []
    for s in sorted(set(e["slot_a"]).union(e["slot_b"])):
        es = e[(e["slot_a"] == s) | (e["slot_b"] == s)]
        partners = set(es["slot_a"]).union(es["slot_b"]) - {s}
        rows.append(dict(slot=int(s), n_edges=len(es), n_partners=len(partners), partners=sorted(int(v) for v in partners),
                         n_crossings=int(((c["slot_a"] == s) | (c["slot_b"] == s)).sum()) if len(c) else 0))
    players = pd.DataFrame(rows, columns=["slot", "n_edges", "n_partners", "partners", "n_crossings"])
    pairs = (e.groupby(["slot_a", "slot_b"]).agg(n_edges=("edge_id", "size"), n_chains=("edge_chain_id", "nunique"),
                                                   first=("start", "min"), last=("end", "max"), d_min=("d_min", "min")).reset_index()
             if len(e) else pd.DataFrame(columns=["slot_a", "slot_b", "n_edges", "n_chains", "first", "last", "d_min"]))
    return WindowEvents(str(match), int(period), float(t0), float(t1), float(radius_m), e, c, g, players, pairs,
                        round((time.perf_counter() - q0) * 1000, 2))


def _fill_connected_max(g: pd.DataFrame, T: dict[str, pd.DataFrame], period: int, radius_m: float) -> pd.DataFrame:
    """The catalogue stores ``max_simultaneous_players`` for link-0 groups only; a link-1 group's value is the maximum over the
    link-0 groups of its member edges (so transitive inflation is visible: n_players > max_simultaneous_players)."""
    one = g["link_s"] > 0
    if not one.any():
        return g
    E = T["edges"]
    E = E[(E["period"] == period) & np.isclose(E["radius_m"], radius_m)]
    G = T["groups"]
    G0 = G[(G["period"] == period) & np.isclose(G["radius_m"], radius_m) & (G["link_s"] == 0)].set_index("group_id")["max_simultaneous_players"]
    m = E.assign(ms=E["group_d0p00"].map(G0)).groupby("group_d1p00")["ms"].max()
    g.loc[one, "max_simultaneous_players"] = g.loc[one, "group_id"].map(m).to_numpy()
    return g


def explain_edge(r: pd.Series) -> list[str]:
    """Why an edge is in the catalogue and what its flags mean (plain language)."""
    out = [f"Included because the two players were within {r['radius_m']} m from {r['start']:.2f} s to {r['end']:.2f} s "
           f"(closest {r['d_min']:.2f} m at {r['t_star']:.2f} s). This is geometric proximity, not visual occlusion."]
    out.append(f"Context: {r.get('context', '?')} ({int(r['n_concurrent_other'])} overlapping and {int(r['n_sequential_other'])} "
               "sequential edges of the same players nearby).")
    out.append(f"Provider status: {r['status_composition']} (ESTIMATED fraction A {r['est_frac_a']:.2f}, B {r['est_frac_b']:.2f}); "
               "ESTIMATED is provider metadata, not a verified visibility label.")
    if bool(r.get("boundary_uncertain")):
        out.append("Boundary-uncertain: the closest distance is within 5 cm of the radius, so membership is sensitive to noise.")
    if bool(r.get("estimated_comotion")):
        out.append("Long co-motion of two ESTIMATED tracks (possible provider clone-like trace).")
    if bool(r.get("label_risk")):
        out.append(f"Label risk: {r['label_risk_reasons']} (provider identity continuity questionable).")
    if str(r.get("reliability")) != "nominal":
        out.append(f"Reduced reliability: {r['reliability_reasons']}.")
    out.append(f"Catalogue difficulty tier (label-derived, evaluation only): {r['cv_tier']}.")
    return out
