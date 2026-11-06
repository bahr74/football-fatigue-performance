"""Opt-in short-gap hand-over re-linking after the identity solver (M7.3 B2).

The frozen solver (M6.1 + M6.3-A deferred association, status cap) leaves most sub-second hand-overs unjoined: one side is usually a
contact-``ambiguous`` tracklet, which the solver does not link. This layer works on the solver OUTPUT, chain by chain (a chain = one
global id), and joins the end of chain X to the start of chain Y only when:

* Y starts at most ``max_gap`` frames after X ends (chains never overlap in time);
* team, class and box size agree, and the motion residual (forward from X and backward from Y, in box heights, divided by the solver's
  own motion spread for that gap) is within ``motion_gate``;
* the colour-descriptor distance of the two ends is within ``max_appearance`` (when set);
* X and Y are each other's best candidate, and every competitor (another chain end for Y, another chain start for X) costs at least
  ``margin`` more. Anything else abstains.

Joined boxes keep their own identity status but are capped at ``provisional`` by default (``cap_status``): the join is a new, unverified
claim, so it must never create a confirmed identity. Predicted positions are never used as evidence; only observed boxes enter.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from football_fatigue.vision.identity import IdentityConfig, Observation, motion_spread

STATUS_RANK = {"confirmed": 3, "provisional": 2, "ambiguous": 1, "unknown": 0}


@dataclass(frozen=True)
class HandoverConfig:
    max_gap: int = 12             # frames between X's last box and Y's first box (0 = consecutive frames)
    motion_gate: float = 0.5      # residual / motion spread
    size_ratio: float = 1.4
    margin: float = 0.5           # cost gap to the best competitor
    max_appearance: float | None = None   # colour-descriptor distance gate (None = not used; missing descriptors then abstain)
    endpoint_frames: int = 10
    cap_status: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chains(boxes: dict, obs: Iterable[Observation]) -> dict[int, list[Observation]]:
    ch = defaultdict(list)
    for o in obs:
        ch[boxes[(o.frame, o.index)]["global_id"]].append(o)
    return {g: sorted(v, key=lambda o: o.frame) for g, v in ch.items()}


def _end(obs: list[Observation], at_end: bool, k: int):
    part = obs[-(k + 1):] if at_end else obs[:k + 1]
    f = np.array([o.frame for o in part], float)
    p = np.array([[(o.box[0] + o.box[2]) / 2, o.box[3]] for o in part])
    h = float(np.median([o.box[3] - o.box[1] for o in part]))
    if len(part) >= 2 and f[-1] > f[0]:
        A = np.c_[np.ones(len(f)), f - f[-1 if at_end else 0]]
        coef = np.linalg.lstsq(A, p, rcond=None)[0]
        pos, vel = coef[0], coef[1]
    else:
        pos, vel = p[-1 if at_end else 0], np.zeros(2)
    app = np.mean([o.appearance for o in part if o.appearance is not None], axis=0) if any(o.appearance is not None for o in part) else None
    return int(f[-1] if at_end else f[0]), pos, vel, max(1.0, h), app


def candidates(boxes: dict, obs: Iterable[Observation], solver_cfg: IdentityConfig, cfg: HandoverConfig) -> list[dict[str, Any]]:
    """Every gated X-end -> Y-start pair with its evidence (no decision yet)."""
    ch = _chains(boxes, obs)
    info = {}
    for g, v in ch.items():
        teams = Counter(boxes[(o.frame, o.index)].get("team_cluster") for o in v)
        team = teams.most_common(1)[0][0]
        cls = Counter(o.detected_class for o in v).most_common(1)[0][0]
        info[g] = {"end": _end(v, True, cfg.endpoint_frames), "start": _end(v, False, cfg.endpoint_frames), "team": team, "cls": cls,
                   "f0": v[0].frame, "f1": v[-1].frame, "n": len(v)}
    starts = sorted(info, key=lambda g: info[g]["f0"])
    f0s = np.array([info[g]["f0"] for g in starts])
    out = []
    for x, ix in info.items():
        fe, pa, va, ha, aa = ix["end"]
        lo, hi = np.searchsorted(f0s, fe + 1), np.searchsorted(f0s, fe + cfg.max_gap + 2)
        for y in starts[lo:hi]:
            if y == x:
                continue
            iy = info[y]
            fs, pb, vb, hb, ab = iy["start"]
            gap = fs - fe - 1
            dt = fs - fe
            ratio = max(ha, hb) / min(ha, hb)
            ef = np.linalg.norm(pa + va * dt - pb)
            eb = np.linalg.norm(pb - vb * dt - pa)
            r = float((ef + eb) / 2.0 / ((ha + hb) / 2.0) / motion_spread(dt, solver_cfg))
            team_conflict = bool(ix["team"] and iy["team"] and ix["team"] != iy["team"])
            app = float(np.linalg.norm(aa - ab)) if aa is not None and ab is not None else None
            gated = (not team_conflict and ix["cls"] == iy["cls"] and ratio <= cfg.size_ratio and r <= 3 * cfg.motion_gate)
            if gated:
                out.append({"from": int(x), "to": int(y), "gap": int(gap), "motion_ratio": round(r, 4), "size_ratio": round(ratio, 4),
                            "appearance_dist": None if app is None else round(app, 4), "team_known": bool(ix["team"] and iy["team"]),
                            "cost": round(r * r + 4.0 * math.log(ratio) ** 2 + (0.0 if ix["team"] and ix["team"] == iy["team"] else 0.25), 4)})
    return out


def decide(cands: list[dict[str, Any]], cfg: HandoverConfig) -> list[dict[str, Any]]:
    """Mutual-best, margin-checked decisions; returns the candidates annotated with ``decision`` and ``reason``."""
    by_from, by_to = defaultdict(list), defaultdict(list)
    for c in cands:
        by_from[c["from"]].append(c); by_to[c["to"]].append(c)
    out = []
    for c in sorted(cands, key=lambda c: c["cost"]):
        d = dict(c)
        rivals = [o["cost"] for o in by_from[c["from"]] + by_to[c["to"]] if o is not c]
        best_from = min(by_from[c["from"]], key=lambda o: o["cost"]) is c
        best_to = min(by_to[c["to"]], key=lambda o: o["cost"]) is c
        if c["motion_ratio"] > cfg.motion_gate or c["gap"] > cfg.max_gap or c["size_ratio"] > cfg.size_ratio:
            d["decision"], d["reason"] = "rejected", "gate"
        elif cfg.max_appearance is not None and (c["appearance_dist"] is None or c["appearance_dist"] > cfg.max_appearance):
            d["decision"], d["reason"] = "rejected", "appearance"
        elif not (best_from and best_to):
            d["decision"], d["reason"] = "abstained", "not_mutual_best"
        elif rivals and min(rivals) < c["cost"] + cfg.margin:
            d["decision"], d["reason"] = "abstained", "competing_candidate"
        else:
            d["decision"], d["reason"] = "linked", "mutual_best_with_margin"
        out.append(d)
    return out


def apply(boxes: dict, decisions: list[dict[str, Any]], cfg: HandoverConfig) -> tuple[dict, dict[int, int]]:
    """New box dict with joined global ids (Y's boxes take X's chain root); joined boxes capped at provisional if ``cap_status``."""
    root = {}

    def find(g):
        while g in root:
            g = root[g]
        return g
    succ, pred = set(), set()
    joined = set()
    for d in decisions:
        if d["decision"] != "linked" or d["from"] in succ or d["to"] in pred:
            continue
        a, b = find(d["from"]), find(d["to"])
        if a == b:
            continue
        root[b] = a
        succ.add(d["from"]); pred.add(d["to"])
        joined.add(d["to"])
    new = {}
    mapping = {}
    for k, v in boxes.items():
        g = v["global_id"]
        r = find(g)
        if r != g:
            mapping[g] = r
            nv = dict(v, global_id=r, association="m73_short_gap_join")
            if cfg.cap_status and STATUS_RANK.get(v["identity_status"], 0) > STATUS_RANK["provisional"]:
                nv["identity_status"] = "provisional"
            new[k] = nv
        else:
            new[k] = v
    # the root chain's own boxes keep their status (the join claim is about the later chain); only boxes reached through a join are capped
    return new, mapping
