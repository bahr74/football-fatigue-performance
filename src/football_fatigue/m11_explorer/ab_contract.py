"""Session A side of the Session A / B contract (``docs/research/m11_session_ab_interface_v1.md``): SYNTHETIC-ONLY adapter.

``request_to_scenario`` turns a ``m11-session-ab-candidate-event-v1`` request into the internal N x M scenario (pitch metres, only
``observed`` samples, outgoing tracks anonymous); ``respond`` runs the bidirectional Kalman + Hungarian-with-null solver and writes a
``m11-session-ab-reassociation-result-v1`` message; ``check_result`` verifies the contract invariants (section 4 of the doc).

This is a contract check, not an integration: no Session B code or data is read, nothing is calibrated on video, and every result
says it is advisory outside the trained domain (provider tracking coordinates).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from . import multiplayer as MP
from . import nxm_compare as NC

REQ, RES = "m11-session-ab-candidate-event-v1", "m11-session-ab-reassociation-result-v1"
TRAINED_DOMAIN = "PFF metric pitch coordinates (FIFA WC 2022 tracking); not validated on video tracking"
SIGMA_P90_ENVELOPE = 0.15          # provider position noise is centimetres; larger video noise is outside the envelope


class ContractError(ValueError):
    pass


def _track(tr: dict[str, Any]) -> dict[str, Any] | None:
    o = tr["observations"]
    st = np.asarray(o["position_status"])
    p = o.get("pitch")
    if p is None:
        return None
    t, x, y = (np.asarray(v, float) for v in (o["timestamp_s"], p["x_m"], p["y_m"]))
    k = (st == "observed") & np.isfinite(x) & np.isfinite(y)
    if k.sum() < 2:
        return None
    return dict(id=tr["tracklet_id"], t=t[k].tolist(), x=x[k].tolist(), y=y[k].tolist())


def request_to_scenario(req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if req.get("schema") != REQ:
        raise ContractError(f"schema {req.get('schema')!r} is not {REQ}")
    for o in req["outgoing_candidates"]:
        if o["identity"].get("global_id") is not None or o["identity"].get("identity_status") != "anonymous":
            raise ContractError(f"outgoing {o['tracklet_id']} carries an identity; B must not pre-fill one")
    cal = req["camera"]["calibration"]
    ins = [_track(t) for t in req["incoming"]]
    outs = [_track(t) for t in req["outgoing_candidates"]]
    sig = [s for t in req["incoming"] + req["outgoing_candidates"] for s in ((t["observations"].get("pitch") or {}).get("sigma_m") or [])]
    p90 = float(np.percentile(sig, 90)) if sig else None
    reasons = []
    if not cal.get("available"):
        reasons.append("no calibration: pitch coordinates unavailable")
    if p90 is not None and p90 > SIGMA_P90_ENVELOPE:
        reasons.append(f"position uncertainty p90 {p90:.2f} m far above the provider data the model was built on")
    if any(t is None for t in ins + outs):
        reasons.append("a tracklet has fewer than 2 observed pitch samples")
    dc = dict(calibration_ok=bool(cal.get("available")), sigma_m_p90=p90, inside_trained_envelope=not reasons, reasons=reasons)
    sc = dict(schema=MP.SCHEMA, incoming=[t for t in ins if t], outgoing=[t for t in outs if t], truth={})
    return sc, dc


def _total(cost, pairs, n, m, lam):
    return float(sum(cost[a, b] for a, b in pairs) + lam * (n - len(pairs)) + lam * (m - len(pairs)))


def respond(req: dict[str, Any], min_margin: float = 1.0) -> dict[str, Any]:
    sc, dc = request_to_scenario(req)
    ids_in, ids_out = [t["tracklet_id"] for t in req["incoming"]], [t["tracklet_id"] for t in req["outgoing_candidates"]]
    gid = {t["tracklet_id"]: t["identity"] for t in req["incoming"]}
    res: dict[str, Any] = dict(schema=RES, synthetic_example=bool(req.get("synthetic_example")), event_id=req["event_id"],
                               producer=dict(session="A", component="identity-reassociation", version="bidir-kalman-hungarian-null",
                                             trained_domain=TRAINED_DOMAIN),
                               input_contract=dict(coordinate_system="pitch_m", observed_samples_only=True, domain_check=dc),
                               candidates_scored=dict(incoming=ids_in, outgoing=ids_out), link_scores=[])
    usable = dc["calibration_ok"] and len(sc["incoming"]) == len(ids_in) and len(sc["outgoing"]) == len(ids_out) and sc["incoming"]
    lam = MP.GATE
    if usable and sc["outgoing"]:
        c = NC.motion_costs(sc)["bi"]
        cost = np.where(c <= lam, c, np.inf)
        for i, j in itertools.product(range(len(ids_in)), range(len(ids_out))):
            res["link_scores"].append(dict(incoming=ids_in[i], outgoing=ids_out[j], score_kind="kalman_bidir_d2",
                                           value=None if not np.isfinite(c[i, j]) else round(float(c[i, j]), 4), inside_gate=bool(np.isfinite(cost[i, j]))))
        pairs = NC.assign2(cost, lam, lam)
        best = _total(cost, pairs, len(ids_in), len(ids_out), lam)
        assignments = []
        for a, b in pairs:
            c2 = cost.copy()
            c2[a, b] = np.inf
            alt = _total(c2, NC.assign2(c2, lam, lam), len(ids_in), len(ids_out), lam)
            assignments.append(dict(incoming=ids_in[a], outgoing=ids_out[b], global_id=gid[ids_in[a]].get("global_id"),
                                    probability=None, margin=round(alt - best, 4)))
        abstain = bool(assignments) and min(x["margin"] for x in assignments) < min_margin
    else:
        assignments, abstain = [], True
    if abstain:
        reason = "no usable pitch input" if not usable else f"smallest assignment margin < {min_margin}"
        assignments = []
    linked_in, linked_out = {x["incoming"] for x in assignments}, {x["outgoing"] for x in assignments}
    status = "abstain" if abstain else ("assigned" if len(assignments) == len(ids_in) else "partial")
    res["decision"] = dict(status=status, solver="hungarian_with_null", assignments=assignments, calibrated=False,
                           unmatched_incoming=[dict(tracklet_id=i, reason="abstained" if abstain else "outside_gate") for i in ids_in if i not in linked_in],
                           unmatched_outgoing=[dict(tracklet_id=o, reason="abstained" if abstain else "outside_gate") for o in ids_out if o not in linked_out],
                           abstention=dict(abstained=abstain, rule=f"abstain when any chosen link has margin < {min_margin} (d2 units; "
                                                                  "UNCALIBRATED placeholder, not fitted on video)", reason=reason if abstain else None),
                           delayed=False)
    ok_ids = all(gid[x["incoming"]].get("identity_status") == "confirmed" for x in assignments)
    res["identity_effect"] = dict(propagates_global_id=bool(assignments) and ok_ids and dc["inside_trained_envelope"],
                                  reason="advisory: outside the trained envelope" if not dc["inside_trained_envelope"] else
                                  ("an incoming identity is not confirmed" if not ok_ids else "inside envelope, confirmed identities"))
    return res


def check_result(res: dict[str, Any], req: dict[str, Any] | None = None) -> list[str]:
    """Contract invariants of a result (empty list = valid)."""
    e = []
    d = res["decision"]
    ins, outs = res["candidates_scored"]["incoming"], res["candidates_scored"]["outgoing"]
    A = d["assignments"]
    if len({a["incoming"] for a in A}) != len(A) or len({a["outgoing"] for a in A}) != len(A):
        e.append("assignment is not one-to-one")
    if any(a["incoming"] not in ins or a["outgoing"] not in outs for a in A):
        e.append("assignment id not in candidates_scored")
    if not d["calibrated"] and any(a["probability"] is not None for a in A):
        e.append("probability given while uncalibrated")
    if d["status"] == "abstain" and (A or len(d["unmatched_incoming"]) != len(ins) or len(d["unmatched_outgoing"]) != len(outs)):
        e.append("abstain must have no assignments and list every candidate as unmatched")
    lin = {a["incoming"] for a in A} | {u["tracklet_id"] for u in d["unmatched_incoming"]}
    lout = {a["outgoing"] for a in A} | {u["tracklet_id"] for u in d["unmatched_outgoing"]}
    if lin != set(ins) or lout != set(outs):
        e.append("every candidate must be assigned or listed as unmatched exactly once")
    if res["identity_effect"]["propagates_global_id"] and not res["input_contract"]["domain_check"]["inside_trained_envelope"]:
        e.append("global id propagated outside the trained envelope")
    if req is not None and res["event_id"] != req["event_id"]:
        e.append("event_id not echoed")
    return e


def synthetic_request(level: int = 2, seed: int = 0, sigma_m: float = 0.05, interpolated_gap: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    """A SYNTHETIC request built from ``multiplayer.generate`` (the truth is returned separately, never placed in the request)."""
    d = MP.generate(level, seed)
    ids = {t["id"]: f"trk-{k}" for k, t in enumerate(d["incoming"] + d["outgoing"])}

    def obs(tr, status="observed"):
        n = len(tr["t"])
        t = list(tr["t"])
        x, y = list(tr["x"]), list(tr["y"])
        st = [status] * n
        return dict(frame=[int(round(v * 25)) for v in t], timestamp_s=t, pixel=None,
                    pitch=dict(x_m=x, y_m=y, sigma_m=[sigma_m] * n), position_status=st)
    inc = []
    for k, t in enumerate(d["incoming"]):
        o = obs(t)
        if interpolated_gap:                                  # B-side interpolation into the hidden interval: must be ignored by A
            t1 = o["timestamp_s"][-1]
            for s in range(1, 4):
                o["timestamp_s"].append(t1 + s / 15)
                o["frame"].append(int(round((t1 + s / 15) * 25)))
                o["pitch"]["x_m"].append(1e3)
                o["pitch"]["y_m"].append(1e3)
                o["pitch"]["sigma_m"].append(sigma_m)
                o["position_status"].append("interpolated")
        inc.append(dict(tracklet_id=ids[t["id"]], identity=dict(global_id=f"home-{k:02d}", identity_status="confirmed",
                                                                identity_confidence=0.9, source="SYNTHETIC"), observations=o))
    out = [dict(tracklet_id=ids[t["id"]], identity=dict(global_id=None, identity_status="anonymous"), observations=obs(t)) for t in d["outgoing"]]
    req = dict(schema=REQ, synthetic_example=True, event_id=f"SYN-{level}-{seed}", match=dict(match_id="SYNTHETIC", period=1, fps=25.0, clock="video_time_s"),
               camera=dict(camera_id="SYNTHETIC", calibration=dict(available=True, calibration_id="SYN", model="homography_ground_plane")),
               trigger=dict(type="simultaneous_loss", reasons=[], occlusion_candidate=dict(verified_visual_occlusion=None)),
               window=dict(), incoming=inc, outgoing_candidates=out, dormant_tracks=[], context_players=[],
               appearance=dict(available=False, reason="SYNTHETIC"), provenance=dict(created="SYNTHETIC"))
    truth = {ids[i]: (ids[o] if o is not None else None) for i, o in d["truth"].items()}
    return req, truth
