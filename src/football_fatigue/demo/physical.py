"""M7.2 physical analytics service for the demo: coach settings x eligibility-checked observed segments of the overlay (v5).

Unit of analysis = one M7-0 safe segment (one tracklet, one tracker id, consecutive observed frames, no jump). A sample is usable only
if it is observed (never a display-only estimate), on the pitch, within ``VALID_RADIUS_M`` of a calibration keypoint (M7.2 C4 rule for
the frozen mapping), and belongs to a player / goalkeeper that is not a touchline person. Positions are the stored pitch positions,
never the display-smoothed ones. Results are level L1 (observed segment) unless a human review confirmed the identity (L2); player totals
exist only for reviewed identities. Nothing is extrapolated, and no fatigue statement is made.

M7.3: an optional motion sidecar (``motion_v1.json``) adds ``positions="h1"``: the same foot points mapped with the continuous inverse of
the same fisheye + TPS mapping (no 0.25 m grid snap). The default remains the stored positions; the plausibility gate is unchanged and is
evaluated separately for each position source and setting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from football_fatigue.analytics import coach as K

VALID_RADIUS_M = 13.0     # m72-sprint-v1/results/c4_coordinate_validity.json (frozen H0 mapping, declared rule)
# match-level plausibility gate over ALL eligible segments (m72 d2): real values are shown only if it passes
PLAUSIBLE_MAX_KMH = 36.0          # above elite sprint speeds: an artefact indicator
MAX_SHARE_OVER = 0.05             # at most 5 % of eligible segments may peak above PLAUSIBLE_MAX_KMH
MAX_JITTER_RATIO = 1.15           # median raw / smoothed distance; larger = position noise dominates distance and speed
TEAM_NAMES = {0: "Blue", 1: "White", 2: "other", -1: "unknown"}


class PhysicalIndex:
    def __init__(self, ov: dict[str, Any], motion: dict[str, Any] | None = None):
        self.ov = ov
        self.motion = motion
        c = ov["cols"]
        self.fps = float(ov.get("fps") or 25.0)
        off = ov["frame_offsets"]
        self.frame = np.repeat(np.arange(len(off) - 1), np.diff(off))
        self.x = np.array([v if v is not None else np.nan for v in c["pxm"]], float)
        self.y = np.array([v if v is not None else np.nan for v in c["pym"]], float)
        src = np.array(c.get("src", c["est"]))
        role = np.array(c["role"]); tl = np.array(c.get("tl", [0] * len(src))); kpd = np.array([v if v is not None else 1e9 for v in c["kpd"]], float)
        self.usable = (src == 0) & (np.array(c["pv"]) == 0) & (kpd <= VALID_RADIUS_M) & np.isin(role, (0, 1)) & (tl == 0) & np.isfinite(self.x)
        self.sources = {"stored": (self.x, self.y, self.usable)}
        self.sigma = None
        if motion is not None:
            if motion.get("rows") != len(src) or motion.get("match_id") != ov.get("match_id"):
                raise ValueError("motion sidecar does not belong to this overlay")
            hx = np.array([v if v is not None else np.nan for v in motion["cols"]["hx"]], float)
            hy = np.array([v if v is not None else np.nan for v in motion["cols"]["hy"]], float)
            self.sources["h1"] = (hx, hy, self.usable & np.isfinite(hx))
            self.sigma = np.array([v if v is not None else np.nan for v in motion["cols"]["sig"]], float)
        rows = defaultdict(list)
        for i, s in enumerate(c["seg"]):
            if s >= 0 and src[i] == 0:
                rows[s].append(i)
        self.rows = {s: np.array(sorted(r, key=lambda i: self.frame[i])) for s, r in rows.items()}
        self.meta = {}
        for s, r in self.rows.items():
            seg = ov["segments"][s]
            team = Counter(c["team"][i] for i in r).most_common(1)[0][0]
            rl = Counter(int(role[i]) for i in r).most_common(1)[0][0]
            self.meta[s] = {"id": s, "local": seg["local"], "trk": seg["trk"], "gid": seg["gid"], "use": seg["use"], "f0": seg["f0"], "f1": seg["f1"],
                            "team": int(team), "role": rl, "touchline": bool(tl[r].any())}

    def _src(self, positions: str):
        if positions not in self.sources:
            raise ValueError(f"positions must be one of {sorted(self.sources)}")
        return self.sources[positions]

    def plausibility(self, s: K.CoachSettings, positions: str = "stored") -> dict[str, Any]:
        """Match-level check of the real positions under these settings (cached); not validation, only an artefact screen."""
        X, Y, U = self._src(positions)
        key = (positions,) + tuple(sorted(s.as_dict().items()))
        cache = self.__dict__.setdefault("_plaus", {})
        if key not in cache:
            mx, ratio = [], []
            for sid, r in self.rows.items():
                m = self.meta[sid]
                fr = self.frame[r]
                el = K.segment_eligibility(coordinate_unit="metre", timestamps_valid=True, identity_safe=True,
                                           role_ok=m["role"] in (0, 1) and not m["touchline"], n_expected=int(fr[-1] - fr[0] + 1),
                                           n_usable=int(self.usable[r].sum()), duration_s=float((fr[-1] - fr[0]) / self.fps), s=s)
                if el["level"] == "L0":
                    continue
                met = K.analyse_segment(fr / self.fps, X[r], Y[r], s, usable=U[r])
                mx.append(met["speed"]["max_kmh"])
                if met["distance_m"] > 1:
                    ratio.append(met["raw_distance_m"] / met["distance_m"])
            reasons = []
            share = float(np.mean(np.array(mx) > PLAUSIBLE_MAX_KMH)) if mx else None
            jit = float(np.median(ratio)) if ratio else None
            if not mx:
                reasons.append("no eligible segment")
            else:
                if share > MAX_SHARE_OVER:
                    reasons.append(f"{share:.0%} of eligible segments peak above {PLAUSIBLE_MAX_KMH:g} km/h (limit {MAX_SHARE_OVER:.0%}): speed artefacts")
                if jit is not None and jit > MAX_JITTER_RATIO:
                    reasons.append(f"raw/smoothed distance ratio {jit:.2f} exceeds {MAX_JITTER_RATIO}: foot-point jitter dominates distance and speed")
            cache[key] = {"passed": not reasons, "reasons": reasons, "eligible_segments": len(mx), "positions": positions,
                          "share_over_max": None if share is None else round(share, 4), "jitter_ratio_median": None if jit is None else round(jit, 3)}
        return cache[key]

    def segments_dict(self) -> dict[int, dict[str, Any]]:
        return {s: {k: m[k] for k in ("trk", "local", "f0", "f1")} for s, m in self.meta.items()}

    def analyse(self, s: K.CoachSettings, *, t0: float | None = None, t1: float | None = None, local: int | None = None,
                team: int | None = None, segment_ids: Sequence[int] | None = None, review_state: dict[str, Any] | None = None,
                series: bool = False, diagnostic: bool = False, positions: str = "stored") -> dict[str, Any]:
        """``diagnostic`` returns the computed values even when the plausibility gate fails (labelled; never for coach display)."""
        X, Y, U = self._src(positions)
        gate = self.plausibility(s, positions)
        f0 = -1 if t0 is None else int(np.floor(t0 * self.fps))
        f1 = 10 ** 9 if t1 is None else int(np.ceil(t1 * self.fps))
        reviewed = {int(k): v for k, v in (review_state or {}).get("segments", {}).items() if v.get("status") == "confirmed"}
        out_segs, blockers = [], Counter()
        for sid, m in sorted(self.meta.items()):
            if segment_ids is not None and sid not in segment_ids:
                continue
            if local is not None and m["local"] != local:
                continue
            if team is not None and m["team"] != team:
                continue
            if m["f1"] < f0 or m["f0"] > f1:
                continue
            r = self.rows[sid]
            r = r[(self.frame[r] >= f0) & (self.frame[r] <= f1)]
            if not len(r):
                continue
            fr = self.frame[r]
            el = K.segment_eligibility(coordinate_unit="metre", timestamps_valid=True, identity_safe=True,
                                       role_ok=m["role"] in (0, 1) and not m["touchline"], n_expected=int(fr[-1] - fr[0] + 1),
                                       n_usable=int(self.usable[r].sum()), duration_s=float((fr[-1] - fr[0]) / self.fps), s=s,
                                       reviewed_same_person=sid in reviewed)
            rec = dict(m, team_name=TEAM_NAMES.get(m["team"], "unknown"), window_f=[int(fr[0]), int(fr[-1])], **el,
                       reviewed_identity=reviewed.get(sid, {}).get("identity"))
            if self.sigma is not None:      # measurement uncertainty of the foot points (pixel noise through the projection; no calibration)
                sg = self.sigma[r][np.isfinite(self.sigma[r])]
                rec["position_sigma_m"] = round(float(np.median(sg)), 3) if len(sg) else None
            blockers.update(el["blocker_codes"])
            if el["level"] in ("L1", "L2"):
                met = K.analyse_segment(fr / self.fps, X[r], Y[r], s, usable=U[r])
                if not series:
                    met.pop("speed_series_kmh")
                rec["metrics"] = met
            out_segs.append(rec)
        elig = [x for x in out_segs if "metrics" in x]
        obs_t = sum(x["metrics"]["observed_time_s"] for x in elig)
        summary = {
            "segments_in_window": len(out_segs), "eligible_segments": len(elig),
            "levels": dict(Counter(x["level"] for x in out_segs)),
            "blocker_counts": dict(blockers),
            "observed_eligible_time_s": round(obs_t, 2),
            "distance_per_observed_min_m": round(sum(x["metrics"]["distance_m"] for x in elig) / obs_t * 60, 1) if obs_t > 0 else None,
            "hsr_per_observed_min_m": round(sum(x["metrics"]["hsr_distance_m"] for x in elig) / obs_t * 60, 2) if obs_t > 0 else None,
            "sprints": sum(x["metrics"]["sprint_count"] for x in elig),
            "max_kmh": max((x["metrics"]["speed"]["max_kmh"] for x in elig), default=None),
            "note": "rates over observed eligible segment time only; unobserved time is not estimated",
        }
        identities = defaultdict(list)
        for x in elig:
            if x["level"] == "L2" and x["reviewed_identity"]:
                identities[x["reviewed_identity"]].append(x["metrics"])
        values = "valid_observed_segment_values" if gate["passed"] else ("diagnostic_values_not_valid" if diagnostic else "withheld")
        if values == "withheld":
            for x in out_segs:
                x.pop("metrics", None)
            totals = {}
            for k in ("distance_per_observed_min_m", "hsr_per_observed_min_m", "sprints", "max_kmh"):
                summary[k] = None
        else:
            totals = {k: K.aggregate(v) for k, v in identities.items()}
        return {"schema": K.RESULT_SCHEMA, "settings": s.as_dict(), "query": {"t0": t0, "t1": t1, "local": local, "team": team, "positions": positions},
                "plausibility": gate, "values": values, "segments": out_segs, "summary": summary,
                "reviewed_identity_totals": totals,
                "provenance": self.provenance(positions)}

    def provenance(self, positions: str = "stored") -> dict[str, Any]:
        cal = self.ov.get("calibration", {})
        pos = ("stored pitch positions of observed boxes (bbox bottom-centre through the frozen hybrid mapping), not display-smoothed"
               if positions == "stored" else f"M7.3 motion sidecar ({(self.motion or {}).get('schema')}): {(self.motion or {}).get('positions')}")
        return {"match_id": self.ov.get("match_id"), "overlay_schema": self.ov.get("schema"), "fps": self.fps,
                "positions": pos, "position_sources_available": sorted(self.sources),
                "coordinate_validity": f"on pitch and within {VALID_RADIUS_M:g} m of a calibration keypoint (M7.2 C4 rule)",
                "calibration_held_out": cal.get("held_out"), "team_method": (self.ov.get("teams") or {}).get("method"),
                "levels": K.LEVELS,
                "limitations": ["One match; no independent validation of positions, speeds or identities (L3 blocked).",
                                "Keypoint calibration error is not foot-point error; detection-box jitter adds noise to speed.",
                                "Thresholds are coach-selected definitions, not validated standards.",
                                "Observed-segment results are not player totals and are not fatigue."]}
