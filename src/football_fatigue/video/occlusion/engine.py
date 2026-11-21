"""Rule-based image-space occlusion events from per-frame tracks (V1 step 1).

Three event classes are kept apart:

* ``box_overlap``: two visible boxes overlap; no claim that anyone is hidden;
* ``probable_occlusion``: overlap plus evidence that the far player is hidden or degraded
  (``partial``: confidence/size drop while overlapped; ``full``: the track is lost while its
  predicted box lies behind a visible player; ``merged``: as ``full`` and the occluder's box grows);
* ``detection_loss``: a track is lost with nobody in front of it, or it leaves the frame.

The engine works offline over a processing range with a bounded look-ahead (``gap_limit`` frames for
reappearance). It never decides identity: a "new track at the predicted position" is a location
match only. Everything it cannot see (boxes removed before the cache, pixels, masks) is out of reach,
and events say so through ``uncertain`` and ``uncertainty_reasons`` instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from football_fatigue.video.occlusion import geometry as g
from football_fatigue.video.schema import Member, OcclusionEvent, SourceInfo


@dataclass(frozen=True)
class TrackObs:
    box: g.Box
    confidence: float
    detected_class: str


@dataclass(frozen=True)
class FrameInput:
    frame: int
    tracks: Mapping[int, TrackObs]
    untracked: Sequence[g.Box] = ()   # cached person boxes the tracker did not output (e.g. below its threshold)


@dataclass(frozen=True)
class EngineConfig:
    widen_fraction: float = 0.10       # horizontal widening for limbs
    overlap_ios: float = 0.20          # box_overlap edge
    strong_ios: float = 0.50           # strong overlap edge
    min_overlap_frames: int = 2        # an event without hiding needs this many evidence frames
    history_frames: int = 25           # look-back for the far member's median confidence / height
    min_history_samples: int = 5
    conf_drop: float = 0.15
    height_shrink: float = 0.85        # far height below this share of its median = degraded
    hidden_cover: float = 0.50         # predicted box covered by a visible box at loss onset
    clear_cover: float = 0.20          # below this for every visible box = nobody in front
    onset_frames: int = 3              # loss onset is judged on the first missing frames
    left_frame_outside: float = 0.50
    gap_limit: int = 25                # frames a lost track is followed (1 s at 25 fps)
    min_loss_frames: int = 2           # isolated losses shorter than this are not events
    merge_area_ratio: float = 1.40     # occluder area jump at loss onset
    reappear_distance_heights: float = 1.0
    untracked_iou: float = 0.30
    close_after_quiet: int = 5         # frames without evidence before an event closes
    # --- added in engine 0.2.0; the defaults reproduce Step 1 (profile "step1") ---
    grouping: str = "component"        # "component" (Step 1: connected groups) | "pair" (one event per player pair)
    prediction: str = "cv"             # "cv" (Step 1: constant velocity, unbounded) | "damped" (median velocity, capped, then held)
    cv_full_frames: int = 5            # damped: frames of velocity extrapolation before holding the position
    max_step_heights: float = 0.25     # damped: per-frame displacement cap, in the track's own box heights
    use_play_area: bool = False        # suppress events involving tracks outside the camera's play-area polygon
    play_area_margin_heights: float = 1.5
    off_pitch_share: float = 0.7
    pair_switch_cover: float = 0.2     # pair grouping: cover needed to re-attribute a hidden player to a new occluder

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


PROFILES = {
    "step1": EngineConfig(),
    "v2": EngineConfig(grouping="pair", prediction="damped", use_play_area=True, pair_switch_cover=0.5),
}


@dataclass
class _Loss:
    track_id: int
    start: int                     # first missing frame
    end: int                       # last followed missing frame
    predicted: dict[int, g.Box]
    kind: str                      # hidden | isolated_miss | left_frame | near_neighbour_uncertain
    onset_cover: float
    occluder: int | None
    outcome: str                   # reappeared_same_track | new_track_at_prediction | not_reappeared | truncated
    reappear_frame: int | None
    new_track_id: int | None
    merged: bool = False
    area_ratio: float | None = None
    untracked_frames: int = 0


@dataclass
class _Open:
    seq: int
    start: int
    last_evidence: int
    nodes: set[int]
    all_nodes: set[int] = field(default_factory=set)
    frames: dict[int, dict[str, Any]] = field(default_factory=dict)
    quiet: int = 0
    parent_seq: int | None = None
    absorbed: list[int] = field(default_factory=list)


@dataclass
class EngineResult:
    events: list[OcclusionEvent]
    frame_rows: list[dict[str, Any]]
    stats: dict[str, Any]
    suppressed: list[OcclusionEvent] = field(default_factory=list)   # kept for review, never mixed with events
    suppressed_rows: list[dict[str, Any]] = field(default_factory=list)


def _bottom_centre(b: g.Box) -> list[float]:
    return [round((b[0] + b[2]) / 2.0, 2), round(b[3], 2)]


def _round_box(b: g.Box) -> list[float]:
    return [round(v, 2) for v in b]


class OcclusionEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _check_frames(frames: Sequence[FrameInput]) -> None:
        if not frames:
            raise ValueError("no frames")
        for a, b in zip(frames, frames[1:]):
            if b.frame != a.frame + 1:
                raise ValueError(f"frames must be contiguous; {a.frame} is followed by {b.frame}")

    # ------------------------------------------------------------------ losses
    def _losses(self, frames: Sequence[FrameInput], width: int, height: int) -> list[_Loss]:
        c = self.config
        by_frame = {f.frame: f for f in frames}
        first, last = frames[0].frame, frames[-1].frame
        history: dict[int, list[tuple[int, g.Box]]] = {}
        births: dict[int, tuple[int, g.Box]] = {}
        for f in frames:
            for tid, obs in f.tracks.items():
                if tid not in history:
                    births[tid] = (f.frame, obs.box)
                history.setdefault(tid, []).append((f.frame, obs.box))
        losses: list[_Loss] = []
        for tid in sorted(history):
            seen = history[tid]
            seen_frames = [s[0] for s in seen]
            for i, (fr, _box) in enumerate(seen):
                nxt = seen_frames[i + 1] if i + 1 < len(seen_frames) else None
                if nxt == fr + 1 or fr == last:
                    continue
                start = fr + 1
                limit_end = min(start + c.gap_limit - 1, last)
                end = min(nxt - 1, limit_end) if nxt is not None else limit_end
                past = seen[: i + 1]
                predicted = {m: self._predict(past, m) for m in range(start, end + 1)}
                kind, cover, occluder = self._classify_onset(predicted, by_frame, tid, width, height)
                if nxt is not None and nxt <= start + c.gap_limit:   # back within the followed gap
                    outcome, reappear, new_id = "reappeared_same_track", nxt, None
                else:
                    outcome, reappear, new_id = self._new_track_match(predicted, births, tid, start, end)
                    if outcome == "not_reappeared" and start + c.gap_limit - 1 > last:
                        outcome = "truncated"
                loss = _Loss(tid, start, end, predicted, kind, cover, occluder, outcome, reappear, new_id)
                loss.untracked_frames = sum(
                    1 for m, pb in predicted.items()
                    if any(g.iou(pb, u) >= c.untracked_iou for u in by_frame[m].untracked)
                )
                if kind == "isolated":
                    # the cache held a person box at the predicted spot in most followed frames: the detector saw the
                    # player and the tracker did not output it; otherwise nothing was cached there (detector miss,
                    # or a box removed before caching).
                    gap = end - start + 1
                    loss.kind = "tracker_gap" if loss.untracked_frames * 2 >= gap else "detector_miss"
                if kind == "hidden" and occluder is not None:
                    loss.area_ratio, loss.merged = self._occluder_growth(history.get(occluder, []), start)
                losses.append(loss)
        return losses

    def _predict(self, past: list[tuple[int, g.Box]], frame: int) -> g.Box:
        c = self.config
        if c.prediction == "cv":
            return g.predict_constant_velocity(past, frame)
        if c.prediction == "damped":
            return g.predict_damped(past, frame, c.cv_full_frames, c.max_step_heights)
        raise ValueError(f"unknown prediction {c.prediction!r}")

    def _classify_onset(self, predicted, by_frame, tid, width, height) -> tuple[str, float, int | None]:
        c = self.config
        onset = sorted(predicted)[: c.onset_frames]
        first_pred = predicted[onset[0]]
        if g.outside_fraction(first_pred, width, height) >= c.left_frame_outside:
            return "left_frame", 0.0, None
        best, best_id = 0.0, None
        for m in onset:
            pb = predicted[m]
            for oid in sorted(by_frame[m].tracks):
                if oid == tid:
                    continue
                cover = g.covered_fraction(pb, g.widen(by_frame[m].tracks[oid].box, c.widen_fraction))
                if cover > best:
                    best, best_id = cover, oid
        if best >= c.hidden_cover:
            return "hidden", best, best_id
        if best < c.clear_cover:
            return "isolated", best, None
        return "near_neighbour_uncertain", best, best_id

    def _new_track_match(self, predicted, births, tid, start, end):
        c = self.config
        candidates = []
        for nid, (bf, bbox) in births.items():
            if nid == tid or not start <= bf <= end:
                continue
            pb = predicted[bf]
            h = max(pb[3] - pb[1], 1.0)
            dist = ((((bbox[0] + bbox[2]) - (pb[0] + pb[2])) / 2) ** 2 + (bbox[3] - pb[3]) ** 2) ** 0.5
            ratio = (bbox[3] - bbox[1]) / h
            if dist <= c.reappear_distance_heights * h and 0.6 <= ratio <= 1.67:
                candidates.append((bf, dist, nid))
        if candidates:
            bf, _d, nid = min(candidates)
            return "new_track_at_prediction", bf, nid
        return "not_reappeared", None, None

    def _occluder_growth(self, occ_hist: list[tuple[int, g.Box]], start: int) -> tuple[float | None, bool]:
        before = [g.area(b) for f, b in occ_hist if start - 10 <= f <= start - 3]
        around = [g.area(b) for f, b in occ_hist if start - 1 <= f <= start + 2]
        if len(before) < 3 or not around:
            return None, False
        ratio = max(around) / statistics.median(before)
        return round(ratio, 3), ratio >= self.config.merge_area_ratio

    # ------------------------------------------------------------------ per-frame evidence
    def _degraded(self, hist: list[tuple[int, float, float]], frame: int, conf: float, hgt: float) -> list[str]:
        c = self.config
        recent = [(cf, h) for f, cf, h in hist if frame - c.history_frames <= f < frame]
        if len(recent) < c.min_history_samples:
            return []
        flags = []
        if conf <= statistics.median(r[0] for r in recent) - c.conf_drop:
            flags.append("far_degraded_conf")
        if hgt < c.height_shrink * statistics.median(r[1] for r in recent):
            flags.append("far_degraded_height")
        return flags

    def _pairs(self, f: FrameInput, track_hist) -> list[dict[str, Any]]:
        c = self.config
        ids = sorted(f.tracks, key=lambda t: (f.tracks[t].box[0], t))
        wide = {t: g.widen(f.tracks[t].box, c.widen_fraction) for t in ids}
        edges = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if wide[b][0] > wide[a][2]:
                    break
                s = g.ios(wide[a], wide[b])
                if s < c.overlap_ios:
                    continue
                ba, bb = f.tracks[a].box, f.tracks[b].box
                near, far = (a, b) if g.nearer_camera(ba, bb) == 0 else (b, a)
                fo = f.tracks[far]
                flags = self._degraded(track_hist.get(far, []), f.frame, fo.confidence, fo.box[3] - fo.box[1]) if s >= c.strong_ios else []
                lo, hi = min(a, b), max(a, b)
                edges.append({"a": lo, "b": hi, "near": near, "far": far, "iou": round(g.iou(ba, bb), 4),
                              "ios": round(s, 4), "strong": s >= c.strong_ios, "far_flags": flags})
        return edges

    # ------------------------------------------------------------------ main
    def _evidence(self, frames: Sequence[FrameInput], hidden_at: dict[int, list[_Loss]]) -> list[tuple[list, list]]:
        """Per-frame pair edges and hidden edges (identical for both grouping modes)."""
        c = self.config
        track_hist: dict[int, list[tuple[int, float, float]]] = {}
        out = []
        for f in frames:
            edges = self._pairs(f, track_hist)
            hidden_edges = []
            for loss in hidden_at.get(f.frame, []):
                pb = loss.predicted[f.frame]
                best, best_id = 0.0, None
                for oid in sorted(f.tracks):
                    cv = g.covered_fraction(pb, g.widen(f.tracks[oid].box, c.widen_fraction))
                    if cv > best:
                        best, best_id = cv, oid
                hidden_edges.append({"hidden": loss.track_id, "occluder": best_id if best >= c.clear_cover else None,
                                     "cover": round(best, 4), "box": pb, "loss": loss})
            out.append((edges, hidden_edges))
            for tid, obs in f.tracks.items():
                track_hist.setdefault(tid, []).append((f.frame, obs.confidence, obs.box[3] - obs.box[1]))
        return out

    def _lifecycle_components(self, frames, evidence) -> list[_Open]:
        """Step 1 grouping: one event per connected group of overlapping players (kept unchanged as the baseline)."""
        c = self.config
        open_events: list[_Open] = []
        closed: list[_Open] = []
        seq = 0
        for f, (edges, hidden_edges) in zip(frames, evidence):
            adj: dict[int, set[int]] = {}
            for e in edges:
                adj.setdefault(e["a"], set()).add(e["b"]); adj.setdefault(e["b"], set()).add(e["a"])
            for h in hidden_edges:
                adj.setdefault(h["hidden"], set())
                if h["occluder"] is not None:
                    adj[h["hidden"]].add(h["occluder"]); adj.setdefault(h["occluder"], set()).add(h["hidden"])
            components = []
            seen: set[int] = set()
            for node in sorted(adj):
                if node in seen:
                    continue
                stack, comp = [node], set()
                while stack:
                    n = stack.pop()
                    if n in comp:
                        continue
                    comp.add(n); stack.extend(adj[n] - comp)
                seen |= comp
                if len(comp) >= 2 or any(h["hidden"] in comp for h in hidden_edges):
                    components.append(comp)
            # Groups first; a lone hidden track (prediction drifted off its occluder) only continues an event
            # it already belongs to: it never starts or splits one.
            components.sort(key=lambda s: (len(s) == 1, min(s)))
            claimed: dict[int, int] = {}
            for comp in components:
                hits = [ev for ev in open_events if ev.nodes & comp or (len(comp) == 1 and ev.all_nodes & comp)]
                if len(comp) == 1:
                    if not hits:
                        continue
                    ev = min(hits, key=lambda e: (e.start, e.seq))
                    h = next(h for h in hidden_edges if h["hidden"] in comp)
                    row = ev.frames.setdefault(f.frame, {"level": "probable", "edges": [], "hidden": [], "boxes": {}})
                    row["level"] = "probable"
                    row["hidden"].append(h)
                    row["boxes"][h["hidden"]] = ("hidden_predicted", h["box"])
                    claimed[ev.seq] = 1
                    ev.nodes |= comp
                    ev.last_evidence, ev.quiet = f.frame, 0
                    continue
                if not hits:
                    seq += 1
                    ev = _Open(seq, f.frame, f.frame, set(comp))
                    open_events.append(ev)
                elif len(hits) == 1 and hits[0].seq in claimed:
                    seq += 1
                    ev = _Open(seq, f.frame, f.frame, set(comp), parent_seq=hits[0].seq)
                    open_events.append(ev)
                else:
                    hits.sort(key=lambda e: (e.start, e.seq))
                    ev = hits[0]
                    for other in hits[1:]:
                        if other.seq in claimed:
                            continue
                        ev.absorbed.append(other.seq)
                        ev.all_nodes |= other.all_nodes
                        for fr, row in other.frames.items():
                            ev.frames.setdefault(fr, row)
                        ev.start = min(ev.start, other.start)
                        open_events.remove(other)
                claimed[ev.seq] = 1
                ev.nodes = set(comp)
                ev.all_nodes |= comp
                ev.last_evidence = f.frame
                ev.quiet = 0
                comp_edges = [e for e in edges if e["a"] in comp and e["b"] in comp]
                comp_hidden = [h for h in hidden_edges if h["hidden"] in comp]
                level = "probable" if comp_hidden or any(e["strong"] and e["far_flags"] for e in comp_edges) else "overlap"
                boxes = {t: ("visible", f.tracks[t].box) for t in comp if t in f.tracks}
                for h in comp_hidden:
                    boxes[h["hidden"]] = ("hidden_predicted", h["box"])
                ev.frames[f.frame] = {"level": level, "edges": comp_edges, "hidden": comp_hidden, "boxes": boxes}
            for ev in list(open_events):
                if ev.seq not in claimed:
                    ev.quiet += 1
                    if ev.quiet > c.close_after_quiet:
                        open_events.remove(ev); closed.append(ev)
        closed.extend(open_events)
        return closed

    def _lifecycle_pairs(self, frames, evidence) -> list[_Open]:
        """v2 grouping: one event per player pair (overlap or hidden-behind); concurrent pairs are linked by group_id
        afterwards, so crowds no longer chain into one long event."""
        c = self.config
        open_by_key: dict[tuple[int, int], _Open] = {}
        closed: list[_Open] = []
        seq = 0
        last_occluder: dict[tuple[int, int], int] = {}   # (hidden track, loss start) -> last occluder seen
        for f, (edges, hidden_edges) in zip(frames, evidence):
            per_key: dict[tuple[int, int], dict[str, Any]] = {}
            for e in edges:
                per_key.setdefault((e["a"], e["b"]), {"edges": [], "hidden": []})["edges"].append(e)
            for h in hidden_edges:
                lk = (h["loss"].track_id, h["loss"].start)
                occ = last_occluder.get(lk, h["loss"].occluder)
                # switch occluder only when another player clearly covers the prediction (a weak cover by a
                # passer-by is not evidence that the hidden player is behind them)
                if h["occluder"] is not None and h["occluder"] != occ and h["cover"] >= c.pair_switch_cover:
                    occ = h["occluder"]
                if occ is None:
                    continue
                last_occluder[lk] = occ
                key = (min(h["hidden"], occ), max(h["hidden"], occ))
                per_key.setdefault(key, {"edges": [], "hidden": []})["hidden"].append(h)
            for key in sorted(per_key):
                data = per_key[key]
                ev = open_by_key.get(key)
                if ev is None:
                    seq += 1
                    ev = _Open(seq, f.frame, f.frame, set(key))
                    open_by_key[key] = ev
                ev.all_nodes |= set(key)
                ev.last_evidence, ev.quiet = f.frame, 0
                level = "probable" if data["hidden"] or any(e["strong"] and e["far_flags"] for e in data["edges"]) else "overlap"
                boxes = {t: ("visible", f.tracks[t].box) for t in key if t in f.tracks}
                for h in data["hidden"]:
                    boxes[h["hidden"]] = ("hidden_predicted", h["box"])
                ev.frames[f.frame] = {"level": level, "edges": data["edges"], "hidden": data["hidden"], "boxes": boxes}
            for key, ev in list(open_by_key.items()):
                if key not in per_key:
                    ev.quiet += 1
                    if ev.quiet > c.close_after_quiet:
                        closed.append(open_by_key.pop(key))
        closed.extend(open_by_key.values())
        return closed

    def _off_pitch_tracks(self, frames: Sequence[FrameInput], play_area) -> tuple[set[int], dict[str, Any]]:
        """Tracks whose foot point lies outside the play area (by more than a margin of their own box heights) in
        at least ``off_pitch_share`` of their observations."""
        import cv2
        import numpy as np

        c = self.config
        if not c.use_play_area:
            return set(), {"status": "not_used"}
        if play_area is None:
            return set(), {"status": "unavailable"}
        poly = np.asarray(play_area, dtype=np.float32).reshape(-1, 1, 2)
        outside: dict[int, list[int]] = {}
        for f in frames:
            for tid, obs in f.tracks.items():
                x1, y1, x2, y2 = obs.box
                d = cv2.pointPolygonTest(poly, ((x1 + x2) / 2.0, y2), True)
                o = outside.setdefault(tid, [0, 0])
                o[0] += d < -c.play_area_margin_heights * (y2 - y1)
                o[1] += 1
        off = {t for t, (n_out, n) in outside.items() if n_out >= c.off_pitch_share * n}
        return off, {"status": "used", "tracks": len(outside), "off_pitch_tracks": len(off)}

    @staticmethod
    def _referee_tracks(frames: Sequence[FrameInput]) -> set[int]:
        counts: dict[int, list[int]] = {}
        for f in frames:
            for tid, obs in f.tracks.items():
                k = counts.setdefault(tid, [0, 0])
                k[0] += obs.detected_class == "referee"
                k[1] += 1
        return {t for t, (r, n) in counts.items() if r * 2 > n}

    def run(self, frames: Sequence[FrameInput], source: SourceInfo, run_id: str,
            emit_range: tuple[int, int], provenance: dict[str, Any], play_area=None) -> EngineResult:
        self._check_frames(frames)
        c = self.config
        if c.grouping not in ("component", "pair"):
            raise ValueError(f"unknown grouping {c.grouping!r}")
        first, last = frames[0].frame, frames[-1].frame
        losses = self._losses(frames, source.width, source.height)
        hidden_at: dict[int, list[_Loss]] = {}
        for loss in losses:
            if loss.kind == "hidden":
                for m in range(loss.start, loss.end + 1):
                    hidden_at.setdefault(m, []).append(loss)
        evidence = self._evidence(frames, hidden_at)
        closed = (self._lifecycle_components if c.grouping == "component" else self._lifecycle_pairs)(frames, evidence)

        transient = 0
        events: list[OcclusionEvent] = []
        rows: list[dict[str, Any]] = []
        seq_to_id: dict[int, str] = {}
        built = []
        for ev in sorted(closed, key=lambda e: (e.start, e.seq)):
            n_evidence = len(ev.frames)
            any_hidden = any(r["hidden"] for r in ev.frames.values())
            if n_evidence < c.min_overlap_frames and not any_hidden:
                transient += 1
                continue
            built.append(ev)
        for ev in built:
            event, frame_rows = self._finalise_group(ev, losses, frames, source, run_id, provenance, first, last)
            seq_to_id[ev.seq] = event.event_id
            events.append(event); rows.extend(frame_rows)
        for ev, event in zip(built, events):
            if ev.parent_seq is not None:
                event.parent_event_id = seq_to_id.get(ev.parent_seq)

        hidden_in_events = {(l.track_id, l.start) for l in losses if l.kind == "hidden"}
        short_losses = 0
        for loss in losses:
            if (loss.track_id, loss.start) in hidden_in_events:
                continue
            if loss.end - loss.start + 1 < c.min_loss_frames:
                short_losses += 1
                continue
            event, frame_rows = self._finalise_loss(loss, frames, source, run_id, provenance, first, last)
            events.append(event); rows.extend(frame_rows)

        lo, hi = emit_range
        keep = {e.event_id for e in events if e.end_frame >= lo and e.start_frame <= hi}
        events = sorted((e for e in events if e.event_id in keep), key=lambda e: (e.start_frame, e.event_id))
        rows = sorted((r for r in rows if r["event_id"] in keep), key=lambda r: (r["frame"], r["event_id"]))

        suppressed: list[OcclusionEvent] = []
        play_stats: dict[str, Any] = {"status": "not_used"}
        suppressed_rows: list[dict[str, Any]] = []
        if c.grouping == "pair" or c.use_play_area:
            off_pitch, play_stats = self._off_pitch_tracks(frames, play_area)
            referees = self._referee_tracks(frames)
            kept = []
            for e in events:
                ids = {m.track_id for m in e.members}
                if ids & referees:
                    e.flags.append("involves_referee_label")
                if ids & off_pitch:
                    e.flags.append("off_pitch_member")
                    e.suppressed_reason = (f"involves track(s) {sorted(ids & off_pitch)} outside the play area "
                                           f"(off-pitch staff, bench or spectators)")
                    suppressed.append(e)
                else:
                    kept.append(e)
            events = kept
            gone = {e.event_id for e in suppressed}
            suppressed_rows = [r for r in rows if r["event_id"] in gone]
            rows = [r for r in rows if r["event_id"] not in gone]
            if c.grouping == "pair":
                self._assign_groups(events)

        counts: dict[str, int] = {}
        for e in events:
            key = f"{e.event_class}/{e.subtype}"
            counts[key] = counts.get(key, 0) + 1
        stats = {
            "processing_frames": [first, last], "emit_range": [lo, hi], "events": len(events),
            "by_class_subtype": dict(sorted(counts.items())),
            "uncertain_events": sum(e.uncertain for e in events),
            "transient_overlaps_dropped": transient, "short_isolated_losses_ignored": short_losses,
            "loss_episodes": {k: sum(1 for l in losses if l.kind == k)
                              for k in ("hidden", "detector_miss", "tracker_gap", "left_frame", "near_neighbour_uncertain")},
        }
        if c.grouping == "pair" or c.use_play_area:
            stats["play_area"] = play_stats
            stats["suppressed_off_pitch_events"] = len(suppressed)
        return EngineResult(events, rows, stats, suppressed, suppressed_rows)

    @staticmethod
    def _assign_groups(events: list[OcclusionEvent]) -> None:
        """Link occlusion/overlap events that share a track while overlapping in time (union-find)."""
        occ = [e for e in events if e.event_class != "detection_loss"]
        parent = {e.event_id: e.event_id for e in occ}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_track: dict[int, list[OcclusionEvent]] = {}
        for e in occ:
            for m in e.members:
                by_track.setdefault(m.track_id, []).append(e)
        for lst in by_track.values():
            lst.sort(key=lambda e: e.start_frame)
            for i, a in enumerate(lst):
                for b in lst[i + 1:]:
                    if b.start_frame > a.end_frame:
                        break
                    ra, rb = find(a.event_id), find(b.event_id)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        groups: dict[str, list[OcclusionEvent]] = {}
        for e in occ:
            groups.setdefault(find(e.event_id), []).append(e)
        for root, members in groups.items():
            tracks = {m.track_id for e in members for m in e.members}
            gid = "grp-" + hashlib.sha1(root.encode()).hexdigest()[:8] if len(members) > 1 else None
            for e in members:
                e.group_id = gid
                e.measurements["group_events"] = len(members)
                e.measurements["group_tracks"] = len(tracks)

    # ------------------------------------------------------------------ finalisation
    @staticmethod
    def _event_id(source: SourceInfo, cls: str, start: int, members: Iterable[int]) -> str:
        key = f"{source.source_id}|{cls}|{start}|{','.join(str(m) for m in sorted(members))}"
        return f"oe-{source.source_id}-{start:06d}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"

    def _truncation(self, start: int, end: int, first: int, last: int) -> str | None:
        if start <= first:
            return "processing_start"
        if end >= last - self.config.close_after_quiet:
            return "processing_end"
        return None

    def _finalise_group(self, ev: _Open, losses, frames, source, run_id, provenance, first, last):
        c = self.config
        frames_sorted = sorted(ev.frames)
        start, end = frames_sorted[0], frames_sorted[-1]
        probable = any(r["level"] == "probable" for r in ev.frames.values())
        hidden_losses = {}
        for r in ev.frames.values():
            for h in r["hidden"]:
                hidden_losses[(h["loss"].track_id, h["loss"].start)] = h["loss"]
        hidden_list = sorted(hidden_losses.values(), key=lambda l: (l.start, l.track_id))
        far_flags = sorted({fl for r in ev.frames.values() for e in r["edges"] for fl in e["far_flags"]})
        if any(l.merged for l in hidden_list):
            cls, subtype = "probable_occlusion", "merged"
        elif hidden_list:
            cls, subtype = "probable_occlusion", "full"
        elif probable:
            cls, subtype = "probable_occlusion", "partial"
        else:
            cls, subtype = "box_overlap", "overlap"
        # members
        near_count: dict[int, int] = {}
        far_count: dict[int, int] = {}
        present: dict[int, list[int]] = {}
        max_ios = max_iou = 0.0
        max_group = 0
        peak, peak_val = start, -1.0
        conf_drops = []
        for fr in frames_sorted:
            r = ev.frames[fr]
            max_group = max(max_group, len(r["boxes"]))
            val = 0.0
            for e in r["edges"]:
                max_ios, max_iou = max(max_ios, e["ios"]), max(max_iou, e["iou"])
                val = max(val, e["ios"])
                if e["strong"]:
                    near_count[e["near"]] = near_count.get(e["near"], 0) + 1
                    far_count[e["far"]] = far_count.get(e["far"], 0) + 1
            for h in r["hidden"]:
                val = max(val, 1.0 + h["cover"])
            if val > peak_val:
                peak, peak_val = fr, val
            for t in r["boxes"]:
                present.setdefault(t, []).append(fr)
        members = []
        hidden_by_track = {}
        for l in hidden_list:
            hidden_by_track.setdefault(l.track_id, []).append(l)
        for t in sorted(present):
            hidden = [[max(l.start, start), min(l.end, end)] for l in hidden_by_track.get(t, [])]
            if t in hidden_by_track:
                role = "far"
            elif near_count.get(t, 0) > far_count.get(t, 0):
                role = "near"
            elif far_count.get(t, 0) > near_count.get(t, 0):
                role = "far"
            else:
                role = "involved"
            outcome = hidden_by_track[t][-1].outcome if t in hidden_by_track else None
            members.append(Member(t, role, present[t][0], present[t][-1], hidden, outcome))
        evidence = {"ios>=0.2"}
        if max_ios >= c.strong_ios:
            evidence.add("ios>=0.5")
        evidence |= set(far_flags)
        reasons = []
        for l in hidden_list:
            evidence.add("hidden_behind_neighbour")
            if l.merged:
                evidence.add("occluder_box_grew")
            evidence.add(l.outcome)
            if l.untracked_frames:
                evidence.add("untracked_detection_in_gap")
            if l.outcome == "not_reappeared":
                reasons.append(f"track {l.track_id} did not reappear within {c.gap_limit} frames (may have left view or been lost)")
            if l.outcome == "new_track_at_prediction":
                reasons.append(f"track {l.track_id} reappearance is a location match with new track {l.new_track_id}, not an identity check")
        if max_group >= 3:
            evidence.add("group_size>=3")
        if subtype == "partial":
            reasons.append("partial occlusion inferred from a confidence/height drop of the far box only")
        truncated = self._truncation(start, end, first, last)
        if truncated:
            reasons.append(f"event touches the {truncated.replace('_', ' ')}")
        if cls == "probable_occlusion":
            strong_hidden = any(l.onset_cover >= 0.8 and l.outcome in ("reappeared_same_track", "new_track_at_prediction")
                                for l in hidden_list)
            level = "low" if reasons else ("high" if strong_hidden else "medium")
        else:
            level = "high" if max_ios >= c.strong_ios and len(frames_sorted) >= 5 else "medium"
            if truncated:
                level = "low"
        area_ratios = [l.area_ratio for l in hidden_list if l.area_ratio is not None]
        peak_boxes = [b for _s, b in ev.frames[peak]["boxes"].values()]
        event_id = self._event_id(source, cls, start, present)
        measurements = {
            "max_iou": round(max_iou, 4), "max_ios_widened": round(max_ios, 4),
            "evidence_frames": len(frames_sorted), "duration_frames": end - start + 1,
            "hidden_frames": sum(h[1] - h[0] + 1 for m in members for h in m.hidden),
            "max_onset_cover": round(max((l.onset_cover for l in hidden_list), default=0.0), 4),
            "max_occluder_area_ratio": max(area_ratios) if area_ratios else None,
            "max_group_size": max_group,
            "absorbed_groups": len(ev.absorbed),
        }
        event = OcclusionEvent(
            event_id=event_id, run_id=run_id, source=source, event_class=cls, subtype=subtype,
            start_frame=start, end_frame=end, start_pts_s=source.pts(start), end_pts_s=source.pts(end),
            peak_frame=peak, members=members, parent_event_id=None, measurements=measurements,
            evidence=sorted(evidence), evidence_level=level, uncertain=bool(reasons), uncertainty_reasons=reasons,
            pixel_xy=_bottom_centre(g.union_box(peak_boxes)),
            pitch={"status": "not_computed", "x_m": None, "y_m": None, "sigma_m": None, "calibration_id": None},
            truncated=truncated, provenance=provenance,
        )
        rows = []
        for fr in frames_sorted:
            r = ev.frames[fr]
            rows.append({
                "event_id": event_id, "frame": fr, "level": r["level"],
                "boxes": {str(t): {"state": s, "box": _round_box(b)} for t, (s, b) in sorted(r["boxes"].items())},
                "pairs": [{k: e[k] for k in ("a", "b", "near", "far", "iou", "ios", "strong", "far_flags")} for e in r["edges"]],
                "hidden": [{"track_id": h["hidden"], "occluder": h["occluder"], "cover": h["cover"]} for h in r["hidden"]],
            })
        return event, rows

    def _finalise_loss(self, loss: _Loss, frames, source, run_id, provenance, first, last):
        start, end = loss.start, loss.end
        reasons = []
        evidence = [loss.kind, loss.outcome]
        if loss.kind == "near_neighbour_uncertain":
            reasons.append(f"predicted box is {loss.onset_cover:.2f} covered by track {loss.occluder}: between the occlusion and clear thresholds")
        if loss.untracked_frames:
            evidence.append("untracked_detection_in_gap")
        if loss.outcome == "new_track_at_prediction":
            reasons.append(f"reappearance is a location match with new track {loss.new_track_id}, not an identity check")
        truncated = self._truncation(start, end, first, last)
        if truncated:
            reasons.append(f"event touches the {truncated.replace('_', ' ')}")
        event_id = self._event_id(source, "detection_loss", start, [loss.track_id])
        member = Member(loss.track_id, "lost", start, end, [[start, end]], loss.outcome)
        event = OcclusionEvent(
            event_id=event_id, run_id=run_id, source=source, event_class="detection_loss", subtype=loss.kind,
            start_frame=start, end_frame=end, start_pts_s=source.pts(start), end_pts_s=source.pts(end),
            peak_frame=start, members=[member], parent_event_id=None,
            measurements={"duration_frames": end - start + 1, "onset_cover": round(loss.onset_cover, 4),
                          "nearest_track": loss.occluder, "untracked_detection_frames": loss.untracked_frames},
            evidence=sorted(set(evidence)), evidence_level="low" if reasons else "medium",
            uncertain=bool(reasons), uncertainty_reasons=reasons,
            pixel_xy=_bottom_centre(loss.predicted[start]),
            pitch={"status": "not_computed", "x_m": None, "y_m": None, "sigma_m": None, "calibration_id": None},
            truncated=truncated, provenance=provenance,
        )
        rows = [{"event_id": event_id, "frame": m, "level": "loss",
                 "boxes": {str(loss.track_id): {"state": "lost_predicted", "box": _round_box(loss.predicted[m])}},
                 "pairs": [], "hidden": []} for m in range(start, end + 1)]
        return event, rows
