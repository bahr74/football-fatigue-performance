"""Event-level evaluation of occlusion events against SoccerTrack MOT ground truth (evaluation only).

Ground truth (GT) = SoccerTrack MOT boxes: 22 players per frame, **no referees, staff or spectators**, boxes are BePro
positions projected with template sizes. "Occlusion" in GT is therefore *geometric projection occlusion*, a proxy for
visual occlusion. GT is never an engine input; it is read here only to score saved events.

Reference episodes, per GT player pair (far = smaller box bottom y, near = the other):

* ``overlap`` tier: share of the far box covered by the near box >= 0.2 for >= 2 frames;
* ``occlusion`` tier: covered share >= 0.5 for >= 3 frames;
  gaps of <= 2 frames are merged; an episode's tier is set by its frames.

Matching: every member box of an engine event (observed or predicted) is mapped to the GT player with the highest
IoU (>= ``GT_IOU``) in that frame. An occlusion/overlap event matches an episode when they share a GT pair in a frame
within the episode's span widened by ``TIME_TOL`` frames.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_fatigue.video.occlusion import geometry as g

EVAL_VERSION = "eval-v1"
GT_IOU = 0.3
TIME_TOL = 5
DETECTABLE_MARGIN = 10
OVERLAP_COVER, OVERLAP_MIN = 0.2, 2
OCCLUSION_COVER, OCCLUSION_MIN = 0.5, 3
MERGE_GAP = 2

GT = Mapping[int, Sequence[tuple[int, g.Box]]]   # frame -> [(gt id, xyxy)]


@dataclass(frozen=True)
class Episode:
    a: int
    b: int
    start: int
    end: int
    tier: str            # "overlap" | "occlusion"
    peak_cover: float

    @property
    def key(self) -> str:
        return f"{self.a}-{self.b}@{self.start}"


def gt_episodes(gt: GT, first: int, last: int) -> list[Episode]:
    """Overlap and occlusion episodes per GT pair (an occlusion episode is also reported at the overlap tier)."""
    per_pair: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for f in range(first, last + 1):
        rows = sorted(gt.get(f, ()))
        for i, (ia, ba) in enumerate(rows):
            for ib, bb in rows[i + 1:]:
                far, near = (ba, bb) if ba[3] < bb[3] else (bb, ba)
                cover = g.covered_fraction(far, near)
                if cover >= OVERLAP_COVER:
                    per_pair[(ia, ib)][f] = cover
    out: list[Episode] = []
    for (a, b), covers in sorted(per_pair.items()):
        for tier, thr, min_len in (("overlap", OVERLAP_COVER, OVERLAP_MIN), ("occlusion", OCCLUSION_COVER, OCCLUSION_MIN)):
            frames = sorted(f for f, cv in covers.items() if cv >= thr)
            if not frames:
                continue
            start = prev = frames[0]
            for f in frames[1:] + [None]:
                if f is not None and f - prev <= MERGE_GAP + 1:
                    prev = f
                    continue
                if prev - start + 1 >= min_len:
                    peak = max(covers[x] for x in range(start, prev + 1) if x in covers)
                    out.append(Episode(a, b, start, prev, tier, round(peak, 4)))
                if f is not None:
                    start = prev = f
    return out


def map_to_gt(box: Sequence[float], frame: int, gt: GT) -> int | None:
    best, best_id = 0.0, None
    for gid, gb in gt.get(frame, ()):
        v = g.iou(tuple(box), gb)
        if v > best:
            best, best_id = v, gid
    return best_id if best >= GT_IOU else None


TrackBoxes = Mapping[int, Mapping[int, g.Box]]   # frame -> {track id: observed box}


def _tracked_gt(frame_tracks: TrackBoxes, gt: GT, first: int, last: int) -> dict[int, set[int]]:
    """GT ids with at least one tracked box (IoU >= GT_IOU) per frame."""
    out: dict[int, set[int]] = {}
    for f in range(first, last + 1):
        ids = set()
        boxes = list(frame_tracks.get(f, {}).values())
        for gid, gb in gt.get(f, ()):
            if any(g.iou(tb, gb) >= GT_IOU for tb in boxes):
                ids.add(gid)
        out[f] = ids
    return out


def track_identity(tid: int, lo: int, hi: int, frame_tracks: TrackBoxes, gt: GT) -> tuple[int | None, int]:
    """Majority GT player of a track's own observed boxes in frames ``lo..hi`` (None = not a GT player)."""
    votes: Counter = Counter()
    for f in range(lo, hi + 1):
        box = frame_tracks.get(f, {}).get(tid)
        if box is not None:
            votes[map_to_gt(box, f, gt)] += 1
    if not votes:
        return None, 0
    return votes.most_common(1)[0][0], sum(votes.values())


def _event_gt(event: dict[str, Any], rows: list[dict[str, Any]], frame_tracks: TrackBoxes, gt: GT):
    """Per event: the GT player of each member, the GT pairs per event frame, and members that are not GT players.

    A member's GT player is the majority mapping of the track's own *observed* boxes within the event span widened
    by DETECTABLE_MARGIN frames (predicted boxes are never used: they sit on the occluder by construction). A member
    whose majority is "no GT player" is a non-GT person (referee, staff, spectator, or a false detection).
    """
    lo, hi = event["start_frame"] - DETECTABLE_MARGIN, event["end_frame"] + DETECTABLE_MARGIN
    present: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        for tid in r["boxes"]:
            present[r["frame"]].add(tid)
    member_gt: dict[str, int | None] = {}
    for m in event["members"]:
        member_gt[str(m["track_id"])] = track_identity(int(m["track_id"]), lo, hi, frame_tracks, gt)[0]
    unmapped = sorted(t for t, v in member_gt.items() if v is None)
    pairs_by_frame: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for fr, tids in present.items():
        vals = sorted({member_gt[t] for t in tids if member_gt.get(t) is not None})
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                pairs_by_frame[fr].add((a, b))
    return member_gt, pairs_by_frame, unmapped


def evaluate_run(events: Sequence[dict[str, Any]], suppressed: Sequence[dict[str, Any]], rows: Iterable[dict[str, Any]],
                 frame_tracks: TrackBoxes, gt: GT, frame_range: tuple[int, int]) -> dict[str, Any]:
    first, last = frame_range
    rows_by_event: dict[str, list] = defaultdict(list)
    for r in rows:
        rows_by_event[r["event_id"]].append(r)
    episodes = [e for e in gt_episodes(gt, first, last)]
    occ_eps = [e for e in episodes if e.tier == "occlusion"]
    ovl_eps = [e for e in episodes if e.tier == "overlap"]
    tracked = _tracked_gt(frame_tracks, gt, first, last)

    def detectable(ep: Episode) -> bool:
        lo, hi = max(first, ep.start - DETECTABLE_MARGIN), min(last, ep.end + DETECTABLE_MARGIN)
        seen_a = any(ep.a in tracked[f] for f in range(lo, hi + 1))
        seen_b = any(ep.b in tracked[f] for f in range(lo, hi + 1))
        return seen_a and seen_b

    def matches(ev: dict, pairs_by_frame, eps: Sequence[Episode]) -> list[Episode]:
        out = []
        for ep in eps:
            lo, hi = ep.start - TIME_TOL, ep.end + TIME_TOL
            if ev["end_frame"] < lo or ev["start_frame"] > hi:
                continue
            if any((ep.a, ep.b) in ps for f, ps in pairs_by_frame.items() if lo <= f <= hi):
                out.append(ep)
        return out

    per_event: list[dict[str, Any]] = []
    occ_hits: dict[str, list[str]] = defaultdict(list)     # episode key -> event ids (occlusion tier)
    ovl_hits: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        rows_e = rows_by_event[ev["event_id"]]
        member_gt, pairs_by_frame, unmapped = _event_gt(ev, rows_e, frame_tracks, gt)
        rec = {"event_id": ev["event_id"], "event_class": ev["event_class"], "subtype": ev["subtype"],
               "uncertain": ev["uncertain"], "start_frame": ev["start_frame"], "end_frame": ev["end_frame"],
               "member_gt": {str(k): v for k, v in sorted(member_gt.items())}}
        if ev["event_class"] in ("box_overlap", "probable_occlusion"):
            m_occ = matches(ev, pairs_by_frame, occ_eps)
            m_ovl = matches(ev, pairs_by_frame, ovl_eps)
            for ep in m_occ:
                occ_hits[ep.key].append(ev["event_id"])
            for ep in m_ovl:
                ovl_hits[ep.key].append(ev["event_id"])
            target = m_occ if ev["event_class"] == "probable_occlusion" else m_ovl
            mapped = [v for v in member_gt.values() if v is not None]
            if unmapped and not target:
                outcome = "non_gt_person"
            elif not target and len(mapped) >= 2 and len(set(mapped)) == 1:
                outcome = "fp_duplicate_track"        # two tracks on one GT player (duplicate detection/track)
            elif target:
                outcome = "tp"
            elif ev["event_class"] == "probable_occlusion" and m_ovl:
                outcome = "fp_gt_overlap_only"        # GT shows overlap (0.2-0.5) but never an occlusion >= 0.5
            else:
                outcome = "fp_no_gt_overlap"
            rec["outcome"] = outcome
            if target:
                best = min(target, key=lambda ep: abs(ep.start - ev["start_frame"]))
                rec["matched_episode"] = {"pair": [best.a, best.b], "start": best.start, "end": best.end, "tier": best.tier,
                                          "peak_cover": best.peak_cover}
                rec["start_error"] = ev["start_frame"] - best.start
                rec["end_error"] = ev["end_frame"] - best.end
        else:
            rec["outcome"] = _loss_outcome(ev, rows_e, frame_tracks, gt)
        per_event.append(rec)

    # missed-episode causes
    suppressed_pairs = []
    for ev in suppressed:
        if ev["event_class"] != "detection_loss":
            _m, pbf, _u = _event_gt(ev, rows_by_event[ev["event_id"]], frame_tracks, gt)
            suppressed_pairs.append((ev, pbf))
    episodes_out = []
    for ep in episodes:
        hits = (occ_hits if ep.tier == "occlusion" else ovl_hits).get(ep.key, [])
        rec = {"pair": [ep.a, ep.b], "start": ep.start, "end": ep.end, "tier": ep.tier, "peak_cover": ep.peak_cover,
               "detectable": detectable(ep), "matched_events": hits}
        if not hits:
            if not rec["detectable"]:
                rec["miss_cause"] = "member_never_tracked"
            elif any(matches(ev, pbf, [ep]) for ev, pbf in suppressed_pairs):
                rec["miss_cause"] = "suppressed_off_pitch"
            else:
                rec["miss_cause"] = "tracked_but_no_event"
        episodes_out.append(rec)

    claims = pair_claims(events, rows_by_event, frame_tracks, gt, episodes)
    summary = summarise(per_event, episodes_out)
    summary["pair_claims"] = claims["summary"]
    return {"eval_version": EVAL_VERSION, "frame_range": [first, last],
            "parameters": {"gt_iou": GT_IOU, "time_tolerance_frames": TIME_TOL, "detectable_margin": DETECTABLE_MARGIN,
                           "overlap_cover": OVERLAP_COVER, "overlap_min_frames": OVERLAP_MIN,
                           "occlusion_cover": OCCLUSION_COVER, "occlusion_min_frames": OCCLUSION_MIN, "merge_gap": MERGE_GAP},
            "summary": summary, "events": per_event, "episodes": episodes_out, "claims": claims["claims"]}


def pair_claims(events, rows_by_event, frame_tracks: TrackBoxes, gt: GT, episodes: Sequence[Episode]) -> dict[str, Any]:
    """Score every player pair an event asserts, independently of how events are grouped.

    Claims: ``occlusion`` = a hidden member and its occluder, or a strong overlap whose far player degraded;
    ``overlap`` = any overlapping pair. A claim is correct when a GT episode of the same tier for the same GT pair
    overlaps the claim's frames (+-TIME_TOL). Claims involving a non-GT person are not scored.
    """
    eps: dict[tuple[str, int, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        eps[(ep.tier, ep.a, ep.b)].append(ep)
    raw: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for ev in events:
        if ev["event_class"] == "detection_loss":
            continue
        for r in rows_by_event[ev["event_id"]]:
            for pr in r.get("pairs", []):
                raw[("overlap", ev["event_id"], pr["a"], pr["b"])].append(r["frame"])
                if pr.get("strong") and pr.get("far_flags"):
                    raw[("occlusion", ev["event_id"], pr["a"], pr["b"])].append(r["frame"])
            for h in r.get("hidden", []):
                if h.get("occluder") is not None:
                    a, b = sorted((int(h["track_id"]), int(h["occluder"])))
                    raw[("occlusion", ev["event_id"], a, b)].append(r["frame"])
    ident_cache: dict[tuple[int, int, int], int | None] = {}
    claims = []
    for (tier, eid, ta, tb), frs in sorted(raw.items()):
        lo, hi = min(frs), max(frs)
        ids = []
        for t in (ta, tb):
            k = (t, lo, hi)
            if k not in ident_cache:
                ident_cache[k] = track_identity(t, lo - DETECTABLE_MARGIN, hi + DETECTABLE_MARGIN, frame_tracks, gt)[0]
            ids.append(ident_cache[k])
        if None in ids:
            outcome = "non_gt_person"
        elif ids[0] == ids[1]:
            outcome = "duplicate_track"
        else:
            ga, gb = sorted(ids)
            hit = any(ep.start - TIME_TOL <= hi and lo <= ep.end + TIME_TOL for ep in eps[(tier, ga, gb)])
            if hit:
                outcome = "correct"
            elif tier == "occlusion" and any(ep.start - TIME_TOL <= hi and lo <= ep.end + TIME_TOL for ep in eps[("overlap", ga, gb)]):
                outcome = "gt_overlap_only"
            else:
                outcome = "no_gt_overlap"
        claims.append({"tier": tier, "event_id": eid, "tracks": [ta, tb], "gt": ids, "frames": [lo, hi], "outcome": outcome})
    summary = {}
    for tier in ("occlusion", "overlap"):
        cs = [c for c in claims if c["tier"] == tier]
        scored = [c for c in cs if c["outcome"] != "non_gt_person"]
        summary[tier] = {"claims": len(cs), "outcomes": dict(Counter(c["outcome"] for c in cs)),
                         "precision": _r(sum(c["outcome"] == "correct" for c in scored) / len(scored)) if scored else None}
    return {"summary": summary, "claims": claims}


def _loss_outcome(ev: dict, rows: list[dict], frame_tracks: TrackBoxes, gt: GT) -> str:
    """Was a detection-loss event really a loss of a GT player that nobody covered?"""
    tid = int(ev["members"][0]["track_id"])
    # identity from the track's own observed boxes just before (and after) the gap
    gid, _n = track_identity(tid, ev["start_frame"] - DETECTABLE_MARGIN, ev["end_frame"] + DETECTABLE_MARGIN, frame_tracks, gt)
    covered = total = 0
    if gid is None:
        return "non_gt_person"
    for r in rows:
        boxes = dict(gt.get(r["frame"], ()))
        if gid not in boxes:
            continue
        total += 1
        mine = boxes[gid]
        if any(other[3] > mine[3] and g.covered_fraction(mine, other) >= OCCLUSION_COVER
               for k, other in boxes.items() if k != gid):
            covered += 1
    if total == 0:
        return "gt_player_absent"
    return "actually_occluded_in_gt" if covered * 2 >= total else "true_loss"


def _prf(tp_e: int, n_e: int, tp_g: int, n_g: int) -> dict[str, Any]:
    p = tp_e / n_e if n_e else None
    r = tp_g / n_g if n_g else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f1), "tp_events": tp_e, "events_scored": n_e,
            "episodes_matched": tp_g, "episodes": n_g}


def _r(v):
    return None if v is None else round(v, 4)


def summarise(per_event: list[dict], episodes: list[dict]) -> dict[str, Any]:
    def ev(cls):
        return [e for e in per_event if e["event_class"] == cls]

    out: dict[str, Any] = {}
    for cls, tier in (("probable_occlusion", "occlusion"), ("box_overlap", "overlap")):
        evs = ev(cls)
        scored = [e for e in evs if e["outcome"] != "non_gt_person"]
        eps = [e for e in episodes if e["tier"] == tier]
        det = [e for e in eps if e["detectable"]]
        hit_by_cls = [e for e in eps if any(x in {v["event_id"] for v in evs} for x in e["matched_events"])]
        hit_det = [e for e in det if e in hit_by_cls]
        out[cls] = {
            "events": len(evs), "outcomes": dict(Counter(e["outcome"] for e in evs)),
            # precision excludes events whose members are not GT players (referees/staff cannot be scored)
            "vs_" + tier + "_episodes": _prf(sum(e["outcome"] == "tp" for e in scored), len(scored), len(hit_by_cls), len(eps)),
            "vs_detectable_" + tier + "_episodes": _prf(sum(e["outcome"] == "tp" for e in scored), len(scored), len(hit_det), len(det)),
            "timing_frames": _timing([e for e in scored if e["outcome"] == "tp"]),
        }
    occ = [e for e in episodes if e["tier"] == "occlusion"]
    det_occ = [e for e in occ if e["detectable"]]
    out["occlusion_episode_recall_any_class"] = {
        "all": _r(sum(bool(e["matched_events"]) for e in occ) / len(occ)) if occ else None,
        "detectable": _r(sum(bool(e["matched_events"]) for e in det_occ) / len(det_occ)) if det_occ else None,
        "episodes": len(occ), "detectable_episodes": len(det_occ),
        "miss_causes": dict(Counter(e.get("miss_cause") for e in occ if not e["matched_events"])),
    }
    loss = [e for e in per_event if e["event_class"] == "detection_loss"]
    out["detection_loss"] = {
        sub: dict(Counter(e["outcome"] for e in loss if e["subtype"] == sub))
        for sub in sorted({e["subtype"] for e in loss})
    }
    scored_loss = [e for e in loss if e["outcome"] in ("true_loss", "actually_occluded_in_gt")]
    out["detection_loss_not_occlusion_precision"] = _r(
        sum(e["outcome"] == "true_loss" for e in scored_loss) / len(scored_loss)) if scored_loss else None
    out["events_total"] = len(per_event)
    out["uncertain_events"] = sum(e["uncertain"] for e in per_event)
    return out


def _timing(tps: list[dict]) -> dict[str, Any]:
    if not tps:
        return {"n": 0}
    s = [e["start_error"] for e in tps]
    t = [e["end_error"] for e in tps]
    return {"n": len(tps), "start_error_median": statistics.median(s), "start_abs_error_median": statistics.median(abs(x) for x in s),
            "end_error_median": statistics.median(t), "end_abs_error_median": statistics.median(abs(x) for x in t)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
