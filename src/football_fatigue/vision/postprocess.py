"""Offline refinement of a saved CV run (``frames.jsonl``): stitch + kit labels.

Pass 1 (detector + ByteTrack) is online and causal, so it fragments tracks and
cannot label a kit colour until it has seen enough frames. Because the export
is offline, a second pass can use the whole run:

1. **Whole-track kit label**: plurality of the raw per-frame colour votes of a
   (stitched) track, so no warm-up gap. ``unknown`` is kept when votes are too
   few or no label has a clear majority; goalkeeper/referee-style kits stay
   ``other`` and are never put into a team.
2. **Tracklet stitching**: a tracklet that ends is linked to one that starts
   within ``max_gap`` frames when (a) the gap in position is physically
   plausible in image space, (b) box size is similar, and (c) kit colours are
   not in conflict (blue vs white, or team vs other, is never linked).
   Assignment is one-to-one and greedy by cost.
3. **Flicker removal**: stitched tracks shorter than ``min_track_frames`` are
   dropped (reported in the stats).

Track IDs remain tracker labels. Stitching raises continuity but is a heuristic:
two nearby players can still be merged or split, so IDs are not verified identities.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Tracklet:
    tid: int
    first: int
    last: int
    first_box: list[float]
    last_box: list[float]
    count: int = 0
    votes: Counter = field(default_factory=Counter)


def _label(votes: Counter, min_votes: int, min_share: float) -> tuple[str, float]:
    decided = sum(votes.values())
    if decided < min_votes:
        return "unknown", 0.0
    best, n = votes.most_common(1)[0]
    share = n / decided
    return (best, round(share, 3)) if share >= min_share else ("unknown", 0.0)


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def refine(
    src: Path,
    dst: Path,
    *,
    max_gap: int = 75,
    speed_px_per_frame: float = 12.0,
    min_track_frames: int = 8,
    kit_min_votes: int = 8,
    kit_min_share: float = 0.75,
    stitch: bool = True,
) -> dict:
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    tracklets: dict[int, _Tracklet] = {}
    for rec in records:
        f = rec["frame_num"]
        for p in rec["players"]:
            box = p.get("bbox_xyxy_px")
            if not box:
                continue
            t = tracklets.get(p["track_id"])
            if t is None:
                t = tracklets[p["track_id"]] = _Tracklet(p["track_id"], f, f, box, box)
            t.last, t.last_box = f, box
            t.count += 1
            if p.get("kit_vote"):
                t.votes[p["kit_vote"]] += 1

    labels = {tid: _label(t.votes, kit_min_votes, kit_min_share)[0] for tid, t in tracklets.items()}
    # fallback for boxes the online classifier left unknown (warm-up): the tracklet's own plurality
    fallback = {tid: _label(t.votes, kit_min_votes, 0.6) for tid, t in tracklets.items()}

    # ---- stitching -------------------------------------------------------
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    if stitch:
        order = sorted(tracklets.values(), key=lambda t: t.first)
        candidates: list[tuple[float, int, int]] = []
        for b in order:
            for a in order:
                gap = b.first - a.last
                if gap < 1 or gap > max_gap or a.tid == b.tid:
                    continue
                la, lb = labels[a.tid], labels[b.tid]
                if la != "unknown" and lb != "unknown" and la != lb:
                    continue  # kit conflict (blue/white/other)
                ha = max(1.0, a.last_box[3] - a.last_box[1])
                hb = max(1.0, b.first_box[3] - b.first_box[1])
                if not 0.6 <= hb / ha <= 1.67:
                    continue
                (ax, ay), (bx, by) = _center(a.last_box), _center(b.first_box)
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                allowed = 0.6 * ha + speed_px_per_frame * gap
                if dist > allowed:
                    continue
                candidates.append((dist / allowed + 0.1 * gap / max_gap, a.tid, b.tid))
        for _, a_tid, b_tid in sorted(candidates):
            if a_tid in successor or b_tid in predecessor:
                continue
            successor[a_tid] = b_tid
            predecessor[b_tid] = a_tid

    # ---- chains -> new ids ----------------------------------------------
    chains: list[list[int]] = []
    for t in sorted(tracklets.values(), key=lambda t: t.first):
        if t.tid in predecessor:
            continue
        chain, cur = [t.tid], t.tid
        while cur in successor:
            cur = successor[cur]
            chain.append(cur)
        chains.append(chain)
    mapping: dict[int, int] = {}
    chain_info: dict[int, tuple[str, float]] = {}
    dropped_short = 0
    new_id = 0
    for chain in chains:
        frames_in_chain = sum(tracklets[t].count for t in chain)
        if frames_in_chain < min_track_frames:
            dropped_short += 1
            continue
        new_id += 1
        votes: Counter = Counter()
        for t in chain:
            mapping[t] = new_id
            votes.update(tracklets[t].votes)
        chain_info[new_id] = _label(votes, kit_min_votes, kit_min_share)  # reported only

    # ---- write ------------------------------------------------------------
    dst.parent.mkdir(parents=True, exist_ok=True)
    kit_boxes = Counter()
    with dst.open("x", encoding="utf-8", newline="\n") as out:
        for rec in records:
            players = []
            for p in rec["players"]:
                nid = mapping.get(p["track_id"])
                if nid is None:
                    continue
                # Pooling votes over a stitched chain mixes lighting regimes and chains bridged by
                # an unlabelled tracklet, so boxes keep the online label and only fall back to
                # their own tracklet's plurality.
                kit, conf = p.get("kit_color") or "unknown", p.get("kit_confidence") or 0.0
                if kit == "unknown":
                    kit, conf = fallback[p["track_id"]]
                q = dict(p, track_id=nid, raw_track_id=p["track_id"], kit_color=kit, kit_confidence=conf)
                kit_boxes[kit] += 1
                players.append(q)
            rec = dict(rec, players=players)
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
    total_boxes = sum(kit_boxes.values())
    return {
        "frames": len(records),
        "tracklets_in": len(tracklets),
        "links": len(successor),
        "tracks_out": new_id,
        "dropped_short_tracks": dropped_short,
        "boxes_out": total_boxes,
        "kit_box_share": {k: round(v / max(1, total_boxes), 4) for k, v in kit_boxes.items()},
        "parameters": {
            "stitch": stitch, "max_gap_frames": max_gap, "speed_px_per_frame": speed_px_per_frame,
            "min_track_frames": min_track_frames, "kit_min_votes": kit_min_votes, "kit_min_share": kit_min_share,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="pass-1 run directory containing frames.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="new refined frames.jsonl path (never overwritten)")
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--max-gap", type=int, default=75)
    ap.add_argument("--min-track-frames", type=int, default=8)
    args = ap.parse_args(argv)
    stats = refine(args.run_dir / "frames.jsonl", args.out, max_gap=args.max_gap,
                   min_track_frames=args.min_track_frames, stitch=not args.no_stitch)
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
