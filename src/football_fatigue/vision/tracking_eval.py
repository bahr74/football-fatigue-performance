"""Compare a saved CV run's tracks with SoccerTrack MOT reference tracks.

Per frame, predictions are assigned to reference boxes with the Hungarian
algorithm on IoU (pairs below ``--iou`` are not matches). Reported:

* detection: TP/FP/FN, precision, recall;
* association: ID switches, fragmentations, IDF1/IDP/IDR (global identity
  assignment between reference and predicted tracks, as in Ristani et al.);
* failure modes: false positives split into *mislocalised* (overlap a reference
  box at IoU 0.1-threshold, i.e. a real player with a poor box or a duplicate)
  and *isolated* (no reference overlap: referees, staff, spectators or
  background), plus persistent isolated tracks; misses split by box height.

Caveat: the MOT file labels exactly the 22 on-pitch players, not referees,
staff or substitutes, so "isolated" false positives include real people who are
simply not annotated. Track IDs are tracker labels, never verified identities.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from football_fatigue.vision.benchmark import load_mot_ids


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def evaluate(
    frames_path: Path,
    mot_path: Path,
    threshold: float = 0.5,
    start: int = 0,
    end: int | None = None,
    kit_reference: dict | None = None,
) -> dict:
    from scipy.optimize import linear_sum_assignment

    ref = load_mot_ids(mot_path)
    last_pred: dict[int, int] = {}
    prev_matched: dict[int, bool] = {}
    tp = fp = fn = idsw = frag = frames = n_ref = 0
    fp_mislocalised = fp_isolated = 0
    fn_small = fn_total_small_ref = 0
    pair_frames: dict[tuple[int, int], int] = defaultdict(int)  # (ref id, pred id) -> co-matched frames
    ref_frames: dict[int, int] = defaultdict(int)
    pred_frames: dict[int, int] = defaultdict(int)
    isolated_by_track: dict[int, int] = defaultdict(int)
    kit_conf: dict[tuple[str, str], int] = defaultdict(int)
    kit_ref = {}
    for cls, ids in (kit_reference or {}).items():
        if not cls.startswith('_'):
            kit_ref.update({i: cls for i in ids})
    conf_fp: list[float] = []
    conf_tp: list[float] = []
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            f = rec["frame_num"]
            if f < start or (end is not None and f >= end):
                continue
            frames += 1
            players = [p for p in rec["players"] if p.get("bbox_xyxy_px")]
            pb = np.array([p["bbox_xyxy_px"] for p in players], dtype=float).reshape(-1, 4)
            pid = [p["track_id"] for p in players]
            rb = np.array([r[1] for r in ref.get(f, [])], dtype=float).reshape(-1, 4)
            rid = [r[0] for r in ref.get(f, [])]
            n_ref += len(rid)
            for t in pid:
                pred_frames[t] += 1
            for r in rid:
                ref_frames[r] += 1
            iou = _iou_matrix(pb, rb)
            used_p: set[int] = set()
            used_r: set[int] = set()
            if iou.size:
                rows, cols = linear_sum_assignment(-iou)
                for pi, ri in zip(rows, cols):
                    if iou[pi, ri] >= threshold:
                        used_p.add(pi)
                        used_r.add(ri)
                        r_id, t_id = rid[ri], pid[pi]
                        if r_id in last_pred and last_pred[r_id] != t_id:
                            idsw += 1
                        if prev_matched.get(r_id) is False:
                            frag += 1
                        last_pred[r_id] = t_id
                        prev_matched[r_id] = True
                        pair_frames[(r_id, t_id)] += 1
                        conf_tp.append(players[pi].get("confidence") or 0.0)
                        if r_id in kit_ref:
                            kit_conf[(kit_ref[r_id], players[pi].get("kit_color") or "unknown")] += 1
            for ri, r_id in enumerate(rid):
                if ri not in used_r:
                    if prev_matched.get(r_id):
                        prev_matched[r_id] = False
                    if rb[ri, 3] - rb[ri, 1] < 35:
                        fn_small += 1
                if rb[ri, 3] - rb[ri, 1] < 35:
                    fn_total_small_ref += 1
            for pi in range(len(pid)):
                if pi in used_p:
                    continue
                conf_fp.append(players[pi].get("confidence") or 0.0)
                if iou.size and iou[pi].max() >= 0.1:
                    fp_mislocalised += 1
                else:
                    fp_isolated += 1
                    isolated_by_track[pid[pi]] += 1
            tp += len(used_p)
            fp += len(pid) - len(used_p)
            fn += len(rid) - len(used_r)

    # IDF1: one-to-one assignment between reference and predicted identities.
    r_ids = sorted(ref_frames)
    p_ids = sorted(pred_frames)
    idtp = 0
    if r_ids and p_ids:
        ri_index = {r: i for i, r in enumerate(r_ids)}
        pi_index = {p: i for i, p in enumerate(p_ids)}
        m = np.zeros((len(r_ids), len(p_ids)))
        for (r, p), c in pair_frames.items():
            m[ri_index[r], pi_index[p]] = c
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    total_ref, total_pred = sum(ref_frames.values()), sum(pred_frames.values())
    idp = idtp / max(1, total_pred)
    idr = idtp / max(1, total_ref)
    idf1 = 2 * idtp / max(1, total_ref + total_pred)
    persistent = sorted(((c, t) for t, c in isolated_by_track.items() if c >= 50), reverse=True)
    result = {
        "frames": frames, "iou": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4),
        "id_switches": idsw, "fragmentations": frag,
        "idf1": round(idf1, 4), "idp": round(idp, 4), "idr": round(idr, 4),
        "mota_like": round(1 - (fn + fp + idsw) / max(1, n_ref), 4),
        "reference_identities": len(r_ids), "predicted_track_ids": len(p_ids),
        "tracks_per_reference_identity": round(len(p_ids) / max(1, len(r_ids)), 1),
        "fp_breakdown": {
            "mislocalised_or_duplicate": fp_mislocalised,
            "isolated_no_reference_overlap": fp_isolated,
            "persistent_isolated_tracks_ge50_frames": len(persistent),
            "frames_of_persistent_isolated_tracks": int(sum(c for c, _ in persistent)),
            "mean_conf_fp": round(float(np.mean(conf_fp)), 3) if conf_fp else None,
            "mean_conf_tp": round(float(np.mean(conf_tp)), 3) if conf_tp else None,
        },
        "fn_breakdown": {
            "missed_small_boxes_h_lt_35px": fn_small,
            "reference_small_boxes_h_lt_35px": fn_total_small_ref,
            "missed_other": fn - fn_small,
        },
    }
    if kit_reference:
        labels = ("blue", "white", "other", "unknown")
        matrix = {r: {l: kit_conf[(r, l)] for l in labels} for r in ("blue", "white", "other")}
        total = sum(sum(row.values()) for row in matrix.values())
        unknown = sum(row["unknown"] for row in matrix.values())
        correct = sum(matrix[r][r] for r in matrix)
        labelled = total - unknown
        bw = sum(matrix[a][b] for a, b in (("blue", "white"), ("white", "blue")))
        outfield = sum(matrix[a][b] for a in ("blue", "white") for b in ("blue", "white"))
        result["kit_color"] = {
            "reference": "derived colour labels, not independent annotation (see kit_reference file)",
            "confusion_reference_by_predicted": matrix,
            "coverage_non_unknown": round(labelled / max(1, total), 4),
            "accuracy_on_labelled": round(correct / max(1, labelled), 4),
            "blue_white_swap_rate": round(bw / max(1, outfield), 4),
            "other_kit_assigned_to_a_team": matrix["other"]["blue"] + matrix["other"]["white"],
        }
    return result



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("mot", type=Path)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--kit-reference", type=Path, help="JSON {white:[ids],blue:[ids],other:[ids]} for kit-colour evaluation")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write tracking_eval.json")
    args = ap.parse_args(argv)
    result = evaluate(
        args.run_dir / "frames.jsonl", args.mot, args.iou, args.start, args.end,
        json.loads(args.kit_reference.read_text(encoding="utf-8")) if args.kit_reference else None,
    )
    if not args.no_write and args.start == 0 and args.end is None:
        (args.run_dir / "tracking_eval.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())