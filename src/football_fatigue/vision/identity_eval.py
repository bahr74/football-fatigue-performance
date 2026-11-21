"""Reference-based identity diagnostics (evaluation and development only; never inference input).

Given predicted boxes with identities (and optionally identity statuses/confidences) and reference
boxes with reference identities, computes per frame a one-to-one IoU match and then:

* purity of every predicted identity and the share of matched boxes in impure identities;
* within-identity transfers (a predicted id moving from one reference player to another), optionally
  split into same-team and cross-team using an evaluation-only team reference;
* fragmentation (predicted ids per reference player, id changes along each reference player);
* identity behaviour after a reference player's gap in coverage, by gap length: same id resumed
  (correct), an id first seen at the resume (new), or an id already used by another player (wrong);
* box correctness (box reference id == majority reference id of its predicted id), selective accuracy
  and coverage per identity status, reliability (ECE, Brier) of identity confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracking_eval import _iou_matrix

GAP_BUCKETS = ((1, 12), (13, 62), (63, 250), (251, 10**9))


def _bucket(gap: int) -> str:
    for lo, hi in GAP_BUCKETS:
        if lo <= gap <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"
    return "0"


def match_frame(ref: dict[int, Sequence[float]], boxes: Sequence[Sequence[float]], iou: float = 0.5) -> dict[int, int]:
    """``{box index: reference id}`` from a Hungarian assignment at IoU >= ``iou``."""
    gids = list(ref)
    if not gids or not len(boxes):
        return {}
    m = _iou_matrix(np.array([ref[g] for g in gids], float), np.array(boxes, float).reshape(-1, 4))
    out = {}
    for a, b in zip(*linear_sum_assignment(-m)):
        if m[a, b] >= iou:
            out[int(b)] = gids[a]
    return out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if not len(conf):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def evaluate_identity(pred: dict[int, list[dict[str, Any]]], ref: dict[int, dict[int, Sequence[float]]],
                      frames: Sequence[int], team_ref: dict[int, str] | None = None,
                      iou: float = 0.5) -> dict[str, Any]:
    """``pred``: ``{frame: [{"id", "box", optional "status", "confidence"}]}``; ``ref``: ``{frame: {gid: box}}``."""
    matched = []  # (frame, pred id, gt id, status, confidence)
    gt_total = 0
    for f in frames:
        rf = ref.get(f, {})
        gt_total += len(rf)
        ps = pred.get(f, [])
        m = match_frame(rf, [p["box"] for p in ps], iou)
        for i, g in m.items():
            p = ps[i]
            matched.append((f, p["id"], g, p.get("status"), p.get("confidence")))
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_gid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for f, pid, g, _, _ in matched:
        by_pid[pid].append((f, g))
        by_gid[g].append((f, pid))
    major = {pid: Counter(g for _, g in v).most_common(1)[0][0] for pid, v in by_pid.items()}
    # purity
    impure_tp = 0
    purities = []
    multi = 0
    for pid, v in by_pid.items():
        c = Counter(g for _, g in v)
        p = c.most_common(1)[0][1] / len(v)
        purities.append(p)
        impure_tp += len(v) if p < 0.95 else 0
        multi += sum(1 for n in c.values() if n >= 10) >= 2
    # within-id transfers
    transfers = Counter()
    for pid, v in by_pid.items():
        v.sort()
        for (_, g1), (_, g2) in zip(v, v[1:]):
            if g1 != g2:
                kind = "transfer"
                if team_ref is not None:
                    t1, t2 = team_ref.get(g1), team_ref.get(g2)
                    kind = "transfer_same_team" if t1 == t2 else "transfer_cross_team"
                transfers[kind] += 1
    # fragmentation and resume after gap
    first_seen: dict[int, int] = {}
    for f, pid, *_ in sorted(matched):
        first_seen.setdefault(pid, f)
    owners: dict[int, set[int]] = defaultdict(set)  # pid -> gids seen so far (time ordered pass below)
    events = sorted((f, g, pid) for f, pid, g, _, _ in matched)
    last_seen: dict[int, tuple[int, int]] = {}
    resume = defaultdict(Counter)
    id_changes = 0
    for f, g, pid in events:
        if g in last_seen:
            f0, p0 = last_seen[g]
            gap = f - f0 - 1
            if pid != p0:
                id_changes += 1
            if gap >= 1:
                if pid == p0:
                    kind = "correct"
                elif first_seen[pid] == f:
                    kind = "new"
                elif owners[pid] - {g}:
                    kind = "wrong"
                else:
                    kind = "other_own_id"
                resume[_bucket(gap)][kind] += 1
        owners[pid].add(g)
        last_seen[g] = (f, pid)
    # correctness, statuses, calibration
    correct = np.array([major[pid] == g for _, pid, g, _, _ in matched], bool)
    status = [s for _, _, _, s, _ in matched]
    out: dict[str, Any] = {
        "gt_boxes": gt_total, "matched_boxes": len(matched), "pred_ids_matched": len(by_pid),
        "gt_ids_matched": len(by_gid),
        "box_correct_share": float(correct.mean()) if len(correct) else float("nan"),
        "purity_mean": float(np.mean(purities)) if purities else float("nan"),
        "ids_purity_lt_0_95": int(sum(p < 0.95 for p in purities)),
        "tp_share_in_impure_ids": impure_tp / max(1, len(matched)),
        "ids_with_two_players_ge10_boxes": int(multi),
        "within_id_transfers": dict(transfers),
        "pred_ids_per_gt_mean": float(np.mean([len({p for _, p in v}) for v in by_gid.values()])) if by_gid else float("nan"),
        "gt_id_changes": id_changes,
        "resume_after_gap": {k: dict(v) for k, v in sorted(resume.items())},
    }
    if any(s is not None for s in status):
        st = {}
        for s in sorted({s for s in status if s is not None}):
            m = np.array([x == s for x in status])
            st[s] = {"boxes": int(m.sum()), "coverage": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None}
        out["by_status"] = st
    conf = np.array([c for *_, c in matched if c is not None], float)
    if len(conf) == len(matched) and len(conf):
        out["ece"] = expected_calibration_error(conf, correct.astype(float))
        out["brier"] = float(np.mean((conf - correct) ** 2))
        curve = []
        for q in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            thr = float(np.quantile(conf, 1 - q)) if q < 1 else -1.0
            m = conf >= thr
            curve.append({"coverage": round(float(m.mean()), 3), "accuracy": round(float(correct[m].mean()), 4),
                          "threshold": round(thr, 4)})
        out["accuracy_coverage"] = curve
    return out
