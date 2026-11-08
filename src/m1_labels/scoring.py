"""Scoring of a per-frame possession prediction against the answer key (plan, Approach section: Scoring).

Definitions (fixed here so every contestant is scored the same way):
- scored frames: label score_mask True and label state in DEAD/CONTROL/TRAVELLING/LOOSE, inside a period, where the
  prediction exists;
- confusion 4x4 on scored frames (rows = answer key);
- player / team accuracy: scored frames labelled CONTROL; correct when the predicted player (team) equals the label's,
  whatever state was predicted;
- possession change: the first frame of a CONTROL run whose team differs from the previous CONTROL run's team in the same
  period (runs of other states in between are ignored); matched one-to-one within the tolerance, F1;
- dead-ball start / end: frames where the state enters / leaves DEAD within a period; F1 within +-0.3 s and +-2 s;
- label events count only when their frame is scored; predicted events count everywhere scored frames exist.
"""
from __future__ import annotations

import numpy as np

from .labeller import CONTROL, DEAD, TOLERANCE_S

STATES4 = ("DEAD", "CONTROL", "TRAVELLING", "LOOSE")


def match_events(a: np.ndarray, b: np.ndarray, tol: int) -> int:
    """Greedy one-to-one matching of two sorted frame-index arrays by smallest |difference| <= tol; returns matches."""
    if len(a) == 0 or len(b) == 0:
        return 0
    pairs = []
    for i, x in enumerate(a):
        lo, hi = np.searchsorted(b, x - tol, "left"), np.searchsorted(b, x + tol, "right")
        pairs.extend((abs(int(b[j]) - int(x)), i, j) for j in range(lo, hi))
    pairs.sort()
    ua, ub, m = set(), set(), 0
    for _, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); m += 1
    return m


def change_events(state, team, period) -> np.ndarray:
    ev, last_team, last_p = [], None, None
    ctrl = state == CONTROL
    start = np.flatnonzero(ctrl & ~np.r_[False, ctrl[:-1]] | (ctrl & np.r_[False, team[1:] != team[:-1]]))
    for i in start:
        if period[i] != last_p:
            last_team, last_p = None, period[i]
        if last_team is not None and team[i] != last_team and team[i] >= 0:
            ev.append(i)
        if team[i] >= 0:
            last_team = team[i]
    return np.asarray(ev, int)


def dead_events(state, period) -> tuple[np.ndarray, np.ndarray]:
    dead = state == DEAD
    same = np.r_[False, period[1:] == period[:-1]]
    starts = np.flatnonzero(dead & ~np.r_[False, dead[:-1]] & same)
    ends = np.flatnonzero(~dead & np.r_[False, dead[:-1]] & same)
    return starts, ends


def f1(tp, n_true, n_pred) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return dict(tp=int(tp), n_true=int(n_true), n_pred=int(n_pred), precision=p, recall=r, f1=(2 * p * r / (p + r)) if p + r else 0.0)


def score(label: dict, pred_state, pred_player, pred_team, fps: float, rule_names=None) -> dict:
    """label: arrays state, player_id, team_id, score_mask, period, rule (same frames as the prediction)."""
    st, pl, tm, mask, per = (np.asarray(label[k]) for k in ("state", "player_id", "team_id", "score_mask", "period"))
    ps, pp, pt = np.asarray(pred_state), np.asarray(pred_player), np.asarray(pred_team)
    scored = mask & (st <= 3) & (per > 0) & (ps >= 0)
    conf = np.zeros((4, 4), np.int64)
    np.add.at(conf, (st[scored], np.clip(ps[scored], 0, 3)), 1)
    c = scored & (st == CONTROL)
    out = dict(frames=int(scored.sum()), confusion=conf.tolist(), accuracy=float(np.trace(conf) / max(conf.sum(), 1)),
               player_acc=float((pp[c] == pl[c]).mean()) if c.any() else None,
               team_acc=float((pt[c] == tm[c]).mean()) if c.any() else None, control_frames=int(c.sum()))
    tol = int(round(TOLERANCE_S * fps))
    le = change_events(st, tm, per)
    le = le[scored[le]]
    pe = change_events(ps, pt, per)
    out["possession_change"] = f1(match_events(le, pe, tol), len(le), len(pe))
    ls, lend = dead_events(st, per)
    pss, pend = dead_events(ps, per)
    ls, lend = ls[scored[ls] | scored[np.maximum(ls - 1, 0)]], lend[scored[lend] | scored[np.maximum(lend - 1, 0)]]
    for name, tl in (("0.3s", tol), ("2s", int(round(2.0 * fps)))):
        out[f"dead_start_{name}"] = f1(match_events(ls, pss, tl), len(ls), len(pss))
        out[f"dead_end_{name}"] = f1(match_events(lend, pend, tl), len(lend), len(pend))
    if rule_names is not None:
        rr = np.asarray(label["rule"])
        by = {}
        for k in np.unique(rr[scored]):
            sel = scored & (rr == k)
            by[str(rule_names[k])] = dict(frames=int(sel.sum()), acc=float((ps[sel] == st[sel]).mean()))
        out["by_rule"] = by
    return out


PASS_WINDOWS_S = (10.0, 2.0, 0.5)   # 10 s = the benchmark window of Vidal-Codina et al. 2022


def score_passes(det_frames, det_passers, true_frames, true_passers, fps: float) -> dict:
    """Pass detection: a detected pass matches a PFF PA event with the same passer within the window (one-to-one)."""
    det_frames, det_passers = np.asarray(det_frames, int), np.asarray(det_passers)
    true_frames, true_passers = np.asarray(true_frames, int), np.asarray(true_passers)
    out = {}
    for w in PASS_WINDOWS_S:
        tol = int(round(w * fps))
        tp = 0
        for pid in np.unique(true_passers):
            a = np.sort(true_frames[true_passers == pid])
            b = np.sort(det_frames[det_passers == pid])
            tp += match_events(a, b, tol)
        out[f"{w:g}s"] = f1(tp, len(true_frames), len(det_frames))
    return out


def merge_passes(scores: list[dict]) -> dict:
    out = {}
    for k in scores[0]:
        tp, nt, npred = (sum(s[k][x] for s in scores) for x in ("tp", "n_true", "n_pred"))
        out[k] = f1(tp, nt, npred)
    return out


HAD_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def had(state, team, period, home_team: int) -> np.ndarray:
    """Team-level view: 0 H, 1 A, 2 D. CONTROL / TRAVELLING with a team -> that team; DEAD -> D; anything else (LOOSE,
    UNKNOWN, no team) -> the last H / A in the same period (-1 before any)."""
    state, team, period = np.asarray(state), np.asarray(team), np.asarray(period)
    v = np.full(len(state), -1, np.int8)
    own = np.isin(state, (CONTROL, 2)) & (team >= 0)
    v[own] = np.where(team[own] == home_team, 0, 1)
    idx = np.maximum.accumulate(np.where(own, np.arange(len(v)), -1))
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1]
    pstart = starts[np.searchsorted(starts, np.arange(len(v)), "right") - 1]
    fill = ~own & (idx >= pstart)
    v[fill] = v[idx[fill]]
    v[state == DEAD] = 2
    return v


def _spells(v, period, keep):
    ok = np.isin(v, keep)
    b = np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (period[1:] != period[:-1])])
    e = np.r_[b[1:], len(v)]
    sel = ok[b]
    return b[sel], e[sel], v[b[sel]]


def score_had(label: dict, pred_state, pred_team, home_team: int) -> dict:
    """macro-F1 over H / A / D on the scored frames; spell-F1: H / A possession spells (maximal runs inside a period) matched
    one-to-one (greedy by IoU, same team) between answer key and prediction, F1 averaged over IoU 0.3-0.7."""
    st, tm, mask, per = (np.asarray(label[k]) for k in ("state", "team_id", "score_mask", "period"))
    inper = per > 0
    lv = had(st, tm, per, home_team)
    pv = had(pred_state, pred_team, per, home_team)
    scored = mask & (st <= 3) & inper
    counts = {}
    for c in (0, 1, 2):
        counts[c] = dict(tp=int(np.sum(scored & (lv == c) & (pv == c))), n_true=int(np.sum(scored & (lv == c))),
                         n_pred=int(np.sum(scored & (pv == c))))
    idx = np.flatnonzero(inper)
    lb, le, lc = _spells(lv[idx], per[idx], (0, 1))
    pb, pe, pc = _spells(pv[idx], per[idx], (0, 1))
    pairs = []
    for i, (b, e, c) in enumerate(zip(lb, le, lc)):
        lo, hi = np.searchsorted(pe, b, "right"), np.searchsorted(pb, e, "left")
        for j in range(lo, hi):
            if pc[j] != c:
                continue
            inter = min(e, pe[j]) - max(b, pb[j])
            if inter > 0:
                pairs.append((inter / (max(e, pe[j]) - min(b, pb[j])), i, j))
    pairs.sort(reverse=True)
    sp = {}
    for thr in HAD_IOUS:
        ua, ub, m = set(), set(), 0
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i not in ua and j not in ub:
                ua.add(i); ub.add(j); m += 1
        sp[f"{thr:g}"] = dict(tp=m, n_true=len(lb), n_pred=len(pb))
    return dict(had_counts=counts, spells=sp)


def merge_had(scores: list[dict]) -> dict:
    f1s = []
    for c in (0, 1, 2):
        tp, nt, npred = (sum(s["had_counts"][c][x] if c in s["had_counts"] else s["had_counts"][str(c)][x] for s in scores)
                         for x in ("tp", "n_true", "n_pred"))
        f1s.append(f1(tp, nt, npred)["f1"])
    sp = []
    for thr in HAD_IOUS:
        k = f"{thr:g}"
        tp, nt, npred = (sum(s["spells"][k][x] for s in scores) for x in ("tp", "n_true", "n_pred"))
        sp.append(f1(tp, nt, npred)["f1"])
    return dict(macro_f1=float(np.mean(f1s)), f1_H=f1s[0], f1_A=f1s[1], f1_D=f1s[2], spell_f1=float(np.mean(sp)),
                spell_f1_by_iou=dict(zip([f"{t:g}" for t in HAD_IOUS], sp)))


def merge(scores: list[dict]) -> dict:
    """Pool per-match scores: sum counts, recompute rates."""
    conf = np.sum([np.asarray(s["confusion"]) for s in scores], axis=0)
    cf = sum(s["control_frames"] for s in scores)
    out = dict(frames=int(conf.sum()), confusion=conf.tolist(), accuracy=float(np.trace(conf) / max(conf.sum(), 1)),
               player_acc=sum((s["player_acc"] or 0) * s["control_frames"] for s in scores) / max(cf, 1),
               team_acc=sum((s["team_acc"] or 0) * s["control_frames"] for s in scores) / max(cf, 1), control_frames=cf)
    for k in ("possession_change", "dead_start_0.3s", "dead_end_0.3s", "dead_start_2s", "dead_end_2s"):
        tp, nt, npred = (sum(s[k][x] for s in scores) for x in ("tp", "n_true", "n_pred"))
        out[k] = f1(tp, nt, npred)
    by = {}
    for s in scores:
        for r, v in (s.get("by_rule") or {}).items():
            b = by.setdefault(r, [0, 0.0])
            b[0] += v["frames"]; b[1] += v["acc"] * v["frames"]
    out["by_rule"] = {r: dict(frames=f, acc=a / f) for r, (f, a) in by.items() if f}
    return out
