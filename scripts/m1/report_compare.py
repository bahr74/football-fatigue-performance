"""Print the M1 comparison report from data/processed/m1_possession/compare/results.json (see compare.py)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from m1_labels.scoring import STATES4, merge, merge_passes
from m1_possession.rules import C2Params
from m1_possession.splits import DEV_MATCHES, LEARNING_MATCHES

OUT = Path("data/processed/m1_possession/compare")
INP = Path("data/processed/m1_possession/inputs")
EVS = Path("data/processed/m1_possession/evalset")
ORDER = ("C0", "C1-published", "C1-tuned", "C1+oracle (not a contestant)", "C2-start", "C2-tuned")
SOURCE = {"control_radius": "[VC] [KM] databallpy", "touch_foot": "[ET] [EL]", "touch_reception": "[ET]", "touch_tackle": "[ET]",
          "duel_radius": "[VC]", "h_foot": "[ET]", "h_header": "[ET]", "h_max": "[EL] [ET]", "kick_acc": "[LH] [EL]",
          "dir_change": "[KM] [VC]", "speed_ratio": "[KM]", "speed_abs": "[VC]", "grace_s": "[KM]", "reach_s": "[LH]",
          "release_m": "[EL]", "arrival_m": "[EL]", "min_failed_pass_m": "[KM]", "shot_post_m": "[KM]", "out_margin": "[GUESS]",
          "out_time_s": "[GUESS]", "stop_context_s": "[MU]", "stop_smooth_s": "[MU]", "min_stop_s": "[MU]",
          "walk_speed": "[GUESS] (added)", "kickoff_tol": "T2 [VC]", "throwin_tol": "T2 [VC]", "corner_m": "T2 [VC]",
          "goal_area_tol": "T2 [VC]", "fk_wall_tol": "T2 [VC] [LAW]"}


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def main() -> int:
    R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    res = R["res"]
    dev = [str(m) for m in DEV_MATCHES]
    allm = [str(m) for m in LEARNING_MATCHES]
    print("## Comparison (cross-validated by match; dev = the 8 dev matches, all = the 30 learning matches)\n")
    print("| contestant | frame acc dev | frame acc all | player acc | team acc | poss.-change F1 | dead start F1 ±0.3/±2 s | dead end F1 ±0.3/±2 s |")
    print("|---|---|---|---|---|---|---|---|")
    merged = {}
    for k in ORDER:
        if k not in res:
            continue
        md = merge([res[k][m] for m in dev])
        ma = merge([res[k][m] for m in allm])
        merged[k] = ma
        print(f"| {k} | {f(md['accuracy'])} | {f(ma['accuracy'])} | {f(ma['player_acc'])} | {f(ma['team_acc'])} | "
              f"{f(ma['possession_change']['f1'])} | {f(ma['dead_start_0.3s']['f1'])} / {f(ma['dead_start_2s']['f1'])} | "
              f"{f(ma['dead_end_0.3s']['f1'])} / {f(ma['dead_end_2s']['f1'])} |")
    for k in ("C0", "C1-published", "C1-tuned", "C2-start", "C2-tuned", "C1+oracle (not a contestant)"):
        if k not in merged:
            continue
        conf = np.asarray(merged[k]["confusion"], float)
        rowp = 100 * conf / np.maximum(conf.sum(1, keepdims=True), 1)
        print(f"\n### Confusion, {k} (all 30, rows = answer key, % of row)\n")
        print("| answer key \\ predicted | " + " | ".join(STATES4) + " | (UNKNOWN or other) |")
        print("|---|---|---|---|---|---|")
        for i, s in enumerate(STATES4):
            print(f"| {s} | " + " | ".join(f"{v:.1f}" for v in rowp[i]) + " | - |")
    print("\n## Pass detection (same passer, one-to-one; all 30 learning matches)\n")
    print("| contestant | ±10 s P/R/F1 | ±2 s P/R/F1 | ±0.5 s P/R/F1 |")
    print("|---|---|---|---|")
    for k in ("C1-published", "C1-tuned", "C2-tuned", "C2-start", "C1+oracle (not a contestant)"):
        if k in res and "passes" in res[k][allm[0]]:
            mp = merge_passes([res[k][m]["passes"] for m in allm])
            print(f"| {k} | " + " | ".join(f"{mp[w]['precision']:.2f} / {mp[w]['recall']:.2f} / {mp[w]['f1']:.2f}" for w in ("10s", "2s", "0.5s")) + " |")
    print("\n## Situation breakdown (state accuracy by answer-key rule, all 30; frames)\n")
    rules = sorted(merged["C0"]["by_rule"].items(), key=lambda kv: -kv[1]["frames"])
    rules = [r for r, v in rules if v["frames"] >= 3000][:18]
    print("| answer-key rule | frames | " + " | ".join(k.split(" ")[0] for k in ORDER if k in merged) + " |")
    print("|---|---|" + "---|" * len(merged))
    for r in rules:
        print(f"| {r} | {merged['C0']['by_rule'][r]['frames']} | " + " | ".join(
            f"{100 * merged[k]['by_rule'].get(r, {}).get('acc', float('nan')):.0f}" for k in ORDER if k in merged) + " |")
    # C2-tuned error types with examples
    print("\n## C2-tuned: biggest error types (held-out predictions)\n")
    err, runs = Counter(), {}
    for m in allm:
        p = dict(np.load(OUT / "c2_pred" / f"{m}.npz"))
        e = dict(np.load(EVS / f"{m}.npz"))
        t = np.load(INP / f"{m}.npz")["t"]
        st, ps = e["state"], p["state"]
        sc = e["score_mask"] & (st <= 3) & (e["period"] > 0)
        wrong = sc & (ps != st)
        lab_r, pred_r = e["rule_names"][e["rule"]], p["rule_names"][p["rule"]]
        key = np.array([f"{STATES4[a]} ({lr}) -> {('DEAD', 'CONTROL', 'TRAVELLING', 'LOOSE', 'UNKNOWN')[b]} ({pr})"
                        for a, lr, b, pr in zip(st[wrong], lab_r[wrong], ps[wrong], pred_r[wrong])])
        err.update(key.tolist())
        idx = np.flatnonzero(wrong)
        if len(idx):
            br = np.flatnonzero(np.diff(idx) > 1)
            starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
            for s_, e_ in zip(starts, ends):
                kk = key[np.searchsorted(idx, s_)]
                runs.setdefault(kk, []).append((float(t[e_] - t[s_]), int(m), round(float(t[s_]), 1)))
    tot = sum(err.values())
    for k, v in err.most_common(5):
        ex = sorted(runs.get(k, []), reverse=True)[:2]
        print(f"- {k}: {v} frames ({100 * v / tot:.1f}% of errors); e.g. " + "; ".join(f"match {m} @ {t0} s ({d:.1f} s)" for d, m, t0 in ex))
    # tuned thresholds
    print("\n## C2 tuned thresholds (search on all 30 learning matches; per-fold choices in brackets)\n")
    start = C2Params().as_dict()
    allp = R["c2_all"]
    folds = [R["c2_choice"][str(k)]["params"] for k in range(5)]
    print("| parameter | start | tuned | per fold | source |")
    print("|---|---|---|---|---|")
    for k in SOURCE:
        print(f"| {k} | {start[k]:g} | {allp[k]:.3g} | " + ", ".join(f"{fp[k]:.3g}" for fp in folds) + f" | {SOURCE[k]} |")
    print(f"\nC1-tuned (zone m, angle deg, speed m/s, spatial m): all learning {R['c1_all']}; per fold {R['c1_choice']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
