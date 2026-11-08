"""Print the M1 C2-v2 / input-upgrade / C3 report (all numbers cross-validated by match on the 30 learning matches).

Reads compare/results.json (C0 / C1 / C2-tuned), compare/c2v2.json, bfp/bfp_eval.json, c3/ablate_*.json, c3/final_*.json and
the saved predictions. H / A / D scores for C0 and C1-tuned are recomputed from their predictions (cached in compare/had.json).

    python scripts/m1/report_c3.py --final <config>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare import load  # noqa: E402
from m1_labels.scoring import STATES4, merge, merge_had, merge_passes, score_had  # noqa: E402
from m1_possession.baselines import c0_naive, c1_finish, c1_prepare  # noqa: E402
from m1_possession.splits import DEV_MATCHES, FOLDS, LEARNING_MATCHES  # noqa: E402

BASE = Path("data/processed/m1_possession")
CMP, C3D = BASE / "compare", BASE / "c3"
ABL = ("none", "smooth", "gapfill", "gapfeat", "bfp", "all")
HOMES = {int(k): v for k, v in json.loads((BASE / "home_team.json").read_text()).items()}


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def had_job(args):
    mid, c1p = args
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    out = {}
    r = c0_naive(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"])
    out["C0"] = score_had(e, r.state, r.team, HOMES[mid])
    zone, ang, sp, spa = c1p
    prep = c1_prepare(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], *z["pitch"], zone_m=zone, spatials=(spa,))
    r = c1_finish(prep, ang, sp, spa)
    out["C1-tuned"] = score_had(e, r.state, r.team, HOMES[mid])
    for name, d in (("C2-tuned", "c2_pred"), ("C2-v2", "c2v2_pred")):
        p = np.load(CMP / d / f"{mid}.npz")
        out[name] = score_had(e, p["state"], p["team"], HOMES[mid])
    return mid, out


def had_scores(R):
    path = CMP / "had.json"
    if path.exists():
        return json.loads(path.read_text())
    c1 = {m: R["c1_choice"][str(k)] for k, fold in enumerate(FOLDS) for m in fold}
    with multiprocessing.Pool(3, maxtasksperchild=2) as pool:
        out = dict(pool.map(had_job, [(m, tuple(c1[m])) for m in LEARNING_MATCHES]))
    res = {}
    for m, d in out.items():
        for k, v in d.items():
            res.setdefault(k, {})[str(m)] = json.loads(json.dumps(v))
    path.write_text(json.dumps(res))
    return res


def row(name, res, had=None, dev=True):
    allm = [str(m) for m in LEARNING_MATCHES]
    ma = merge([res[m] for m in allm])
    md = merge([res[str(m)] for m in DEV_MATCHES]) if dev else None
    cells = [name, f(md["accuracy"]) if md else "-", f(ma["accuracy"]), f(ma["player_acc"]), f(ma["team_acc"]),
             f(ma["possession_change"]["f1"]), f"{f(ma['dead_start_0.3s']['f1'], 2)} / {f(ma['dead_start_2s']['f1'], 2)}",
             f"{f(ma['dead_end_0.3s']['f1'], 2)} / {f(ma['dead_end_2s']['f1'], 2)}"]
    if had is not None:
        mh = merge_had([had[m] for m in allm])
        cells += [f(mh["macro_f1"]), f(mh["spell_f1"])]
    return "| " + " | ".join(cells) + " |", ma


HDR = ("| contestant | frame acc dev | frame acc all | player acc | team acc | poss.-change F1 | dead start F1 ±0.3 / ±2 s | "
       "dead end F1 ±0.3 / ±2 s | H/A/D macro-F1 | spell-F1 |\n|---|---|---|---|---|---|---|---|---|---|")


def conf_table(ma, title):
    conf = np.asarray(ma["confusion"], float)
    rowp = 100 * conf / np.maximum(conf.sum(1, keepdims=True), 1)
    print(f"\n### {title} (all 30, rows = answer key, % of row)\n")
    print("| answer key \\ predicted | " + " | ".join(STATES4) + " |\n|---|---|---|---|---|")
    for i, s in enumerate(STATES4):
        print(f"| {s} | " + " | ".join(f"{v:.1f}" for v in rowp[i]) + " |")


def main() -> int:
    from m1_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    R = json.loads((CMP / "results.json").read_text(encoding="utf-8"))
    V = json.loads((CMP / "c2v2.json").read_text(encoding="utf-8"))
    had = had_scores(R)
    # 1. C2-v2 vs C2-tuned
    print("## 1. C2-v2 vs C2-tuned (cross-validated)\n")
    print(HDR)
    for name, res in (("C2-tuned", R["res"]["C2-tuned"]), ("C2-v2", V["res"])):
        print(row(name, res, had[name])[0])
    print("\nNew v2 parameters per fold (start: restart_limit_s 60, clear_margin_m 1.0, gap_long_s 4, gap_same_spot_m 3, "
          "still_speed_v2 2.0):")
    for k in range(5):
        c = V["choice"][str(k)]
        p = c["params"]
        print(f"- fold {k}: " + ", ".join(f"{x} {p[x]:.3g}" for x in ("restart_limit_s", "clear_margin_m", "gap_long_s",
                                                                     "gap_same_spot_m", "still_speed_v2")) +
              f"; train acc start {c['train_acc_start']:.4f} -> {c['train_acc']:.4f}")
    p = V["all"]["params"]
    print("- all 30: " + ", ".join(f"{x} {p[x]:.3g}" for x in ("restart_limit_s", "clear_margin_m", "gap_long_s", "gap_same_spot_m",
                                                            "still_speed_v2")))
    # 2. ball from players
    B = json.loads((BASE / "bfp" / "bfp_eval.json").read_text())
    print("\n## 2. Ball-from-players error (hidden visible ball, held-out folds; metres)\n")
    print("| ball | time missing | samples | median | p95 | last-seen median | last-seen p95 |\n|---|---|---|---|---|---|---|")
    for r in B["table"]:
        print(f"| {r['ball']} | {r['h']} | {r['n']} | {r['med']:.1f} | {r['p95']:.1f} | {r['base_med']:.1f} | {r['base_p95']:.1f} |")
    print("\nTop features: " + ", ".join(n for n, _ in B["top_features"][:8]))
    # 3. with / without
    print("\n## 3. C3 with / without input upgrades (fixed LightGBM settings, default decoder)\n")
    print(HDR.replace("contestant", "C3 inputs"))
    base_acc = None
    for c in ABL + tuple(sorted(p.stem[7:] for p in C3D.glob("ablate_custom-*.json"))):
        pth = C3D / f"ablate_{c}.json"
        if not pth.exists():
            continue
        A = json.loads(pth.read_text())
        line, ma = row(c, A["res"], {m: s["had"] for m, s in A["res"].items()})
        print(line)
    # 4. comparison
    F = json.loads((C3D / f"final_{a.final}.json").read_text())
    print(f"\n## 4. All contestants (cross-validated; C3 = inputs '{a.final}', nested tuning)\n")
    print(HDR)
    rows = [("C0", R["res"]["C0"], had["C0"]), ("C1-tuned", R["res"]["C1-tuned"], had["C1-tuned"]),
            ("C2-tuned", R["res"]["C2-tuned"], had["C2-tuned"]), ("C2-v2", V["res"], had["C2-v2"]),
            ("C3", F["res"], {m: s["had"] for m, s in F["res"].items()})]
    merged = {}
    for name, res, h in rows:
        line, ma = row(name, res, h)
        merged[name] = ma
        print(line)
    allm = [str(m) for m in LEARNING_MATCHES]
    print("\nPass detection (same passer, one-to-one), P / R / F1:\n")
    print("| contestant | ±10 s | ±2 s | ±0.5 s |\n|---|---|---|---|")
    for name, res in (("C1-tuned", R["res"]["C1-tuned"]), ("C2-tuned", R["res"]["C2-tuned"]), ("C2-v2", V["res"]), ("C3", F["res"])):
        mp = merge_passes([res[m]["passes"] for m in allm])
        print(f"| {name} | " + " | ".join(f"{mp[w]['precision']:.2f} / {mp[w]['recall']:.2f} / {mp[w]['f1']:.2f}"
                                          for w in ("10s", "2s", "0.5s")) + " |")
    conf_table(merged["C3"], "Confusion, C3")
    conf_table(merged["C2-v2"], "Confusion, C2-v2")
    print("\nSituation breakdown (state accuracy % by answer-key rule, all 30):\n")
    rules = sorted(merged["C0"]["by_rule"].items(), key=lambda kv: -kv[1]["frames"])
    rules = [r for r, v in rules if v["frames"] >= 3000][:14]
    print("| answer-key rule | frames | " + " | ".join(merged) + " |\n|---|---|" + "---|" * len(merged))
    for r in rules:
        print(f"| {r} | {merged['C0']['by_rule'][r]['frames']} | " +
              " | ".join(f"{100 * merged[k]['by_rule'].get(r, {}).get('acc', float('nan')):.0f}" for k in merged) + " |")
    # 5. features
    print("\n## 5. C3 top 15 features (normalised gain, summed over the three models, mean over folds)\n")
    for i, (n, v) in enumerate(sorted(F["importance"].items(), key=lambda x: -x[1])[:15], 1):
        print(f"{i}. {n} ({v:.3f})")
    print("\nDecoder / LightGBM choice per fold:")
    for k, c in F["choice"].items():
        print(f"- fold {k}: LightGBM {json.dumps({x: round(y, 3) if isinstance(y, float) else y for x, y in c['lgb'].items()})}; "
              f"decoder {json.dumps({x: round(y, 2) for x, y in c['hmm'].items()})}")
    # 6. errors
    print("\n## 6. C3 biggest error types (held-out predictions; time = video seconds)\n")
    err, runs = Counter(), {}
    for m in LEARNING_MATCHES:
        p = np.load(C3D / "pred" / f"final_{a.final}" / f"{m}.npz")
        e = dict(np.load(BASE / "evalset" / f"{m}.npz"))
        t = np.load(BASE / "inputs" / f"{m}.npz")["t"]
        st, ps = e["state"], p["state"]
        sc = e["score_mask"] & (st <= 3) & (e["period"] > 0)
        wrong = sc & (ps != st)
        lab_r = e["rule_names"][e["rule"]]
        key = np.array([f"{STATES4[x]} ({lr}) -> {STATES4[y]}" for x, lr, y in zip(st[wrong], lab_r[wrong], ps[wrong])])
        err.update(key.tolist())
        idx = np.flatnonzero(wrong)
        if len(idx):
            br = np.flatnonzero(np.diff(idx) > 1)
            for s_, e_ in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]):
                kk = key[np.searchsorted(idx, s_)]
                runs.setdefault(kk, []).append((float(t[e_] - t[s_]), int(m), round(float(t[s_]), 1)))
    tot = sum(err.values())
    for k, v in err.most_common(5):
        ex = sorted(runs.get(k, []), reverse=True)[:2]
        print(f"- {k}: {v} frames ({100 * v / tot:.1f}% of errors); e.g. " + "; ".join(f"match {m} @ {t0} s ({d:.1f} s)" for d, m, t0 in ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
