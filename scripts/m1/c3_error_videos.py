"""M1: video check list for C3's long errors (held-out C3 predictions on the 30 learning matches; no training).

For C3's 5 biggest error types (answer-key state + rule -> C3 state, by wrong frames) the 2 longest contiguous stretches
each. Times are video seconds (the clock of PFF eventTime and film-room links; the inputs' `t`). The film-room link reuses
the nearest PFF event's videoUrl with the time set to the stretch start. PFF events are read here only to describe the
answer key and to build the link; nothing is fed to a model.

    python scripts/m1/c3_error_videos.py --final custom-1110
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from m1_labels.scoring import STATES4
from m1_possession.smoother import FLAG_ESTIMATED, FLAG_MISSING
from m1_possession.splits import LEARNING_MATCHES

BASE = Path("data/processed/m1_possession")
EVENTS = Path("data/raw/FIFA World Cup 2022/Event Data")
OUT = Path("reports/m1_possession/c3_error_videos.md")
PE_NAMES = {"PA": "pass", "CR": "cross", "SH": "shot", "CL": "clearance", "CH": "challenge", "RE": "rebound",
            "TC": "touch", "BC": "ball carry", "IT": "initial touch"}
GE_NAMES = {"OUT": "ball out", "END": "period end", "SUB": "substitution", "FIRSTKICKOFF": "kick-off",
            "SECONDKICKOFF": "kick-off", "OFF": "offside", "VAR": "VAR"}


def runs_of(key: np.ndarray, idx: np.ndarray):
    """Contiguous runs (consecutive frame indices with the same error key)."""
    br = np.flatnonzero((np.diff(idx) > 1) | (key[1:] != key[:-1]))
    starts, ends = np.r_[0, br + 1], np.r_[br, len(idx) - 1]
    return [(idx[s], idx[e], key[s]) for s, e in zip(starts, ends)]


def load_events(mid: int):
    rows = []
    for e in json.loads((EVENTS / f"{mid}.json").read_text(encoding="utf-8")):
        ge, pe = e.get("gameEvents") or {}, e.get("possessionEvents") or {}
        if e.get("eventTime") is None:
            continue
        rows.append(dict(t=float(e["eventTime"]), url=ge.get("videoUrl"), clock=ge.get("startFormattedGameClock"),
                         ge=ge.get("gameEventType"), pe=pe.get("possessionEventType"),
                         out=pe.get("passOutcomeType") or pe.get("shotOutcomeType") or pe.get("crossOutcomeType")))
    rows.sort(key=lambda r: r["t"])
    return rows


def describe(ev, t0, t1, rules):
    def name(r):
        s = PE_NAMES.get(r["pe"]) or GE_NAMES.get(r["ge"]) or (r["pe"] or r["ge"] or "?")
        return s + (f" ({r['out']})" if r["out"] else "")
    before = [r for r in ev if r["t"] <= t0 and (r["pe"] or r["ge"] in GE_NAMES)]
    inside = [r for r in ev if t0 < r["t"] <= t1 and (r["pe"] or r["ge"] in GE_NAMES)]
    after = [r for r in ev if r["t"] > t1 and (r["pe"] or r["ge"] in GE_NAMES)]
    parts = []
    if before:
        parts.append(f"after {name(before[-1])} at {before[-1]['t']:.1f}")
    parts.append(f"{len(inside)} PFF events inside" + (": " + ", ".join(name(r) for r in inside[:4]) + (" …" if len(inside) > 4 else "")
                                                        if inside else ""))
    if after:
        parts.append(f"next {name(after[0])} at {after[0]['t']:.1f}")
    parts.append("key rules " + ", ".join(f"{r} {p:.0f}%" for r, p in rules))
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    err, stretches = Counter(), []
    for m in LEARNING_MATCHES:
        p = np.load(BASE / "c3" / "pred" / f"final_{a.final}" / f"{m}.npz")
        e = dict(np.load(BASE / "evalset" / f"{m}.npz"))
        st, ps = e["state"], p["state"]
        wrong = e["score_mask"] & (st <= 3) & (e["period"] > 0) & (ps != st)
        idx = np.flatnonzero(wrong)
        if not len(idx):
            continue
        rn = e["rule_names"][e["rule"]]
        key = np.array([f"{STATES4[x]} ({r}) -> {STATES4[y]}" for x, r, y in zip(st[idx], rn[idx], ps[idx])])
        err.update(key.tolist())
        stretches += [(m, s, en, k) for s, en, k in runs_of(key, idx)]
    top = [k for k, _ in err.most_common(5)]
    tot = sum(err.values())
    cache, lines = {}, []
    for k in top:
        cand = []
        for m, s, en, kk in stretches:
            if kk != k:
                continue
            if m not in cache:
                cache[m] = np.load(BASE / "inputs" / f"{m}.npz")["t"]
            cand.append((float(cache[m][en] - cache[m][s]), m, s, en))
        for dur, m, s, en in sorted(cand, reverse=True)[:2]:
            z = np.load(BASE / "inputs" / f"{m}.npz")
            e = dict(np.load(BASE / "evalset" / f"{m}.npz"))
            t, fl = z["t"], z["bflag"][s:en + 1]
            t0, t1 = float(t[s]), float(t[en])
            ev = load_events(m)
            near = min((r for r in ev if r["url"]), key=lambda r: abs(r["t"] - t0))
            url = re.sub(r"/[0-9.]+$", f"/{t0:.3f}", near["url"])
            c = Counter(e["rule_names"][e["rule"][s:en + 1]].tolist())
            rules = [(r, 100 * v / (en - s + 1)) for r, v in c.most_common(2)]
            lines.append(dict(type=k, share=100 * err[k] / tot, match=m, period=int(e["period"][s]), dur=dur, t0=t0, t1=t1,
                              clock=near["clock"], url=url, what=describe(ev, t0, t1, rules),
                              miss=100 * float(np.mean(fl == FLAG_MISSING)), est=100 * float(np.mean(fl == FLAG_ESTIMATED))))
    md = [f"# C3 long-error video check list (C3 inputs '{a.final}', held-out predictions, 30 learning matches)\n",
          "Video time = PFF eventTime / film-room seconds. Ball missing = our smoothed raw ball has no position; ESTIMATED = "
          "raw ball visibility ESTIMATED. Match clock is the nearest PFF event's clock (approximate).\n",
          "| # | error type (answer key → C3) | % of error frames | match | period | length s | video start | video end | "
          "match clock ≈ | film room (start) | answer key: what happens | ball missing % | ball ESTIMATED % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(lines, 1):
        md.append(f"| {i} | {r['type']} | {r['share']:.1f} | {r['match']} | {r['period']} | {r['dur']:.1f} | {r['t0']:.1f} | "
                  f"{r['t1']:.1f} | {r['clock']} | [link]({r['url']}) | {r['what']} | {r['miss']:.0f} | {r['est']:.0f} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
