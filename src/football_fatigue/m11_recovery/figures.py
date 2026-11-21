"""Report figures: correct, wrong and abstained recoveries of a model's development out-of-fold predictions (seeded selection,
no cherry-picking: the first examples in a seeded random order of each outcome). Writes PNGs to
``m11-recovery-v1/figures/<experiment>-<model>/`` once. Run:
``python -m football_fatigue.m11_recovery.figures --exp e1-baselines-gbm --model gbm``.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root

from .experiments import SOURCE, out_root
from .views import load_match, make_sample


def draw(ax, s, p, outcome):
    cols = ("tab:blue", "tab:orange")
    for i in (0, 1):
        ax.plot(*s.before[i].T, "-o", ms=2, color=cols[i])
    for j in (0, 1):
        ax.plot(*s.after[j].T, ":x", ms=3, color="0.4")
        ax.annotate("AB"[j], s.after[j][0], fontsize=8)
    for i in (0, 1):                                              # provider continuity (reference)
        j = i if s.y == 1 else 1 - i
        ax.plot([s.before[i][-1, 0], s.after[j][0, 0]], [s.before[i][-1, 1], s.after[j][0, 1]], "-", color=cols[i], lw=0.8, alpha=0.6)
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"{outcome} · gap {s.gap_s:g} s · noise {s.noise_m:g} m\nP(keep)={p:.2f} provider={'keep' if s.y else 'swap'}", fontsize=7)
    ax.tick_params(labelsize=6)


def build(raw_root, exp: str, model: str, per: int = 8, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from football_fatigue.intersections.__main__ import write_dir

    dest = out_root(raw_root) / "figures" / f"{exp}-{model}"
    if dest.exists():
        raise IngestionError(f"{dest} already exists and is never overwritten")
    edir = out_root(raw_root) / "experiments" / exp
    with np.load(edir / "oof.npz") as z:
        d = {k: z[k] for k in z.files}
    c = json.loads((edir / "results.json").read_text(encoding="utf-8"))["results"][model]["thresholds_fitted_on_this_oof"]["fa_05"]
    p = d[f"p_{model}"]
    conf = np.maximum(p, 1 - p)
    ok = (p > 0.5).astype(int) == d["y"]
    outcome = np.where(conf < c, "abstained", np.where(ok, "correct", "wrong"))
    rng = np.random.default_rng(seed)
    cache: dict[str, dict] = {}
    blobs, picked = {}, {}
    for o in ("correct", "wrong", "abstained"):
        idx = rng.permutation(np.flatnonzero(outcome == o))[:per]
        fig, axes = plt.subplots(2, per // 2, figsize=(3 * per // 2, 6.4))
        for ax, i in zip(axes.flat, idx):
            mid = str(d["match"][i])
            if mid not in cache:
                cache[mid] = {it.event_id: it for it in load_match(approved_output_root(raw_root) / SOURCE / mid)}
            s = make_sample(cache[mid][str(d["event"][i])], float(d["gap"][i]), float(d["noise"][i]))
            draw(ax, s, float(p[i]), o)
        fig.suptitle(f"{model}: {o} (5 % false-association operating point, threshold {c:.3f}); development out-of-fold; "
                     "thin lines = provider continuity", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        blobs[f"{o}.png"] = buf.getvalue()
        picked[o] = [dict(match=str(d["match"][i]), event=str(d["event"][i]), gap=float(d["gap"][i]), noise=float(d["noise"][i]), p=float(p[i])) for i in idx]
    blobs["examples.json"] = json.dumps(dict(experiment=exp, model=model, threshold=c, seed=seed, examples=picked), indent=1).encode()
    write_dir(dest, blobs)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_recovery.figures")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    print(build(str(Path(a.root).resolve()), a.exp, a.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
