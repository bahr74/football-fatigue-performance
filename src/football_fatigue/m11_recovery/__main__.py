"""CLI: ``python -m football_fatigue.m11_recovery dashboard [--port 8502]`` launches the read-only recovery viewer;
``python -m football_fatigue.m11_recovery infer --model gbm-v1 --input tracks.json [--target fa_05]`` recovers the assignment of
two before tracks to two after tracks. ``tracks.json``: ``{"t_before": [...], "before": [[[x, y], ...], [[x, y], ...]],
"t_after": [...], "after": [...]}`` (seconds, metres; null = missing). Prints the decision as JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_recovery")
    ap.add_argument("--root", default="data/raw")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dashboard")
    d.add_argument("--port", type=int, default=8502)
    d.add_argument("--address", default="127.0.0.1")
    d.add_argument("--headless", action="store_true")
    i = sub.add_parser("infer")
    i.add_argument("--model", required=True, help="frozen model directory name under m11-recovery-v1/models")
    i.add_argument("--input", required=True)
    i.add_argument("--target", default=None, help="operating point, e.g. fa_01 / fa_05 (default: the model's)")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    if a.cmd == "infer":
        from football_fatigue.ingestion import approved_output_root

        from .experiments import ROOT_NAME
        from .infer import RecoveryModel

        rm = RecoveryModel.load(approved_output_root(root) / ROOT_NAME / "models" / a.model)
        x = json.loads(Path(a.input).read_text(encoding="utf-8"))
        arr = lambda v: np.array([[[np.nan, np.nan] if p is None else p for p in tr] for tr in v], float)  # noqa: E731
        r = rm.recover(arr(x["before"]), x["t_before"], arr(x["after"]), x["t_after"], a.target)
        print(json.dumps(r.as_dict() | dict(model=a.model, scope=rm.spec["scope"]), indent=1))
        return 0
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port), "--server.address", a.address,
                            "--server.headless", str(a.headless).lower(), "--browser.gatherUsageStats", "false", "--", "--root", root])


if __name__ == "__main__":
    sys.exit(main())
