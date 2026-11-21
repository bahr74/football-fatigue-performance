"""CLI: ``python -m football_fatigue.m11_dashboard [--root data/raw] [--port 8501]`` launches the Streamlit app;
``python -m football_fatigue.m11_dashboard verify --match <development id> [--limit N]`` compares the dashboard's per-encounter
inference with the Step 6 batch path on one development match and prints the result (nothing is written);
``python -m football_fatigue.m11_dashboard [--port 8503] encounters`` (options before the subcommand) launches the synchronized-encounter explorer
(``encounters_app.py``);
``python -m football_fatigue.m11_dashboard --port 8610 lab`` launches the M11 Trajectory Recovery Lab (``lab_app.py``);
``python -m football_fatigue.m11_dashboard --port 8620 studio`` launches the M11 Studio (``studio_app.py``: explorer, recovery lab,
N x M lab, review queue, evidence)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_dashboard")
    ap.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--address", default="127.0.0.1", help="interface to bind (default: this computer only)")
    ap.add_argument("--headless", action="store_true", help="do not open a browser or ask Streamlit's first-run questions")
    sub = ap.add_subparsers(dest="cmd")
    v = sub.add_parser("verify", help="reproduce the batch Step 6 inference on a development match")
    v.add_argument("--match", required=True)
    v.add_argument("--limit", type=int, default=None, help="first N instances only")
    sub.add_parser("encounters", help="launch the synchronized-encounter explorer (v2 events, v3 contexts; development matches)")
    sub.add_parser("lab", help="launch the M11 Trajectory Recovery Lab (live GRU C / HGB C inference on development scenarios)")
    sub.add_parser("studio", help="launch the M11 Studio (match & interaction explorer, recovery lab, N x M lab, review queue, evidence)")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    if a.cmd == "verify":
        from .data import DashboardError, load_match
        from .inference import compare_with_batch, load_frozen_model

        try:
            t0 = time.perf_counter()
            fm = load_frozen_model(root)
            md = load_match(root, a.match)
            insts = list(md.instances.values())[: a.limit]
            res = compare_with_batch(insts, fm)
        except DashboardError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        res.update(match=md.match_id, model_sha256=fm.model_sha256, spec_sha256=fm.spec_sha256, sources=md.sources,
                   seconds=round(time.perf_counter() - t0, 1))
        print(json.dumps(res, indent=1))
        return 0 if res["max_abs_probability_diff"] == 0.0 and not res["decision_mismatches"] and not res["assignment_problems"] else 1
    app = Path(__file__).with_name({"encounters": "encounters_app.py", "lab": "lab_app.py", "studio": "studio_app.py"}.get(a.cmd, "app.py"))
    from football_fatigue.worktree import child_env
    # the Streamlit child imports football_fatigue from the same src as this launcher (not from an inherited PYTHONPATH)
    return subprocess.call(env=child_env(), args=[sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port), "--server.address", a.address, "--server.headless",
                            str(a.headless).lower(), "--browser.gatherUsageStats", "false", "--", "--root", root])


if __name__ == "__main__":
    sys.exit(main())
