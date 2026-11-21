"""Worktree import isolation for Session A.

Session A (``F:\\Football-Fatigue-Project``) and Session B (``F:\\Football-Fatigue-Video``, a git worktree of the same repository)
share one virtual environment. An inherited ``PYTHONPATH`` or an editable reinstall from the other worktree can make
``import football_fatigue`` resolve to the wrong ``src`` (observed: ``m11_dashboard`` loaded from the Video worktree, so the
``studio`` command did not exist). Nothing global is changed here; launchers set ``PYTHONPATH`` for their own process only.

``python -m football_fatigue.worktree [--expect <repo root>] [--modules a,b]`` prints where each module was imported from and
exits 1 when any of them lies outside ``<expect>/src`` (default: the repository that contains the current directory).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

CHECK_MODULES = ("football_fatigue", "football_fatigue.m11_dashboard", "football_fatigue.m11_fair", "football_fatigue.m11_explorer",
                 "football_fatigue.m11_phase2", "football_fatigue.m11_benchmark")


def repo_root(start: str | os.PathLike | None = None) -> Path | None:
    """Nearest ancestor containing ``pyproject.toml`` and ``src/football_fatigue`` (a worktree root)."""
    p = Path(start or os.getcwd()).resolve()
    for d in (p, *p.parents):
        if (d / "pyproject.toml").is_file() and (d / "src" / "football_fatigue").is_dir():
            return d
    return None


def this_src() -> Path:
    """The ``src`` directory this module itself was imported from."""
    return Path(__file__).resolve().parents[1]


def child_env(src: str | os.PathLike | None = None, env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child Python process that must import from ``src`` first (any inherited PYTHONPATH is dropped,
    because it is exactly the channel through which the other worktree leaks in)."""
    e = dict(os.environ if env is None else env)
    e["PYTHONPATH"] = str(Path(src or this_src()).resolve())
    e["PYTHONNOUSERSITE"] = "1"
    return e


def diagnose(expect: str | os.PathLike | None = None, modules=CHECK_MODULES) -> dict:
    root = Path(expect).resolve() if expect else repo_root()
    want = (root / "src").resolve() if root else None
    rows = []
    for m in modules:
        try:
            mod = importlib.import_module(m)
            f = Path(mod.__file__).resolve()
            ok = want is not None and (f == want or want in f.parents)
            rows.append(dict(module=m, file=str(f), inside_expected=bool(ok)))
        except Exception as e:  # missing subpackage = wrong worktree too
            rows.append(dict(module=m, file=None, inside_expected=False, error=repr(e)[:200]))
    return dict(expected_src=str(want) if want else None, cwd=os.getcwd(), executable=sys.executable,
                pythonpath=os.environ.get("PYTHONPATH", ""), sys_path_head=sys.path[:5], modules=rows,
                ok=want is not None and all(r["inside_expected"] for r in rows))


def assert_isolated(expect: str | os.PathLike | None = None, modules=CHECK_MODULES) -> None:
    d = diagnose(expect, modules)
    if not d["ok"]:
        bad = [r for r in d["modules"] if not r["inside_expected"]]
        raise RuntimeError(f"football_fatigue imported outside {d['expected_src']}: {bad}. Launch with scripts\\session_a_python.cmd "
                           "or set PYTHONPATH to this worktree's src for the process.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.worktree")
    ap.add_argument("--expect", help="worktree root whose src must win (default: repository containing the current directory)")
    ap.add_argument("--modules", help="comma-separated module list", default=",".join(CHECK_MODULES))
    a = ap.parse_args(argv)
    d = diagnose(a.expect, tuple(m for m in a.modules.split(",") if m))
    print(json.dumps(d, indent=1))
    print("IMPORT ISOLATION OK" if d["ok"] else "IMPORT ISOLATION FAILED", file=sys.stderr)
    return 0 if d["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
