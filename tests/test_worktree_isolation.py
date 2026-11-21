"""Session A import isolation: a decoy ``football_fatigue`` on an inherited PYTHONPATH (as from the Session B worktree) must be
detected, and the Session A launch path (PYTHONPATH = this worktree's src) must win over it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from football_fatigue import worktree as WT

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _decoy(tmp_path: Path) -> Path:
    d = tmp_path / "decoy_src" / "football_fatigue"
    (d / "m11_dashboard").mkdir(parents=True)
    (d / "__init__.py").write_text("DECOY = True\n")
    (d / "m11_dashboard" / "__init__.py").write_text("")
    return tmp_path / "decoy_src"


def _run(env: dict, code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=120).stdout.strip()


def test_current_process_imports_this_worktree():
    d = WT.diagnose(ROOT)
    assert d["ok"], d
    assert WT.this_src() == SRC.resolve()


def test_inherited_pythonpath_decoy_is_detected(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(_decoy(tmp_path)))
    out = _run(env, "import football_fatigue as f; print(f.__file__)")
    assert "decoy_src" in out                                           # the failure mode is real
    rc = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); from football_fatigue import worktree" % SRC], env=env,
                        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert rc.returncode == 0
    probe = ("import sys, json; sys.path.insert(0, r'%s'); import football_fatigue.worktree as W; "
             "import importlib; d = W.diagnose(r'%s', ('football_fatigue',)); print(d['ok'])") % (SRC, ROOT)
    assert _run(env, probe) == "True"                                   # src inserted first wins


def test_child_env_overrides_decoy(tmp_path):
    env = WT.child_env(SRC, dict(os.environ, PYTHONPATH=str(_decoy(tmp_path))))
    assert env["PYTHONPATH"] == str(SRC.resolve()) and env["PYTHONNOUSERSITE"] == "1"
    out = _run(env, "import football_fatigue.m11_dashboard as m; print(m.__file__)")
    assert Path(out).resolve().is_relative_to(SRC.resolve())


def test_diagnose_flags_wrong_expectation(tmp_path):
    d = WT.diagnose(tmp_path, ("football_fatigue",))
    assert not d["ok"] and not d["modules"][0]["inside_expected"]
    with pytest.raises(RuntimeError):
        WT.assert_isolated(tmp_path, ("football_fatigue",))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_session_a_launcher_beats_inherited_pythonpath(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(_decoy(tmp_path)))
    r = subprocess.run(["cmd", "/c", str(ROOT / "scripts" / "session_a_python.cmd"), "-m", "football_fatigue.worktree"], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "IMPORT ISOLATION OK" in r.stderr
