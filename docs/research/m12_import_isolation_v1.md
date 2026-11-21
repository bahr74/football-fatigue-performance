# M12 report 1: Session A import isolation (v1)

Status: done; regression-tested. Date 2026-11-21. No global environment change, Session B untouched.

## Problem (reproduced)

Session B (`F:\Football-Fatigue-Video`) has no virtual environment of its own and runs Session A's `.venv`. A shell that inherited
`PYTHONPATH=F:\Football-Fatigue-Video\src` made `import football_fatigue` resolve to Session B's copy of the package. Session A
commands, the Studio and its Streamlit child process then silently ran Session B's code.

## Fix (process-local only)

| Piece | What it does |
|---|---|
| `scripts/session_a_python.cmd` | Sets `PYTHONPATH=<repo>\src` and `PYTHONNOUSERSITE=1` for this process only, `cd`s to the repository and runs `.venv\Scripts\python.exe` with the given arguments |
| `scripts/m11_studio.cmd` | Runs the import check through the launcher first and refuses to start if it fails; then starts the Studio on port 8620 |
| `football_fatigue.worktree` | `diagnose()`, `assert_isolated()` and `child_env()`; the CLI is `python -m football_fatigue.worktree` (JSON report, exit 1 on failure) |
| `m11_dashboard/__main__.py` | The Streamlit subprocess gets `child_env()`, so it imports the same `src` as its parent whatever the shell inherited |
| `pyproject.toml` | `[tool.pytest.ini_options] pythonpath = ["src"]`, so pytest imports this repository's `src` first |

Session B's files, `.venv` contents and user or system environment variables were not changed.

## Safe launch

```bat
scripts\session_a_python.cmd -m football_fatigue.worktree            REM prints IMPORT ISOLATION OK / FAILED
scripts\m11_studio.cmd                                                REM Studio, http://localhost:8620
scripts\session_a_python.cmd -m football_fatigue.m11_fair plan        REM any Session A module
```

A bare `.venv\Scripts\python.exe -m ...` is only safe when `PYTHONPATH` is unset or points at this repository's `src`.

## Tests

`tests/test_worktree_isolation.py` (5 tests) covers:

- a decoy `football_fatigue` package on `PYTHONPATH` is detected;
- `child_env` overrides the decoy;
- `diagnose` flags a wrong expected root;
- the Windows launcher wins over an inherited `PYTHONPATH`.

The suite also passed with `PYTHONPATH=F:/Football-Fatigue-Video/src` exported. As a negative control, collection fails when the
pytest `pythonpath` option is cleared, which shows the test really depends on the fix.
