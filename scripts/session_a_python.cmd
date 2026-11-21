@echo off
rem Session A Python: runs the shared .venv with THIS worktree's src first, for this process only (no global change).
rem Usage:  scripts\session_a_python.cmd -m football_fatigue.worktree
rem         scripts\session_a_python.cmd -m pytest tests\test_worktree_isolation.py
setlocal
set "SA_ROOT=%~dp0.."
for %%I in ("%SA_ROOT%") do set "SA_ROOT=%%~fI"
if not exist "%SA_ROOT%\.venv\Scripts\python.exe" (
  echo Virtual environment not found: "%SA_ROOT%\.venv\Scripts\python.exe"
  exit /b 1
)
rem Replace (not extend) any inherited PYTHONPATH: it is how the Session B worktree's src leaks in.
set "PYTHONPATH=%SA_ROOT%\src"
set "PYTHONNOUSERSITE=1"
cd /d "%SA_ROOT%"
"%SA_ROOT%\.venv\Scripts\python.exe" %*
exit /b %ERRORLEVEL%
