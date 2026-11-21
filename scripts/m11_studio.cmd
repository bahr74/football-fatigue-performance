@echo off
rem M11 Studio (Session A): double-click to start, then open http://127.0.0.1:8620 (close this window to stop).
rem Imports are pinned to this worktree's src (see scripts\session_a_python.cmd); the check below refuses to start otherwise.
cd /d "%~dp0.."
call "%~dp0session_a_python.cmd" -m football_fatigue.worktree >nul
if errorlevel 1 (
  echo Import isolation check FAILED: football_fatigue does not resolve to %CD%\src
  call "%~dp0session_a_python.cmd" -m football_fatigue.worktree
  pause
  exit /b 1
)
echo Starting the M11 Studio on http://127.0.0.1:8620 ...
call "%~dp0session_a_python.cmd" -m football_fatigue.m11_dashboard --port 8620 studio
pause
