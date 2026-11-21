@echo off
rem M11 Trajectory Recovery Lab: double-click to start, then open http://127.0.0.1:8610 (close this window to stop).
rem Port 8610 is used because Windows reserves 8494-8593 on this machine.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found: .venv\Scripts\python.exe
  pause
  exit /b 1
)
echo Starting the M11 Trajectory Recovery Lab on http://127.0.0.1:8610 ...
".venv\Scripts\python.exe" -m football_fatigue.m11_dashboard --port 8610 lab
pause
