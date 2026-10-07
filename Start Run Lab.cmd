@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run setup.cmd once first - it needs Python 3.11.
  pause
  exit /b 1
)
echo Starting RunTrack - running form lab...
echo Browser opens at http://127.0.0.1:8780/ - keep this window open.
".venv\Scripts\python.exe" app.py %*
pause
