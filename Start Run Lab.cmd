@echo off
cd /d "%~dp0"
echo Starting RunTrack (running form lab)...
echo Browser will open at http://127.0.0.1:8780/ — keep this window open.
".venv\Scripts\python.exe" app.py
pause
