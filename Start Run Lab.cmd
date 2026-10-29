@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run setup.cmd once first - it needs Python 3.11.
  pause
  exit /b 1
)
echo Starting RunTrack - running form lab...
REM Full-capacity pose: RTMPose (HALPE-26, heels+toes) on the GPU via DirectML.
REM Measured on this PC: 179 fps / 100 pct frames with pose vs MediaPipe 106 fps
REM / 36 pct, ankle landmarks 46.7 pct vs 10.7 pct. Force the old engine with
REM   set RUNTRACK_POSE_BACKEND=mediapipe
if "%RUNTRACK_POSE_BACKEND%"=="" set RUNTRACK_POSE_BACKEND=rtmpose
if "%RUNTRACK_ORT_PROVIDER%"=="" set RUNTRACK_ORT_PROVIDER=auto
echo Pose engine: %RUNTRACK_POSE_BACKEND% (provider: %RUNTRACK_ORT_PROVIDER%)
echo Browser opens at http://127.0.0.1:8780/ - keep this window open.
".venv\Scripts\python.exe" app.py %*
pause
