# RunTrack — running form tracker (local, Windows)

Live joint tracking and running-form analytics on a treadmill, for USB cameras
and phone cameras. Local only (binds 127.0.0.1); nothing is uploaded.

## Run

Double-click **Start Run Lab.cmd**, or:

```
.venv\Scripts\python.exe app.py
```

then open http://127.0.0.1:8780/ (opens automatically). Keep the window open.

Flow: **Scan** → pick camera → **Open camera** → pick the **camera view**
(sagittal / frontal) → fill patient **name + phone** (required) → **Start run
session** (default 60 s) → the report builds automatically when the run ends.
Live joint angles appear on the skeleton while running.

## Camera sources

- **USB camera** — built-in or external USB/UVC webcams (Microsoft LifeCam and
  similar work as-is). Up to 8 indexes are scanned and devices are listed with
  their Windows device names when available; capture uses DirectShow with an
  automatic Media Foundation fallback. Close other apps (Zoom/Teams) that may
  own the camera; check Windows camera privacy settings.
- **Phone camera (URL)** — install an IP-camera app on the phone (Android:
  “IP Webcam” → Start server → `http://PHONE-IP:8080/video`). Any app that
  exposes an MJPEG/RTSP URL works; phone and PC must be on the same Wi-Fi.
- **Video file** — a recording (MP4 / MOV / WebM, e.g. filmed with the phone at
  the treadmill). Upload it and it **rolls (plays) through the tracker while
  joint tracking and metrics update live**, frame by frame; you can record a
  session on it. A playback bar offers **slow motion (0.25× / 0.5× / 0.75× /
  1.0×)** and frame/time **scrubbing** for close inspection.

Offline analysis of a video file without the app UI:

```
.venv\Scripts\python.exe tools\analyze_video.py video.mp4 ^
   --start 5 --duration 60 --name "Jane" --phone 555 --weight 70 --height 175 --speed 10
```

## What is measured

- Joint angles, live + stored per frame: ankle, knee, hip, shoulder, elbow,
  neck–trunk, head tilt, trunk lean, shoulder tilt (33-point pose model;
  landmarks are smoothed with a 1-Euro filter to remove jitter without lag).
- Movement: per-joint range of motion and mean angular velocity (deg/s).
- Cadence (steps/min) from foot-strike detection, cross-checked with
  hip-oscillation frequency.
- Vertical oscillation (cm) of the hips; step/stride length from treadmill speed.
- Left/right symmetry per joint (mean-angle agreement, %; also an asymmetry
  index from stance time and knee angles).
- Estimates: vertical mechanical power `m·g·h·f` (needs weight + height), also
  reported per kg. All estimates are labelled as such in the report.

### Biomechanics (view-plane aware)

Pick the camera view when you open a session — the other plane’s warnings are
suppressed instead of guessed:

- **Sagittal (side view)** — touchdown shin angle (knee→ankle vs vertical at
  contact), overstride vector (landing ankle ahead of the pelvis at contact,
  in leg lengths; a height-based estimate when no scale is known), dynamic
  knee flexion at initial contact and mid-stance, anterior trunk lean
  (forward-normalised).
- **Frontal (front/back view)** — contralateral pelvic drop (Trendelenburg
  proxy), dynamic knee valgus/varus (medial collapse positive), arm-swing
  symmetry.

Metrics from the other plane display as “not assessed (wrong camera view)”
everywhere: live view, session summary, risk flags and the report.

### Gait events & keyframes

The three gait phases are latched automatically — **initial contact** (lowest
foot / contact), **mid-stance** (deepest load-bearing knee flexion), **toe-off**
(foot lift-off) — and annotated freeze-frame snapshots are saved into the
session and embedded in the report. When the lower body is not visible enough,
the session states “gait events not assessable (…)” — never a silent zero.

### Audio biofeedback (treadmill, eyes forward)

- **Metronome**: Web-Audio click, 120–240 BPM, ±5 steppers, 170 / 180 presets,
  “sync to live cadence”.
- **Spoken cues** (“Cadence low, increase step rate”, “Form stable”): max one
  cue per ~12 s, thresholds in `config.py`.

Both are off by default; one click enables audio (browser autoplay rule).

### Patient history & before/after

Sessions are permanent under the patient (name + phone). The History panel
lists a patient’s sessions (latest 30 in the UI); **Compare** puts any
**Baseline** vs **Follow-up** side by side with Δ values (ΔCadence, ΔSymmetry,
ΔKnee flexion, ΔOscillation, ΔForm score) and colour-coded
Improved / Neutral / Regressed badges.

## Reports

Everything is stored under `data/sessions/<id>/`:

- `report.html` — summary cards, **overall Form score (0–100)**, **injury-risk
  strata (Low / Moderate / Elevated / High)** with PT drill prescriptions,
  joint table, symmetry, form bands vs reference bands, gait keyframes, charts,
  a **Print / Save as PDF** button (A4/Letter print CSS), AI review
- `metrics.csv` — full per-frame timeseries (all legacy columns preserved)
- `summary.json` — includes `form_score`, `risk`, and honesty notes
- `chart_*.png`, `frame_*.jpg` (annotated snapshots), `keyframes/` (gait phases)
- `coach.json` — when the local-AI review has been run

Form score = weighted scoring of each measured metric (100 inside its
reference band, linear decay to 0 at the per-metric tolerance); weights are
re-normalised over what the session actually measured (documented in
`advisor.py`; e.g. sagittal: symmetry 0.22, cadence 0.18, touchdown shin 0.15,
mid-stance knee flexion 0.13, overstride 0.12, trunk lean 0.10, oscillation
0.10).

“Run local AI form review” sends the summary + snapshots to the **local** model
(Ollama, `qwen3.8:latest`); it never leaves the PC. The first call loads the
model (~40 s) and the whole review can take 1–3 minutes.

## Tests

```
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

109 tests: metrics + biomechanics math, 1-Euro filter, synthetic gait-cycle
ground-truth latching, advisor scoring/compare, storage, real-video pipeline,
report/API endpoints.

## Limits (be honest about them)

- 2D angles from a single camera view. Best results: **side (sagittal) view**,
  whole body in frame, camera steady; frontal view for the frontal metrics.
- Gait events need the **lower body visible** (ankle/foot landmarks). Without
  it the session says “gait events not assessable” — thresholds are never
  relaxed to force events out of bad footage.
- Slow motion and scrubbing apply to video files; a live camera stream has no
  playback.
- Overstride (and any cm-based estimate without a known scale) is a
  height-based estimate — labelled as such.
- MediaPipe left/right labels follow its own convention — verify against the
  camera orientation.
- Reference bands, the form score, risk strata and PT drills are screening
  heuristics from running-gait literature and common physiotherapy reasoning —
  decision support, not a diagnosis, and not validated on this camera setup.
  This is not a medical device.
- If the frame rate is low, reduce `PROC_WIDTH` in `config.py` (e.g. 720) or
  use a USB cable for the phone instead of Wi-Fi streaming.
