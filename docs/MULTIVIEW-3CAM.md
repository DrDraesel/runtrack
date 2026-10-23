# Three-camera setup — side + rear + front

RunTrack can analyse a session filmed from **three cameras at once**: a side
(sagittal) view as the primary source, a rear camera as the second view and a
front camera as the third view. Each clip is an independent recording with its
own time base and its own view plane; the reports show the views **side by
side** and never average them into one number.

This document is both the operator sheet for building the rig and the
engineering reference for how the views are analysed.

- Third view default plane: **frontal**
- Third view direction: `front` (metadata only — see *Mirror invariance*)
- Report heading: `Third view — front view (<source3_label>)`
- Report sync line: per-view frame counts + the software-clock note
- Wiring: `summary["third_view"]`, `summary["plane3"]`,
  `summary["direction3"]`, `summary["source3_label"]`,
  `summary["multi_view_sync"]`

---

## Quick start — running a three-camera session

1. **Camera card → primary source** (side camera / side clip) and set
   **Camera view = sagittal**.
2. Tick **Second view (rear / side)**, pick the rear source, set
   **Second view plane = Rear / front (frontal)**.
3. Tick **Third view (front camera)**, pick the front source, set
   **camera facing = Front (camera faces the runner)** and
   **Third view plane = Front / rear (frontal)**.
4. Fill patient name + phone and start the session. When it finalizes, the
   report carries the second-view and third-view blocks plus the multi-view
   sync line.

The **Import side + rear videos** shortcut loads the first two clips only; the
front clip is added through the third-view checkbox (USB camera, phone stream
or video file).

---

## 1. What each view measures

| View | Plane | Metrics | Role |
| --- | --- | --- | --- |
| **SIDE** | sagittal | trunk lean (forward +), touchdown shin angle vs vertical, overstride vector (landing ankle ahead of the pelvis, leg-length units), knee flexion at initial contact and mid-stance, hip extension at toe-off, push-off ankle angle, stance time, cadence, vertical oscillation, power estimate | **Quantitative core** — most metrics come from here; also the critical view for bike fitting |
| **REAR** | frontal | contralateral pelvic drop (Trendelenburg proxy), frontal knee collapse (`knee_valgus_deg`), arm-swing symmetry (`arm_swing_sym_pct`), inter-hip pelvic tilt (`pelvic_tilt_deg`, + = right hip low) | Frontal control that a side view cannot measure |
| **FRONT** | frontal | the same frontal family seen from the front (pelvic drop, knee valgus, arm-swing symmetry, pelvic tilt), plus an upper-body / treadmill check: shoulder level, head-and-neck posture, whether the runner drifts or crosses the deck centreline | Same frontal metrics as the rear, from the opposite side; least metric-critical view |

Plane gating is absolute: a sagittal metric is never invented from a frontal
clip and vice versa. Metrics from the other plane are reported as
`not assessed (wrong camera view)` in the live view, the summary, the risk
flags and the report.

### Mirror invariance (front vs rear needs no axis correction)

The frontal maths is mirror-invariant, so a front clip and a rear clip produce
comparable numbers without flipping axes:

- the body **midline is derived from the pelvis** (mid-hip x), not from a fixed
  image axis, so a left-right mirror of the frame moves the midline with it;
- **pelvic tilt is y-based** (the vertical difference between the left and right
  hip landmarks), so it is unaffected by which side of the body faces the
  camera;
- knee valgus/varus is measured as the knee's deviation from the hip–ankle
  axis, and arm-swing symmetry compares left/right wrist amplitudes — both are
  internal angles/ratios that survive a mirror.

`direction3` (`front` | `rear`) is therefore **metadata only**: it selects the
word in the report heading and tells the operator which tripod fired. It never
triggers a coordinate flip, and it does not change the metric definitions.

---

## 2. Camera placement

| View | Distance | Lens height | Position / aim | Why |
| --- | --- | --- | --- | --- |
| **SIDE** | 2.5–3.5 m, perpendicular to the treadmill | ~1.0–1.1 m (hip height) | Runner centred, full body in frame plus a margin of floor visible | Sagittal angles need the leg silhouette perpendicular to the image plane. Hip height keeps the pelvis and knee near the optical axis, so 2D angles are not skewed by perspective; the floor margin captures foot contact and keyframes. |
| **REAR** | 2.5–3 m | ~1.0–1.1 m | Directly behind, centred on the runner's midline, no obstructions between lens and hips | Pelvic tilt is a y-difference in the image, so the sensor must be level (camera roll fakes a drop). Centred framing keeps left/right scale comparable. Handle bars, rails or a person in the way will hide exactly the landmarks the view exists for. |
| **FRONT** | 3–4 m | ~1.1–1.2 m, slight downward tilt | **Offset ~0.5 m to the left or right of the console line**, runner centred, deck visible in the lower part of the frame | The treadmill display/console is the main failure mode: straight-on framing hides the legs behind it. The offset keeps the legs clear while the slight tilt keeps both the torso (upper-body check) and the feet/deck in frame. A longer distance reduces wide-angle distortion. |

Practical rules for all three:

- Put all three tripods at the **same height where practical** — it makes the
  views easier to compare visually and keeps perspective effects consistent.
- **Record the measured heights (and distances) in the session notes.** If a
  camera moves, the session is no longer comparable with earlier ones.
- Lock focus/exposure/white balance on webcams that allow it; autofocus
  hunting changes limb apparent size frame to frame.
- Keep the runner's whole body inside the frame for the entire set. A foot that
  leaves the frame is a missing landmark, not a measurement.

### Recommendation: best sensor on the SIDE

Sagittal carries the most metrics, the form-score weights and the plane-gated
risk rules, and it is also the critical view for bike fitting. Put the best
sensor there.

The **front is the least metric-critical view** — it repeats the frontal family
already covered by the rear and adds an upper-body/treadmill sanity check — so
it is a good place for a Kinect Azure. The **rear needs decent light more than
resolution**: pelvic tilt and knee valgus are computed from landmarks, and a
noisy, under-lit silhouette hurts far more than a lower pixel count.

---

## 3. Heterogeneous cameras

Mixing camera models, frame rates and exposures is fine. Each view is analysed
on its own time base and its own plane (`second_view.analyze` is run once per
extra view; the same code produces the second and third blocks).

- **Ordinary webcams:** 720p @ 30 fps is the recommended operating point.
- **Kinect Azure / Femto:** use the **1080p RGB stream** and its **RGB pin
  only — never the depth/IR pins.** Depth and IR frames are not RGB pose input
  and will not produce comparable landmarks.
- Per-view fps is shown in the UI; check it before trusting a session.
- **Do not compare pixel-level values across views.** Different lenses, fields
  of view, exposure and distance make pixels meaningless across cameras; the
  metrics are angles and normalized ratios precisely so that this does not
  matter. Compare trends within a view, not raw pixels between views.

---

## 4. USB and bandwidth

- A **Kinect Azure needs a USB 3.x port on its own controller.** It will not
  behave on a shared USB 2 hub or a busy controller.
- Put the webcams on **separate ports**; avoid unpowered hubs (three cameras on
  one bus is a classic source of dropped frames).
- If frames drop, **lower the webcams to 720p / 30 fps first**, then move them
  to another controller. Dropped frames shorten a view's usable time base; they
  do not corrupt the other views.
- Close Zoom/Teams and any other app that may own a camera; check the Windows
  camera privacy settings if a device is not listed.

---

## 5. Lighting

- **Even front lighting.** Avoid a bright window behind the runner — backlight
  turns the body into a silhouette and the pose model loses limbs.
- **Avoid flickering fluorescents.** Mains flicker plus a rolling shutter
  produces banding and exposure pumping, which shows up as jitter in the
  metrics (especially pelvic tilt and knee valgus).
- Matte clothing that contrasts with the floor and the treadmill deck helps the
  landmark model; avoid logos/patterns that break up the limb edges.
- No strobe or stage lighting; keep the light the same across sessions you
  intend to compare.

---

## 6. Synchronisation

There is **no hardware genlock** in this rig. Each view carries its own
`t_rel` timestamps, relative to that view's own start, and each view is analysed
on that time base.

- Expect up to **~1 frame (33 ms at 30 fps)** of offset between views.
- If frame-level alignment matters (e.g. comparing the same foot-strike across
  views), do a **single clap at the start of the session** in view of all three
  cameras and use it as a visual sync marker. Do not clap repeatedly.
- Per-view frame counts are stored in `summary["multi_view_sync"]["views"]` and
  printed in the report's `Multi-view sync` line, together with the
  software-clock / no-genlock note.
- `multi_view_sync["clocks"]` is `"software"`. No resampling or frame shifting
  is applied: the numbers in each view are that view's own numbers.

---

## 7. Cycling fitting with the same rig

The same three cameras serve bike fitting; the metric set already exists in
`cycling.py` / `config.CYCLING_BANDS`.

| View | What to read | Reference (config) |
| --- | --- | --- |
| **SIDE** (critical) | knee angle at bottom dead centre (BDC), hip angle at top dead centre (TDC), torso angle vs horizontal, ankle angle | knee BDC 140–150° road / 140–145° TT-tri; hip TDC 70–90° road / 45–75° TT-tri; torso 35–45° road / 10–25° TT / 15–30° tri; ankle 80–100° road / 100–120° TT-tri |
| **REAR** | hip rocking (pelvic obliquity), knee tracking (lateral travel), shoulder/ankle control | pelvic obliquity ≤ 5°; knee lateral travel avg ≤ 40 mm / good ≤ 20 mm (peak-to-peak) |
| **FRONT** | left-right knee symmetry, arm position, that the console does not hide the legs | read as left/right agreement, no separate band |

Change one fit dimension at a time, **10 mm per step** (5 mm is below 2-D resolution), and re-film. The rear
travel numbers are peak-to-peak excursions relative to a rolling mean — treat
them as control indicators, not laboratory kinematics.

---

## 8. Limitations (be honest about them)

- **Per-view 2D estimates, not 3D triangulation.** Three cameras do not give
  you a 3D reconstruction here; each view is measured independently in its own
  image plane.
- **Plane gating.** Wrong-plane metrics read `not assessed (wrong camera view)`
  rather than a guessed number. A frontal clip cannot produce overstride or
  shin angle, and a side clip cannot produce pelvic drop.
- **No hardware genlock.** Software timestamps only; up to ~1 frame of offset
  between views.
- **No medical diagnosis.** Reference bands, form scores, risk strata and PT
  drills are screening heuristics, not a diagnosis; RunTrack is not a medical
  device.
- **Camera roll and bumps matter.** A tilted rear camera fabricates pelvic
  tilt; a tripod knocked mid-session invalidates comparison with earlier sets.
- **Scale-dependent values** (cm/mm) need height or shoulder width; otherwise
  they are reported as normalized ratios or labelled estimates.
- **MediaPipe left/right labels** follow its own convention; verify against the
  camera orientation.
- **Occlusion is the dominant error source**: console, hands, treadmill rails
  and loose clothing hide the landmarks the maths depends on.

---

## 9. Engineering contract (for maintainers)

At finalize, a three-camera session may carry:

```jsonc
{
  "third_view": { /* same shape as second_view:
                    view_plane, form, form_score, form_score_detail,
                    risk, not_assessed, rows, duration_s */ },
  "plane3": "frontal",
  "direction3": "front",          // "front" | "rear" — metadata only
  "source3_label": "front.mp4",
  "multi_view_sync": {
    "clocks": "software",
    "views": [
      {"role": "primary", "label": "side.mp4",  "rows": 1800},
      {"role": "second",  "label": "rear.mp4",  "rows": 1795},
      {"role": "third",   "label": "front.mp4", "rows": 1800}
    ],
    "note": "no hardware genlock; aligned by per-view t_rel timestamps; expect up to ~1 frame offset"
  }
}
```

`report.py` renders the third-view block **only when `third_view` is present**;
`plane3` / `direction3` / `source3_label` alone produce nothing. The block
contains its own metric table, its own `not_assessed` rows, its own form score,
its own risk flags, the plane-gating note, and — when `multi_view_sync` exists —
the per-view frame-count line. The second-view block is untouched: existing
two-camera sessions render exactly as before.

---

## 10. This room — measured rig and target positions (TRUE Alpine Runner)

Measured on the installation 2026-10-23 from the live feeds, not from spec sheets.
Machine: TRUE Alpine Runner, deck 152 × 56 cm, footprint 180–206 × 101 cm,
grade −3 % to +30 %.

### As found (before re-aiming)

| View | Camera | Distance | Lens height | Aim | What it costs |
|---|---|---|---|---|---|
| Side (primary) | Femto RGB 1280×720 | ~2 m, behind-right | ~1.0–1.1 m | rear-right oblique, ~30–40° off the sagittal plane | trunk/shin/knee are projections of an oblique view; the console upright crosses the left third |
| Front | RealSense 455f RGB (SDK: + depth) | ~1.1–1.5 m from the deck front corner | ~1.2 m | yawed ~20–25°, level | near deck fills the lower frame, runner off-centre, console arm in shot; step width foreshortened by the yaw |

Both streams were healthy (`22 fps`, `error: null`) while framed like that — a green
state proves nothing about aim. Depth on the 455f at 640×480: **85.8 % valid,
fx 384.5 → 154 px/m at 2.5 m**, 22 fps alongside the Femto.

### Target

| View | Camera | Position | Aim | Metrics it gates |
|---|---|---|---|---|
| Side | Femto RGB + ToF (after moving the cable to a direct USB 3 port) | 2.4–2.8 m perpendicular to the belt centreline | lens 1.0–1.2 m, level 0° (5° max) | trunk, shin, knee, hip sagittal angles, cadence, oscillation, metric stride |
| Front | RealSense 455f via the SDK (`source2: {kind:"depth"}`) | 3.0 m from the deck front edge on the centreline, **or** 2.5–3.0 m front-quarter, ≤15° yaw | 2.0–2.2 m high, pitched 12–18° down | pelvic drop, knee valgus, step width, foot progression (metric), crossover |
| Rear | third camera (ordinary 720p) | 1.6–2.0 m behind, centred | lens 0.4–0.6 m, level to 5° up | heel strike, rearfoot eversion (proxy ±3–5°), lateral foot placement, crossover |

Rules that matter more than the numbers:

- **Perpendicular is literal.** A 10° yaw on the side view shortens the shank by
  cos 10° = 1.5 % but reads as several degrees on knee flexion; 30° is fatal.
- **Frame for 30 % grade.** The Alpine Runner lifts the deck 30 cm+ at the front and the
  runner's head rises ~0.5 m; a frame that just fits at 0 % crops the head at 20 %.
  Lock one grade for comparison sessions (1 % is the reference) and record it — the
  metric bands are level-running norms and the app has no grade compensation.
- **Do not use the deck as a level reference** for the depth pitch fit when the belt is
  inclined; fit the room floor outside the machine.
- **Clear the background.** A chair and a person at the desk sit inside the current front
  frame; MediaPipe will lock onto whoever fits best when the runner is small or absent.
