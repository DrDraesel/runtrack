# Cycling / triathlon / TT position analysis

RunTrack measures a rider's position from ordinary video and reports joint angles against
common **video bike-fit** target bands. It is a coaching and research aid — not a medical
device, not a diagnosis, and not a substitute for an in-person fitter or clinician.

## What it measures (and what it does not)

Side camera (sagittal view):

- knee angle at the **bottom of the pedal stroke (BDC)** — the classic saddle-height reference
  (included angle; 140–150° road, 140–145° TT/tri, i.e. 30–40° / 35–40° knee flexion)
- knee angle at the **top of the stroke (TDC)** — how closed the knee gets
- **hip angle at TDC** — available hip range in an aero position (road 70–90°, TT/tri 45–75°)
- **torso angle vs horizontal** (road 35–45°, TT 10–25°, tri 15–30°)
- **ankle angle**, **elbow angle** (road 150–170°, TT/tri 85–105°), shoulder angle
- **cadence** from the knee-angle oscillation frequency (FFT)
- per-side values, revolutions counted, and a BDC/TDC keyframe gallery with an HD re-detect pass

Rear/front camera (frontal plane):

- hip vertical/horizontal travel (rocking), pelvic obliquity
- knee lateral travel and knee-travel angle vs vertical
- shoulder travel and ankle swivel

Front camera with segmentation enabled:

- projected frontal-area silhouette share — only a **relative comparator** on the same camera
  setup; it is not CdA and not a wind-tunnel result.

Not measurable from one side camera: saddle setback / knee-over-pedal-spindle (no
bottom-bracket reference), true 3-D joint angles, hip rocking (needs the rear view), and any
diagnosis or injury prediction.

## Camera setup

| View | Position | Why |
|---|---|---|
| Side (primary) | perpendicular to the bike, 2.5–4 m away, lens at hip height (~0.9–1.1 m) | knee/hip/torso angles; keep the **whole rider and both wheels** in frame |
| Rear (second) | 2–3 m behind, hip height | hip rocking, knee tracking, pelvic obliquity |
| Front (third, optional) | 3–4 m ahead, hip height, slight offset | frontal area with segmentation |

- 60 fps if the camera supports it; 30 fps is usable but BDC/TDC keyframes are coarser.
- Lock exposure/shutter if possible; avoid zoom changes between positions.
- Turbo trainer / stationary setup: the bike must not translate. A moving road shot is only a
  demo — the camera must be fixed relative to the bike for fit work.
- Use the **same camera distance and zoom** for before/after comparisons.

## Capture protocol (about 8–10 minutes)

1. Warm up 10–15 minutes; settle into the position being tested.
2. Ride at a steady self-selected cadence (85–95 rpm is a useful reference) for 60–90 s.
3. Record one **baseline** clip per position (road hoods / aerobars).
4. Change **one thing only**, **10 mm** at a time (saddle height, saddle fore/aft, pad stack,
   pad reach; record crank length). Re-film 60 s at the same cadence. 5 mm is below the
   measurement resolution — keep it for fine-tuning after a 10 mm effect is confirmed.
5. Mark keyframes / save both sessions; compare the angle table and the report's AI fit review.
6. Repeat until the knee-at-BDC, hip-at-TDC and torso/elbow values are inside the chosen band
   **and the rider reports comfort and power**. Numbers do not overrule the rider.

## Targets shown by the app

| Metric | Road | TT | Triathlon | Evidence |
|---|---|---|---|---|
| Knee at BDC (included°) | 140–150 | 140–147 | 140–147 | B — dynamic 2-D method validated; ~2° bias vs 3-D |
| Knee at TDC (included°) | ≥68 (floor) | ≥68 (floor) | ≥68 (floor) | C — floor only; the old 68–95 upper bound was implausible |
| Hip closed at TDC (°) | 55–70 | 40–60 | 40–60 | C/D — video-fit / TT convention (old road 70–90 passed a hip that never closes) |
| Torso vs horizontal (°) | 35–50 | 10–25 | 15–30 | C — convention; elite vs general differs |
| Ankle (°) | descriptive | descriptive | descriptive | C — markerless foot tracking is the weakest landmark; not a pass/fail |
| Elbow (°) | 150–170 | 85–105 | 85–105 | C/D — hoods vs aerobars; TT band is convention, not validated |
| Cadence (rpm) | 70–100 | 70–100 | 70–100 | C — self-selected reference, not a norm |

These are **fit guidance bands**, not clinical norms or population percentiles. The evidence
grades, source list and disagreements (static vs dynamic knee, hip definitions, torso ranges)
are in **[research/CYCLING-TT-FIT-EVIDENCE.md](../research/CYCLING-TT-FIT-EVIDENCE.md)** — read
it before changing a band. A value outside a band is a prompt to test a change, not a
diagnosis.

**Measurement resolution:** 2-D markerless knee angles carry a 3–9° error and a ~2° bias vs
3-D; 5 mm of saddle height is only ≈1.2° at BDC. Use **10 mm steps** (5 mm only as a fine-tune
after a 10 mm effect is confirmed) and repeat the baseline. A difference below ~3° (knee/hip)
or ~5° (torso) is not detectable from this camera. Never compare absolute degrees with a 3-D
fit or a different camera setup; same-setup comparisons only.

## Reading the live overlay

- Tracking lines: hip→knee→ankle (cyan), shoulder→hip→knee (white), torso (magenta), arm
  (orange), dashed plumb and horizontal references.
- Angle labels turn **green** inside the selected bike type's band and **amber** outside.
  The knee and hip bands are **BDC/TDC reference values**, not targets for every crank
  position: an instantaneous mid-stroke knee can legitimately show amber while the BDC/TDC
  values in the panel are in range. Use the panel/report for the fit decision.
- The live panel is a rolling 5 s estimate; the saved report uses per-revolution BDC/TDC
  keyframes. Use the report for decisions, the live panel for feedback while adjusting.

## AI fit review

The report's **Run local AI form review** button sends only derived numbers (no video) to the
configured local model or OpenAI-compatible endpoint. The prompt is grounded in the same target
bands and asks for prioritized 10 mm changes, one aero comment, and one data-limits caveat.
It never diagnoses; the deterministic fit suggestions in the report are the primary output.

## Limitations

- 2-D pose estimation from one camera: 3–9° typical error and ~2° bias vs 3-D; more in
  out-of-plane motion. Do not compare absolute degrees across different camera setups.
- Aero: silhouette area explained only ~54 % of drag-area variance in field data, and smaller
  area did not always mean less drag. The watt figure is a Cd-assumption scenario, not a
  measured saving; never claim CdA, watts saved or predicted speed.
- Bike fit is not a medical device; it does not prevent or treat injury. Refer pain with
  swelling/locking/giving way or neurological symptoms to a clinician.
- Knee/hip angles depend on landmark accuracy; clothing, saddle/bar occlusion and motion blur
  reduce it.
- Aero frontal area is a silhouette proportion, not CdA; use it only for same-setup comparisons.
- TT/tri positions trade comfort and power for aerodynamics; a more aggressive position is not
  automatically better.
- No saddle setback/KOPS, no 3-D kinematics, no injury diagnosis, no medical clearance.

## Tomorrow's live acceptance test

1. Side camera: full body, 3 m, hip height; 60 fps if available. Rear camera at 2.5 m.
2. Open the app → **Cycling** → **TT** (or Triathlon) → open camera → patient → start.
3. Ride 60 s at ~90 rpm in the aerobars. Confirm: pose tracked, overlay lines/angles visible,
   live panel shows BDC/TDC/hip/torso/elbow/cadence with target colors.
4. Stop; open the report. Confirm: bike type TT, angle table, per-side values, BDC/TDC keyframes,
   rear-view table (if rear camera), TT notes, AI fit review button.
5. Change saddle height 5 mm; repeat; confirm the report shows the new value and the AI review
   references the change.

