"""RunTrack rig check: is the camera placed so the metrics can be measured?

Pure functions over one MediaPipe landmark array of shape ``(33, 4)``
(x, y, z, visibility; x and y normalised 0..1 with y growing downward).
No camera, no I/O, no module state: every result is a function of the arguments.

What a pass means
-----------------
``ok=True`` from :func:`check_view` means the *preconditions* are met: the
runner is fully in frame at a usable size, the camera is not pitched, and the
view plane matches the plane being analysed.  It says nothing about the
validity of any joint angle -- an angle still depends on landmark accuracy,
clothing, occlusion and motion blur.  These are operator placement aids, not
clinical measurements.

Units and conventions
---------------------
* ``w``/``h`` are the frame size in pixels that the normalised landmarks refer
  to (normalisation is resolution independent, so any matching size works).
* x spans the frame width and y the frame height, so pixel distances are
  ``dx * w`` and ``dy * h``; nothing here computes an angle, and every ratio
  that matters (nose offset / shoulder width, bbox / frame) is taken within a
  single axis.
* ``pitch_deg`` is the measured camera pitch from the depth floor-plane fit,
  positive = aimed down.  ``None`` means not measured, which is not the same
  as zero: an unmeasured pitch is never warned about.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Landmark groups.  Feet are split from legs because the foot metrics (strike
# angle, toe-off) fail independently of the knee/hip metrics.
BODY_GROUPS: dict[str, tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "torso": (11, 12, 23, 24),
    "arms": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
    "legs": (25, 26, 27, 28),
    "feet": (29, 30, 31, 32),
}

_GROUP_NAMES = ("head", "torso", "arms", "legs", "feet")
_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12

# ---------------------------------------------------------------- thresholds
# Cited against the app where the number already exists, so the placement check
# cannot drift away from the report that consumes the same footage.
PITCH_MAX_SAGITTAL_DEG = 5.0   # sagittal angles are measured against image vertical
PITCH_MAX_FRONTAL_DEG = 20.0   # frontal metrics live in the image plane
LOWER_BODY_MIN = 0.70          # gait events need the legs through most of a cycle
FEET_MIN = 0.50                # foot strike / toe-off need the foot landmarks
HEIGHT_FRAC_MIN = 0.55         # below this, landmark noise dominates joint angles
EDGE_MARGIN = 0.02             # 2% of the frame dimension counts as clipped
PX_PER_M_MIN = 60.0            # 60 px/m turns 1 cm into 0.6 px: too coarse for angles
PLANE_FRONTAL_MAX = 0.25       # mirrors metrics.camera_orientation
PLANE_SAGITTAL_MIN = 0.60      # mirrors metrics.camera_orientation
PLANE_MIN_SAMPLES = 5          # a median over fewer frames is not a median
SHOULDER_MIN_NORM = 0.02       # guard against dividing by a collapsed shoulder line


# ------------------------------------------------------------------- helpers
def _finite(value: Any) -> float | None:
    """float(value) if it is a finite number, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute-style object."""
    if obj is None:
        return default
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError, AttributeError):
        return getattr(obj, key, default)


def _as_landmarks(lms: Any) -> np.ndarray | None:
    """(N, >=4) float array, or None when the input cannot be used."""
    if lms is None:
        return None
    try:
        arr = np.asarray(lms, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 4:
        return None
    return arr


def _visible_mask(arr: np.ndarray, vis_min: float) -> np.ndarray:
    """Landmarks with visibility >= vis_min and finite x/y."""
    vis = arr[:, 3]
    return (np.isfinite(vis) & (vis >= vis_min)
            & np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]))


def _num(value: Any) -> str:
    """Short human number: 34, 34.5, 69.9 (never 34.0)."""
    v = _finite(value)
    if v is None:
        return "?"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def _num2(value: Any) -> str:
    """Like :func:`_num` but keeps a second decimal (metres, pixels)."""
    v = _finite(value)
    if v is None:
        return "?"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _pct(frac: float) -> str:
    """A 0..1 fraction as a percentage string, one decimal at most."""
    return _num(max(0.0, float(frac)) * 100.0)


def _pct_below(frac: float, limit: float | None = None) -> str:
    """Percentage for a 'below threshold' message, floored at one decimal.

    Flooring keeps the printed number on the same side of the threshold as the
    test that produced the warning (0.6996 prints 69.9, not a contradictory 70).
    The value is snapped to 1e-6 first so float32 landmarks do not turn a true
    0.40 into 39.9, and ``limit`` (in percent) caps the display just under the
    threshold for the same reason.
    """
    tenths = math.floor(round(max(0.0, float(frac)), 6) * 1000.0) / 10.0
    if limit is not None and tenths >= limit:
        tenths = max(0.0, limit - 0.1)
    return _num(tenths)


def _int_below(value: float, limit: float | None = None) -> str:
    """Integer for an 'under threshold' message, floored for the same reason."""
    whole = math.floor(round(max(0.0, float(value)), 6))
    if limit is not None and whole >= limit:
        whole = max(0, int(limit) - 1)
    return str(whole)


# ------------------------------------------------------------------ coverage
def body_coverage(lms: Any, vis_min: float = 0.5) -> dict:
    """Share of each body group that is visible, 0..1.

    A landmark counts when its visibility reaches ``vis_min`` and its x/y are
    finite.  ``lower`` averages legs+feet (gait events need both); ``overall``
    averages the five groups so every region carries the same weight.
    ``None`` or an unusable array yields all zeros, never an exception.
    """
    frac = {name: 0.0 for name in _GROUP_NAMES}
    arr = _as_landmarks(lms)
    if arr is not None:
        vis_min_v = _finite(vis_min)
        mask = _visible_mask(arr, 0.5 if vis_min_v is None else vis_min_v)
        n_rows = arr.shape[0]
        for name, idx in BODY_GROUPS.items():
            present = [i for i in idx if i < n_rows]
            if present:
                frac[name] = float(np.count_nonzero(mask[present])) / len(present)
    out: dict[str, float] = {name: round(frac[name], 4) for name in _GROUP_NAMES}
    out["lower"] = round((frac["legs"] + frac["feet"]) / 2.0, 4)
    out["overall"] = round(sum(frac.values()) / len(_GROUP_NAMES), 4)
    return out


# ------------------------------------------------------------------ framing
def framing(lms: Any, w: float, h: float, vis_min: float = 0.5) -> dict:
    """Pixel bounding box of the visible landmarks, plus edge margins.

    ``margins`` gives each side's gap in fractions of its own frame dimension
    (negative = the landmark is outside the frame); ``touches_edge`` is True
    when any margin is under 2%.  Every pixel value is None when nothing is
    visible or the frame size is unusable.
    """
    empty = {"top_px": None, "bottom_px": None, "left_px": None, "right_px": None,
             "height_frac": None, "width_frac": None, "touches_edge": False,
             "body_h_px": None, "margins": None}
    wf, hf = _finite(w), _finite(h)
    if wf is None or hf is None or wf <= 0 or hf <= 0:
        return empty
    arr = _as_landmarks(lms)
    if arr is None:
        return empty
    vis_min_v = _finite(vis_min)
    mask = _visible_mask(arr, 0.5 if vis_min_v is None else vis_min_v)
    if not mask.any():
        return empty

    left, right = float(arr[mask, 0].min()), float(arr[mask, 0].max())
    top, bottom = float(arr[mask, 1].min()), float(arr[mask, 1].max())
    left_px, right_px = left * wf, right * wf
    top_px, bottom_px = top * hf, bottom * hf
    body_h = bottom_px - top_px
    margins = {"left": left_px / wf, "right": (wf - right_px) / wf,
               "top": top_px / hf, "bottom": (hf - bottom_px) / hf}
    return {
        "top_px": top_px,
        "bottom_px": bottom_px,
        "left_px": left_px,
        "right_px": right_px,
        "height_frac": body_h / hf,
        "width_frac": (right_px - left_px) / wf,
        "touches_edge": any(m < EDGE_MARGIN for m in margins.values()),
        "body_h_px": body_h,
        "margins": margins,
    }


# --------------------------------------------------------------- view plane
def _as_frames(lms: Any) -> np.ndarray | None:
    """Normalise input to (N, >=13, >=4): one frame or a stack of frames."""
    if lms is None:
        return None
    try:
        arr = np.asarray(lms, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 2:
        if arr.shape[0] < _L_SHOULDER + 2 or arr.shape[1] < 4:
            return None
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[1] >= _L_SHOULDER + 2 and arr.shape[2] >= 4:
        return arr
    return None


def _nose_offset_ratio(frame: np.ndarray, vis_min: float) -> float | None:
    """|nose_x - shoulder_mid_x| / shoulder_width for one frame."""
    mask = _visible_mask(frame, vis_min)
    if not (mask[_NOSE] and mask[_L_SHOULDER] and mask[_R_SHOULDER]):
        return None
    nose_x = float(frame[_NOSE, 0])
    lx, rx = float(frame[_L_SHOULDER, 0]), float(frame[_R_SHOULDER, 0])
    mid, width = (lx + rx) / 2.0, abs(lx - rx)
    if width < SHOULDER_MIN_NORM:
        # A collapsed shoulder line makes the ratio explode; a true profile can
        # legitimately have the shoulders almost overlapping, hence the small
        # floor rather than the report's 0.05 (which is tuned over many frames).
        return None
    return abs(nose_x - mid) / width


def plane_guess(lms: Any, vis_min: float = 0.5) -> str:
    """Guess the camera plane from the nose offset: frontal, sagittal or unknown.

    Signal (same as the report's): median |nose_x - shoulder_mid_x| / shoulder
    width.  A face-on runner sits near 0; a profile is typically above 1 and
    crosses the 0.60 threshold.  Uses the report's thresholds (0.25 / 0.60);
    the in-between band is reported as ``unknown`` rather than forced into a
    plane.

    ``lms`` may be one (33,4) frame -- judged on its own -- or an (N,33,4)
    stack, which needs at least 5 usable frames before a median is trusted.
    """
    frames = _as_frames(lms)
    if frames is None:
        return "unknown"
    vis_min_v = _finite(vis_min)
    vis = 0.5 if vis_min_v is None else vis_min_v
    ratios = [r for r in (_nose_offset_ratio(fr, vis) for fr in frames) if r is not None]
    if not ratios:
        return "unknown"
    if frames.shape[0] > 1 and len(ratios) < PLANE_MIN_SAMPLES:
        return "unknown"
    med = float(np.median(ratios))
    if med < PLANE_FRONTAL_MAX:
        return "frontal"
    if med > PLANE_SAGITTAL_MIN:
        return "sagittal"
    return "unknown"


# --------------------------------------------------------------- resolution
def px_per_m_from_body(lms: Any, w: float, h: float,
                       height_m: float = 1.75) -> float | None:
    """Fallback pixels-per-metre from the visible body height.

    Only valid when the whole body is in frame (head and feet coverage >= 0.5),
    otherwise the bbox height is a fragment of a person and the estimate would
    be silently wrong.  A raised arm above the head inflates the bbox, so this
    is a fallback, not a calibration.
    """
    height = _finite(height_m)
    if height is None or height <= 0:
        return None
    cov = body_coverage(lms)
    if cov["head"] < 0.5 or cov["feet"] < 0.5:
        return None
    box = framing(lms, w, h)
    body_h = _finite(box["body_h_px"])
    if body_h is None or body_h <= 0:
        return None
    return body_h / height


def _effective_fx(intrinsics: Any, w: float) -> tuple[float | None, str | None]:
    """fx in pixels of the current frame, rescaling if the calibration differs."""
    fx = _finite(_field(intrinsics, "fx"))
    if fx is None or fx <= 0:
        return None, None
    cal_w = _finite(_field(intrinsics, "width"))
    if cal_w is not None and cal_w > 0 and abs(cal_w - w) > 0.01 * cal_w:
        return fx * (w / cal_w), (f"fx rescaled from a {_num2(cal_w)} px wide "
                                  f"calibration to the {_num2(w)} px frame")
    return fx, None


def _subject_distance(depth: Any, box: dict) -> tuple[float | None, str | None]:
    """Take the scene median depth as subject distance only when that is credible.

    ``median_m`` is a scene median, not a tracked runner distance.  It is used
    only when most depth pixels are valid, the value is in a plausible capture
    range, and the runner dominates the frame; otherwise the caller gets None
    plus a note saying why, so scene depth is never presented as the distance
    to the runner.
    """
    if depth is None:
        return None, None
    med = _finite(_field(depth, "median_m"))
    valid = _finite(_field(depth, "valid_pct"))
    if med is None or med <= 0:
        return None, "no usable scene depth - distance unknown"
    reasons: list[str] = []
    if valid is None:
        reasons.append("valid-pixel share unknown")
    elif valid < 50.0:
        reasons.append(f"only {_num(valid)}% of depth pixels valid")
    if not 1.0 <= med <= 15.0:
        reasons.append(f"median {_num2(med)} m is outside a plausible capture range")
    height_frac = _finite(box.get("height_frac"))
    if height_frac is None or height_frac < 0.5:
        reasons.append(f"runner fills {_pct(height_frac or 0.0)}% of frame height, "
                       "so the median is mostly background")
    if reasons:
        return None, (f"scene depth {_num2(med)} m is not subject distance "
                      f"({'; '.join(reasons)}) - distance left unknown")
    return med, (f"distance from the scene median depth {_num2(med)} m "
                 f"({_num(valid)}% valid pixels) - an approximation, not a tracked runner")


# ---------------------------------------------------------------- warnings
def _clipped_edges(box: dict) -> list[tuple[str, float]]:
    """Edges under the margin, worst (smallest margin) first."""
    margins = box.get("margins")
    if not isinstance(margins, dict):
        return []
    found = [(name, _finite(margins.get(name))) for name in ("left", "right", "top", "bottom")]
    found = [(name, m) for name, m in found if m is not None and m < EDGE_MARGIN]
    found.sort(key=lambda item: item[1])
    return found


def warnings_for(plane: str, coverage: dict, framing: dict, px_per_m: float | None,
                 pitch_deg: float | None, plane_guess: str) -> list[str]:
    """Operator-facing warnings, shortest useful form, each with its number.

    Rules are plane-gated: a sagittal view tolerates almost no pitch because its
    angles are measured against image vertical, while a frontal view is mostly
    in the image plane and tolerates a downward tilt.  Nothing here validates a
    joint angle; every message is about placement.
    """
    plane_s = str(plane or "").lower().strip()
    out: list[str] = []

    # -- camera pitch ------------------------------------------------------
    pitch = _finite(pitch_deg)
    if pitch is not None:
        if plane_s == "sagittal" and abs(pitch) > PITCH_MAX_SAGITTAL_DEG:
            out.append(
                f"camera pitched {_num(abs(pitch))} deg "
                f"(max {_num(PITCH_MAX_SAGITTAL_DEG)} for a side view: "
                "sagittal angles are measured against image vertical)")
        elif plane_s == "frontal" and pitch > PITCH_MAX_FRONTAL_DEG:
            out.append(f"camera pitched {_num(pitch)} deg down "
                       f"(max {_num(PITCH_MAX_FRONTAL_DEG)} for a front view)")

    # -- what is visible ---------------------------------------------------
    coverage = coverage or {}
    lower = _finite(_field(coverage, "lower"))
    if lower is not None and lower < LOWER_BODY_MIN:
        shown = _pct_below(lower, LOWER_BODY_MIN * 100)
        out.append(f"lower body {shown}% visible - "
                   f"gait events cannot latch below {_num(LOWER_BODY_MIN * 100)}%")
    feet = _finite(_field(coverage, "feet"))
    if feet is not None and feet < FEET_MIN:
        shown = _pct_below(feet, FEET_MIN * 100)
        out.append(f"feet {shown}% visible - foot metrics unavailable")

    # -- how big the runner is in frame ------------------------------------
    box = framing or {}
    height_frac = _finite(_field(box, "height_frac"))
    if height_frac is not None and height_frac < HEIGHT_FRAC_MIN:
        shown = _pct_below(height_frac, HEIGHT_FRAC_MIN * 100)
        out.append(f"runner fills only {shown}% of frame height - "
                   "move camera closer or zoom")

    # -- clipping ----------------------------------------------------------
    if _field(box, "touches_edge"):
        clipped = _clipped_edges(box)
        if clipped:
            parts = [f"{name} edge ({_pct(max(0.0, margin))}% margin)"
                     for name, margin in clipped]
            where = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
            out.append(f"body clipped at the {where} - widen the shot to fit the runner")
        else:
            out.append("body clipped at a frame edge - widen the shot to fit the runner")

    # -- resolution --------------------------------------------------------
    ppm = _finite(px_per_m)
    if ppm is not None and ppm < PX_PER_M_MIN:
        out.append(f"resolution {_int_below(ppm, PX_PER_M_MIN)} px/m - too coarse, aim for 100+")

    # -- plane mismatch ----------------------------------------------------
    guess = str(plane_guess or "unknown").lower().strip()
    if plane_s in ("sagittal", "frontal") and guess not in ("unknown", plane_s):
        out.append(f"view looks {guess} but is being analysed as {plane_s} - "
                   "check camera position")
    return out


# ----------------------------------------------------------------- the check
def check_view(plane: str, lms: Any, w: float, h: float, depth: Any = None,
               intrinsics: Any = None, pitch_deg: float | None = None,
               body_height_m: float = 1.75) -> dict:
    """Score one camera view for placement and return the warnings.

    ``plane`` is the plane being analysed ("sagittal" or "frontal"); it is the
    operator's choice, not re-derived from the landmarks.  ``depth`` is an
    optional ``{"valid_pct", "median_m"}`` scene-depth summary, ``intrinsics``
    an optional ``{fx, fy, cx, cy, width, height}``, and ``pitch_deg`` the
    measured pitch from the depth floor-plane fit (positive = aimed down).

    ``px_per_m`` comes from ``fx / distance`` when a credible distance exists,
    otherwise from the visible body height.  ``distance_m`` is None whenever
    the scene median cannot stand in for the subject distance, with a note
    saying so.  ``ok`` is True only when ``warnings`` is empty -- and only
    means the framing/pitch preconditions are met, never that a joint angle is
    trustworthy.
    """
    plane_s = str(plane).lower().strip() if plane is not None else ""
    if plane_s not in ("sagittal", "frontal"):
        raise ValueError(f"plane must be 'sagittal' or 'frontal', got {plane!r}")

    coverage = body_coverage(lms)
    box = framing(lms, w, h)
    guess = plane_guess(lms)
    pitch = _finite(pitch_deg)
    distance_m, distance_note = _subject_distance(depth, box)

    px_note: str | None = None
    fx, fx_note = _effective_fx(intrinsics, float(_finite(w) or 0.0))
    if fx is not None and distance_m is not None and distance_m > 0:
        px_per_m: float | None = fx / distance_m
        px_note = (f"resolution from camera intrinsics: fx {_num2(fx)} px "
                   f"at {_num2(distance_m)} m")
        if fx_note:
            px_note = f"{px_note} ({fx_note})"
    else:
        px_per_m = px_per_m_from_body(lms, w, h, body_height_m)
        if px_per_m is not None:
            px_note = (f"resolution from body height: {_num2(px_per_m * float(body_height_m))} px "
                       f"for an assumed {_num2(body_height_m)} m runner")
        else:
            px_note = ("resolution unknown: head and feet both need to be visible "
                       "to estimate it from body height")

    warnings = warnings_for(plane_s, coverage, box, px_per_m, pitch, guess)
    notes = ["placement check only: framing, pitch and resolution preconditions, "
             "not joint angles"]
    if distance_note:
        notes.append(distance_note)
    if px_note:
        notes.append(px_note)

    return {
        "ok": not warnings,
        "plane": plane_s,
        "plane_guess": guess,
        "coverage": coverage,
        "framing": box,
        "px_per_m": px_per_m,
        "pitch_deg": pitch,
        "distance_m": distance_m,
        "warnings": warnings,
        "notes": notes,
    }
