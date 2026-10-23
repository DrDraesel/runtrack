"""Metric-scale foot metrics from an aligned RealSense depth frame.

The depth stream is sampled at foot landmarks so lateral distance and foot
progression are expressed in centimetres/degrees instead of frame fractions.
That is the whole point of using depth: an image-only metric cannot say how
wide a step is, only how wide it looks at an unknown distance.

Honest limits - state these whenever the numbers are quoted:
  * the D455F has roughly 3-4 cm RMS depth noise per frame at 2.5-3 m, which
    is the same order as the step width itself.  Every reported figure is a
    median over many frames; a single frame is never a measurement.
  * step width here is a contact-to-contact proxy: contacts come from a
    stance heuristic on ankle height, not from force plates.
  * foot progression is a magnitude.  Its sign depends on which side of the
    body the camera sits on, so the sign is not reported.
  * the camera must be square to the deck for "forward" to mean the line of
    progression.  A yawed camera biases every angle by that yaw.

Coordinates: camera frame, metres.  X is image-right (lateral), Y is
image-down (vertical), Z is distance from the camera (forward).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

VIS_MIN = 0.5


def point_at(lms, idx, depth_at, intr, vis_min=VIS_MIN, r=4):
    """Back-project one landmark to (X, Y, Z) metres, or None.

    ``depth_at(nx, ny, r)`` returns metres at normalised image coords.
    ``intr`` carries fx, fy, cx, cy in pixels plus width/height.
    """
    if lms is None or depth_at is None or not intr:
        return None
    try:
        if float(lms[idx, 3]) < vis_min:
            return None
        nx, ny = float(lms[idx, 0]), float(lms[idx, 1])
    except (TypeError, ValueError, IndexError):
        return None
    z = depth_at(nx, ny, r)
    if z is None:
        return None
    try:
        z = float(z)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z) or z <= 0.05 or z > 12.0:
        return None
    try:
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
        w, h = float(intr["width"]), float(intr["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if fx <= 0 or fy <= 0:
        return None
    return ((nx * w - cx) / fx * z, (ny * h - cy) / fy * z, z)


def foot_progression_deg(heel, toe):
    """|toe-out| angle of the heel->toe axis in the floor plane, degrees.

    0 deg = foot points straight along the camera's forward axis (assumed
    parallel to the belt).  Magnitude only - see the module docstring.
    """
    if heel is None or toe is None:
        return None
    dx = float(toe[0] - heel[0])
    dz = float(toe[2] - heel[2])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return None
    return abs(math.degrees(math.atan2(dx, dz)))


class DepthGait:
    """Accumulates metric foot metrics across a session.

    ``update`` is called once per processed frame with that view's landmarks;
    it returns the per-frame values (JSON-safe floats) to store in the row,
    and keeps the history needed for the medians in ``summary``.
    """

    SIDES = ("left", "right")
    WINDOW = 600            # ~27 s at 22 fps, enough for the stance percentiles
    STANCE_HI = 60.0        # ankle Y >= this percentile of the window = stance
    STANCE_LO = 40.0        # ankle Y <  this percentile = swing (hysteresis)

    def __init__(self):
        self.frames = 0
        self.frames_with_depth = 0
        self.fpa = {s: [] for s in self.SIDES}          # degrees, stance frames
        self.sep_cm = []                                 # |x_l - x_r| per frame
        self.steps_cm = []                               # contact-to-contact
        self.dist = []                                   # metres at the feet
        self._ys = {s: deque(maxlen=self.WINDOW) for s in self.SIDES}
        self._stance = {s: False for s in self.SIDES}
        self._last_x = {s: None for s in self.SIDES}     # last contact lateral
        self._warned_no_intr = False

    # ---------------------------------------------------------------- update
    def update(self, lms, depth_at, intr, t=None) -> dict:
        """Compute this frame's metric foot values; returns row fields."""
        self.frames += 1
        out: dict = {}
        if lms is None or depth_at is None or not intr:
            return out
        pts = {}
        for side, ankle, heel, toe in (("left", L_ANKLE, L_HEEL, L_TOE),
                                       ("right", R_ANKLE, R_HEEL, R_TOE)):
            a = point_at(lms, ankle, depth_at, intr)
            h = point_at(lms, heel, depth_at, intr)
            f = point_at(lms, toe, depth_at, intr)
            if a is None:
                continue
            pts[side] = (a, h, f)
            out[f"ankle_x_{side}_cm"] = round(a[0] * 100.0, 1)
            out[f"ankle_y_{side}_cm"] = round(a[1] * 100.0, 1)
            fp = foot_progression_deg(h, f)
            if fp is not None:
                out[f"fpa_{side}_deg"] = round(fp, 1)
            self._ys[side].append(a[1])
        if not pts:
            return out
        self.frames_with_depth += 1
        self.dist.append(float(np.median([p[0][2] for p in pts.values()])))

        if "left" in pts and "right" in pts:
            self.sep_cm.append(abs(pts["left"][0][0] - pts["right"][0][0]) * 100.0)

        # stance heuristic -> contacts -> step width
        for side in self.SIDES:
            if side not in pts:
                continue
            ys = self._ys[side]
            if len(ys) < 30:
                continue
            y = pts[side][0][1]
            hi = float(np.percentile(np.asarray(ys), self.STANCE_HI))
            lo = float(np.percentile(np.asarray(ys), self.STANCE_LO))
            stance = self._stance[side]
            if not stance and y >= hi:
                stance = True
                self._contact(side, pts[side][0][0])
            elif stance and y < lo:
                stance = False
            self._stance[side] = stance
            fp = out.get(f"fpa_{side}_deg")
            if stance and fp is not None:
                self.fpa[side].append(float(fp))
        return out

    def _contact(self, side, x_m):
        """Record a contact; a contact opposite the previous one gives a step."""
        other = "right" if side == "left" else "left"
        prev_other = self._last_x[other]
        if prev_other is not None:
            self.steps_cm.append(abs(x_m - prev_other) * 100.0)
        self._last_x[side] = x_m

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        """JSON-safe medians with the sample counts they rest on."""
        def med(vals):
            return round(float(np.median(vals)), 1) if vals else None

        fpa = {s: med(self.fpa[s]) for s in self.SIDES}
        asym = None
        if fpa["left"] is not None and fpa["right"] is not None:
            asym = round(abs(fpa["left"] - fpa["right"]), 1)
        out = {
            "frames": self.frames,
            "frames_with_depth": self.frames_with_depth,
            "step_width_cm": med(self.steps_cm),
            "step_width_n": len(self.steps_cm),
            "fpa_left_deg": fpa["left"],
            "fpa_right_deg": fpa["right"],
            "fpa_n_left": len(self.fpa["left"]),
            "fpa_n_right": len(self.fpa["right"]),
            "fpa_asymmetry_deg": asym,
            "lateral_separation_cm": med(self.sep_cm),
            "distance_m": med(self.dist),
            "depth_note": ("medians over stance frames; per-frame depth RMS is "
                           "3-4 cm at 2.5-3 m, so no single frame is a "
                           "measurement, and step width is a contact-to-contact "
                           "proxy, not a force-plate value"),
        }
        return out
