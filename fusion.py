"""Fused 3D skeleton and cross-view coordination metrics (P2 step 2).

Two calibrated views see the same runner at the same time.  Matching their 2D
landmarks and triangulating them gives one 3D skeleton in metres — that is what
turns "the pelvis and the thorax rotate" into a number, and what makes
arm-leg coupling and left/right asymmetry measurable at all.

What this module does NOT fix
-----------------------------
* **Time.** The cameras share no hardware trigger; frames are matched by the
  software clock, so assume up to ~1 frame of skew (~45 ms at 22 fps).  A phase
  lag is reported in milliseconds, so the skew is visible in the number — it is
  not corrected, and it is the floor on any coupling claim.
* **Gravity.** Pelvis/thorax rotation are measured about the vertical axis.
  The vertical comes from the calibration document's ``gravity`` vector (the
  depth floor fit at calibration time); with the default ``[0, 1, 0]`` the
  primary camera is ASSUMED level.  A camera pitched 5° biases every transverse
  angle by a few degrees, and nothing here detects that.
* **Soft tissue and landmark error.** Triangulated joints inherit MediaPipe's
  error; a residual gate drops the worst frames but cannot make a hip centre
  accurate.

Conventions
-----------
World frame: the primary camera's optical frame at calibration time (see
``calibration.py``), x right, y down, z forward.  ``down`` may be overridden by
the calibration's gravity vector; ``right`` and ``forward`` are rebuilt
orthogonal to it, so "transverse" always means "about true vertical".
"""
from __future__ import annotations

import math

import numpy as np

# The joints the fused metrics use.  MediaPipe Pose indices.
JOINTS = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}

MIN_VISIBILITY = 0.5        # both views must see the joint
MAX_RESIDUAL_PX = 12.0      # per-view reprojection gate for a fused joint
MIN_SERIES_S = 2.0          # two gait cycles before any coupling number

# Working volume in front of the primary camera, metres.  A *wrong* match (one
# landmark paired with the other view's other limb) still triangulates exactly,
# so the reprojection residual cannot catch it; a depth gate catches the gross
# cases and the rest is left to the operator.
MIN_DEPTH_M, MAX_DEPTH_M = 0.2, 6.0

# Running stride cycles are ~0.65-0.75 s, so a +-1 s search window is longer
# than the signal's period and the correlation has several equal peaks.  Cap the
# search at half a stride and flag a result that lands on the cap.
MAX_LAG_S = 0.4


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------
def basis_from_gravity(gravity=(0.0, 1.0, 0.0), forward=(0.0, 0.0, 1.0)):
    """Orthonormal (right, down, forward) with ``down`` = gravity.

    ``forward`` only fixes the rotation about the vertical: it is projected
    onto the horizontal plane and re-normalised, so a tilted calibration frame
    still yields a horizontal plane.
    """
    down = np.asarray(gravity, float).reshape(3)
    n = np.linalg.norm(down)
    if n < 1e-9:
        raise ValueError("gravity must be a non-zero vector")
    down = down / n

    f = np.asarray(forward, float).reshape(3)
    f = f - down * float(f @ down)
    if np.linalg.norm(f) < 1e-6:
        # forward is parallel to gravity: pick any orthogonal axis
        f = np.array([1.0, 0.0, 0.0])
        f = f - down * float(f @ down)
    f = f / np.linalg.norm(f)

    right = np.cross(down, f)
    right = right / np.linalg.norm(right)
    return right, down, f


def transverse_angle(left_xyz, right_xyz, basis) -> float:
    """Yaw (deg) of a left->right joint pair about the vertical.

    0° = the pair lies along the lateral axis (pelvis square to the camera);
    positive = the RIGHT joint is forward.  This is the standard convention for
    pelvic and thoracic rotation, and it is what lets the two be compared.
    """
    right, down, fwd = basis
    v = np.asarray(right_xyz, float) - np.asarray(left_xyz, float)
    return math.degrees(math.atan2(float(v @ fwd), float(v @ right)))


def sagittal_angle(proximal_xyz, distal_xyz, basis) -> float:
    """Swing (deg) of a proximal->distal segment in the sagittal plane.

    0° = the segment hangs straight down; positive = the distal joint is
    forward.  Used for arm and leg swing, so the two can be cross-correlated.
    """
    right, down, fwd = basis
    v = np.asarray(distal_xyz, float) - np.asarray(proximal_xyz, float)
    return math.degrees(math.atan2(float(v @ fwd), float(v @ down)))


# --------------------------------------------------------------------------
# triangulation of one frame
# --------------------------------------------------------------------------
def _pixels(lms, size):
    """Normalised landmarks -> pixel coordinates (N, 2)."""
    a = np.asarray(lms, float)
    return np.stack([a[:, 0] * size[0], a[:, 1] * size[1]], axis=1)


def fuse_frame(lms1, size1, P1, lms2, size2, P2,
               min_visibility: float = MIN_VISIBILITY,
               max_residual_px: float = MAX_RESIDUAL_PX) -> dict:
    """Triangulate the tracked joints seen by both views in one frame.

    Returns ``{"points": {name: (x, y, z)}, "residual_px": {name: r},
    "dropped": {name: reason}}``.  A joint is kept only when both views report
    it visible and the 3D point reprojects within ``max_residual_px`` in both —
    a bad match otherwise triangulates to a plausible-looking wrong place.
    """
    from calibration import reprojection_error  # local: keeps this module light

    a = np.asarray(lms1, float)
    b = np.asarray(lms2, float)
    if a.shape[0] < 33 or b.shape[0] < 33:
        raise ValueError("need MediaPipe pose landmarks (33 rows)")

    pa, pb = _pixels(a, size1), _pixels(b, size2)
    names = [n for n, i in JOINTS.items()
             if a[i, 3] >= min_visibility and b[i, 3] >= min_visibility]
    points: dict[str, tuple[float, float, float]] = {}
    residual: dict[str, float] = {}
    dropped: dict[str, str] = {
        n: "not visible in both views" for n, i in JOINTS.items()
        if a[i, 3] < min_visibility or b[i, 3] < min_visibility}
    if not names:
        return {"points": points, "residual_px": residual, "dropped": dropped}

    idx = np.array([JOINTS[n] for n in names])
    from calibration import triangulate
    X = triangulate(P1, P2, pa[idx], pb[idx])

    ra = reprojection_error(P1, X, pa[idx])
    rb = reprojection_error(P2, X, pb[idx])
    for k, name in enumerate(names):
        worst = max(float(ra[k]), float(rb[k]))
        if worst > max_residual_px or not np.all(np.isfinite(X[k])):
            dropped[name] = f"reprojection {worst:.1f} px > {max_residual_px:.0f}"
            continue
        if not (MIN_DEPTH_M <= X[k, 2] <= MAX_DEPTH_M):
            dropped[name] = (f"depth {X[k, 2]:.1f} m outside "
                             f"[{MIN_DEPTH_M}, {MAX_DEPTH_M}]")
            continue
        points[name] = (float(X[k, 0]), float(X[k, 1]), float(X[k, 2]))
        residual[name] = worst
    return {"points": points, "residual_px": residual, "dropped": dropped}


# --------------------------------------------------------------------------
# time series
# --------------------------------------------------------------------------
def phase_lag(sig_a, sig_b, fps: float, max_lag_s: float = MAX_LAG_S) -> dict | None:
    """Cross-correlation lag between two periodic series, at ``fps``.

    Positive ``lag_s`` means **b happens after a** (b lags a).  Returns None
    when either series is flat or too short to overlap meaningfully.
    """
    a = np.asarray(sig_a, float)
    b = np.asarray(sig_b, float)
    n = min(len(a), len(b))
    if n < 8 or fps <= 0:
        return None
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return None
    max_lag = max(1, int(round(max_lag_s * fps)))
    max_lag = min(max_lag, n - 8)
    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:n + lag]
        if len(x) < 8:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r) and r > best_r:
            best_lag, best_r = lag, r
    return {"lag_s": best_lag / fps, "lag_frames": best_lag,
            "r": best_r, "n": n, "fps": fps,
            "ambiguous": abs(best_lag) >= max_lag - 1,
            "max_lag_s": max_lag / fps}


def _series(fused_frames: list[dict], key: str, fps: float) -> np.ndarray:
    """One joint coordinate per fused frame, NaN where the joint was dropped."""
    out = []
    for f in fused_frames:
        p = f["points"].get(key)
        out.append(np.nan if p is None else float(p[1]))
    return np.asarray(out, float)


def coordination(fused_frames: list[dict], fps: float,
                 gravity=(0.0, 1.0, 0.0), max_lag_s: float = MAX_LAG_S) -> dict:
    """Cross-view coordination metrics from a fused-frame sequence.

    Reports pelvis-thorax transverse lag and contralateral arm-leg lag, each
    with its correlation coefficient and amplitude.  Anything that cannot be
    computed is listed in ``not_assessed`` with the reason — never as a zero.
    """
    basis = basis_from_gravity(gravity)
    out = {"frames": len(fused_frames), "fps": fps,
           "min_series_s": MIN_SERIES_S, "not_assessed": []}
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(fused_frames) < int(MIN_SERIES_S * fps):
        out["not_assessed"].append(
            f"series {len(fused_frames)/fps:.1f} s < {MIN_SERIES_S:.0f} s")
        return out

    def _angle(fn, p, a, b):
        # Each joint is appended independently: a frame can have the hips fused
        # and the shoulders dropped, and the two series must stay aligned.
        try:
            return fn(p[a], p[b], basis)
        except KeyError:
            return np.nan

    # ---- pelvis vs thorax (transverse rotation) ----
    pelvis, thorax = [], []
    for f in fused_frames:
        p = f["points"]
        pelvis.append(_angle(transverse_angle, p, "l_hip", "r_hip"))
        thorax.append(_angle(transverse_angle, p, "l_shoulder", "r_shoulder"))
    pelvis = np.asarray(pelvis, float)
    thorax = np.asarray(thorax, float)
    if np.isfinite(pelvis).sum() >= int(MIN_SERIES_S * fps * 0.5) and \
       np.isfinite(thorax).sum() >= int(MIN_SERIES_S * fps * 0.5):
        ok = np.isfinite(pelvis) & np.isfinite(thorax)
        lag = phase_lag(pelvis[ok], thorax[ok], fps, max_lag_s)
        if lag:
            out["pelvis_thorax"] = {
                **lag,
                "pelvis_deg": float(np.nanmax(pelvis) - np.nanmin(pelvis)),
                "thorax_deg": float(np.nanmax(thorax) - np.nanmin(thorax)),
                "coverage_pct": float(100.0 * ok.mean()),
                "note": "positive lag = thorax follows pelvis; software clock, "
                        "~1 frame skew not corrected",
            }
        else:
            out["not_assessed"].append("pelvis-thorax: flat or unusable signal")
    else:
        out["not_assessed"].append("pelvis-thorax: hips/shoulders not fused often enough")

    # ---- contralateral arm-leg swing ----
    for arm, leg, label in (("l_wrist", "r_ankle", "left_arm_right_leg"),
                            ("r_wrist", "l_ankle", "right_arm_left_leg")):
        arm_shoulder = "l_shoulder" if arm.startswith("l") else "r_shoulder"
        leg_hip = "l_hip" if leg.startswith("l") else "r_hip"
        arm_sig, leg_sig = [], []
        for f in fused_frames:
            p = f["points"]
            arm_sig.append(_angle(sagittal_angle, p, arm_shoulder, arm))
            leg_sig.append(_angle(sagittal_angle, p, leg_hip, leg))
        arm_sig = np.asarray(arm_sig, float)
        leg_sig = np.asarray(leg_sig, float)
        ok = np.isfinite(arm_sig) & np.isfinite(leg_sig)
        if ok.sum() < int(MIN_SERIES_S * fps * 0.5):
            out["not_assessed"].append(f"{label}: not fused often enough")
            continue
        # leg first, arm second: a positive lag then reads "the arm lags the
        # contralateral leg", which is the clinically meaningful direction.
        lag = phase_lag(leg_sig[ok], arm_sig[ok], fps, max_lag_s)
        if not lag:
            out["not_assessed"].append(f"{label}: flat or unusable signal")
            continue
        out.setdefault("arm_leg", {})[label] = {
            **lag,
            "note": "positive lag = arm lags the contralateral leg",
            "arm_amp_deg": float(np.nanmax(arm_sig) - np.nanmin(arm_sig)),
            "leg_amp_deg": float(np.nanmax(leg_sig) - np.nanmin(leg_sig)),
            "coverage_pct": float(100.0 * ok.mean()),
        }
    return out
