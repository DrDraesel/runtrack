"""RunTrack metrics: per-frame joint angles/positions and session analysis.

All angles in degrees. Positions normalized (0-1) image coordinates, y down.
All session-level estimates are clearly-labelled heuristics / research
estimates, not clinical measurements.
"""
from __future__ import annotations

import math
import numpy as np

import config
import gait_events as GA
from pose_engine import (
    NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)

G = 9.81
VIS = config.VIS_MIN

# Joint angle specs: name -> (a, b, c) with b the vertex landmark
JOINT_SPECS = {
    "left_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "left_shoulder": (L_ELBOW, L_SHOULDER, L_HIP),
    "right_shoulder": (R_ELBOW, R_SHOULDER, R_HIP),
    "left_hip": (L_SHOULDER, L_HIP, L_KNEE),
    "right_hip": (R_SHOULDER, R_HIP, R_KNEE),
    "left_knee": (L_HIP, L_KNEE, L_ANKLE),
    "right_knee": (R_HIP, R_KNEE, R_ANKLE),
    "left_ankle": (L_KNEE, L_ANKLE, 31),
    "right_ankle": (R_KNEE, R_ANKLE, 32),
}

# Pairs used for symmetry reporting
SYM_PAIRS = [("left_elbow", "right_elbow"), ("left_shoulder", "right_shoulder"),
             ("left_hip", "right_hip"), ("left_knee", "right_knee"),
             ("left_ankle", "right_ankle"),
             ("wrist_amp", "wrist_amp"), ("ankle_amp", "ankle_amp")]


def angle_deg(a, b, c) -> float | None:
    """Angle at vertex b formed by points a-b-c, in degrees (2D)."""
    if a is None or b is None or c is None:
        return None
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cosang = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _vis(lms, idx) -> bool:
    return float(lms[idx, 3]) >= VIS


def _xy(lms, idx):
    return (float(lms[idx, 0]), float(lms[idx, 1]))


# Per-frame keys that may legitimately be negative (they encode a direction).
# Everything else in the per-frame metric dict is a magnitude 0..~180.
SIGNED_FRAME_KEYS = frozenset({
    "trunk_lean", "trunk_lean_fwd", "head_tilt", "shoulder_tilt",
    "pelvic_tilt_deg", "shin_angle_l", "shin_angle_r",
    "knee_valgus_l", "knee_valgus_r", "overstride_l", "overstride_r",
    "hip_ext_l", "hip_ext_r",
})


def _facing_x(lms, fm) -> float | None:
    """+1 / -1: which image-x direction the runner's head points (facing).

    Used to give forward/backward quantities a stable sign in a side view.
    Returns None when the head is not visible enough to tell.
    """
    ref = fm.get("_midhip_x")
    if ref is None:
        return None
    for idx in (NOSE, L_EAR, R_EAR):
        if _vis(lms, idx):
            d = float(lms[idx, 0]) - ref
            if abs(d) > 0.004:
                return 1.0 if d > 0 else -1.0
    return None


def shin_angle_deg(knee, ankle, facing: float = 1.0) -> float | None:
    """Angle of the knee->ankle (shin) vector vs TRUE VERTICAL, in degrees.

    Signed: positive = ankle ahead of the knee (shin inclined forward) — the
    braking/over-stride posture at touchdown; negative = ankle behind the knee.
    0 deg = shin exactly vertical, which is the optimal touchdown posture
    (an alternative `|value|` reading is used for scoring).

    Returns None when the shin does not point downward (swing phase / bad pose).
    """
    if knee is None or ankle is None:
        return None
    dx = (ankle[0] - knee[0]) * (1.0 if facing is None else facing)
    dy = ankle[1] - knee[1]                 # image y grows downward
    if dy <= 0.02:                          # ankle not below the knee
        return None
    return math.degrees(math.atan2(dx, dy))


def frontal_knee_deviation(hip, knee, ankle, pelvis) -> float | None:
    """Frontal-plane knee collapse angle relative to the hip-ankle axis.

    Magnitude: 180 deg minus the hip-knee-ankle angle in the frontal plane —
    i.e. how far the knee falls off the straight hip->ankle line.
    Sign: positive = knee displaced TOWARD the body midline (valgus /
    medial collapse), negative = away from the midline (varus).  The
    midline comparison is done in x only (a frontal-plane quantity), so the
    large vertical knee->pelvis component cannot flip the sign.
    """
    if hip is None or knee is None or ankle is None or pelvis is None:
        return None
    ang = angle_deg(hip, knee, ankle)
    if ang is None:
        return None
    mag = 180.0 - ang
    # perpendicular foot of the knee on the hip->ankle line
    vx, vy = ankle[0] - hip[0], ankle[1] - hip[1]
    n2 = vx * vx + vy * vy
    if n2 < 1e-12:
        return None
    t = ((knee[0] - hip[0]) * vx + (knee[1] - hip[1]) * vy) / n2
    px, py = hip[0] + t * vx, hip[1] + t * vy
    ox, oy = knee[0] - px, knee[1] - py            # offset of the knee
    mx = pelvis[0] - px          # x-direction from the axis toward the midline
    if abs(mx) > 1e-6:
        toward_midline = ox * mx > 0
    else:                        # axis passes through the pelvis x: fall back
        toward_midline = (ox * (pelvis[0] - knee[0])
                          + oy * (pelvis[1] - knee[1])) > 0
    return mag if toward_midline else -mag


def pelvic_tilt_deg(l_hip, r_hip) -> float | None:
    """Signed frontal tilt of the inter-hip line, in degrees.

    Positive = the RIGHT hip is lower than the left hip (right-side drop);
    negative = the left hip is lower.
    """
    if l_hip is None or r_hip is None:
        return None
    dx = abs(r_hip[0] - l_hip[0])
    dy = r_hip[1] - l_hip[1]
    if dx < 0.01:
        return None
    return math.degrees(math.atan2(dy, dx))


def frame_metrics(lms) -> dict:
    """Compute all per-frame metrics from (33,4) landmarks. None-safe."""
    fm: dict = {}
    if lms is None:
        return fm

    for name, (a, b, c) in JOINT_SPECS.items():
        if _vis(lms, a) and _vis(lms, b) and _vis(lms, c):
            fm[name] = angle_deg(_xy(lms, a), _xy(lms, b), _xy(lms, c))

    # trunk lean: angle of midhip->midshoulder vs vertical (up = -y)
    if _vis(lms, L_SHOULDER) and _vis(lms, R_SHOULDER) and _vis(lms, L_HIP) and _vis(lms, R_HIP):
        sx = (lms[L_SHOULDER, 0] + lms[R_SHOULDER, 0]) / 2
        sy = (lms[L_SHOULDER, 1] + lms[R_SHOULDER, 1]) / 2
        hx = (lms[L_HIP, 0] + lms[R_HIP, 0]) / 2
        hy = (lms[L_HIP, 1] + lms[R_HIP, 1]) / 2
        fm["trunk_lean"] = math.degrees(math.atan2(sx - hx, max(1e-6, hy - sy)))
        fm["_midsh_x"], fm["_midsh_y"] = float(sx), float(sy)
        fm["_midhip_x"], fm["_midhip_y"] = float(hx), float(hy)

    # neck-trunk angle at shoulder (ear-midpoint vs hip-midpoint)
    if _vis(lms, L_EAR) and _vis(lms, R_EAR) and _vis(lms, L_SHOULDER) and _vis(lms, R_SHOULDER) and _vis(lms, L_HIP) and _vis(lms, R_HIP):
        ex = (lms[L_EAR, 0] + lms[R_EAR, 0]) / 2
        ey = (lms[L_EAR, 1] + lms[R_EAR, 1]) / 2
        if "_midsh_x" in fm:
            fm["neck_trunk"] = angle_deg((ex, ey), (fm["_midsh_x"], fm["_midsh_y"]),
                                         (fm["_midhip_x"], fm["_midhip_y"]))

    # head tilt: ear-ear line vs horizontal
    if _vis(lms, L_EAR) and _vis(lms, R_EAR):
        dy = lms[R_EAR, 1] - lms[L_EAR, 1]
        dx = lms[R_EAR, 0] - lms[L_EAR, 0]
        fm["head_tilt"] = abs(math.degrees(math.atan2(dy, max(1e-6, abs(dx)))))

    # shoulder tilt (roll)
    if _vis(lms, L_SHOULDER) and _vis(lms, R_SHOULDER):
        dy = lms[R_SHOULDER, 1] - lms[L_SHOULDER, 1]
        dx = lms[R_SHOULDER, 0] - lms[L_SHOULDER, 0]
        fm["shoulder_tilt"] = abs(math.degrees(math.atan2(dy, max(1e-6, abs(dx)))))

    # ------------------------------------------------ facing + pelvis centre
    facing = _facing_x(lms, fm)
    fm["_facing"] = facing if facing is not None else 0.0
    pelvis = None
    if _vis(lms, L_HIP) and _vis(lms, R_HIP):
        # float(): landmarks can arrive as float32, and the session analysis
        # (metrics._col) only accepts real Python numbers
        pelvis = (float((lms[L_HIP, 0] + lms[R_HIP, 0]) / 2.0),
                  float((lms[L_HIP, 1] + lms[R_HIP, 1]) / 2.0))
        fm["_pelvis_x"], fm["_pelvis_y"] = pelvis
        pt = pelvic_tilt_deg(_xy(lms, L_HIP), _xy(lms, R_HIP))
        if pt is not None:
            fm["pelvic_tilt_deg"] = pt

    # ------------------------------------------------ forward-normalised trunk
    if facing and "_midsh_x" in fm:
        dx = (fm["_midsh_x"] - fm["_midhip_x"]) * facing
        dy = fm["_midhip_y"] - fm["_midsh_y"]
        fm["trunk_lean_fwd"] = math.degrees(math.atan2(dx, max(1e-6, abs(dy))))
        fm["trunk_lean_abs"] = abs(fm["trunk_lean_fwd"])

    # ------------------------------------------------ sagittal leg metrics
    for side, hip_i, knee_i, ank_i in (("l", L_HIP, L_KNEE, L_ANKLE),
                                       ("r", R_HIP, R_KNEE, R_ANKLE)):
        have_leg = _vis(lms, hip_i) and _vis(lms, knee_i) and _vis(lms, ank_i)
        if have_leg:
            kn, an = _xy(lms, knee_i), _xy(lms, ank_i)
            sa = shin_angle_deg(kn, an, facing if facing else 1.0)
            if sa is not None:
                fm[f"shin_angle_{side}"] = sa
            ang = angle_deg(_xy(lms, hip_i), kn, an)
            if ang is not None:
                fm[f"knee_flex_{side}"] = 180.0 - ang     # 0 = straight leg
            if pelvis is not None:
                fm[f"overstride_{side}"] = (an[0] - pelvis[0]) * (facing if facing else 1.0)
            # sagittal thigh angle vs vertical: + = knee forward of the hip
            # (flexion), - = thigh behind the hip (hip extension, toe-off proxy)
            dxth = (_xy(lms, knee_i)[0] - _xy(lms, hip_i)[0]) * (facing if facing else 1.0)
            dyth = _xy(lms, knee_i)[1] - _xy(lms, hip_i)[1]
            if dyth > 0.02:
                fm[f"hip_ext_{side}"] = -math.degrees(math.atan2(dxth, dyth))
        # -------------------------------------------- frontal knee collapse
        if have_leg and pelvis is not None:
            kv = frontal_knee_deviation(_xy(lms, hip_i), _xy(lms, knee_i),
                                        _xy(lms, ank_i), pelvis)
            if kv is not None:
                fm[f"knee_valgus_{side}"] = kv

    # key positions (raw xy for time series)
    pos = {"nose": NOSE, "l_ankle": L_ANKLE, "r_ankle": R_ANKLE,
           "l_hip": L_HIP, "r_hip": R_HIP, "l_wrist": L_WRIST, "r_wrist": R_WRIST,
           "l_knee": L_KNEE, "r_knee": R_KNEE, "l_foot": 31, "r_foot": 32,
           "l_heel": 29, "r_heel": 30, "l_sh": L_SHOULDER, "r_sh": R_SHOULDER}
    for name, idx in pos.items():
        if _vis(lms, idx):
            fm[f"{name}_x"] = float(lms[idx, 0])
            fm[f"{name}_y"] = float(lms[idx, 1])
    return fm


# ---------------------------------------------------------------- session --

def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return x.copy()
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _col(rows, key):
    """Column as a float array; numpy scalars (float32/float64) are accepted."""
    out = []
    for r in rows:
        v = r.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f if math.isfinite(f) else float("nan"))
    return np.array(out, dtype=float)


def _find_strikes(t, y, min_gap=0.2):
    """Foot strikes = local maxima of ankle y (lowest point), robust to noise."""
    idx = np.where(~np.isnan(y))[0]
    if len(idx) < 10:
        return []
    yy = y[idx]
    if len(yy) >= 5:
        k = np.ones(3) / 3
        yy = np.convolve(yy, k, mode="same")
    med = float(np.median(yy))
    span = float(np.percentile(yy, 95) - np.percentile(yy, 5))
    thr = med + 0.25 * span
    strikes = []
    last_t = None
    for i in range(1, len(idx) - 1):
        if yy[i] >= yy[i - 1] and yy[i] > yy[i + 1] and yy[i] > thr:
            j = idx[i]
            if last_t is None or t[j] - last_t >= min_gap:
                strikes.append((t[j], j))
                last_t = t[j]
    return strikes


def _cadence_from_peak_intervals(strike_times):
    if len(strike_times) < 3:
        return None
    dts = np.diff(strike_times)
    dts = dts[(dts > 0.15) & (dts < 1.6)]
    if len(dts) < 2:
        return None
    return 60.0 / float(np.median(dts))


def _cadence_fft(t, y, band=(2.0, 3.6)):
    """Dominant oscillation frequency (Hz) in `band`, or None.

    Note: the hip/COM oscillates at STEP frequency; a single ankle oscillates
    at STRIDE frequency (half the step frequency).
    """
    idx = np.where(~np.isnan(y))[0]
    if len(idx) < 40:
        return None
    tt = t[idx]
    yy = y[idx]
    if tt[-1] - tt[0] < 3.0:
        return None
    # resample to uniform grid
    grid = np.linspace(tt[0], tt[-1], max(64, len(tt)))
    sig = np.interp(grid, tt, yy)
    sig = sig - _rolling_mean(sig, max(3, int(len(sig) * 0.2)))
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), d=(grid[1] - grid[0]))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any() or spec[mask].max() <= 0:
        return None
    f = freqs[mask][np.argmax(spec[mask])]
    return float(f)


def _robust_range(y):
    v = y[~np.isnan(y)]
    if len(v) < 10:
        return None
    return float(np.percentile(v, 95) - np.percentile(v, 5))


def _dominant_facing(rows) -> float:
    """Runner's facing direction (+1 / -1 in image x) over the whole session."""
    vals = []
    for r in rows:
        try:
            f = float(r.get("_facing") or 0.0)
        except (TypeError, ValueError):
            f = 0.0
        if f:
            vals.append(f)
    if not vals:
        return 1.0
    return 1.0 if float(np.sum(vals)) >= 0 else -1.0


def _signed_series(rows, key, dom: float) -> np.ndarray:
    """Signed per-frame series re-signed with the session-dominant facing.

    Per-frame signed values use the facing detected in that frame; frames where
    head detection jittered to the opposite side are flipped so the series is
    consistent (sign flips are exactly +/-1 so this is lossless).
    """
    v = _col(rows, key)
    f = _col(rows, "_facing")
    agree = np.where(np.isnan(f) | (np.abs(f) < 0.5), True, (f * dom) >= 0)
    return np.where(agree, v, -v)


def _nearest(times: np.ndarray, t0) -> int:
    d = np.abs(times - float(t0))
    d = np.where(np.isnan(d), np.inf, d)
    if d.size == 0 or not np.isfinite(d).any():
        return -1
    return int(np.argmin(d))


def _t_sort_key(r):
    try:
        v = float(r.get("t_rel"))
    except (TypeError, ValueError):
        return float("inf")
    return v if math.isfinite(v) else float("inf")


def _asym_index(a, b):
    """Asymmetry index per spec: (1 - |L-R| / ((L+R)/2)) * 100."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    m = (abs(a) + abs(b)) / 2.0
    if m <= 1e-9:
        return None
    return max(0.0, 100.0 * (1.0 - abs(a - b) / m))


def analyze_session(rows, *, weight_kg=None, height_cm=None, speed_kmh=None,
                    view_plane=None, events=None, discipline=None) -> dict:
    """Compute session summary from logged frame metrics rows.

    view_plane : "sagittal" (side view) or "frontal" (front/back view).  Metrics
                 that cannot be measured from the chosen plane are never
                 invented: they are removed and listed in out["not_assessed"]
                 with the reason (config.NOT_ASSESSED_TEXT).
    events     : optional pre-latched gait events (gait_events.detect_events or
                 a GaitLatch); detected from the rows when omitted.
    discipline : "running" (default) or "cycling".  For cycling the running-gait
                 blocks (contact/takeoff events, step cadence, overstride) are
                 not produced at all — they are moved to not_assessed with a
                 "not applicable" reason, because pedalling is not a gait.

    Returns a JSON-safe dict with summary values, per-joint stats, symmetry,
    biomechanical gait metrics, form metrics vs reference bands, and heuristic
    form indices.
    """
    disc = str(discipline or "running").lower()
    out: dict = {"n_frames": len(rows), "labels": []}
    if not rows:
        out["error"] = "no data"
        return out
    # stable sort: a video seek can feed rows out of order, never a broken axis
    rows = sorted(rows, key=_t_sort_key)
    plane = str(view_plane or config.DEFAULT_VIEW_PLANE).lower()
    if plane not in config.VIEW_PLANES:
        plane = config.DEFAULT_VIEW_PLANE
    out["view_plane"] = plane

    t = _col(rows, "t_rel")
    mask_t = ~np.isnan(t)
    t = t[mask_t]
    if len(t) == 0:
        out["error"] = "no timestamps"
        return out
    out["duration_s"] = float(t[-1] - t[0])
    if len(t) > 2:
        dts = np.diff(t)
        dts = dts[dts > 0]
        if len(dts):
            out["mean_fps"] = round(1.0 / float(np.median(dts)), 1)

    def series(key):
        v = _col(rows, key)[mask_t]
        return v

    # ---------- joint angles: stats per joint
    joints = {}
    for jname in JOINT_SPECS:
        v = series(jname)
        ok = ~np.isnan(v)
        if ok.sum() < 5:
            continue
        vv = v[ok]
        ang_vel = np.abs(np.diff(vv)) / np.maximum(np.diff(t[ok]), 1e-3)
        joints[jname] = {
            "mean": round(float(np.mean(vv)), 1),
            "min": round(float(np.min(vv)), 1),
            "max": round(float(np.max(vv)), 1),
            "rom": round(float(np.max(vv) - np.min(vv)), 1),
            "ang_vel_mean": round(float(np.mean(ang_vel)), 1),   # deg/s
            "coverage_pct": round(100.0 * ok.sum() / len(v), 1),
        }
    out["joints"] = joints

    # ---------- symmetry (mean angle L vs R)
    symmetry = {}
    sym_vals = []
    for lname, rname in [("left_knee", "right_knee"), ("left_hip", "right_hip"),
                         ("left_ankle", "right_ankle"), ("left_elbow", "right_elbow"),
                         ("left_shoulder", "right_shoulder")]:
        if lname in joints and rname in joints:
            a, b = joints[lname]["mean"], joints[rname]["mean"]
            denom = max(1e-6, (abs(a) + abs(b)) / 2)
            sym = max(0.0, 100.0 * (1.0 - abs(a - b) / denom))
            symmetry[lname.replace("left_", "")] = round(sym, 1)
            sym_vals.append(sym)
    out["symmetry"] = symmetry
    out["symmetry_mean_pct"] = round(float(np.mean(sym_vals)), 1) if sym_vals else None

    # ---------- trunk
    trunk = series("trunk_lean")
    trunk = trunk[np.abs(trunk) < 60]  # reject wild outliers
    if np.sum(~np.isnan(trunk)) > 5:
        out["trunk_lean_deg"] = {"mean": round(float(np.nanmean(trunk)), 1),
                                 "std": round(float(np.nanstd(trunk)), 1)}
    head_tilt = series("head_tilt")
    if np.sum(~np.isnan(head_tilt)) > 5:
        out["head_tilt_deg_mean"] = round(float(np.nanmean(head_tilt)), 1)

    # ---------- cadence (steps/min, both feet)
    l_ank_y = series("l_ankle_y")
    r_ank_y = series("r_ankle_y")
    strikes_all, per_side = [], []
    for ank in (l_ank_y, r_ank_y):
        st = [s[0] for s in _find_strikes(t, ank)]
        strikes_all.extend(st)
        per_side.append(st)
    strikes_all.sort()
    merged = []
    for ts_ in strikes_all:
        if not merged or ts_ - merged[-1] > 0.12:
            merged.append(ts_)
    cand = []
    c = _cadence_from_peak_intervals(merged)
    if c and 100 <= c <= 250:
        cand.append(c)
    else:
        # per-side strikes give half of total cadence (one strike per stride)
        for st in per_side:
            c2 = _cadence_from_peak_intervals(st)
            if c2 and 50 <= c2 <= 130:
                cand.append(c2 * 2.0)
        if not cand:
            hip_hz = _cadence_fft(t, series("_midhip_y"), band=(2.0, 3.6))
            if hip_hz:
                c3 = hip_hz * 60.0
                if 100 <= c3 <= 250:
                    cand.append(c3)
                    out["labels"].append("cadence from hip frequency estimate")
            if not cand:
                for ank in (l_ank_y, r_ank_y):
                    stride_hz = _cadence_fft(t, ank, band=(1.0, 1.9))
                    if stride_hz:
                        c4 = stride_hz * 120.0   # stride -> steps (x2), Hz -> /min (x60)
                        if 100 <= c4 <= 250:
                            cand.append(c4)
                            out["labels"].append("cadence from ankle frequency estimate")
                            break
    if cand:
        out["cadence_spm"] = round(float(np.median(cand)), 1)
        out["strike_count"] = len(merged)
    if speed_kmh:
        out["speed_kmh"] = float(speed_kmh)
        if out.get("cadence_spm"):
            step_len_m = (speed_kmh * 1000 / 60) / out["cadence_spm"]
            out["step_length_m"] = round(step_len_m, 2)
            out["stride_length_m"] = round(step_len_m * 2, 2)

    # ---------- scale calibration (needs height)
    leg_norms = []
    leg_px = []
    for lhip, lank in (("l_hip_y", "l_ankle_y"), ("r_hip_y", "r_ankle_y")):
        lg = np.abs(_col(rows, lank)[mask_t] - _col(rows, lhip)[mask_t])
        lg = lg[~np.isnan(lg)]
        if len(lg):
            leg_norms.append(float(np.median(lg)))
    leg_norm = max(leg_norms) if leg_norms else None
    m_per_unit = None
    if leg_norm and height_cm:
        leg_m = 0.47 * (height_cm / 100.0)
        m_per_unit = leg_m / leg_norm
        out["scale_m_per_unit"] = round(m_per_unit, 4)

    # ---------- vertical oscillation (hip midpoint y)
    mh_y = _col(rows, "_midhip_y")[mask_t]
    osc_px = _robust_range(mh_y)
    if osc_px is not None:
        out["vertical_osc_norm"] = round(float(osc_px), 4)
        if m_per_unit:
            osc_cm = osc_px * m_per_unit * 100.0
            out["vertical_osc_cm"] = round(float(osc_cm), 1)

    # ---------- wrist / head / ankle amplitudes
    for side in ("l", "r"):
        wy = _col(rows, f"{side}_wrist_y")[mask_t]
        r_ = _robust_range(wy)
        if r_ is not None:
            out[f"wrist_amp_{side}"] = round(float(r_), 4)
    wx_l = _robust_range(_col(rows, "l_wrist_x")[mask_t])
    wx_r = _robust_range(_col(rows, "r_wrist_x")[mask_t])
    nose_x = _robust_range(_col(rows, "nose_x")[mask_t])
    nose_y = _robust_range(_col(rows, "nose_y")[mask_t])
    if nose_x is not None:
        out["head_sway_norm"] = round(float(nose_x), 4)
    if nose_y is not None:
        out["head_bob_norm"] = round(float(nose_y), 4)
    if wx_l is not None and wx_r is not None:
        out["wrist_amp_sym_norm"] = round(float(min(wx_l, wx_r) / max(wx_l, wx_r, 1e-6)), 3)

    # ---------- overstride proxy at strikes (ankle ahead of hip)
    overs = []
    for side, hipk, ankck in (("l", "l_hip_x", "l_ankle_x"), ("r", "r_hip_x", "r_ankle_x")):
        ank_y = series(f"{side}_ankle_y")
        st = _find_strikes(t, ank_y)
        ax = series(ankck)
        hx = series(hipk)
        for _, j in st[:80]:
            if j < len(ax) and j < len(hx) and not (np.isnan(ax[j]) or np.isnan(hx[j])):
                overs.append(abs(ax[j] - hx[j]))
    if overs and leg_norm:
        out["overstride_body_units"] = round(float(np.mean(overs)) / leg_norm, 3)

    # ---------- power estimate (vertical mechanical power, kinematic proxy)
    if weight_kg and out.get("vertical_osc_cm") and out.get("cadence_spm"):
        m = float(weight_kg)
        h = out["vertical_osc_cm"] / 100.0
        f_step = out["cadence_spm"] / 60.0
        p_vert = m * G * h * f_step
        out["power_est_watts"] = round(p_vert, 1)
        out["labels"].append("power is a kinematic estimate (mass x g x vertical oscillation x step rate)")

    if speed_kmh and weight_kg:
        out["speed_mps"] = round(speed_kmh / 3.6, 2)
        out["external_power_est_watts"] = round(float(weight_kg) * G * (speed_kmh / 3.6) * 0.25, 1)
        out["labels"].append("external power proxy assumes 0.25 vertical cost factor")

    # ======================================================================
    # Biomechanical metric set — per leg where applicable.
    # Values are sampled at the latched gait events (initial contact,
    # mid-stance, toe-off) so they describe a phase of the gait cycle instead
    # of an arbitrary frame.
    # ======================================================================
    dom = _dominant_facing(rows)

    def sseries(key):
        """Signed per-frame series, re-signed to the session facing."""
        return _signed_series(rows, key, dom)[mask_t]

    def vals_at(key, evs, signed=False):
        arr = sseries(key) if signed else series(key)
        got = []
        for e in evs:
            j = _nearest(t, e.get("t"))
            if j < 0 or j >= arr.size:
                continue
            v = arr[j]
            if v is not None and np.isfinite(float(v)):
                got.append(float(v))
        return got

    events = list(events) if events is not None else GA.detect_events(rows)
    out["gait_events"] = GA.summarize(events)
    # ---- data quality: is the lower body even in this footage? -------------
    lb_cov = GA.lower_body_coverage(rows)
    lb_ok = not GA.events_not_assessable(lb_cov)
    out["data_quality"] = {
        "lower_body_coverage_pct": round(lb_cov, 1),
        "lower_body_ok": bool(lb_ok),
        "min_lower_body_coverage_pct": config.LOWER_BODY_MIN_COVERAGE_PCT,
    }
    if not lb_ok:
        out["data_quality"]["note"] = config.GAIT_NOT_ASSESSABLE_TEXT
        out["gait_events"]["note"] = config.GAIT_NOT_ASSESSABLE_TEXT
        out["labels"].append(config.GAIT_NOT_ASSESSABLE_TEXT)
    elif out["gait_events"]["n"]:
        out["labels"].append(
            "gait events latched from ankle velocity (initial contact / toe-off) "
            "and knee flexion (mid-stance)")

    ev_by = {(typ, side): [e for e in events
                           if e.get("type") == typ and e.get("side") == side]
             for typ in ("IC", "MS", "TO") for side in GA.SIDES}
    # fallback: when no IC was latched, use the classic strike detection so the
    # sagittal metrics are still reported (labelled as strike-based).
    ic_fallback = {}
    for side in GA.SIDES:
        if ev_by[("IC", side)]:
            ic_fallback[side] = False
            continue
        st = _find_strikes(t, series(GA.ANK_Y_KEY[side]))[:60]
        ev_by[("IC", side)] = [{"type": "IC", "side": side, "t": s[0]} for s in st]
        ic_fallback[side] = bool(st)

    # ---- touchdown shin angle (sagittal) ----------------------------------
    shin_all, shin_side = [], {}
    for side in GA.SIDES:
        v = vals_at(GA.SHIN_KEY[side], ev_by[("IC", side)], signed=True)
        if v:
            shin_side[side] = round(float(np.mean(np.abs(v))), 1)
            shin_all += [abs(x) for x in v]
    if shin_all:
        out["touchdown_shin_deg"] = {
            "mean": round(float(np.mean(shin_all)), 1),
            "max": round(float(np.max(shin_all)), 1),
            "n": len(shin_all),
            "per_side": shin_side,
            "band_note": "knee->ankle vector vs true vertical at footstrike; "
                         "0-5 deg optimal, >8 deg suggests high braking force",
        }

    # ---- overstride vector at initial contact -----------------------------
    over_all, over_side = [], {}
    for side in GA.SIDES:
        v = vals_at(GA.OVER_KEY[side], ev_by[("IC", side)], signed=True)
        if v:
            over_side[side] = round(float(np.mean(v)), 3)
            over_all += v
    if over_all:
        arr = np.array(over_all, dtype=float)
        ahead = arr[arr > 0]
        frac = None
        cm_est = None
        if leg_norm:
            frac = float(np.mean(ahead)) / leg_norm if ahead.size else 0.0
            if m_per_unit:
                cm_est = float(np.mean(arr)) * m_per_unit * 100.0
        out["overstride_leg_frac"] = {
            "mean": round(float(frac), 3) if frac is not None else None,
            "mean_ahead_leg_frac": round(frac, 3) if frac is not None else None,
            "signed_mean": round(float(np.mean(arr)), 3),
            "signed_min": round(float(np.min(arr)), 3),
            "signed_max": round(float(np.max(arr)), 3),
            "n": int(arr.size),
            "per_side": over_side,
            "cm_est": round(cm_est, 1) if cm_est is not None else None,
            "is_estimate": True,
        }
        out["labels"].append(
            "overstride: landing ankle -> pelvis centre, expressed in leg lengths; "
            "the cm value is a height-based ESTIMATE (leg length = 0.47 x height)")

    # ---- dynamic knee flexion at initial contact / mid-stance -------------
    def flexion_block(typ):
        per, allv = {}, []
        for side in GA.SIDES:
            v = vals_at(GA.KNEE_KEY[side], ev_by[(typ, side)])
            if v:
                f = [180.0 - x for x in v]
                per[side] = round(float(np.mean(f)), 1)
                allv += f
        if not allv:
            return None
        return {"mean": round(float(np.mean(allv)), 1),
                "max": round(float(np.max(allv)), 1),
                "per_side": per, "n": len(allv)}

    for key, typ, extra in (("knee_flexion_ic_deg", "IC",
                             "hip-knee-ankle angle at initial contact (absorption begins)"),
                            ("knee_flexion_ms_deg", "MS",
                             "peak stance knee flexion under load; 35-45 deg = normal absorption")):
        blk = flexion_block(typ)
        if blk:
            blk["note"] = extra
            out[key] = blk

    # ---- contralateral pelvic drop during single-leg stance ---------------
    tilt = sseries("pelvic_tilt_deg")           # + = right hip lower
    drop_left, drop_right = [], []              # stance side -> contralateral drop
    for side, bucket in (("left", drop_right), ("right", drop_left)):
        for e in ev_by[("MS", side)]:
            j = _nearest(t, e.get("t"))
            if 0 <= j < tilt.size and np.isfinite(tilt[j]):
                # during LEFT stance the right hip is the contralateral one
                bucket.append(float(tilt[j]) if side == "left" else -float(tilt[j]))
    if drop_left or drop_right:
        both = drop_left + drop_right
        out["pelvic_drop_deg"] = {
            "mean": round(float(np.mean(both)), 1),
            "max": round(float(np.max(both)), 1),
            "n": len(both),
            "drop_left_stance": round(float(np.mean(drop_left)), 1) if drop_left else None,
            "drop_right_stance": round(float(np.mean(drop_right)), 1) if drop_right else None,
            "note": "frontal tilt of the inter-hip line during single-leg stance "
                    "(positive = contralateral hip lower); >4-5 deg flags potential "
                    "gluteus-medius weakness",
        }

    # ---- dynamic knee valgus / varus (frontal) ----------------------------
    valgus = {}
    for side in GA.SIDES:
        v = vals_at(GA.VALGUS_KEY[side], ev_by[("MS", side)])
        if not v:
            v = vals_at(GA.VALGUS_KEY[side], events)
        if v:
            valgus[side] = {
                "mean": round(float(np.mean(v)), 1),
                "max": round(float(np.max(v)), 1),
                "n": len(v),
            }
    if valgus:
        allv = [d["mean"] for d in valgus.values()]
        out["knee_valgus_deg"] = {
            "mean": round(float(np.mean(allv)), 1),
            "max": round(float(max(d["max"] for d in valgus.values())), 1),
            "per_side": valgus,
            "n": int(sum(d["n"] for d in valgus.values())),
            "note": "frontal knee collapse vs the hip-ankle axis (deviation from a "
                    "straight hip-knee-ankle line); positive = medial collapse "
                    "(valgus), negative = varus; measured under load",
        }

    # ---- trunk lean, forward-normalised -----------------------------------
    tl = sseries("trunk_lean_fwd")
    if np.sum(~np.isnan(tl)) > 5:
        out["trunk_lean_fwd_deg"] = {"mean": round(float(np.nanmean(tl)), 1),
                                     "std": round(float(np.nanstd(tl)), 1)}
        out["labels"].append("trunk lean is reported signed (raw) and "
                             "forward-normalised (positive = anterior lean)")

    # ---- hip extension + push-off ankle angle at toe-off ------------------
    he, po = [], []
    for side in GA.SIDES:
        he += vals_at(GA.HIPEXT_KEY[side], ev_by[("TO", side)], signed=True)
        po += vals_at(GA.ANKLE_ANG_KEY[side], ev_by[("TO", side)])
    if he:
        out["hip_extension_toeoff_deg"] = {
            "mean": round(float(np.mean(he)), 1),
            "max": round(float(np.max(he)), 1),
            "n": len(he),
            "note": "thigh angle behind the vertical at toe-off (hip-extension proxy)",
        }
    if po:
        out["pushoff_ankle_deg"] = {"mean": round(float(np.mean(po)), 1), "n": len(po)}

    # ---- stance time + left/right asymmetry index (spec formula) ----------
    st_times = GA.stance_times(events)
    asym: dict = {}
    if st_times["left"] and st_times["right"]:
        l_t, r_t = float(np.mean(st_times["left"])), float(np.mean(st_times["right"]))
        out["stance_time_s"] = {"left": round(l_t, 3), "right": round(r_t, 3),
                                "mean": round((l_t + r_t) / 2.0, 3),
                                "n": len(st_times["left"]) + len(st_times["right"])}
        ai = _asym_index(l_t, r_t)
        if ai is not None:
            asym["stance_time_pct"] = round(ai, 1)
    knee_ic = out.get("knee_flexion_ic_deg", {}).get("per_side") or {}
    knee_ms = out.get("knee_flexion_ms_deg", {}).get("per_side") or {}
    for label, per in (("knee_angle_pct", knee_ic), ("knee_flexion_ms_pct", knee_ms)):
        if "left" in per and "right" in per:
            ai = _asym_index(per["left"], per["right"])
            if ai is not None:
                asym[label] = round(ai, 1)
    l_kneeq = joints.get("left_knee", {}).get("mean")
    r_kneeq = joints.get("right_knee", {}).get("mean")
    if l_kneeq is not None and r_kneeq is not None:
        ai = _asym_index(l_kneeq, r_kneeq)
        if ai is not None:
            asym["knee_mean_angle_pct"] = round(ai, 1)
    if asym:
        asym["formula"] = "(1 - |L-R| / ((L+R)/2)) * 100"
        out["asymmetry"] = asym

    if out.get("wrist_amp_sym_norm") is not None:
        out["arm_swing_sym_pct"] = round(100.0 * out["wrist_amp_sym_norm"], 1)

    # ---- mechanical power per kg body mass --------------------------------
    if out.get("power_est_watts") and weight_kg:
        out["power_per_kg_watts"] = round(out["power_est_watts"] / float(weight_kg), 2)
        out["labels"].append("power per kg = vertical mechanical power / body mass (W/kg)")

    # ---------- form metrics vs reference bands (plane aware) --------------
    form = {}
    bands = config.BANDS
    checks = []

    def add_band(name, value, band_key, label=None, band=None):
        if value is None:
            return
        b = band if band is not None else bands.get(band_key)
        entry = {"value": value, "band": tuple(b) if b else None}
        entry["in_band"] = (bool(b[0] <= value <= b[1]) if b else None)
        if label:
            entry["label"] = label
        form[name] = entry

    add_band("cadence_spm", out.get("cadence_spm"), "cadence_spm")
    if plane == "sagittal":
        tl_fwd = out.get("trunk_lean_fwd_deg", {}).get("mean")
        tl_raw = out.get("trunk_lean_deg", {}).get("mean")
        add_band("trunk_lean_deg", tl_fwd if tl_fwd is not None else tl_raw,
                 "trunk_lean_deg", "anterior trunk lean (forward-normalised)")
    add_band("vertical_osc_cm", out.get("vertical_osc_cm"), "vertical_osc_cm")
    if "left_knee" in joints or "right_knee" in joints:
        vals = [joints[j]["rom"] for j in ("left_knee", "right_knee") if j in joints]
        if vals:
            add_band("knee_rom_deg", round(float(np.mean(vals)), 1), "knee_flexion_max_deg")
    add_band("symmetry_pct", out.get("symmetry_mean_pct"), "symmetry_pct")
    if plane == "sagittal":
        shin = out.get("touchdown_shin_deg", {}).get("mean")
        add_band("touchdown_shin_deg", shin, "touchdown_shin_deg",
                 "knee->ankle vs vertical at footstrike (0-5 deg optimal)")
        over = out.get("overstride_leg_frac", {}) or {}
        add_band("overstride_leg_frac", over.get("mean"), "overstride_leg_frac",
                 "landing ankle ahead of the pelvis, in leg lengths")
        add_band("knee_flexion_ms_deg", (out.get("knee_flexion_ms_deg") or {}).get("mean"),
                 "knee_flexion_ms_deg", "stance knee flexion (absorption)")
        ic_flex = (out.get("knee_flexion_ic_deg") or {}).get("mean")
        if ic_flex is not None:
            form["knee_flexion_ic_deg"] = {"value": ic_flex, "band": None,
                                           "in_band": None,
                                           "label": "knee flexion at initial contact "
                                                    "(reported, no band applied)"}
    else:
        add_band("pelvic_drop_deg", (out.get("pelvic_drop_deg") or {}).get("mean"),
                 "pelvic_drop_deg", "contralateral drop in single-leg stance")
        add_band("knee_valgus_deg", (out.get("knee_valgus_deg") or {}).get("mean"),
                 "knee_valgus_deg", "frontal knee collapse vs hip-ankle axis")
        add_band("arm_swing_sym_pct", out.get("arm_swing_sym_pct"), "arm_swing_sym_pct")
    for k, v in form.items():
        if v.get("in_band") is not None:
            checks.append(1.0 if v["in_band"] else 0.0)
    out["form"] = form
    if checks:
        out["form_index"] = round(100.0 * float(np.mean(checks)), 0)
        out["labels"].append("form index = share of tracked metrics within heuristic reference bands")

    # ---------- never invent wrong-plane values ----------------------------
    blocked = config.FRONTAL_ONLY if plane == "sagittal" else config.SAGITTAL_ONLY
    not_assessed = {}
    for k in blocked:
        out.pop(k, None)
        form.pop(k, None)
        not_assessed[k] = config.NOT_ASSESSED_TEXT
    # missing lower body: say WHY, never leave a bare zero / empty section
    if not lb_ok:
        not_assessed["gait_events"] = config.GAIT_NOT_ASSESSABLE_TEXT
        for k in ("touchdown_shin_deg", "overstride_leg_frac", "knee_flexion_ic_deg",
                  "knee_flexion_ms_deg", "pelvic_drop_deg", "knee_valgus_deg",
                  "hip_extension_toeoff_deg", "pushoff_ankle_deg", "stance_time_s",
                  "cadence_spm", "vertical_osc_cm", "power_est_watts",
                  "power_per_kg_watts"):
            if out.get(k) is None:
                not_assessed[k] = config.GAIT_NOT_ASSESSABLE_TEXT
                form.pop(k, None)
    # ======================================================================
    # Cycling sessions: the running-gait blocks above are not meaningful for
    # pedalling.  Never delete silently — move them into not_assessed with an
    # explicit reason; the bike metrics live in summary["bike"] (cycling.py).
    # ======================================================================
    if disc == "cycling":
        na = not_assessed
        for k in ("gait_events", "cadence_spm", "strike_count", "step_length_m",
                  "stride_length_m", "vertical_osc_cm", "vertical_osc_norm",
                  "overstride_leg_frac", "overstride_body_units",
                  "touchdown_shin_deg", "knee_flexion_ic_deg",
                  "knee_flexion_ms_deg", "hip_extension_toeoff_deg",
                  "pushoff_ankle_deg", "stance_time_s", "power_est_watts",
                  "power_per_kg_watts", "external_power_est_watts"):
            # running-gait metrics are not applicable to a cycling session —
            # say so even when no value was computed (never leave a silent blank)
            out.pop(k, None)
            na[k] = config.CYCLING_NOT_APPLICABLE_TEXT
            form.pop(k, None)
        out["form"] = {}
        out.pop("form_index", None)
        out["discipline"] = "cycling"
        out["labels"].append(
            "cycling session: running-gait metrics (contact events, step cadence, "
            "overstride) are not applicable; see the cycling position analysis")
    out["not_assessed"] = not_assessed
    return out
