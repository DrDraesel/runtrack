"""Cycling analysis for RunTrack — pedalling metrics, bike-fit targets,
rear-view control metrics and a video-based frontal-area (aero) estimate.

Design notes (kept honest on purpose):

* Angles come from the same 2D MediaPipe landmarks as the running pipeline
  (hip-knee-ankle included angle, shoulder-hip-knee hip angle, trunk lean).
  A side view is required for the angles, a front/back view for the travel
  metrics and the frontal area.
* Reference targets are the ranges used by industry video-fit tools
  (Velogic Studio metric pages), the Holmes method (25-35 deg knee flexion
  at bottom dead centre) and common fit practice (torso angle road 35-45,
  tri/TT 10-30).  They are guidance bands for a fit session — not norms.
* Frontal area is measured from the MediaPipe segmentation mask and scaled
  with the rider's shoulder width (measured in-frame) or 0.245 x height.
  Absolute FA is an ESTIMATE (perspective, clothing); position-to-position
  comparisons on the same camera setup are the reliable output.
"""
from __future__ import annotations

import math

import numpy as np

import config
import gait_events as GA

SIDES = ("left", "right")
KNEE_KEY = {"left": "left_knee", "right": "right_knee"}
HIP_KEY = {"left": "left_hip", "right": "right_hip"}
ANKLE_KEY = {"left": "left_ankle", "right": "right_ankle"}
SIDE_SHORT = {"left": "l", "right": "r"}

BIKE_NOTES = (
    "Cycling targets follow common video bike-fit ranges (Velogic Studio "
    "metric guides; Holmes method 25-35 deg knee flexion at BDC; tri/TT "
    "conventions). They are fit guidance, not clinical norms."
)


# ------------------------------------------------------------------ helpers
def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f if math.isfinite(f) else float("nan"))
    return np.array(out, dtype=float)


def _robust_range(v, lo=2.0, hi=98.0):
    x = v[~np.isnan(v)]
    if x.size < 10:
        return None
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def _smooth3(v):
    if v.size < 3:
        return v
    k = np.ones(3) / 3.0
    return np.convolve(v, k, mode="same")


def _peaks(v: np.ndarray, min_gap_n: int = 6):
    """Indices of local maxima (with a minimum separation in frames)."""
    idx = []
    for i in range(1, v.size - 1):
        if math.isnan(v[i]) or math.isnan(v[i - 1]) or math.isnan(v[i + 1]):
            continue
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            if not idx or i - idx[-1] >= min_gap_n:
                idx.append(i)
    return idx


def rpm_from_series(t: np.ndarray, y: np.ndarray, band=(0.6, 2.4)):
    """Pedalling rate (rpm) from the dominant oscillation frequency (FFT).

    One leg passes top+dead centre once per crank revolution, so the knee
    angle oscillates at exactly the cadence frequency.  Band 0.6-2.4 Hz
    = 36-144 rpm.
    """
    ok = ~np.isnan(y)
    if ok.sum() < 60:
        return None
    tt, yy = t[ok], y[ok]
    if tt[-1] - tt[0] < 4.0:
        return None
    grid = np.linspace(tt[0], tt[-1], max(128, len(tt)))
    sig = np.interp(grid, tt, yy)
    sig = sig - float(np.mean(sig))          # mean only: the band excludes DC
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), d=(grid[1] - grid[0]))
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any() or spec[m].max() <= 0:
        return None
    # the dominant oscillation must itself lie in the pedalling band — an
    # in-band reading that is only spectral leakage of a faster motion (or
    # noise) is refused instead of reported as a cadence
    gp = int(np.argmax(spec))
    if not (band[0] <= float(freqs[gp]) <= band[1]):
        return None
    return float(freqs[m][np.argmax(spec[m])]) * 60.0


# ------------------------------------------------------------ sagittal view
def analyze_cycling(rows, bike="road", params=None):
    """Cycling position metrics from frame-metric rows (side view)."""
    params = params or {}
    bike = bike if bike in config.BIKE_TYPES else config.DEFAULT_BIKE_TYPE
    out: dict = {"bike_type": bike, "labels": [BIKE_NOTES]}
    if not rows:
        out["error"] = "no data"
        return out
    t = _col(rows, "t_rel")
    ok_t = ~np.isnan(t)
    if ok_t.sum() < 30:
        out["error"] = "not enough frames"
        return out

    def s(key):
        return _col(rows, key)[ok_t]

    per_side: dict = {}
    rpm_candidates = []
    tt = t[ok_t]
    dts = np.diff(tt)
    dts = dts[(dts > 0) & np.isfinite(dts)]
    fps_est = (1.0 / float(np.median(dts))) if dts.size else 30.0
    min_gap_n = int(max(3, round(fps_est * 0.35)))     # allows up to ~170 rpm
    for side in SIDES:
        knee = _smooth3(s(KNEE_KEY[side]))
        hip = _smooth3(s(HIP_KEY[side]))
        ank = s(ANKLE_KEY[side])
        st: dict = {}
        n_ok = int(np.sum(~np.isnan(knee)))
        if n_ok > 30:
            rpm = rpm_from_series(tt, knee)
            if rpm and config.CYCLING_CADENCE_HARD[0] <= rpm <= config.CYCLING_CADENCE_HARD[1]:
                rpm_candidates.append(rpm)
            pk = _peaks(knee, min_gap_n)
            bdc_vals = [float(knee[i]) for i in pk if knee[i] > 95]
            tdc_vals, hip_tdc = [], []
            for a, b in zip(pk, pk[1:]):
                if b - a < 3:
                    continue
                seg = knee[a + 1:b]
                if seg.size and not np.all(np.isnan(seg)):
                    tdc_vals.append(float(np.nanmin(seg)))
                hseg = hip[a + 1:b]
                if hseg.size and not np.all(np.isnan(hseg)):
                    hip_tdc.append(float(np.nanmin(hseg)))
            if bdc_vals:
                st["knee_bdc_mean_deg"] = round(float(np.mean(bdc_vals)), 1)
                st["knee_bdc_max_deg"] = round(float(np.max(bdc_vals)), 1)
                st["knee_bdc_spread_deg"] = round(float(np.std(bdc_vals)), 1)
                st["revolutions"] = len(bdc_vals)
            if tdc_vals:
                st["knee_tdc_min_deg"] = round(float(np.min(tdc_vals)), 1)
                st["knee_tdc_mean_deg"] = round(float(np.mean(tdc_vals)), 1)
            if hip_tdc:
                st["hip_tdc_min_deg"] = round(float(np.min(hip_tdc)), 1)
                st["hip_tdc_mean_deg"] = round(float(np.mean(hip_tdc)), 1)
            rom = _robust_range(knee)
            if rom is not None:
                st["knee_rom_deg"] = round(rom, 1)
        a_ok = ank[~np.isnan(ank)]
        if a_ok.size > 10:
            st["ankle_mean_deg"] = round(float(np.mean(a_ok)), 1)
        per_side[side] = st
    out["per_side"] = per_side

    # ---- session aggregates
    def _mean_of(key):
        vals = [d[key] for d in per_side.values() if d.get(key) is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    out["knee_bdc_deg"] = _mean_of("knee_bdc_mean_deg")
    out["knee_bdc_max_deg"] = _mean_of("knee_bdc_max_deg")
    out["knee_tdc_min_deg"] = _mean_of("knee_tdc_min_deg")
    out["hip_tdc_min_deg"] = _mean_of("hip_tdc_min_deg")
    if rpm_candidates:
        out["cadence_rpm"] = round(float(np.median(rpm_candidates)), 1)
        out["labels"].append("cadence rpm from the knee-angle oscillation frequency (FFT)")

    # ---- torso angle vs horizontal (side view trunk lean)
    tl = s("trunk_lean")
    tl = tl[np.abs(tl) < 85]
    if np.sum(~np.isnan(tl)) > 10:
        torso = 90.0 - float(np.nanmean(np.abs(tl)))
        out["torso_deg"] = round(torso, 1)
        out["torso_note"] = ("trunk inclination vs horizontal (90 - trunk lean vs "
                             "vertical), measured mid-hip to mid-shoulder")
    el = [x for x in (s("left_elbow"), s("right_elbow")) if np.sum(~np.isnan(x)) > 5]
    if el:
        out["elbow_deg"] = round(float(np.mean([np.nanmean(x) for x in el])), 1)
    sh = [x for x in (s("left_shoulder"), s("right_shoulder")) if np.sum(~np.isnan(x)) > 5]
    if sh:
        out["shoulder_deg"] = round(float(np.mean([np.nanmean(x) for x in sh])), 1)

    # ---- targets for this bike type
    def band(key):
        return config.CYCLING_BANDS.get(key, {}).get(bike)

    def entry(key, value, label, band_key=None, unit="deg"):
        b = band(band_key or key)
        e = {"value": value, "band": list(b) if b else None, "label": label, "unit": unit}
        e["in_band"] = (bool(b[0] <= value <= b[1]) if (b and value is not None) else None)
        return e

    form = {}
    if out.get("knee_bdc_deg") is not None:
        form["bike_knee_bdc"] = entry(
            "knee_bdc_deg", out["knee_bdc_deg"],
            f"Knee angle at bottom of stroke (BDC), {bike.upper()}")
    if out.get("knee_tdc_min_deg") is not None:
        form["bike_knee_tdc"] = entry(
            "knee_tdc_deg", out["knee_tdc_min_deg"],
            "Knee angle at top of stroke (TDC, most flexed)")
    if out.get("hip_tdc_min_deg") is not None:
        form["bike_hip_tdc"] = entry(
            "hip_tdc_deg", out["hip_tdc_min_deg"],
            "Hip angle at top of stroke (closed position)")
    if out.get("torso_deg") is not None:
        form["bike_torso"] = entry("torso_deg", out["torso_deg"],
                                   "Torso angle vs horizontal")
    ank_vals = [d["ankle_mean_deg"] for d in per_side.values()
                if d.get("ankle_mean_deg") is not None]
    if ank_vals:
        form["bike_ankle"] = entry("ankle_deg", round(float(np.mean(ank_vals)), 1),
                                   "Ankle angle (mean)")
    if out.get("cadence_rpm") is not None:
        cb = config.CYCLING_CADENCE_BAND
        form["bike_cadence"] = {
            "value": out["cadence_rpm"], "band": list(cb),
            "in_band": bool(cb[0] <= out["cadence_rpm"] <= cb[1]),
            "label": "Cadence", "unit": "rpm",
        }
    out["form"] = form

    # ---- fit suggestions (direction of correction, classic reasoning)
    sug = []
    kb = out.get("knee_bdc_deg")
    ktgt = band("knee_bdc_deg")
    if kb is not None and ktgt:
        if kb > ktgt[1]:
            sug.append({
                "finding": f"Knee extension at BDC is {kb:.1f} deg — {kb - ktgt[1]:.0f} deg "
                           f"above the target {ktgt[0]:.0f}-{ktgt[1]:.0f} (saddle too high "
                           f"suspected).",
                "action": "Saddle is probably 3-8 mm too high: lower it 3-5 mm, re-film and "
                          "re-measure. Above ~150 deg the patellofemoral and hamstring load rises "
                          "and the hips start rocking to reach the pedal."})
        elif kb < ktgt[0]:
            sug.append({
                "finding": f"Knee flexion at BDC is {180 - kb:.1f} deg (included {kb:.1f}, target "
                           f"{ktgt[0]:.0f}-{ktgt[1]:.0f}).",
                "action": "Saddle is probably 3-8 mm too low: raise it 3-5 mm at a time and re-film. "
                          "Low saddles compress the knee at the top of the stroke and increase "
                          "anterior knee load."})
        else:
            sug.append({
                "finding": f"Knee angle at BDC {kb:.1f} deg is inside the classic target range.",
                "action": "Saddle height is consistent with the 25-35 deg flexion guideline from "
                          "this camera view — verify with the rider's feel and knee tracking "
                          "(rear camera) under load."})
    ht = out.get("hip_tdc_min_deg")
    htgt = band("hip_tdc_deg")
    if ht is not None and htgt and ht < htgt[0]:
        sug.append({
            "finding": f"Hip angle closes to {ht:.1f} deg at the top of the stroke "
                       f"(target >= {htgt[0]:.0f} for this discipline).",
            "action": "Very closed hip: consider shorter crank arms, raising the front end "
                      "slightly, or moving the saddle back 3-5 mm; watch breathing comfort "
                      "and power at high torque."})
    tr = out.get("torso_deg")
    ttgt = band("torso_deg")
    if tr is not None and ttgt and tr > ttgt[1] and bike in ("tt", "tri"):
        sug.append({
            "finding": f"Torso is {tr:.1f} deg above horizontal (typical aero range "
                       f"{ttgt[0]:.0f}-{ttgt[1]:.0f} for {bike.upper()}).",
            "action": "Position is not yet aero: lower the front end progressively (10 mm steps) "
                      "and re-check the hip angle at TDC each time; aero gains should not close "
                      "the hip below the target."})
    out["suggestions"] = sug
    return out


# ------------------------------------------------------- rear / frontal view
def analyze_rear(rows, params=None, prefix="r_"):
    """Rear (frontal-plane) control metrics from the second camera rows.

    Travel values are peak-to-peak robust ranges.  With a known shoulder
    width they are converted to millimetres; otherwise they are reported as
    a share of the rider's shoulder width (normalised, still comparable
    between sessions of the same rider/setup).
    """
    params = params or {}
    if not rows:
        return None
    t = _col(rows, "t_rel")
    ok_t = ~np.isnan(t)

    def s(key):
        return _col(rows, prefix + key)[ok_t]

    # is there any rear data at all?
    probes = [s("l_knee_x"), s("r_knee_x"), s("_midhip_y"), s("pelvic_tilt_deg")]
    if all(np.sum(~np.isnan(p)) < 10 for p in probes):
        return None

    # scale: shoulder width measured in this view
    lsh, rsh = s("l_sh_x"), s("r_sh_x")
    sh_norm = np.abs(lsh - rsh)
    sh_norm = sh_norm[~np.isnan(sh_norm)]
    sh_med = float(np.median(sh_norm)) if sh_norm.size > 10 else None
    shoulder_m = None
    height_cm = params.get("height_cm")
    shoulder_cm = params.get("shoulder_cm")
    if shoulder_cm:
        shoulder_m = float(shoulder_cm) / 100.0
    elif height_cm:
        shoulder_m = float(height_cm) / 100.0 * config.SHOULDER_WIDTH_HEIGHT_FRACTION
    aspect = float(params.get("aspect") or config.VIDEO_ASPECT_DEFAULT)
    m_per_norm_x = None
    if shoulder_m and sh_med and sh_med > 0.04:
        m_per_norm_x = shoulder_m / sh_med
    m_per_norm_y = m_per_norm_x / aspect if m_per_norm_x else None

    def travel(xy_key, axis):
        v = s(xy_key)
        r = _robust_range(v)
        if r is None:
            return None
        mm = None
        if axis == "x" and m_per_norm_x:
            mm = r * m_per_norm_x * 1000.0
        elif axis == "y" and m_per_norm_y:
            mm = r * m_per_norm_y * 1000.0
        return round(r, 4), (round(mm, 1) if mm is not None else None)

    out: dict = {"prefix": prefix, "labels": [
        "rear/front-view travel metrics: peak-to-peak robust ranges (5-95%); "
        "targets follow Velogic joint-motion guide values"]}
    if m_per_norm_x:
        out["scale_note"] = (
            f"mm scale from shoulder width ({shoulder_m * 100:.0f} cm) measured in-frame; "
            "keep the camera at the same distance for comparisons")
    else:
        out["scale_note"] = ("no shoulder width / height given — travels are reported as a "
                             "share of shoulder width (normalised)")

    def add(name, key, axis, label, target_key=None):
        r = travel(key, axis)
        if r is None:
            return
        norm, mm = r
        e = {"value_norm": norm, "value_mm": mm, "label": label}
        tgt = config.REAR_TRAVEL_TARGETS.get(target_key or name)
        if tgt:
            avg, good = tgt
            if mm is not None:
                e["target_mm_avg"] = avg
                e["target_mm_good"] = good
                e["within_avg"] = bool(mm <= avg)
                e["within_good"] = bool(mm <= good)
            else:
                e["target_norm_avg"] = round(avg / (shoulder_m * 1000.0), 3) if shoulder_m else None
                e["target_norm_good"] = round(good / (shoulder_m * 1000.0), 3) if shoulder_m else None
                if e["target_norm_avg"]:
                    e["within_avg"] = bool(norm <= e["target_norm_avg"])
                    e["within_good"] = bool(norm <= e["target_norm_good"])
        out[name] = e

    add("hip_vertical_travel", "_midhip_y", "y", "Hip vertical travel (rocking)")
    add("hip_horizontal_travel", "_midhip_x", "x", "Hip horizontal travel")
    add("knee_lateral_travel_left", "l_knee_x", "x", "Knee lateral travel — left")
    add("knee_lateral_travel_right", "r_knee_x", "x", "Knee lateral travel — right")
    add("shoulder_lateral_travel_left", "l_sh_x", "x", "Shoulder lateral travel — left")
    add("shoulder_lateral_travel_right", "r_sh_x", "x", "Shoulder lateral travel — right")
    add("ankle_swivel_left", "l_ankle_x", "x", "Ankle swivel — left")
    add("ankle_swivel_right", "r_ankle_x", "x", "Ankle swivel — right")

    # pelvic obliquity = peak-to-peak of the inter-hip tilt (deg)
    pt = s("pelvic_tilt_deg")
    r = _robust_range(pt)
    if r is not None:
        out["pelvic_obliquity_deg"] = round(r, 1)
        out["pelvic_obliquity_note"] = ("peak-to-peak tilt of the inter-hip line; "
                                        ">5 deg flags hip rocking")

    # frontal knee tracking angle vs vertical (hip->knee line), deg p2p
    for side in SIDES:
        sh = SIDE_SHORT[side]
        kx = s(f"{sh}_knee_x")
        ky = s(f"{sh}_knee_y")
        hx = s(f"{sh}_hip_x")
        hy = s(f"{sh}_hip_y")
        ok = ~np.isnan(kx) & ~np.isnan(ky) & ~np.isnan(hx) & ~np.isnan(hy)
        if ok.sum() < 10:
            continue
        dx = kx[ok] - hx[ok]
        dy = np.maximum(1e-6, hy[ok] - ky[ok])       # downward positive
        ang = np.degrees(np.arctan2(dx, dy))
        r = _robust_range(ang)
        if r is not None:
            out[f"knee_travel_angle_{side}"] = {
                "value_deg": round(r, 1),
                "target_deg_avg": 8.0, "target_deg_good": 2.0,
                "within_avg": bool(r <= 8.0), "within_good": bool(r <= 2.0),
                "label": f"Knee travel angle vs vertical — {side}",
            }
    return out


# ------------------------------------------------------------------- aero --
def frontal_area(rows, params=None, prefix=""):
    """Projected frontal area from the segmentation-mask fraction series.

    ``fa_frac`` (mask share of the frame) is recorded per frame by the pose
    engine when segmentation is enabled.  With a scale (shoulder width or
    height) the fraction is converted to m^2 for the assumed frame size.
    """
    params = params or {}
    if not rows:
        return None
    v = _col(rows, prefix + "fa_frac")
    v = v[~np.isnan(v)]
    if v.size < 20:
        return None
    med = float(np.median(v))
    out = {
        "fa_frac_median": round(med, 4),
        "fa_frac_mean": round(float(np.mean(v)), 4),
        "n": int(v.size),
        "labels": ["frontal area = MediaPipe segmentation-mask share of the frame "
                   "(front view); scale from shoulder width / height"],
        "assumptions": ("absolute value is an estimate (perspective, clothing); compare "
                        "positions only on the same camera distance and lens"),
    }
    lsh = _col(rows, prefix + "l_sh_x")
    rsh = _col(rows, prefix + "r_sh_x")
    d = np.abs(lsh - rsh)
    d = d[~np.isnan(d)]
    sh_med = float(np.median(d)) if d.size > 10 else None
    shoulder_m = None
    if params.get("shoulder_cm"):
        shoulder_m = float(params["shoulder_cm"]) / 100.0
    elif params.get("height_cm"):
        shoulder_m = float(params["height_cm"]) / 100.0 * config.SHOULDER_WIDTH_HEIGHT_FRACTION
    if shoulder_m and sh_med and sh_med > 0.04:
        aspect = float(params.get("aspect") or config.VIDEO_ASPECT_DEFAULT)
        frame_w_m = shoulder_m / sh_med
        frame_h_m = frame_w_m / aspect
        out["frame_width_m_est"] = round(frame_w_m, 2)
        out["fa_m2_median"] = round(med * frame_w_m * frame_h_m, 3)
        out["fa_m2_mean"] = round(float(np.mean(v)) * frame_w_m * frame_h_m, 3)
    else:
        out["scale_note"] = ("no shoulder width / height — FA reported as the mask "
                             "share of the frame only")
    return out


def aero_delta(fa_base_m2, fa_new_m2, speed_kmh=None, cd=None):
    """Estimated effect of a frontal-area change at a given road speed.

    P_drag = 0.5 * rho * CdA * v^3.  With Cd held constant, dP = 0.5 rho Cd
    dA v^3 — clearly an ESTIMATE (Cd is assumed, and the video measures area,
    not real flow).
    """
    if not fa_base_m2 or not fa_new_m2:
        return None
    dA = float(fa_new_m2) - float(fa_base_m2)
    pct = 100.0 * dA / float(fa_base_m2)
    out = {
        "delta_area_m2": round(dA, 3),
        "delta_area_pct": round(pct, 1),
        "direction": "smaller area (faster)" if dA < 0 else "larger area (slower)",
    }
    if speed_kmh:
        v = float(speed_kmh) / 3.6
        c = float(cd if cd else config.AERO_CD_DEFAULT)
        watts = 0.5 * config.AIR_DENSITY_KGM3 * c * dA * (v ** 3)
        out.update({
            "speed_kmh": float(speed_kmh), "cd_assumed": c,
            "watts_delta_est": round(watts, 1),
            "watts_note": ("estimated change in aero drag power at this speed, assuming "
                           f"Cd = {c:g} and unchanged position aerodynamics beyond area"),
        })
    return out


# ------------------------------------------------------------- keyframe latch
class CycleLatch:
    """Latches BDC / TDC keyframes from the knee-angle waveform (live push).

    Mirrors the long-runner keyframe behaviour of gait_events.GaitLatch
    (clean jpeg + landmarks + metrics per event; HD refinement + annotated
    snapshots at session end) but for the pedalling cycle: one 'BDC' event per
    revolution per side (most extended knee — the fit reference position) and
    one 'TDC' event (most flexed).
    """

    def __init__(self, per_type=None, max_total=None):
        self.per_type = int(per_type or 2)
        self.max_total = int(max_total or config.KEYFRAMES_MAX)
        self.latched: list = []
        self._i = 0
        self._win: dict = {s: [] for s in SIDES}      # recent (i, t, knee) per side
        self._last: dict = {}
        self._buf: dict = {}                          # frame i -> {jpeg, lms, fm}
        self._revolutions = {s: 0 for s in SIDES}

    def push(self, rel, fm, jpeg, lms):
        i = self._i
        self._i += 1
        if jpeg is not None and lms is not None:
            self._buf[i] = {"jpeg": jpeg, "lms": lms, "fm": dict(fm or {})}
            if len(self._buf) > 24:
                for k in sorted(self._buf)[:-16]:
                    self._buf.pop(k, None)
        new = []
        for side in SIDES:
            knee = fm.get(KNEE_KEY[side])
            if knee is None:
                self._win[side].append((i, rel, None))
            else:
                self._win[side].append((i, rel, float(knee)))
                w = self._win[side]
                if len(w) > 5:
                    del w[:-5]
                # middle frame = lookahead of 1 completed frame
                if len(w) == 5:
                    j = w[2][0]
                    kj = w[2][2]
                    if kj is not None:
                        lo_prev, lo_next = w[1][2], w[3][2]
                        typ = None
                        if lo_prev is not None and lo_next is not None:
                            if kj >= lo_prev and kj > lo_next and kj > 95:
                                typ = "BDC"
                            elif kj <= lo_prev and kj < lo_next and kj < 105:
                                typ = "TDC"
                        if typ:
                            key = (typ, side)
                            last = self._last.get(key)
                            if last is None or rel - last >= 0.45:
                                self._last[key] = rel
                                b = self._buf.get(j) or {}
                                ev = {"type": typ, "side": side, "t": float(w[2][1]),
                                      "i": j, "frame": b.get("jpeg"),
                                      "lms": b.get("lms"), "fm": dict(b.get("fm") or {}),
                                      "knee": kj, "quality": self._quality(b, kj)}
                                self.latched.append(ev)
                                new.append(ev)
                                if typ == "BDC":
                                    self._revolutions[side] += 1
        return new

    @staticmethod
    def _quality(b, knee) -> float:
        fm = b.get("fm") or {}
        n = 0
        for k in ("left_knee", "right_knee", "left_hip", "right_hip", "left_ankle"):
            if fm.get(k) is not None:
                n += 1
        vis = fm.get("_mean_vis")
        try:
            vis = float(vis)
        except (TypeError, ValueError):
            vis = 0.0
        if not math.isfinite(vis):
            vis = 0.0
        return round(n / 5.0 + max(0.0, min(1.0, vis)) + min(0.5, knee / 300.0), 3)

    def revolutions(self) -> dict:
        return dict(self._revolutions)

    def events(self) -> list:
        return list(self.latched)

    def best_events(self) -> list:
        out, by = [], {}
        for ev in sorted(self.latched, key=lambda e: -float(e.get("quality") or 0.0)):
            k = (ev.get("type"), ev.get("side"))
            if by.get(k, 0) >= self.per_type:
                continue
            by[k] = by.get(k, 0) + 1
            out.append(ev)
        out.sort(key=lambda e: (e.get("t") or 0.0))
        return out[: self.max_total]

    def save_snapshots(self, out_dir, refine=None) -> list:
        snaps = self.best_events()
        if not snaps:
            return []
        return GA.save_events(snaps, out_dir, refine=refine)


# ------------------------------------------------------------------ enrich --
def enrich(summary: dict, rows, params=None, bike_type=None, discipline=None) -> dict:
    """Attach bike / rear-view / aero blocks to a session summary.

    Called from the app finalize step and the offline CLI right after
    metrics.analyze_session, so live and file sessions produce identical data.
    """
    if not isinstance(summary, dict) or not rows:
        return summary
    params = params or {}
    disc = str(discipline or summary.get("discipline") or "running").lower()
    summary["discipline"] = disc
    if disc == "cycling":
        bike = bike_type or summary.get("bike_type") or config.DEFAULT_BIKE_TYPE
        summary["bike_type"] = bike if bike in config.BIKE_TYPES else config.DEFAULT_BIKE_TYPE
        summary["bike"] = analyze_cycling(rows, summary["bike_type"], params)
    rear = analyze_rear(rows, params)
    if rear:
        summary["rear"] = rear
    aero = frontal_area(rows, params, prefix="")
    if not aero:
        aero = frontal_area(rows, params, prefix="r_")
        if aero:
            aero["camera"] = "second camera"
    if aero:
        speed = params.get("speed_kmh")
        aero["speed_kmh"] = float(speed) if speed else None
        cd = params.get("cd_est")
        aero["cd_assumed"] = float(cd) if cd else config.AERO_CD_DEFAULT
        summary["aero"] = aero
    return summary
