"""Form scoring, injury-risk stratification and session comparison for RunTrack.

Everything here is rule-based and documented in code + README, because the
numbers are used for coaching decisions:

  form_score(summary)          -> 0-100 composite (weighted, plane aware)
  risk_stratification(summary) -> Low / Moderate / Elevated / High + PT drills
  compare(a, b)                -> baseline vs follow-up table with delta badges

Honest evidence note (also shown in the report): these are screening heuristics
derived from running-gait literature and common clinical reasoning for
recreational runners.  They are NOT a diagnosis and do not replace a
physiotherapist's assessment.
"""
from __future__ import annotations

import math

import config

# ---------------------------------------------------------------------------
# Overall form score
#
# Each available component is scored 0-100 against its reference band
# (100 inside the band, decaying linearly to 0 at `tol` outside it).  The
# component scores are combined with the weights below.  Weights are
# re-normalised over the components that could actually be measured, so a
# session without a scale (no height) or a frontal-view session is still scored
# on what was really measured — never on filled-in guesses.
# ---------------------------------------------------------------------------
SAGITTAL_WEIGHTS = {
    "symmetry_pct": 0.22,
    "cadence_spm": 0.18,
    "touchdown_shin_deg": 0.15,
    "knee_flexion_ms_deg": 0.13,
    "overstride_leg_frac": 0.12,
    "trunk_lean_deg": 0.10,
    "vertical_osc_cm": 0.10,
}
FRONTAL_WEIGHTS = {
    "symmetry_pct": 0.30,
    "knee_valgus_deg": 0.20,
    "pelvic_drop_deg": 0.20,
    "cadence_spm": 0.20,
    "vertical_osc_cm": 0.10,
}
# component -> (band key, tolerance used for the linear decay, label, unit)
COMPONENTS = {
    "cadence_spm": ("cadence_spm", 40.0, "Cadence", "spm"),
    "symmetry_pct": ("symmetry_pct", 15.0, "L/R symmetry", "%"),
    "vertical_osc_cm": ("vertical_osc_cm", 6.0, "Vertical oscillation", "cm"),
    "touchdown_shin_deg": ("touchdown_shin_deg", 12.0, "Touchdown shin angle", "deg"),
    "overstride_leg_frac": ("overstride_leg_frac", 0.25, "Overstride", "leg lengths"),
    "knee_flexion_ms_deg": ("knee_flexion_ms_deg", 25.0, "Mid-stance knee flexion", "deg"),
    "trunk_lean_deg": ("trunk_lean_deg", 10.0, "Anterior trunk lean", "deg"),
    "pelvic_drop_deg": ("pelvic_drop_deg", 8.0, "Contralateral pelvic drop", "deg"),
    "knee_valgus_deg": ("knee_valgus_deg", 12.0, "Frontal knee collapse", "deg"),
    "arm_swing_sym_pct": ("arm_swing_sym_pct", 20.0, "Arm-swing symmetry", "%"),
}

EVIDENCE_CAVEAT = (
    "Screening aid only: the bands, the form score and the risk level come from "
    "running-gait literature and common physiotherapy reasoning for recreational "
    "runners. They are not a diagnosis, were not validated on this camera setup, "
    "and must not replace an in-person assessment by a clinician or coach."
)

EVIDENCE_CAVEAT_CYCLING = (
    "Fit guidance only: the bike-position bands are industry video-fit ranges "
    "(Velogic Studio metric guides, the Holmes method 25-35 deg knee flexion at "
    "BDC, tri/TT position conventions) measured from 2D video landmarks. They "
    "are not a medical assessment, were not validated on this camera setup, and "
    "must not replace an in-person assessment by a clinician or bike fitter."
)


def scalar(v):
    """Unwrap a metric that may be a plain number or a stats dict."""
    if isinstance(v, dict):
        for k in ("mean", "value", "mean_ahead_leg_frac", "mean_signed"):
            if v.get(k) is not None:
                return v[k]
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def metric(summary: dict, key: str):
    """Value of a summary metric, mapping a few aliases to their stored form."""
    if not summary:
        return None
    src = {
        "trunk_lean_deg": "trunk_lean_fwd_deg",
        "symmetry_pct": "symmetry_mean_pct",
    }.get(key, key)
    v = scalar(summary.get(src))
    if v is None and key in ("knee_flexion_ms_deg", "pelvic_drop_deg"):
        v = scalar(summary.get(key))
    if v is None and key == "touchdown_shin_deg":
        v = scalar(summary.get("touchdown_shin_deg"))
    if v is None and key == "overstride_leg_frac":
        v = scalar(summary.get("overstride_leg_frac"))
    return v


def not_assessed(summary: dict, key: str):
    na = (summary or {}).get("not_assessed") or {}
    return key in na and scalar(summary.get(key)) is None


def band_score(value, lo, hi, tol) -> float:
    """100 inside [lo, hi], 0 at `tol` beyond the nearer edge, linear between."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if lo <= v <= hi:
        return 100.0
    d = (lo - v) if v < lo else (v - hi)
    tol = float(tol) if tol else max(1e-6, 0.5 * (hi - lo))
    return max(0.0, min(100.0, 100.0 * (1.0 - d / max(tol, 1e-6))))


def _cycling_form_score(summary: dict) -> dict:
    """Composite score for a cycling session from the bike form entries."""
    bike = summary.get("bike") or {}
    bform = bike.get("form") or {}
    comps, acc, total_w = [], 0.0, 0.0
    for key, w in CYCLING_WEIGHTS.items():
        if key == "symmetry_pct":
            val = metric(summary, "symmetry_pct")
            band = config.BANDS.get("symmetry_pct")
            if val is None or not band:
                comps.append({"metric": key, "label": "L/R symmetry", "value": val,
                              "band": band, "score": None, "weight": w,
                              "status": "not tracked"})
                continue
            s = band_score(val, band[0], band[1], 15.0)
            comps.append({"metric": key, "label": "L/R symmetry", "value": round(float(val), 1),
                          "band": list(band), "unit": "%", "score": round(s, 1),
                          "weight": w, "status": "scored"})
            acc += s * w
            total_w += w
            continue
        e = bform.get(key)
        if not e or e.get("value") is None:
            comps.append({"metric": key, "label": key.replace("bike_", "").replace("_", " "),
                          "value": None, "band": None, "score": None, "weight": w,
                          "status": "not tracked"})
            continue
        band = e.get("band")
        tol = (0.25 * (band[1] - band[0]) + 2.0) if band else 10.0
        s = band_score(e["value"], band[0], band[1], tol) if band else (
            100.0 if e.get("in_band") else 0.0)
        comps.append({"metric": key, "label": e.get("label") or key,
                      "value": round(float(e["value"]), 1), "band": band,
                      "unit": e.get("unit", "deg"), "score": round(s, 1),
                      "weight": w, "status": "scored"})
        acc += s * w
        total_w += w
    score = round(acc / total_w, 0) if total_w > 0 else None
    return {
        "score": score, "components": comps, "weights": CYCLING_WEIGHTS,
        "coverage": round(total_w, 3),
        "method": ("cycling position score: knee/hip/torso angles against the "
                   "industry fit bands for this bike type, plus symmetry; weights "
                   "re-normalised over the components that were measured"),
        "plane": str(summary.get("view_plane") or "sagittal"),
        "discipline": "cycling",
        "bike_type": summary.get("bike_type"),
    }


def form_score(summary: dict) -> dict:
    """0-100 composite form score for a session summary (plane aware)."""
    if not summary:
        return {"score": None, "components": [], "method": "no data"}
    if str(summary.get("discipline") or "").lower() == "cycling":
        return _cycling_form_score(summary)
    plane = str(summary.get("view_plane") or config.DEFAULT_VIEW_PLANE).lower()
    weights = FRONTAL_WEIGHTS if plane == "frontal" else SAGITTAL_WEIGHTS
    comps, total_w, acc = [], 0.0, 0.0
    for key, w in weights.items():
        if not_assessed(summary, key):
            comps.append({"metric": key, "label": COMPONENTS[key][2], "value": None,
                          "band": None, "score": None, "weight": w,
                          "status": config.NOT_ASSESSED_TEXT})
            continue
        val = metric(summary, key)
        band_key, tol, label, unit = COMPONENTS[key]
        band = config.BANDS.get(band_key)
        if val is None or not band:
            comps.append({"metric": key, "label": label, "value": None, "band": band,
                          "score": None, "weight": w, "status": "not tracked"})
            continue
        s = band_score(val, band[0], band[1], tol)
        comps.append({"metric": key, "label": label, "value": round(float(val), 2),
                      "band": list(band), "unit": unit, "score": round(s, 1),
                      "weight": w, "status": "scored"})
        acc += s * w
        total_w += w
    score = round(acc / total_w, 0) if total_w > 0 else None
    return {
        "score": score,
        "components": comps,
        "weights": weights,
        "coverage": round(total_w, 3),
        "method": ("weighted reference-band score: each component scores 100 inside "
                   "its band and decays linearly to 0 at the component tolerance; "
                   "weights are re-normalised over the components that were measured"),
        "plane": plane,
    }


# ---------------------------------------------------------------------------
# Cycling form score (used when summary["discipline"] == "cycling")
#
# Components come from summary["bike"]["form"] (computed by cycling.py) and
# carry their own per-bike-type bands; the symmetry component is shared with
# the running scoring.  Weights are re-normalised over what was measured.
# ---------------------------------------------------------------------------
CYCLING_WEIGHTS = {
    "bike_knee_bdc": 0.30,
    "bike_torso": 0.20,
    "bike_hip_tdc": 0.15,
    "bike_cadence": 0.15,
    "symmetry_pct": 0.20,
}

CYCLING_DRILLS = {
    "knee_bdc_high": [
        "Lower the saddle 3-5 mm, re-film at the same camera position, re-measure "
        "the knee angle at the bottom of the stroke",
        "Check for hip rocking in the rear-view recording at the same effort — "
        "reaching for the pedal is a classic too-high-saddle sign",
    ],
    "knee_bdc_low": [
        "Raise the saddle 3-5 mm at a time; low saddles compress the knee at the "
        "top of the stroke",
        "If the target angle still needs more height, check crank length and shoe "
        "stack before raising the saddle beyond the seatpost limit",
    ],
    "hip_tdc_low": [
        "Open the hip: consider shorter cranks, raise the front end 10 mm, or move "
        "the saddle back 3-5 mm (re-check reach afterwards)",
        "Hip flexor mobility work 3 x 30 s per side before sessions",
    ],
    "obliquity": [
        "Side-lying hip abduction 3 x 12 per side, single-leg bridge 3 x 10",
        "Re-check saddle height and cleat shims; a persistent one-sided drop is "
        "worth a leg-length check with the fitter",
    ],
    "knee_track": [
        "Band-resisted lateral walks 3 x 10 steps each way",
        "Step-down (20-30 cm) with knee tracking over the second toe, 3 x 8/side",
        "Check cleat stance/Q-factor with the fitter if the knee keeps drifting",
    ],
    "cadence": [
        "Cadence drills: 3 x 3 min at +5-10 rpm over self-selected, keep the pelvis "
        "quiet in the rear-view camera",
    ],
}

CYCLING_RISK_RULES = [
    {"id": "knee_bdc_high", "path": ("bike", "knee_bdc_deg"), "op": ">", "thr": 152.0,
     "points": 2,
     "finding": "Knee extension at the bottom of the stroke {v:.1f} deg (>152): the "
                "saddle is likely too high — patellofemoral and posterior-chain load "
                "rise, and hips typically start rocking to reach the pedal."},
    {"id": "knee_bdc_low", "path": ("bike", "knee_bdc_deg"), "op": "<", "thr": 135.0,
     "points": 2,
     "finding": "Knee flexion at the bottom of the stroke is {v:.1f} deg included "
                "(<135): the saddle is likely too low — high anterior knee load at "
                "the top of the stroke."},
    {"id": "hip_tdc_low", "path": ("bike", "hip_tdc_min_deg"), "op": "<", "thr": 42.0,
     "points": 1,
     "finding": "Hip angle closes to {v:.1f} deg at the top of the stroke (<42): a "
                "very compressed position; consider shorter cranks or a higher "
                "front end."},
    {"id": "obliquity", "path": ("rear", "pelvic_obliquity_deg"), "op": ">", "thr": 5.0,
     "points": 2,
     "finding": "Pelvic obliquity (hip rocking) peak-to-peak {v:.1f} deg in the "
                "rear view (>5): control deficit or fit asymmetry loading the "
                "lumbar spine and hips."},
    {"id": "cadence", "path": ("bike", "cadence_rpm"), "op": "<",
     "thr": config.CYCLING_CADENCE_HARD[0], "points": 1,
     "finding": "Cadence {v:.0f} rpm (<{thr:.0f}) at the recorded effort: very low "
                "cadence raises knee torque; check gearing for the terrain."},
    {"id": "cadence", "path": ("bike", "cadence_rpm"), "op": ">",
     "thr": config.CYCLING_CADENCE_HARD[1], "points": 1,
     "finding": "Cadence {v:.0f} rpm (>{thr:.0f}): unusually high — check the "
                "measurement effort and gearing."},
    {"id": "asymmetry", "key": "symmetry_pct", "op": "<", "thr": 90.0, "points": 1,
     "finding": "Left/right symmetry {v:.1f} % (<90): a persistent side-to-side "
                "difference is worth investigating."},
]


def _cyc_val(summary, path):
    node = summary
    for part in path:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return scalar(node) if node is not None else None


# ---------------------------------------------------------------------------
# Injury risk stratification (rule based) + PT drill prescriptions
# ---------------------------------------------------------------------------
RISK_LEVELS = ((0, "Low"), (2, "Moderate"), (4, "Elevated"), (6, "High"))

PT_DRILLS = {
    "braking": [
        "Metronome cadence work at 175-180 spm, 4 x 60 s (shortens the braking "
        "stride and pulls the foot closer under the hips)",
        "Wall drill / A-skip: 3 x 20 m focusing on a quiet, vertical shin at contact",
    ],
    "overstride": [
        "Lean-and-fall drill from standing: 3 x 8 reps, land under the hips",
        "Downhill-free cadence ladder: 4 x 30 s at +5 spm over your comfortable rate",
    ],
    "knee_absorption": [
        "Split squat isometric hold 3 x 30 s per leg building knee flexor "
        "absorption capacity",
        "Soft, quiet landing drills (drop from a 20 cm box, 3 x 8, land with a "
        "bent, silent knee)",
    ],
    "pelvic_drop": [
        "Side-lying hip abduction 3 x 12 per side, slow eccentric",
        "Single-leg bridge 3 x 10 per side with a level pelvis",
        "Standing hip-hike on a step 3 x 12 per side (front leg stays straight)",
    ],
    "valgus": [
        "Band-resisted lateral walks 3 x 10 steps each way",
        "Single-leg squat to a target with the knee tracking over the second toe, "
        "3 x 8 per side",
        "Step-down (20-30 cm) with frontal-plane control, 3 x 8 per side",
    ],
    "low_cadence": [
        "Metronome sessions: raise cadence by 5 % only, 2-3 runs per week",
        "Short, quick ground-contact cues: 'light, fast, quiet feet'",
    ],
    "oscillation": [
        "Skip-and-run drills 4 x 20 m to shorten ground contact",
        "Run tall with a slight forward lean; avoid reaching with the toes",
    ],
}

RISK_RULES = [
    {"id": "braking", "key": "touchdown_shin_deg", "op": ">", "thr": 8.0, "points": 2,
     "plane": "sagittal",
     "finding": "Touchdown shin angle {v:.1f} deg (>8): the shin lands inclined "
                "forward, which increases braking force through the knee and hip."},
    {"id": "overstride", "key": "overstride_leg_frac", "op": ">", "thr": 0.15, "points": 2,
     "plane": "sagittal",
     "finding": "Overstride vector {v:.2f} leg lengths ahead of the pelvis at "
                "contact (>0.15): a longer braking phase and higher impact load."},
    {"id": "knee_absorption", "key": "knee_flexion_ms_deg", "op": "<", "thr": 35.0,
     "points": 1, "plane": "sagittal",
     "finding": "Mid-stance knee flexion {v:.1f} deg (<35): stiffer absorption "
                "phase, load moved to the joint instead of the muscles."},
    {"id": "knee_absorption", "key": "knee_flexion_ms_deg", "op": ">", "thr": 45.0,
     "points": 1, "plane": "sagittal",
     "finding": "Mid-stance knee flexion {v:.1f} deg (>45): deep collapse under "
                "load, often paired with hip-control deficits."},
    {"id": "pelvic_drop", "key": "pelvic_drop_deg", "op": ">", "thr": 5.0, "points": 2,
     "plane": "frontal",
     "finding": "Contralateral pelvic drop {v:.1f} deg (>4-5): potential "
                "gluteus-medius weakness and control deficit in single-leg stance."},
    {"id": "valgus", "key": "knee_valgus_deg", "op": ">", "thr": 12.0, "points": 2,
     "plane": "frontal",
     "finding": "Frontal knee collapse {v:.1f} deg (>12): medial knee motion "
                "relative to the hip-ankle axis, a primary patellofemoral / "
                "IT-band risk marker."},
    {"id": "low_cadence", "key": "cadence_spm", "op": "<", "thr": 160.0, "points": 1,
     "plane": None,
     "finding": "Cadence {v:.0f} spm (<160): longer stride time, higher impact "
                "per step and more vertical braking."},
    {"id": "oscillation", "key": "vertical_osc_cm", "op": ">", "thr": 10.0, "points": 1,
     "plane": "sagittal",
     "finding": "Vertical oscillation {v:.1f} cm (>10): extra vertical work is "
                "not converted into forward motion."},
    {"id": "asymmetry", "key": "symmetry_pct", "op": "<", "thr": 90.0, "points": 1,
     "plane": None,
     "finding": "Left/right symmetry {v:.1f} % (<90): a persistent side-to-side "
                "difference is worth investigating."},
]


def _cmp(v, op, thr) -> bool:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return v > thr if op == ">" else v < thr


def risk_level(points: int) -> str:
    level = RISK_LEVELS[0][1]
    for thr, name in RISK_LEVELS:
        if points >= thr:
            level = name
    return level


def risk_stratification(summary: dict) -> dict:
    """Rule-based injury-risk stratification with PT drill prescriptions."""
    flags = []
    if not summary:
        return {"level": "Low", "points": 0, "flags": [], "drills": [],
                "caveat": EVIDENCE_CAVEAT}
    if str(summary.get("discipline") or "").lower() == "cycling":
        return _cycling_risk(summary)
    plane = str(summary.get("view_plane") or config.DEFAULT_VIEW_PLANE).lower()
    points = 0
    for rule in RISK_RULES:
        if rule["plane"] and rule["plane"] != plane:
            continue                       # wrong camera view: never flag it
        val = metric(summary, rule["key"])
        if val is None:
            continue
        if _cmp(val, rule["op"], rule["thr"]):
            points += int(rule["points"])
            flags.append({
                "id": rule["id"],
                "metric": rule["key"],
                "value": round(float(val), 2),
                "finding": rule["finding"].format(v=float(val)),
                "points": rule["points"],
            })
    drills = []
    seen = set()
    for f in flags:
        for d in PT_DRILLS.get(f["id"], []):
            if d not in seen:
                seen.add(d)
                drills.append({"drill": d, "for": f["id"]})
    return {
        "level": risk_level(points),
        "points": points,
        "flags": flags,
        "drills": drills,
        "caveat": EVIDENCE_CAVEAT,
        "plane": plane,
        "rules_version": "1.0",
    }


# ---------------------------------------------------------------------------
# Baseline vs follow-up comparison
# ---------------------------------------------------------------------------
# direction: "up" = higher is better, "down" = lower is better, "band" = closer
# to the reference band is better.  eps = neutral zone (|delta| <= eps => Neutral).
COMPARE_METRICS = [
    {"key": "form_score", "label": "Form score", "unit": "/100", "direction": "up",
     "eps": 3.0, "digits": 0},
    {"key": "cadence_spm", "label": "Cadence", "unit": "spm", "direction": "up",
     "eps": 2.0, "digits": 1},
    {"key": "symmetry_pct", "label": "L/R symmetry", "unit": "%", "direction": "up",
     "eps": 1.5, "digits": 1},
    {"key": "knee_flexion_ms_deg", "label": "Mid-stance knee flexion", "unit": "deg",
     "direction": "band", "band": "knee_flexion_ms_deg", "eps": 2.0, "digits": 1},
    {"key": "vertical_osc_cm", "label": "Vertical oscillation", "unit": "cm",
     "direction": "down", "eps": 0.6, "digits": 1},
    {"key": "touchdown_shin_deg", "label": "Touchdown shin angle", "unit": "deg",
     "direction": "down", "eps": 1.0, "digits": 1},
    {"key": "overstride_leg_frac", "label": "Overstride", "unit": "leg lengths",
     "direction": "down", "eps": 0.02, "digits": 3},
    {"key": "pelvic_drop_deg", "label": "Contralateral pelvic drop", "unit": "deg",
     "direction": "down", "eps": 0.8, "digits": 1},
    {"key": "knee_valgus_deg", "label": "Frontal knee collapse", "unit": "deg",
     "direction": "down", "eps": 1.0, "digits": 1},
    {"key": "power_per_kg_watts", "label": "Power per kg", "unit": "W/kg",
     "direction": "band", "band": "power_per_kg_w", "eps": 0.05, "digits": 2},
]


def _value_for(summary, spec):
    """Value of a compare metric (handles the composite form score)."""
    if spec["key"] == "form_score":
        fs = form_score(summary or {})
        return fs.get("score")
    return metric(summary or {}, spec["key"])


def _verdict(direction, base, follow, spec) -> str:
    """Improved / Neutral / Regressed for one metric."""
    if base is None or follow is None:
        return "n/a"
    delta = float(follow) - float(base)
    eps = float(spec.get("eps", 0.0))
    if abs(delta) <= eps:
        return "Neutral"
    if direction == "up":
        return "Improved" if delta > 0 else "Regressed"
    if direction == "down":
        return "Improved" if delta < 0 else "Regressed"
    # toward the reference band
    band = config.BANDS.get(spec.get("band")) or spec.get("band_values")
    if not band:
        return "Neutral"
    lo, hi = band

    def dist(v):
        v = float(v)
        if v < lo:
            return lo - v
        if v > hi:
            return v - hi
        return 0.0

    return "Improved" if dist(follow) < dist(base) else "Regressed"


def _cycling_risk(summary: dict) -> dict:
    """Risk stratification for a cycling session (bike-fit + rear-view rules)."""
    points = 0
    flags = []
    for rule in CYCLING_RISK_RULES:
        val = (_cyc_val(summary, rule["path"]) if rule.get("path")
               else metric(summary, rule["key"]))
        if val is None:
            continue
        if _cmp(val, rule["op"], rule["thr"]):
            points += int(rule["points"])
            try:
                finding = rule["finding"].format(v=float(val), thr=float(rule["thr"]))
            except Exception:
                finding = rule["finding"]
            flags.append({"id": rule["id"], "metric": rule.get("key") or ".".join(rule["path"]),
                          "value": round(float(val), 2), "finding": finding,
                          "points": rule["points"]})
    # rear-view lateral knee travel (flagged by the rear targets, mm or normalised)
    rear = summary.get("rear") or {}
    for side in ("left", "right"):
        e = rear.get(f"knee_lateral_travel_{side}") or {}
        if e and e.get("within_avg") is False:
            points += 1
            flags.append({
                "id": "knee_track", "metric": f"knee_lateral_travel_{side}",
                "value": e.get("value_mm") if e.get("value_mm") is not None else e.get("value_norm"),
                "finding": (f"Lateral knee travel on the {side} side is above the "
                            "fit-tool guide value in the rear view: patellofemoral "
                            "tracking is worth checking (cleat stance, Q-factor, "
                            "lateral hip control)."),
                "points": 1})
            break
    drills = []
    seen = set()
    for f in flags:
        for d in CYCLING_DRILLS.get(f["id"], []):
            if d not in seen:
                seen.add(d)
                drills.append({"drill": d, "for": f["id"]})
    return {
        "level": risk_level(points),
        "points": points,
        "flags": flags,
        "drills": drills,
        "caveat": EVIDENCE_CAVEAT_CYCLING,
        "discipline": "cycling",
        "rules_version": "1.0",
    }


def enrich(summary: dict) -> dict:
    """Attach the composite score + risk stratification to a session summary.

    Called once when a session is finalized (app + CLI), so that the values are
    stored in summary.json and every consumer (list, history, report, compare)
    reads the same numbers instead of recomputing them.
    """
    if not isinstance(summary, dict):
        return summary
    fs = form_score(summary)
    rs = risk_stratification(summary)
    summary["form_score"] = fs.get("score")
    summary["form_score_detail"] = fs
    summary["risk"] = rs
    return summary


def _aero_watts(fa_base, fa_new, speed_kmh, cd) -> float | None:
    """dP = 0.5 rho Cd dA v^3 (estimate; see cycling.aero_delta)."""
    try:
        import cycling as CY
        d = CY.aero_delta(fa_base, fa_new, speed_kmh=speed_kmh, cd=cd)
        return None if not d else d.get("watts_delta_est")
    except Exception:
        return None


def compare(a_summary: dict, b_summary: dict, a_meta=None, b_meta=None) -> dict:
    """Baseline (a) vs follow-up (b) comparison table with colour-coded badges."""
    rows = []
    for spec in COMPARE_METRICS:
        if not_assessed(a_summary or {}, spec["key"]) and \
                not_assessed(b_summary or {}, spec["key"]):
            continue
        base = _value_for(a_summary, spec)
        follow = _value_for(b_summary, spec)
        verdict = _verdict(spec["direction"], base, follow, spec)
        row = {
            "metric": spec["key"],
            "label": spec["label"],
            "unit": spec["unit"],
            "direction": spec["direction"],
            "baseline": None if base is None else round(float(base), spec["digits"]),
            "followup": None if follow is None else round(float(follow), spec["digits"]),
            "delta": (None if (base is None or follow is None)
                      else round(float(follow) - float(base), spec["digits"])),
            "verdict": verdict,
        }
        if verdict == "n/a":
            row["note"] = config.NOT_ASSESSED_TEXT
        rows.append(row)
    improved = sum(1 for r in rows if r["verdict"] == "Improved")
    regressed = sum(1 for r in rows if r["verdict"] == "Regressed")
    neutral = sum(1 for r in rows if r["verdict"] == "Neutral")

    # ---------------------------------------------------------------- cycling
    # Position rows read from summary["bike"]; the frontal-area pair is the
    # aero A/B comparison (same camera setup assumed between sessions).
    ab = (a_summary or {}).get("bike") or {}
    bb = (b_summary or {}).get("bike") or {}
    if ab and bb:
        btype = ((b_meta or {}).get("bike_type") or (a_meta or {}).get("bike_type")
                 or config.DEFAULT_BIKE_TYPE)
        if btype not in config.BIKE_TYPES:
            btype = config.DEFAULT_BIKE_TYPE

        def cband(k):
            return config.CYCLING_BANDS.get(k, {}).get(btype)

        cyc_specs = [
            ("cadence_rpm", "Cadence", "rpm", config.CYCLING_CADENCE_BAND, 2.0, 0),
            ("knee_bdc_deg", "Knee angle at BDC", "deg", cband("knee_bdc_deg"), 2.0, 1),
            ("knee_tdc_min_deg", "Knee angle at TDC", "deg", cband("knee_tdc_deg"), 2.0, 1),
            ("hip_tdc_min_deg", "Hip angle at TDC", "deg", cband("hip_tdc_deg"), 2.0, 1),
            ("torso_deg", "Torso angle", "deg", cband("torso_deg"), 2.0, 1),
        ]
        for key, label, unit, band, eps, digits in cyc_specs:
            base = scalar(ab.get(key))
            follow = scalar(bb.get(key))
            if base is None and follow is None:
                continue
            verdict = _verdict("band", base, follow, {"eps": eps, "band_values": band})
            rows.append({
                "metric": key, "label": label, "unit": unit, "direction": "band",
                "group": "bike", "band": list(band) if band else None,
                "baseline": None if base is None else round(float(base), digits),
                "followup": None if follow is None else round(float(follow), digits),
                "delta": (None if (base is None or follow is None)
                          else round(float(follow) - float(base), digits)),
                "verdict": verdict,
            })

    # ---------------------------------------------------------- aero (area)
    aa = (a_summary or {}).get("aero") or {}
    ba = (b_summary or {}).get("aero") or {}
    fa1, fa2 = scalar(aa.get("fa_m2_median")), scalar(ba.get("fa_m2_median"))
    if fa1 and fa2:
        speed = ((b_meta or {}).get("speed_kmh") or (a_meta or {}).get("speed_kmh")
                 or 40.0)
        cd = scalar(ba.get("cd_assumed")) or config.AERO_CD_DEFAULT
        delta = float(fa2) - float(fa1)
        verdict = _verdict("down", fa1, fa2, {"eps": 0.002})
        rows.append({
            "metric": "fa_m2", "label": "Frontal area (aero)", "unit": "m²",
            "direction": "down", "group": "aero",
            "baseline": round(float(fa1), 3), "followup": round(float(fa2), 3),
            "delta": round(delta, 3), "verdict": verdict,
        })
        d = _aero_watts(float(fa1), float(fa2), speed, cd)
        if d is not None:
            rows.append({
                "metric": "aero_watts",
                "label": f"Est. aero drag power change @ {float(speed):.0f} km/h "
                         f"(Cd {float(cd):g})",
                "unit": "W", "direction": "down", "group": "aero",
                "baseline": None, "followup": round(d, 1), "delta": round(d, 1),
                "verdict": "Improved" if d < -1.0 else ("Regressed" if d > 1.0 else "Neutral"),
                "note": "assumes Cd and all else equal; area measured from video "
                        "silhouette",
            })
    improved = sum(1 for r in rows if r["verdict"] == "Improved")
    regressed = sum(1 for r in rows if r["verdict"] == "Regressed")
    neutral = sum(1 for r in rows if r["verdict"] == "Neutral")
    return {
        "baseline": a_meta or {},
        "followup": b_meta or {},
        "rows": rows,
        "summary": {"improved": improved, "neutral": neutral, "regressed": regressed,
                    "n": len(rows)},
        "note": ("Improved/Neutral/Regressed compare the follow-up against the "
                 "baseline session; neutral thresholds are per metric (see "
                 "advisor.COMPARE_METRICS)."),
    }
