"""Joint-angle study: live snapshot, operator captures, session summary.

The capture loop already computes a per-frame included angle for every joint in
``metrics.JOINT_SPECS`` plus ``trunk_lean`` (180 deg = straight, aspect
corrected by ``metrics.frame_metrics``).  This module only reshapes those
numbers for the UI and the report:

* ``snapshot(live)``          -> {joint: deg} for the joints actually measured
* ``capture(session, live, label)`` -> one labelled snapshot stored on the
  running session (the operator's "capture this angle" tap)
* ``summarize(rows)``         -> per-joint mean/min/max/ROM/n for the session

Nothing here invents a value: a joint the pose model did not see on that frame
stays absent, and it stays absent in every aggregate too.
"""
from __future__ import annotations

import math

# Joints the panel and the report show, in reading order (side, then trunk).
JOINTS = (
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "trunk_lean",
)

LABELS = {
    "left_knee": "Left knee",
    "right_knee": "Right knee",
    "left_hip": "Left hip",
    "right_hip": "Right hip",
    "left_ankle": "Left ankle",
    "right_ankle": "Right ankle",
    "left_elbow": "Left elbow",
    "right_elbow": "Right elbow",
    "left_shoulder": "Left shoulder",
    "right_shoulder": "Right shoulder",
    "trunk_lean": "Trunk lean",
}

# The report groups joints by segment so a long table stays readable.
GROUPS = (
    ("Knee", ("left_knee", "right_knee")),
    ("Hip", ("left_hip", "right_hip")),
    ("Ankle", ("left_ankle", "right_ankle")),
    ("Elbow", ("left_elbow", "right_elbow")),
    ("Shoulder", ("left_shoulder", "right_shoulder")),
    ("Trunk", ("trunk_lean",)),
)

MAX_CAPTURES = 40          # per session; the operator taps a handful, not 500
LABEL_MAX_CHARS = 60


def _num(value):
    """float or None - never a string, never NaN/inf."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def snapshot(live: dict) -> dict:
    """{joint: degrees} for every joint present in a live metrics dict."""
    out = {}
    for joint in JOINTS:
        val = _num((live or {}).get(joint))
        if val is not None:
            out[joint] = round(val, 1)
    return out


def capture(session: dict, live: dict, label: str = "") -> dict:
    """Store one labelled angle snapshot on the running session.

    Raises ``ValueError`` when the frame has no measurable joint or the session
    already holds ``MAX_CAPTURES`` snapshots, so the API can answer 400 with a
    reason instead of writing an empty record.
    """
    if not isinstance(session, dict):
        raise ValueError("no running session")
    angles = snapshot(live)
    if not angles:
        raise ValueError("no joint angles on this frame yet - wait for the runner")
    caps = session.setdefault("angle_captures", [])
    if len(caps) >= MAX_CAPTURES:
        raise ValueError(f"capture limit reached ({MAX_CAPTURES})")
    text = " ".join(str(label or "").split())[:LABEL_MAX_CHARS]
    record = {
        "t": round(float(session.get("elapsed") or 0.0), 2),
        "label": text or f"Capture {len(caps) + 1}",
        "angles": angles,
    }
    caps.append(record)
    return record


def summarize(rows) -> dict:
    """Per-joint mean/min/max/ROM/n over the session's rows (degrees)."""
    out = {}
    for joint in JOINTS:
        vals = []
        for row in rows or ():
            val = _num(row.get(joint) if isinstance(row, dict) else None)
            if val is not None:
                vals.append(val)
        if not vals:
            continue
        out[joint] = {
            "mean": round(sum(vals) / len(vals), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "rom": round(max(vals) - min(vals), 1),
            "n": len(vals),
        }
    return out


def overview(summary: dict) -> dict:
    """Compact counts for /api/state: how many captures, which joints seen."""
    caps = (summary or {}).get("angle_captures") or []
    seen = sorted({j for c in caps for j in (c.get("angles") or {})})
    return {"captures": len(caps), "joints": seen,
            "last": (caps[-1] if caps else None)}
