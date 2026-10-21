"""Second-view (rear / side) analysis for two-source sessions.

The primary source sets the session's view plane; the second source is an
independent recording (or camera) with its own time base and its own plane.
Running the same metrics pipeline over it yields the metric family the primary
view cannot measure — a rear (frontal) clip gives pelvic drop, frontal knee
collapse and arm-swing symmetry, while a side (sagittal) clip gives overstride,
shin angle, knee flexion and trunk lean.

Gait events for this view are detected from its own rows (no keyframe latch is
needed here).  The block is stored under ``summary["second_view"]`` so the
report, the API and any downstream consumer read one shape.
"""
from __future__ import annotations

import advisor
import config
import metrics as M

# fewer frames than this is a camera glitch, not a usable second view
MIN_ROWS = 15


def analyze(rows, params=None, plane=None, events=None, discipline="running",
            direction=None):
    """Analyse the second view; returns a summary dict or None.

    ``rows`` are that view's own frame-metric rows with its own ``t_rel``.
    Returning None (instead of a half-empty block) keeps old reports and the
    UI honest when the second source contributed nothing usable.

    ``direction`` (``"front"`` / ``"rear"``) is a camera-facing label carried
    into the block for the report.  It deliberately does NOT flip any sign:
    frontal metric math is mirror-invariant (``frontal_knee_deviation`` derives
    the midline from the pelvis itself and ``pelvic_tilt`` is y-based), so a
    front camera needs no axis correction — the direction is metadata only.
    """
    params = params or {}
    disc = str(discipline or "running").lower()
    plane = str(plane or config.DEFAULT_PLANE2).lower()
    if plane not in config.VIEW_PLANES:
        plane = config.DEFAULT_PLANE2
    rows = list(rows or [])
    if len(rows) < MIN_ROWS or disc != "running":
        return None
    sv = M.analyze_session(
        rows, weight_kg=params.get("weight_kg"),
        height_cm=params.get("height_cm"), speed_kmh=params.get("speed_kmh"),
        view_plane=plane, events=events, discipline=disc)
    if sv.get("error"):
        return None
    try:
        advisor.enrich(sv)          # its own form score + frontal risk rules
    except Exception:
        pass
    sv["source"] = "second view"
    sv["rows"] = len(rows)
    d = str(direction or "").lower()
    if d in config.DIRECTIONS:
        sv["direction"] = d        # label only — see docstring: no sign flips
    return sv
