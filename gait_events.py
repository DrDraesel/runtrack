"""Gait phase event latching + annotated freeze-frame keyframes for RunTrack.

Per gait cycle three events are latched, per leg:

  IC  Initial Contact — the END OF THE LANDING DESCENT.  On the (smoothed)
      ankle-height signal the swing descent accelerates, then decays as the leg
      reaches for the ground; the contact is the frame where the descent speed
      has decayed to GAIT_CONTACT_STOP_FRAC of its own peak — the foot has
      stopped coming down.  A session that OPENS with the foot already inside
      the ground band (recording started mid-stance) gets its first IC at the
      first frame instead; the band alone never invents a contact mid-swing.
  MS  Mid-Stance — maximum knee flexion under load, i.e. the frame with the
      smallest knee angle between the latched IC and the following TO.
  TO  Toe-Off — ankle lift-off: the frame of fastest upward ankle motion
      (largest negative dy/dt) of the first real rise after the stance.  A rise
      only counts when it reaches at least one band-height per second (and
      GAIT_RISE_MIN) over GAIT_EPISODE_MIN_FRAMES frames, so landmark jitter
      mid-stance cannot fake a toe-off.

Two entry points share the SAME state machine (gait_events._advance_side), so
the online and offline passes latch the same events from the same velocity
profile:

  detect_events(rows)   — offline, pure numpy over logged per-frame metric rows
  GaitLatch             — online, rolling window + ring buffer of recent frames,
                          used by the live app and the offline CLI so that both
                          can store annotated keyframe images.

Honest limits: 2D ankle-position heuristics from a single camera; the ground
band is a percentile estimate of the foot-height signal, so a session where the
whole foot never leaves the frame or where the runner walks produces no events
rather than made-up ones.
"""
from __future__ import annotations

import math

import numpy as np

import config

SIDES = ("left", "right")
SIDE_SHORT = {"left": "l", "right": "r"}
KNEE_KEY = {"left": "left_knee", "right": "right_knee"}
# shoulder-hip-knee included angle (hip angle) — cycling BDC/TDC keyframes
SHOULDER_HIP_KEY = {"left": "left_hip", "right": "right_hip"}
FLEX_KEY = {"left": "knee_flex_l", "right": "knee_flex_r"}
ANK_Y_KEY = {"left": "l_ankle_y", "right": "r_ankle_y"}
SHIN_KEY = {"left": "shin_angle_l", "right": "shin_angle_r"}
OVER_KEY = {"left": "overstride_l", "right": "overstride_r"}
VALGUS_KEY = {"left": "knee_valgus_l", "right": "knee_valgus_r"}
HIPEXT_KEY = {"left": "hip_ext_l", "right": "hip_ext_r"}
ANKLE_ANG_KEY = {"left": "left_ankle", "right": "right_ankle"}

EVENT_LABEL = {
    "IC": "Initial contact",
    "MS": "Mid-stance",
    "TO": "Toe-off",
    "BDC": "Bottom dead centre (knee fit ref)",
    "TDC": "Top dead centre (closed hip)",
}


# ------------------------------------------------------------------ helpers --
def _clean(values):
    """NaN-aware numpy array from a list of optional numbers."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = np.nan
        out.append(f if math.isfinite(f) else np.nan)
    return np.array(out, dtype=float)


def _smooth3(y):
    """3-point moving average that keeps NaNs where the input is invalid."""
    if y.size < 3:
        return y.copy()
    out = y.copy()
    for i in range(1, y.size - 1):
        w = y[i - 1:i + 2]
        if not np.isnan(w).any():
            out[i] = float(np.mean(w))
    return out


def _vel(t, y):
    """Central-difference dy/dt (NaN-aware)."""
    v = np.full(y.shape, np.nan)
    n = y.size
    for i in range(n):
        a = i - 1 if i > 0 else i
        b = i + 1 if i < n - 1 else i
        if b <= a:
            continue
        if np.isnan(y[a]) or np.isnan(y[b]):
            continue
        dt = t[b] - t[a]
        if dt <= 1e-6:
            continue
        v[i] = (y[b] - y[a]) / dt
    return v


def lower_body_coverage(rows_or_frames) -> float:
    """Share of frames (%) with usable ankle data — the gate for gait events."""
    vals = []
    for r in rows_or_frames or []:
        fm = r.get("fm") if isinstance(r, dict) and "fm" in r else r
        if not isinstance(fm, dict):
            continue
        v = fm.get("l_ankle_y")
        if v is None:
            v = fm.get("r_ankle_y")
        try:
            f = float(v)
        except (TypeError, ValueError):
            vals.append(0.0)
            continue
        vals.append(1.0 if math.isfinite(f) else 0.0)
    if not vals:
        return 0.0
    return 100.0 * sum(vals) / len(vals)


def events_not_assessable(coverage_pct: float | None) -> bool:
    return coverage_pct is not None and coverage_pct < config.LOWER_BODY_MIN_COVERAGE_PCT


def summarize(events) -> dict:
    """Counts per event type + per side (JSON-safe)."""
    out = {"IC": 0, "MS": 0, "TO": 0}
    sides = {"IC": {"left": 0, "right": 0}, "MS": {"left": 0, "right": 0},
             "TO": {"left": 0, "right": 0}}
    for e in events or []:
        typ = e.get("type")
        if typ in out:
            out[typ] += 1
            side = e.get("side")
            if side in sides[typ]:
                sides[typ][side] += 1
    out["by_side"] = sides
    out["n"] = int(sum(v for k, v in out.items() if isinstance(v, int)))
    return out


def _score_event(e) -> float:
    q = e.get("quality")
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = 0.0
    if not math.isfinite(q):
        q = 0.0
    return q


def select_best(snaps, per_type=None, max_total=None) -> list:
    """Pick the best freeze-frames: up to `per_type` per event type, balanced.

    `snaps` are event dicts that carry a 'frame' (image bytes). Keeps the
    highest-quality frames of each type so the report shows one good example of
    every phase instead of nine frames of the same moment.
    """
    per_type = int(per_type or config.KEYFRAMES_PER_TYPE)
    max_total = int(max_total or config.KEYFRAMES_MAX)
    with_frame = [s for s in (snaps or []) if s.get("frame")]
    picked: list = []
    for typ in ("IC", "MS", "TO"):
        group = [s for s in with_frame if s.get("type") == typ]
        group.sort(key=_score_event, reverse=True)
        picked.extend(group[:per_type])
    if len(picked) > max_total:
        picked.sort(key=_score_event, reverse=True)
        picked = picked[:max_total]
    picked.sort(key=lambda e: (e.get("t") or 0.0))
    return picked


# ------------------------------------------- ground band + shared machine ---
def _ground_band(y):
    """Percentile ground band of one ankle-height series, or None.

    Returns (thr, hi, band): `band` is the p5..p95 excursion of the ankle
    height and `thr` is the "foot is down" threshold (the top
    GAIT_STANCE_FRAC of that band, remember: image y grows downward, so a
    planted foot has a LARGER y).  None when the series is too short or the
    foot never leaves the ground — no lift means no events, never invented ones.
    """
    if y is None:
        return None
    v = y[~np.isnan(y)]
    if v.size < 12:
        return None
    lo = float(np.percentile(v, 5))
    hi = float(np.percentile(v, 95))
    band = hi - lo
    if band < 0.02:
        return None
    return hi - config.GAIT_STANCE_FRAC * band, hi, band


def _as_lms(lms):
    """Coerce to a landmark array shaped (>=33, >=4); None when it is not one.

    Guards the keyframe renderer and the latch buffer against callers that hand
    in a plain metric dict by mistake — an un-annotated keyframe is better than
    a crash mid-session.
    """
    if lms is None:
        return None
    try:
        arr = np.asarray(lms, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 33 or arr.shape[1] < 4:
        return None
    return arr


def _new_side_state() -> dict:
    """Fresh state for one leg's event machine (used by both entry points)."""
    return {
        "next": 0,          # absolute index of the next frame to consume
        "phase": "up",      # "up" = foot off the ground (swing); "down" = stance
        "armed": False,     # a real foot-lift was seen: the next descent can land
        "start_done": False,
        "peak_v": 0.0,      # fastest descent speed since the last foot lift
        "stop_i": None,     # landing frame waiting for its confirmation frame
        "stop_v": None,
        "rise_i": None,     # first frame of the open rising episode (the toe-off)
        "rise_j": None,     # fastest frame of that episode
        "rise_v": None,     # fastest upward speed of that episode
        "rise_n": 0,        # frames in the open rising episode
        "open_ic": None,    # (t, i) of the stance waiting for its toe-off
        "last_ic_t": None,  # most recent initial contact (min-gap guard)
        "last_to_t": None,  # most recent toe-off (min-gap guard)
    }


def _rise_gate(band_width) -> float:
    """Minimum upward speed (norm units / s) that counts as a real foot lift."""
    return max(config.GAIT_RISE_MIN, config.GAIT_RISE_BAND_FRAC * float(band_width))


def _open_contact(st, side, t0, y0, v0, band, events) -> bool:
    """Session opening case: the recording may have started with the foot down.

    Only latches when the very first frame really sits inside the ground band
    and the foot is not rising — a session that opens mid-swing gets no IC here,
    it gets one when the landing descent completes.  The check is one-shot, so
    it waits (without deciding) until the band is a mature foot-lift estimate:
    an immature rolling window must not be able to answer it wrongly.
    """
    if st["start_done"] or band is None:
        return False
    _thr, _hi, b = band
    if float(b) < config.GAIT_START_BAND_MIN:
        return False                            # retry once the band is mature
    st["start_done"] = True
    thr = _thr
    try:
        y0f, t0f, v0f = float(y0), float(t0), float(v0)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(y0f) and math.isfinite(t0f) and y0f >= thr):
        return False
    if not math.isfinite(v0f) or v0f <= -_rise_gate(b):
        return False                            # foot is mid-swing: rising
    events.append({"type": "IC", "side": side, "t": t0f, "i": 0, "velocity": 0.0})
    st["phase"] = "down"
    st["armed"] = False
    st["open_ic"] = (t0f, 0)
    st["last_ic_t"] = t0f
    return True


def _min_knee_frame(knee, base, i0, i1):
    """Absolute index of the smallest (most flexed) knee angle in [i0, i1]."""
    best, best_v = None, None
    for i in range(int(i0), int(i1) + 1):
        p = i - base
        if p < 0 or p >= knee.size:
            continue
        try:
            v = float(knee[p])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if best_v is None or v < best_v:
            best, best_v = i, v
    return best


def _close_rise(st, side, t, knee, base, events) -> None:
    """The rising episode ended — latch its toe-off and the mid-stance before it.

    The toe-off is the FIRST frame of the episode (ankle lift-off), not the
    fastest one: the fastest ankle motion is already early swing, past the knee
    flexion that mid-stance is measured against.  A rise that lasted fewer than
    GAIT_EPISODE_MIN_FRAMES leaves the stance open (it was jitter, not a step).
    """
    if st["rise_i"] is not None and st["rise_n"] >= int(config.GAIT_EPISODE_MIN_FRAMES):
        j = int(st["rise_i"])
        to_t = float(t[j - base])
        if st["last_to_t"] is None or to_t - st["last_to_t"] >= config.GAIT_MIN_GAP_S:
            events.append({"type": "TO", "side": side, "t": to_t, "i": j,
                           "velocity": (round(float(st["rise_v"]), 3)
                                        if st["rise_v"] is not None else None)})
            st["last_to_t"] = to_t
            ic = st["open_ic"]
            if ic is not None and ic[1] < j:
                # mid-stance: maximum knee flexion (smallest knee angle) between
                # the latched IC and this toe-off — the load-bearing phase
                k = _min_knee_frame(knee, base, ic[1], j)
                if k is not None and ic[1] < k < j:
                    events.append({"type": "MS", "side": side,
                                   "t": float(t[k - base]), "i": k})
            st["phase"] = "up"
            st["armed"] = True
            st["open_ic"] = None
    st["rise_i"] = None
    st["rise_j"] = None
    st["rise_v"] = None
    st["rise_n"] = 0


def _latch_ic(st, side, t, base, events) -> None:
    """The pending landing was confirmed: latch the initial contact."""
    i = int(st["stop_i"])
    v = float(st["stop_v"])
    events.append({"type": "IC", "side": side, "t": float(t[i - base]), "i": i,
                   "velocity": round(v, 3)})
    st["last_ic_t"] = float(t[i - base])
    st["phase"] = "down"
    st["armed"] = False
    st["peak_v"] = 0.0
    st["stop_i"] = None
    st["stop_v"] = None
    st["open_ic"] = (float(t[i - base]), i)


def _advance_side(st, side, t, y, knee, vel, base, band, upto, events) -> None:
    """Consume frames [st['next'], upto] of one leg and latch IC / MS / TO.

    The machine is purely velocity based (the band only gates the session-start
    case and the rise/landing speeds) because a single ankle-height signal
    cannot tell a mid-swing hover from a planted foot.  Both detect_events
    (base 0, whole session) and GaitLatch (base = first buffered frame, called
    as frames arrive) drive this function, so the two latching paths agree.

    Per frame:
      * a fast RISE (|v| >= the rise gate) opens/continues a foot-lift episode
        and arms the next landing; when the rise stops the episode is closed
        (toe-off + mid-stance);
      * a descent that decelerates under GAIT_CONTACT_STOP_FRAC of its own peak
        becomes a PENDING landing, latched as the initial contact once the next
        frame confirms the foot really stopped coming down (so a truncated
        session tail cannot invent a contact).
    """
    if band is None:
        return
    gate = _rise_gate(band[2])
    stop = float(config.GAIT_CONTACT_STOP_FRAC)
    min_gap = float(config.GAIT_MIN_GAP_S)
    i = int(st["next"])
    n = y.size
    while i <= upto and i < n:
        p = i - base
        try:
            v = float(vel[p]) if 0 <= p < vel.size else float("nan")
            yy = float(y[p]) if 0 <= p < y.size else float("nan")
        except (TypeError, ValueError):
            v, yy = float("nan"), float("nan")
        if math.isfinite(v) and math.isfinite(yy):
            # ---- confirm or move a pending landing
            if st["stop_i"] is not None:
                if v <= 0.0 or v >= abs(st["stop_v"]):
                    _latch_ic(st, side, t, base, events)
                elif v > 0.0 and v < abs(st["stop_v"]):
                    st["stop_i"], st["stop_v"] = i, v   # still slowing down
            # ---- foot-lift (swing) episode tracking, in both phases
            if v <= -gate:
                if st["rise_i"] is None:
                    st["rise_i"], st["rise_j"] = i, i
                    st["rise_v"] = v
                    st["rise_n"] = 0
                elif v < st["rise_v"]:
                    st["rise_j"], st["rise_v"] = i, v
                st["rise_n"] += 1
                st["armed"] = True
                st["peak_v"] = 0.0
            else:
                if st["rise_i"] is not None:
                    if st["phase"] == "down":
                        _close_rise(st, side, t, knee, base, events)
                    else:
                        st["rise_i"] = None
                        st["rise_j"] = None
                        st["rise_v"] = None
                        st["rise_n"] = 0
            # ---- landing: the descent decelerates after its fastest frame
            if (st["phase"] == "up" and st["stop_i"] is None and st["armed"]
                    and v > 0.0):
                if st["peak_v"] <= 0.0 or v > st["peak_v"]:
                    st["peak_v"] = v
                if (v <= stop * st["peak_v"]
                        and (st["last_ic_t"] is None
                             or float(t[p]) - st["last_ic_t"] >= min_gap)):
                    st["stop_i"], st["stop_v"] = i, v
        i += 1
    st["next"] = i


# ------------------------------------------------------- offline detection --
def detect_events(rows, min_gap: float | None = None) -> list:
    """Latch IC / MS / TO events from logged per-frame metric rows.

    `rows` is the same list of per-frame dicts that is written to metrics.csv
    (needs t_rel + the ankle y and knee angle series).  Pure numpy — no images,
    no OpenCV — so it can run during analysis too.  Drives the same state
    machine as the online GaitLatch (see gait_events._advance_side).
    """
    if not rows:
        return []
    min_gap = float(config.GAIT_MIN_GAP_S if min_gap is None else min_gap)
    t = _clean([r.get("t_rel") for r in rows])
    ok = ~np.isnan(t)
    if ok.sum() < 10 or (np.max(t[ok]) - np.min(t[ok])) < 1.0:
        return []
    events: list = []
    for side in SIDES:
        y = np.where(ok, _clean([r.get(ANK_Y_KEY[side]) for r in rows]), np.nan)
        band = _ground_band(y)
        if band is None:
            continue                        # no measurable foot lift: no events
        knee = np.where(ok, _clean([r.get(KNEE_KEY[side]) for r in rows]), np.nan)
        vel = _vel(t, _smooth3(y))
        st = _new_side_state()
        _open_contact(st, side, t[0], y[0], vel[0], band, events)
        # the last frame is only consumed once its velocity and its local
        # extrema are confirmed by the frames after it
        _advance_side(st, side, t, y, knee, vel, 0, band, len(rows) - 2, events)
    events.sort(key=lambda e: e["t"])
    return events


def stance_times(events) -> dict:
    """Stance duration per side from IC -> TO pairs (seconds)."""
    out = {"left": [], "right": []}
    open_ic = {}
    for e in sorted(events or [], key=lambda x: x.get("t", 0.0)):
        side, typ = e.get("side"), e.get("type")
        if side not in out:
            continue
        if typ == "IC":
            open_ic[side] = e.get("t")
        elif typ == "TO" and side in open_ic:
            t0 = open_ic.pop(side)
            try:
                d = float(e.get("t")) - float(t0)
            except (TypeError, ValueError):
                continue
            if 0.05 < d < 1.5:
                out[side].append(d)
    return out


def event_values(events, times, series, side_map) -> dict:
    """Sample a per-frame series at each event time, per side.

    Used by metrics.analyze_session to attach biomechanical values to latched
    events (shin angle at IC, knee flexion at MS, ...).
    """
    out: dict = {s: [] for s in SIDES}
    for e in events or []:
        side = e.get("side")
        if side not in out:
            continue
        key = side_map.get(side)
        if key is None:
            continue
        arr = series(key) if callable(series) else series.get(key)
        if arr is None:
            continue
        d = np.abs(np.asarray(times) - float(e.get("t")))
        d = np.where(np.isnan(d), np.inf, d)
        if d.size == 0 or not np.isfinite(d).any():
            continue
        j = int(np.argmin(d))
        if j < len(arr):
            v = arr[j]
            if v is not None and math.isfinite(float(v)):
                out[side].append(float(v))
    return out


# --------------------------------------------------------- online latching --
class GaitLatch:
    """Online gait event latch with a ring buffer of recent annotated frames.

    push(t, fm, frame_jpeg, lms) once per processed frame; the latch returns the
    events that became final during that call.  Call save_snapshots(out_dir) at
    the end of the session to write the annotated freeze-frames.  It drives the
    same state machine as detect_events (gait_events._advance_side), so a live
    session and a re-analysis of its metrics.csv latch the same events.
    """

    _KEEP_FRAMES = 600            # per-leg series kept for the machine (~20 s)

    def __init__(self, buffer_frames: int | None = None, per_type: int | None = None,
                 max_total: int | None = None, window: int = 90):
        self.buffer_frames = int(buffer_frames or config.GAIT_BUFFER_FRAMES)
        self.per_type = int(per_type or config.KEYFRAMES_PER_TYPE)
        self.max_total = int(max_total or config.KEYFRAMES_MAX)
        self.window = int(window)
        self.buf: list = []
        self._i = 0
        self._state = {s: _new_side_state() for s in SIDES}
        self._series = {s: {"t": [], "y": [], "knee": [], "base": 0} for s in SIDES}
        self._first: dict = {}
        self.latched: list = []

    # ------------------------------------------------------------------ push
    def push(self, t, fm: dict, frame_jpeg=None, lms=None) -> list:
        fm = fm or {}
        arr = _as_lms(lms)
        if arr is not None:
            fm = dict(fm)
            try:
                fm["_mean_vis"] = float(np.mean(arr[:, 3]))
            except Exception:
                pass
        i = self._i
        self.buf.append({"i": i, "t": float(t), "fm": dict(fm),
                         "jpeg": frame_jpeg, "lms": arr})
        if len(self.buf) > self.buffer_frames:
            self.buf = self.buf[-self.buffer_frames:]
        for side in SIDES:
            ser = self._series[side]
            ser["t"].append(float(t))
            ser["y"].append(fm.get(ANK_Y_KEY[side]))
            ser["knee"].append(fm.get(KNEE_KEY[side]))
            self._trim(side)
        if i == 0:
            self._first = {s: {"t": float(t), "y": fm.get(ANK_Y_KEY[s])} for s in SIDES}
        elif i == 1:
            # once the second frame exists the session-opening velocity is known
            for s in SIDES:
                f = self._first.get(s) or {}
                try:
                    y0 = float(f.get("y"))
                    y1 = float(fm.get(ANK_Y_KEY[s]))
                    dt = float(t) - float(f.get("t"))
                    f["v"] = (y1 - y0) / dt if dt > 1e-6 else float("nan")
                except (TypeError, ValueError):
                    f["v"] = float("nan")
        self._i += 1
        new: list = []
        for side in SIDES:
            new.extend(self._advance(side))
        new.sort(key=lambda e: e["t"])
        return new

    def _trim(self, side) -> None:
        """Drop the part of a leg's series the machine will never look at again."""
        ser = self._series[side]
        st = self._state[side]
        n = len(ser["t"])
        if n <= self._KEEP_FRAMES:
            return
        lowest = int(st["next"]) - 2
        if st["open_ic"] is not None:
            lowest = min(lowest, int(st["open_ic"][1]))
        if st["stop_i"] is not None:
            lowest = min(lowest, int(st["stop_i"]))
        if st["rise_i"] is not None:
            lowest = min(lowest, int(st["rise_i"]))
        if st["rise_j"] is not None:
            lowest = min(lowest, int(st["rise_j"]))
        drop = min(n - self._KEEP_FRAMES, max(0, lowest - int(ser["base"])))
        if drop <= 0:
            return
        for k in ("t", "y", "knee"):
            del ser[k][:drop]
        ser["base"] = int(ser["base"]) + drop

    def _advance(self, side) -> list:
        """Feed the new frames of one leg to the shared event machine."""
        ser = self._series[side]
        st = self._state[side]
        band = _ground_band(_clean(ser["y"][-self.window:]))
        if band is None:
            return []
        base = int(ser["base"])
        t = np.array(ser["t"], dtype=float)
        y = _clean(ser["y"])
        knee = _clean(ser["knee"])
        vel = _vel(t, _smooth3(y))
        raw: list = []
        if not st["start_done"]:
            f = self._first.get(side) or {}
            _open_contact(st, side, f.get("t"), f.get("y"), f.get("v"),
                          band, raw)
        upto = base + t.size - 2            # velocity needs the frame after it
        _advance_side(st, side, t, y, knee, vel, base, band, upto, raw)
        if not raw:
            return []
        by_i = {b["i"]: b for b in self.buf}
        out: list = []
        for ev in raw:
            b = by_i.get(ev.get("i")) or {}
            rich = {"type": ev["type"], "side": ev["side"], "t": float(ev["t"]),
                    "i": ev.get("i"), "velocity": ev.get("velocity"),
                    "frame": b.get("jpeg"), "lms": b.get("lms"),
                    "fm": dict(b.get("fm") or {}),
                    "quality": self._quality(b) if b else 0.0}
            out.append(rich)
            self.latched.append(rich)
        return out

    def _quality(self, b) -> float:
        """Rough snapshot quality: pose present + the event metric available."""
        fm = b.get("fm") or {}
        n = 0
        for k in ("shin_angle_l", "shin_angle_r", "left_knee", "right_knee",
                  "pelvic_tilt_deg", "l_ankle_y", "r_ankle_y", "left_hip", "right_hip"):
            if fm.get(k) is not None:
                n += 1
        vis = fm.get("_mean_vis")
        try:
            vis = float(vis)
        except (TypeError, ValueError):
            vis = 0.0
        if not math.isfinite(vis):
            vis = 0.0
        return round(n / 9.0 + max(0.0, min(1.0, vis)), 3)

    # ------------------------------------------------------------- snapshots
    def lower_body_coverage(self) -> float:
        """Share of pushed frames (%) that carried usable ankle data."""
        return lower_body_coverage(self.buf)

    def events(self) -> list:
        """Finalized events, oldest first (they carry frame bytes + metrics)."""
        return list(self.latched)

    def save_snapshots(self, out_dir, refine=None) -> list:
        """Write annotated freeze-frames for the best latched events.

        Returns a list of JSON-safe metadata dicts (file name, event, values).
        ``refine`` (optional) upgrades the retained keyframes to
        native-resolution landmarks first (see gait_events.refine_event).
        """
        snaps = select_best(self.latched, self.per_type, self.max_total)
        if not snaps:
            return []
        return save_events(snaps, out_dir, refine=refine)

    def best_events(self) -> list:
        return select_best(self.latched, self.per_type, self.max_total)


# ------------------------------------------------------ keyframe rendering --
def _pt(lms, idx, w, h):
    return int(lms[idx, 0] * w), int(lms[idx, 1] * h)


def _caption(img, lines, color=(255, 255, 255)):
    import cv2
    h, w = img.shape[:2]
    pad, lh = 8, 20
    box_h = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (12, 16, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    for k, txt in enumerate(lines):
        y = h - box_h + pad + lh * k + 14
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return img


def _arrow(img, p0, p1, color, thickness=2):
    import cv2
    cv2.arrowedLine(img, p0, p1, (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.18)


def _vertical_ref(img, x, y, color=(200, 200, 200), span=110):
    import cv2
    cv2.line(img, (x, y - span), (x, y + span), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (x, y - span), (x, y + span), color, 1, cv2.LINE_AA)


def annotate_event(img, ev):
    """Draw the event overlay: skeleton metrics + the plane-specific geometry."""
    import cv2
    lms = _as_lms(ev.get("lms"))
    fm = ev.get("fm") or {}
    typ = ev.get("type")
    side = ev.get("side")
    short = SIDE_SHORT.get(side, "l")
    h, w = img.shape[:2]
    if lms is None:
        return _caption(img, [f"{EVENT_LABEL.get(typ, typ)} ({side}) — no landmarks"])
    hip_i = 23 if short == "l" else 24
    knee_i = 25 if short == "l" else 26
    ank_i = 27 if short == "l" else 28
    lines = [f"{EVENT_LABEL.get(typ, typ)} — {side} leg @ t = {ev.get('t', 0):.2f} s"]

    if typ == "IC":
        sa = fm.get(SHIN_KEY[side])
        if sa is not None:
            lines.append(f"Touchdown shin angle {abs(float(sa)):.1f} deg "
                         f"(target 0-5, >8 flags braking)")
        ov = fm.get(OVER_KEY[side])
        if ov is not None:
            lines.append(f"Overstride vector {abs(float(ov)):.3f} norm-units ahead of pelvis")
        kx, ky = _pt(lms, knee_i, w, h)
        _vertical_ref(img, kx, ky)
        _arrow(img, _pt(lms, knee_i, w, h), _pt(lms, ank_i, w, h), (61, 217, 255), 3)
        if ov is not None:
            px = int(((lms[23, 0] + lms[24, 0]) / 2.0) * w)
            py = int(((lms[23, 1] + lms[24, 1]) / 2.0) * h)
            ay = _pt(lms, ank_i, w, h)[1]
            _arrow(img, (px, ay), _pt(lms, ank_i, w, h), (255, 170, 60), 2)
            cv2.line(img, (px, ay - 60), (px, ay + 60), (255, 170, 60), 1, cv2.LINE_AA)
    elif typ == "MS":
        kf = fm.get(KNEE_KEY[side])
        if kf is not None:
            lines.append(f"Stance knee angle {float(kf):.1f} deg "
                         f"(flexion {180 - float(kf):.1f} deg, target 35-45)")
        pd = fm.get("pelvic_tilt_deg")
        if pd is not None:
            lines.append(f"Pelvic drop line {float(pd):+.1f} deg "
                         f"(+ = right hip lower; >4-5 flags)")
        cv2.line(img, _pt(lms, 23, w, h), _pt(lms, 24, w, h), (255, 170, 60), 3, cv2.LINE_AA)
        x0, y0 = _pt(lms, 23, w, h)
        x1, y1 = _pt(lms, 24, w, h)
        cv2.line(img, (x0, y0), (x1 + int((x1 - x0) * 0.6) if x1 > x0 else x1 - int((x0 - x1) * 0.6), y1),
                 (255, 170, 60), 1, cv2.LINE_AA)
    elif typ == "TO":
        he = fm.get(HIPEXT_KEY[side])
        if he is not None:
            lines.append(f"Hip extension (thigh) {float(he):+.1f} deg behind vertical")
        aa = fm.get(ANKLE_ANG_KEY[side])
        if aa is not None:
            lines.append(f"Push-off ankle angle {float(aa):.1f} deg")
        _arrow(img, _pt(lms, hip_i, w, h), _pt(lms, knee_i, w, h), (61, 217, 255), 3)
        kx, ky = _pt(lms, knee_i, w, h)
        _vertical_ref(img, kx, ky, (170, 220, 255), 90)
        cv2.circle(img, _pt(lms, ank_i, w, h), 8, (255, 170, 60), 2, cv2.LINE_AA)
    elif typ in ("BDC", "TDC"):
        ka = fm.get(KNEE_KEY[side])
        if ka is not None:
            if typ == "BDC":
                lines.append(f"Knee {float(ka):.1f} deg at bottom of stroke "
                             f"(fit ref: 140-150 road / 140-145 TT-tri)")
            else:
                lines.append(f"Knee {float(ka):.1f} deg at top of stroke "
                             f"(most flexed; keep > 68)")
        ha = fm.get(SHOULDER_HIP_KEY.get(side, "left_hip"))
        if ha is not None:
            lines.append(f"Hip angle {float(ha):.1f} deg (shoulder-hip-knee)")
        cv2.line(img, _pt(lms, hip_i, w, h), _pt(lms, knee_i, w, h), (61, 217, 255), 3,
                 cv2.LINE_AA)
        cv2.line(img, _pt(lms, knee_i, w, h), _pt(lms, ank_i, w, h), (61, 217, 255), 3,
                 cv2.LINE_AA)
        cv2.circle(img, _pt(lms, knee_i, w, h), 8, (255, 170, 60), 2, cv2.LINE_AA)
    return _caption(img, lines)


def refine_event(ev, refine) -> bool:
    """HD refinement of one event's landmarks, in place.

    ``refine(jpeg_bytes, box)`` re-detects the pose on a native-resolution crop
    of the event frame with the (heavy) still-image model and returns
    ``{"lms", "fm", "lms_crop", "crop_img"}``; the landmarks come back in
    full-frame coordinates and the metrics are recomputed from them.  Returns
    True when the event was upgraded.
    """
    if refine is None:
        return False
    hi = ev.get("hi")
    box = ev.get("hi_box")
    if hi is None or box is None:
        return False
    try:
        ref = refine(hi, box)
    except Exception:
        ref = None
    if not ref or ref.get("lms") is None:
        return False
    ev["lms"] = ref["lms"]
    ev["fm"] = {**(ev.get("fm") or {}), **(ref.get("fm") or {})}
    ev["_ref_crop"] = ref.get("lms_crop")
    ev["_ref_img"] = ref.get("crop_img")
    return True


def save_events(snaps, out_dir, refine=None) -> list:
    """Write one annotated JPEG per latched event into out_dir/keyframes/.

    ``refine`` (optional): see :func:`refine_event` — upgrades the retained
    keyframes to native-resolution landmarks before they are drawn, and writes
    an additional zoomed lower-body still per refined keyframe.
    """
    import cv2
    from pose_engine import draw_skeleton, overlay_angles
    kdir = out_dir / config.KEYFRAME_DIR
    kdir.mkdir(parents=True, exist_ok=True)
    meta = []
    for n, ev in enumerate(snaps):
        frame = ev.get("frame")
        if frame is None:
            continue
        # landmarks are optional: without them the keyframe still carries the
        # event caption + metrics, it just cannot draw the skeleton overlay
        try:
            arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            continue
        refined = refine_event(ev, refine)
        lms = _as_lms(ev.get("lms"))
        fm = ev.get("fm") or {}
        img = draw_skeleton(img, lms)
        if lms is not None:
            try:
                overlay_angles(img, lms, fm)
            except Exception:
                pass
        img = annotate_event(img, ev)
        name = (f"{n:02d}_{ev.get('type','EV')}_{ev.get('side','x')}"
                f"_t{float(ev.get('t', 0.0)):05.2f}.jpg".replace(" ", "0"))
        path = kdir / name
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        zoom_file = None
        zimg = ev.get("_ref_img")
        zlms = _as_lms(ev.get("_ref_crop"))
        if refined and zimg is not None and zlms is not None:
            try:
                zoom = draw_skeleton(zimg.copy(), zlms)
                overlay_angles(zoom, zlms, fm)
                zev = dict(ev, lms=zlms)
                zoom = annotate_event(zoom, zev)
                zname = f"zoom_{name}"
                cv2.imwrite(str(kdir / zname), zoom,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                zoom_file = f"{config.KEYFRAME_DIR}/{zname}"
            except Exception:
                zoom_file = None
        meta.append({
            "file": f"{config.KEYFRAME_DIR}/{name}",
            "zoom_file": zoom_file,
            "refined": bool(refined),
            "type": ev.get("type"),
            "label": EVENT_LABEL.get(ev.get("type"), ev.get("type")),
            "side": ev.get("side"),
            "t": round(float(ev.get("t", 0.0)), 2),
            "quality": ev.get("quality"),
            "shin_angle_deg": _r(fm.get(SHIN_KEY.get(ev.get("side"), "shin_angle_l"))),
            "knee_angle_deg": _r(fm.get(KNEE_KEY.get(ev.get("side"), "left_knee"))),
            "knee_flexion_deg": _r(180 - float(fm[KNEE_KEY.get(ev.get("side"), "left_knee")]))
            if isinstance(fm.get(KNEE_KEY.get(ev.get("side"), "left_knee")), (int, float)) else None,
            "pelvic_tilt_deg": _r(fm.get("pelvic_tilt_deg")),
            "hip_extension_deg": _r(fm.get(HIPEXT_KEY.get(ev.get("side"), "hip_ext_l"))),
        })
    return meta


def _r(v, nd=1):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)
