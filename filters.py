"""Temporal smoothing for RunTrack landmarks.

Implements the 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012) without any
extra dependency.  The filter is a low-pass filter whose cutoff frequency rises
with the observed speed of the signal: it removes the high-frequency jitter of
pose landmarks when the limb is slow, and opens up (low lag) when the limb moves
fast — exactly what a running gait needs, where the ankle can travel ~0.5 body
heights in 100 ms.

Signal chain in RunTrack:

    raw landmarks ──► 1-Euro ──► per-frame metrics ──► session analysis
           └────────────► event detection (raw + filtered ankle velocity)

Parameters live in config.py (ONEEURO_*).
"""
from __future__ import annotations

import math

import numpy as np

import config

# Landmark columns: x, y, z, visibility. Only x/y/z are smoothed.
_COORD_COLS = slice(0, 3)
_VIS_COL = 3


def alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a given cutoff frequency (Hz) and timestep (s)."""
    if cutoff <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Multi-channel 1-Euro filter over arbitrary-shaped float arrays.

    Usage::

        f = OneEuroFilter()
        lms = f(lms_raw, t)          # t = seconds (monotonic)

    `x` is any np.ndarray of float coordinates; the filter keeps one state per
    element.  `reset()` clears the state (call it when the subject is lost or
    after a seek in a video file).
    """

    def __init__(self, min_cutoff: float | None = None, beta: float | None = None,
                 d_cutoff: float | None = None, freq: float | None = None):
        self.min_cutoff = float(config.ONEEURO_MIN_CUTOFF if min_cutoff is None else min_cutoff)
        self.beta = float(config.ONEEURO_BETA if beta is None else beta)
        self.d_cutoff = float(config.ONEEURO_DCUTOFF if d_cutoff is None else d_cutoff)
        self.freq = float(config.ONEEURO_FREQ if freq is None else freq)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    # ------------------------------------------------------------------ state
    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def _dt(self, t) -> float:
        if t is None:
            return 1.0 / max(self.freq, 1e-6)
        if self._t_prev is None:
            self._t_prev = float(t)
            return 1.0 / max(self.freq, 1e-6)
        dt = float(t) - float(self._t_prev)
        if dt <= 1e-6:                     # duplicate or out-of-order timestamp
            dt = 1.0 / max(self.freq, 1e-6)
        else:
            self._t_prev = float(t)
        return dt

    # ------------------------------------------------------------------- call
    def __call__(self, x, t=None, mask=None):
        """Filter one sample.

        x    : array-like of floats (e.g. (33, 4) landmarks)
        t    : timestamp in seconds (optional; falls back to config.ONEEURO_FREQ)
        mask : optional boolean array, True where a value is valid.  Invalid
               entries are passed through untouched and do not update state.
        """
        arr = np.asarray(x, dtype=np.float32)
        if not config.ONEEURO_ENABLED:
            return arr
        flat = arr.reshape(-1).copy()
        if mask is not None:
            valid = np.asarray(mask, dtype=bool).reshape(-1)
        else:
            valid = np.ones(flat.shape[0], dtype=bool)
        if self._x_prev is None or self._x_prev.shape != flat.shape:
            self._x_prev = flat.copy()
            self._dx_prev = np.zeros_like(flat)
            self._t_prev = None if t is None else float(t)
            self._dt(t)
            return arr
        dt = self._dt(t)
        a_d = alpha(self.d_cutoff, dt)
        # derivative (per second) with the previous filtered value
        dx = (flat - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = (1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt))
        x_hat = a * flat + (1.0 - a) * self._x_prev
        out = np.where(valid, x_hat, flat).astype(np.float32)
        self._x_prev = np.where(valid, x_hat, self._x_prev)
        self._dx_prev = np.where(valid, dx_hat, self._dx_prev)
        return out.reshape(arr.shape)


class LandmarkFilter:
    """1-Euro filter specialised for (33, 4) MediaPipe landmark arrays.

    x/y/z are smoothed; visibility (column 3) is passed through unchanged so
    that confidence gating keeps working on raw values.  Low-visibility
    landmarks are still filtered (their coordinates jump around when the model
    is unsure) but the confidence flag tells downstream code not to trust them.
    """

    def __init__(self, **kw):
        self._f = OneEuroFilter(**kw)

    def reset(self):
        self._f.reset()

    def __call__(self, lms, t=None):
        if lms is None:
            # Subject lost: keep the state so the filter does not have to
            # re-converge from scratch, but report "no pose".
            return None
        arr = np.asarray(lms, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            return arr
        mask = np.zeros(arr.shape, dtype=bool)
        mask[:, _COORD_COLS] = True        # smooth x/y/z only
        out = self._f(arr, t=t, mask=mask)
        out[:, _VIS_COL] = arr[:, _VIS_COL]
        return out


def smooth_landmark_series(landmark_list, fps: float | None = None, **kw):
    """Convenience helper: filter a list of (33,4) landmarks at a fixed rate.

    Used by tests and offline re-analysis; the live path uses LandmarkFilter
    directly so that it can pass real timestamps.
    """
    f = LandmarkFilter(**kw)
    rate = float(fps or config.ONEEURO_FREQ)
    out = []
    for i, lms in enumerate(landmark_list):
        if lms is None:
            out.append(None)
            continue
        out.append(f(lms, t=i / max(rate, 1e-6)))
    return out


def signal_smooth(x, t=None, min_cutoff=None, beta=None):
    """Smooth a 1-D signal with the same filter (used for gait signals)."""
    f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    v = np.asarray(x, dtype=float)
    tt = None if t is None else np.asarray(t, dtype=float)
    out = np.empty_like(v)
    for i in range(v.size):
        val = f(np.array([v[i]], dtype=np.float32),
                t=None if tt is None else tt[i])[0]
        out[i] = float(val)
    return out
