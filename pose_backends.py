"""Pluggable pose backends for RunTrack — optional RTMPose (via rtmlib).

The app's native backend is MediaPipe (``pose_engine.PoseEngine``).  This module
adds an alternative backend that runs RTMPose through ``rtmlib`` and implements
the same duck-typed contract, so it can be dropped in wherever a PoseEngine is
used without touching the callers:

    backend.detect(frame_bgr, timestamp_ms) -> np.ndarray (33, 4) or None
    backend.detect_still(frame_bgr)         -> np.ndarray (33, 4) or None
    backend.seg_frac                        -> None (RTMPose has no mask)
    backend.last_world                      -> None (2-D only, no metric depth)

Every row is ``[x, y, z, visibility]`` with x/y normalised 0..1 to the FULL
frame that was passed in — also when an ROI crop was used internally.  The 33
slots are the MediaPipe layout, because every downstream module (metrics.py,
gait_events.py, side_identity.py) indexes that layout: 11/12 shoulders,
13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles, 29/30 heels,
31/32 foot_index.

Two honest limits of a 2-D RTMPose model versus MediaPipe:

* ``z`` is 0.0 for every slot (HALPE-26 carries no depth), and
* ``last_world`` stays None (no metric 3-D landmarks), so a caller that needs
  metres must keep using the MediaPipe path.

``rtmlib`` is imported lazily: importing this module and constructing the
backend work without the package; only the first frame that needs a model
raises ``RuntimeError`` when it is missing.  Nothing here imports mediapipe, so
the default MediaPipe path is unaffected by the optional dependency.

Environment:
    RUNTRACK_RTMPOSE_MODE    lightweight | balanced | performance (default: lightweight)
    RUNTRACK_RTMPOSE_DEVICE  cpu | cuda | mps                    (default: cpu)
"""
from __future__ import annotations

import math
import os

import numpy as np

# ------------------------------------------------------------------ constants

DEFAULT_MODE = "lightweight"
DEFAULT_DEVICE = "cpu"


def _env(name, default):
    """Environment override as a clean lowercase string, else ``default``."""
    try:
        value = os.environ.get(name)
    except Exception:
        return default
    if value is None:
        return default
    value = str(value).strip().lower()
    return value or default


MODE = _env("RUNTRACK_RTMPOSE_MODE", DEFAULT_MODE)
DEVICE = _env("RUNTRACK_RTMPOSE_DEVICE", DEFAULT_DEVICE)

# Size of the source model's keypoint set (HALPE-26) and of the layout every
# RunTrack consumer indexes (MediaPipe's 33-point pose).  Both tables below are
# checked against these numbers by tests/test_pose_backends.py.
HALPE26_COUNT = 26
MP33_COUNT = 33

# HALPE-26 order as returned by rtmlib.BodyWithFeet.
HALPE26_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "head", "neck", "hip_mid",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)

# MediaPipe-33 slot -> HALPE-26 source index.  One entry per destination slot,
# so ``len(HALPE26_TO_MP33) == MP33_COUNT``.
#
# Slots in DERIVED_MP33 are not real model outputs: HALPE-26 has no eye inner /
# outer, no mouth and no finger points, so they are filled with the nearest
# sensible point (the eye, the nose, the wrist) and their visibility is scaled
# down by DERIVED_VIS_SCALE to mark them as stand-ins.  Every anatomical joint
# the metrics use (11-32 and the ears) maps to its own HALPE point.
HALPE26_TO_MP33 = (
    # face (MediaPipe slot: HALPE source)
    0,    # 0  nose            <- nose
    1,    # 1  left_eye_inner  <- left_eye   (derived)
    1,    # 2  left_eye        <- left_eye
    1,    # 3  left_eye_outer  <- left_eye   (derived)
    2,    # 4  right_eye_inner <- right_eye  (derived)
    2,    # 5  right_eye       <- right_eye
    2,    # 6  right_eye_outer <- right_eye  (derived)
    3,    # 7  left_ear        <- left_ear
    4,    # 8  right_ear       <- right_ear
    0,    # 9  mouth_left      <- nose       (derived)
    0,    # 10 mouth_right     <- nose       (derived)
    # arms
    5,    # 11 left_shoulder   <- left_shoulder
    6,    # 12 right_shoulder  <- right_shoulder
    7,    # 13 left_elbow      <- left_elbow
    8,    # 14 right_elbow     <- right_elbow
    9,    # 15 left_wrist      <- left_wrist
    10,   # 16 right_wrist     <- right_wrist
    # hands (derived from the wrist — HALPE-26 has no finger keypoints)
    9,    # 17 left_pinky
    10,   # 18 right_pinky
    9,    # 19 left_index
    10,   # 20 right_index
    9,    # 21 left_thumb
    10,   # 22 right_thumb
    # legs
    11,   # 23 left_hip        <- left_hip
    12,   # 24 right_hip       <- right_hip
    13,   # 25 left_knee       <- left_knee
    14,   # 26 right_knee      <- right_knee
    15,   # 27 left_ankle      <- left_ankle
    16,   # 28 right_ankle     <- right_ankle
    24,   # 29 left_heel       <- left_heel
    25,   # 30 right_heel      <- right_heel
    20,   # 31 left_foot_index <- left_big_toe
    21,   # 32 right_foot_index<- right_big_toe
)

# MediaPipe slots filled from a neighbour instead of their own HALPE point.
DERIVED_MP33 = frozenset({1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22})
# A derived slot is a stand-in, never an independent measurement: halving its
# confidence keeps it under the usual 0.5 visibility thresholds (so it is not
# drawn as a landmark of its own) without inventing a new number.
DERIVED_VIS_SCALE = 0.5


# --------------------------------------------------------------- box geometry

def _norm_box(box):
    """Clamp ``box`` to a normalized (x0, y0, x1, y1), or None when unusable."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except Exception:
        return None
    vals = (x0, y0, x1, y1)
    if not all(math.isfinite(v) for v in vals):
        return None
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    if (x1 - x0) < 1e-4 or (y1 - y0) < 1e-4:
        return None
    return (x0, y0, x1, y1)


def _expand_box(box, pad, min_span):
    """Grow a normalized box by ``pad`` of its span and to ``min_span``."""
    x0, y0, x1, y1 = box
    pad = max(0.0, float(pad))
    if pad:
        px = (x1 - x0) * pad
        py = (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    span = max(1e-3, float(min_span))
    if (x1 - x0) < span:
        mid = (x0 + x1) * 0.5
        x0, x1 = mid - span * 0.5, mid + span * 0.5
    if (y1 - y0) < span:
        mid = (y0 + y1) * 0.5
        y0, y1 = mid - span * 0.5, mid + span * 0.5
    return _norm_box((x0, y0, x1, y1))


def _is_full_frame(box):
    """True when the box covers the whole frame (nothing to crop off)."""
    return (box[0] <= 0.0 and box[1] <= 0.0
            and box[2] >= 1.0 and box[3] >= 1.0)


def box_from_landmarks(lms, pad=0.15, min_span=0.05, vis_min=0.2):
    """Normalized (x0, y0, x1, y1) around a (33, 4) landmark array, or None.

    Used to auto-pad the ROI crop from the previous detection.  Mirrors
    ``pose_engine.person_crop_box`` but takes the already-mapped 33-slot array
    and never raises.
    """
    if lms is None:
        return None
    try:
        arr = np.asarray(lms, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] < 1:
        return None
    try:
        sel = arr[:, 3] >= float(vis_min)
    except Exception:
        return None
    if int(sel.sum()) < 6:
        return None
    xs = arr[sel, 0]
    ys = arr[sel, 1]
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if (x1 - x0) < float(min_span) or (y1 - y0) < float(min_span):
        return None
    return _expand_box(_norm_box((x0, y0, x1, y1)), pad, min_span)


# ------------------------------------------------------------- keypoint mapping

def map_halpe26_to_mp33(kps_px, scores=None, width=None, height=None, box=None):
    """Map one person's HALPE-26 pixels onto the 33-slot MediaPipe layout.

    ``kps_px`` is (26, 2) or (people, 26, 2) in PIXELS of the image the model
    saw — the first person is used, like ``PoseEngine.detect``.  ``scores`` is
    (26,) or (people, 26) and becomes the visibility column.  ``width`` /
    ``height`` are that image's pixel size, so the result is normalized.

    ``box`` is the normalized (x0, y0, x1, y1) crop the image came from; when
    given, the result is mapped back to full-frame coordinates.  Its z column
    is 0.0 (a 2-D model has no depth) and derived slots get their donor's
    coordinates with the confidence scaled by DERIVED_VIS_SCALE.

    Returns float32 (33, 4) or None when there is nothing usable.
    """
    if kps_px is None:
        return None
    try:
        k = np.asarray(kps_px, dtype=np.float32)
    except Exception:
        return None
    s = None
    if scores is not None:
        try:
            s = np.asarray(scores, dtype=np.float32)
        except Exception:
            s = None
    if k.ndim == 3:                       # (people, 26, 2) -> first person
        if k.shape[0] < 1:
            return None
        k = k[0]
        if s is not None and s.ndim >= 2 and s.shape[0] >= 1:
            s = s[0]
    if k.ndim != 2 or k.shape[1] < 2 or k.shape[0] < 1:
        return None
    try:
        w = float(width) if width is not None else 0.0
        h = float(height) if height is not None else 0.0
    except Exception:
        return None
    if not (math.isfinite(w) and math.isfinite(h) and w > 0.0 and h > 0.0):
        return None

    conf = np.ones((k.shape[0],), dtype=np.float32) if s is None else s.reshape(-1)
    if conf.size < k.shape[0]:
        conf = np.concatenate([conf, np.ones((k.shape[0] - conf.size,), np.float32)])
    if conf.size == 0:
        conf = np.ones((k.shape[0],), dtype=np.float32)

    xs = k[:, 0].astype(np.float64) / w
    ys = k[:, 1].astype(np.float64) / h
    if box is not None:
        rect = _norm_box(box)
        if rect is None:
            raise ValueError("box must be a normalized (x0, y0, x1, y1) box")
        bx0, by0, bx1, by1 = rect
        xs = bx0 + xs * (bx1 - bx0)
        ys = by0 + ys * (by1 - by0)

    out = np.zeros((MP33_COUNT, 4), dtype=np.float32)
    for dst, src in enumerate(HALPE26_TO_MP33):
        if src is None or src < 0 or src >= k.shape[0]:
            continue
        vis = float(conf[src])
        if not math.isfinite(vis):
            vis = 0.0
        vis = min(max(vis, 0.0), 1.0)
        if dst in DERIVED_MP33:
            vis *= DERIVED_VIS_SCALE
        out[dst] = (float(xs[src]), float(ys[src]), 0.0, vis)
    return out


# -------------------------------------------------------------------- backend

def _import_rtmlib():
    """Import rtmlib lazily, raising a RuntimeError that says what to do."""
    try:
        import rtmlib  # noqa: F401  (optional dependency, imported on purpose)
    except Exception as exc:
        raise RuntimeError(
            "the RTMPose pose backend needs the optional 'rtmlib' package "
            "(pip install rtmlib onnxruntime). MediaPipe remains the default: "
            "set RUNTRACK_POSE_BACKEND=mediapipe, or install rtmlib to use "
            "'rtmpose'."
        ) from exc
    return rtmlib


def is_available():
    """True when rtmlib can be imported (no model is loaded or downloaded)."""
    try:
        _import_rtmlib()
    except RuntimeError:
        return False
    return True


class RTMPoseBackend:
    """RTMPose body-with-feet detector behind the PoseEngine contract.

    ``mode`` / ``device`` default to the module ``MODE`` / ``DEVICE`` (which in
    turn read ``RUNTRACK_RTMPOSE_MODE`` / ``RUNTRACK_RTMPOSE_DEVICE``).
    ``model`` may be injected (tests, a pre-built rtmlib model); when it is
    None the rtmlib model is built on the first detection, so constructing a
    backend never downloads anything.

    ``roi`` is off by default.  Pass a normalized (x0, y0, x1, y1) box — or
    ``"auto"`` / True — to run on a cropped region instead of the whole frame:
    the seed box is used for the first frame, after that the crop auto-pads
    (``roi_pad``) around the previous detection and is re-centred every frame,
    which keeps the athlete at native pixel density.  Keypoints are mapped back
    to full-frame normalized coordinates either way, and a frame where the crop
    finds nothing is retried once on the full frame (no lock-out).
    """

    name = "rtmpose"
    model_backend = "onnxruntime"

    def __init__(self, mode=None, device=None, model=None, roi=None,
                 roi_pad=0.15, roi_min_span=0.05, min_score=0.0,
                 model_path=None, seg=False, num_poses=1):
        # ``model_path``/``seg``/``num_poses`` exist only so callers written for
        # PoseEngine (the app, tools/analyze_video.py) can pass the same kwargs
        # to any backend. RTMPose picks its own ONNX model and has no
        # segmentation, so they are accepted and ignored — never silently
        # repurposed.
        self.mode = str(mode or MODE)
        self.device = str(device or DEVICE)
        self.roi_pad = float(roi_pad)
        self.roi_min_span = float(roi_min_span)
        self.min_score = float(min_score)
        self._model = model

        self.roi = False
        self._roi_seed = None
        if roi is None or roi is False:
            self.roi = False
        elif roi is True:
            self.roi = True
        elif isinstance(roi, str):
            word = roi.strip().lower()
            if word not in ("auto", "on", "true"):
                raise ValueError("roi must be None, False, 'auto' or a "
                                 "normalized (x0, y0, x1, y1) box")
            self.roi = True
        else:
            box = _norm_box(roi)
            if box is None:
                raise ValueError("roi must be None, False, 'auto' or a "
                                 "normalized (x0, y0, x1, y1) box")
            self.roi = True
            self._roi_seed = box

        # Kept as attributes (never a change to detect()'s return type) so
        # every existing caller keeps working; RTMPose cannot fill them.
        self.seg_frac = None
        self.last_world = None
        # HALPE confidences of the most recent successful inference, (26,).
        self.last_scores = None
        # Normalized full-frame box actually used by the last detect(); None
        # means the full frame was used.
        self.last_box = None
        self._prev_box = None

    # -- model ------------------------------------------------------------
    @property
    def label(self):
        return f"{self.name} ({self.mode}/{self.device})"

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        rtmlib = _import_rtmlib()
        cls = getattr(rtmlib, "BodyWithFeet", None)
        if cls is None:
            raise RuntimeError(
                "rtmlib is installed but exposes no BodyWithFeet class — "
                "install a newer rtmlib (pip install -U rtmlib)"
            )
        try:
            try:
                self._model = cls(mode=self.mode, backend=self.model_backend,
                                  device=self.device)
            except TypeError:
                # older rtmlib has no backend= kwarg (it defaults to onnxruntime)
                self._model = cls(mode=self.mode, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"could not create rtmlib.BodyWithFeet(mode={self.mode!r}, "
                f"device={self.device!r}): {exc}"
            ) from exc
        return self._model

    def _infer(self, image):
        """Run the model on one image; returns (pixels, scores) or None."""
        model = self._ensure_model()
        try:
            out = model(image)
        except Exception as exc:
            raise RuntimeError(f"rtmlib inference failed: {exc}") from exc
        if out is None:
            return None
        if isinstance(out, (tuple, list)) and len(out) == 2:
            kps, scores = out
        else:                                  # a model that returns keypoints only
            kps, scores = out, None
        if kps is None:
            return None
        try:
            k = np.asarray(kps, dtype=np.float32)
        except Exception:
            return None
        if k.size == 0 or k.ndim < 2:
            return None
        s = None
        if scores is not None:
            try:
                s = np.asarray(scores, dtype=np.float32)
            except Exception:
                s = None
        if k.ndim == 3:
            if k.shape[0] < 1:
                return None
            k = k[0]
            if s is not None and s.ndim >= 2 and s.shape[0] >= 1:
                s = s[0]
        if k.ndim != 2 or k.shape[1] < 2 or k.shape[0] < 1:
            return None
        if s is not None:
            s = s.reshape(-1)
        if self.min_score > 0.0:
            best = 1.0 if s is None or s.size == 0 else float(np.nanmax(s))
            if not (best >= self.min_score):
                return None
        self.last_scores = s
        return k, s

    # -- detection --------------------------------------------------------
    def _full_frame(self, frame):
        got = self._infer(frame)
        if got is None:
            return None
        h, w = frame.shape[:2]
        return map_halpe26_to_mp33(got[0], got[1], w, h)

    def _run(self, frame_bgr, update_roi):
        try:
            frame = np.asarray(frame_bgr)
        except Exception:
            frame = None
        if frame is None or frame.ndim < 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
            self.seg_frac = None
            self.last_world = None
            self.last_box = None
            return None
        h, w = frame.shape[:2]

        arr = None
        used = None
        box = self.next_roi_box() if self.roi else None
        if box is not None and _is_full_frame(box):
            box = None                         # nothing to crop off
        if box is not None:
            px0 = min(max(int(round(box[0] * w)), 0), max(w - 1, 0))
            py0 = min(max(int(round(box[1] * h)), 0), max(h - 1, 0))
            px1 = min(max(int(round(box[2] * w)), px0 + 1), w)
            py1 = min(max(int(round(box[3] * h)), py0 + 1), h)
            crop = frame[py0:py1, px0:px1]
            if crop.size and crop.shape[0] > 0 and crop.shape[1] > 0:
                got = self._infer(crop)
                if got is not None:
                    # The effective box comes from the integer slice, so the
                    # mapping is exact w.r.t. the pixels the model actually saw.
                    used = (px0 / w, py0 / h, px1 / w, py1 / h)
                    arr = map_halpe26_to_mp33(got[0], got[1],
                                              crop.shape[1], crop.shape[0],
                                              box=used)
        if arr is None:
            arr = self._full_frame(frame)      # no ROI, or the crop found nothing
            used = None

        self.seg_frac = None
        self.last_world = None
        self.last_box = used
        if arr is None:
            return None
        if self.roi and update_roi:
            grown = box_from_landmarks(arr, pad=self.roi_pad,
                                       min_span=self.roi_min_span)
            if grown is not None:
                self._prev_box = grown
        return arr

    def detect(self, frame_bgr, timestamp_ms: int = 0):
        """Detect one BGR frame; returns (33, 4) float32 or None.

        ``timestamp_ms`` is accepted for PoseEngine compatibility and ignored:
        RTMPose is a per-frame model (it does not need monotonic timestamps).
        """
        return self._run(frame_bgr, update_roi=True)

    def detect_still(self, frame_bgr):
        """Stateless detection on one image; never touches the ROI history."""
        return self._run(frame_bgr, update_roi=False)

    # -- roi --------------------------------------------------------------
    def next_roi_box(self):
        """Normalized box the next detect() crops to, or None when ROI is off.

        Before the first detection this is the seed box (or the full frame for
        ``roi='auto'``).
        """
        if not self.roi:
            return None
        if self._prev_box is not None:
            return self._prev_box
        if self._roi_seed is not None:
            return self._roi_seed
        return (0.0, 0.0, 1.0, 1.0)

    def reset_roi(self):
        """Forget the auto-padded box and go back to the seed (or full frame)."""
        self._prev_box = None

    def close(self):
        """Drop the model; rtmlib sessions need no explicit shutdown."""
        self._model = None

    def __repr__(self):
        return (f"RTMPoseBackend(mode={self.mode!r}, device={self.device!r}, "
                f"roi={self.roi})")


def make_backend(which=None, **kwargs):
    """Build a pose backend by name.

    ``which`` defaults to ``config.POSE_BACKEND`` (env ``RUNTRACK_POSE_BACKEND``,
    default ``"mediapipe"``).  Extra kwargs go to the backend constructor, e.g.
    ``make_backend("rtmpose", roi="auto")`` or
    ``make_backend("mediapipe", model_path=path)``.
    """
    if which is None:
        which = "mediapipe"
        try:
            import config
            which = getattr(config, "POSE_BACKEND", "mediapipe")
        except Exception:
            pass
    name = str(which).strip().lower()
    if name in ("rtmpose", "rtm", "rtmlib"):
        return RTMPoseBackend(**kwargs)
    if name in ("mediapipe", "mp"):
        from pose_engine import PoseEngine      # lazy: mediapipe is the default
        return PoseEngine(**kwargs)
    raise ValueError(f"unknown pose backend {which!r} "
                     "(expected 'mediapipe' or 'rtmpose')")
