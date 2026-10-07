"""MediaPipe PoseLandmarker wrapper + skeleton/angle drawing for RunTrack."""
from __future__ import annotations

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# Standard 33-point pose model landmarks we care about
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Connection pairs for drawing (subset of full 33-point topology, body only)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
LEFT_IDX = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 7}
RIGHT_IDX = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 8}
FACE_IDX = {0, 1, 2, 3, 4, 5, 6, 9, 10}

CLR_LEFT = (107, 107, 255)     # BGR
CLR_RIGHT = (196, 205, 78)
CLR_CENTER = (61, 217, 255)
CLR_FACE = (180, 180, 180)


def _try_pose_connections():
    try:
        conns = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        return [(c.start, c.end) for c in conns]
    except Exception:
        return None


class PoseEngine:
    """Thin wrapper around the MediaPipe tasks PoseLandmarker.

    ``running_mode`` ``"VIDEO"`` keeps the model's internal temporal tracking
    between frames (the live path).  ``"IMAGE"`` makes the instance stateless —
    use it for one-off stills (keyframe HD refinement) so its state cannot be
    disturbed by / disturb the live stream.
    """

    def __init__(self, model_path, num_poses=1,
                 min_det=0.5, min_pres=0.5, min_track=0.5, running_mode="VIDEO",
                 seg=False):
        self.video_mode = str(running_mode).upper() != "IMAGE"
        self.seg_enabled = bool(seg)
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO if self.video_mode else RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_pres,
            min_tracking_confidence=min_track,
            output_segmentation_masks=self.seg_enabled,
        )
        self._lm = PoseLandmarker.create_from_options(opts)
        self._last_ts = 0
        # mask share of the frame from the most recent detection (None when
        # segmentation is off or no pose was found) — used for frontal-area
        self.seg_frac = None

    def _seg_frac(self, res) -> float | None:
        if not self.seg_enabled:
            return None
        try:
            masks = getattr(res, "segmentation_masks", None)
            if not masks:
                return None
            arr = masks[0].numpy_view()
            if arr is None or getattr(arr, "size", 0) == 0:
                return None
            return float((arr > 0.5).mean())
        except Exception:
            return None

    def _array(self, pose) -> np.ndarray:
        arr = np.zeros((len(pose), 4), dtype=np.float32)
        for i, p in enumerate(pose):
            arr[i, 0] = p.x
            arr[i, 1] = p.y
            arr[i, 2] = p.z
            vis = getattr(p, "visibility", None)
            arr[i, 3] = 1.0 if vis is None else float(vis)
        return arr

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Run pose detection on one BGR frame (VIDEO mode).

        Returns np.ndarray (33, 4) of [x, y, z, visibility] normalized coords
        for the first pose, or None if no pose found. Raises RuntimeError on
        mediapipe failure.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self._lm.detect_for_video(mp_img, ts)
        self.seg_frac = self._seg_frac(res)
        if not res.pose_landmarks:
            return None
        return self._array(res.pose_landmarks[0])

    def detect_still(self, frame_bgr: np.ndarray):
        """Stateless detection on a single image (IMAGE mode engines).

        Coordinates are normalized to the image that was passed in.
        """
        if self.video_mode:
            raise RuntimeError("detect_still needs a PoseEngine(running_mode='IMAGE')")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        res = self._lm.detect(mp_img)
        self.seg_frac = self._seg_frac(res)
        if not res.pose_landmarks:
            return None
        return self._array(res.pose_landmarks[0])

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


def _pt(lms, idx, w, h):
    return int(lms[idx, 0] * w), int(lms[idx, 1] * h)


# ------------------------------------------------------- crop / refine geometry

def person_crop_box(lms, pad=0.12, min_span=0.15, vis_min=0.2):
    """Normalized (x0, y0, x1, y1) box around the visible person, or None.

    Used to crop a full-resolution frame for the keyframe HD refinement pass:
    the model then sees the athlete at native pixel density instead of the
    downscaled processing frame.
    """
    if lms is None:
        return None
    try:
        arr = np.asarray(lms, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < 33:
        return None
    sel = arr[:33, 3] >= vis_min
    if int(sel.sum()) < 6:
        return None
    xs = arr[:33, 0][sel]
    ys = arr[:33, 1][sel]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if (x1 - x0) < min_span or (y1 - y0) < min_span:
        return None
    px = (x1 - x0) * float(pad)
    py = (y1 - y0) * float(pad)
    return (max(0.0, x0 - px), max(0.0, y0 - py),
            min(1.0, x1 + px), min(1.0, y1 + py))


def map_crop_landmarks(lms_crop, box):
    """Map landmarks normalized to a crop back to full-frame coordinates."""
    if lms_crop is None or box is None:
        return None
    x0, y0, x1, y1 = (float(b) for b in box)
    out = np.array(lms_crop, dtype=np.float32, copy=True)
    out[:, 0] = x0 + out[:, 0] * (x1 - x0)
    out[:, 1] = y0 + out[:, 1] * (y1 - y0)
    out[:, 2] = out[:, 2] * (x1 - x0)          # depth scales with the crop width
    return out


def make_crop_refiner(model_path=None):
    """HD keyframe refiner: still-image pose model on native-res crops.

    Returns ``(engine, refine)`` — ``refine(jpeg_bytes, box)`` re-detects the
    pose on the crop and returns ``{"lms", "fm", "lms_crop", "crop_img"}`` in
    full-frame coordinates — or None when refinement is disabled
    (``config.REFINE_KEYFRAMES``).  The engine is a separate, stateless
    IMAGE-mode instance (the heavy model when present) so the HD pass can never
    disturb a live VIDEO-mode tracker.  The caller closes the engine.
    """
    import config
    import metrics          # lazy: metrics imports this module

    if not config.REFINE_KEYFRAMES:
        return None
    path = model_path or (config.MODEL_PATH_HEAVY
                          if config.MODEL_PATH_HEAVY.exists() else config.MODEL_PATH)
    engine = PoseEngine(path, running_mode="IMAGE")

    def refine(hi_jpeg, box):
        try:
            arr = np.frombuffer(hi_jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            lms_crop = engine.detect_still(img)
            if lms_crop is None:
                return None
            lms_full = map_crop_landmarks(lms_crop, box)
            return {"lms": lms_full, "fm": metrics.frame_metrics(lms_full),
                    "lms_crop": lms_crop, "crop_img": img}
        except Exception:
            return None

    return engine, refine


def draw_skeleton(img: np.ndarray, lms, vis_min=0.5):
    """Draw the tracked skeleton. Returns img (in place)."""
    if lms is None:
        return img
    h, w = img.shape[:2]
    conns = _try_pose_connections() or CONNECTIONS
    conns = [c for c in conns if c[0] <= 32 and c[1] <= 32]
    for a, b in conns:
        if lms[a, 3] < vis_min or lms[b, 3] < vis_min:
            continue
        if a in FACE_IDX and b in FACE_IDX:
            continue
        color = CLR_LEFT if (a in LEFT_IDX and b in LEFT_IDX) else \
                CLR_RIGHT if (a in RIGHT_IDX and b in RIGHT_IDX) else CLR_CENTER
        cv2.line(img, _pt(lms, a, w, h), _pt(lms, b, w, h), color, 2, cv2.LINE_AA)
    for i in range(min(33, len(lms))):
        if lms[i, 3] < vis_min:
            continue
        if i in FACE_IDX:
            color, r = CLR_FACE, 2
        elif i in LEFT_IDX:
            color, r = CLR_LEFT, 4
        elif i in RIGHT_IDX:
            color, r = CLR_RIGHT, 4
        else:
            color, r = CLR_CENTER, 4
        cv2.circle(img, _pt(lms, i, w, h), r, color, -1, cv2.LINE_AA)
    return img


def overlay_angles(img: np.ndarray, lms, fm: dict, color=(255, 255, 255)):
    """Draw key joint angles near their vertices using frame metrics."""
    if lms is None or not fm:
        return img
    h, w = img.shape[:2]
    shown = [("left_knee", 25), ("right_knee", 26),
             ("left_hip", 23), ("right_hip", 24),
             ("left_elbow", 13), ("right_elbow", 14),
             ("left_ankle", 27), ("right_ankle", 28)]
    for name, idx in shown:
        v = fm.get(name)
        if v is None or lms[idx, 3] < 0.5:
            continue
        x, y = _pt(lms, idx, w, h)
        txt = f"{v:.0f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        px = min(max(x + 8, 0), w - tw - 4)
        py = min(max(y - 6, th), h - 4)
        cv2.putText(img, txt, (px + 1, py + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return img
