"""Multi-camera calibration for the RunTrack rig (P2).

The rig analyses the same runner from two or three cameras, but every view is
currently reduced on its own: there is no shared 3D frame, so "the pelvis
rotates relative to the thorax" or "the left arm is out of phase with the right
leg" cannot be measured across views.  This module produces the missing piece:
one calibration per camera plus the rigid transform that puts every view in the
same world frame, in metres.

Conventions
-----------
* World frame = the **primary camera's optical frame** at calibration time.
  Camera 2/3 extrinsics follow the OpenCV stereo convention::

      X_cam2 = R @ X_world + t

  so ``R`` and ``t`` are exactly what ``cv2.stereoCalibrate`` returns when the
  primary camera's own extrinsics are the identity.
* Camera frame: x right, y down, z forward (the usual optical frame).  Depth is
  z, so a triangulated point lands in the same axes ``depth.py`` uses.
* Units are metres everywhere.  The board's square size is the only scale
  reference, so a wrong square size scales the whole 3D skeleton — measure it
  with a caliper, not from the print dialog.

Honest limits (state these with any fused number)
-------------------------------------------------
* Planar target: intrinsics from a flat board are less stable than a 3D target;
  the reported RMS is a *fit* residual and is optimistic at the image edges.
  Keep the board moving across the frame, including the corners.
* The cameras share no hardware trigger.  Frames are matched by the software
  clock, so expect up to ~1 frame of skew — at 22 fps that is ~45 ms, which is
  a few percent of a stride and enough to blur a phase-coupling estimate.  This
  module does not fix that; it only makes it visible.
* Extrinsics drift if a camera is knocked.  Re-run the solve (the captured
  images are kept) and compare RMS to the stored value before trusting a
  session.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# target
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Board:
    """A printed chessboard, counted in INNER corners.

    ``cols`` and ``rows`` are the inner-corner counts (a 9x6 board printed with
    10x7 squares); ``square_m`` is the measured edge length of one square.
    """

    cols: int = 9
    rows: int = 6
    square_m: float = 0.025

    @property
    def n_points(self) -> int:
        return self.cols * self.rows

    def object_points(self) -> np.ndarray:
        """(N, 3) float32 grid on the z=0 plane, origin at the first corner."""
        xs, ys = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        pts = np.zeros((self.n_points, 3), np.float32)
        pts[:, 0] = xs.ravel() * self.square_m
        pts[:, 1] = ys.ravel() * self.square_m
        return pts

    def as_dict(self) -> dict:
        return {"cols": self.cols, "rows": self.rows, "square_m": self.square_m}


def find_board(gray: np.ndarray, board: Board) -> np.ndarray | None:
    """Detect the board's inner corners, sub-pixel, or None.

    ``findChessboardCornersSB`` already returns sub-pixel corners, so no
    ``cornerSubPix`` refinement is needed (and refining SB output tends to
    wander on the small squares of a printed A4 board).
    """
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows), flags)
    if not ok or corners is None or len(corners) != board.n_points:
        return None
    return np.asarray(corners, dtype=np.float32).reshape(-1, 2)


# --------------------------------------------------------------------------
# single camera
# --------------------------------------------------------------------------
def solve_intrinsics(shots: list[np.ndarray], board: Board,
                     image_size: tuple[int, int]) -> dict:
    """Calibrate one camera from detected corner sets.

    ``image_size`` is (width, height).  Returns a JSON-safe dict; on failure the
    ``error`` key says why instead of raising, because calibration is an
    operator action and the UI needs a message, not a traceback.
    """
    shots = [s for s in shots if s is not None and len(s) == board.n_points]
    if len(shots) < 8:
        return {"error": f"need at least 8 usable views, got {len(shots)}"}

    obj = board.object_points()
    objpoints = [obj for _ in shots]
    flags = (cv2.CALIB_ZERO_TANGENT_DIST if hasattr(cv2, "CALIB_ZERO_TANGENT_DIST") else 0)
    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, shots, image_size, None, None, flags=flags)
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
        return {"error": f"calibrateCamera failed: {exc}"}

    per_view = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - shots[i], axis=1)))
        per_view.append(err)

    return {
        "K": np.asarray(K, dtype=float).tolist(),
        "dist": np.asarray(dist, dtype=float).ravel().tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "rms_px": float(rms),
        "views": len(shots),
        "per_view_px": per_view,
        "board": board.as_dict(),
    }


# --------------------------------------------------------------------------
# stereo (primary + one more view)
# --------------------------------------------------------------------------
def solve_stereo(shots1: list[np.ndarray | None], shots2: list[np.ndarray | None],
                 board: Board, size1: tuple[int, int], size2: tuple[int, int],
                 K1: np.ndarray, d1: np.ndarray, K2: np.ndarray, d2: np.ndarray,
                 ) -> dict:
    """Rigid transform between two calibrated cameras.

    Only shots where BOTH cameras saw the board are used.  Returns R, t with
    ``X2 = R @ X1 + t`` (metres), plus the RMS and the number of pairs.
    """
    pairs = [(a, b) for a, b in zip(shots1, shots2)
             if a is not None and b is not None
             and len(a) == board.n_points and len(b) == board.n_points]
    if len(pairs) < 5:
        return {"error": f"need at least 5 paired views, got {len(pairs)}"}

    obj = board.object_points()
    objpoints = [obj for _ in pairs]
    c1 = [p[0] for p in pairs]
    c2 = [p[1] for p in pairs]
    try:
        rms, K1o, d1o, K2o, d2o, R, T, E, F = cv2.stereoCalibrate(
            objpoints, c1, c2, K1.copy(), d1.copy(), K2.copy(), d2.copy(),
            (int(size1[0]), int(size1[1])),
            flags=cv2.CALIB_FIX_INTRINSIC)
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
        return {"error": f"stereoCalibrate failed: {exc}"}

    return {
        "R": np.asarray(R, dtype=float).tolist(),
        "t": np.asarray(T, dtype=float).ravel().tolist(),
        "rms_px": float(rms),
        "pairs": len(pairs),
        "baseline_m": float(np.linalg.norm(T)),
        "K": [np.asarray(K1o, float).tolist(), np.asarray(K2o, float).tolist()],
        "dist": [np.asarray(d1o, float).ravel().tolist(),
                 np.asarray(d2o, float).ravel().tolist()],
    }


def projection_matrix(K, dist, R, t) -> np.ndarray:
    """3x4 P for a camera whose world->camera transform is (R, t).

    Distortion is ignored here: for triangulation only the projection matrix
    matters, and both cameras are already being sampled at undistorted landmark
    positions in the app (the pose pipeline uses normalised coordinates).  When
    the caller has distorted pixels, undistort them first.
    """
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3, 1)
    return K @ np.hstack([R, t])


def triangulate(P1, P2, pts1, pts2) -> np.ndarray:
    """Closed-form (DLT) triangulation of matched image points.

    ``pts1``/``pts2`` are (N, 2) pixel coordinates; the return is (N, 3) metres
    in the world frame.  No outlier rejection is done here — the caller checks
    the reprojection residual, because a bad match silently triangulates to a
    plausible-looking wrong point.
    """
    p1 = np.asarray(pts1, float).reshape(-1, 2).T
    p2 = np.asarray(pts2, float).reshape(-1, 2).T
    if p1.shape != p2.shape:
        raise ValueError("point sets must have the same length")
    hom = cv2.triangulatePoints(np.asarray(P1, float), np.asarray(P2, float), p1, p2)
    hom = hom.reshape(4, -1)
    w = hom[3]
    w[np.abs(w) < 1e-12] = 1e-12
    return (hom[:3] / w).T


def reprojection_error(P: np.ndarray, pts3: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Per-point pixel distance between projected and observed image points."""
    X = np.hstack([np.asarray(pts3, float), np.ones((len(pts3), 1))])
    proj = (np.asarray(P, float) @ X.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - np.asarray(pts2, float).reshape(-1, 2), axis=1)


# --------------------------------------------------------------------------
# file format
# --------------------------------------------------------------------------
def build_calibration(primary: dict, others: dict[str, dict], board: Board) -> dict:
    """Assemble the JSON document the app reads.

    ``primary`` is the camera-0 solution; ``others`` maps a view role
    (``"second"`` / ``"third"``) to that camera's intrinsics plus the stereo
    result from :func:`solve_stereo`.
    """
    doc = {
        "schema": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "units": "metres",
        "world": "primary camera optical frame at calibration time",
        "convention": "X_other = R @ X_world + t",
        "board": board.as_dict(),
        "primary": {k: primary[k] for k in
                    ("K", "dist", "image_size", "rms_px", "views") if k in primary},
        "views": {},
    }
    for role, spec in others.items():
        cam = {k: spec[k] for k in ("K", "dist", "image_size", "rms_px", "views")
               if k in spec}
        cam["R"] = np.asarray(spec["R"], float).tolist()
        cam["t"] = np.asarray(spec["t"], float).tolist()
        cam["stereo_rms_px"] = float(spec.get("rms_px", float("nan")))
        cam["baseline_m"] = float(np.linalg.norm(np.asarray(spec["t"], float)))
        cam["pairs"] = int(spec.get("pairs", 0))
        cam["P"] = projection_matrix(cam["K"], cam["dist"],
                                    cam["R"], cam["t"]).tolist()
        doc["views"][role] = cam
    return doc


def save(doc: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("schema") != SCHEMA_VERSION:
        return None
    return doc


def quality(doc: dict | None) -> dict:
    """Operator-facing verdict on a calibration file.

    Thresholds are deliberately blunt: sub-pixel stereo RMS is the normal
    outcome of a good board session, and a baseline under 0.5 m makes depth
    from triangulation worse than the sensors' own depth stream.
    """
    if not doc:
        return {"ok": False, "reasons": ["no calibration file"]}
    reasons = []
    views = doc.get("views") or {}
    if not views:
        reasons.append("no other views calibrated")
    if doc.get("primary", {}).get("rms_px", 1e9) > 1.0:
        reasons.append(f"primary reprojection RMS {doc['primary']['rms_px']:.2f} px > 1.0")
    for role, cam in views.items():
        if cam.get("stereo_rms_px", 1e9) > 1.0:
            reasons.append(f"{role}: stereo RMS {cam['stereo_rms_px']:.2f} px > 1.0")
        if cam.get("baseline_m", 0.0) < 0.5:
            reasons.append(f"{role}: baseline {cam['baseline_m']:.2f} m < 0.5 m "
                           "(triangulation will be noisy)")
    return {"ok": not reasons, "reasons": reasons,
            "views": sorted(views), "board": doc.get("board")}
