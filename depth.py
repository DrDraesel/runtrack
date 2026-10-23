"""RealSense D455F depth + colour source for RunTrack.

This is the app's first depth consumer.  ``DepthSource`` is a drop-in stand-in
for the ``cv2.VideoCapture`` objects ``cameras.open_source`` returns: the same
``read`` / ``get`` / ``set`` / ``isOpened`` / ``release`` surface, plus depth
sampling and a floor-plane pitch estimate.

What is measured (on this rig, D455F over USB 2.1)
--------------------------------------------------
* Colour: the aligned BGR colour stream (bgr8) -- the same pixels the pose
  tracker sees through the camera's UVC colour pin.
* Depth: the z16 depth stream aligned to the colour frame, converted to
  float32 metres with ``depth_frame.get_units()`` (0.001 m per unit on this
  unit).  0 means "no measurement" (out of range, shadow, invalid pixel).
* Depth at 640x480@30 delivers about 20 fps with ~85% valid pixels; colour
  intrinsics are fx = fy = 391.1 (brown_conrady).
* Depth noise grows roughly with the square of distance: expect a per-frame
  RMS of about 3-4 cm at 2.5-3 m.  A plane fit over thousands of samples
  averages *random* noise down; it cannot remove systematic depth bias, and it
  is only as good as the surface it locks onto -- a treadmill deck, mat or wall
  inside the fitting band can dominate.  Treat the pitch as a setup aid, not a
  calibration-grade measurement, and check it against a level before trusting
  sagittal angles.
* Synthetic check with these intrinsics (floor 1.2 m below the camera, pitch
  20 deg): 3 cm RMS random depth noise at 3 m moves the fitted pitch by under
  0.05 deg over ~140k inliers, but a systematic depth offset of 1 mm biases it
  by ~0.17 deg (about 1 deg per 6 mm).  The sensor's depth *offset* error, not
  its per-frame noise, bounds the pitch accuracy: treat +-1 deg as the
  realistic floor.  One 640x480 fit costs ~10 ms, hence the 1 s cache.

Conventions
-----------
Camera frame (the RealSense optical frame), metres, no re-projection:

* x -> right in the image
* y -> down in the image
* z -> forward along the optical axis (this is the depth value)

A point at pixel (u, v) with depth z back-projects to
``((u - cx) * z / fx, (v - cy) * z / fy, z)``.

Plane: ``normal . p + d = 0`` in that camera frame.  The normal is unit length
and oriented with ``nz >= 0`` (``nz > 0`` for any camera pitched down, the
normal pointing away from the camera towards the floor); when ``nz`` is
(almost) exactly 0 the tie-break is ``ny >= 0``.  ``d`` is negative while the
normal points at the floor, and then ``|d|`` is the camera height; the
orientation flip carries both signs together, so an upward-pitched camera
yields ``d > 0``.

Pitch: ``camera_pitch_from_plane`` returns degrees the optical axis is pitched
*below* horizontal (positive = down, negative = up).  It is derived from the
floor normal with no roll assumption: see its docstring.

pyrealsense2 is optional.  Importing this module never fails without it; the
capture class then reports ``ok = False`` with an error string so callers can
fall back to a plain colour camera.  The pure helpers work on any machine.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence

import numpy as np

try:  # pyrealsense2 is optional: the pure helpers and tests must run without it
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - depends on the machine
    rs = None

__all__ = [
    "CAP_PROP_POS_MSEC",
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FPS",
    "DEFAULT_BAND",
    "DepthSource",
    "camera_pitch_from_plane",
    "fit_floor_plane",
    "px_per_m",
    "sample_depth",
]

# cv2 CAP_PROP ids as plain ints, so cv2 is not a dependency and callers can
# pass cv2.CAP_PROP_* unchanged (the numeric values are part of cv2's ABI).
CAP_PROP_POS_MSEC = 0
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5

#: Fraction of the image height used for the floor fit: 0.55 = lower 45%.
DEFAULT_BAND = (0.55, 1.0)
#: Residual (m) inside which a point counts as an inlier of the floor plane.
TRIM_M = 0.02
#: A plane needs at least this many valid depth points ...
MIN_POINTS = 200
#: ... and at least this share of the band, or the fit is rejected.
MIN_INLIER_PCT = 20.0
#: floor_pitch_deg() recomputes at most this often.
FLOOR_PITCH_CACHE_S = 1.0


def _short(exc: BaseException) -> str:
    """One-line, bounded error text for the UI/fallback path."""
    text = " ".join(str(exc).split())
    return text[:160]


def sample_depth(depth_m: np.ndarray | None, nx: float, ny: float,
                 r: int = 4) -> float | None:
    """Median depth in metres around a normalised image point.

    ``depth_m`` is a float32 HxW array of metres with 0 = invalid.  ``nx``/``ny``
    are 0..1 image coordinates (0 = left/top edge, 1 = right/bottom edge) and
    are mapped to the nearest pixel.  The window is the (2r+1)^2 box around it,
    clipped at the image border; ``r=0`` samples a single pixel.

    Returns the median of the finite values > 0 in that window, or None when
    the pixel is outside 0..1, the array is empty, or nothing in the window is
    valid.  The median is a measured value: no interpolation, no hole filling.
    """
    if depth_m is None:
        return None
    arr = np.asarray(depth_m)
    if arr.ndim != 2 or arr.size == 0:
        return None
    # NaN fails these comparisons, so non-finite coordinates land here too.
    if not (0.0 <= nx <= 1.0) or not (0.0 <= ny <= 1.0):
        return None
    h, w = arr.shape
    u = int(round(float(nx) * (w - 1)))
    v = int(round(float(ny) * (h - 1)))
    rr = max(0, int(r))
    window = arr[max(0, v - rr): min(h, v + rr + 1),
                 max(0, u - rr): min(w, u + rr + 1)]
    vals = window[np.isfinite(window) & (window > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def px_per_m(fx: float, distance_m: float) -> float | None:
    """Pixels per metre at a distance, from the focal length in pixels.

    ``fx`` is the horizontal focal length (391.1 on the D455F colour stream at
    640x480), ``distance_m`` the distance along the optical axis.  Returns
    ``fx / distance_m``, or None for non-positive or non-finite inputs.  The
    reciprocal is metres per pixel, so 1 cm/px at 3.9 m falls out as ~100.
    """
    try:
        f = float(fx)
        d = float(distance_m)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(f) and np.isfinite(d)) or f <= 0.0 or d <= 0.0:
        return None
    return f / d


def _plane_lsq(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Total-least-squares plane through points; (unit normal, d) or None.

    Solves the eigenproblem of the 3x3 covariance (smallest eigenvector =
    normal) -- deterministic, unlike a random RANSAC sample.  Returns the
    normal in an arbitrary +/- sense plus d with ``normal . p + d = 0``.
    """
    n_pts = int(points.shape[0])
    if n_pts < 3:
        return None
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = (centred.T @ centred) / max(1, n_pts - 1)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    # A collinear (or single-point) cloud has no plane: second spread ~ 0.
    if not np.isfinite(evals).all() or evals[1] <= 1e-12:
        return None
    normal = evecs[:, 0]
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    normal = normal / norm
    return normal, -float(normal @ centroid)


def fit_floor_plane(depth_m: np.ndarray | None, intrinsics: dict,
                    band: tuple[float, float] = DEFAULT_BAND) -> dict | None:
    """Fit the dominant plane in the lower part of a depth image.

    ``intrinsics`` needs fx, fy, cx, cy, width, height (the aligned colour
    intrinsics).  ``band`` is the fraction of image *height* used: the default
    (0.55, 1.0) is the lower 45%, where a floor-mounted camera sees the floor.

    Every valid pixel in the band is back-projected with the pinhole model,
    then fitted deterministically: least squares (3x3 covariance eigenvector),
    followed by two rounds of trimming to points within ``TRIM_M`` (2 cm) of
    the plane, refitting on the survivors.  No random sampling, so identical
    input gives identical output.

    Returns ``{"normal": (nx, ny, nz), "d": float, "rms_m": float,
    "inlier_pct": float, "points": int}``:

    * ``normal`` -- unit normal with ``nz >= 0`` (tie-break ``ny >= 0``); see
      the module docstring for the full convention.
    * ``d`` -- offset in ``normal . p + d = 0``.  Negative while the normal
      points at the floor (any downward-pitched camera), where ``|d|`` is the
      camera height in metres; flipping the normal for the ``nz >= 0`` rule
      flips ``d`` too.
    * ``rms_m`` -- RMS residual of the inliers in metres (a *fit* residual, not
      the sensor's absolute accuracy).
    * ``inlier_pct`` -- percentage of the valid band points within 2 cm.
    * ``points`` -- number of valid depth points the fit considered.

    Returns None when fewer than 200 valid points exist, or when the band has
    no dominant plane (under 20% of points within 2 cm of the best fit, e.g.
    pure noise or heavy clutter).  The truthfulness of the result depends on
    the floor actually being the dominant surface in the band.

    At 0 deg pitch a level camera only sees the floor tens of metres away, so
    real (range-limited) depth data has few floor pixels there; the estimate is
    most useful with the camera tilted down 15-30 deg, where the band sits
    ~1.5-3.5 m away.
    """
    if depth_m is None or not intrinsics:
        return None
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        return None

    arr = np.asarray(depth_m)
    if arr.ndim != 2 or arr.size == 0:
        return None
    h, w = arr.shape

    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError, IndexError):
        return None
    lo = min(max(lo, 0.0), 1.0)
    hi = min(max(hi, 0.0), 1.0)
    y0 = int(round(lo * h))
    y1 = int(round(hi * h))
    if y1 <= y0:
        return None

    sub = arr[y0:y1]
    valid = np.isfinite(sub) & (sub > 0)
    rows, cols = np.nonzero(valid)
    n_pts = int(rows.size)
    if n_pts < MIN_POINTS:
        return None

    z = sub[rows, cols].astype(np.float64)
    v = (rows + y0).astype(np.float64)
    u = cols.astype(np.float64)
    pts = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)

    fit = _plane_lsq(pts)
    if fit is None:
        return None
    normal, d = fit
    inliers = np.ones(n_pts, dtype=bool)
    for _ in range(2):
        residual = np.abs(pts @ normal + d)
        keep = residual < TRIM_M
        if int(keep.sum()) < MIN_POINTS:
            return None
        if np.array_equal(keep, inliers):
            break
        inliers = keep
        refit = _plane_lsq(pts[inliers])
        if refit is None:
            return None
        normal, d = refit

    residuals = np.abs(pts[inliers] @ normal + d)
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else float("inf")
    inlier_pct = 100.0 * float(inliers.sum()) / float(n_pts)

    if int(inliers.sum()) < MIN_POINTS or inlier_pct < MIN_INLIER_PCT:
        return None
    if not np.isfinite(rms):
        return None

    # Orient consistently: nz >= 0, tie-break ny >= 0 at the horizon.
    if normal[2] < 0.0 or (abs(normal[2]) < 1e-12 and normal[1] < 0.0):
        normal = -normal
    centroid = pts[inliers].mean(axis=0)
    d = -float(normal @ centroid)
    return {
        "normal": (float(normal[0]), float(normal[1]), float(normal[2])),
        "d": d,
        "rms_m": rms,
        "inlier_pct": inlier_pct,
        "points": n_pts,
    }


def camera_pitch_from_plane(normal: np.ndarray | Sequence[float] | None) -> float | None:
    """Camera pitch in degrees (positive = optical axis aimed down).

    Derivation, with a camera that has no roll and a horizontal floor
    ``height_m`` below it.  In the camera frame (x right, y down, z along the
    optical axis), pitching the camera down by ``theta`` leaves the right axis
    alone and gives the optical axis ``(0, -sin theta, cos theta)`` and the
    image-down axis ``(0, -cos theta, -sin theta)``.  A camera point
    ``(X, Y, Z)`` then has world height ``-Y cos theta - Z sin theta``, so the
    floor (height ``-height_m``) is the plane

        ``Y cos(theta) + Z sin(theta) = height_m``
        ``n . p + d = 0`` with ``n = (0, cos theta, sin theta)``, ``d = -height_m``.

    Therefore ``tan(theta) = nz / ny`` and ``theta = atan2(nz, ny)`` for the
    floor normal pointing away from the camera (``ny > 0``).  The +/-n
    ambiguity is removed by flipping to ``ny >= 0`` first, which the visible
    floor always satisfies; for a straight-down view (``ny == 0``) the ``nz>0``
    orientation gives +90.  A level camera that sees the floor therefore reads
    ~0, and one aimed 30 deg down reads ~+30.

    Roll (``nx != 0``) is ignored: the result is the tilt in the camera's y-z
    plane, i.e. the pitch component about the camera's x axis.  A wall or a
    steeply inclined surface fitted by mistake yields a large angle rather than
    a plausible one -- check the magnitude before trusting it.

    Returns None for a missing, non-finite or zero-length normal.
    """
    try:
        n = np.asarray(normal, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(n)):
        return None
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return None
    n = n / norm
    if n[1] < 0.0 or (n[1] == 0.0 and n[2] < 0.0):
        n = -n
    return float(np.degrees(np.arctan2(n[2], n[1])))


class DepthSource:
    """Live D455F colour + aligned depth, presenting a cv2-like capture API.

    ``__init__`` never raises: if pyrealsense2 is missing, no device is found,
    or the pipeline cannot start, it returns with ``ok = False`` and a short
    ``error`` string so the caller can fall back to the UVC colour camera.

    A daemon thread pulls framesets continuously and keeps only the newest
    aligned colour frame (BGR) and aligned depth frame (float32 metres); the
    camera's own frame queue is never allowed to build up.  ``read()`` hands
    out the newest colour frame once, like ``cv2.VideoCapture.read``, but
    without blocking: a frame already seen is never returned twice.

    Threading: the producer only ever *replaces* the ``_latest``/``_depth``
    references (never mutates an array in place), so readers can use the
    returned arrays freely.  Colour frames are copied out of the SDK buffer;
    depth is a fresh array from the float conversion.

    If the stream dies after a successful start (unplug, USB reset), ``ok``
    flips to False and ``error`` says why; callers should then ``release()``
    and fall back to the UVC colour camera.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 warmup: int = 8) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.ok = False
        self.error = ""

        self._lock = threading.Lock()
        self._pitch_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipe = None
        self._align = None
        self._latest: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intr: dict | None = None
        self._frames = 0
        self._times: deque[float] = deque(maxlen=31)
        self._pitch_deg: float | None = None
        self._pitch_ts = 0.0

        if rs is None:
            self.error = "pyrealsense2 is not installed"
            return
        try:
            self._open(max(0, int(warmup)))
        except Exception as exc:  # noqa: BLE001 - init must never raise
            self.error = _short(exc) or "could not open the RealSense depth pipeline"
            self._close_pipe()
            self.ok = False

    # -- setup ---------------------------------------------------------
    def _open(self, warmup: int) -> None:
        """Start the pipeline, settle it with ``warmup`` frames, start the thread.

        At least one frame must arrive during warm-up (``warmup=0`` still pulls
        one): a pipeline that starts but never delivers is reported as a
        failure so callers fall back instead of reading an empty source.
        """
        sdk = rs
        if sdk is None:
            self.error = "pyrealsense2 is not installed"
            return
        ctx = sdk.context()
        if len(ctx.query_devices()) == 0:
            self.error = "no RealSense device found"
            return

        pipe = sdk.pipeline()
        config = sdk.config()
        try:
            config.enable_stream(sdk.stream.depth, self.width, self.height,
                                 sdk.format.z16, int(self.fps))
            config.enable_stream(sdk.stream.color, self.width, self.height,
                                 sdk.format.bgr8, int(self.fps))
        except Exception:
            # The requested mode may not exist at this link speed; the defaults
            # always do.  intrinsics report what was actually negotiated.
            config = sdk.config()
            config.enable_stream(sdk.stream.depth, sdk.format.z16, int(self.fps))
            config.enable_stream(sdk.stream.color, sdk.format.bgr8, int(self.fps))

        self._pipe = pipe
        self._align = sdk.align(sdk.stream.color)
        pipe.start(config)

        grabbed = 0
        for _ in range(max(1, warmup)):
            if self._grab():
                grabbed += 1
        if grabbed == 0:
            self.error = "depth pipeline started but delivered no frames"
            self._close_pipe()
            return

        # ok first, so a thread that dies immediately can flip it back to False.
        self.ok = True
        self._thread = threading.Thread(target=self._run, name="realsense-depth",
                                        daemon=True)
        self._thread.start()

    def _close_pipe(self) -> None:
        pipe = self._pipe
        self._pipe = None
        self._align = None
        if pipe is not None:
            try:
                pipe.stop()
            except Exception:
                pass

    # -- producer ------------------------------------------------------
    def _grab(self) -> bool:
        """Pull one frameset; store the aligned colour + depth frames."""
        pipe = self._pipe
        align = self._align
        if pipe is None or align is None:
            return False
        frames = pipe.wait_for_frames(2000)
        if frames is None or frames.size() == 0:
            return False
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return False

        scale = float(depth_frame.get_units())  # metres per z16 unit
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
        # Copy: the SDK buffer behind get_data() dies with the frameset.
        colour = np.asanyarray(color_frame.get_data()).copy()

        intr = None
        profile = color_frame.get_profile().as_video_stream_profile()
        if profile is not None:
            i = profile.get_intrinsics()
            # pyrealsense2 names the principal point ppx/ppy (cx/cy in the
            # OpenCV convention); accept either name so a wrapper that renames
            # them cannot take the whole depth source down at runtime.
            cx = float(getattr(i, "ppx", getattr(i, "cx", 0.0)))
            cy = float(getattr(i, "ppy", getattr(i, "cy", 0.0)))
            intr = {"fx": float(i.fx), "fy": float(i.fy),
                    "cx": cx, "cy": cy,
                    "width": int(i.width), "height": int(i.height)}

        now = time.perf_counter()
        with self._lock:
            self._latest = colour
            self._depth = depth
            if intr is not None:
                self._intr = intr
            self._frames += 1
            self._times.append(now)
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._grab()
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set():
                    break
                with self._lock:
                    self.error = _short(exc) or "RealSense stream stopped"
                self.ok = False
                break

    # -- cv2-compatible surface ---------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Newest aligned BGR frame, consumed once; (False, None) if none yet."""
        with self._lock:
            frame = self._latest
            self._latest = None
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop) -> float:
        """cv2.CAP_PROP_* lookup, read-only.

        FRAME_WIDTH / FRAME_HEIGHT report the negotiated size once intrinsics
        are known (otherwise the requested size); FPS reports the requested
        rate (the measured rate is in ``depth_stats``); POS_MSEC and anything
        else return 0.0 because a live stream cannot seek.
        """
        try:
            p = int(prop)
        except (TypeError, ValueError):
            return 0.0
        if p == CAP_PROP_FPS:
            return float(self.fps)
        if p in (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT):
            with self._lock:
                intr = self._intr
            if intr is not None:
                key = "width" if p == CAP_PROP_FRAME_WIDTH else "height"
                return float(int(intr[key]))
            return float(self.width if p == CAP_PROP_FRAME_WIDTH else self.height)
        return 0.0

    def set(self, prop, value) -> bool:
        """No-op: a live source cannot be resized or seeked.  Always False."""
        return False

    def isOpened(self) -> bool:
        return bool(self.ok)

    def release(self) -> None:
        """Stop the thread and the pipeline; safe to call more than once."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=2.5)
        self._close_pipe()
        self.ok = False

    # -- depth ---------------------------------------------------------
    @property
    def intrinsics(self) -> dict | None:
        """Aligned colour-stream intrinsics {fx,fy,cx,cy,width,height} or None."""
        with self._lock:
            intr = self._intr
        return dict(intr) if intr is not None else None

    def latest_depth(self) -> np.ndarray | None:
        """Newest float32 depth frame in metres aligned to the colour frame (0=invalid)."""
        with self._lock:
            return self._depth

    def depth_at(self, nx: float, ny: float, r: int = 4) -> float | None:
        """Median depth in metres around a normalised colour-image point."""
        depth = self.latest_depth()
        if depth is None:
            return None
        return sample_depth(depth, nx, ny, r)

    def depth_stats(self) -> dict:
        """{"valid_pct", "median_m", "fps", "frames"} for the newest depth frame.

        Empty shape before the first frame: every value None and frames 0.
        ``fps`` is measured from frame arrival times (not the requested rate),
        and ``valid_pct`` counts non-zero, finite pixels.
        """
        with self._lock:
            depth = self._depth
            frames = int(self._frames)
            times = list(self._times)
        stats: dict = {"valid_pct": None, "median_m": None, "fps": None,
                       "frames": frames}
        if len(times) >= 2 and times[-1] > times[0]:
            stats["fps"] = round((len(times) - 1) / (times[-1] - times[0]), 1)
        if depth is not None and depth.size:
            valid = np.isfinite(depth) & (depth > 0)
            n_valid = int(valid.sum())
            if n_valid:
                stats["valid_pct"] = round(100.0 * n_valid / float(depth.size), 1)
                stats["median_m"] = round(float(np.median(depth[valid])), 3)
        return stats

    def floor_pitch_deg(self) -> float | None:
        """Camera pitch in degrees below horizontal, cached for 1 s.

        Positive = optical axis aimed down.  None when there is no depth frame,
        no intrinsics, or no dominant floor plane in the lower 45% of the
        image.  Computed at most once per second regardless of call rate.
        """
        now = time.monotonic()
        with self._pitch_lock:
            if self._pitch_ts > 0.0 and (now - self._pitch_ts) < FLOOR_PITCH_CACHE_S:
                return self._pitch_deg
            # Stamp first so a slow fit cannot be re-entered by every caller.
            self._pitch_ts = now
            pitch = None
            depth = self.latest_depth()
            intr = self.intrinsics
            if depth is not None and intr is not None:
                try:
                    plane = fit_floor_plane(depth, intr, DEFAULT_BAND)
                    if plane is not None:
                        pitch = camera_pitch_from_plane(plane["normal"])
                except Exception:
                    pitch = None
            self._pitch_deg = pitch
            return pitch

    def px_per_m(self, distance_m: float) -> float | None:
        """Pixels per metre at a distance, using this camera's fx."""
        intr = self.intrinsics
        if intr is None:
            return None
        return px_per_m(intr["fx"], distance_m)  # module-level helper


# ---------------------------------------------------------------------------
# Hardware smoke test: never imported by the unit tests.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    src = DepthSource()
    try:
        print(f"ok={src.ok} error={src.error!r}")
        if src.ok:
            for _ in range(30):
                ok, frame = src.read()
                if ok and frame is not None:
                    print(f"colour frame {frame.shape} {frame.dtype}")
                    break
                time.sleep(0.1)
            print(f"intrinsics={src.intrinsics}")
            print(f"stats={src.depth_stats()}")
            print(f"depth at centre={src.depth_at(0.5, 0.5)} m")
            print(f"pixels per metre at 2.5 m={src.px_per_m(2.5)}")
            print(f"floor pitch={src.floor_pitch_deg()} deg")
    finally:
        src.release()
