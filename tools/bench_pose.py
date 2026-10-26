"""Benchmark the MediaPipe and RTMPose pose backends on one video file.

Usage:
  python tools/bench_pose.py VIDEO [--frames 300] [--proc-width 0] [--roi]
        [--backend both|mediapipe|rtmpose] [--mode lightweight] [--device cpu]

Prints one line per backend: inference fps, keypoint count per detection, how
many frames produced a pose and the cost of the first frame (rtmlib builds or
downloads its ONNX model there, MediaPipe creates its landmarker up front).

A backend whose library — or, for MediaPipe, whose model file — is missing is
skipped with the reason instead of failing the run.  Both backends read the
video independently and see the same frames at the same processing width.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import config  # noqa: E402


def _iter_frames(path, limit, proc_width):
    """Yield BGR frames from ``path`` (optionally downscaled to ``proc_width``)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        n = 0
        while limit <= 0 or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if proc_width and frame.shape[1] > proc_width:
                scale = proc_width / float(frame.shape[1])
                height = max(1, int(round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, (proc_width, height),
                                   interpolation=cv2.INTER_AREA)
            n += 1
            yield frame
    finally:
        cap.release()


def _make_mediapipe(args):
    try:
        from pose_engine import PoseEngine
    except Exception as exc:                       # mediapipe not installed
        raise RuntimeError(f"mediapipe unavailable: {exc}") from exc
    model = config.MODEL_PATH
    if not model.exists():
        raise RuntimeError(f"MediaPipe model missing: {model}")
    return PoseEngine(model, running_mode="VIDEO")


def _make_rtmpose(args):
    try:
        import pose_backends
    except Exception as exc:
        raise RuntimeError(f"pose_backends unavailable: {exc}") from exc
    if not pose_backends.is_available():
        raise RuntimeError("rtmlib not installed (pip install rtmlib onnxruntime)")
    return pose_backends.RTMPoseBackend(
        mode=args.mode or pose_backends.MODE,
        device=args.device or pose_backends.DEVICE,
        roi="auto" if args.roi else None,
    )


def _run(label, make, args, video):
    """Benchmark one backend; returns its fps, or None when it was skipped."""
    try:
        engine = make(args)
    except Exception as exc:
        print(f"{label:<26} skipped: {exc}")
        return None

    total = posed = 0
    kps = 0
    first_ms = None
    elapsed = 0.0
    timed = 0
    try:
        for frame in _iter_frames(video, args.frames, args.proc_width):
            ts = int(round(total * 1000.0 / 30.0))     # monotonic, any rate
            t0 = time.perf_counter()
            lms = engine.detect(frame, ts)
            dt = time.perf_counter() - t0
            total += 1
            if first_ms is None:
                first_ms = dt * 1000.0                # pays the model build
                continue
            elapsed += dt
            timed += 1
            if lms is not None:
                posed += 1
                kps = int(len(lms))
    except Exception as exc:
        print(f"{label:<26} failed: {exc}")
        return None
    finally:
        try:
            engine.close()
        except Exception:
            pass

    if timed <= 0:
        print(f"{label:<26} no frames decoded from {video}")
        return None
    fps = timed / elapsed if elapsed > 0.0 else 0.0
    print(f"{label:<26} {fps:7.1f} fps  {kps:3d} keypoints  "
          f"{posed:4d}/{timed:<4d} frames with pose  "
          f"first frame {first_ms:6.0f} ms")
    return fps


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark MediaPipe vs RTMPose pose backends on a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file to run both backends on")
    parser.add_argument("--frames", type=int, default=300,
                        help="frames to process per backend (0 = whole video)")
    parser.add_argument("--proc-width", type=int, default=0,
                        help="downscale frames to this width first (0 = native)")
    parser.add_argument("--backend", choices=("both", "mediapipe", "rtmpose"),
                        default="both")
    parser.add_argument("--mode", default=None,
                        help="rtmlib model size: lightweight|balanced|performance "
                             "(default RUNTRACK_RTMPOSE_MODE or lightweight)")
    parser.add_argument("--device", default=None,
                        help="rtmlib device: cpu|cuda|mps "
                             "(default RUNTRACK_RTMPOSE_DEVICE or cpu)")
    parser.add_argument("--roi", action="store_true",
                        help="RTMPose: crop to the auto-padded previous detection")
    args = parser.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    print(f"{video} | max {args.frames or 'all'} frames | "
          f"width {args.proc_width or 'native'}")
    print(f"{'backend':<26} {'fps':>10}  keypoints  pose hit            "
          f"first frame")

    jobs = []
    if args.backend in ("both", "mediapipe"):
        jobs.append(("mediapipe (VIDEO)", _make_mediapipe))
    if args.backend in ("both", "rtmpose"):
        label = "rtmpose" + (" +roi" if args.roi else "")
        jobs.append((label, _make_rtmpose))

    ran = 0
    for label, make in jobs:
        if _run(label, make, args, video) is not None:
            ran += 1
    if ran == 0:
        print("no backend could run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
