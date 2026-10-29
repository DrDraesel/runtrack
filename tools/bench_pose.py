"""Benchmark the MediaPipe and RTMPose pose backends on one video file.

Usage:
  python tools/bench_pose.py VIDEO [--backend both|mediapipe|rtmpose]
        [--provider auto|cpu|directml|cuda] [--limit N] [--repeat N]
        [--proc-width 960] [--mode lightweight] [--device cpu] [--roi] [--json]

  python tools/bench_pose.py assets/test_run.mp4 --backend rtmpose \
        --provider directml --limit 119 --repeat 3

Frames go through the same preprocessing the live app uses: downscale to
``config.PROC_WIDTH`` wide (never upscale), then ``metrics.frame_metrics(lms,
aspect=w/h)`` on the result — so a green bench also proves the pose is
consumable by the metric layer.  The metric call is outside the pose timing.

The first detect of each backend pays the model build (rtmlib builds or
downloads its ONNX model there, MediaPipe creates the landmarker up front); it
is reported as "first frame" and excluded from the fps.  ``--repeat`` runs the
clip several times and reports the median, which is what you want when the
difference between two providers is small.

Prints one line per backend: frames processed, frames with a pose, lower-body
coverage, seconds, fps and the ONNX Runtime provider that actually ran (for
RTMPose).  A backend whose library — or, for MediaPipe, whose model file — is
missing is skipped with the reason instead of failing the run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import metrics as M  # noqa: E402
from pose_backends import backend_info  # noqa: E402

LOWER_BODY = (25, 26, 27, 28)          # knees + ankles in the MediaPipe layout


def _iter_frames(path, limit, proc_width):
    """Yield app-preprocessed BGR frames from ``path``.

    ``limit`` <= 0 reads the whole clip.  ``proc_width`` <= 0 keeps native size.
    """
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
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            n += 1
            yield frame
    finally:
        cap.release()


def _make_mediapipe(args):
    try:
        import pose_backends
        from pose_engine import PoseEngine  # noqa: F401  (presence check)
    except Exception as exc:                       # mediapipe not installed
        raise RuntimeError(f"mediapipe unavailable: {exc}") from exc
    model = config.MODEL_PATH
    if not model.exists():
        raise RuntimeError(f"MediaPipe model missing: {model}")
    # built through make_backend, exactly like the app, so ``info`` is attached
    return pose_backends.make_backend("mediapipe", model_path=str(model),
                                      seg=False)


def _make_rtmpose(args):
    try:
        import pose_backends
    except Exception as exc:
        raise RuntimeError(f"pose_backends unavailable: {exc}") from exc
    if not pose_backends.is_available():
        raise RuntimeError("rtmlib not installed (pip install rtmlib onnxruntime)")
    return pose_backends.make_backend(
        "rtmpose",
        model_path=str(config.MODEL_PATH),
        mode=args.mode,
        device=args.device,
        provider=args.provider,
        roi="auto" if args.roi else None,
    )


def _lower_body(lms):
    """True when both knees and both ankles clear config.VIS_MIN."""
    if lms is None:
        return False
    try:
        return all(float(lms[i, 3]) >= config.VIS_MIN for i in LOWER_BODY)
    except Exception:
        return False


def _finite(value):
    """True when ``value`` is a real number (frame_metrics returns None)."""
    try:
        return float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _ankle_usable(fm):
    """gait_events.lower_body_coverage's per-frame test: an ankle y exists."""
    return _finite(fm.get("l_ankle_y")) or _finite(fm.get("r_ankle_y"))


def _warmup(engine, video, proc_width):
    """First detect: builds the model.  Returns its cost in ms."""
    for frame in _iter_frames(video, 1, proc_width):
        t0 = time.perf_counter()
        engine.detect(frame, 1)
        return (time.perf_counter() - t0) * 1000.0
    return 0.0


def _pass(engine, video, args, pass_index):
    """One timed pass; returns frames/pose hits/seconds for that pass."""
    frames = posed = lower = ankles = metrics_ok = 0
    elapsed = 0.0
    for frame in _iter_frames(video, args.limit, args.proc_width):
        # monotonic across passes and independent of the clip rate
        ts = 1 + pass_index * 1_000_000 + int(round(frames * 1000.0 / 30.0))
        t0 = time.perf_counter()
        lms = engine.detect(frame, ts)
        elapsed += time.perf_counter() - t0
        frames += 1
        if lms is not None:
            posed += 1
            if _lower_body(lms):
                lower += 1
        fm = M.frame_metrics(lms, aspect=frame.shape[1] / frame.shape[0])
        if fm:
            metrics_ok += 1
            if _ankle_usable(fm):
                ankles += 1
    return {"frames": frames, "pose_frames": posed, "lower_body_frames": lower,
            "ankle_frames": ankles, "metric_frames": metrics_ok,
            "seconds": elapsed,
            "fps": (frames / elapsed) if elapsed > 0 else 0.0}


def _run(label, make, args, video):
    """Benchmark one backend; returns its result dict, or None when skipped."""
    try:
        engine = make(args)
    except Exception as exc:
        print(f"{label:<22} skipped: {exc}")
        return None
    info = dict(backend_info(engine))
    try:
        first_ms = _warmup(engine, video, args.proc_width)
        passes = [_pass(engine, video, args, i) for i in range(args.repeat)]
    except Exception as exc:
        print(f"{label:<22} failed: {exc}")
        return None
    finally:
        try:
            engine.close()
        except Exception:
            pass

    last = passes[-1]
    if last["frames"] <= 0:
        print(f"{label:<22} no frames decoded from {video}")
        return None
    result = {
        "backend": label,
        "engine": info.get("engine"),
        "provider": info.get("provider"),
        "model": info.get("model"),
        "first_frame_ms": round(first_ms, 1),
        "frames": last["frames"],
        "pose_frames": last["pose_frames"],
        "lower_body_frames": last["lower_body_frames"],
        "ankle_frames": last["ankle_frames"],
        "metric_frames": last["metric_frames"],
        "pose_pct": round(100.0 * last["pose_frames"] / last["frames"], 1),
        "ankle_pct": round(100.0 * last["ankle_frames"] / last["frames"], 1),
        "lower_body_pct": round(100.0 * last["lower_body_frames"] / last["frames"], 1),
        "seconds": round(statistics.median([p["seconds"] for p in passes]), 3),
        "fps": round(statistics.median([p["fps"] for p in passes]), 1),
        "fps_passes": [round(p["fps"], 1) for p in passes],
        "passes": args.repeat,
    }
    print(f"{label:<22} {str(result['provider'] or '-'):<22} "
          f"{result['frames']:>5} frames  pose {result['pose_frames']:>4}"
          f" ({result['pose_pct']:>5.1f}%)  ankle {result['ankle_pct']:>5.1f}%"
          f"  legs {result['lower_body_pct']:>5.1f}%  "
          f"{result['seconds']:>7.3f} s  {result['fps']:>6.1f} fps"
          f"  first {result['first_frame_ms']:>6.0f} ms")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark MediaPipe vs RTMPose pose backends on a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file to run the backend(s) on")
    parser.add_argument("--backend", choices=("both", "mediapipe", "rtmpose"),
                        default="both")
    parser.add_argument("--provider", default=getattr(config, "ORT_PROVIDER", "auto"),
                        choices=("auto", "cpu", "directml", "cuda"),
                        help="ONNX Runtime provider for rtmpose "
                             "(default: config.ORT_PROVIDER)")
    parser.add_argument("--limit", "--frames", dest="limit", type=int, default=0,
                        help="process at most N frames (0 = whole clip)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="timed passes over the clip; the median is reported")
    parser.add_argument("--proc-width", type=int, default=config.PROC_WIDTH,
                        help=f"downscale frames to this width first "
                             f"(default: {config.PROC_WIDTH} = the app's; 0 = native)")
    parser.add_argument("--mode", default=None,
                        help="rtmlib model size: lightweight|balanced|performance "
                             "(default RUNTRACK_RTMPOSE_MODE or lightweight)")
    parser.add_argument("--device", default=None,
                        help="rtmlib device string: cpu|cuda|mps "
                             "(default RUNTRACK_RTMPOSE_DEVICE or cpu)")
    parser.add_argument("--roi", action="store_true",
                        help="RTMPose: crop to the auto-padded previous detection")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object instead of the table")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        args.repeat = 1

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    jobs = []
    if args.backend in ("both", "mediapipe"):
        jobs.append(("mediapipe (VIDEO)", _make_mediapipe))
    if args.backend in ("both", "rtmpose"):
        jobs.append(("rtmpose" + (" +roi" if args.roi else ""), _make_rtmpose))

    if not args.json:
        print(f"{video} | limit {args.limit or 'all'} | width "
              f"{args.proc_width or 'native'} | repeat {args.repeat} | "
              f"requested provider {args.provider!r}")
        print(f"{'backend':<22} {'provider':<22} {'frames':>11}  {'pose':>17}  "
              f"{'ankle':>11}  {'legs':>10}  {'seconds':>9}  {'fps':>9}  "
              f"{'first frame':>14}")

    results = []
    for label, make in jobs:
        got = _run(label, make, args, video)
        if got is not None:
            results.append(got)
    if not results:
        print("no backend could run", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"clip": str(video), "limit": args.limit,
                          "proc_width": args.proc_width,
                          "provider_requested": args.provider,
                          "results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
