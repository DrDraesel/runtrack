"""Calibrate the RunTrack rig from a printed chessboard.

Two cameras, one board, one world frame (the primary camera's optical frame at
capture time).  Produces ``data/calibration.json`` — intrinsics per camera, the
rigid transform between them, and the projection matrices the fusion step uses.

Usage
-----
    python tools/calibrate.py --cam1 usb:4 --cam2 depth --auto --shots 18
    python tools/calibrate.py --cam1 usb:4 --cam2 usb:0 --from-dir data/calibration/shots_20261023_1300

    --cam1 / --cam2   usb:<index> | depth | file:<path>
    --board 9x6       INNER corners (a board printed with 10x7 squares)
    --square 25       square edge in millimetres — measure it, do not assume
    --shots 18        pairs to collect before solving
    --out data/calibration.json

Capture protocol (this is what makes the numbers good, not the solver)
---------------------------------------------------------------------
* Fill a third to a half of each frame with the board; keep all four board
  edges inside the image, including the quiet zone.
* Tilt the board ±30-40° in yaw and pitch between shots; rotate it so the
  corners are sampled, not just the centre.  A flat board held square to the
  camera in every shot calibrates a camera that only works for that one pose.
* Move it across the frame: top-left, bottom-right, near the edges.
* Both cameras must SEE the board in the same shot (the pair is what links
  the views); a board visible to one camera is skipped.

The captures are kept as PNGs next to the JSON, so a re-solve never needs the
board again — only a fresh capture does.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import calibration as C  # noqa: E402
import cameras as CAM  # noqa: E402
import config  # noqa: E402


def parse_source(spec: str) -> dict:
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "usb":
        return {"kind": "usb", "index": int(rest or 0)}
    if kind == "depth":
        return {"kind": "depth"}
    if kind == "file":
        p = Path(rest)
        if not p.exists():
            raise SystemExit(f"file not found: {p}")
        return {"kind": "file", "path": str(p)}
    raise SystemExit(f"bad source {spec!r}; use usb:<i> | depth | file:<path>")


def open_camera(src: dict):
    cap = CAM.open_source(src)
    if cap is None:
        raise SystemExit(f"could not open {src}")
    return cap


def grab(cap):
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def side_by_side(a, b, width=1500):
    h = int(width * 0.5)
    def fit(img):
        s = min(h / img.shape[0], (width // 2) / img.shape[1])
        return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    a, b = fit(a), fit(b)
    out = np.zeros((max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1], 3), np.uint8)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1]:] = b
    return out


def draw(frame, corners, board, label, colour=(0, 255, 0)):
    vis = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(vis, (board.cols, board.rows),
                                  corners.reshape(-1, 1, 2), True)
    cv2.putText(vis, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return vis


def save_shots(shots1, shots2, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (f1, f2) in enumerate(zip(shots1, shots2)):
        cv2.imwrite(str(out_dir / f"cam1_{i:02d}.png"), f1)
        cv2.imwrite(str(out_dir / f"cam2_{i:02d}.png"), f2)


def load_shots(folder: Path):
    f1s = sorted(folder.glob("cam1_*.png"))
    f2s = sorted(folder.glob("cam2_*.png"))
    if not f1s or len(f1s) != len(f2s):
        raise SystemExit(f"{folder}: need matching cam1_*/cam2_* PNGs "
                         f"(found {len(f1s)} and {len(f2s)})")
    shots = []
    for a, b in zip(f1s, f2s):
        shots.append((cv2.imread(str(a)), cv2.imread(str(b))))
    return shots


def solve(frames, board: C.Board, out: Path | None = None) -> int:
    """Detect, solve both intrinsics and the stereo pair, write the JSON."""
    c1 = [C.find_board(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), board) for f1, _ in frames]
    c2 = [C.find_board(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), board) for _, f2 in frames]
    found1, found2 = sum(x is not None for x in c1), sum(x is not None for x in c2)
    print(f"board found: cam1 {found1}/{len(frames)}  cam2 {found2}/{len(frames)}")
    if found1 < 8 or found2 < 8:
        print("not enough detections — retry with the board filling more of the frame")
        return 2

    size1 = (frames[0][0].shape[1], frames[0][0].shape[0])
    size2 = (frames[0][1].shape[1], frames[0][1].shape[0])
    prim = C.solve_intrinsics(c1, board, size1)
    if "error" in prim:
        print("primary:", prim["error"])
        return 2
    sec = C.solve_intrinsics(c2, board, size2)
    if "error" in sec:
        print("second:", sec["error"])
        return 2
    print(f"cam1 intrinsics RMS {prim['rms_px']:.3f} px   cam2 {sec['rms_px']:.3f} px")

    stereo = C.solve_stereo(c1, c2, board, size1, size2,
                            np.array(prim["K"]), np.array(prim["dist"]),
                            np.array(sec["K"]), np.array(sec["dist"]))
    if "error" in stereo:
        print("stereo:", stereo["error"])
        return 2
    print(f"stereo RMS {stereo['rms_px']:.3f} px over {stereo['pairs']} pairs, "
          f"baseline {stereo['baseline_m']:.3f} m")

    second = dict(stereo)
    second.update({"K": sec["K"], "dist": sec["dist"], "image_size": list(size2),
                   "views": sec["views"]})
    doc = C.build_calibration(prim, {"second": second}, board)
    path = C.save(doc, out or (config.DATA_DIR / "calibration.json"))
    verdict = C.quality(doc)
    print(f"wrote {path}")
    print("quality:", "OK" if verdict["ok"] else "CHECK", *verdict["reasons"], sep="\n  ")
    return 0 if verdict["ok"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam1", default="usb:0", help="primary camera (usb:<i>|depth|file:<p>)")
    ap.add_argument("--cam2", default="usb:1", help="second camera")
    ap.add_argument("--board", default="9x6", help="inner corners, e.g. 9x6")
    ap.add_argument("--square", type=float, default=25.0, help="square edge, millimetres")
    ap.add_argument("--shots", type=int, default=18, help="pairs to collect")
    ap.add_argument("--auto", action="store_true", help="capture automatically, then solve")
    ap.add_argument("--from-dir", help="re-solve from saved cam1_*/cam2_* PNGs")
    ap.add_argument("--out", default=None, help="calibration JSON path")
    args = ap.parse_args(argv)

    cols, _, rows = args.board.lower().partition("x")
    board = C.Board(int(cols), int(rows), args.square / 1000.0)
    out = Path(args.out) if args.out else (config.DATA_DIR / "calibration.json")

    if args.from_dir:
        frames = load_shots(Path(args.from_dir))
        print(f"{len(frames)} pairs from {args.from_dir}")
        return solve(frames, board, out)

    cap1 = open_camera(parse_source(args.cam1))
    cap2 = open_camera(parse_source(args.cam2))
    tag = time.strftime("%Y%m%d_%H%M%S")
    shots_dir = config.DATA_DIR / "calibration" / f"shots_{tag}"
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    last = 0.0
    print(__doc__.split("Capture protocol")[0])
    print("SPACE = capture pair   ENTER = solve and save   ESC = quit"
          + ("   (auto mode: capturing)" if args.auto else ""))
    try:
        while True:
            f1, f2 = grab(cap1), grab(cap2)
            if f1 is None or f2 is None:
                time.sleep(0.05)
                continue
            c1 = C.find_board(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), board)
            c2 = C.find_board(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), board)
            vis = side_by_side(draw(f1, c1, board, f"cam1  {len(frames)}/{args.shots}"),
                               draw(f2, c2, board, "cam2"))
            cv2.putText(vis, f"pairs {len(frames)}/{args.shots}", (12, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("RunTrack calibration", vis)

            auto_hit = (args.auto and c1 is not None and c2 is not None
                        and time.time() - last > 0.8 and len(frames) < args.shots)
            key = cv2.waitKey(1) & 0xFF
            capture = (key == 32 and c1 is not None and c2 is not None) or auto_hit
            if capture:
                frames.append((f1.copy(), f2.copy()))
                last = time.time()
                print(f"captured {len(frames)}/{args.shots}")
                if len(frames) >= args.shots:
                    break
            if key == 27:
                print("aborted")
                return 1
            if key == 13:
                break
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

    if len(frames) < 8:
        print(f"only {len(frames)} pairs — need at least 8")
        return 2
    save_shots([a for a, _ in frames], [b for _, b in frames], shots_dir)
    print(f"shots saved to {shots_dir}")
    return solve(frames, board, out)


if __name__ == "__main__":
    raise SystemExit(main())
