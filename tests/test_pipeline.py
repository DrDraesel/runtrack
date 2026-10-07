"""Pipeline smoke test: pose + metrics on a real test clip.

Skips automatically if assets/test_run.mp4 is missing.
"""
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import cv2  # noqa: E402

import config  # noqa: E402
from pose_engine import PoseEngine, draw_skeleton  # noqa: E402
import metrics as M  # noqa: E402

CLIP = BASE / "assets" / "test_run.mp4"


@unittest.skipUnless(CLIP.exists(), "test clip not present")
class TestPipeline(unittest.TestCase):
    def test_video_pipeline(self):
        cap = cv2.VideoCapture(str(CLIP))
        self.assertTrue(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.set(cv2.CAP_PROP_POS_MSEC, 5500)   # first 5.5 s of this clip has no full body
        engine = PoseEngine(config.MODEL_PATH)
        det, n = 0, 0
        rows = []
        try:
            while n < 100:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                h, w = frame.shape[:2]
                scale = 640 / w if w > 640 else 1.0
                proc = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                vt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                lms = engine.detect(proc, int(vt * 1000))
                fm = M.frame_metrics(lms)
                if lms is not None:
                    det += 1
                    drawn = draw_skeleton(proc.copy(), lms)
                    self.assertEqual(drawn.shape, proc.shape)
                    for k, v in fm.items():
                        if k.startswith("_") or k.endswith(("_x", "_y")):
                            continue
                        if k in M.SIGNED_FRAME_KEYS:
                            # signed metrics: angles/directions that may be
                            # negative (see metrics.SIGNED_FRAME_KEYS)
                            self.assertLessEqual(abs(v), 60.0, k)
                        else:
                            self.assertGreaterEqual(v, 0.0, k)
                            self.assertLessEqual(v, 181.0, k)
                rows.append(fm)
        finally:
            engine.close()
            cap.release()
        self.assertGreater(n, 50, "did not read enough frames")
        self.assertGreater(det, 5, f"pose detected in only {det}/{n} frames")
        summary = M.analyze_session([dict(r, t_rel=i / fps) for i, r in enumerate(rows)])
        self.assertGreater(summary["n_frames"], 50)


if __name__ == "__main__":
    unittest.main()
