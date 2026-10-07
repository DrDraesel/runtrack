"""Tests for the keyframe HD refinement pass (pose_engine crop helpers +
gait_events.refine_event / save_events with a refiner)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import cv2  # noqa: E402

import gait_events as GA  # noqa: E402
from pose_engine import map_crop_landmarks, person_crop_box  # noqa: E402


def _lms(points, vis=0.9):
    """(33,4) frame: zeroed, only the listed joints are visible."""
    arr = np.zeros((33, 4), dtype=np.float32)
    for idx, (x, y, z) in points.items():
        arr[idx, 0], arr[idx, 1], arr[idx, 2], arr[idx, 3] = x, y, z, vis
    return arr


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


PERSON = {11: (0.40, 0.15, 0.0), 12: (0.44, 0.17, 0.0),
          23: (0.42, 0.45, 0.0), 24: (0.46, 0.46, 0.0),
          25: (0.41, 0.60, 0.0), 26: (0.45, 0.61, 0.0),
          27: (0.35, 0.80, 0.0), 28: (0.39, 0.81, 0.0),
          31: (0.25, 0.88, 0.0), 32: (0.55, 0.89, 0.0)}


class TestCropGeometry(unittest.TestCase):
    def test_person_crop_box(self):
        lms = _lms(PERSON)
        box = person_crop_box(lms, pad=0.12)
        self.assertIsNotNone(box)
        x0, y0, x1, y1 = box
        # x span 0.25..0.55 -> padded by 12 % of 0.30 = 0.036
        self.assertAlmostEqual(x0, 0.25 - 0.036, places=4)
        self.assertAlmostEqual(x1, 0.55 + 0.036, places=4)
        self.assertAlmostEqual(y0, 0.15 - 0.12 * (0.89 - 0.15), places=4)
        self.assertGreater(y1, 0.89)
        self.assertGreaterEqual(x0, 0.0)
        self.assertLessEqual(y1, 1.0)

    def test_crop_box_none_when_not_enough_points(self):
        lms = _lms({11: (0.5, 0.5, 0.0), 12: (0.52, 0.5, 0.0)})
        self.assertIsNone(person_crop_box(lms))
        self.assertIsNone(person_crop_box(None))

    def test_map_crop_landmarks(self):
        lms_crop = _lms({27: (0.5, 0.5, 0.1), 28: (0.25, 0.75, -0.2)})
        out = map_crop_landmarks(lms_crop, (0.2, 0.1, 0.6, 0.9))
        self.assertAlmostEqual(float(out[27, 0]), 0.2 + 0.5 * 0.4, places=6)
        self.assertAlmostEqual(float(out[27, 1]), 0.1 + 0.5 * 0.8, places=6)
        self.assertAlmostEqual(float(out[27, 2]), 0.1 * 0.4, places=6)
        self.assertAlmostEqual(float(out[28, 0]), 0.2 + 0.25 * 0.4, places=6)
        self.assertAlmostEqual(float(out[28, 2]), -0.2 * 0.4, places=6)
        self.assertIsNone(map_crop_landmarks(None, (0, 0, 1, 1)))


class TestRefineEvent(unittest.TestCase):
    def _ev(self):
        lms = _lms(PERSON)
        return {"frame": _jpeg(np.full((80, 120, 3), 100, np.uint8)),
                "hi": b"jpegbytes", "hi_box": [0.1, 0.1, 0.9, 0.9],
                "lms": lms, "fm": {"left_knee": 150.0, "extra": 1},
                "type": "IC", "side": "left", "t": 1.0, "quality": 1.5}

    def test_refine_applies(self):
        ev = self._ev()
        ref_lms = _lms({27: (0.4, 0.8, 0.0), 25: (0.4, 0.6, 0.0)}, vis=0.8)

        def refine(hi, box):
            self.assertEqual(box, [0.1, 0.1, 0.9, 0.9])
            return {"lms": ref_lms, "fm": {"left_knee": 140.0, "shin_angle_l": 4.0},
                    "lms_crop": ref_lms, "crop_img": np.full((60, 60, 3), 50, np.uint8)}

        self.assertTrue(GA.refine_event(ev, refine))
        self.assertIs(ev["lms"], ref_lms)
        self.assertEqual(ev["fm"]["left_knee"], 140.0)      # refined value wins
        self.assertEqual(ev["fm"]["extra"], 1)              # old values kept

    def test_refine_failures_are_soft(self):
        ev = self._ev()
        self.assertFalse(GA.refine_event(dict(ev, hi=None), lambda *a: (_ for _ in ()).throw(RuntimeError())))
        self.assertFalse(GA.refine_event(ev, None))
        self.assertFalse(GA.refine_event(ev, lambda *a: None))
        self.assertFalse(GA.refine_event(ev, lambda *a: (_ for _ in ()).throw(RuntimeError())))

    def test_save_events_with_refiner(self):
        ev = self._ev()
        ref_lms = _lms({27: (0.4, 0.8, 0.0), 25: (0.4, 0.6, 0.0),
                        29: (0.42, 0.81, 0.0), 31: (0.44, 0.82, 0.0)}, vis=0.85)

        def refine(hi, box):
            return {"lms": ref_lms,
                    "fm": {"left_knee": 141.0, "shin_angle_l": 3.5},
                    "lms_crop": ref_lms, "crop_img": np.full((60, 60, 3), 50, np.uint8)}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta = GA.save_events([ev], out, refine=refine)
        self.assertEqual(len(meta), 1)
        m = meta[0]
        self.assertTrue(m["refined"])
        self.assertIsNotNone(m["zoom_file"])
        self.assertEqual(m["knee_angle_deg"], 141.0)
        self.assertEqual(m["shin_angle_deg"], 3.5)

    def test_save_events_without_refiner_still_works(self):
        ev = self._ev()
        ev.pop("hi")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta = GA.save_events([ev], out)          # no refiner at all
            files = list((out / "keyframes").glob("*.jpg"))
        self.assertEqual(len(meta), 1)
        self.assertFalse(meta[0]["refined"])
        self.assertIsNone(meta[0]["zoom_file"])
        self.assertEqual(len(files), 1)
        self.assertIsNotNone(meta[0]["knee_angle_deg"])


if __name__ == "__main__":
    unittest.main()
