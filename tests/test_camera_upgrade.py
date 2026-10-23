"""Camera-upgrade plumbing: depth source, rig check, 3D landmarks, depth metrics.

These tests are hardware-free: the depth stream is stubbed and the API is
exercised through Flask's test client with the real session store swapped to a
temporary directory.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import depth_metrics as DM  # noqa: E402
import metrics as M  # noqa: E402


def lms33():
    """Neutral 33x4 landmark array (x, y, z, visibility), all visible."""
    a = np.zeros((33, 4), dtype=np.float32)
    a[:, 0] = 0.5
    a[:, 1] = 0.5
    a[:, 3] = 1.0
    return a


def world33():
    """World landmarks with a right angle at the left knee."""
    a = np.zeros((33, 4), dtype=np.float32)
    a[:, 3] = 1.0
    a[23] = [0.0, 0.0, 0.0, 1.0]      # left hip
    a[25] = [0.0, 0.45, 0.0, 1.0]     # left knee
    a[27] = [0.0, 0.45, 0.4, 1.0]     # left ankle -> 90 deg at the knee
    a[24] = [0.2, 0.0, 0.0, 1.0]      # right hip
    a[26] = [0.2, 0.45, 0.0, 1.0]
    a[28] = [0.2, 0.9, 0.0, 1.0]      # straight leg -> 180 deg
    return a


INTR = {"fx": 391.1, "fy": 391.1, "cx": 320.0, "cy": 240.0,
        "width": 640, "height": 480}


class TestWorldLandmarks(unittest.TestCase):
    def test_knee_angles_from_world_landmarks(self):
        out = M.world_joint_angles(world33())
        self.assertAlmostEqual(out["knee_l_3d"], 90.0, places=3)
        self.assertAlmostEqual(out["knee_r_3d"], 180.0, places=3)

    def test_missing_or_short_array_is_empty(self):
        self.assertEqual(M.world_joint_angles(None), {})
        self.assertEqual(M.world_joint_angles(np.zeros((5, 4))), {})

    def test_low_visibility_is_skipped(self):
        w = world33()
        w[25, 3] = 0.1
        self.assertNotIn("knee_l_3d", M.world_joint_angles(w))


class TestPointAt(unittest.TestCase):
    def _depth(self, z):
        return lambda nx, ny, r=4: z

    def test_centre_pixel_maps_to_optical_axis(self):
        a = lms33()
        a[27] = [0.5, 0.5, 0.0, 1.0]
        p = DM.point_at(a, 27, self._depth(2.0), INTR)
        self.assertAlmostEqual(p[0], (0.5 * 640 - 320) / 391.1 * 2.0, places=4)
        self.assertAlmostEqual(p[1], (0.5 * 480 - 240) / 391.1 * 2.0, places=4)
        self.assertAlmostEqual(p[2], 2.0, places=6)

    def test_offset_pixel_is_lateral_and_metric(self):
        a = lms33()
        a[27] = [0.6, 0.5, 0.0, 1.0]     # 64 px right of centre
        p = DM.point_at(a, 27, self._depth(2.5), INTR)
        expected = (0.6 * 640 - 320) / 391.1 * 2.5
        self.assertAlmostEqual(p[0], expected, places=4)
        self.assertGreater(p[0], 0)

    def test_rejects_invalid_depth_and_missing_inputs(self):
        a = lms33()
        self.assertIsNone(DM.point_at(a, 27, lambda nx, ny, r=4: None, INTR))
        self.assertIsNone(DM.point_at(a, 27, self._depth(0.0), INTR))
        self.assertIsNone(DM.point_at(a, 27, self._depth(99.0), INTR))
        self.assertIsNone(DM.point_at(None, 27, self._depth(2.0), INTR))
        self.assertIsNone(DM.point_at(a, 27, self._depth(2.0), None))

    def test_invisible_landmark_is_skipped(self):
        a = lms33()
        a[27, 3] = 0.2
        self.assertIsNone(DM.point_at(a, 27, self._depth(2.0), INTR))


class TestFootProgression(unittest.TestCase):
    def test_straight_foot_is_zero(self):
        self.assertAlmostEqual(
            DM.foot_progression_deg((0.0, 0.0, 0.0), (0.0, 0.0, 0.25)), 0.0,
            places=6)

    def test_toe_out_is_positive_magnitude(self):
        # 10 cm lateral over 25 cm forward -> 21.8 deg
        v = DM.foot_progression_deg((0.0, 0.0, 0.0), (0.10, 0.0, 0.25))
        self.assertAlmostEqual(v, math.degrees(math.atan2(0.10, 0.25)), places=4)
        v2 = DM.foot_progression_deg((0.0, 0.0, 0.0), (-0.10, 0.0, 0.25))
        self.assertAlmostEqual(v, v2, places=6)     # magnitude only

    def test_degenerate_inputs(self):
        self.assertIsNone(DM.foot_progression_deg(None, (0, 0, 1)))
        self.assertIsNone(DM.foot_progression_deg((0, 0, 0), (0, 0, 0)))


class TestDepthGait(unittest.TestCase):
    def _frame(self, y_left, y_right, side_x=0.05):
        """Landmarks for one frame: feet at known normalised positions."""
        a = lms33()
        a[27] = [0.5 - side_x, y_left, 0.0, 1.0]
        a[28] = [0.5 + side_x, y_right, 0.0, 1.0]
        a[29], a[30] = a[27].copy(), a[28].copy()
        a[31] = [0.5 - side_x, y_left + 0.02, 0.0, 1.0]
        a[32] = [0.5 + side_x, y_right + 0.02, 0.0, 1.0]
        return a

    def test_summary_shape_and_json_safety(self):
        g = DM.DepthGait()
        depth = lambda nx, ny, r=4: 3.0
        for i in range(80):
            ph = 2 * math.pi * i / 20.0
            g.update(self._frame(0.72 + 0.06 * math.sin(ph),
                                 0.72 + 0.06 * math.sin(ph + math.pi)),
                     depth, INTR)
        s = g.summary()
        json.dumps(s)                       # must be JSON-safe
        self.assertEqual(s["frames"], 80)
        self.assertGreater(s["frames_with_depth"], 0)
        self.assertIsNotNone(s["lateral_separation_cm"])
        self.assertGreater(s["lateral_separation_cm"], 0.0)
        self.assertIn("depth_note", s)

    def test_missing_depth_yields_empty_row_fields(self):
        g = DM.DepthGait()
        out = g.update(lms33(), lambda nx, ny, r=4: None, INTR)
        self.assertEqual(out, {})
        self.assertEqual(g.summary()["frames_with_depth"], 0)

    def test_step_width_is_metric_not_image_distance(self):
        g = DM.DepthGait()
        # feet 0.1 m apart in the world, depth 3 m -> ~57 px apart at fx=391
        g.update(self._frame(0.7, 0.75, side_x=0.05), lambda nx, ny, r=4: 3.0, INTR)
        self.assertAlmostEqual(g.sep_cm[-1], 2 * 0.05 * 640 / 391.1 * 3.0 * 100,
                               places=2)


class TestApiWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        import app as A  # noqa: E402
        from session_store import Store
        cls.A = A
        cls.A.STORE = Store(config.DB_PATH, sessions_dir=Path(cls.tmp.name))
        cls.client = A.app.test_client()

    def test_state_exposes_rig(self):
        d = self.client.get("/api/state").get_json()
        self.assertIn("rig", d)
        self.assertIsInstance(d["rig"], dict)

    def test_rig_endpoint_without_camera(self):
        r = self.client.get("/api/rig/check")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(set(d), {"ok", "views", "active"})
        self.assertFalse(d["ok"])
        self.assertEqual(d["views"], {})

    def test_depth_source_disabled_returns_none(self):
        import cameras
        with mock.patch.object(config, "DEPTH_ENABLED", False):
            self.assertIsNone(cameras.open_source(kind="depth"))

    def test_depth_source_failure_is_soft(self):
        """No RealSense / SDK failure must fall back to None, never raise."""
        import cameras
        with mock.patch.dict(sys.modules, {"depth": None}):
            self.assertIsNone(cameras.open_source(kind="depth"))

    def test_depth_kind_accepted_by_open_payload_parser(self):
        # a depth source needs no index; the handler must not reject the kind
        r = self.client.post("/api/camera/open",
                             json={"kind": "depth", "width": 640, "height": 480},
                             environ_base={"REMOTE_ADDR": "127.0.0.1"})
        self.assertIn(r.status_code, (200, 400))
        if r.status_code == 400:
            self.assertNotIn("bad kind", (r.get_json() or {}).get("error", ""))
        # never leave a capture thread running for the next test
        self.client.post("/api/camera/close",
                         environ_base={"REMOTE_ADDR": "127.0.0.1"})


if __name__ == "__main__":
    unittest.main()
