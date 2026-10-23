"""Unit tests for depth.py.

Everything here runs on synthetic numpy arrays: no RealSense, no camera, no
threads.  The capture class is only exercised through its documented
"pyrealsense2 missing" fallback (depth.rs monkeypatched to None).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import depth  # noqa: E402

# Synthetic camera: the measured D455F colour intrinsics at 640x480.
INTR = {"fx": 391.1, "fy": 391.1, "cx": 319.5, "cy": 239.5,
        "width": 640, "height": 480}
CAM_HEIGHT_M = 1.2
MAX_RANGE_M = 6.0


def render_floor(pitch_deg, intr=INTR, height_m=CAM_HEIGHT_M,
                 max_range_m=MAX_RANGE_M):
    """Ray-cast a flat floor into a float32 HxW metres depth image.

    Camera at ``height_m`` above the floor, pitched down by ``pitch_deg``
    (positive = optical axis below horizontal), no roll.  Uses the same
    pinhole convention as depth.py: point = ((u-cx)z/fx, (v-cy)z/fy, z) and
    the floor is Y*cos(theta) + Z*sin(theta) = height.

    Pixels whose ray never meets the floor (above the horizon) or that would be
    farther than ``max_range_m`` (the sensor's useful range) are 0 = invalid,
    exactly like a real depth frame.
    """
    w, h = int(intr["width"]), int(intr["height"])
    fx, fy, cx, cy = (float(intr["fx"]), float(intr["fy"]),
                      float(intr["cx"]), float(intr["cy"]))
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    dy = (vv - cy) / fy
    theta = np.radians(float(pitch_deg))
    denom = dy * np.cos(theta) + np.sin(theta)
    depth_img = np.zeros((h, w), dtype=np.float32)
    ahead = denom > 1e-9
    z = np.where(ahead, height_m / np.where(ahead, denom, 1.0), 0.0)
    keep = ahead & (z <= max_range_m)
    depth_img[keep] = z[keep].astype(np.float32)
    return depth_img


class TestSampleDepth(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((21, 21), dtype=np.float32)
        self.img[10:13, 10:13] = np.array([[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]], dtype=np.float32)

    def test_median_of_valid_window(self):
        self.assertEqual(depth.sample_depth(self.img, 0.55, 0.55, r=1), 5.0)

    def test_zero_pixels_do_not_drag_the_median(self):
        # A wider window is mostly zeros (invalid); only the 9 real samples count.
        self.assertEqual(depth.sample_depth(self.img, 0.55, 0.55, r=4), 5.0)

    def test_r_zero_samples_one_pixel(self):
        self.assertEqual(depth.sample_depth(self.img, 0.55, 0.55, r=0), 5.0)
        self.assertIsNone(depth.sample_depth(self.img, 0.1, 0.1, r=0))

    def test_window_clipped_at_border(self):
        corner = np.zeros((9, 9), dtype=np.float32)
        corner[0, 0] = 2.0
        self.assertEqual(depth.sample_depth(corner, 0.0, 0.0, r=4), 2.0)

    def test_r_larger_than_image_is_clipped(self):
        self.assertEqual(depth.sample_depth(self.img, 0.55, 0.55, r=50), 5.0)

    def test_all_zero_window_is_none(self):
        self.assertIsNone(depth.sample_depth(self.img, 0.1, 0.1, r=4))

    def test_nan_and_inf_are_skipped(self):
        img = np.zeros((5, 5), dtype=np.float32)
        img[1:4, 1:4] = np.array([[np.nan, np.inf, 2.0],
                                  [3.0, 4.0, np.nan],
                                  [np.inf, 1.0, 6.0]], dtype=np.float32)
        # Valid values 2,3,4,1,6 -> median 3.
        self.assertEqual(depth.sample_depth(img, 0.5, 0.5, r=2), 3.0)

    def test_negative_values_are_invalid(self):
        img = np.zeros((5, 5), dtype=np.float32)
        img[2, 2] = -1.5
        img[2, 3] = 3.0
        self.assertEqual(depth.sample_depth(img, 0.5, 0.5, r=1), 3.0)

    def test_out_of_range_coords_return_none(self):
        for nx, ny in ((-0.01, 0.5), (1.01, 0.5), (0.5, -0.01), (0.5, 1.01),
                       (float("nan"), 0.5), (0.5, float("inf"))):
            with self.subTest(nx=nx, ny=ny):
                self.assertIsNone(depth.sample_depth(self.img, nx, ny, r=2))

    def test_none_or_empty_array_returns_none(self):
        self.assertIsNone(depth.sample_depth(None, 0.5, 0.5))
        self.assertIsNone(depth.sample_depth(np.zeros((0, 0), np.float32), 0.5, 0.5))


class TestPxPerM(unittest.TestCase):
    def test_known_value(self):
        self.assertAlmostEqual(depth.px_per_m(400.0, 2.0), 200.0, places=9)

    def test_doubling_distance_halves_pixels_per_metre(self):
        near = depth.px_per_m(391.1, 2.0)
        far = depth.px_per_m(391.1, 4.0)
        assert near is not None and far is not None
        self.assertAlmostEqual(near, 2.0 * far, places=9)

    def test_one_centimetre_per_pixel_at_3_91_m(self):
        ppm = depth.px_per_m(391.1, 3.91)
        assert ppm is not None
        self.assertAlmostEqual(ppm, 100.026, delta=0.01)

    def test_reciprocal_is_metres_per_pixel(self):
        ppm = depth.px_per_m(391.1, 2.5)
        assert ppm is not None
        self.assertAlmostEqual(ppm * (2.5 / 391.1), 1.0, places=9)

    def test_guards(self):
        for fx, d in ((391.1, 0.0), (391.1, -1.0), (0.0, 2.0), (-391.1, 2.0),
                      (float("nan"), 2.0), (391.1, float("inf"))):
            with self.subTest(fx=fx, d=d):
                self.assertIsNone(depth.px_per_m(fx, d))


class TestFitFloorPlane(unittest.TestCase):
    def test_recovers_pitch_and_height(self):
        for pitch in (0.0, 15.0, 30.0):
            with self.subTest(pitch=pitch):
                img = render_floor(pitch)
                plane = depth.fit_floor_plane(img, INTR)
                assert plane is not None
                self.assertGreaterEqual(plane["points"], depth.MIN_POINTS)
                self.assertGreater(plane["inlier_pct"], 99.0)
                self.assertLess(plane["rms_m"], 1e-3)
                self.assertAlmostEqual(float(np.linalg.norm(plane["normal"])),
                                       1.0, places=6)
                self.assertGreaterEqual(plane["normal"][2], 0.0)
                recovered = depth.camera_pitch_from_plane(plane["normal"])
                assert recovered is not None
                self.assertAlmostEqual(recovered, pitch, delta=0.5)
                # Normal points at the floor for a downward-pitched camera,
                # so |d| is the camera height.
                self.assertAlmostEqual(plane["d"], -CAM_HEIGHT_M, delta=1e-3)

    def test_recovers_upward_pitch_sign(self):
        img = render_floor(-10.0)
        plane = depth.fit_floor_plane(img, INTR)
        assert plane is not None
        self.assertLess(plane["rms_m"], 1e-3)
        recovered = depth.camera_pitch_from_plane(plane["normal"])
        assert recovered is not None
        self.assertAlmostEqual(recovered, -10.0, delta=0.5)

    def test_clutter_is_trimmed(self):
        img = render_floor(20.0)
        rng = np.random.default_rng(7)
        h, w = img.shape
        rows = rng.integers(int(0.55 * h), h, size=4000)
        cols = rng.integers(0, w, size=4000)
        img[rows, cols] = rng.uniform(0.5, 4.0, size=4000).astype(np.float32)
        plane = depth.fit_floor_plane(img, INTR)
        assert plane is not None
        recovered = depth.camera_pitch_from_plane(plane["normal"])
        assert recovered is not None
        self.assertAlmostEqual(recovered, 20.0, delta=0.5)
        self.assertGreater(plane["inlier_pct"], 85.0)
        self.assertLess(plane["rms_m"], 0.005)

    def test_random_noise_is_rejected(self):
        rng = np.random.default_rng(11)
        noise = rng.uniform(0.5, 4.0, size=(480, 640)).astype(np.float32)
        plane = depth.fit_floor_plane(noise, INTR)
        # No dominant plane: either refused outright or reported with a large rms.
        if plane is not None:
            self.assertGreater(plane["rms_m"], 0.01)
            self.assertLess(plane["inlier_pct"], 20.0)

    def test_fewer_than_200_valid_points_is_none(self):
        img = np.zeros((480, 640), dtype=np.float32)
        img[400:410, 100:110] = 2.0          # 100 valid pixels
        self.assertIsNone(depth.fit_floor_plane(img, INTR))

    def test_exactly_200_points_is_accepted(self):
        floor = render_floor(15.0)
        img = np.zeros((480, 640), dtype=np.float32)
        img[400:420, 100:110] = floor[400:420, 100:110]   # 20 x 10 = 200
        self.assertEqual(int(np.count_nonzero(img)), 200)
        plane = depth.fit_floor_plane(img, INTR)
        assert plane is not None
        self.assertEqual(plane["points"], 200)
        recovered = depth.camera_pitch_from_plane(plane["normal"])
        assert recovered is not None
        self.assertAlmostEqual(recovered, 15.0, delta=0.5)
        img[400, 100] = 0.0                  # 199 points now
        self.assertIsNone(depth.fit_floor_plane(img, INTR))

    def test_all_zero_image_is_none(self):
        self.assertIsNone(depth.fit_floor_plane(np.zeros((480, 640), np.float32), INTR))

    def test_bad_intrinsics_are_rejected(self):
        img = render_floor(15.0)
        self.assertIsNone(depth.fit_floor_plane(img, None))
        self.assertIsNone(depth.fit_floor_plane(img, {}))
        broken = dict(INTR)
        del broken["cy"]
        self.assertIsNone(depth.fit_floor_plane(img, broken))
        zero_fx = dict(INTR, fx=0.0)
        self.assertIsNone(depth.fit_floor_plane(img, zero_fx))

    def test_empty_band_is_none(self):
        img = render_floor(15.0)
        self.assertIsNone(depth.fit_floor_plane(img, INTR, band=(0.6, 0.6)))


class TestCameraPitchFromPlane(unittest.TestCase):
    def test_level_camera_on_floor_is_zero(self):
        self.assertAlmostEqual(depth.camera_pitch_from_plane((0.0, 1.0, 0.0)),
                               0.0, places=9)

    def test_tilted_down_is_positive(self):
        for pitch in (5.0, 15.0, 30.0, 45.0):
            with self.subTest(pitch=pitch):
                theta = np.radians(pitch)
                n = (0.0, np.cos(theta), np.sin(theta))
                self.assertAlmostEqual(depth.camera_pitch_from_plane(n), pitch,
                                       places=9)

    def test_tilted_up_is_negative(self):
        for pitch in (-10.0, -30.0):
            with self.subTest(pitch=pitch):
                theta = np.radians(pitch)
                n = (0.0, np.cos(theta), np.sin(theta))
                self.assertAlmostEqual(depth.camera_pitch_from_plane(n), pitch,
                                       places=9)

    def test_orientation_and_scale_invariance(self):
        theta = np.radians(30.0)
        n = np.array([0.0, np.cos(theta), np.sin(theta)])
        for variant in (-n, 5.0 * n, 0.1 * n, list(n)):
            with self.subTest(variant=variant):
                self.assertAlmostEqual(depth.camera_pitch_from_plane(variant),
                                       30.0, places=9)

    def test_straight_down_is_90(self):
        self.assertAlmostEqual(depth.camera_pitch_from_plane((0.0, 0.0, 1.0)),
                               90.0, places=9)

    def test_invalid_normals_return_none(self):
        for n in ((0.0, 0.0, 0.0), (float("nan"), 1.0, 0.0), None, (1.0, 2.0)):
            with self.subTest(normal=n):
                self.assertIsNone(depth.camera_pitch_from_plane(n))


class TestDepthSourceWithoutHardware(unittest.TestCase):
    """rs = None is exactly the 'pyrealsense2 missing / no device' fallback."""

    def setUp(self):
        self._saved_rs = depth.rs
        depth.rs = None
        self.src = depth.DepthSource()

    def tearDown(self):
        self.src.release()
        depth.rs = self._saved_rs

    def test_construction_does_not_raise_and_reports_failure(self):
        self.assertFalse(self.src.ok)
        self.assertIsInstance(self.src.error, str)
        self.assertTrue(self.src.error)
        self.assertFalse(self.src.isOpened())

    def test_read_returns_nothing(self):
        self.assertEqual(self.src.read(), (False, None))
        self.assertEqual(self.src.read(), (False, None))

    def test_get_matches_cv2_property_ids(self):
        self.assertEqual(self.src.get(depth.CAP_PROP_FPS), 30.0)
        self.assertEqual(self.src.get(depth.CAP_PROP_FRAME_WIDTH), 640.0)
        self.assertEqual(self.src.get(depth.CAP_PROP_FRAME_HEIGHT), 480.0)
        self.assertEqual(self.src.get(depth.CAP_PROP_POS_MSEC), 0.0)
        self.assertEqual(self.src.get(9999), 0.0)
        self.assertEqual(self.src.get(None), 0.0)

    def test_set_is_a_no_op(self):
        self.assertFalse(self.src.set(depth.CAP_PROP_FPS, 15))
        self.assertFalse(self.src.set(depth.CAP_PROP_POS_MSEC, 1000))

    def test_depth_surface_is_empty(self):
        self.assertIsNone(self.src.intrinsics)
        self.assertIsNone(self.src.latest_depth())
        self.assertIsNone(self.src.depth_at(0.5, 0.5))
        self.assertIsNone(self.src.floor_pitch_deg())
        self.assertIsNone(self.src.px_per_m(2.0))
        self.assertEqual(self.src.depth_stats(),
                         {"valid_pct": None, "median_m": None,
                          "fps": None, "frames": 0})

    def test_release_is_idempotent(self):
        self.src.release()
        self.src.release()
        self.assertFalse(self.src.isOpened())
        self.assertEqual(self.src.read(), (False, None))

    def test_custom_size_is_reported(self):
        src = depth.DepthSource(width=848, height=480, fps=15)
        try:
            self.assertEqual(src.get(depth.CAP_PROP_FRAME_WIDTH), 848.0)
            self.assertEqual(src.get(depth.CAP_PROP_FRAME_HEIGHT), 480.0)
            self.assertEqual(src.get(depth.CAP_PROP_FPS), 15.0)
        finally:
            src.release()


if __name__ == "__main__":
    unittest.main()
