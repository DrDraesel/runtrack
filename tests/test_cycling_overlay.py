"""Cycling fit overlay (pose_engine.draw_cycling_overlay) tests.

Pure drawing logic on a blank canvas with synthetic side-view landmarks: the
fit layer must be additive, band-aware and utterly non-throwing on bad input.
"""
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import pose_engine as PE  # noqa: E402

GREEN = (80, 210, 120)
AMBER = (0, 190, 255)
WHITE = (235, 235, 235)
GRAY_REF = (90, 90, 90)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)

# Side-view rider facing +x, hips high-left, feet low-centre (normalized).
SIDE_POINTS = {
    0: (0.60, 0.28), 7: (0.57, 0.27), 8: (0.58, 0.28),
    11: (0.47, 0.34), 12: (0.48, 0.345), 13: (0.56, 0.42), 14: (0.57, 0.425),
    15: (0.64, 0.36), 16: (0.65, 0.365),
    23: (0.36, 0.44), 24: (0.37, 0.445), 25: (0.44, 0.60), 26: (0.45, 0.61),
    27: (0.38, 0.78), 28: (0.39, 0.79), 29: (0.36, 0.80), 30: (0.37, 0.81),
    31: (0.40, 0.81), 32: (0.41, 0.82),
}


def landmarks(points=None, vis=1.0):
    """(33, 4) landmarks; unlisted points sit mid-frame with the given vis."""
    a = np.zeros((33, 4), dtype=np.float32)
    a[:, 0] = 0.5
    a[:, 1] = 0.5
    a[:, 3] = vis
    for i, (x, y) in (points if points is not None else SIDE_POINTS).items():
        a[i, 0], a[i, 1] = x, y
    return a


def blank(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def count_near(img, color, tol=14):
    """Pixels whose worst channel differs from ``color`` by <= ``tol``.

    Max-channel (not summed) distance keeps the near-identical pairs apart:
    the orange arm (0,165,255) sits 25 units from the amber label on the green
    channel alone.
    """
    d = np.abs(img.astype(np.int16) - np.array(color, np.int16)).max(axis=2)
    return int((d <= tol).sum())


def changed_px(img):
    return int(img.any(axis=2).sum())


FM = {
    "left_knee": 145.0, "right_knee": 145.0,
    "left_hip": 20.0, "right_hip": 20.0,
    "left_elbow": 160.0, "right_elbow": 160.0,
    "trunk_lean": -40.0,            # torso = 90 - 40 = 50 deg vs horizontal
}


class TestAngleColor(unittest.TestCase):
    def test_inside_band_is_green(self):
        self.assertEqual(PE.angle_color(145.0, (140.0, 150.0)), GREEN)
        self.assertEqual(PE.angle_color(140.0, (140.0, 150.0)), GREEN)  # edge

    def test_outside_band_is_amber(self):
        self.assertEqual(PE.angle_color(151.0, (140.0, 150.0)), AMBER)
        self.assertEqual(PE.angle_color(120.0, (140.0, 150.0)), AMBER)

    def test_missing_input_is_white(self):
        self.assertEqual(PE.angle_color(None, (140.0, 150.0)), WHITE)
        self.assertEqual(PE.angle_color(145.0, None), WHITE)
        self.assertEqual(PE.angle_color(float("nan"), (140.0, 150.0)), WHITE)
        self.assertEqual(PE.angle_color(145.0, (float("inf"), 150.0)), WHITE)
        self.assertEqual(PE.angle_color("n/a", (140.0, 150.0)), WHITE)
        self.assertEqual(PE.angle_color(145.0, (140.0,)), WHITE)

    def test_reversed_band_is_tolerated(self):
        self.assertEqual(PE.angle_color(145.0, (150.0, 140.0)), GREEN)

    def test_returns_bgr_tuple(self):
        for v, b in ((145.0, (140.0, 150.0)), (10.0, (140.0, 150.0)),
                     (None, None)):
            c = PE.angle_color(v, b)
            self.assertIsInstance(c, tuple)
            self.assertEqual(len(c), 3)
            self.assertTrue(all(isinstance(x, int) for x in c))


class TestDrawCyclingOverlay(unittest.TestCase):
    def test_draws_on_blank_frame_and_returns_same_array(self):
        img = blank()
        out = PE.draw_cycling_overlay(img, landmarks(), FM, bike_type="road")
        self.assertIs(out, img)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(changed_px(out), 0)

    def test_all_fit_layers_present(self):
        out = PE.draw_cycling_overlay(blank(), landmarks(), FM)
        self.assertGreater(count_near(out, CYAN), 0, "hip-knee-ankle missing")
        self.assertGreater(count_near(out, MAGENTA), 0, "torso/shoulder-hip missing")
        self.assertGreater(count_near(out, ORANGE), 0, "arm chain missing")
        self.assertGreater(count_near(out, GRAY_REF), 0, "reference lines missing")

    def test_low_visibility_landmarks_draw_nothing(self):
        img = blank()
        out = PE.draw_cycling_overlay(img, landmarks(vis=0.2), FM)
        self.assertIs(out, img)
        self.assertEqual(changed_px(out), 0)

    def test_none_landmarks_is_byte_identical(self):
        img = blank()
        before = img.copy()
        out = PE.draw_cycling_overlay(img, None, FM)
        self.assertIs(out, img)
        self.assertTrue(np.array_equal(out, before))

    def test_empty_fm_draws_lines_but_no_labels(self):
        out = PE.draw_cycling_overlay(blank(), landmarks(), {})
        self.assertGreater(changed_px(out), 0)          # tracking lines remain
        self.assertEqual(count_near(out, GREEN), 0)
        self.assertEqual(count_near(out, AMBER), 0)
        self.assertEqual(count_near(out, WHITE, tol=60), 0)

    def test_short_landmark_array_does_not_raise(self):
        for n in (0, 1, 5, 26):                        # 26 has hips, no knees
            img = blank()
            arr = np.zeros((n, 4), dtype=np.float32)
            arr[:, 3] = 1.0
            try:
                out = PE.draw_cycling_overlay(img, arr, FM)
            except Exception as exc:                   # noqa: BLE001
                self.fail(f"{n}-row landmarks raised {exc!r}")
            self.assertIs(out, img)

    def test_narrow_landmarks_and_nonfinite_values_do_not_raise(self):
        for arr in (np.zeros((33, 2), dtype=np.float32),      # no visibility col
                    np.full((33, 4), np.nan, dtype=np.float32),
                    np.full((33, 4), np.inf, dtype=np.float32)):
            img = blank()
            try:
                out = PE.draw_cycling_overlay(img, arr, FM)
            except Exception as exc:                   # noqa: BLE001
                self.fail(f"bad landmarks raised {exc!r}")
            self.assertIs(out, img)

    def test_single_nonfinite_joint_only_drops_that_segment(self):
        lms = landmarks()
        lms[25, 0] = np.nan                             # left knee
        img = blank()
        out = PE.draw_cycling_overlay(img, lms, FM)
        self.assertGreater(count_near(out, CYAN), 0)    # right leg still drawn

    def test_tiny_frame_does_not_raise(self):
        for w, h in ((3, 3), (1, 1)):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                out = PE.draw_cycling_overlay(img, landmarks(), FM)
            except Exception as exc:                   # noqa: BLE001
                self.fail(f"{w}x{h} frame raised {exc!r}")
            self.assertIs(out, img)

    def test_thickness_parameter_widens_lines(self):
        thin = PE.draw_cycling_overlay(blank(), landmarks(), FM, thickness=1)
        thick = PE.draw_cycling_overlay(blank(), landmarks(), FM, thickness=6)
        self.assertGreater(count_near(thick, CYAN), count_near(thin, CYAN))


class TestBandsAndLabels(unittest.TestCase):
    def test_config_band_colours_labels_by_bike_type(self):
        # knee 148: inside road (140-150), outside tt (140-147)
        fm = dict(FM, left_knee=148.0, right_knee=148.0)
        road = PE.draw_cycling_overlay(blank(), landmarks(), fm, bike_type="road")
        tt = PE.draw_cycling_overlay(blank(), landmarks(), fm, bike_type="tt")
        self.assertGreater(count_near(road, GREEN), 0)
        self.assertEqual(count_near(tt, GREEN, tol=20), 0)
        self.assertGreater(count_near(tt, AMBER), 0)

    def test_explicit_targets_override_config(self):
        # same 145 deg knee, but a band that excludes it -> no green anywhere
        out = PE.draw_cycling_overlay(
            blank(), landmarks(), FM,
            targets={"knee_bdc_deg": (100.0, 110.0),
                     "hip_tdc_deg": (70.0, 90.0),
                     "torso_deg": (35.0, 45.0)})
        self.assertEqual(count_near(out, GREEN, tol=20), 0)
        self.assertGreater(count_near(out, AMBER), 0)

    def test_torso_uses_90_minus_abs_trunk_lean(self):
        # trunk_lean = -40 -> torso 50 deg; inside a (45, 55) band = green
        out = PE.draw_cycling_overlay(blank(), landmarks(), FM,
                                      targets={"torso_deg": (45.0, 55.0)})
        self.assertGreater(count_near(out, GREEN), 0)

    def test_tt_position_tag_and_aerobar_line(self):
        road = PE.draw_cycling_overlay(blank(), landmarks(), FM, bike_type="road")
        tt = PE.draw_cycling_overlay(blank(), landmarks(), FM, bike_type="tt")
        self.assertEqual(changed_px(road[:32, :150]), 0)   # nothing top-left
        self.assertGreater(changed_px(tt[:32, :150]), 0)   # "TT position" tag
        self.assertGreater(changed_px(tt), changed_px(road))

    def test_tri_runs_tt_path(self):
        out = PE.draw_cycling_overlay(blank(), landmarks(), FM, bike_type="tri")
        self.assertGreater(changed_px(out[:32, :150]), 0)

    def test_unknown_bike_type_falls_back_to_default(self):
        img = blank()
        try:
            out = PE.draw_cycling_overlay(img, landmarks(), FM, bike_type="mtb")
        except Exception as exc:                       # noqa: BLE001
            self.fail(f"unknown bike type raised {exc!r}")
        self.assertGreater(changed_px(out), 0)


if __name__ == "__main__":
    unittest.main()
