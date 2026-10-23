"""Tests for rigcheck: synthetic landmarks only, no camera, no depth hardware.

All fixtures come from one full-body side-view pose on a 576x324 frame, where
the body spans exactly 80% of the frame height.  Each test changes one thing
(visibility, scale, vertical offset, nose position, plane) so a failing
assertion points at a single rule.

Run: ./.venv/Scripts/python.exe -m unittest tests.test_rigcheck -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import rigcheck as RC  # noqa: E402

W, H = 576, 324          # 16:9, small enough that the numbers stay readable

# Full-body profile, y down: head top 0.10, toes 0.90 (80% of the frame).
# Shoulders 0.45/0.55 (width 0.10), nose at 0.65 -> offset ratio 1.5 = sagittal.
BASE_POSE = {
    0: (0.65, 0.145),
    1: (0.62, 0.125), 2: (0.635, 0.10), 3: (0.615, 0.115), 4: (0.60, 0.13),
    5: (0.645, 0.11), 6: (0.635, 0.155), 7: (0.575, 0.155), 8: (0.60, 0.16),
    9: (0.63, 0.17), 10: (0.645, 0.165),
    11: (0.45, 0.26), 12: (0.55, 0.26),
    13: (0.44, 0.36), 14: (0.56, 0.36),
    15: (0.42, 0.44), 16: (0.58, 0.44),
    17: (0.415, 0.455), 18: (0.585, 0.455),
    19: (0.42, 0.465), 20: (0.58, 0.465),
    21: (0.425, 0.44), 22: (0.575, 0.44),
    23: (0.47, 0.50), 24: (0.53, 0.50),
    25: (0.48, 0.70), 26: (0.52, 0.70),
    27: (0.49, 0.88), 28: (0.53, 0.88),
    29: (0.46, 0.895), 30: (0.54, 0.895),
    31: (0.56, 0.90), 32: (0.58, 0.90),
}


def lms_from(pose: dict, vis: float = 0.95) -> np.ndarray:
    """(33, 4) landmark array from {index: (x, y)}."""
    arr = np.zeros((33, 4), dtype=np.float32)
    arr[:, 0:2] = 0.5
    for i, (x, y) in pose.items():
        arr[i] = (x, y, 0.0, vis)
    return arr


def scaled(pose: dict, factor: float) -> dict:
    """Shrink the body about the frame centre."""
    return {i: (0.5 + (x - 0.5) * factor, 0.5 + (y - 0.5) * factor)
            for i, (x, y) in pose.items()}


def shifted(pose: dict, dy: float) -> dict:
    """Move the body down the frame by ``dy``."""
    return {i: (x, y + dy) for i, (x, y) in pose.items()}


def frontal_pose() -> dict:
    """Same body with the nose on the shoulder midline (a rear/front view)."""
    pose = dict(BASE_POSE)
    pose[0] = (0.50, 0.145)
    return pose


def clipped_feet_lms() -> np.ndarray:
    """Feet below the frame and barely visible: gait events cannot latch."""
    arr = lms_from(BASE_POSE)
    for i, (x, y) in {29: (0.46, 1.02), 30: (0.54, 1.02),
                      31: (0.56, 1.03), 32: (0.58, 1.03)}.items():
        arr[i] = (x, y, 0.0, 0.1)
    return arr


def head_hidden_lms() -> np.ndarray:
    arr = lms_from(BASE_POSE)
    arr[0:11, 3] = 0.1
    return arr


class TestBodyGroups(unittest.TestCase):
    def test_body_groups_are_the_mediapipe_sets(self):
        self.assertEqual(
            RC.BODY_GROUPS,
            {"head": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
             "torso": (11, 12, 23, 24),
             "arms": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
             "legs": (25, 26, 27, 28),
             "feet": (29, 30, 31, 32)})


class TestBodyCoverage(unittest.TestCase):
    def test_full_body_is_fully_visible(self):
        cov = RC.body_coverage(lms_from(BASE_POSE))
        for key in ("head", "torso", "arms", "legs", "feet", "lower", "overall"):
            self.assertEqual(cov[key], 1.0, key)

    def test_none_returns_zeros_for_every_key(self):
        cov = RC.body_coverage(None)
        self.assertEqual(set(cov), {"head", "torso", "arms", "legs", "feet",
                                    "lower", "overall"})
        self.assertEqual(set(cov.values()), {0.0})

    def test_hidden_feet_drop_lower_and_overall(self):
        cov = RC.body_coverage(clipped_feet_lms())
        self.assertEqual(cov["feet"], 0.0)
        self.assertEqual(cov["legs"], 1.0)
        self.assertAlmostEqual(cov["lower"], 0.5, places=6)
        self.assertAlmostEqual(cov["overall"], 0.8, places=6)

    def test_visibility_threshold_is_inclusive(self):
        arr = lms_from(BASE_POSE, vis=0.5)
        self.assertEqual(RC.body_coverage(arr)["overall"], 1.0)
        arr = lms_from(BASE_POSE, vis=0.49)
        self.assertEqual(RC.body_coverage(arr)["overall"], 0.0)


class TestFraming(unittest.TestCase):
    def test_bbox_matches_the_visible_landmarks(self):
        fr = RC.framing(lms_from(BASE_POSE), W, H)
        self.assertAlmostEqual(fr["top_px"], 0.10 * H, places=3)
        self.assertAlmostEqual(fr["bottom_px"], 0.90 * H, places=3)
        self.assertAlmostEqual(fr["left_px"], 0.415 * W, places=3)
        self.assertAlmostEqual(fr["right_px"], 0.65 * W, places=3)
        self.assertAlmostEqual(fr["height_frac"], 0.80, places=4)
        self.assertAlmostEqual(fr["body_h_px"], 0.80 * H, places=3)
        self.assertAlmostEqual(fr["width_frac"], 0.235, places=4)
        self.assertFalse(fr["touches_edge"])
        for margin in fr["margins"].values():
            self.assertGreaterEqual(margin, RC.EDGE_MARGIN)

    def test_none_landmarks_are_none_safe(self):
        fr = RC.framing(None, W, H)
        for key in ("top_px", "bottom_px", "left_px", "right_px",
                    "height_frac", "width_frac", "body_h_px"):
            self.assertIsNone(fr[key], key)
        self.assertFalse(fr["touches_edge"])

    def test_all_invisible_or_bad_frame_size_is_none(self):
        invisible = np.zeros((33, 4), dtype=np.float32)
        self.assertIsNone(RC.framing(invisible, W, H)["top_px"])
        self.assertIsNone(RC.framing(lms_from(BASE_POSE), 0, H)["top_px"])
        self.assertIsNone(RC.framing(lms_from(BASE_POSE), W, -1)["top_px"])

    def test_body_at_the_bottom_edge_flags_touches_edge(self):
        fr = RC.framing(lms_from(shifted(BASE_POSE, 0.10)), W, H)
        self.assertTrue(fr["touches_edge"])
        self.assertAlmostEqual(fr["margins"]["bottom"], 0.0, places=6)
        self.assertLess(fr["margins"]["bottom"], RC.EDGE_MARGIN)


class TestPlaneGuess(unittest.TestCase):
    def test_frontal_when_nose_sits_on_the_shoulder_midline(self):
        self.assertEqual(RC.plane_guess(lms_from(frontal_pose())), "frontal")

    def test_sagittal_when_nose_is_well_off_the_midline(self):
        self.assertEqual(RC.plane_guess(lms_from(BASE_POSE)), "sagittal")

    def test_unknown_when_a_stack_has_fewer_than_five_usable_frames(self):
        four = np.stack([lms_from(frontal_pose())] * 4)
        five = np.stack([lms_from(frontal_pose())] * 5)
        self.assertEqual(RC.plane_guess(four), "unknown")
        self.assertEqual(RC.plane_guess(five), "frontal")

    def test_unknown_when_shoulders_are_not_visible(self):
        arr = lms_from(BASE_POSE)
        arr[11:13, 3] = 0.1
        self.assertEqual(RC.plane_guess(arr), "unknown")
        self.assertEqual(RC.plane_guess(None), "unknown")

    def test_mid_range_offset_is_not_forced_into_a_plane(self):
        # ratio 0.35 sits between the report's 0.25 / 0.60 thresholds
        pose = dict(BASE_POSE)
        pose[0] = (0.535, 0.145)
        self.assertEqual(RC.plane_guess(lms_from(pose)), "unknown")


class TestPxPerM(unittest.TestCase):
    def test_body_height_gives_the_fallback_resolution(self):
        ppm = RC.px_per_m_from_body(lms_from(BASE_POSE), W, H)
        self.assertIsNotNone(ppm)
        assert ppm is not None          # narrows the type for the checker
        self.assertAlmostEqual(ppm, 0.80 * H / 1.75, places=3)
        self.assertGreater(ppm, 130.0)

    def test_none_when_the_head_is_not_visible(self):
        self.assertIsNone(RC.px_per_m_from_body(head_hidden_lms(), W, H))

    def test_none_when_the_feet_are_not_visible(self):
        self.assertIsNone(RC.px_per_m_from_body(clipped_feet_lms(), W, H))

    def test_none_for_an_unusable_body_height(self):
        arr = lms_from(BASE_POSE)
        self.assertIsNone(RC.px_per_m_from_body(arr, W, H, height_m=0.0))
        self.assertIsNone(RC.px_per_m_from_body(arr, W, H, height_m=float("nan")))


class TestWarningsFor(unittest.TestCase):
    def setUp(self):
        self.cov = RC.body_coverage(lms_from(BASE_POSE))
        self.fr = RC.framing(lms_from(BASE_POSE), W, H)

    def test_frontal_pitch_rule_fires_only_above_20_deg_down(self):
        self.assertEqual(
            RC.warnings_for("frontal", self.cov, self.fr, 150.0, 12.0, "frontal"), [])
        self.assertEqual(
            RC.warnings_for("frontal", self.cov, self.fr, 150.0, 26.0, "frontal"),
            ["camera pitched 26 deg down (max 20 for a front view)"])

    def test_sagittal_pitch_rule_uses_the_absolute_angle(self):
        w = RC.warnings_for("sagittal", self.cov, self.fr, 150.0, -8.0, "sagittal")
        self.assertEqual(
            w, ["camera pitched 8 deg (max 5 for a side view: "
                "sagittal angles are measured against image vertical)"])

    def test_warning_numbers_stay_on_the_correct_side_of_the_threshold(self):
        # 0.6999999 rounds to 70.0%; the message must still read below 70%
        cov = {"lower": 0.6999999, "feet": 1.0}
        w = RC.warnings_for("sagittal", cov, self.fr, 150.0, None, "sagittal")
        self.assertEqual(
            w, ["lower body 69.9% visible - gait events cannot latch below 70%"])
        w = RC.warnings_for("sagittal", self.cov, self.fr, 59.9999999, None, "sagittal")
        self.assertEqual(w, ["resolution 59 px/m - too coarse, aim for 100+"])

    def test_resolution_warning_floors_the_number(self):
        w = RC.warnings_for("sagittal", self.cov, self.fr, 59.4, None, "sagittal")
        self.assertEqual(w, ["resolution 59 px/m - too coarse, aim for 100+"])
        # 59.7 must not be rounded up to a self-contradicting "60 px/m".
        w = RC.warnings_for("sagittal", self.cov, self.fr, 59.7, None, "sagittal")
        self.assertEqual(w, ["resolution 59 px/m - too coarse, aim for 100+"])

    def test_valid_rig_produces_no_warnings(self):
        self.assertEqual(
            RC.warnings_for("sagittal", self.cov, self.fr,
                            RC.px_per_m_from_body(lms_from(BASE_POSE), W, H),
                            0.0, "sagittal"), [])


class TestCheckView(unittest.TestCase):
    def test_full_body_side_view_passes_with_no_warnings(self):
        res = RC.check_view("sagittal", lms_from(BASE_POSE), W, H, pitch_deg=0.0)
        self.assertEqual(
            set(res), {"ok", "plane", "plane_guess", "coverage", "framing",
                       "px_per_m", "pitch_deg", "distance_m", "warnings", "notes"})
        self.assertEqual(res["warnings"], [])
        self.assertTrue(res["ok"])
        self.assertEqual(res["plane"], "sagittal")
        self.assertEqual(res["plane_guess"], "sagittal")
        self.assertEqual(res["coverage"]["lower"], 1.0)
        self.assertEqual(res["coverage"]["overall"], 1.0)
        self.assertAlmostEqual(res["framing"]["height_frac"], 0.80, places=4)
        self.assertGreater(res["framing"]["height_frac"], 0.55)
        self.assertAlmostEqual(res["px_per_m"], 0.80 * H / 1.75, places=3)
        self.assertGreater(res["px_per_m"], 130.0)      # ~150 px/m in this fixture
        self.assertLess(res["px_per_m"], 170.0)
        self.assertIsNone(res["distance_m"])
        self.assertAlmostEqual(res["pitch_deg"], 0.0, places=6)
        # The result must never claim a joint angle was validated.
        self.assertIn("not joint angles", res["notes"][0])

    def test_pitched_camera_produces_exactly_one_warning_naming_34(self):
        res = RC.check_view("sagittal", lms_from(BASE_POSE), W, H, pitch_deg=34.0)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["warnings"]), 1)
        self.assertEqual(
            res["warnings"][0],
            "camera pitched 34 deg (max 5 for a side view: "
            "sagittal angles are measured against image vertical)")
        self.assertAlmostEqual(res["pitch_deg"], 34.0, places=6)

    def test_clipped_feet_warn_feet_and_lower_body(self):
        res = RC.check_view("sagittal", clipped_feet_lms(), W, H)
        self.assertFalse(res["ok"])
        self.assertIn("feet 0% visible - foot metrics unavailable", res["warnings"])
        self.assertIn("lower body 50% visible - gait events cannot latch below 70%",
                      res["warnings"])
        self.assertEqual(len(res["warnings"]), 2)
        self.assertEqual(res["coverage"]["feet"], 0.0)
        self.assertAlmostEqual(res["coverage"]["lower"], 0.5, places=6)
        # No whole body -> no body-height resolution estimate, and no bogus one.
        self.assertIsNone(res["px_per_m"])

    def test_small_runner_warns_on_frame_height(self):
        res = RC.check_view("sagittal", lms_from(scaled(BASE_POSE, 0.5)), W, H)
        self.assertFalse(res["ok"])
        self.assertEqual(
            res["warnings"],
            ["runner fills only 40% of frame height - move camera closer or zoom"])
        self.assertAlmostEqual(res["framing"]["height_frac"], 0.40, places=4)
        # 40% of 324 px is still ~74 px/m, above the resolution floor.
        self.assertGreater(res["px_per_m"], RC.PX_PER_M_MIN)

    def test_frontal_fixture_analysed_as_sagittal_warns(self):
        res = RC.check_view("sagittal", lms_from(frontal_pose()), W, H)
        self.assertFalse(res["ok"])
        self.assertEqual(res["plane_guess"], "frontal")
        self.assertEqual(
            res["warnings"],
            ["view looks frontal but is being analysed as sagittal - check camera position"])

    def test_intrinsics_plus_depth_give_px_per_m(self):
        res = RC.check_view(
            "sagittal", lms_from(BASE_POSE), W, H,
            depth={"valid_pct": 92.0, "median_m": 3.0},
            intrinsics={"fx": 450.0, "fy": 450.0, "cx": 288.0, "cy": 162.0,
                        "width": W, "height": H})
        self.assertTrue(res["ok"])
        self.assertEqual(res["warnings"], [])
        self.assertAlmostEqual(res["distance_m"], 3.0, places=6)
        self.assertAlmostEqual(res["px_per_m"], 150.0, places=6)
        self.assertTrue(any("intrinsics" in n for n in res["notes"]))

    def test_implausible_scene_depth_is_not_called_subject_distance(self):
        res = RC.check_view("sagittal", lms_from(BASE_POSE), W, H,
                            depth={"valid_pct": 8.0, "median_m": 0.32})
        self.assertIsNone(res["distance_m"])
        self.assertTrue(any("not subject distance" in n for n in res["notes"]))
        self.assertTrue(any("body height" in n for n in res["notes"]))
        self.assertAlmostEqual(res["px_per_m"], 0.80 * H / 1.75, places=3)

    def test_scene_depth_is_rejected_when_the_runner_is_small_in_frame(self):
        # A plausible-looking 3 m median, but the runner fills 40% of the frame:
        # the median is mostly background, so it must not be called the subject.
        res = RC.check_view("sagittal", lms_from(scaled(BASE_POSE, 0.5)), W, H,
                            depth={"valid_pct": 95.0, "median_m": 3.0})
        self.assertIsNone(res["distance_m"])
        self.assertTrue(any("not subject distance" in n for n in res["notes"]))

    def test_touching_edge_names_the_clipped_edge(self):
        res = RC.check_view("sagittal", lms_from(shifted(BASE_POSE, 0.10)), W, H)
        self.assertFalse(res["ok"])
        self.assertEqual(
            res["warnings"],
            ["body clipped at the bottom edge (0% margin) - widen the shot to fit the runner"])

    def test_none_landmarks_do_not_raise(self):
        res = RC.check_view("sagittal", None, W, H)
        self.assertFalse(res["ok"])
        self.assertEqual(res["plane_guess"], "unknown")
        self.assertIsNone(res["px_per_m"])
        self.assertIsNone(res["framing"]["top_px"])
        self.assertEqual(res["coverage"]["overall"], 0.0)
        self.assertEqual(len(res["warnings"]), 2)

    def test_rejects_an_unknown_plane(self):
        with self.assertRaises(ValueError):
            RC.check_view("side", lms_from(BASE_POSE), W, H)
        with self.assertRaises(ValueError):
            RC.check_view(None, lms_from(BASE_POSE), W, H)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
