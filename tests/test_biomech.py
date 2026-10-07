"""Tests for the biomechanical metrics: sagittal/frontal math, plane gating,
data-quality honesty, and metrics.csv column backward compatibility."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import metrics as M  # noqa: E402

# columns that existed before the enrichment — metrics.csv consumers rely on
# these names never disappearing
LEGACY_FRAME_KEYS = (
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "trunk_lean", "neck_trunk", "head_tilt", "shoulder_tilt",
    "nose_x", "nose_y", "l_ankle_x", "l_ankle_y", "r_ankle_x", "r_ankle_y",
    "l_hip_x", "l_hip_y", "r_hip_x", "r_hip_y", "l_wrist_x", "l_wrist_y",
    "r_wrist_x", "r_wrist_y", "l_knee_x", "l_knee_y", "r_knee_x", "r_knee_y",
    "l_foot_x", "l_foot_y", "r_foot_x", "r_foot_y", "l_heel_x", "l_heel_y",
    "r_heel_x", "r_heel_y", "l_sh_x", "l_sh_y", "r_sh_x", "r_sh_y",
    "_midsh_x", "_midsh_y", "_midhip_x", "_midhip_y",
)


def _lms(pose: dict, vis=1.0):
    arr = np.zeros((33, 4), dtype=np.float32)
    arr[:, 0:2] = 0.5
    arr[:, 3] = vis
    for i, (x, y) in pose.items():
        arr[i] = [x, y, 0.0, vis]
    return arr


def _pose(facing=1.0, ankle_l=(0.5, 0.85), ankle_r=(0.5, 0.85),
          knee_l=(0.5, 0.65), knee_r=(0.5, 0.65), hips_y=0.5, lean=0.0):
    """A standing pose with configurable ankle/knee geometry.

    `lean` shifts both shoulders toward the facing direction by that x amount
    (0.0 = perfectly vertical trunk).
    """
    return {
        0: (0.50 + 0.02 * facing, 0.10), 7: (0.48, 0.12), 8: (0.52, 0.12),
        11: (0.50 - 0.03 * facing + lean * facing, 0.30),
        12: (0.50 + 0.03 * facing + lean * facing, 0.30),
        23: (0.48, hips_y), 24: (0.52, hips_y),
        25: knee_l, 26: knee_r, 27: ankle_l, 28: ankle_r,
        29: (ankle_l[0] - 0.02, ankle_l[1] + 0.02),
        30: (ankle_r[0] - 0.02, ankle_r[1] + 0.02),
        31: (ankle_l[0] + 0.02, ankle_l[1] + 0.02),
        32: (ankle_r[0] + 0.02, ankle_r[1] + 0.02),
    }


class TestShinAngle(unittest.TestCase):
    def test_known_angle(self):
        # knee at (0.5,0.65), ankle 0.2 below and dx ahead -> atan(dx/dy)
        dy = 0.2
        for deg in (0.0, 5.0, 10.0, 25.0):
            dx = dy * math.tan(math.radians(deg))
            got = M.shin_angle_deg((0.5, 0.65), (0.5 + dx, 0.85), facing=1.0)
            self.assertAlmostEqual(got, deg, places=4)

    def test_facing_flips_sign(self):
        got = M.shin_angle_deg((0.5, 0.65), (0.53, 0.85), facing=-1.0)
        self.assertLess(got, 0.0)

    def test_none_when_ankle_above_knee(self):
        self.assertIsNone(M.shin_angle_deg((0.5, 0.65), (0.5, 0.55)))

    def test_frame_metric_at_known_pose(self):
        fm = M.frame_metrics(_lms(_pose(ankle_l=(0.535, 0.85), knee_l=(0.5, 0.65))))
        # dx = 0.035, dy = 0.20 -> 9.93 deg, ankle ahead of the knee (facing +1)
        self.assertAlmostEqual(fm["shin_angle_l"], 9.93, places=1)
        self.assertGreater(fm["shin_angle_l"], 0.0)


class TestFrontalMetrics(unittest.TestCase):
    def test_straight_leg_is_zero_valgus(self):
        v = M.frontal_knee_deviation((0.45, 0.5), (0.45, 0.65), (0.45, 0.8), (0.5, 0.5))
        self.assertAlmostEqual(v, 0.0, places=1)

    def test_knee_towards_midline_is_positive_valgus(self):
        # left leg: hip at x=0.45, pelvis centre at 0.5 -> medial is +x
        v = M.frontal_knee_deviation((0.45, 0.5), (0.49, 0.65), (0.45, 0.8), (0.50, 0.5))
        self.assertGreater(v, 0.0)

    def test_knee_away_from_midline_is_varus(self):
        v = M.frontal_knee_deviation((0.45, 0.5), (0.41, 0.65), (0.45, 0.8), (0.50, 0.5))
        self.assertLess(v, 0.0)

    def test_pelvic_tilt_sign(self):
        self.assertGreater(M.pelvic_tilt_deg((0.45, 0.50), (0.55, 0.53)), 0.0)  # right lower
        self.assertLess(M.pelvic_tilt_deg((0.45, 0.53), (0.55, 0.50)), 0.0)     # left lower

    def test_pelvic_tilt_needs_span(self):
        self.assertIsNone(M.pelvic_tilt_deg((0.5, 0.5), (0.505, 0.52)))

    def test_frame_metric_valgus(self):
        fm = M.frame_metrics(_lms(_pose(knee_l=(0.49, 0.65), knee_r=(0.51, 0.65),
                                        ankle_l=(0.44, 0.85), ankle_r=(0.56, 0.85))))
        self.assertIn("knee_valgus_l", fm)
        self.assertGreater(fm["knee_valgus_l"], 0.0)     # both knees collapse inward
        self.assertGreater(fm["knee_valgus_r"], 0.0)


class TestFrameMetricSurface(unittest.TestCase):
    def test_legacy_csv_columns_still_emitted(self):
        fm = M.frame_metrics(_lms(_pose()))
        missing = [k for k in LEGACY_FRAME_KEYS if k not in fm]
        self.assertEqual(missing, [], f"legacy metrics.csv columns disappeared: {missing}")

    def test_new_frame_metrics_present(self):
        fm = M.frame_metrics(_lms(_pose()))
        for k in ("shin_angle_l", "shin_angle_r", "overstride_l", "overstride_r",
                  "knee_flex_l", "knee_flex_r", "pelvic_tilt_deg", "knee_valgus_l",
                  "knee_valgus_r", "trunk_lean_fwd", "hip_ext_l", "hip_ext_r"):
            self.assertIn(k, fm, k)

    def test_signed_keys_declared(self):
        fm = M.frame_metrics(_lms(_pose()))
        negatives = {k for k, v in fm.items()
                     if isinstance(v, float) and v < 0 and not k.startswith("_")}
        self.assertTrue(negatives.issubset(M.SIGNED_FRAME_KEYS), negatives)

    def test_trunk_lean_forward_normalised(self):
        back = M.frame_metrics(_lms(_pose(facing=-1.0, lean=0.05)))
        fwd = M.frame_metrics(_lms(_pose(facing=1.0, lean=0.05)))
        self.assertGreater(fwd["trunk_lean_fwd"], 0.0)
        self.assertGreater(back["trunk_lean_fwd"], 0.0)   # same lean, seen mirrored

    def test_overstride_positive_when_ankle_ahead(self):
        fm = M.frame_metrics(_lms(_pose(facing=1.0, ankle_l=(0.60, 0.85))))
        self.assertGreater(fm["overstride_l"], 0.0)
        fm2 = M.frame_metrics(_lms(_pose(facing=1.0, ankle_l=(0.40, 0.85))))
        self.assertLess(fm2["overstride_l"], 0.0)


def _rows(n=120, fps=30.0, with_legs=True, t0=0.0):
    rows = []
    for i in range(n):
        t = t0 + i / fps
        pose = _pose(ankle_l=(0.50 + 0.05 * math.sin(2 * math.pi * 1.5 * t), 0.85),
                     ankle_r=(0.50 - 0.05 * math.sin(2 * math.pi * 1.5 * t), 0.85))
        lms = _lms(pose)
        if not with_legs:                      # simulate a clip with no lower body
            lms[23:33, 3] = 0.0
            lms[23:33, :2] = np.nan
        fm = M.frame_metrics(lms)
        fm["t_rel"] = t
        rows.append(fm)
    return rows


class TestPlaneGating(unittest.TestCase):
    def test_sagittal_suppresses_frontal(self):
        s = M.analyze_session(_rows(), view_plane="sagittal")
        self.assertEqual(s["view_plane"], "sagittal")
        for k in ("pelvic_drop_deg", "knee_valgus_deg", "arm_swing_sym_pct"):
            self.assertIsNone(s.get(k), k)
            self.assertEqual(s["not_assessed"][k], config.NOT_ASSESSED_TEXT)
        self.assertNotIn("pelvic_drop_deg", s["form"])

    def test_frontal_suppresses_sagittal(self):
        s = M.analyze_session(_rows(), view_plane="frontal")
        self.assertEqual(s["view_plane"], "frontal")
        for k in ("touchdown_shin_deg", "overstride_leg_frac", "trunk_lean_deg"):
            self.assertIsNone(s.get(k), k)
            self.assertEqual(s["not_assessed"][k], config.NOT_ASSESSED_TEXT)

    def test_plane_neutral_metrics_always_present(self):
        for plane in ("sagittal", "frontal"):
            s = M.analyze_session(_rows(), view_plane=plane)
            self.assertIn("cadence_spm", s.get("form", {}), plane)

    def test_unknown_plane_falls_back(self):
        s = M.analyze_session(_rows(), view_plane="sideways")
        self.assertEqual(s["view_plane"], config.DEFAULT_VIEW_PLANE)


class TestDataQualityHonesty(unittest.TestCase):
    def test_low_lower_body_reports_not_assessable(self):
        s = M.analyze_session(_rows(with_legs=False), view_plane="sagittal")
        dq = s["data_quality"]
        self.assertFalse(dq["lower_body_ok"])
        self.assertLess(dq["lower_body_coverage_pct"], config.LOWER_BODY_MIN_COVERAGE_PCT)
        self.assertEqual(dq["note"], config.GAIT_NOT_ASSESSABLE_TEXT)
        self.assertEqual(s["gait_events"]["n"], 0)
        self.assertEqual(s["gait_events"]["note"], config.GAIT_NOT_ASSESSABLE_TEXT)
        self.assertEqual(s["not_assessed"]["gait_events"], config.GAIT_NOT_ASSESSABLE_TEXT)
        self.assertEqual(s["not_assessed"]["touchdown_shin_deg"], config.GAIT_NOT_ASSESSABLE_TEXT)
        self.assertIn(config.GAIT_NOT_ASSESSABLE_TEXT, s["labels"])

    def test_good_lower_body_has_no_such_note(self):
        s = M.analyze_session(_rows(), view_plane="sagittal")
        self.assertTrue(s["data_quality"]["lower_body_ok"])
        self.assertNotIn("note", s["data_quality"])
        self.assertNotIn("gait_events", s["not_assessed"])


class TestAsymmetryIndex(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(M._asym_index(0.30, 0.30), 100.0, places=4)
        # |L-R| = 0.05, mean = 0.30 -> (1 - 0.1667) * 100
        self.assertAlmostEqual(M._asym_index(0.325, 0.275), 83.33, places=1)
        self.assertIsNone(M._asym_index(0.0, 0.0))


class TestRowsSorting(unittest.TestCase):
    def test_out_of_order_rows_are_sorted(self):
        rows = _rows(n=60)
        shuffled = rows[30:] + rows[:30]
        s = M.analyze_session(shuffled, view_plane="sagittal")
        self.assertLessEqual(s["duration_s"], 2.1)
        self.assertGreaterEqual(s["duration_s"], 1.9)
        self.assertEqual(s["n_frames"], 60)

    def test_nan_timestamps_are_dropped(self):
        rows = _rows(n=40)
        rows += [{"t_rel": float("nan"), "left_knee": 10.0}]
        s = M.analyze_session(rows, view_plane="sagittal")
        self.assertEqual(s["n_frames"], 41)


if __name__ == "__main__":
    unittest.main()
