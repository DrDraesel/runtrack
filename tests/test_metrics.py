"""Tests for RunTrack metrics math (numerical fixtures only here, never live)."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import metrics as M  # noqa: E402


class TestAngleMath(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(M.angle_deg((0, 0), (1, 0), (1, 1)), 90.0, places=4)

    def test_straight_line(self):
        self.assertAlmostEqual(M.angle_deg((0, 0), (1, 0), (2, 0)), 180.0, places=4)

    def test_45_degrees(self):
        self.assertAlmostEqual(M.angle_deg((0, 1), (0, 0), (1, 1)), 45.0, places=4)

    def test_none_input(self):
        self.assertIsNone(M.angle_deg(None, (0, 0), (1, 1)))

    def test_degenerate(self):
        self.assertIsNone(M.angle_deg((1, 0), (1, 0), (1, 0)))


def _lms(pose: dict):
    arr = np.zeros((33, 4), dtype=np.float32)
    arr[:, 0:2] = 0.5
    for i, (x, y, v) in pose.items():
        arr[i] = [x, y, 0.0, v]
    return arr


class TestFrameMetrics(unittest.TestCase):
    def _standing_pose(self):
        return {
            0: (0.50, 0.10, 1.0), 7: (0.48, 0.12, 1.0), 8: (0.52, 0.12, 1.0),
            11: (0.47, 0.30, 1.0), 12: (0.53, 0.30, 1.0),
            13: (0.45, 0.45, 1.0), 14: (0.55, 0.45, 1.0),
            15: (0.44, 0.60, 1.0), 16: (0.56, 0.60, 1.0),
            23: (0.48, 0.62, 1.0), 24: (0.52, 0.62, 1.0),
            25: (0.48, 0.76, 1.0), 26: (0.52, 0.76, 1.0),
            27: (0.48, 0.90, 1.0), 28: (0.52, 0.90, 1.0),
            29: (0.47, 0.93, 1.0), 30: (0.53, 0.93, 1.0),
            31: (0.49, 0.95, 1.0), 32: (0.51, 0.95, 1.0),
        }

    def test_basic_angles(self):
        fm = M.frame_metrics(_lms(self._standing_pose()))
        self.assertIn("left_knee", fm)
        self.assertIn("right_knee", fm)
        self.assertGreater(fm["left_knee"], 150.0)   # near-straight leg
        self.assertLess(fm["left_knee"], 181.0)
        self.assertIn("left_elbow", fm)
        self.assertIn("trunk_lean", fm)
        self.assertLess(abs(fm["trunk_lean"]), 30.0)

    def test_none_frame(self):
        self.assertEqual(M.frame_metrics(None), {})

    def test_low_visibility_gated(self):
        pose = self._standing_pose()
        pose[25] = (0.48, 0.76, 0.1)   # left knee barely visible
        fm = M.frame_metrics(_lms(pose))
        self.assertNotIn("left_knee", fm)
        self.assertIn("right_knee", fm)


class TestSessionAnalysis(unittest.TestCase):
    def _synth_rows(self, fps=30, T=8.0, step_hz=1.5, osc_amp=0.02):
        rows = []
        for i in range(int(fps * T)):
            t = i / fps
            rows.append({
                "t_rel": t,
                "l_ankle_y": 0.90 + 0.05 * math.sin(2 * math.pi * step_hz * t),
                "r_ankle_y": 0.90 + 0.05 * math.sin(2 * math.pi * step_hz * t + math.pi),
                "l_hip_y": 0.62 - 0.5 * osc_amp * math.sin(2 * math.pi * step_hz * t * 2),
                "r_hip_y": 0.62 - 0.5 * osc_amp * math.sin(2 * math.pi * step_hz * t * 2),
                "l_ankle_x": 0.48, "r_ankle_x": 0.52,
                "l_hip_x": 0.48, "r_hip_x": 0.52,
                "_midhip_y": 0.62 - 0.5 * osc_amp * math.sin(2 * math.pi * step_hz * t * 2),
            })
        return rows

    def test_cadence_180spm(self):
        # each foot strikes at 1.5 Hz -> stride rate 90/min per foot -> 180 spm total
        s = M.analyze_session(self._synth_rows(step_hz=1.5))
        self.assertIsNotNone(s.get("cadence_spm"))
        self.assertTrue(150 <= s["cadence_spm"] <= 210, s.get("cadence_spm"))

    def test_cadence_160spm(self):
        s = M.analyze_session(self._synth_rows(step_hz=1.3333))
        self.assertTrue(140 <= s["cadence_spm"] <= 190, s.get("cadence_spm"))

    def test_oscillation_and_power(self):
        s = M.analyze_session(self._synth_rows(), weight_kg=70, height_cm=175)
        self.assertIsNotNone(s.get("vertical_osc_cm"))
        self.assertTrue(1.0 < s["vertical_osc_cm"] < 40.0, s.get("vertical_osc_cm"))
        self.assertIsNotNone(s.get("power_est_watts"))
        self.assertTrue(50 < s["power_est_watts"] < 600, s.get("power_est_watts"))

    def test_form_bands_present(self):
        s = M.analyze_session(self._synth_rows(), weight_kg=70, height_cm=175, speed_kmh=10)
        self.assertIn("form", s)
        self.assertIn("cadence_spm", s["form"])
        self.assertTrue(s["form"]["cadence_spm"]["in_band"])
        self.assertIsNotNone(s.get("form_index"))

    def test_empty_rows(self):
        s = M.analyze_session([])
        self.assertEqual(s.get("n_frames"), 0)

    def test_step_length(self):
        s = M.analyze_session(self._synth_rows(), speed_kmh=10)
        self.assertIsNotNone(s.get("step_length_m"))
        self.assertTrue(0.5 < s["step_length_m"] < 3.0, s.get("step_length_m"))


if __name__ == "__main__":
    unittest.main()
