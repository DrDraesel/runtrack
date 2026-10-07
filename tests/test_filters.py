"""Tests for the 1-Euro landmark smoothing (filters.py)."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import filters  # noqa: E402


class TestOneEuro(unittest.TestCase):
    def test_constant_input_is_passed_through(self):
        f = filters.OneEuroFilter()
        out = [f(np.array([0.25, 0.5, 0.0], dtype=np.float32), t=i / 30.0) for i in range(40)]
        self.assertAlmostEqual(float(out[-1][0]), 0.25, places=4)
        self.assertAlmostEqual(float(out[-1][1]), 0.5, places=4)

    def test_jitter_is_reduced(self):
        rng = np.random.default_rng(7)
        f = filters.OneEuroFilter()
        raw, filt = [], []
        for i in range(120):
            v = 0.5 + 0.01 * rng.standard_normal()
            raw.append(v)
            filt.append(float(f(np.array([v], dtype=np.float32), t=i / 30.0)[0]))
        raw_sd = float(np.std(raw[30:]))
        filt_sd = float(np.std(filt[30:]))
        self.assertLess(filt_sd, 0.6 * raw_sd, f"jitter {filt_sd:.4f} vs raw {raw_sd:.4f}")

    def test_fast_movement_is_preserved(self):
        # 2 Hz limb movement at 30 fps: the filter must not eat the amplitude
        f = filters.OneEuroFilter()
        vals, out = [], []
        for i in range(120):
            v = 0.1 * math.sin(2 * math.pi * 2.0 * (i / 30.0))
            vals.append(v)
            out.append(float(f(np.array([v], dtype=np.float32), t=i / 30.0)[0]))
        self.assertGreater(float(np.max(out[30:])), 0.45 * 0.1,
                           "fast movement was over-smoothed (too much lag)")
        self.assertLess(float(np.max(out[30:])), 0.105)

    def test_disabled_filter_is_identity(self):
        with mock.patch.object(config, "ONEEURO_ENABLED", False):
            f = filters.OneEuroFilter()
            x = np.arange(6, dtype=np.float32).reshape(3, 2)
            np.testing.assert_allclose(f(x, t=0.0), x)
            np.testing.assert_allclose(f(x + 5, t=1 / 30), x + 5)

    def test_reset_restarts_state(self):
        f = filters.OneEuroFilter()
        f(np.array([0.0], dtype=np.float32), t=0.0)
        f(np.array([0.9], dtype=np.float32), t=1 / 30)
        f.reset()
        out = f(np.array([0.9], dtype=np.float32), t=2 / 30)
        self.assertAlmostEqual(float(out[0]), 0.9, places=5)


class TestLandmarkFilter(unittest.TestCase):
    def _lms(self, x=0.5, vis=0.9):
        a = np.zeros((33, 4), dtype=np.float32)
        a[:, 0] = x
        a[:, 1] = 0.6
        a[:, 3] = vis
        return a

    def test_none_passthrough(self):
        self.assertIsNone(filters.LandmarkFilter()(None))

    def test_visibility_column_untouched(self):
        lf = filters.LandmarkFilter()
        for i in range(10):
            lms = self._lms(x=0.5 + 0.01 * i, vis=0.42)
            out = lf(lms, t=i / 30.0)
        self.assertAlmostEqual(float(out[5, 3]), 0.42, places=6)
        self.assertEqual(out.shape, (33, 4))

    def test_smoothing_matches_manual_series(self):
        lf = filters.LandmarkFilter()
        f = filters.OneEuroFilter()
        for i in range(20):
            t = i / 30.0
            v = 0.5 + 0.02 * math.sin(i)
            lms = self._lms(x=v)
            a = float(lf(lms, t=t)[7, 0])
            b = float(f(np.array([v, v, 0.0, 0.0], dtype=np.float32), t=t)[0])
            self.assertAlmostEqual(a, b, places=5)

    def test_signal_smooth_helper(self):
        x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        out = filters.signal_smooth(x, min_cutoff=1.0, beta=0.2)
        self.assertEqual(len(out), len(x))
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
