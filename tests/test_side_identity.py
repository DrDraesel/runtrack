"""Tests for side_identity.SideIdentity — left/right label stabilisation.

The scenarios mirror the real failure: MediaPipe keeps a limb's position
continuous but flips its left/right label (typically while one leg or arm is
raised / crossing).  The stabiliser must (a) undo flips, (b) never invent a
flip on a genuinely continuous (even crossing) trajectory, (c) hold identity
while the two sides are glued, (d) report an occluded side as hidden.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from side_identity import SideIdentity  # noqa: E402

FPS = 30.0


def leg_points(ankle_xy, side="left"):
    """A limb chain around an ankle position, for the given side's indices."""
    ax, ay = ankle_xy
    base = 23 if side == "left" else 24
    offs = {0: (ax + 0.01, ay - 0.42),        # hip
            2: (ax - 0.01, ay - 0.16),        # knee
            4: (ax, ay),                      # ankle
            6: (ax + 0.02, ay + 0.01),        # heel
            8: (ax + 0.04, ay + 0.02)}        # foot
    return {base + k: v for k, v in offs.items()}


def arm_points(wrist_xy, side="left"):
    wx, wy = wrist_xy
    base = 11 if side == "left" else 12
    offs = {0: (wx + 0.02, wy - 0.36),        # shoulder
            2: (wx + 0.03, wy - 0.18),        # elbow
            4: (wx, wy),                      # wrist
            6: (wx, wy + 0.02), 8: (wx, wy + 0.03), 10: (wx, wy + 0.04)}
    return {base + k: v for k, v in offs.items()}


def make_frame(legs, arms, vis=0.9):
    """Assemble a (33,4) landmark frame from per-side point dicts."""
    arr = np.zeros((33, 4), dtype=np.float32)
    for pts in (legs[0], legs[1], arms[0], arms[1]):
        for idx, (x, y) in pts.items():
            arr[idx, 0], arr[idx, 1], arr[idx, 3] = x, y, vis
    return arr


def trajectories(n, phase_l=0.0, phase_r=math.pi, amp=0.17, freq=1.4,
                 glue_legs=False):
    """Per-side ankle ground-truth positions for n frames."""
    ts = np.arange(n) / FPS
    out = {}
    for side, phase in (("left", phase_l), ("right", phase_r)):
        y = 0.55 + amp * np.sin(2 * math.pi * freq * ts + phase)
        x = 0.50 + 0.10 * np.cos(2 * math.pi * freq * ts + phase + 0.5)
        out[side] = np.stack([x, y], axis=1)
    if glue_legs:                        # both legs move together (glued)
        out["right"] = out["left"].copy()
    return out


def wrist_targets(i):
    """Ground-truth wrist positions, left first."""
    return ((0.42, 0.32 + 0.05 * math.sin(i / 9.0)),
            (0.58, 0.32 - 0.05 * math.sin(i / 9.0)))


def build_frame(truth, i, flip_legs=False, flip_arms=False):
    """One input frame. The flip_* flags make the LEFT indices carry the
    right limb's positions — exactly what a label flip looks like."""
    l, r = truth["left"][i], truth["right"][i]
    al, ar = wrist_targets(i)
    if flip_legs:
        l, r = r, l
    if flip_arms:
        al, ar = ar, al
    return make_frame(
        (leg_points(tuple(l), "left"), leg_points(tuple(r), "right")),
        (arm_points(al, "left"), arm_points(ar, "right")))


def leg_err(out_lms, truth, side, t):
    """|ankle_y(out) - truth| for one side (normalized units)."""
    idx = 27 if side == "left" else 28
    return abs(float(out_lms[idx, 1]) - float(truth[side][t, 1]))

class TestSideIdentity(unittest.TestCase):
    def _run(self, frames, si=None, t0=0.0):
        si = si or SideIdentity()
        outs, infos = [], []
        for i, f in enumerate(frames):
            out, info = si.update(f, t0 + i / FPS)
            outs.append(out)
            infos.append(info)
        return si, outs, infos

    # ---------------------------------------------------------------- stable
    def test_clean_sequence_is_untouched(self):
        truth = trajectories(120)
        frames = [build_frame(truth, i) for i in range(120)]
        si, outs, infos = self._run(frames)
        self.assertEqual(si.info()["total_swaps"], 0)
        for i in range(120):
            np.testing.assert_allclose(outs[i][:, :3], frames[i][:, :3], atol=1e-6)
        self.assertEqual(infos[-1]["hidden"]["leg"], [])
        self.assertEqual(infos[-1]["hidden"]["arm"], [])

    # ------------------------------------------------------------- label flip
    def test_flip_is_corrected(self):
        truth = trajectories(120)
        frames = [build_frame(truth, i, flip_legs=i >= 40, flip_arms=i >= 40)
                  for i in range(120)]
        si, outs, infos = self._run(frames)
        self.assertGreaterEqual(si.info()["swaps"]["leg"], 1)
        self.assertGreaterEqual(si.info()["swaps"]["arm"], 1)
        # after a short confirmation window the tracked sides follow the truth
        err = np.mean([leg_err(outs[t], truth, "left", t) for t in range(46, 120)])
        self.assertLess(err, 0.012)

    def test_flip_back_is_corrected_again(self):
        truth = trajectories(180)
        frames = [build_frame(truth, i, flip_legs=40 <= i < 100,
                              flip_arms=40 <= i < 100) for i in range(180)]
        si, outs, infos = self._run(frames)
        self.assertGreaterEqual(si.info()["swaps"]["leg"], 2)
        for lo, hi in ((46, 100), (106, 180)):
            err = np.mean([leg_err(outs[t], truth, "left", t) for t in range(lo, hi)])
            self.assertLess(err, 0.012, f"left side wrong between {lo}-{hi}")

    # ------------------------------------------------------- real crossings
    def test_no_false_flip_on_real_crossing(self):
        # both legs swing at a small phase offset -> the paths cross twice per
        # cycle; the labels are CORRECT throughout, so no correction may fire
        truth = trajectories(150, phase_l=0.0, phase_r=0.35, amp=0.25)
        frames = [build_frame(truth, i) for i in range(150)]
        si, outs, infos = self._run(frames)
        self.assertEqual(si.info()["total_swaps"], 0)

    # ------------------------------------------------------------ occlusion
    def test_hidden_side_reported(self):
        truth = trajectories(120)
        frames = [build_frame(truth, i) for i in range(120)]
        for i in range(30, 70):                    # far (right) leg occluded
            frames[i][[24, 26, 28, 30, 32], 3] = 0.05
        si, outs, infos = self._run(frames)
        self.assertEqual(si.info()["total_swaps"], 0)
        self.assertIn("right", infos[55]["hidden"]["leg"])
        self.assertEqual(infos[110]["hidden"]["leg"], [])
        # after the leg reappears the tracked left side is still the truth
        err = np.mean([leg_err(outs[t], truth, "left", t) for t in range(75, 120)])
        self.assertLess(err, 0.012)

    # --------------------------------------------------------------- glued
    def test_glued_sides_never_swap(self):
        truth = trajectories(90, glue_legs=True)
        frames = [build_frame(truth, i) for i in range(90)]
        si, outs, infos = self._run(frames)
        self.assertEqual(si.info()["total_swaps"], 0)

    def test_pose_lost_is_safe(self):
        si = SideIdentity()
        out, info = si.update(None, 0.0)
        self.assertIsNone(out)
        self.assertIn("swaps", info)
        # tracking survives a short dropout and continues
        truth = trajectories(60)
        frames = [build_frame(truth, i) for i in range(60)]
        frames[20] = None
        for i, f in enumerate(frames):
            si.update(f, i / FPS)
        self.assertEqual(si.info()["total_swaps"], 0)

    # ---------------------------------------------------------------- arms
    def test_arm_flip_corrected(self):
        truth = trajectories(120)
        frames = [build_frame(truth, i, flip_arms=i >= 50) for i in range(120)]
        si, outs, infos = self._run(frames)
        self.assertGreaterEqual(si.info()["swaps"]["arm"], 1)
        # left wrist (idx 15) follows the left arm target after correction
        err = 0.0
        for t in range(56, 120):
            err += abs(float(outs[t][15, 1]) - wrist_targets(t)[0][1])
        self.assertLess(err / 64.0, 0.01)


if __name__ == "__main__":
    unittest.main()
