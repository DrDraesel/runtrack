"""Fusion tests — a synthetic runner with known rotation and known lag.

The 3D ground truth is built first (pelvis yaw, thorax yaw delayed by k frames,
arm swing delayed relative to the contralateral leg), then projected through two
calibrated cameras and pushed back through the real fusion path.  If the module
recovers the inputs, the geometry and the sign conventions are right.
"""
import math
import unittest

import cv2
import numpy as np

import fusion as F

FPS = 30.0
W, H = 640, 480
K1 = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
K2 = np.array([[590.0, 0, 315], [0, 590, 235], [0, 0, 1]])
R = cv2.Rodrigues(np.array([0.02, -0.05, 0.01]))[0]
T = np.array([-0.65, 0.02, 0.08])
def proj(K, Rm, t):
    from calibration import projection_matrix
    return projection_matrix(K, np.zeros(5), Rm, t)


P1 = proj(K1, np.eye(3), np.zeros(3))
P2 = proj(K2, R, T)


def _proj_pts(P, pts3):
    X = np.hstack([np.asarray(pts3, float), np.ones((len(pts3), 1))])
    q = (P @ X.T).T
    return q[:, :2] / q[:, 2:3]


def _lms(P, pts3, visibility=1.0):
    """33x4 landmark array with the JOINT rows filled from a projection."""
    a = np.zeros((33, 4), np.float32)
    a[:, 3] = visibility
    a[:, 0] = 0.5
    a[:, 1] = 0.5
    px = _proj_pts(P, pts3)          # order follows _points3d()
    for k, idx in enumerate(F.JOINTS.values()):
        a[idx, 0] = px[k, 0] / W
        a[idx, 1] = px[k, 1] / H
    return a


def _rot_y(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = v
    return np.array([c * x + s * z, y, -s * x + c * z])


def _points3d(t, pelvis_amp=10.0, thorax_amp=10.0, thorax_lag_frames=5,
              cycle_s=1.2, arm_lag_s=0.20):
    """Runner in the world frame (y down, z forward), metres.

    Pelvis yaws ±pelvis_amp about vertical; the thorax does the same with a
    known delay; the left wrist swings with a known delay relative to the right
    ankle.  Everything else is static, which is enough for the metrics tested.
    """
    ph = 2 * math.pi * t / cycle_s
    p_yaw = math.radians(pelvis_amp) * math.sin(ph)
    t_yaw = math.radians(thorax_amp) * math.sin(
        2 * math.pi * (t - thorax_lag_frames / FPS) / cycle_s)

    l_hip = _rot_y(np.array([-0.12, 0.00, 0.0]), p_yaw)
    r_hip = _rot_y(np.array([0.12, 0.00, 0.0]), p_yaw)
    l_sho = _rot_y(np.array([-0.19, -0.52, 0.0]), t_yaw)
    r_sho = _rot_y(np.array([0.19, -0.52, 0.0]), t_yaw)

    swing = 0.28 * math.sin(ph)                     # right leg forward
    r_ank = np.array([0.10, 0.98, swing])
    l_ank = np.array([-0.10, 0.98, -swing])
    r_knee = np.array([0.10, 0.50, swing * 0.5])
    l_knee = np.array([-0.10, 0.50, -swing * 0.5])

    arm = 0.30 * math.sin(2 * math.pi * (t - arm_lag_s) / cycle_s)
    l_wri = np.array([-0.24, 0.06, arm])            # left arm swings with delay
    r_wri = np.array([0.24, 0.06, -arm])
    l_elb = np.array([-0.22, -0.24, arm * 0.5])
    r_elb = np.array([0.22, -0.24, -arm * 0.5])

    order = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
             "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
    vals = {"l_shoulder": l_sho, "r_shoulder": r_sho, "l_elbow": l_elb,
            "r_elbow": r_elb, "l_wrist": l_wri, "r_wrist": r_wri,
            "l_hip": l_hip, "r_hip": r_hip, "l_knee": l_knee, "r_knee": r_knee,
            "l_ankle": l_ank, "r_ankle": r_ank}
    # the runner stands ~3 m in front of the primary camera (z is depth)
    return np.array([vals[n] for n in order], float) + np.array([0.0, 0.0, 3.0])


class BasisTests(unittest.TestCase):
    def test_default_is_orthonormal_and_right_handed(self):
        right, down, fwd = F.basis_from_gravity()
        for a in (right, down, fwd):
            self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=9)
        self.assertAlmostEqual(float(right @ down), 0.0, places=9)
        self.assertAlmostEqual(float(right @ fwd), 0.0, places=9)
        self.assertAlmostEqual(float(down @ fwd), 0.0, places=9)
        self.assertTrue(np.allclose(right, [1, 0, 0], atol=1e-9))

    def test_tilted_gravity_stays_horizontal(self):
        g = np.array([0.1, 0.95, 0.05])
        g = g / np.linalg.norm(g)
        right, down, fwd = F.basis_from_gravity(g)
        self.assertAlmostEqual(float(down @ fwd), 0.0, places=9)
        self.assertAlmostEqual(float(down @ right), 0.0, places=9)
        self.assertTrue(np.allclose(down, g))

    def test_zero_gravity_raises(self):
        with self.assertRaises(ValueError):
            F.basis_from_gravity([0, 0, 0])


class AngleTests(unittest.TestCase):
    def setUp(self):
        self.b = F.basis_from_gravity()

    def test_square_pelvis_is_zero(self):
        a = F.transverse_angle((-0.12, 0, 0), (0.12, 0, 0), self.b)
        self.assertAlmostEqual(a, 0.0, places=6)

    def test_right_hip_forward_is_positive(self):
        a = F.transverse_angle((-0.12, 0, 0.05), (0.12, 0, 0.05), self.b)
        self.assertAlmostEqual(a, 0.0, places=6)   # both moved: pure translation
        a = F.transverse_angle((-0.12, 0, -0.05), (0.12, 0, 0.05), self.b)
        self.assertGreater(a, 20.0)                # right hip forward
        a2 = F.transverse_angle((-0.12, 0, 0.05), (0.12, 0, -0.05), self.b)
        self.assertLess(a2, -20.0)

    def test_sagittal_angle_sign(self):
        # distal joint forward of the proximal one -> positive
        self.assertGreater(F.sagittal_angle((0, 0, 0), (0, 1, 0.3), self.b), 10.0)
        self.assertLess(F.sagittal_angle((0, 0, 0), (0, 1, -0.3), self.b), -10.0)
        self.assertAlmostEqual(F.sagittal_angle((0, 0, 0), (0, 1, 0), self.b), 0.0)


class FuseFrameTests(unittest.TestCase):
    def test_recovers_joint_positions(self):
        pts = _points3d(0.3)
        f = F.fuse_frame(_lms(P1, pts), (W, H), P1, _lms(P2, pts), (W, H), P2)
        self.assertEqual(len(f["points"]), len(F.JOINTS))
        self.assertEqual(f["dropped"], {})
        for k, name in enumerate(F.JOINTS):
            got = np.array(f["points"][name])
            self.assertLess(np.abs(got - pts[k]).max(), 1e-4, name)

    def test_invisible_joint_is_dropped(self):
        pts = _points3d(0.3)
        a = _lms(P1, pts)
        a[F.JOINTS["l_ankle"], 3] = 0.1
        f = F.fuse_frame(a, (W, H), P1, _lms(P2, pts), (W, H), P2)
        self.assertNotIn("l_ankle", f["points"])
        self.assertIn("l_ankle", f["dropped"])

    def test_inconsistent_point_is_dropped(self):
        pts = _points3d(0.3)
        b = _lms(P2, pts)
        b[F.JOINTS["r_knee"], 0] += 60.0 / W      # 60 px mismatch
        f = F.fuse_frame(_lms(P1, pts), (W, H), P1, b, (W, H), P2)
        # A mismatched pair still intersects cleanly, so the residual cannot
        # catch it; the point lands far behind the runner and the depth gate
        # rejects it instead.
        self.assertNotIn("r_knee", f["points"])
        self.assertIn("depth", f["dropped"]["r_knee"])

    def test_short_landmark_array_raises(self):
        with self.assertRaises(ValueError):
            F.fuse_frame(np.zeros((10, 4)), (W, H), P1,
                         np.zeros((33, 4)), (W, H), P2)


class PhaseLagTests(unittest.TestCase):
    def test_known_delay(self):
        t = np.arange(0, 3, 1 / FPS)
        a = np.sin(2 * np.pi * t / 1.2)
        b = np.sin(2 * np.pi * (t - 0.2) / 1.2)
        lag = F.phase_lag(a, b, FPS, max_lag_s=0.4)
        self.assertIsNotNone(lag)
        self.assertAlmostEqual(lag["lag_s"], 0.2, delta=1.0 / FPS)
        self.assertAlmostEqual(lag["lag_frames"], 6, delta=1)
        self.assertGreater(lag["r"], 0.9)

    def test_sign_is_b_lags_a(self):
        t = np.arange(0, 3, 1 / FPS)
        a = np.sin(2 * np.pi * t / 1.2)
        b = np.sin(2 * np.pi * (t - 0.2) / 1.2)
        self.assertGreater(F.phase_lag(a, b, FPS)["lag_s"], 0)
        self.assertLess(F.phase_lag(b, a, FPS)["lag_s"], 0)

    def test_flat_signal_is_none(self):
        self.assertIsNone(F.phase_lag(np.zeros(90), np.ones(90), FPS))


class CoordinationTests(unittest.TestCase):
    def _frames(self, seconds=4.0, **kw):
        frames = []
        for i in range(int(seconds * FPS)):
            pts = _points3d(i / FPS, **kw)
            frames.append(F.fuse_frame(_lms(P1, pts), (W, H), P1,
                                       _lms(P2, pts), (W, H), P2))
        return frames

    def test_pelvis_thorax_lag_and_amplitude(self):
        frames = self._frames(thorax_lag_frames=5)
        out = F.coordination(frames, FPS)
        self.assertIn("pelvis_thorax", out)
        pt = out["pelvis_thorax"]
        self.assertAlmostEqual(pt["lag_s"], 5 / FPS, delta=1.5 / FPS)
        self.assertGreater(pt["r"], 0.8)
        self.assertAlmostEqual(pt["pelvis_deg"], 20.0, delta=3.0)
        self.assertAlmostEqual(pt["thorax_deg"], 20.0, delta=3.0)

    def test_contralateral_arm_leg_lag(self):
        frames = self._frames(arm_lag_s=0.2)
        out = F.coordination(frames, FPS)
        self.assertIn("arm_leg", out)
        al = out["arm_leg"]["left_arm_right_leg"]
        self.assertAlmostEqual(al["lag_s"], 0.2, delta=1.5 / FPS)
        self.assertGreater(al["r"], 0.8)
        self.assertGreater(al["arm_amp_deg"], 10.0)
        self.assertGreater(al["leg_amp_deg"], 10.0)

    def test_short_series_is_not_assessed(self):
        out = F.coordination(self._frames(0.5), FPS)
        self.assertNotIn("pelvis_thorax", out)
        self.assertTrue(out["not_assessed"])

    def test_bad_fps_raises(self):
        with self.assertRaises(ValueError):
            F.coordination([], 0.0)


if __name__ == "__main__":
    unittest.main()
