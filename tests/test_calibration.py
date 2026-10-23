"""Calibration solver tests — synthetic ground truth, no camera.

Every test projects a known board through known intrinsics/extrinsics, so any
error is the solver's, not the rig's.  Noise-free data must recover the inputs
to numerical precision; a noisy case is checked for a sane magnitude only.
"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import calibration as C


def _board_image(cols=9, rows=6, sq=40, margin=2):
    """A rendered chessboard: (cols+1)x(rows+1) squares with a white quiet zone."""
    w = (cols + 1 + 2 * margin) * sq
    h = (rows + 1 + 2 * margin) * sq
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = margin * sq + r * sq
                x0 = margin * sq + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 0
    return img


def _pose(rv, t):
    return cv2.Rodrigues(np.asarray(rv, float))[0], np.asarray(t, float)


def _synthetic_board_views(K, dist, R_world, t_world, board, n=12, seed=0):
    """Project the board at n random poses seen through one camera.

    ``R_world``/``t_world`` place that camera in the world frame (default:
    identity, i.e. the world frame is this camera's frame).
    """
    obj = board.object_points()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        R1, t1 = _pose([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                        rng.uniform(-0.3, 0.3)],
                       [rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1),
                        rng.uniform(0.8, 1.4)])
        R = R_world @ R1
        t = (R_world @ t1.reshape(3, 1)).ravel() + t_world
        pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)
        out.append(pts.reshape(-1, 2).astype(np.float32))
    return out


K1 = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
K2 = np.array([[590.0, 0, 315], [0, 590, 235], [0, 0, 1]])
D0 = np.zeros(5)


class BoardTests(unittest.TestCase):
    def test_object_points_shape_and_spacing(self):
        b = C.Board(9, 6, 0.025)
        obj = b.object_points()
        self.assertEqual(obj.shape, (54, 3))
        self.assertTrue(np.allclose(obj[:, 2], 0))
        self.assertAlmostEqual(float(obj[1, 0] - obj[0, 0]), 0.025, places=6)
        self.assertAlmostEqual(float(obj[9, 1] - obj[0, 1]), 0.025, places=6)

    def test_find_board_rendered(self):
        b = C.Board(9, 6, 0.025)
        found = C.find_board(_board_image(), b)
        self.assertIsNotNone(found)
        assert found is not None  # for the type checker
        pts = np.asarray(found, float)
        self.assertEqual(pts.shape, (54, 2))
        # findChessboardCornersSB starts at its own corner and orientation, so
        # assert the grid itself: 9 x-clusters and 6 y-clusters, one square
        # apart.  Corners carry ~0.05 px of sub-pixel jitter, so cluster them
        # instead of rounding (rounding splits a column across two integers).
        def clusters(vals, tol=1.0):
            vals = np.sort(vals)
            groups = [[vals[0]]]
            for v in vals[1:]:
                if v - groups[-1][-1] <= tol:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            return np.array([float(np.mean(g)) for g in groups])

        xs = clusters(pts[:, 0])
        ys = clusters(pts[:, 1])
        self.assertEqual(len(xs), 9)
        self.assertEqual(len(ys), 6)
        self.assertTrue(np.allclose(np.diff(xs), 40, atol=1.5), xs)
        self.assertTrue(np.allclose(np.diff(ys), 40, atol=1.5), ys)

    def test_find_board_blank_is_none(self):
        self.assertIsNone(C.find_board(np.full((480, 640), 127, np.uint8),
                                       C.Board(9, 6, 0.025)))


class IntrinsicsTests(unittest.TestCase):
    def test_recovers_intrinsics(self):
        board = C.Board(9, 6, 0.025)
        shots = _synthetic_board_views(K1, D0, np.eye(3), np.zeros(3), board)
        res = C.solve_intrinsics(shots, board, (640, 480))
        self.assertNotIn("error", res)
        K = np.array(res["K"])
        self.assertAlmostEqual(K[0][0], 600.0, places=2)
        self.assertAlmostEqual(K[1][1], 600.0, places=2)
        self.assertAlmostEqual(K[0][2], 320.0, places=2)
        self.assertAlmostEqual(K[1][2], 240.0, places=2)
        self.assertLess(res["rms_px"], 0.01)
        self.assertEqual(res["views"], 12)

    def test_needs_enough_views(self):
        board = C.Board(9, 6, 0.025)
        shots = _synthetic_board_views(K1, D0, np.eye(3), np.zeros(3), board, n=3)
        res = C.solve_intrinsics(shots, board, (640, 480))
        self.assertIn("error", res)
        self.assertIn("at least 8", res["error"])


class StereoTests(unittest.TestCase):
    def _pair(self, board, seed=0):
        Rtrue = cv2.Rodrigues(np.array([0.05, -0.55, 0.02]))[0]
        ttrue = np.array([-0.85, 0.03, 0.10])
        c1 = _synthetic_board_views(K1, D0, np.eye(3), np.zeros(3), board, seed=seed)
        c2 = _synthetic_board_views(K2, D0, Rtrue, ttrue, board, seed=seed)
        return Rtrue, ttrue, c1, c2

    def test_recovers_rigid_transform(self):
        board = C.Board(9, 6, 0.025)
        Rtrue, ttrue, c1, c2 = self._pair(board)
        res = C.solve_stereo(c1, c2, board, (640, 480), (640, 480), K1, D0, K2, D0)
        self.assertNotIn("error", res)
        R = np.array(res["R"])
        t = np.array(res["t"])
        ang = np.degrees(np.arccos(np.clip((np.trace(R @ Rtrue.T) - 1) / 2, -1, 1)))
        self.assertLess(ang, 0.01)
        self.assertLess(np.linalg.norm(t - ttrue), 1e-4)
        self.assertAlmostEqual(res["baseline_m"], float(np.linalg.norm(ttrue)), places=5)
        self.assertEqual(res["pairs"], 12)

    def test_pairs_only_where_both_saw_the_board(self):
        board = C.Board(9, 6, 0.025)
        _, _, c1, c2 = self._pair(board)
        c2 = list(c2)
        for i in (0, 1, 2):
            c2[i] = None
        res = C.solve_stereo(c1, c2, board, (640, 480), (640, 480), K1, D0, K2, D0)
        self.assertEqual(res["pairs"], 9)

    def test_needs_enough_pairs(self):
        board = C.Board(9, 6, 0.025)
        _, _, c1, c2 = self._pair(board)
        c2 = [None] * 10 + list(c2[10:])
        res = C.solve_stereo(c1, c2, board, (640, 480), (640, 480), K1, D0, K2, D0)
        self.assertIn("error", res)
        self.assertIn("at least 5", res["error"])


class TriangulationTests(unittest.TestCase):
    def test_dlt_recovers_points(self):
        R, t = _pose([0.1, -0.5, 0.0], [-0.8, 0.05, 0.15])
        P1 = C.projection_matrix(K1, D0, np.eye(3), np.zeros(3))
        P2 = C.projection_matrix(K2, D0, R, t)
        pts3 = np.array([[0.1, 0.2, 2.0], [-0.3, 0.0, 2.5], [0.05, -0.15, 1.5]])
        q1, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K1, D0)
        q2, _ = cv2.projectPoints(pts3, cv2.Rodrigues(R)[0], t, K2, D0)
        X = C.triangulate(P1, P2, q1.reshape(-1, 2), q2.reshape(-1, 2))
        self.assertLess(np.abs(X - pts3).max(), 1e-9)

    def test_projection_matrix_shape(self):
        P = C.projection_matrix(K1, D0, np.eye(3), [0, 0, 0])
        self.assertEqual(P.shape, (3, 4))
        self.assertTrue(np.allclose(P[:, :3], K1))

    def test_reprojection_error_zero_for_exact(self):
        P = C.projection_matrix(K1, D0, np.eye(3), np.zeros(3))
        pts3 = np.array([[0.0, 0.0, 2.0], [0.2, 0.1, 1.8]])
        q, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K1, D0)
        q = q.reshape(-1, 2)
        err = C.reprojection_error(P, pts3, q)
        self.assertLess(float(np.max(err)), 1e-9)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            C.triangulate(np.eye(3, 4), np.eye(3, 4),
                          np.zeros((3, 2)), np.zeros((4, 2)))


class DocumentTests(unittest.TestCase):
    def _doc(self, baseline=0.85):
        board = C.Board(9, 6, 0.025)
        Rtrue = cv2.Rodrigues(np.array([0.05, -0.55, 0.02]))[0]
        ttrue = np.array([-baseline, 0.03, 0.10])
        c1 = _synthetic_board_views(K1, D0, np.eye(3), np.zeros(3), board)
        c2 = _synthetic_board_views(K2, D0, Rtrue, ttrue, board)
        prim = C.solve_intrinsics(c1, board, (640, 480))
        stereo = C.solve_stereo(c1, c2, board, (640, 480), (640, 480),
                                K1, D0, K2, D0)
        second = dict(stereo)
        second.update({"K": K2.tolist(), "dist": D0.tolist(),
                       "image_size": [640, 480], "views": 12})
        return C.build_calibration(prim, {"second": second}, board)

    def test_round_trip_and_quality_pass(self):
        doc = self._doc()
        with tempfile.TemporaryDirectory() as tmp:
            p = C.save(doc, Path(tmp) / "calibration.json")
            self.assertTrue(p.exists())
            back = C.load(p)
        self.assertIsNotNone(back)
        self.assertIn("second", back["views"])
        self.assertEqual(np.array(back["views"]["second"]["P"]).shape, (3, 4))
        q = C.quality(back)
        self.assertTrue(q["ok"], q["reasons"])

    def test_short_baseline_is_flagged(self):
        doc = self._doc(baseline=0.25)
        q = C.quality(doc)
        self.assertFalse(q["ok"])
        self.assertTrue(any("baseline" in r for r in q["reasons"]))

    def test_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calibration.json"
            p.write_text(json.dumps({"schema": 999}), encoding="utf-8")
            self.assertIsNone(C.load(p))

    def test_missing_file(self):
        self.assertIsNone(C.load(Path(tempfile.gettempdir()) / "nope-none.json"))


if __name__ == "__main__":
    unittest.main()


class ApiTests(unittest.TestCase):
    """`/api/calibration` is an operator status endpoint, not a gate."""

    def setUp(self):
        import app as A
        from session_store import Store
        import config
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        A.STORE = Store(root / "t.db", sessions_dir=root / "sessions")
        self.client = A.app.test_client()
        self._old = config.CALIBRATION_PATH
        config.CALIBRATION_PATH = root / "calibration.json"

    def tearDown(self):
        import config
        config.CALIBRATION_PATH = self._old

    def test_missing_file_is_reported_not_raised(self):
        d = self.client.get("/api/calibration").get_json()
        self.assertFalse(d["ok"])
        self.assertFalse(d["quality"]["ok"])
        self.assertTrue(d["quality"]["reasons"])

    def test_present_file_reports_views(self):
        import config
        doc = DocumentTests._doc(DocumentTests("test_round_trip_and_quality_pass"))
        C.save(doc, config.CALIBRATION_PATH)
        d = self.client.get("/api/calibration").get_json()
        self.assertTrue(d["ok"])
        self.assertIn("second", d["views"])
        self.assertGreater(d["views"]["second"]["baseline_m"], 0.5)
        self.assertLess(d["views"]["second"]["stereo_rms_px"], 1.0)
        self.assertTrue(d["quality"]["ok"])
