"""Tests for the optional RTMPose backend (pose_backends.py).

No model is downloaded and rtmlib is never really imported: a fake ``rtmlib``
module is injected into ``sys.modules`` whose ``BodyWithFeet`` returns known
pixel keypoints, so the mapping, the normalization, the ROI round-trip and the
None-on-empty behaviour are all exercised offline.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pose_backends as PB  # noqa: E402


# --------------------------------------------------------------------- helpers

def _halpe(points, default=(0.0, 0.0)):
    """(26, 2) HALPE keypoints in pixels; ``points`` is {index: (x, y)}."""
    arr = np.zeros((PB.HALPE26_COUNT, 2), dtype=np.float32)
    arr[:] = default
    for idx, (x, y) in points.items():
        arr[idx] = (x, y)
    return arr


def _fake_module(predict):
    """A stand-in ``rtmlib`` module.

    ``predict(image_shape) -> (keypoints, scores)`` gets the shape of the image
    the backend actually passes to the model (a full frame or a crop), so a
    test can return pixels relative to that image.
    """
    mod = types.ModuleType("rtmlib")

    class BodyWithFeet:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            BodyWithFeet.instances.append(self)

        def __call__(self, image):
            self.calls.append(tuple(int(v) for v in image.shape[:2]))
            return predict(tuple(image.shape[:2]))

    setattr(mod, "BodyWithFeet", BodyWithFeet)   # stand-in module attribute
    return mod


def _pixels(kps_px, batch=True):
    """predict() returning fixed pixel keypoints with full confidence."""
    def predict(shape):
        k = np.asarray(kps_px, dtype=np.float32)
        s = np.ones((k.shape[0],), dtype=np.float32)
        return (k[None], s[None]) if batch else (k, s)
    return predict


def _fractional(frac, score=0.9, fill=(0.5, 0.5)):
    """predict() placing HALPE points at fractions of the image it receives."""
    def predict(shape):
        h, w = shape
        k = np.zeros((PB.HALPE26_COUNT, 2), dtype=np.float32)
        k[:] = (fill[0] * w, fill[1] * h)
        for idx, (fx, fy) in frac.items():
            k[idx] = (fx * w, fy * h)
        return k[None], np.full((1, PB.HALPE26_COUNT), score, np.float32)
    return predict


def _empty(shape):
    return (np.zeros((0, PB.HALPE26_COUNT, 2), np.float32),
            np.zeros((0, PB.HALPE26_COUNT), np.float32))


FRAME = np.zeros((400, 800, 3), dtype=np.uint8)


class _BackendCase(unittest.TestCase):
    """Base case that injects a fake rtmlib for the duration of one test."""

    def backend(self, predict, **kwargs):
        mod = _fake_module(predict)
        patcher = mock.patch.dict(sys.modules, {"rtmlib": mod})
        patcher.start()
        self.addCleanup(patcher.stop)
        return PB.RTMPoseBackend(**kwargs), mod


# ---------------------------------------------------------------- mapping table

class TestMappingTable(unittest.TestCase):
    def test_table_has_one_entry_per_mediapipe_slot(self):
        self.assertEqual(PB.MP33_COUNT, 33)
        self.assertEqual(PB.HALPE26_COUNT, 26)
        self.assertEqual(len(PB.HALPE26_TO_MP33), PB.MP33_COUNT)
        self.assertEqual(len(PB.HALPE26_NAMES), PB.HALPE26_COUNT)
        for src in PB.HALPE26_TO_MP33:
            self.assertIsInstance(src, int)
            self.assertGreaterEqual(src, 0)
            self.assertLess(src, PB.HALPE26_COUNT)

    def test_body_joints_use_their_own_halpe_point(self):
        # the slots metrics.py / gait_events.py / side_identity.py index
        expected = {
            0: 0,                                          # nose
            7: 3, 8: 4,                                    # ears
            11: 5, 12: 6,                                  # shoulders
            13: 7, 14: 8,                                  # elbows
            15: 9, 16: 10,                                 # wrists
            23: 11, 24: 12,                                # hips
            25: 13, 26: 14,                                # knees
            27: 15, 28: 16,                                # ankles
            29: 24, 30: 25,                                # heels
            31: 20, 32: 21,                                # foot_index <- big toes
        }
        for dst, src in expected.items():
            self.assertEqual(PB.HALPE26_TO_MP33[dst], src, f"mp33 slot {dst}")

    def test_derived_slots_never_stand_in_for_a_body_joint(self):
        # 11-16 and 23-32 are the metric joints; 17-22 are MediaPipe's hands,
        # which HALPE-26 lacks and which no metric consumes.
        metric_joints = set(range(11, 17)) | set(range(23, 33))
        self.assertFalse(PB.DERIVED_MP33 & metric_joints)
        self.assertFalse(PB.DERIVED_MP33 & {0, 7, 8})
        self.assertTrue(PB.DERIVED_MP33 <= set(range(PB.MP33_COUNT)))
        self.assertGreater(PB.DERIVED_VIS_SCALE, 0.0)
        self.assertLess(PB.DERIVED_VIS_SCALE, 1.0)


# -------------------------------------------------------------- map function

class TestMapHalpe26(unittest.TestCase):
    def test_normalizes_pixels_and_keeps_the_layout(self):
        kps = _halpe({5: (480.0, 120.0), 11: (400.0, 360.0),
                      24: (420.0, 700.0), 20: (430.0, 760.0)})
        arr = PB.map_halpe26_to_mp33(kps, None, 960, 800)
        self.assertEqual(arr.shape, (33, 4))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr[11, :2], [0.5, 0.15], atol=1e-6)
        np.testing.assert_allclose(arr[23, :2], [400 / 960, 0.45], atol=1e-6)
        np.testing.assert_allclose(arr[29, :2], [420 / 960, 0.875], atol=1e-6)
        np.testing.assert_allclose(arr[31, :2], [430 / 960, 0.95], atol=1e-6)
        self.assertTrue(np.all(arr[:, 2] == 0.0))       # 2-D model: no depth
        self.assertAlmostEqual(float(arr[11, 3]), 1.0, places=6)   # no scores -> 1.0

    def test_scores_become_visibility_and_are_clipped(self):
        kps = _halpe({5: (10.0, 10.0)})
        scores = np.full(PB.HALPE26_COUNT, 1.7, dtype=np.float32)
        arr = PB.map_halpe26_to_mp33(kps, scores, 100, 100)
        self.assertAlmostEqual(float(arr[11, 3]), 1.0, places=6)
        scores[:] = -0.4
        arr = PB.map_halpe26_to_mp33(kps, scores, 100, 100)
        self.assertAlmostEqual(float(arr[11, 3]), 0.0, places=6)

    def test_derived_slots_copy_their_donor_and_are_down_weighted(self):
        kps = _halpe({0: (100.0, 50.0), 1: (110.0, 60.0), 2: (130.0, 60.0),
                      9: (200.0, 400.0), 10: (300.0, 400.0)})
        scores = np.full(PB.HALPE26_COUNT, 0.8, dtype=np.float32)
        arr = PB.map_halpe26_to_mp33(kps, scores, 400, 800)
        np.testing.assert_allclose(arr[1, :2], arr[2, :2])
        np.testing.assert_allclose(arr[3, :2], arr[2, :2])
        np.testing.assert_allclose(arr[9, :2], arr[0, :2])       # mouth <- nose
        np.testing.assert_allclose(arr[17, :2], arr[15, :2])     # pinky <- wrist
        np.testing.assert_allclose(arr[22, :2], arr[16, :2])
        self.assertAlmostEqual(float(arr[2, 3]), 0.8, places=6)          # real
        self.assertAlmostEqual(float(arr[3, 3]), 0.8 * PB.DERIVED_VIS_SCALE,
                               places=6)                                 # derived

    def test_roi_box_round_trip(self):
        kps = _halpe({13: (250.0, 200.0)})     # left knee at the crop centre
        arr = PB.map_halpe26_to_mp33(kps, None, 500, 400,
                                     box=(0.25, 0.5, 0.75, 1.0))
        np.testing.assert_allclose(arr[25, :2], [0.5, 0.75], atol=1e-6)
        with self.assertRaises(ValueError):
            PB.map_halpe26_to_mp33(kps, None, 500, 400, box=(0.5, 0.5, 0.5, 1.0))

    def test_empty_or_unusable_input_is_none(self):
        self.assertIsNone(PB.map_halpe26_to_mp33(None, None, 100, 100))
        self.assertIsNone(PB.map_halpe26_to_mp33(
            np.zeros((0, PB.HALPE26_COUNT, 2), np.float32), None, 100, 100))
        self.assertIsNone(PB.map_halpe26_to_mp33(np.zeros((3,), np.float32),
                                                 None, 100, 100))
        self.assertIsNone(PB.map_halpe26_to_mp33(_halpe({}), None, 0, 0))
        self.assertIsNone(PB.map_halpe26_to_mp33(_halpe({}), None, None, None))

    def test_unbatched_model_output_is_accepted(self):
        kps = _halpe({11: (400.0, 200.0)})
        arr = PB.map_halpe26_to_mp33(kps, np.full(26, 0.9, np.float32), 800, 400)
        np.testing.assert_allclose(arr[23, :2], [0.5, 0.5], atol=1e-6)

    def test_mapped_pose_is_consumable_by_metrics(self):
        # A standing athlete in a 1000 x 1000 frame, in HALPE pixels: if the
        # layout were wrong, frame_metrics would read nonsense.
        import metrics as M
        k = np.zeros((PB.HALPE26_COUNT, 2), dtype=np.float32)
        for idx, (x, y) in {0: (500, 100), 3: (470, 110), 4: (530, 110),
                            5: (460, 250), 6: (540, 250),
                            11: (470, 500), 12: (530, 500),
                            13: (490, 650), 14: (510, 650),
                            15: (480, 820), 16: (520, 820),
                            20: (470, 900), 21: (530, 900),
                            24: (485, 880), 25: (515, 880)}.items():
            k[idx] = (x, y)
        arr = PB.map_halpe26_to_mp33(k, np.full(26, 0.95, np.float32), 1000, 1000)
        fm = M.frame_metrics(arr, aspect=1.0)
        self.assertAlmostEqual(fm["left_knee"], fm["right_knee"], places=3)
        self.assertAlmostEqual(fm["left_hip"], fm["right_hip"], places=3)
        self.assertGreater(fm["left_knee"], 160.0)
        self.assertLess(fm["left_knee"], 180.0)
        self.assertGreater(fm["left_hip"], 160.0)
        self.assertLess(fm["left_hip"], 180.0)
        self.assertAlmostEqual(fm["trunk_lean"], 0.0, places=3)
        # 21 real joints are visible; the 12 derived stand-ins stay < 0.5
        self.assertEqual(int((arr[:, 3] >= 0.5).sum()), 21)

    def test_box_from_landmarks_needs_enough_visible_points(self):
        self.assertIsNone(PB.box_from_landmarks(None))
        self.assertIsNone(PB.box_from_landmarks(np.zeros((33, 4), np.float32)))
        one = np.zeros((33, 4), dtype=np.float32)
        one[11, 3] = 0.9
        self.assertIsNone(PB.box_from_landmarks(one))


# -------------------------------------------------------------------- backend

class TestBackendContract(_BackendCase):
    def test_detect_returns_33x4_normalized_to_the_full_frame(self):
        predict = _fractional({5: (0.10, 0.20), 11: (0.30, 0.60),
                               16: (0.80, 0.95)})
        be, mod = self.backend(predict)
        arr = be.detect(FRAME, 0)
        assert arr is not None
        self.assertEqual(arr.shape, (33, 4))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr[11, :2], [0.10, 0.20], atol=1e-6)
        np.testing.assert_allclose(arr[23, :2], [0.30, 0.60], atol=1e-6)
        np.testing.assert_allclose(arr[28, :2], [0.80, 0.95], atol=1e-6)
        self.assertIsNone(be.seg_frac)
        self.assertIsNone(be.last_world)
        self.assertEqual(mod.BodyWithFeet.instances[-1].kwargs["backend"],
                         "onnxruntime")
        self.assertEqual(mod.BodyWithFeet.instances[-1].calls[0], (400, 800))

    def test_model_is_built_lazily_on_the_first_frame(self):
        be, mod = self.backend(_fractional({}))
        self.assertEqual(mod.BodyWithFeet.instances, [])   # construction is free
        self.assertIsNone(be._model)
        be.detect(FRAME, 0)
        self.assertEqual(len(mod.BodyWithFeet.instances), 1)
        be.detect(FRAME, 1)
        self.assertEqual(len(mod.BodyWithFeet.instances), 1)   # reused

    def test_unbatched_model_output_is_accepted(self):
        be, _ = self.backend(_pixels(_halpe({11: (400.0, 200.0)}), batch=False))
        arr = be.detect(FRAME, 0)
        assert arr is not None
        np.testing.assert_allclose(arr[23, :2], [0.5, 0.5], atol=1e-6)

    def test_detect_does_not_need_a_timestamp(self):
        be, _ = self.backend(_fractional({11: (0.5, 0.5)}))
        self.assertIsNotNone(be.detect(FRAME))

    def test_mode_and_device_flow_into_the_model(self):
        be, mod = self.backend(_fractional({}), mode="performance", device="cuda")
        be.detect(FRAME, 0)
        self.assertEqual(mod.BodyWithFeet.instances[-1].kwargs,
                         {"mode": "performance", "backend": "onnxruntime",
                          "device": "cuda"})
        be2, mod2 = self.backend(_fractional({}))
        be2.detect(FRAME, 0)
        self.assertEqual(mod2.BodyWithFeet.instances[-1].kwargs["mode"], PB.MODE)
        self.assertEqual(mod2.BodyWithFeet.instances[-1].kwargs["device"], PB.DEVICE)

    def test_none_when_the_model_finds_nothing(self):
        be, _ = self.backend(_empty)
        self.assertIsNone(be.detect(FRAME, 0))
        self.assertIsNone(be.detect_still(FRAME))
        self.assertIsNone(be.seg_frac)
        self.assertIsNone(be.last_world)

    def test_accepts_the_poseengine_kwargs_and_ignores_them(self):
        # the app and tools/analyze_video.py build any backend with the same
        # kwargs (model_path/seg/num_poses). RTMPose must accept them silently —
        # a TypeError here used to kill the capture loop at camera open.
        be, _ = self.backend(_fractional({}), model_path="models/pose.task",
                             seg=False, num_poses=1)
        out = be.detect(FRAME, 0)
        assert out is not None
        self.assertEqual(out.shape, (33, 4))
        self.assertIsNone(be.seg_frac)

    def test_none_when_the_model_returns_none(self):
        be, _ = self.backend(lambda shape: None)
        self.assertIsNone(be.detect(FRAME, 0))

    def test_bad_frames_are_none_not_an_exception(self):
        be, _ = self.backend(_fractional({}))
        self.assertIsNone(be.detect(None, 0))
        self.assertIsNone(be.detect(np.zeros((0, 0, 3), np.uint8), 0))
        self.assertIsNone(be.detect_still(np.zeros((5,), np.uint8)))

    def test_min_score_rejects_weak_detections(self):
        weak = lambda shape: (np.zeros((1, PB.HALPE26_COUNT, 2), np.float32),
                              np.full((1, PB.HALPE26_COUNT), 0.05, np.float32))
        be, _ = self.backend(weak, min_score=0.2)
        self.assertIsNone(be.detect(FRAME, 0))
        strong = lambda shape: (np.zeros((1, PB.HALPE26_COUNT, 2), np.float32),
                                np.full((1, PB.HALPE26_COUNT), 0.8, np.float32))
        be2, _ = self.backend(strong, min_score=0.2)
        self.assertIsNotNone(be2.detect(FRAME, 0))

    def test_detect_still_is_stateless(self):
        be, _ = self.backend(_fractional({11: (0.4, 0.4), 15: (0.5, 0.9)}),
                             roi="auto")
        first = be.detect_still(FRAME)
        second = be.detect_still(FRAME)
        assert first is not None and second is not None
        np.testing.assert_allclose(first, second)
        self.assertIsNone(be._prev_box)          # no ROI history from stills
        self.assertIsNone(be.last_box)           # stills never crop

    def test_bad_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            PB.RTMPoseBackend(roi=(0.5, 0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            PB.RTMPoseBackend(roi="maybe")


class TestRoi(_BackendCase):
    def test_roi_crop_is_used_and_mapped_back_to_the_full_frame(self):
        predict = _fractional({11: (0.5, 0.5), 13: (0.5, 0.25)})
        be, mod = self.backend(predict, roi=(0.25, 0.5, 0.75, 1.0))
        frame = np.zeros((800, 1000, 3), np.uint8)
        arr = be.detect(frame, 0)
        assert arr is not None
        # the model saw only the crop: 500 x 400 px
        self.assertEqual(mod.BodyWithFeet.instances[-1].calls[0], (400, 500))
        np.testing.assert_allclose(arr[23, :2], [0.5, 0.75], atol=1e-6)
        np.testing.assert_allclose(arr[25, :2], [0.5, 0.625], atol=1e-6)
        self.assertEqual(be.last_box, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(arr.shape, (33, 4))

    def test_auto_roi_pads_from_the_previous_detection(self):
        frac = {5: (0.40, 0.30), 9: (0.60, 0.70),
                13: (0.42, 0.62), 15: (0.58, 0.90)}
        # padded bbox of those points: x 0.40..0.60, y 0.30..0.90, pad 15 %
        box = (0.40 - 0.15 * 0.20, 0.30 - 0.15 * 0.60,
               0.60 + 0.15 * 0.20, 0.90 + 0.15 * 0.60)

        def predict(shape):
            h, w = shape
            full = (h, w) == (800, 1000)
            k = np.zeros((PB.HALPE26_COUNT, 2), dtype=np.float32)
            k[:] = (0.5 * w, 0.5 * h)
            for idx, (fx, fy) in frac.items():
                if full:                       # full frame: normalized position
                    k[idx] = (fx * w, fy * h)
                else:                          # crop: same position inside the box
                    k[idx] = ((fx - box[0]) / (box[2] - box[0]) * w,
                              (fy - box[1]) / (box[3] - box[1]) * h)
            return k[None], np.full((1, PB.HALPE26_COUNT), 0.9, np.float32)

        be, mod = self.backend(predict, roi="auto")
        frame = np.zeros((800, 1000, 3), np.uint8)
        be.detect(frame, 0)
        self.assertEqual(mod.BodyWithFeet.instances[-1].calls[0], (800, 1000))

        got = be.next_roi_box()
        assert got is not None
        for a, b in zip(got, box):
            self.assertAlmostEqual(a, b, places=6)

        arr = be.detect(frame, 1)
        assert arr is not None
        crop_h, crop_w = mod.BodyWithFeet.instances[-1].calls[-1]
        self.assertEqual((crop_h, crop_w), (624, 260))
        # ... and the keypoints still land on the same full-frame coordinates
        np.testing.assert_allclose(arr[11, :2], [0.40, 0.30], atol=1e-5)
        np.testing.assert_allclose(arr[27, :2], [0.58, 0.90], atol=1e-5)

    def test_roi_falls_back_to_the_full_frame_when_the_crop_is_empty(self):
        calls = []

        def predict(shape):
            calls.append(shape)
            if shape != (800, 1000):        # inside the crop: nothing
                return _empty(shape)
            return _pixels(_halpe({11: (500.0, 400.0)}))(shape)

        be, _ = self.backend(predict, roi=(0.25, 0.5, 0.75, 1.0))
        frame = np.zeros((800, 1000, 3), np.uint8)
        arr = be.detect(frame, 0)
        assert arr is not None
        self.assertEqual(calls[0], (400, 500))     # crop first
        self.assertEqual(calls[1], (800, 1000))    # then the full frame
        np.testing.assert_allclose(arr[23, :2], [0.5, 0.5], atol=1e-6)
        self.assertIsNone(be.last_box)

    def test_reset_roi_returns_to_the_seed(self):
        be, _ = self.backend(_fractional({11: (0.4, 0.4), 15: (0.6, 0.8)}),
                             roi=(0.2, 0.2, 0.8, 0.8))
        frame = np.zeros((400, 800, 3), np.uint8)
        be.detect(frame, 0)
        self.assertNotEqual(be.next_roi_box(), (0.2, 0.2, 0.8, 0.8))
        be.reset_roi()
        self.assertEqual(be.next_roi_box(), (0.2, 0.2, 0.8, 0.8))

    def test_roi_off_by_default(self):
        be, mod = self.backend(_fractional({}))
        self.assertFalse(be.roi)
        self.assertIsNone(be.next_roi_box())
        be.detect(FRAME, 0)
        self.assertEqual(mod.BodyWithFeet.instances[-1].calls, [(400, 800)])


class TestMissingRtmlib(unittest.TestCase):
    def test_import_and_construction_work_without_rtmlib(self):
        with mock.patch.dict(sys.modules, {"rtmlib": None}):
            be = PB.RTMPoseBackend()          # must not import rtmlib
            self.assertFalse(PB.is_available())
            with self.assertRaises(RuntimeError) as ctx:
                be.detect(FRAME, 0)
        message = str(ctx.exception)
        self.assertIn("rtmlib", message)
        self.assertIn("pip install", message)

    def test_import_error_is_chained(self):
        with mock.patch.dict(sys.modules, {"rtmlib": None}):
            with self.assertRaises(RuntimeError) as ctx:
                PB.RTMPoseBackend().detect_still(FRAME)
        self.assertIsNotNone(ctx.exception.__cause__)


class TestEnvAndFactory(_BackendCase):
    def test_env_defaults(self):
        self.assertEqual(PB.DEFAULT_MODE, "lightweight")
        self.assertEqual(PB.DEFAULT_DEVICE, "cpu")
        with mock.patch.dict(os.environ, {"RUNTRACK_RTMPOSE_MODE": " Balanced "}):
            self.assertEqual(PB._env("RUNTRACK_RTMPOSE_MODE", PB.DEFAULT_MODE),
                             "balanced")
        with mock.patch.dict(os.environ, {"RUNTRACK_RTMPOSE_DEVICE": ""}):
            self.assertEqual(PB._env("RUNTRACK_RTMPOSE_DEVICE", PB.DEFAULT_DEVICE),
                             "cpu")
        self.assertEqual(PB.MODE, PB._env("RUNTRACK_RTMPOSE_MODE", PB.DEFAULT_MODE))
        self.assertEqual(PB.DEVICE, PB._env("RUNTRACK_RTMPOSE_DEVICE",
                                           PB.DEFAULT_DEVICE))

    def test_make_backend_selects_and_rejects(self):
        be, _ = self.backend(_fractional({}))
        made = PB.make_backend("rtmpose", mode="lightweight")
        self.assertIsInstance(made, PB.RTMPoseBackend)
        self.assertEqual(made.mode, "lightweight")
        self.assertIsNotNone(be)
        with self.assertRaises(ValueError):
            PB.make_backend("nope")


if __name__ == "__main__":
    unittest.main()
