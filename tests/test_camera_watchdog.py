"""Capture-loop camera watchdogs (session-43 freeze + blocked extra views).

Two live bugs are pinned here:

  * A primary camera that stops delivering used to spin forever in silence —
    ``rows``/``elapsed`` frozen, ``error`` null, ``fps`` stuck at its last
    value, the UI looked hung.  Now the state snapshot keeps refreshing and the
    stall becomes a loud, final error that finalizes the running session with
    ``camera_stalled``.
  * ``cap2.read()`` / ``cap3.read()`` ran on the capture thread, so one camera
    blocking ~1 s per frame capped EVERY view at ~1 fps (measured live: a dead
    RealSense made fps = fps2 = fps3 = 1.0 with error null).  Extra views are
    now read on their own threads; one that never delivers is dropped with a
    warning while the others keep recording at full rate.

Hardware-free: cameras and the pose backend are fakes, the store is mocked and
the sessions directory is a temp directory.  No real camera is ever opened.
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import config             # noqa: E402

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeCap:
    """Capture double: ``good`` frames, then not-ok forever (or a slow read)."""

    def __init__(self, good=10 ** 9, block_s=0.0, fps=30.0):
        self.good = int(good)
        self.block_s = float(block_s)
        self.fps = float(fps)
        self.n = 0
        self.released = False
        self._lock = threading.Lock()

    def isOpened(self):
        return True

    def read(self):
        if self.block_s:
            time.sleep(self.block_s)
        with self._lock:
            self.n += 1
            n = self.n
        if n > self.good:
            return False, None
        return True, FRAME.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True


class FakeEngine:
    """Pose backend double: no landmarks, so rows carry only timing."""

    seg_frac = None
    last_world = None

    def __init__(self, *a, **kw):
        pass

    def detect(self, frame, ts):
        return None

    def close(self):
        pass


class _LoopCase(unittest.TestCase):
    """Isolates the global Live state, temp sessions dir, faked hardware."""

    ATTRS = ("active", "error", "session", "source", "source_label", "source2",
             "source2_label", "source3", "source3_label", "live", "live2",
             "live3", "fps", "fps2", "fps3", "warn2", "warn3", "frame_t1",
             "frame_t2", "frame_t3", "frames1", "frames2", "frames3", "open_t",
             "stalled", "stalled2", "stalled3", "latest_jpeg", "latest_jpeg2",
             "latest_jpeg3", "stop_evt", "thread", "media", "rig", "view_plane",
             "paused")

    def setUp(self):
        self._saved = {k: getattr(appmod.L, k) for k in self.ATTRS}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sdir = Path(self.tmp.name)
        self._patches = [
            mock.patch.object(config, "SESSIONS_DIR", self.sdir),
            mock.patch.object(config, "RIG_CHECK_ENABLED", False),
            mock.patch.object(appmod, "make_backend", FakeEngine),
        ]
        for p in self._patches:
            p.start()
        with appmod.L.lock:
            appmod.L.reset()
        self.client = appmod.app.test_client()
        appmod.app.config["TESTING"] = True

    def tearDown(self):
        appmod.L.stop_evt.set()
        t = appmod.L.thread
        if t is not None and t.is_alive():
            t.join(timeout=6)
        appmod.L.thread = None
        for k, v in self._saved.items():
            setattr(appmod.L, k, v)
        for p in reversed(self._patches):
            p.stop()

    # -- helpers --------------------------------------------------------
    def start_loop(self, caps, src):
        """Run the real capture loop against fake captures."""
        it = iter(caps)

        def _open(*a, **k):
            return next(it)

        p = mock.patch.object(appmod.cameras, "open_source", side_effect=_open)
        p.start()
        self.addCleanup(p.stop)
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source = dict(src)
            appmod.L.source_label = "USB camera 2"
            appmod.L.source2 = src.get("source2")
            appmod.L.source2_label = ("USB camera 4" if src.get("source2")
                                      else "none")
            appmod.L.source3 = src.get("source3")
            appmod.L.source3_label = ("USB camera 6" if src.get("source3")
                                      else "none")
        appmod.L.thread = threading.Thread(target=appmod._loop, args=(src,),
                                           daemon=True)
        appmod.L.thread.start()

    def start_session(self, duration=30.0):
        """Start a session through the API with the row-writing store mocked."""
        with appmod.L.lock:               # the API requires an open camera
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
        with mock.patch.object(appmod.STORE, "upsert_patient", return_value=1), \
                mock.patch.object(appmod.STORE, "create_session", return_value=77):
            r = self.client.post("/api/session/start",
                                 json={"name": "Dana", "phone": "444",
                                       "duration_s": duration})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        with appmod.L.lock:
            sess = appmod.L.session
            sess["latch"] = None            # no model, no keyframes in this test
            sess["stills"] = None
            sess["source1_label"] = "USB camera 2"
            sid = sess["id"]
        return sid

    def wait_until(self, cond, timeout=8.0, interval=0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(interval)
        return bool(cond())

    def state(self):
        return self.client.get("/api/state").get_json()


class TestViewStatusShape(_LoopCase):
    def test_state_exposes_all_three_views(self):
        d = self.state()
        self.assertEqual(set(d["view_status"]), {"1", "2", "3"})
        for key, v in d["view_status"].items():
            self.assertEqual(set(v), {"ok", "stalled", "age_s", "frames"},
                             f"view {key} keys")
            self.assertIsInstance(v["ok"], bool)
            self.assertIsInstance(v["stalled"], bool)
            self.assertIsInstance(v["frames"], int)
            self.assertTrue(v["age_s"] is None or isinstance(v["age_s"], float))
        # top-level additives, and every pre-existing key is still there
        for k in ("frame_age_s", "frame_age2_s", "frame_age3_s",
                  "stalled", "stalled2", "stalled3", "joint_study"):
            self.assertIn(k, d)
        for k in ("active", "source", "error", "fps", "fps2", "fps3", "live",
                  "live2", "live3", "view_plane", "plane2", "plane3",
                  "warning2", "warning3", "has_camera2", "has_camera3", "rig",
                  "session", "playback", "media"):
            self.assertIn(k, d)
        # idle: no camera open, so nothing is stalled and nothing is "ok"
        self.assertFalse(d["stalled"])
        self.assertIsNone(d["frame_age_s"])
        self.assertIsNone(d["joint_study"])


class TestSilentPrimary(_LoopCase):
    def test_state_keeps_refreshing_while_reads_fail(self):
        """Bug 1: the snapshot must move even when the primary read fails."""
        primary = FakeCap(good=3)
        with mock.patch.object(config, "CAM_STALL_S", 30.0):
            self.start_loop([primary], {"kind": "usb", "index": 2})
            self.assertTrue(self.wait_until(
                lambda: appmod.L.frames1 >= 3, timeout=5.0))
            t0 = time.monotonic()
            d1 = self.state()
            self.wait_until(lambda: False, timeout=0.9)   # let time pass
            d2 = self.state()
        self.assertIsNone(d2["error"], "a stall below CAM_STALL_S is not an error")
        self.assertTrue(d2["active"])
        self.assertGreaterEqual(d2["frame_age_s"], 0.5,
                                "frame_age_s must grow while no frame arrives")
        self.assertGreater(d2["frame_age_s"], d1["frame_age_s"])
        self.assertEqual(d2["fps"], 0.0, "fps must decay instead of freezing")
        self.assertLess(time.monotonic() - t0, 30.0)

    def test_stalled_primary_is_a_loud_final_error(self):
        primary = FakeCap(good=4)
        self.start_session()
        finish = mock.MagicMock(return_value=True)
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session", finish), \
                mock.patch.object(appmod.R, "build"):
            self.start_loop([primary], {"kind": "usb", "index": 2})
            ok = self.wait_until(lambda: appmod.L.error is not None, timeout=10.0)
        self.assertTrue(ok, "the stall must raise L.error")
        err = appmod.L.error
        self.assertIn("USB camera 2", err)
        self.assertIn("stopped delivering frames", err)
        self.assertIn("no frame for", err)
        self.assertFalse(appmod.L.active)
        # the state carries the same verdict + the age of the last frame
        d = self.state()
        self.assertTrue(d["stalled"])
        self.assertTrue(d["view_status"]["1"]["stalled"])
        self.assertFalse(d["view_status"]["1"]["ok"])
        self.assertGreaterEqual(d["frame_age_s"], 2.0)
        self.assertIsNone(d["session"], "the stalled session must be finalized")

        self.assertTrue(finish.called, "the running session must be finalized")
        args, kwargs = finish.call_args
        self.assertEqual(kwargs.get("status"), "incomplete")
        summary = args[3]
        self.assertTrue(summary["data_quality"]["camera_stalled"])
        self.assertEqual(summary["data_quality"]["camera_stalled_source"],
                         "USB camera 2")
        self.assertIn("stopped delivering frames", summary["note"])
        self.assertIn("angle_captures", summary)
        self.assertIn("joint_study", summary)


class TestDeadExtraView(_LoopCase):
    def test_never_delivering_third_view_is_dropped(self):
        primary = FakeCap()                  # keeps delivering forever
        second = FakeCap()                   # healthy other extra view
        src = {"kind": "usb", "index": 2,
               "source2": {"kind": "usb", "index": 4},
               "source3": {"kind": "usb", "index": 6}}
        with mock.patch.object(config, "CAM_VIEW_DROP_AFTER", 3):
            self.start_loop([primary, second, FakeCap(good=0)], src)
            ok = self.wait_until(lambda: appmod.L.warn3 is not None, timeout=10.0)
        self.assertTrue(ok, "a view that never delivers must be dropped")
        msg = appmod.L.warn3
        self.assertIn("stopped delivering frames", msg)
        self.assertIn("dropped", msg)
        self.assertIn("keep recording", msg)
        d = self.state()
        self.assertTrue(d["stalled3"])
        self.assertTrue(d["view_status"]["3"]["stalled"])
        self.assertFalse(d["view_status"]["3"]["ok"])
        self.assertEqual(d["view_status"]["3"]["frames"], 0)
        # view 1 and the OTHER extra view are untouched and still recording
        self.assertFalse(d["stalled"])
        self.assertTrue(d["view_status"]["1"]["ok"])
        self.assertTrue(d["view_status"]["2"]["ok"], "the healthy second view")
        self.assertFalse(d["stalled2"])
        before1, before2 = appmod.L.frames1, appmod.L.frames2
        self.wait_until(lambda: appmod.L.frames1 > before1 + 3
                        and appmod.L.frames2 > before2 + 3, timeout=3.0)
        self.assertGreater(appmod.L.frames1, before1 + 3)
        self.assertGreater(appmod.L.frames2, before2 + 3)
        self.assertTrue(appmod.L.active)
        self.assertIsNone(appmod.L.error)

    def test_blocked_second_view_does_not_slow_the_primary(self):
        """Bug 2: a 0.4 s blocking read on view 2 must not cap view 1."""
        primary = FakeCap()
        slow = FakeCap(block_s=0.4)
        src = {"kind": "usb", "index": 2,
               "source2": {"kind": "usb", "index": 4}}
        with mock.patch.object(config, "CAM_VIEW_DROP_AFTER", 10 ** 6):
            # keep view 2 alive (never dropped) so the *blocking* read is what
            # would have throttled the primary
            self.start_loop([primary, slow], src)
            t0 = time.monotonic()
            n0 = appmod.L.frames1
            self.wait_until(lambda: False, timeout=1.5)
            got = appmod.L.frames1 - n0
            elapsed = time.monotonic() - t0
        # the old sequential loop managed ~1 frame per blocking read (~3)
        self.assertGreater(got, 15,
                           f"only {got} primary frames in {elapsed:.1f}s")
        self.assertTrue(appmod.L.active)
        self.assertIsNone(appmod.L.error)


class TestSessionStartGuard(_LoopCase):
    def test_refuses_to_start_on_a_stalled_primary(self):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
            appmod.L.stalled = True
            appmod.L.error = ("camera 'USB camera 2' stopped delivering frames: "
                              "no frame for 3.0 s (limit 2.0 s) — capture stopped")
        r = self.client.post("/api/session/start",
                             json={"name": "Dana", "phone": "444"})
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertFalse(d["ok"])
        self.assertIn("not delivering frames", d["error"])
        self.assertIn("Reopen the camera", d["error"])

    def test_warns_when_an_extra_view_was_dropped(self):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
            appmod.L.source2 = {"kind": "usb", "index": 4}
            appmod.L.source2_label = "USB camera 4"
            appmod.L.stalled2 = True
            appmod.L.warn2 = ("second view (USB camera 4) stopped delivering "
                              "frames — dropped; the other views keep recording")
        with mock.patch.object(appmod.STORE, "upsert_patient", return_value=1), \
                mock.patch.object(appmod.STORE, "create_session", return_value=78):
            r = self.client.post("/api/session/start",
                                 json={"name": "Dana", "phone": "444"})
            appmod.L.session = self._saved["session"]
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn("warning", d)
        self.assertIn("second view", d["warning"])
        self.assertIn("not being recorded", d["warning"])


class TestPausedPlayback(_LoopCase):
    def test_pause_freezes_extra_views_and_is_not_a_stall(self):
        """A paused video is not a stalled camera, and its views must not drift."""
        primary = FakeCap(fps=30.0)
        second = FakeCap(fps=30.0)
        src = {"kind": "file", "path": "clip.mp4",
               "source2": {"kind": "file", "path": "clip.mp4"}}
        with mock.patch.object(config, "CAM_STALL_S", 1.0):
            self.start_loop([primary, second], src)
            self.assertTrue(self.wait_until(lambda: appmod.L.frames2 >= 2,
                                           timeout=5.0))
            with appmod.L.lock:
                appmod.L.paused = True
            self.wait_until(lambda: False, timeout=0.4)   # let the pause engage
            n1, n2 = appmod.L.frames1, appmod.L.frames2
            self.wait_until(lambda: False, timeout=1.6)   # > CAM_STALL_S
            self.assertEqual(appmod.L.frames1, n1)
            self.assertEqual(appmod.L.frames2, n2,
                             "a paused extra view must not keep advancing")
            d = self.state()
            self.assertFalse(d["stalled"], "a pause is not a camera stall")
            self.assertFalse(d["view_status"]["1"]["stalled"])
            with appmod.L.lock:
                appmod.L.paused = False
            self.wait_until(lambda: appmod.L.frames1 > n1, timeout=3.0)
            self.assertGreater(appmod.L.frames1, n1)


class TestDepthTravelsWithFrame(unittest.TestCase):
    """cap2.depth_at / intrinsics must stay paired with the frame they came from."""

    def test_reader_payload_carries_the_depth_sampler(self):
        class DepthFake(FakeCap):
            intrinsics = {"fx": 391.1, "fy": 391.1, "cx": 320.0,
                          "cy": 240.0, "width": 640, "height": 480}

            def __init__(self):
                super().__init__()
                self.depth_at = lambda nx, ny, r=4: 2.5

        cap = DepthFake()
        reader = appmod._ViewReader(cap, "view2").start()
        try:
            payload = None
            deadline = time.monotonic() + 3.0
            while payload is None and time.monotonic() < deadline:
                payload = reader.latest(timeout=0.05)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["intrinsics"]["fx"], 391.1)
            self.assertAlmostEqual(payload["depth_at"](0.5, 0.5), 2.5)
        finally:
            reader.stop()
        self.assertTrue(cap.released, "the owner thread releases its capture")


if __name__ == "__main__":
    unittest.main()
