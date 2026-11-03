"""F3 — abnormal end states must not leave a session 'running'.

* An exception in the capture loop used to set ``L.error`` and leave the DB row
  running forever.  It now finalizes as ``capture_error`` / ``incomplete``.
* A process that dies (crash, power loss) leaves orphan ``running`` rows; the
  store closes them at startup as ``incomplete`` with a note.
* A view the watchdog dropped is recorded in the summary's data_quality, with a
  plain-English note, instead of surviving only in the live state.
"""
import json
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
from session_store import Store  # noqa: E402

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeCap:
    def __init__(self, fps=30.0):
        self.fps = float(fps)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, FRAME.copy()

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True


class FakeEngine:
    seg_frac = None
    last_world = None

    def __init__(self, *a, **kw):
        pass

    def detect(self, frame, ts):
        return None

    def close(self):
        pass


class EndStateCase(unittest.TestCase):
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
        root = Path(self.tmp.name)
        self.store = Store(root / "t.db", sessions_dir=root / "sessions")
        self._patches = [
            mock.patch.object(config, "SESSIONS_DIR", root / "sessions"),
            mock.patch.object(config, "RIG_CHECK_ENABLED", False),
            mock.patch.object(appmod, "make_backend", FakeEngine),
            mock.patch.object(appmod, "STORE", self.store),
        ]
        for p in self._patches:
            p.start()
        with appmod.L.lock:
            appmod.L.reset()
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()

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

    def start_session(self, duration=30.0):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
        r = self.client.post("/api/session/start",
                             json={"name": "Dana", "phone": "444",
                                   "duration_s": duration})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        with appmod.L.lock:
            sess = appmod.L.session
            sess["latch"] = None          # no model / keyframes in these tests
            sess["stills"] = None
            sess["source1_label"] = "USB camera 2"
            return sess["id"]

    def wait_until(self, cond, timeout=10.0, interval=0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(interval)
        return bool(cond())


class TestCaptureException(EndStateCase):
    def test_capture_exception_finalizes_as_incomplete(self):
        sid = self.start_session()
        with mock.patch.object(appmod.cameras, "open_source",
                               return_value=FakeCap()), \
                mock.patch.object(appmod.M, "frame_metrics",
                                  side_effect=RuntimeError("boom")):
            appmod.L.thread = threading.Thread(
                target=appmod._loop, args=({"kind": "usb", "index": 2},),
                daemon=True)
            appmod.L.thread.start()
            ok = self.wait_until(
                lambda: appmod.L.error is not None and appmod.L.session is None)
        self.assertTrue(ok, "the loop must fail loudly and finalize the session")
        self.assertIn("capture error", appmod.L.error)
        row = self.store.get_session(sid)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "incomplete")
        self.assertNotEqual(row["status"], "running")
        self.assertIsNotNone(row["ended_ts"])
        summary = json.loads(row["summary_json"])
        self.assertTrue(summary["data_quality"]["capture_error"])
        self.assertIn("boom", summary["data_quality"]["capture_error_detail"])
        self.assertIn("capture error", summary["note"])


class TestOrphanReconciliation(EndStateCase):
    def test_running_rows_are_closed_at_startup_with_a_note(self):
        pid = self.store.upsert_patient("Orphan", "777")
        orphan = self.store.create_session(pid, {}, "cam", view_plane="sagittal")
        done = self.store.create_session(pid, {}, "cam", view_plane="sagittal")
        self.store.finish_session(done, 10.0, 100, {"form_score": 70},
                                  status="complete")
        closed = appmod._reconcile_orphan_sessions(self.store)
        self.assertEqual(closed, [orphan])
        row = self.store.get_session(orphan)
        self.assertEqual(row["status"], "incomplete")
        self.assertIsNotNone(row["ended_ts"])
        summary = json.loads(row["summary_json"])
        self.assertIn("startup", summary["note"])
        self.assertTrue(summary["data_quality"]["orphaned_running"])
        self.assertEqual(self.store.get_session(done)["status"], "complete",
                         "a finished session must not be touched")
        # idempotent: nothing left to close
        self.assertEqual(appmod._reconcile_orphan_sessions(self.store), [])

    def test_app_init_runs_the_reconciliation(self):
        self.assertTrue(hasattr(appmod, "ORPHANED_SESSIONS"))
        self.assertIsInstance(appmod.ORPHANED_SESSIONS, list)


class TestDroppedViewNote(EndStateCase):
    def test_dropped_second_view_is_recorded_in_the_summary(self):
        sid = self.start_session()
        with appmod.L.lock:
            appmod.L.warn2 = ("second view (USB camera 4) stopped delivering "
                              "frames — dropped; the other views keep recording")
        appmod._finalize("stopped")
        row = self.store.get_session(sid)
        summary = json.loads(row["summary_json"])
        dq = summary["data_quality"]
        self.assertEqual(len(dq["dropped_views"]), 1)
        self.assertEqual(dq["dropped_views"][0]["view"], "second")
        self.assertTrue(dq["second_view_dropped"])
        self.assertFalse(dq["third_view_dropped"])
        self.assertIn("dropped", dq["warn2"])
        self.assertIn("fewer views", summary["note"])
        self.assertIn("was dropped", summary["note"])

    def test_no_dropped_view_adds_no_note(self):
        sid = self.start_session()
        appmod._finalize("stopped")
        summary = json.loads(self.store.get_session(sid)["summary_json"])
        self.assertNotIn("dropped_views", summary.get("data_quality") or {})
        self.assertNotIn("fewer views", summary.get("note") or "")


if __name__ == "__main__":
    unittest.main()
