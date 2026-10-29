"""POST /api/session/capture_angles + joint-study wiring into the session.

The endpoint stores one labelled snapshot of the live angles on the running
session (``joint_study.capture``); ``_finalize`` writes
``summary["angle_captures"]`` / ``summary["joint_study"]``; ``/api/state``
exposes the capture count for the UI.

No camera and no session files: ``L`` is isolated per test and the store is
mocked, following the existing API-test pattern.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod        # noqa: E402
import config               # noqa: E402
import joint_study as JS    # noqa: E402


def running_session(captures=None, elapsed=12.5, sid=5):
    """Minimal running-session dict (the shape api_session_start builds)."""
    return {
        "id": sid, "patient": {"name": "Dana", "phone": "444"},
        "t0": 0.0, "elapsed": elapsed, "duration": 30.0,
        "rows": [], "params": {"weight_kg": 70.0, "height_cm": 175.0,
                               "speed_kmh": 10.0},
        "finalized": False, "finalizing": False,
        "snap_count": 0, "last_snap": -1e9,
        "view_plane": "sagittal", "discipline": "running",
        "bike_type": config.DEFAULT_BIKE_TYPE, "latch": None,
        "stills": None, "aspect": 16.0 / 9.0, "seeks": 0,
        "last_rel": None, "t0_video": None,
        "source1_label": "USB camera 2",
        "plane2": "frontal", "source2_label": "none",
        "rows2": [], "t0_video2": None, "t0_2": None,
        "plane3": "frontal", "direction3": "front",
        "source3_label": "none",
        "rows3": [], "t0_video3": None, "t0_3": None,
        "angle_captures": list(captures or []),
    }


LIVE = {"pose_found": True, "fps": 21.9, "left_knee": 152.44,
        "right_knee": 149.06, "trunk_lean": -5.04, "left_hip": None}


class _Case(unittest.TestCase):
    ATTRS = ("session", "live", "active", "source_label", "error",
             "latest_jpeg", "fps", "last_angles", "last_angles_t")

    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()
        self._saved = {k: getattr(appmod.L, k) for k in self.ATTRS}
        with appmod.L.lock:
            appmod.L.session = None
            appmod.L.live = {}
            appmod.L.active = False
            appmod.L.error = None
            appmod.L.last_angles = {}
            appmod.L.last_angles_t = None

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(appmod.L, k, v)

    def live(self, session, live=None):
        with appmod.L.lock:
            appmod.L.session = session
            appmod.L.live = dict(LIVE if live is None else live)


class TestCaptureEndpoint(_Case):
    def test_no_session_is_400(self):
        r = self.c.post("/api/session/capture_angles", json={"label": "mid stance"})
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertFalse(d["ok"])
        self.assertIn("no running session", d["error"])

    def test_finalized_session_is_400(self):
        sess = running_session()
        sess["finalized"] = True
        self.live(sess)
        r = self.c.post("/api/session/capture_angles", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no running session", r.get_json()["error"])

    def test_capture_returns_the_record(self):
        sess = running_session(elapsed=12.5)
        self.live(sess)
        r = self.c.post("/api/session/capture_angles", json={"label": "  mid   swing "})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertTrue(d["ok"])
        rec = d["capture"]
        self.assertEqual(rec["label"], "mid swing")      # whitespace collapsed
        self.assertEqual(rec["t"], 12.5)
        self.assertEqual(rec["angles"]["left_knee"], 152.4)   # 1 dp, measured only
        self.assertEqual(rec["angles"]["trunk_lean"], -5.0)
        self.assertNotIn("left_hip", rec["angles"])       # None stays absent
        self.assertNotIn("fps", rec["angles"])
        self.assertEqual(sess["angle_captures"], [rec])

    def test_label_is_optional(self):
        sess = running_session()
        self.live(sess)
        d = self.c.post("/api/session/capture_angles", json={}).get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["capture"]["label"], "Capture 1")
        # a non-string label must not blow up
        d2 = self.c.post("/api/session/capture_angles",
                         json={"label": 1234}).get_json()
        self.assertEqual(d2["capture"]["label"], "1234")

    def test_empty_angles_is_400(self):
        self.live(running_session(), {"pose_found": False, "fps": 0.0})
        r = self.c.post("/api/session/capture_angles", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no joint angles", r.get_json()["error"])

    def test_capture_limit_is_400(self):
        full = [{"t": float(i), "label": f"c{i}", "angles": {"left_knee": 150.0}}
                for i in range(JS.MAX_CAPTURES)]
        self.live(running_session(captures=full))
        r = self.c.post("/api/session/capture_angles", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("capture limit", r.get_json()["error"])

    def test_falls_back_to_the_last_tracked_frame(self):
        """One frame without a body must not discard the operator's tap."""
        import time as _t
        self.live(running_session(elapsed=30.0), {"pose_found": False, "fps": 22.0})
        with appmod.L.lock:
            appmod.L.last_angles = {"left_knee": 152.0, "trunk_lean": 6.0}
            appmod.L.last_angles_t = _t.monotonic() - 0.8
        r = self.c.post("/api/session/capture_angles", json={"label": "toe-off"})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        rec = r.get_json()["capture"]
        self.assertEqual(rec["source"], "last tracked frame")
        self.assertEqual(rec["angles"]["left_knee"], 152.0)
        self.assertGreaterEqual(rec["age_s"], 0.5)

    def test_live_frame_wins_over_the_fallback(self):
        import time as _t
        self.live(running_session(), {"left_knee": 100.0})
        with appmod.L.lock:
            appmod.L.last_angles = {"left_knee": 999.0}
            appmod.L.last_angles_t = _t.monotonic()
        rec = self.c.post("/api/session/capture_angles", json={}).get_json()["capture"]
        self.assertEqual(rec["source"], "live frame")
        self.assertEqual(rec["angles"]["left_knee"], 100.0)

    def test_origin_guard(self):
        self.live(running_session())
        r = self.c.post("/api/session/capture_angles", json={},
                        environ_overrides={"HTTP_HOST": "10.0.0.9:8780"})
        self.assertEqual(r.status_code, 403)


class TestStateAndSummary(_Case):
    def test_state_reports_captures_and_last(self):
        self.assertIsNone(self.c.get("/api/state").get_json()["joint_study"],
                          "no session -> null")
        sess = running_session()
        self.live(sess)
        empty = self.c.get("/api/state").get_json()["joint_study"]
        self.assertEqual(empty["captures"], 0)
        self.assertIsNone(empty["last"])
        first = self.c.post("/api/session/capture_angles",
                            json={"label": "one"}).get_json()["capture"]
        second = self.c.post("/api/session/capture_angles",
                             json={"label": "two"}).get_json()["capture"]
        js = self.c.get("/api/state").get_json()["joint_study"]
        self.assertEqual(js["captures"], 2)
        self.assertIsInstance(js["captures"], int)
        self.assertEqual(js["last"]["label"], "two")
        self.assertEqual(js["last"], second)
        self.assertNotEqual(js["last"], first)

    def test_finalize_writes_captures_and_joint_study(self):
        rows = [{"t_rel": i * 0.03,
                 "left_knee": 150.0 + i,
                 "right_knee": 160.0,
                 "trunk_lean": 5.0} for i in range(20)]
        sess = running_session(elapsed=1.0)
        sess["rows"] = rows
        sess["angle_captures"] = [
            {"t": 0.4, "label": "mid swing", "angles": {"left_knee": 158.0}},
            {"t": 0.8, "label": "contact", "angles": {"right_knee": 160.0}},
        ]
        with appmod.L.lock:
            appmod.L.session = sess
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as finish, \
                mock.patch.object(appmod.R, "build"):
            sid = appmod._finalize("stopped")
        self.assertEqual(sid, 5)
        summary = finish.call_args[0][3]
        self.assertEqual(summary["angle_captures"], sess["angle_captures"])
        expected = JS.summarize(rows)
        self.assertEqual(summary["joint_study"], expected)
        self.assertEqual(summary["joint_study"]["left_knee"]["n"], 20)
        self.assertEqual(summary["joint_study"]["left_knee"]["min"], 150.0)
        self.assertEqual(summary["joint_study"]["left_knee"]["max"], 169.0)
        self.assertEqual(summary["joint_study"]["left_knee"]["rom"], 19.0)
        self.assertEqual(summary["joint_study"]["right_knee"]["rom"], 0.0)
        # the running session is closed, the report id is remembered
        self.assertIsNone(appmod.L.session)
        self.assertEqual(appmod.L.last_report_sid, 5)

    def test_finalize_without_captures_writes_empty_structures(self):
        sess = running_session()
        sess["rows"] = [{"t_rel": 0.0, "left_knee": 150.0}]
        with appmod.L.lock:
            appmod.L.session = sess
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as finish, \
                mock.patch.object(appmod.R, "build"):
            appmod._finalize("stopped")
        summary = finish.call_args[0][3]
        self.assertEqual(summary["angle_captures"], [])
        self.assertEqual(summary["joint_study"]["left_knee"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
