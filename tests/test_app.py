"""API surface tests for the RunTrack Flask app (no camera required)."""
import math
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402


class TestApp(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()

    def test_index(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"RunTrack", r.data)

    def test_state(self):
        r = self.c.get("/api/state")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn("active", d)
        self.assertIn("session", d)

    def test_session_needs_name(self):
        r = self.c.post("/api/session/start", json={"name": "", "phone": "123"})
        self.assertEqual(r.status_code, 400)

    def test_session_needs_camera(self):
        r = self.c.post("/api/session/start", json={"name": "A", "phone": "123"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("camera", r.get_json()["error"])

    def test_camera_open_bad_kind(self):
        r = self.c.post("/api/camera/open", json={"kind": "nope"})
        self.assertEqual(r.status_code, 400)

    def test_camera_open_missing_file(self):
        r = self.c.post("/api/camera/open", json={"kind": "file", "path": "Z:/nope.mp4"})
        self.assertEqual(r.status_code, 400)

    def test_report_missing(self):
        r = self.c.get("/report/999999/")
        self.assertEqual(r.status_code, 404)

    def test_sessions_list(self):
        r = self.c.get("/api/sessions")
        self.assertEqual(r.status_code, 200)
        self.assertIn("sessions", r.get_json())

    def test_cycling_fit_ui_is_present(self):
        html = self.c.get("/").get_data(as_text=True)
        self.assertIn("cycFitCard", html)
        self.assertIn("Knee at BDC", html)
        self.assertIn("updateCyclingFit", html)
        self.assertIn("AI fit review is in the session report", html)

    def test_state_reports_bike_live_slot(self):
        d = self.c.get("/api/state").get_json()
        self.assertIn("bike_live", d)
        self.assertIsNone(d["bike_live"])          # idle: never invented
        self.assertIn("bike_type", d)

    def test_live_bike_returns_a_rolling_tt_fit(self):
        rows = []
        for i in range(300):
            tt = i / 30.0
            rows.append({"t_rel": tt,
                         "left_knee": 120.0 + 22.0 * math.cos(2 * math.pi * 1.5 * tt),
                         "right_knee": 120.0 + 22.0 * math.cos(2 * math.pi * 1.5 * tt + 0.4),
                         "left_hip": 70.0 + 10.0 * math.cos(2 * math.pi * 1.5 * tt),
                         "right_hip": 70.0 + 10.0 * math.cos(2 * math.pi * 1.5 * tt + 0.4),
                         "left_elbow": 95.0, "right_elbow": 95.0,
                         "trunk_lean": 70.0})
        appmod.L.bike_type = "tt"
        fit = appmod._live_bike({"rows": rows}, "cycling", True)
        self.assertIsNotNone(fit)
        self.assertGreater(fit["n_frames"], 100)
        self.assertGreater(fit["knee_bdc_deg"], 135.0)
        self.assertLess(fit["knee_tdc_deg"], 105.0)
        self.assertIn("knee_bdc_deg", fit["targets"])
        self.assertIsNone(appmod._live_bike({"rows": rows}, "running", True))
        self.assertIsNone(appmod._live_bike({"rows": rows}, "cycling", False))

    def test_foreign_host_rejected(self):
        r = self.c.post("/api/camera/close", headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
