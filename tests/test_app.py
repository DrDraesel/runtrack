"""API surface tests for the RunTrack Flask app (no camera required)."""
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

    def test_foreign_host_rejected(self):
        r = self.c.post("/api/camera/close", headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
