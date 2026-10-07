"""Editable patient/session records: store methods + API endpoints."""
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402
from session_store import Store  # noqa: E402


class TestStoreEdits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = Store(root / "t.db", sessions_dir=root / "sessions")

    def tearDown(self):
        self.tmp.cleanup()

    def _session(self, pid):
        return self.store.create_session(pid, {"weight_kg": 70, "height_cm": 170,
                                               "speed_kmh": 9}, "test")

    def test_update_patient_renames(self):
        pid = self.store.upsert_patient("Ann", "111")
        p = self.store.update_patient(pid, name="Anne", phone="222")
        self.assertEqual(p["name"], "Anne")
        self.assertEqual(p["phone"], "222")

    def test_update_patient_partial_keeps_other(self):
        pid = self.store.upsert_patient("Ann", "111")
        p = self.store.update_patient(pid, name="Anne")
        self.assertEqual(p["phone"], "111")

    def test_update_patient_duplicate_phone_rejected(self):
        self.store.upsert_patient("Ann", "111")
        pid = self.store.upsert_patient("Bob", "222")
        with self.assertRaises(ValueError):
            self.store.update_patient(pid, phone="111")

    def test_update_patient_blank_rejected(self):
        pid = self.store.upsert_patient("Ann", "111")
        with self.assertRaises(ValueError):
            self.store.update_patient(pid, name="   ")

    def test_update_patient_unknown(self):
        self.assertIsNone(self.store.update_patient(999, name="X"))

    def test_update_session_fields(self):
        pid = self.store.upsert_patient("Ann", "111")
        sid = self._session(pid)
        s = self.store.update_session(sid, weight_kg=83, speed_kmh=10)
        self.assertEqual(s["weight_kg"], 83)
        self.assertEqual(s["speed_kmh"], 10)
        self.assertEqual(s["height_cm"], 170)   # untouched

    def test_update_session_unknown(self):
        self.assertIsNone(self.store.update_session(999, weight_kg=80))

    def test_session_list_exposes_editable_fields(self):
        pid = self.store.upsert_patient("Ann", "111")
        sid = self._session(pid)
        rows = self.store.list_sessions(10)
        row = next(r for r in rows if r["id"] == sid)
        for key in ("weight_kg", "height_cm", "speed_kmh"):
            self.assertIn(key, row)


class TestEditAPI(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()

    def test_patient_edit_unknown(self):
        r = self.c.post("/api/patient/999999/edit", json={"name": "X", "phone": "1"})
        self.assertEqual(r.status_code, 404)

    def test_session_edit_unknown(self):
        r = self.c.post("/api/session/999999/edit", json={"weight_kg": 80})
        self.assertEqual(r.status_code, 404)

    def test_foreign_host_rejected(self):
        r = self.c.post("/api/session/1/edit", json={},
                        headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 403)

    def test_cameras_preferred_shape(self):
        r = self.c.get("/api/cameras")
        self.assertEqual(r.status_code, 200)
        self.assertIn("preferred", r.get_json())


if __name__ == "__main__":
    unittest.main()
