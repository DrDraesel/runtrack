"""Tests for RunTrack session storage."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_store import Store  # noqa: E402


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.store = Store(self.db, sessions_dir=Path(self.tmp.name) / "sessions")

    def tearDown(self):
        self.tmp.cleanup()

    def test_patient_upsert(self):
        a = self.store.upsert_patient("Jane Doe", "555-0001")
        b = self.store.upsert_patient("Jane Doe", "555-0001")
        self.assertEqual(a, b)
        c = self.store.upsert_patient("Jane D.", "555-0001")   # name update
        self.assertEqual(a, c)
        p = self.store.get_patient(a)
        self.assertEqual(p["name"], "Jane D.")
        self.assertEqual(len(self.store.list_patients()), 1)

    def test_session_lifecycle(self):
        pid = self.store.upsert_patient("Bob", "555-0002")
        sid = self.store.create_session(pid, {"weight_kg": 80, "height_cm": 180,
                                              "speed_kmh": 11}, "USB camera 0")
        rows = [{"t_rel": 0.0, "left_knee": 90.0}, {"t_rel": 0.033, "left_knee": 91.2}]
        self.store.save_csv(sid, rows)
        self.store.finish_session(sid, 0.033, len(rows), {"cadence_spm": 180.0},
                                  status="complete")
        s = self.store.get_session(sid)
        self.assertEqual(s["status"], "complete")
        self.assertEqual(s["patient_name"], "Bob")
        self.assertIn("cadence_spm", s["summary_json"])
        sess_list = self.store.list_sessions()
        self.assertEqual(len(sess_list), 1)
        self.assertEqual(sess_list[0]["cadence_spm"], 180.0)

    def test_json_roundtrip(self):
        pid = self.store.upsert_patient("Zoe", "555-0003")
        sid = self.store.create_session(pid, {}, "test")
        self.store.save_json(sid, "coach.json", {"text": "nice form", "model": "m"})
        got = self.store.read_json(sid, "coach.json")
        self.assertEqual(got["text"], "nice form")
        self.assertIsNone(self.store.read_json(sid, "missing.json"))


if __name__ == "__main__":
    unittest.main()
