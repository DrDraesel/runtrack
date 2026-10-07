"""Report + history/compare API tests (temp DB, no camera, no real data dir)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import metrics as M  # noqa: E402
import report as R  # noqa: E402
from session_store import Store  # noqa: E402


def synth_rows(n=240, fps=30.0, cadence_spm=180.0, shin=3.0, osc=0.02,
               with_legs=True):
    """Semi-synthetic rows: a running cadence with a configurable shin angle."""
    f_step = cadence_spm / 60.0 / 2.0        # one strike per foot per stride
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        y_l = 0.90 + 0.05 * math.sin(ph)
        y_r = 0.90 + 0.05 * math.sin(ph + math.pi)
        dx = 0.2 * math.tan(math.radians(shin))
        row = {
            "t_rel": t,
            "l_ankle_y": y_l, "r_ankle_y": y_r,
            "l_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "r_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_x": 0.50, "_midsh_x": 0.51, "_midsh_y": 0.38,
            "_facing": 1.0,
            "l_ankle_x": 0.48 + dx, "r_ankle_x": 0.52 + dx,
            "l_hip_x": 0.48, "r_hip_x": 0.52,
            "l_knee_x": 0.48, "r_knee_x": 0.52,
            "shin_angle_l": shin, "shin_angle_r": shin,
            "overstride_l": 0.05, "overstride_r": 0.05,
            "knee_flex_l": 25.0, "knee_flex_r": 25.0,
            "left_knee": 155.0, "right_knee": 155.0,
            "left_hip": 170.0, "right_hip": 170.0,
            "pelvic_tilt_deg": 1.0, "knee_valgus_l": 3.0, "knee_valgus_r": 3.0,
            "trunk_lean_fwd": 7.0,
        }
        if not with_legs:
            for k in ("l_ankle_y", "r_ankle_y", "l_hip_y", "r_hip_y", "_midhip_y",
                      "shin_angle_l", "shin_angle_r", "overstride_l", "overstride_r",
                      "left_knee", "right_knee"):
                row[k] = None
        rows.append(row)
    return rows


class ReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(
            config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def seed(self, name, phone, **kw):
        plane = kw.pop("plane", "sagittal")   # synth_rows() has no plane knob
        rows = synth_rows(**kw)
        summary = M.analyze_session(rows, weight_kg=75, height_cm=178, speed_kmh=10,
                                    view_plane=plane)
        pid = self.store.upsert_patient(name, phone)
        sid = self.store.create_session(pid, {"weight_kg": 75, "height_cm": 178,
                                             "speed_kmh": 10}, "test source",
                                        view_plane=plane)
        self.store.save_csv(sid, rows)
        self.store.finish_session(sid, summary.get("duration_s"), len(rows), summary,
                                  status="complete")
        return sid, summary


class TestReportContent(ReportFixture):
    def test_report_contains_new_sections(self):
        sid, _ = self.seed("Ann", "111", cadence_spm=180, shin=12.0, osc=0.05)
        p = Path(R.build(sid, self.store))
        self.assertTrue(p.exists())
        html = p.read_text(encoding="utf-8")
        for needle in ("Print / Save as PDF", "Overall form score", "Injury risk stratification",
                       "@media print", "@page", "Gait event keyframes", "Gait phases latched",
                       "Left / right symmetry", "not assessed (wrong camera view)",
                       "camera view"):
            self.assertIn(needle, html, needle)
        self.assertIn("form_score", json.dumps(json.loads(
            (p.parent / "summary.json").read_text(encoding="utf-8"))))
        charts = list(p.parent.glob("chart_*.png"))
        self.assertGreaterEqual(len(charts), 7)

    def test_report_is_honest_without_lower_body(self):
        sid, _ = self.seed("Bob", "222", with_legs=False)
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn(config.GAIT_NOT_ASSESSABLE_TEXT, html)
        self.assertIn("too low for gait events", html)

    def test_frontal_report_suppresses_sagittal(self):
        sid, _ = self.seed("Cara", "333", plane="frontal")
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("camera view: <b>frontal</b>", html)
        self.assertIn(config.NOT_ASSESSED_TEXT, html)

    def test_keyframes_gallery_embeds_images(self):
        sid, _ = self.seed("Dan", "444")
        kdir = self.root / "sessions" / str(sid) / config.KEYFRAME_DIR
        kdir.mkdir(parents=True, exist_ok=True)
        import cv2
        import numpy as np
        img = np.zeros((80, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(kdir / "00_IC_left_t00.10.jpg"), img)
        # summary without stored metadata -> report must pick the folder up
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn(config.KEYFRAME_DIR + "/00_IC_left_t00.10.jpg", html)
        self.assertIn("Gait event keyframes (1)", html)


class TestHistoryAndCompareAPI(ReportFixture):
    def setUp(self):
        super().setUp()
        import app as appmod
        self.appmod = appmod
        self._store_patch = mock.patch.object(appmod, "STORE", self.store)
        self._store_patch.start()
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()

    def tearDown(self):
        self._store_patch.stop()
        super().tearDown()

    def test_state_has_playback_and_plane(self):
        d = self.c.get("/api/state").get_json()
        self.assertIn("playback", d)
        self.assertIn("view_plane", d)
        self.assertIn("speeds", d["playback"])
        self.assertFalse(d["playback"]["can_control"])

    def test_playback_requires_file_source(self):
        r = self.c.post("/api/playback", json={"speed": 0.5})
        self.assertEqual(r.status_code, 400)
        self.assertIn("video file", r.get_json()["error"])

    def test_history_endpoint(self):
        a, _ = self.seed("Eve", "555", cadence_spm=170)
        b, _ = self.seed("Eve", "555", cadence_spm=180)
        d = self.c.get("/api/history").get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["patients"]), 1)
        self.assertEqual(d["patients"][0]["n_sessions"], 2)
        self.assertEqual(len(d["patients"][0]["sessions"]), 2)
        self.assertEqual(d["limit_per_patient"], 30)
        pid = d["patients"][0]["id"]
        h = self.c.get(f"/api/patients/{pid}/history").get_json()
        self.assertEqual(h["n"], 2)
        self.assertEqual(h["limit"], 30)
        self.assertEqual(h["sessions"][0]["id"], b)      # newest first

    def test_patient_history_unknown(self):
        self.assertEqual(self.c.get("/api/patients/9999/history").status_code, 404)

    def test_compare_endpoint(self):
        a, _ = self.seed("Fay", "666", cadence_spm=160, osc=0.05, shin=12.0)
        b, _ = self.seed("Fay", "666", cadence_spm=180, osc=0.02, shin=3.0)
        d = self.c.get(f"/api/compare?a={a}&b={b}").get_json()
        self.assertTrue(d["ok"])
        rows = {r["metric"]: r for r in d["rows"]}
        self.assertEqual(rows["cadence_spm"]["verdict"], "Improved")
        self.assertEqual(rows["vertical_osc_cm"]["verdict"], "Improved")
        self.assertGreater(d["summary"]["improved"], 0)
        self.assertEqual(d["baseline"]["id"], a)
        self.assertEqual(d["followup"]["id"], b)

    def test_compare_errors(self):
        self.assertEqual(self.c.get("/api/compare").status_code, 400)
        a, _ = self.seed("Gus", "777")
        self.assertEqual(self.c.get(f"/api/compare?a={a}&b=99999").status_code, 404)

    def test_sessions_list_exposes_new_fields(self):
        self.seed("Hana", "888")
        s = self.c.get("/api/sessions").get_json()["sessions"][0]
        self.assertIn("view_plane", s)
        self.assertIn("form_score", s)
        self.assertIn("risk_level", s)


if __name__ == "__main__":
    unittest.main()
