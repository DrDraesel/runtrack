"""Second-view (rear + side) analysis: plane gating, scoring and report wiring."""
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
import second_view as SV  # noqa: E402
from session_store import Store  # noqa: E402


def ms_events(rows, win=6):
    """Mid-stance events from local ankle-y minima (the latch's own criterion).

    The synthetic rows are not a full gait generator, so the test supplies the
    stance events explicitly — this is the same seam the capture loop uses when
    it hands a GaitLatch's events to the analysis.
    """
    out = []
    for side, key in (("left", "l_ankle_y"), ("right", "r_ankle_y")):
        y = [r.get(key) for r in rows]
        t = [r.get("t_rel") for r in rows]
        for i in range(win, len(rows) - win):
            v = y[i]
            if v is None:
                continue
            seg = [x for x in y[i - win:i + win + 1] if x is not None]
            if v == min(seg):
                out.append({"type": "MS", "side": side, "t": t[i], "i": i})
    return out


def frontal_rows(n=240, fps=30.0, knee_valgus=4.0, cadence_spm=180.0):
    """Rear-view (frontal-plane) rows: alternating legs, knee collapse, hip drop."""
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        valgus = knee_valgus + 2.0 * math.sin(ph)
        rows.append({
            "t_rel": t,
            "l_ankle_y": 0.90 + 0.05 * math.sin(ph),
            "r_ankle_y": 0.90 + 0.05 * math.sin(ph + math.pi),
            "l_hip_y": 0.62, "r_hip_y": 0.62,
            "_midhip_y": 0.62, "_midhip_x": 0.50,
            "_midsh_x": 0.50, "_midsh_y": 0.38, "_facing": 1.0,
            "l_hip_x": 0.48, "r_hip_x": 0.52,
            "l_knee_x": 0.47, "r_knee_x": 0.53,
            "l_ankle_x": 0.47, "r_ankle_x": 0.53,
            "l_sh_x": 0.44, "r_sh_x": 0.56,
            "left_knee": 160.0, "right_knee": 160.0,
            "pelvic_tilt_deg": 3.0 * math.sin(ph),
            "knee_valgus_l": valgus, "knee_valgus_r": valgus,
        })
    return rows


def sagittal_rows(n=240, fps=30.0, cadence_spm=180.0, shin=3.0, osc=0.02):
    """Side-view rows (same generator style as test_report_api.synth_rows)."""
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        dx = 0.2 * math.tan(math.radians(shin))
        rows.append({
            "t_rel": t,
            "l_ankle_y": 0.90 + 0.05 * math.sin(ph),
            "r_ankle_y": 0.90 + 0.05 * math.sin(ph + math.pi),
            "l_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "r_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_x": 0.50, "_midsh_x": 0.51, "_midsh_y": 0.38, "_facing": 1.0,
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
        })
    return rows


PARAMS = {"weight_kg": 75, "height_cm": 178, "speed_kmh": 10}


class TestSecondViewAnalysis(unittest.TestCase):
    def test_rear_clip_gives_frontal_metrics_only(self):
        rows = frontal_rows()
        sv = SV.analyze(rows, PARAMS, "frontal", events=ms_events(rows))
        self.assertIsNotNone(sv)
        self.assertEqual(sv["view_plane"], "frontal")
        self.assertIn("knee_valgus_deg", sv)
        self.assertIn("pelvic_drop_deg", sv)
        self.assertIn("knee_valgus_deg", sv["form"])
        self.assertIn("pelvic_drop_deg", sv["form"])
        # the wrong plane is never invented in either direction
        self.assertNotIn("touchdown_shin_deg", sv)
        self.assertIn("touchdown_shin_deg", sv["not_assessed"])
        self.assertIsNotNone(sv["form_score"])
        self.assertEqual(sv["form_score_detail"]["plane"], "frontal")
        self.assertGreaterEqual(sv["rows"], 200)

    def test_side_clip_as_second_view_keeps_sagittal_metrics(self):
        sv = SV.analyze(sagittal_rows(), PARAMS, "sagittal")
        self.assertIsNotNone(sv)
        self.assertEqual(sv["view_plane"], "sagittal")
        self.assertIn("touchdown_shin_deg", sv)
        self.assertNotIn("knee_valgus_deg", sv)
        self.assertIn("knee_valgus_deg", sv["not_assessed"])
        self.assertEqual(sv["form_score_detail"]["plane"], "sagittal")

    def test_high_valgus_raises_a_rear_view_risk_flag(self):
        rows = frontal_rows(knee_valgus=15.0)
        sv = SV.analyze(rows, PARAMS, "frontal", events=ms_events(rows))
        ids = [f.get("id") for f in (sv.get("risk") or {}).get("flags", [])]
        self.assertIn("valgus", ids)

    def test_too_few_rows_returns_none(self):
        self.assertIsNone(SV.analyze(frontal_rows(n=10), PARAMS, "frontal"))

    def test_cycling_has_no_second_view_block(self):
        self.assertIsNone(SV.analyze(frontal_rows(), PARAMS, "frontal",
                                     discipline="cycling"))

    def test_bad_plane_falls_back_to_default(self):
        sv = SV.analyze(frontal_rows(), PARAMS, "diagonal")
        self.assertEqual(sv["view_plane"], config.DEFAULT_PLANE2)


class TestSecondViewReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def test_report_shows_both_views(self):
        rows = sagittal_rows()
        summary = M.analyze_session(rows, view_plane="sagittal", **{
            "weight_kg": 75, "height_cm": 178, "speed_kmh": 10})
        sv = SV.analyze(frontal_rows(), PARAMS, "frontal",
                        events=ms_events(frontal_rows()))
        summary["second_view"] = sv
        summary["plane2"] = "frontal"
        summary["source2_label"] = "rear.mp4"
        pid = self.store.upsert_patient("Dana", "444")
        sid = self.store.create_session(pid, PARAMS, "side.mp4 + rear.mp4",
                                        view_plane="sagittal")
        self.store.save_csv(sid, rows)
        self.store.finish_session(sid, summary.get("duration_s"), len(rows), summary,
                                  status="complete")
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("Second view — frontal view (rear.mp4)", html)
        self.assertIn("Second-view score", html)
        self.assertIn("knee valgus", html.lower())
        self.assertIn("second view: <b>frontal</b>", html)


if __name__ == "__main__":
    unittest.main()
