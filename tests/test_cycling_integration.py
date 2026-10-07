"""Cycling end-to-end integration tests (no camera, no app).

Covers the full offline path the app finalize step performs for a cycling
session: analyze_session(discipline="cycling") → cycling.enrich →
advisor.enrich → report.build, plus the compare table's bike/aero rows.

Rows come from the synthetic cyclist (ground-truth kinematics), so the checks
are against known pedalling values, not against the pipeline itself.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

import numpy as np  # noqa: E402

import advisor  # noqa: E402
import config  # noqa: E402
import cycling as C  # noqa: E402
import metrics as M  # noqa: E402
import report as R  # noqa: E402
import synth_cyclist as SC  # noqa: E402
from session_store import Store  # noqa: E402


def cycling_rows(seconds=20.0, fps=30, rpm=75.0, fa=0.28, with_rear=True,
                 geom=None):
    """Synthetic pedalling rows in the live-loop layout (rear rows r_-prefixed)."""
    g = dict(SC.GEOM)
    g.update(geom or {})
    rows = []
    n = int(seconds * fps)
    omega = 2.0 * math.pi * (rpm / 60.0)
    for k in range(n):
        phase = omega * (k / fps)
        lm = SC.landmarks_sagittal(phase, g)
        fm = M.frame_metrics(lm)
        fm["_mean_vis"] = 1.0
        if fa is not None:
            fm["fa_frac"] = fa + 0.004 * math.sin(phase * 2.0)
            # aero filming is frontal: give the row a frontal shoulder span
            # (the sagittal model has the shoulders nearly overlapping in x)
            fm["l_sh_x"], fm["r_sh_x"] = 0.50, 0.85
        row = dict(fm)
        row["t_rel"] = k / fps
        if with_rear:
            lm2 = SC.landmarks_rear(phase, g)
            fm2 = M.frame_metrics(lm2)
            fm2["_mean_vis"] = 1.0
            if fa is not None:
                fm2["fa_frac"] = fa + 0.003
                fm2["l_sh_x"], fm2["r_sh_x"] = 0.50, 0.85
            for kk, vv in fm2.items():
                row["r_" + kk] = vv
        rows.append(row)
    return rows


PARAMS = {"weight_kg": 75.0, "height_cm": 178.0, "speed_kmh": 40.0, "cd_est": 0.8}


def full_summary(rows, bike_type="tt"):
    summary = M.analyze_session(rows, weight_kg=PARAMS["weight_kg"],
                                height_cm=PARAMS["height_cm"],
                                speed_kmh=PARAMS["speed_kmh"],
                                view_plane="sagittal", events=[],
                                discipline="cycling")
    C.enrich(summary, rows, PARAMS, bike_type=bike_type, discipline="cycling")
    advisor.enrich(summary)
    return summary


class TestCyclingSession(unittest.TestCase):
    def test_running_gait_metrics_are_moved_out_not_invented(self):
        rows = cycling_rows()
        s = M.analyze_session(rows, weight_kg=75, height_cm=178, speed_kmh=40,
                              view_plane="sagittal", events=[], discipline="cycling")
        self.assertEqual(s.get("discipline"), "cycling")
        na = s.get("not_assessed") or {}
        self.assertEqual(na.get("cadence_spm"), config.CYCLING_NOT_APPLICABLE_TEXT)
        self.assertIsNone(s.get("cadence_spm"))
        self.assertIsNone(s.get("gait_events"))
        self.assertFalse(s.get("form"))          # no running bands in a bike session

    def test_enrich_produces_bike_rear_and_aero_blocks(self):
        rows = cycling_rows(rpm=75.0)
        s = full_summary(rows)
        bike = s["bike"]
        self.assertIsNotNone(bike.get("cadence_rpm"))
        self.assertAlmostEqual(bike["cadence_rpm"], 75.0, delta=5.0)
        kb = bike.get("knee_bdc_deg")
        self.assertIsNotNone(kb)
        self.assertGreater(kb, 120.0)
        self.assertLess(kb, 165.0)
        tp = bike.get("torso_deg")
        self.assertIsNotNone(tp)
        # torso model geometry puts the shoulders well above the hips
        self.assertGreater(tp, 15.0)
        self.assertLess(tp, 90.0)
        # rear-view block from the r_-prefixed second camera rows
        rear = s.get("rear") or {}
        self.assertIn("pelvic_obliquity_deg", rear)
        self.assertIn("hip_vertical_travel", rear)
        self.assertIn("knee_lateral_travel_left", rear)
        # aero frontal area from fa_frac + height
        aero = s.get("aero") or {}
        self.assertIsNotNone(aero.get("fa_m2_median"))
        self.assertGreater(aero["fa_m2_median"], 0.2)
        self.assertLess(aero["fa_m2_median"], 1.2)
        self.assertAlmostEqual(aero.get("cd_assumed"), 0.8)

    def test_advisor_gives_cycling_score_and_cycling_rules(self):
        rows = cycling_rows(rpm=75.0)
        s = full_summary(rows)
        fs = s.get("form_score_detail") or advisor.form_score(s)
        self.assertEqual(fs.get("discipline"), "cycling")
        comps = {c["metric"] for c in fs.get("components") or []}
        self.assertTrue(comps & {"bike_knee_bdc", "bike_torso", "bike_cadence",
                                 "bike_hip_tdc", "symmetry_pct"}, comps)
        self.assertIsNotNone(fs.get("score"))
        risk = s.get("risk") or {}
        self.assertEqual(risk.get("discipline"), "cycling")
        self.assertIn(risk.get("level"), ("Low", "Moderate", "Elevated", "High"))
        self.assertIn("fit", (risk.get("caveat") or "").lower())

    def test_compare_table_has_bike_and_aero_rows(self):
        a = full_summary(cycling_rows(rpm=70.0, fa=0.30))
        b = full_summary(cycling_rows(rpm=80.0, fa=0.27))
        cmp_ = advisor.compare(a, b, {"bike_type": "tt"}, {"bike_type": "tt"})
        rows = cmp_.get("rows") or []
        metrics = {r.get("metric") for r in rows}
        self.assertIn("knee_bdc_deg", metrics)
        self.assertIn("torso_deg", metrics)
        self.assertIn("cadence_rpm", metrics)
        self.assertIn("fa_m2", metrics)
        self.assertIn("aero_watts", metrics)
        fa_row = next(r for r in rows if r.get("metric") == "fa_m2")
        self.assertEqual(fa_row.get("verdict"), "Improved")     # 0.30 → 0.27 m²
        watt_row = next(r for r in rows if r.get("metric") == "aero_watts")
        self.assertLess(watt_row.get("delta"), 0.0)             # less area = less W


class TestCyclingReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def seed(self, rows, bike_type="tt", name="Cyclist"):
        summary = full_summary(rows, bike_type=bike_type)
        pid = self.store.upsert_patient(name, "000")
        sid = self.store.create_session(
            pid, {"weight_kg": 75, "height_cm": 178, "speed_kmh": 40},
            "synthetic turbo", view_plane="sagittal", discipline="cycling",
            bike_type=bike_type)
        self.store.save_csv(sid, rows)
        self.store.finish_session(sid, summary.get("duration_s"), len(rows),
                                  summary, status="complete")
        return sid

    def test_report_has_professional_sections(self):
        sid = self.seed(cycling_rows())
        p = Path(R.build(sid, self.store))
        html = p.read_text(encoding="utf-8")
        for needle in ("cycling position analysis", "Clinical review",
                       "Coach review", "Injury risk stratification",
                       "Rear-view control", "Aero analysis",
                       "Cycling position score", "bike"):
            self.assertIn(needle, html, needle)
        self.assertNotIn("<h2>Gait phases latched</h2>", html)   # running-only
        self.assertNotIn("not applicable to a cycling session", html.split("<h2 class=\"group\">")[0],
                         "the position table should not show running not-assessed text")
        charts = {c.name for c in p.parent.glob("chart_*.png")}
        self.assertIn("chart_bike_knee.png", charts)
        self.assertIn("chart_bike_rear.png", charts)
        self.assertTrue(any(c.startswith("chart_aero") for c in charts), charts)

    def test_report_survives_a_session_without_rear_or_aero(self):
        sid = self.seed(cycling_rows(with_rear=False, fa=None))
        p = Path(R.build(sid, self.store))
        html = p.read_text(encoding="utf-8")
        self.assertIn("cycling position analysis", html)
        self.assertNotIn("Rear-view control", html)
        self.assertNotIn("Aero analysis", html)


if __name__ == "__main__":
    unittest.main()
