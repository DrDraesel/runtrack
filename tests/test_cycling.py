"""Cycling analysis tests: synthetic pedalling ground truth.

The synthetic cyclist (tools/synth_cyclist.py) generates landmarks from a
2-link leg model riding a crank circle, so the true knee angle at BDC/TDC and
the true cadence are known analytically and the analysis is checked against
them — not against itself.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

import numpy as np  # noqa: E402

import config  # noqa: E402
import cycling as C  # noqa: E402
import gait_events as GA  # noqa: E402
import metrics as M  # noqa: E402
import synth_cyclist as SC  # noqa: E402


def _rows(geom=None, seconds=20.0, fps=30, rpm=60.0, view="sagittal",
          with_rear=False, prefix_rear="r_"):
    """Frame-metric rows from the synthetic model (like the live loop does)."""
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
        row = dict(fm)
        row["t_rel"] = k / fps
        if with_rear:
            lm2 = SC.landmarks_rear(phase, g)
            fm2 = M.frame_metrics(lm2)
            fm2["_mean_vis"] = 1.0
            for kk, vv in fm2.items():
                row[prefix_rear + kk] = vv
        rows.append(row)
    return rows


class TestCadence(unittest.TestCase):
    def test_rpm_60(self):
        rows = _rows(rpm=60.0)
        bike = C.analyze_cycling(rows, "road")
        self.assertIsNotNone(bike.get("cadence_rpm"))
        self.assertAlmostEqual(bike["cadence_rpm"], 60.0, delta=4.0)

    def test_rpm_90(self):
        rows = _rows(rpm=90.0)
        bike = C.analyze_cycling(rows, "road")
        self.assertAlmostEqual(bike["cadence_rpm"], 90.0, delta=5.0)

    def test_revolutions_counted(self):
        rows = _rows(rpm=60.0, seconds=20.0)
        bike = C.analyze_cycling(rows, "road")
        rev = bike["per_side"]["left"].get("revolutions")
        self.assertIsNotNone(rev)
        # 20 s at 60 rpm = 20 revolutions per leg (±2 for edge effects)
        self.assertAlmostEqual(float(rev), 20.0, delta=2.0)


class TestKneeAngles(unittest.TestCase):
    def test_bdc_matches_analytic(self):
        rows = _rows(rpm=60.0)
        bike = C.analyze_cycling(rows, "road")
        # analytic ground truth: knee angle at the bottom of the stroke
        truth = SC.knee_included_angle(0.0)     # phase 0 = pedal at bottom
        self.assertAlmostEqual(bike["knee_bdc_deg"], round(truth, 1), delta=2.5)

    def test_tdc_matches_analytic(self):
        rows = _rows(rpm=60.0)
        bike = C.analyze_cycling(rows, "road")
        truth = SC.knee_included_angle(math.pi)   # pedal at top
        self.assertAlmostEqual(bike["knee_tdc_min_deg"], round(truth, 1), delta=3.5)

    def test_saddle_default_in_band(self):
        """The default geometry was sized for ~145 deg at BDC (hemisphere)."""
        rows = _rows(rpm=60.0)
        bike = C.analyze_cycling(rows, "road")
        self.assertTrue(bike["form"]["bike_knee_bdc"]["in_band"],
                        bike["form"]["bike_knee_bdc"])

    def test_saddle_too_high_flagged_with_direction(self):
        rows = _rows(rpm=60.0, geom={"hip_h": 0.6785 + 0.020})
        bike = C.analyze_cycling(rows, "road")
        self.assertGreater(bike["knee_bdc_deg"], 150.0)
        self.assertFalse(bike["form"]["bike_knee_bdc"]["in_band"])
        joined = " ".join(s["action"] for s in bike["suggestions"]).lower()
        self.assertIn("lower", joined)
        self.assertIn("high", " ".join(s["finding"].lower() for s in bike["suggestions"]))

    def test_saddle_too_low_flagged_with_direction(self):
        rows = _rows(rpm=60.0, geom={"hip_h": 0.6785 - 0.020})
        bike = C.analyze_cycling(rows, "road")
        self.assertLess(bike["knee_bdc_deg"], 140.0)
        joined = " ".join(s["action"] for s in bike["suggestions"]).lower()
        self.assertIn("raise", joined)


class TestPositionTargets(unittest.TestCase):
    def test_torso_road_in_band(self):
        rows = _rows(rpm=60.0, geom={"torso_deg": 40.0})
        bike = C.analyze_cycling(rows, "road")
        self.assertIsNotNone(bike.get("torso_deg"))
        self.assertTrue(bike["form"]["bike_torso"]["in_band"],
                        bike["form"]["bike_torso"])

    def test_torso_tt_too_upright_gets_aero_suggestion(self):
        rows = _rows(rpm=60.0, geom={"torso_deg": 42.0})
        bike = C.analyze_cycling(rows, "tt")
        self.assertFalse(bike["form"]["bike_torso"]["in_band"])
        self.assertTrue(any("aero" in s["action"].lower() or "front end" in s["action"].lower()
                            for s in bike["suggestions"]), bike["suggestions"])

    def test_hip_angle_present(self):
        rows = _rows(rpm=60.0)
        bike = C.analyze_cycling(rows, "tri")
        self.assertIsNotNone(bike.get("hip_tdc_min_deg"))
        self.assertIn("bike_hip_tdc", bike["form"])


class TestRearMetrics(unittest.TestCase):
    def test_knee_lateral_travel_measured(self):
        # knee swings +/-0.02 m laterally  -> p2p = 0.04 m = 40 mm (approx, 2-98%)
        rows = _rows(rpm=60.0, with_rear=True,
                     geom={"knee_lat_amp": 0.02, "obliquity_deg": 0.0})
        rear = C.analyze_rear(rows, {"height_cm": 175.0})
        self.assertIsNotNone(rear)
        e = rear.get("knee_lateral_travel_left")
        self.assertIsNotNone(e)

    def test_pelvic_obliquity(self):
        rows = _rows(rpm=60.0, with_rear=True, geom={"obliquity_deg": 3.0})
        rear = C.analyze_rear(rows, {"height_cm": 175.0})
        self.assertIsNotNone(rear.get("pelvic_obliquity_deg"))
        # synthetic: inter-hip tilt oscillates +/-3 deg -> p2p ~ 6 deg
        self.assertAlmostEqual(rear["pelvic_obliquity_deg"], 6.0, delta=1.5)

    def test_no_rear_data_returns_none(self):
        rows = _rows(rpm=60.0)
        self.assertIsNone(C.analyze_rear(rows, {}))

    def test_scale_from_shoulder_width(self):
        rows = _rows(rpm=60.0, with_rear=True)
        rear = C.analyze_rear(rows, {"shoulder_cm": 42.0})
        self.assertIn("mm scale", rear.get("scale_note", ""))
        e = rear.get("hip_vertical_travel")
        self.assertIsNotNone(e)
        self.assertIsNotNone(e.get("value_mm"))


class TestAero(unittest.TestCase):
    def test_frontal_area_math(self):
        rows = []
        for k in range(60):
            rows.append({
                "t_rel": k / 30.0,
                "fa_frac": 0.08,
                "l_sh_x": 0.35 - 0.11,   # shoulder width 0.22 norm
                "r_sh_x": 0.35 + 0.11,
            })
        aero = C.frontal_area(rows, {"height_cm": 175.0})
        self.assertIsNotNone(aero)
        shoulder_m = 1.75 * config.SHOULDER_WIDTH_HEIGHT_FRACTION
        frame_w_m = shoulder_m / 0.22
        frame_h_m = frame_w_m / config.VIDEO_ASPECT_DEFAULT
        expect = 0.08 * frame_w_m * frame_h_m
        self.assertAlmostEqual(aero["fa_m2_median"], round(expect, 3), delta=0.002)

    def test_aero_delta_watts(self):
        d = C.aero_delta(0.400, 0.370, speed_kmh=40.0, cd=0.8)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d["delta_area_pct"], -7.5, delta=0.1)
        v = 40.0 / 3.6
        expect_w = 0.5 * config.AIR_DENSITY_KGM3 * 0.8 * (-0.03) * v ** 3
        self.assertAlmostEqual(d["watts_delta_est"], round(expect_w, 1), delta=0.5)

    def test_enrich_attaches_blocks(self):
        rows = _rows(rpm=60.0, with_rear=True)
        for r in rows:
            r["fa_frac"] = 0.07
        summary = {"duration_s": 20.0}
        C.enrich(summary, rows, {"height_cm": 175.0}, bike_type="road",
                 discipline="cycling")
        self.assertIn("bike", summary)
        self.assertIn("rear", summary)
        self.assertIn("aero", summary)
        self.assertEqual(summary["discipline"], "cycling")

    def test_enrich_running_no_bike(self):
        rows = _rows(rpm=60.0)
        summary = {}
        C.enrich(summary, rows, {}, discipline="running")
        self.assertNotIn("bike", summary)


class TestCycleLatch(unittest.TestCase):
    def test_latches_bdc_and_saves(self):
        import cv2
        latch = C.CycleLatch(per_type=1, max_total=4)
        n = 0
        fps = 30.0
        omega = 2.0 * math.pi * (60.0 / 60.0)
        img = np.full((120, 160, 3), 200, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        jpeg = buf.tobytes()
        for k in range(int(20.0 * fps)):
            phase = omega * (k / fps)
            lm = SC.landmarks_sagittal(phase)
            fm = M.frame_metrics(lm)
            latch.push(k / fps, fm, jpeg, lm)
            n += 1
        evs = latch.events()
        bdc = [e for e in evs if e["type"] == "BDC"]
        tdc = [e for e in evs if e["type"] == "TDC"]
        self.assertGreaterEqual(len(bdc), 30)      # ~2 legs x ~20 revolutions
        self.assertGreaterEqual(len(tdc), 30)
        rev = latch.revolutions()
        self.assertGreaterEqual(rev["left"], 18)
        with tempfile.TemporaryDirectory() as td:
            meta = latch.save_snapshots(Path(td))
            self.assertTrue(meta)
            for m in meta:
                self.assertTrue((Path(td) / m["file"]).exists(), m)

    def test_bdc_events_near_ground_truth_angle(self):
        latch = C.CycleLatch(per_type=2, max_total=8)
        import cv2
        img = np.full((120, 160, 3), 200, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        jpeg = buf.tobytes()
        fps = 30.0
        omega = 2.0 * math.pi * 1.0
        for k in range(int(20.0 * fps)):
            phase = omega * (k / fps)
            lm = SC.landmarks_sagittal(phase)
            fm = M.frame_metrics(lm)
            latch.push(k / fps, fm, jpeg, lm)
        truth = SC.knee_included_angle(0.0)      # BDC
        bdc_vals = [e["knee"] for e in latch.events() if e["type"] == "BDC"]
        self.assertTrue(bdc_vals)
        self.assertAlmostEqual(max(bdc_vals), truth, delta=4.0)


class TestLabelsAndHelpers(unittest.TestCase):
    def test_event_labels_have_bdc_tdc(self):
        self.assertIn("BDC", GA.EVENT_LABEL)
        self.assertIn("TDC", GA.EVENT_LABEL)

    def test_rpm_band_guard(self):
        self.assertIsNone(C.rpm_from_series(np.array([]), np.array([])))
        t = np.arange(0, 20, 0.05)
        y = np.sin(2 * np.pi * 5.0 * t)      # 300 rpm: outside band
        self.assertIsNone(C.rpm_from_series(t, y))


if __name__ == "__main__":
    unittest.main()
