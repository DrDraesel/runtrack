"""Tests for the advisor: form score weights, risk stratification, comparison."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import advisor  # noqa: E402
import config  # noqa: E402


def sagittal_summary(**kw):
    s = {
        "view_plane": "sagittal",
        "cadence_spm": 180.0,
        "symmetry_mean_pct": 96.0,
        "vertical_osc_cm": 8.0,
        "touchdown_shin_deg": {"mean": 3.0},
        "overstride_leg_frac": {"mean": 0.05},
        "knee_flexion_ms_deg": {"mean": 40.0},
        "trunk_lean_fwd_deg": {"mean": 7.0},
    }
    s.update(kw)
    return s


def bad_sagittal_summary():
    return sagittal_summary(
        cadence_spm=140.0, symmetry_mean_pct=80.0, vertical_osc_cm=14.0,
        touchdown_shin_deg={"mean": 14.0}, overstride_leg_frac={"mean": 0.30},
        knee_flexion_ms_deg={"mean": 10.0}, trunk_lean_fwd_deg={"mean": 25.0})


class TestFormScore(unittest.TestCase):
    def test_band_score_endpoints(self):
        self.assertEqual(advisor.band_score(5.0, 0, 10, 10), 100.0)
        self.assertEqual(advisor.band_score(0.0, 10, 20, 10), 0.0)
        self.assertEqual(advisor.band_score(15.0, 10, 20, 10), 100.0)  # inside the band
        self.assertEqual(advisor.band_score(25.0, 10, 20, 10), 50.0)   # halfway out
        self.assertEqual(advisor.band_score(30.0, 10, 20, 10), 0.0)
        self.assertIsNone(advisor.band_score(None, 0, 10, 5))

    def test_good_session_scores_high(self):
        fs = advisor.form_score(sagittal_summary())
        self.assertGreaterEqual(fs["score"], 90.0)
        self.assertEqual(fs["coverage"], 1.0)
        self.assertEqual(fs["plane"], "sagittal")

    def test_bad_session_scores_low(self):
        fs = advisor.form_score(bad_sagittal_summary())
        self.assertLess(fs["score"], 30.0)
        worst = [c for c in fs["components"] if c["metric"] == "touchdown_shin_deg"][0]
        # 14 deg shin with band (0, 5) and a 12-deg tolerance -> 100*(1-9/12) = 25
        b = config.BANDS["touchdown_shin_deg"]
        self.assertAlmostEqual(worst["score"],
                               advisor.band_score(14.0, b[0], b[1], 12.0), places=3)
        self.assertLess(worst["score"], 50.0)

    def test_missing_components_are_renormalised(self):
        s = sagittal_summary()
        s.pop("overstride_leg_frac")
        s.pop("trunk_lean_fwd_deg")
        fs = advisor.form_score(s)
        self.assertLess(fs["coverage"], 1.0)
        self.assertIsNotNone(fs["score"])
        statuses = {c["metric"]: c["status"] for c in fs["components"]}
        self.assertEqual(statuses["overstride_leg_frac"], "not tracked")
        self.assertEqual(statuses["trunk_lean_deg"], "not tracked")

    def test_wrong_plane_components_are_not_scored(self):
        s = {"view_plane": "frontal", "cadence_spm": 180.0, "symmetry_mean_pct": 96.0,
             "vertical_osc_cm": 8.0, "pelvic_drop_deg": {"mean": 2.0},
             "knee_valgus_deg": {"mean": 4.0},
             "not_assessed": {"touchdown_shin_deg": config.NOT_ASSESSED_TEXT,
                              "overstride_leg_frac": config.NOT_ASSESSED_TEXT,
                              "knee_flexion_ms_deg": config.NOT_ASSESSED_TEXT,
                              "trunk_lean_deg": config.NOT_ASSESSED_TEXT}}
        fs = advisor.form_score(s)
        self.assertEqual(fs["plane"], "frontal")
        comps = {c["metric"]: c for c in fs["components"]}
        self.assertNotIn("touchdown_shin_deg", comps)
        self.assertIn("pelvic_drop_deg", comps)
        self.assertEqual(comps["pelvic_drop_deg"]["score"], 100.0)
        self.assertGreaterEqual(fs["score"], 90.0)

    def test_no_data(self):
        self.assertIsNone(advisor.form_score({})["score"])


class TestRisk(unittest.TestCase):
    def test_level_mapping(self):
        self.assertEqual(advisor.risk_level(0), "Low")
        self.assertEqual(advisor.risk_level(1), "Low")
        self.assertEqual(advisor.risk_level(2), "Moderate")
        self.assertEqual(advisor.risk_level(3), "Moderate")
        self.assertEqual(advisor.risk_level(4), "Elevated")
        self.assertEqual(advisor.risk_level(6), "High")

    def test_clean_session_is_low_risk(self):
        r = advisor.risk_stratification(sagittal_summary())
        self.assertEqual(r["level"], "Low")
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["drills"], [])

    def test_flagged_session_gets_drills(self):
        s = sagittal_summary(touchdown_shin_deg={"mean": 12.0},
                             overstride_leg_frac={"mean": 0.30},
                             cadence_spm=150.0)
        r = advisor.risk_stratification(s)
        ids = {f["id"] for f in r["flags"]}
        self.assertIn("braking", ids)
        self.assertIn("overstride", ids)
        self.assertIn("low_cadence", ids)
        self.assertEqual(r["points"], 5)
        self.assertEqual(r["level"], "Elevated")
        self.assertTrue(r["drills"])
        self.assertIn("caveat", r)

    def test_frontal_only_flags_ignored_in_sagittal(self):
        s = sagittal_summary()
        s["pelvic_drop_deg"] = {"mean": 9.0}
        s["knee_valgus_deg"] = {"mean": 20.0}
        s["not_assessed"] = {"pelvic_drop_deg": config.NOT_ASSESSED_TEXT,
                             "knee_valgus_deg": config.NOT_ASSESSED_TEXT}
        r = advisor.risk_stratification(s)
        self.assertEqual(r["flags"], [])

    def test_frontal_session_flags_pelvic_drop(self):
        s = {"view_plane": "frontal", "cadence_spm": 180.0, "symmetry_mean_pct": 96.0,
             "vertical_osc_cm": 8.0, "pelvic_drop_deg": {"mean": 7.0},
             "knee_valgus_deg": {"mean": 4.0}}
        r = advisor.risk_stratification(s)
        ids = {f["id"] for f in r["flags"]}
        self.assertIn("pelvic_drop", ids)
        self.assertEqual(r["level"], "Moderate")
        self.assertTrue(any("glute" in d["drill"].lower() or "hip" in d["drill"].lower()
                            for d in r["drills"]))

    def test_caveat_mentions_screening(self):
        self.assertIn("not a diagnosis", advisor.EVIDENCE_CAVEAT.lower())


class TestCompare(unittest.TestCase):
    def _pair(self):
        a = sagittal_summary(cadence_spm=162.0, symmetry_mean_pct=88.0,
                             vertical_osc_cm=11.0, knee_flexion_ms_deg={"mean": 30.0})
        b = sagittal_summary(cadence_spm=178.0, symmetry_mean_pct=95.0,
                             vertical_osc_cm=8.0, knee_flexion_ms_deg={"mean": 40.0})
        return a, b

    def test_deltas_and_verdicts(self):
        a, b = self._pair()
        c = advisor.compare(a, b, {"id": 1}, {"id": 2})
        rows = {r["metric"]: r for r in c["rows"]}
        self.assertEqual(rows["cadence_spm"]["delta"], 16.0)
        self.assertEqual(rows["cadence_spm"]["verdict"], "Improved")
        self.assertEqual(rows["symmetry_pct"]["verdict"], "Improved")
        self.assertEqual(rows["vertical_osc_cm"]["verdict"], "Improved")   # lower is better
        self.assertEqual(rows["knee_flexion_ms_deg"]["verdict"], "Improved")  # toward band
        self.assertEqual(rows["form_score"]["verdict"], "Improved")
        self.assertEqual(c["summary"]["regressed"], 0)
        self.assertGreaterEqual(c["summary"]["improved"], 4)

    def test_regression_detected(self):
        a, b = self._pair()
        c = advisor.compare(b, a)                       # reversed = worse
        rows = {r["metric"]: r for r in c["rows"]}
        self.assertEqual(rows["cadence_spm"]["verdict"], "Regressed")
        self.assertEqual(rows["vertical_osc_cm"]["verdict"], "Regressed")
        self.assertEqual(rows["knee_flexion_ms_deg"]["verdict"], "Regressed")

    def test_neutral_zone(self):
        a = sagittal_summary(cadence_spm=180.0, vertical_osc_cm=8.0)
        b = sagittal_summary(cadence_spm=181.0, vertical_osc_cm=8.2)
        c = advisor.compare(a, b)
        rows = {r["metric"]: r for r in c["rows"]}
        self.assertEqual(rows["cadence_spm"]["verdict"], "Neutral")
        self.assertEqual(rows["vertical_osc_cm"]["verdict"], "Neutral")

    def test_wrong_plane_metrics_dropped(self):
        a = sagittal_summary()
        b = sagittal_summary()
        a["not_assessed"] = b["not_assessed"] = {"pelvic_drop_deg": config.NOT_ASSESSED_TEXT}
        c = advisor.compare(a, b)
        self.assertNotIn("pelvic_drop_deg", {r["metric"] for r in c["rows"]})

    def test_missing_value_is_not_assessed(self):
        a = sagittal_summary()
        b = sagittal_summary()
        a.pop("touchdown_shin_deg")
        c = advisor.compare(a, b)
        row = {r["metric"]: r for r in c["rows"]}["touchdown_shin_deg"]
        self.assertEqual(row["verdict"], "n/a")
        self.assertIsNone(row["delta"])

    def test_required_metrics_present(self):
        a, b = self._pair()
        c = advisor.compare(a, b)
        keys = {r["metric"] for r in c["rows"]}
        for k in ("cadence_spm", "symmetry_pct", "knee_flexion_ms_deg",
                  "vertical_osc_cm", "form_score"):
            self.assertIn(k, keys)


class TestEnrich(unittest.TestCase):
    def test_enrich_adds_score_and_risk(self):
        s = advisor.enrich(sagittal_summary())
        self.assertIsNotNone(s["form_score"])
        self.assertIn("risk", s)
        self.assertIn("form_score_detail", s)
        self.assertEqual(s["risk"]["level"], "Low")


if __name__ == "__main__":
    unittest.main()
