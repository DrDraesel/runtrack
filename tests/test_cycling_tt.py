"""TT/tri bike-fit targets, elbow form entry and the live rolling fit estimate.

Covers the additive cycling.py API: fit_targets(), the bike_elbow form entry,
TT/tri elbow suggestions and live_fit()/live_status().  Uses only synthetic
rows, so no hardware, video or app process is involved.
"""
import json
import math
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import config  # noqa: E402
import cycling as C  # noqa: E402

FIT_KEYS = ("knee_bdc_deg", "knee_tdc_deg", "hip_tdc_deg",
            "torso_deg", "ankle_deg", "elbow_deg")


def _elbow_rows(value, seconds=2.0, fps=20.0):
    """Minimal side-view rows carrying only an elbow angle (deg)."""
    n = int(seconds * fps)
    return [{"t_rel": k / fps, "left_elbow": float(value),
             "right_elbow": float(value)} for k in range(n)]


def _live_rows(seconds=10.0, fps=30.0, rpm=78.0, knee_lo=100.0, knee_hi=140.0,
               trunk_lean=-40.0, elbow=160.0, start=0.0):
    """Synthetic live frame-metric rows for a steady pedalling effort.

    The knee angle sweeps knee_lo..knee_hi at the cadence frequency (one
    oscillation per crank revolution), both legs in phase so max/min are the
    extremes of that envelope.  NOTE: the envelope is written as mid + amp*cos;
    100+40*cos would sweep 60..140, so 100..140 is 120+20*cos.
    """
    mid = 0.5 * (knee_lo + knee_hi)
    amp = 0.5 * (knee_hi - knee_lo)
    freq = rpm / 60.0
    rows = []
    n = int(round(seconds * fps))
    for k in range(n + 1):
        t = start + k / fps
        knee = mid + amp * math.cos(2.0 * math.pi * freq * t)
        rows.append({
            "t_rel": float(t),
            "left_knee": knee, "right_knee": knee,
            "left_hip": 75.0 + 0.2 * (knee - mid),
            "right_hip": 75.0 + 0.2 * (knee - mid),
            "left_elbow": float(elbow), "right_elbow": float(elbow),
            "trunk_lean": float(trunk_lean),
        })
    return rows


class TestFitTargets(unittest.TestCase):
    def test_road_targets(self):
        t = C.fit_targets("road")
        self.assertEqual(tuple(sorted(t)), tuple(sorted(FIT_KEYS)))
        self.assertEqual(t["knee_bdc_deg"], (140.0, 150.0))
        self.assertEqual(t["elbow_deg"], (150.0, 170.0))
        for key, band in t.items():
            self.assertIsInstance(band, tuple, key)
            self.assertEqual(len(band), 2, key)
            self.assertIsInstance(band[0], float, key)
            self.assertLess(band[0], band[1], key)

    def test_tt_targets_are_aerobar(self):
        t = C.fit_targets("tt")
        self.assertEqual(t["elbow_deg"], (85.0, 105.0))
        self.assertEqual(t["torso_deg"], (10.0, 25.0))
        self.assertEqual(t["hip_tdc_deg"], (40.0, 60.0))
        self.assertEqual(C.fit_targets("tri")["elbow_deg"], (85.0, 105.0))

    def test_unknown_bike_falls_back_to_road(self):
        road = C.fit_targets("road")
        self.assertEqual(C.fit_targets("gravel"), road)
        self.assertEqual(C.fit_targets(None), road)
        self.assertEqual(C.fit_targets(""), road)

    def test_default_is_road(self):
        self.assertEqual(C.fit_targets(), C.fit_targets("road"))

    def test_returns_a_copy_not_config(self):
        t = C.fit_targets("road")
        t["knee_bdc_deg"] = (0.0, 1.0)
        self.assertEqual(config.CYCLING_BANDS["knee_bdc_deg"]["road"],
                         (140.0, 150.0))
        self.assertEqual(C.fit_targets("road")["knee_bdc_deg"], (140.0, 150.0))


class TestElbowFormEntry(unittest.TestCase):
    def test_road_elbow_in_band(self):
        bike = C.analyze_cycling(_elbow_rows(160.0), "road")
        e = bike["form"]["bike_elbow"]
        self.assertEqual(e["label"], "Elbow angle (mean)")
        self.assertEqual(e["unit"], "deg")
        self.assertEqual(e["band"], [150.0, 170.0])
        self.assertTrue(e["in_band"])
        self.assertAlmostEqual(e["value"], 160.0, delta=0.1)

    def test_road_locked_elbow_out_of_band(self):
        bike = C.analyze_cycling(_elbow_rows(176.0), "road")
        e = bike["form"]["bike_elbow"]
        self.assertFalse(e["in_band"])
        self.assertEqual(e["band"], [150.0, 170.0])

    def test_tt_elbow_in_band(self):
        bike = C.analyze_cycling(_elbow_rows(96.0), "tt")
        e = bike["form"]["bike_elbow"]
        self.assertEqual(e["band"], [85.0, 105.0])
        self.assertTrue(e["in_band"])

    def test_tri_elbow_out_of_band(self):
        bike = C.analyze_cycling(_elbow_rows(150.0), "tri")
        self.assertFalse(bike["form"]["bike_elbow"]["in_band"])

    def test_no_elbow_data_no_entry(self):
        rows = [{"t_rel": k / 30.0, "left_knee": 140.0} for k in range(40)]
        bike = C.analyze_cycling(rows, "road")
        self.assertNotIn("bike_elbow", bike["form"])


class TestElbowSuggestions(unittest.TestCase):
    def _actions(self, bike):
        return " | ".join(s["action"].lower() for s in bike["suggestions"])

    def _findings(self, bike):
        return " | ".join(s["finding"].lower() for s in bike["suggestions"])

    def test_tt_too_straight_shortens_pad_reach(self):
        bike = C.analyze_cycling(_elbow_rows(130.0), "tt")
        actions = self._actions(bike)
        self.assertIn("shorten pad reach", actions)
        self.assertIn("raise pads", actions)
        self.assertIn("straighter", self._findings(bike))

    def test_tt_overbent_extends_reach(self):
        bike = C.analyze_cycling(_elbow_rows(78.0), "tt")
        actions = self._actions(bike)
        self.assertIn("extend reach", actions)
        self.assertIn("lower pads", actions)

    def test_tri_gets_the_same_aerobar_direction(self):
        bike = C.analyze_cycling(_elbow_rows(130.0), "tri")
        self.assertIn("shorten pad reach", self._actions(bike))

    def test_road_locked_arms_cue(self):
        bike = C.analyze_cycling(_elbow_rows(176.0), "road")
        actions = self._actions(bike)
        self.assertIn("arms locked", actions)
        self.assertIn("bend elbows", actions)

    def test_road_in_band_has_no_elbow_suggestion(self):
        bike = C.analyze_cycling(_elbow_rows(160.0), "road")
        self.assertNotIn("elbow", self._actions(bike))

    def test_wording_is_coaching_not_diagnosis(self):
        for bike_type, value in (("tt", 130.0), ("tt", 78.0), ("road", 176.0)):
            bike = C.analyze_cycling(_elbow_rows(value), bike_type)
            text = (self._actions(bike) + " " + self._findings(bike)).lower()
            for word in ("injur", "pain", "syndrome", "tendin", "diagnos"):
                self.assertNotIn(word, text, (bike_type, value, text))


class TestLiveFit(unittest.TestCase):
    def test_bdc_and_tdc_from_sine(self):
        fit = C.live_fit(_live_rows(seconds=10.0, rpm=78.0), "road")
        # true envelope 100..140 deg; BDC = most extended, TDC = most flexed
        self.assertIsNotNone(fit["knee_bdc_deg"])
        self.assertIsNotNone(fit["knee_tdc_deg"])
        self.assertAlmostEqual(fit["knee_bdc_deg"], 140.0, delta=2.0)
        self.assertAlmostEqual(fit["knee_tdc_deg"], 100.0, delta=2.0)

    def test_tdc_rejects_bad_pose_readings(self):
        rows = []
        knees = [30.0, 45.0, 100.0, 140.0, 120.0]
        for i, k in enumerate(knees):
            rows.append({"t_rel": i / 30.0, "left_knee": k, "right_knee": k})
        fit = C.live_fit(rows, "road")
        self.assertAlmostEqual(fit["knee_tdc_deg"], 100.0, delta=0.1)
        self.assertAlmostEqual(fit["knee_bdc_deg"], 140.0, delta=0.1)

    def test_cadence_near_78_rpm(self):
        rows = _live_rows(seconds=10.0, rpm=78.0)
        # a 5 s FFT window quantises to ~12 rpm/bin, so assert within a bin;
        # a 10 s window is the accurate read of the same series.
        short = C.live_fit(rows, "road")
        self.assertIsNotNone(short["cadence_rpm"])
        self.assertAlmostEqual(short["cadence_rpm"], 78.0, delta=6.5)
        long = C.live_fit(rows, "road", window_s=10.0)
        self.assertAlmostEqual(long["cadence_rpm"], 78.0, delta=2.0)

    def test_cadence_none_under_4_seconds(self):
        fit = C.live_fit(_live_rows(seconds=3.0, rpm=78.0), "road")
        self.assertIsNone(fit["cadence_rpm"])
        self.assertIsNotNone(fit["knee_bdc_deg"])     # angles still measured

    def test_honours_the_last_window_only(self):
        rows = []
        for k in range(0, 20 * 30 + 1):
            t = k / 30.0
            elbow = 160.0 if t < 15.0 else 95.0
            rows.append({"t_rel": t, "left_knee": 120.0, "right_knee": 120.0,
                         "left_elbow": elbow, "right_elbow": elbow,
                         "trunk_lean": -40.0})
        fit = C.live_fit(rows, "tt", window_s=5.0)
        self.assertEqual(fit["n_frames"], 151)        # 15.0 .. 20.0 s at 30 fps
        self.assertAlmostEqual(fit["elbow_deg"], 95.0, delta=0.1)
        self.assertAlmostEqual(fit["window_s"], 5.0, delta=1e-9)
        # a full-session window sees the mixed position instead
        whole = C.live_fit(rows, "tt", window_s=20.0)
        self.assertGreater(whole["elbow_deg"], 140.0)

    def test_torso_and_elbow_means(self):
        rows = _live_rows(seconds=6.0, trunk_lean=-40.0, elbow=100.0)
        fit = C.live_fit(rows, "tt")
        self.assertAlmostEqual(fit["torso_deg"], 50.0, delta=0.5)   # 90 - |lean|
        self.assertAlmostEqual(fit["elbow_deg"], 100.0, delta=0.1)

    def test_hip_tdc_is_the_minimum(self):
        fit = C.live_fit(_live_rows(seconds=8.0, knee_lo=100.0, knee_hi=140.0),
                         "tri")
        self.assertIsNotNone(fit["hip_tdc_deg"])
        self.assertAlmostEqual(fit["hip_tdc_deg"], 71.0, delta=1.0)  # 75 - 0.2*20

    def test_one_leg_only_still_measured(self):
        rows = [{"t_rel": k / 30.0,
                 "right_knee": 120.0 + 20.0 * math.cos(2 * math.pi * 1.3 * k / 30.0),
                 "right_elbow": 96.0, "right_hip": 70.0}
                for k in range(301)]
        fit = C.live_fit(rows, "tt", window_s=10.0)
        self.assertAlmostEqual(fit["knee_bdc_deg"], 140.0, delta=2.0)
        self.assertAlmostEqual(fit["elbow_deg"], 96.0, delta=0.1)
        self.assertIsNotNone(fit["cadence_rpm"])     # falls back to the right leg

    def test_empty_rows_is_none_safe(self):
        for rows in ([], None, ()):
            fit = C.live_fit(rows, "tt")
            self.assertEqual(fit["n_frames"], 0)
            self.assertEqual(fit["bike_type"], "tt")
            for key in ("knee_bdc_deg", "knee_tdc_deg", "hip_tdc_deg",
                        "torso_deg", "elbow_deg", "cadence_rpm"):
                self.assertIsNone(fit[key], key)
            self.assertTrue(all(v is None for v in fit["status"].values()))

    def test_too_short_rows_do_not_invent_values(self):
        fit = C.live_fit(_live_rows(seconds=2.0)[:2], "road")
        self.assertEqual(fit["n_frames"], 2)
        for key in ("knee_bdc_deg", "knee_tdc_deg", "hip_tdc_deg",
                    "torso_deg", "elbow_deg", "cadence_rpm"):
            self.assertIsNone(fit[key], key)

    def test_rows_without_time_are_ignored(self):
        fit = C.live_fit([{"left_knee": 140.0}] * 10, "road")
        self.assertEqual(fit["n_frames"], 0)
        self.assertIsNone(fit["knee_bdc_deg"])

    def test_all_nan_metrics_are_none(self):
        rows = [{"t_rel": k / 30.0, "left_knee": float("nan"),
                 "right_knee": float("nan"), "left_elbow": float("nan"),
                 "trunk_lean": float("nan")} for k in range(120)]
        fit = C.live_fit(rows, "road")
        self.assertEqual(fit["n_frames"], 120)
        for key in ("knee_bdc_deg", "knee_tdc_deg", "hip_tdc_deg",
                    "torso_deg", "elbow_deg", "cadence_rpm"):
            self.assertIsNone(fit[key], key)
        json.dumps(fit, allow_nan=False)            # no NaN leaks out

    def test_window_arg_is_validated(self):
        rows = _live_rows(seconds=6.0)
        self.assertAlmostEqual(C.live_fit(rows, window_s=0)["window_s"], 5.0)
        self.assertAlmostEqual(C.live_fit(rows, window_s=-2)["window_s"], 5.0)
        self.assertAlmostEqual(C.live_fit(rows, window_s="bad")["window_s"], 5.0)

    def test_unknown_bike_falls_back_to_road(self):
        fit = C.live_fit(_live_rows(seconds=6.0), "gravel")
        self.assertEqual(fit["bike_type"], "road")
        self.assertEqual(fit["targets"],
                         {k: list(v) for k, v in C.fit_targets("road").items()})


class TestLiveFitShape(unittest.TestCase):
    def setUp(self):
        self.fit = C.live_fit(_live_rows(seconds=10.0), "tt")

    def test_json_safe(self):
        text = json.dumps(self.fit, allow_nan=False)
        self.assertIn("knee_bdc_deg", text)
        self.assertIsInstance(self.fit["n_frames"], int)
        self.assertIsInstance(self.fit["window_s"], float)

    def test_targets_shape(self):
        targets = self.fit["targets"]
        self.assertEqual(tuple(sorted(targets)), tuple(sorted(FIT_KEYS)))
        for key, band in targets.items():
            self.assertIsInstance(band, list, key)
            self.assertEqual(len(band), 2, key)
            self.assertTrue(all(isinstance(x, float) for x in band), key)
            self.assertLess(band[0], band[1], key)
        self.assertEqual(targets["elbow_deg"], [85.0, 105.0])

    def test_status_keys_and_values(self):
        status = self.fit["status"]
        self.assertEqual(set(status), set(FIT_KEYS) | {"cadence_rpm"})
        for key, value in status.items():
            self.assertIn(value, ("in", "low", "high", None), key)

    def test_status_directions_match_bands(self):
        self.assertEqual(self.fit["status"]["knee_bdc_deg"], "in")   # 140 in 140-145
        self.assertEqual(self.fit["status"]["elbow_deg"], "high")    # 160 > 105
        self.assertEqual(self.fit["status"]["torso_deg"], "high")    # 50 > 25 (tt)
        self.assertEqual(self.fit["status"]["ankle_deg"], None)      # not measured
        road = C.live_fit(_live_rows(seconds=10.0, trunk_lean=-50.0), "road")
        self.assertEqual(road["status"]["torso_deg"], "in")          # 40 in 35-45

    def test_live_status_helper(self):
        self.assertEqual(C.live_status({"knee_bdc_deg": 155.0}, "road")
                         ["knee_bdc_deg"], "high")
        self.assertEqual(C.live_status({"knee_bdc_deg": 130.0}, "road")
                         ["knee_bdc_deg"], "low")
        self.assertEqual(C.live_status({"knee_bdc_deg": 145.0}, "road")
                         ["knee_bdc_deg"], "in")
        self.assertIsNone(C.live_status({"knee_bdc_deg": None}, "road")
                          ["knee_bdc_deg"])
        self.assertIsNone(C.live_status({}, "road")["cadence_rpm"])
        self.assertEqual(C.live_status({"cadence_rpm": 120.0}, "road")
                         ["cadence_rpm"], "high")
        self.assertIsNone(C.live_status("not a dict", "road")["elbow_deg"])

    def test_analyze_and_live_use_the_same_bands(self):
        bike = C.analyze_cycling(_elbow_rows(96.0), "tt")
        self.assertEqual(bike["form"]["bike_elbow"]["band"],
                         list(C.fit_targets("tt")["elbow_deg"]))


if __name__ == "__main__":
    unittest.main()
