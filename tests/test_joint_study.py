"""Joint-angle study helpers: snapshot, capture, summarize, overview."""
import math
import unittest

import joint_study as JS


class TestSnapshot(unittest.TestCase):
    def test_keeps_only_measured_joints(self):
        live = {"left_knee": 158.24, "right_knee": None,
                "trunk_lean": -4.06, "left_hip": float("nan"),
                "fps": 22.1, "pose_found": True}
        out = JS.snapshot(live)
        self.assertEqual(out, {"left_knee": 158.2, "trunk_lean": -4.1})

    def test_empty_and_garbage(self):
        self.assertEqual(JS.snapshot({}), {})
        self.assertEqual(JS.snapshot(None), {})
        self.assertEqual(JS.snapshot({"left_knee": "n/a"}), {})

    def test_angles_are_1dp_floats(self):
        out = JS.snapshot({"left_elbow": 179.94})
        self.assertIsInstance(out["left_elbow"], float)
        self.assertEqual(out["left_elbow"], 179.9)


class TestCapture(unittest.TestCase):
    def test_appends_record(self):
        session = {"elapsed": 12.345, "angle_captures": []}
        rec = JS.capture(session, {"left_knee": 155.0, "trunk_lean": 7.0}, "  mid   swing ")
        self.assertEqual(rec["t"], 12.35)
        self.assertEqual(rec["label"], "mid swing")      # whitespace collapsed
        self.assertEqual(rec["angles"], {"left_knee": 155.0, "trunk_lean": 7.0})
        self.assertEqual(session["angle_captures"], [rec])

    def test_creates_list_when_missing(self):
        session = {"elapsed": 1.0}
        JS.capture(session, {"right_knee": 90.0})
        self.assertEqual(len(session["angle_captures"]), 1)
        self.assertEqual(session["angle_captures"][0]["label"], "Capture 1")

    def test_refuses_without_angles(self):
        session = {"elapsed": 1.0, "angle_captures": []}
        with self.assertRaises(ValueError):
            JS.capture(session, {"pose_found": False}, "x")
        self.assertEqual(session["angle_captures"], [])

    def test_refuses_without_session(self):
        with self.assertRaises(ValueError):
            JS.capture(None, {"left_knee": 100.0})

    def test_label_is_truncated(self):
        session = {"elapsed": 0.0}
        rec = JS.capture(session, {"left_knee": 100.0}, "z" * 200)
        self.assertEqual(len(rec["label"]), JS.LABEL_MAX_CHARS)

    def test_limit(self):
        session = {"elapsed": 0.0, "angle_captures": []}
        for _ in range(JS.MAX_CAPTURES):
            JS.capture(session, {"left_knee": 100.0})
        with self.assertRaises(ValueError):
            JS.capture(session, {"left_knee": 100.0})
        self.assertEqual(len(session["angle_captures"]), JS.MAX_CAPTURES)


class TestSummarize(unittest.TestCase):
    def test_stats(self):
        rows = [{"left_knee": 150.0}, {"left_knee": 170.0},
                {"left_knee": None}, {"right_knee": 90.0}]
        out = JS.summarize(rows)
        self.assertEqual(out["left_knee"],
                         {"mean": 160.0, "min": 150.0, "max": 170.0,
                          "rom": 20.0, "n": 2})
        self.assertEqual(out["right_knee"]["n"], 1)
        self.assertEqual(out["right_knee"]["rom"], 0.0)

    def test_skips_non_finite_and_unknown(self):
        rows = [{"trunk_lean": float("inf")}, {"trunk_lean": 3.0},
                {"nope": 1.0}, "not a dict"]
        out = JS.summarize(rows)
        self.assertEqual(out["trunk_lean"]["n"], 1)
        self.assertNotIn("nope", out)

    def test_empty(self):
        self.assertEqual(JS.summarize([]), {})
        self.assertEqual(JS.summarize(None), {})

    def test_only_known_joints(self):
        out = JS.summarize([{"fps": 30.0, "left_knee": 100.0}])
        self.assertEqual(list(out), ["left_knee"])


class TestOverview(unittest.TestCase):
    def test_counts_and_last(self):
        caps = [{"t": 1.0, "label": "a", "angles": {"left_knee": 100.0}},
                {"t": 2.0, "label": "b", "angles": {"right_knee": 90.0}}]
        out = JS.overview({"angle_captures": caps})
        self.assertEqual(out["captures"], 2)
        self.assertEqual(out["last"]["label"], "b")
        self.assertEqual(out["joints"], ["left_knee", "right_knee"])

    def test_empty(self):
        out = JS.overview({})
        self.assertEqual(out["captures"], 0)
        self.assertIsNone(out["last"])

    def test_labels_cover_every_joint(self):
        for joint in JS.JOINTS:
            self.assertIn(joint, JS.LABELS)
        grouped = [j for _, js in JS.GROUPS for j in js]
        self.assertEqual(sorted(grouped), sorted(JS.JOINTS))


if __name__ == "__main__":
    unittest.main()
