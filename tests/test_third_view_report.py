"""Third-view (front camera) report section: heading, scores, flags, sync note.

The third view is built by the same ``second_view.analyze`` call as the second
view, so the report consumes one frozen shape: ``summary["third_view"]`` plus
``plane3`` / ``direction3`` / ``source3_label`` / ``multi_view_sync``.  These
tests feed a hand-built summary (no camera, no real data dir) and check the
printed HTML.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import report as R  # noqa: E402
from session_store import Store  # noqa: E402


def third_view_block(**over):
    """Minimal stand-in for second_view.analyze() output (front camera)."""
    sv3 = {
        "view_plane": "frontal",
        "form_score": 71.0,
        "rows": 240,
        "duration_s": 8.0,
        "form": {
            "pelvic_drop_deg": {"label": "Contralateral pelvic drop", "value": 6.4,
                                "band": [0, 5], "in_band": False},
            "knee_valgus_deg": {"label": "Frontal knee collapse", "value": 3.1,
                                "band": [0, 8], "in_band": True},
        },
        "not_assessed": {
            "touchdown_shin_deg": config.NOT_ASSESSED_TEXT,
            "overstride_leg_frac": config.NOT_ASSESSED_TEXT,
        },
        "risk": {"level": "Moderate", "points": 12, "flags": [
            {"id": "pelvic_drop", "metric": "pelvic_drop_deg",
             "value": 6.4, "points": 12,
             "finding": "Contralateral pelvic drop 6.4 deg — hip abductor control"},
        ]},
    }
    sv3.update(over)
    return sv3


def second_view_block():
    """Minimal stand-in for the existing rear-clip block (unchanged output)."""
    return {
        "view_plane": "frontal",
        "form_score": 84.0,
        "rows": 238,
        "duration_s": 8.0,
        "form": {"knee_valgus_deg": {"label": "Frontal knee collapse", "value": 2.0,
                                     "band": [0, 8], "in_band": True}},
        "not_assessed": {"touchdown_shin_deg": config.NOT_ASSESSED_TEXT},
        "risk": {"level": "Low", "points": 0, "flags": []},
    }


_DEFAULT_THIRD = object()


def fake_summary(third_view=_DEFAULT_THIRD):
    summary = {
        "discipline": "running",
        "view_plane": "sagittal",
        "duration_s": 8.0,
        "form_score": 78.0,
        "form_score_detail": {"method": "weighted composite", "coverage": 0.5,
                              "components": [], "weights": {}},
        "risk": {"level": "Low", "points": 0, "flags": [], "drills": [],
                 "caveat": "test caveat"},
        "second_view": second_view_block(),
        "plane2": "frontal",
        "source2_label": "rear.mp4",
    }
    if third_view is _DEFAULT_THIRD:
        third_view = third_view_block()
    if third_view is not None:
        summary["third_view"] = third_view
        summary["plane3"] = "frontal"
        summary["direction3"] = "front"
        summary["source3_label"] = "front.mp4"
        summary["multi_view_sync"] = {
            "clocks": "software",
            "views": [
                {"role": "primary", "label": "side.mp4", "rows": 240},
                {"role": "second", "label": "rear.mp4", "rows": 238},
                {"role": "third", "label": "front.mp4", "rows": 240},
            ],
            "note": "no hardware genlock; aligned by per-view t_rel timestamps; "
                    "expect up to ~1 frame offset",
        }
    return summary


class ThirdViewReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def build_html(self, summary, name="Nia", phone="999"):
        pid = self.store.upsert_patient(name, phone)
        sid = self.store.create_session(pid, {"weight_kg": 75, "height_cm": 178,
                                              "speed_kmh": 10},
                                        "side.mp4 + rear.mp4 + front.mp4",
                                        view_plane="sagittal")
        # no metrics.csv: the report must still build from the summary alone
        self.store.finish_session(sid, summary.get("duration_s"), 240, summary,
                                  status="complete")
        return Path(R.build(sid, self.store)).read_text(encoding="utf-8")


class TestThirdViewReport(unittest.TestCase):
    def setUp(self):
        self.fx = ThirdViewReportFixture()
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_third_view_section_renders(self):
        html = self.fx.build_html(fake_summary())
        # heading uses the direction word + the third source label
        self.assertIn("Third view — front view (front.mp4)", html)
        self.assertIn("third-view form score <b>71 / 100</b>", html)
        self.assertIn("Third-view score", html)
        self.assertIn("third view: <b>front</b> (front.mp4)", html)
        # its own metric table + not-assessed rows
        self.assertIn("Contralateral pelvic drop", html)
        self.assertIn("Frontal knee collapse", html)
        self.assertIn(config.NOT_ASSESSED_TEXT, html)
        self.assertIn("touchdown shin", html)
        # its own risk flags
        self.assertIn("Third-view risk flags", html)
        self.assertIn("Contralateral pelvic drop 6.4 deg", html)
        self.assertIn("(12 pt)", html)
        # software-clock sync line: per-view frame counts + no-genlock note
        self.assertIn("Multi-view sync", html)
        self.assertIn("side.mp4 (240 frames)", html)
        self.assertIn("rear.mp4 (238 frames)", html)
        self.assertIn("front.mp4 (240 frames)", html)
        self.assertIn("software clock only", html)
        self.assertIn("no hardware genlock", html)
        # each view gated to its own plane; never averaged
        self.assertIn("Each view is gated to its own plane", html)
        self.assertIn("never averaged", html)

    def test_direction_word_and_plane_fallback(self):
        html = self.fx.build_html(
            fake_summary(third_view_block(view_plane="frontal")), name="Ola", phone="1001")
        self.assertIn("Third view — front view", html)

        summary = fake_summary(third_view_block(view_plane="frontal"))
        summary["direction3"] = "rear"
        html = self.fx.build_html(summary, name="Ola", phone="1001")
        self.assertIn("Third view — rear view (front.mp4)", html)

        # no direction word: fall back to the plane the clip was analysed in
        summary = fake_summary(third_view_block(view_plane="frontal"))
        summary.pop("direction3")
        html = self.fx.build_html(summary, name="Ola", phone="1001")
        self.assertIn("Third view — frontal view (front.mp4)", html)

    def test_no_third_view_means_no_section(self):
        summary = fake_summary(third_view=None)
        # metadata may linger from the capture loop; without the block nothing renders
        summary["plane3"] = "frontal"
        summary["direction3"] = "front"
        summary["source3_label"] = "front.mp4"
        html = self.fx.build_html(summary, name="Pia", phone="1002")
        self.assertNotIn("Third view —", html)
        self.assertNotIn("Third-view score", html)
        self.assertNotIn("Multi-view sync", html)

    def test_sync_line_requires_multi_view_sync(self):
        summary = fake_summary()
        summary.pop("multi_view_sync")
        html = self.fx.build_html(summary, name="Rae", phone="1004")
        # section still renders; the frame-count line is conditional on the block
        self.assertIn("Third view — front view (front.mp4)", html)
        self.assertNotIn("Multi-view sync", html)

    def test_second_view_output_is_unchanged(self):
        html = self.fx.build_html(fake_summary(), name="Quinn", phone="1003")
        self.assertIn("Second view — frontal view (rear.mp4)", html)
        self.assertIn("Second-view score", html)
        self.assertIn("second view: <b>frontal</b>", html)
        # second view comes first, third view is added after it
        self.assertLess(html.index("Second view —"), html.index("Third view —"))
        # no second-view risk flags -> the old fallback text is still there
        self.assertIn("No risk flags were triggered in the second view.", html)

        # without a third view the second-view block alone reads as before
        html = self.fx.build_html(fake_summary(third_view=None), name="Quinn", phone="1003")
        self.assertIn("Second view — frontal view (rear.mp4)", html)
        self.assertIn("Second-view score", html)
        self.assertNotIn("Third view —", html)


if __name__ == "__main__":
    unittest.main()
