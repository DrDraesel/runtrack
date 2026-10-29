"""Report sections for the operator's angle captures and the per-session joint study.

``summary["angle_captures"]`` and ``summary["joint_study"]`` are written by the
capture loop for new sessions only, so every assertion here also has a twin that
proves an older summary (no such keys) renders neither section.  No camera and
no real data dir: a temp DB + temp sessions dir, a hand-built summary, then
``report.build``.

Honesty rules pinned here: the wording must stay "included angle (180 deg =
straight), averaged over tracked frames — not peak flexion, not clinical ROM",
and a label from the operator is escaped exactly like any other text.
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


def base_summary(**over):
    summary = {
        "discipline": "running",
        "view_plane": "sagittal",
        "duration_s": 12.0,
        "form_score": 76.0,
        "form_score_detail": {"method": "weighted composite", "coverage": 0.6,
                              "components": [], "weights": {}},
        "risk": {"level": "Low", "points": 0, "flags": [], "drills": [],
                 "caveat": "test caveat"},
    }
    summary.update(over)
    return summary


# left_knee is measured twice, right_ankle once, trunk_lean once -> three columns
# in JOINTS order (left_knee, right_ankle, trunk_lean); every other joint is absent.
CAPTURES = [
    {"t": 3.24, "label": "mid stance, left leg",
     "angles": {"left_knee": 158.34, "left_hip": 171.0}},
    {"t": 7.56, "label": "toe-off", "angles": {"left_knee": 162.0, "right_ankle": 118.2}},
    {"t": 9.0, "label": "<script>alert(1)</script>", "angles": {"trunk_lean": -6.4}},
]

STUDY = {
    "left_knee": {"mean": 160.06, "min": 148.2, "max": 172.4, "rom": 24.2, "n": 240},
    "right_knee": {"mean": 159.5, "min": 147.0, "max": 170.1, "rom": 23.1, "n": 231},
    "trunk_lean": {"mean": 7.26, "min": 3.1, "max": 11.9, "rom": 8.8, "n": 240},
}


class JointReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def build_html(self, summary, name="Nia", phone="9001"):
        pid = self.store.upsert_patient(name, phone)
        sid = self.store.create_session(pid, {"weight_kg": 75, "height_cm": 178,
                                              "speed_kmh": 10},
                                        "side.mp4", view_plane="sagittal")
        self.store.finish_session(sid, summary.get("duration_s"), 240, summary,
                                  status="complete")
        path = R.build(sid, self.store)          # no metrics.csv: summary-only build
        self.assertTrue(Path(path).exists())
        return Path(path).read_text(encoding="utf-8")


class TestCapturedAnglesSection(unittest.TestCase):
    def setUp(self):
        self.fx = JointReportFixture()
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_one_row_per_capture_with_union_of_joint_columns(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        self.assertIn("<h2>Captured angles (3)</h2>", html)
        self.assertIn("<th>Time (s)</th><th>Label</th>", html)
        # only joints measured in ANY capture get a column, in joint_study.JOINTS order
        for header in ("<th>Left knee</th>", "<th>Left hip</th>", "<th>Right ankle</th>",
                       "<th>Trunk lean</th>"):
            self.assertIn(header, html)
        self.assertLess(html.index("<th>Left knee</th>"), html.index("<th>Left hip</th>"))
        self.assertLess(html.index("<th>Left hip</th>"), html.index("<th>Right ankle</th>"))
        self.assertLess(html.index("<th>Right ankle</th>"), html.index("<th>Trunk lean</th>"))
        # a joint no capture measured has no column at all
        self.assertNotIn("<th>Right knee</th>", html)
        self.assertNotIn("<th>Left shoulder</th>", html)

    def test_values_are_1dp_degrees_and_missing_joint_is_a_dash(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        # columns are left_knee, left_hip, right_ankle, trunk_lean (JOINTS order)
        self.assertIn("<tr><td>3.2</td><td>mid stance, left leg</td>", html)
        self.assertIn("<td>158.3°</td>", html)
        self.assertIn("<td>171.0°</td>", html)
        # toe-off measured knee + ankle; left_hip and trunk_lean were not seen on
        # that frame, so each stays a dash instead of borrowing another value
        self.assertIn("<td>162.0°</td><td>—</td><td>118.2°</td><td>—</td></tr>", html)
        # no 2-dp leakage of a raw capture value
        self.assertNotIn("158.34", html)

    def test_operator_label_is_escaped(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_rows_are_capped_at_40_and_the_cut_is_stated(self):
        caps = [{"t": float(i), "label": f"cap-{i:03d}",
                 "angles": {"left_knee": 150.0 + i}} for i in range(1, 46)]
        html = self.fx.build_html(base_summary(angle_captures=caps))
        self.assertIn("<h2>Captured angles (45)</h2>", html)
        self.assertIn("Showing the first 40 of 45 captures.", html)
        self.assertIn("cap-040", html)
        self.assertNotIn("cap-041", html)
        self.assertNotIn("cap-045", html)

    def test_junk_containers_render_nothing(self):
        for junk in ("not a list", 7, {"t": 1.0}, [None, 3, "x"]):
            html = self.fx.build_html(base_summary(angle_captures=junk))
            self.assertNotIn("Captured angles", html)


class TestJointStudySection(unittest.TestCase):
    def setUp(self):
        self.fx = JointReportFixture()
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_grouped_table_with_1dp_degrees_and_n_frames(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        self.assertIn("<h2>Joint study — per-joint summary</h2>", html)
        self.assertIn("<th>Joint</th><th>Mean</th><th>Min</th><th>Max</th><th>ROM</th>"
                      "<th>n frames</th>", html)
        # groups come from joint_study.GROUPS and only populated ones appear
        self.assertIn("<b>Knee</b>", html)
        self.assertIn("<b>Trunk</b>", html)
        self.assertNotIn("<b>Hip</b>", html)
        self.assertNotIn("<b>Ankle</b>", html)
        self.assertLess(html.index("<b>Knee</b>"), html.index("<b>Trunk</b>"))
        # per-joint mean/min/max/rom at 1 dp, tracked frame count as an int
        self.assertIn("<tr><td>Left knee</td><td>160.1°</td><td>148.2°</td><td>172.4°</td>"
                      "<td>24.2°</td><td>240</td></tr>", html)
        self.assertIn("<tr><td>Right knee</td><td>159.5°</td><td>147.0°</td><td>170.1°</td>"
                      "<td>23.1°</td><td>231</td></tr>", html)
        self.assertIn("<tr><td>Trunk lean</td><td>7.3°</td>", html)

    def test_caveat_keeps_the_honest_wording(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        self.assertIn("180° means the segment is straight", html)
        self.assertIn("Included joint angle", html)
        self.assertIn("<b>not</b> peak flexion", html)
        self.assertIn("clinical ROM", html)
        self.assertIn("clinical norms", html)
        self.assertIn("<b>n frames</b>", html)
        self.assertIn("tracking coverage", html)

    def test_unknown_joint_key_is_not_hidden(self):
        study = dict(STUDY)
        study["left_wrist"] = {"mean": 170.0, "min": 160.0, "max": 179.0,
                               "rom": 19.0, "n": 12}
        html = self.fx.build_html(base_summary(joint_study=study))
        self.assertIn("<b>Other</b>", html)
        self.assertIn("<td>left wrist</td><td>170.0°</td>", html)


class TestPlacementAndAbsence(unittest.TestCase):
    def setUp(self):
        self.fx = JointReportFixture()
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_sections_sit_between_joint_angles_and_symmetry(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        i_joint = html.index("<h2>Joint angles</h2>")
        i_caps = html.index("<h2>Captured angles (3)</h2>")
        i_study = html.index("<h2>Joint study — per-joint summary</h2>")
        i_sym = html.index("<h2>Left / right symmetry")
        self.assertLess(i_joint, i_caps)
        self.assertLess(i_caps, i_study)
        self.assertLess(i_study, i_sym)
        # printed material: neither new section carries the noprint class
        for start, end in ((i_caps, i_study), (i_study, i_sym)):
            self.assertNotIn("noprint", html[start:end])

    def test_forward_reference_points_at_the_new_sections(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES, joint_study=STUDY))
        self.assertIn("Captured angles and the joint study further down", html)
        # the reference lives inside the existing Joint angles block
        i_joint = html.index("<h2>Joint angles</h2>")
        i_ref = html.index("reuse these same included angles")
        self.assertLess(i_joint, i_ref)
        self.assertLess(i_ref, html.index("<h2>Captured angles (3)</h2>"))
        # the old joint table itself is untouched
        self.assertIn("<th>Joint</th><th>Mean</th><th>Min</th><th>Max</th><th>ROM</th>"
                      "<th>Angular vel.</th><th>Tracked</th>", html)

    def test_old_session_without_the_keys_renders_neither_section(self):
        html = self.fx.build_html(base_summary())
        self.assertNotIn("Captured angles", html)
        self.assertNotIn("Joint study", html)
        self.assertNotIn("reuse these same included angles", html)
        # the pre-existing report is intact
        self.assertIn("<h2>Joint angles</h2>", html)
        self.assertIn("<h2>Left / right symmetry", html)

    def test_empty_containers_render_nothing(self):
        html = self.fx.build_html(base_summary(angle_captures=[], joint_study={}))
        self.assertNotIn("Captured angles", html)
        self.assertNotIn("Joint study", html)

    def test_one_block_alone_renders_only_its_section(self):
        html = self.fx.build_html(base_summary(angle_captures=CAPTURES))
        self.assertIn("<h2>Captured angles (3)</h2>", html)
        self.assertNotIn("Joint study", html)
        self.assertIn("Captured angles further down", html)

        html = self.fx.build_html(base_summary(joint_study=STUDY), name="Ola", phone="9002")
        self.assertIn("<h2>Joint study — per-joint summary</h2>", html)
        self.assertNotIn("Captured angles", html)
        self.assertIn("The joint study further down", html)


if __name__ == "__main__":
    unittest.main()
