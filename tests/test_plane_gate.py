"""F1 — the camera-plane gate.

An unverified plane (stored ``plane_match`` False, an ``ambiguous`` guess, or no
orientation block at all on a sagittal-labelled session) makes every
plane-specific number invalid.  The report must print one blocking block and
withhold the form score, risk level, joint-angle table and gait metric table —
it used to print a red warning and then render them anyway.

``/api/session/start`` must also refuse to record a plane that contradicts the
live rig's own ``plane_guess`` unless the caller explicitly confirms it.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import config             # noqa: E402
import metrics as M       # noqa: E402
import report as R        # noqa: E402
from session_store import Store  # noqa: E402


def synth_rows(n=240, fps=30.0, cadence_spm=180.0, shin=3.0, osc=0.02):
    """Same shape as the report fixture: a running cadence with legs."""
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        y_l = 0.90 + 0.05 * math.sin(ph)
        y_r = 0.90 + 0.05 * math.sin(ph + math.pi)
        dx = 0.2 * math.tan(math.radians(shin))
        rows.append({
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
        })
    return rows


ORIENTS = {
    "valid": {"guess": "sagittal", "guess_label": "sagittal (side)",
              "nose_offset_ratio": 1.42, "samples": 240, "plane_match": True},
    "mismatch": {"guess": "frontal", "guess_label": "frontal (front/rear)",
                 "nose_offset_ratio": 0.17, "samples": 240, "plane_match": False},
    "ambiguous": {"guess": "ambiguous", "guess_label": "ambiguous",
                  "nose_offset_ratio": 0.42, "samples": 240, "plane_match": None},
}


class ReportGateFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def build(self, orient="valid", plane="sagittal", save_csv=True):
        rows = synth_rows()
        summary = M.analyze_session(rows, weight_kg=75, height_cm=178,
                                    speed_kmh=10, view_plane=plane)
        dq = summary.setdefault("data_quality", {})
        if orient == "missing":
            dq.pop("camera_orientation", None)
            save_csv = False       # the metrics.csv fallback must not rescue it
        else:
            dq["camera_orientation"] = dict(ORIENTS[orient])
        pid = self.store.upsert_patient("Nia", "9001")
        sid = self.store.create_session(
            pid, {"weight_kg": 75, "height_cm": 178, "speed_kmh": 10},
            "side.mp4", view_plane=plane)
        if save_csv:                       # "missing" also removes the CSV fallback
            self.store.save_csv(sid, rows)
        self.store.finish_session(sid, summary.get("duration_s"), len(rows),
                                  summary, status="complete")
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        return summary, html


class TestReportPlaneGate(ReportGateFixture):
    BLOCKED_NEEDLES = (
        "Overall form score",
        "Injury risk screening",
        "<h2>Joint angles</h2>",
        "Form metrics vs reference bands",
        "Gait phases latched",
        "Left / right symmetry",
        'class="k">Form score',
        'class="k">Risk level',
    )

    def assert_blocked(self, html):
        self.assertIn("Not scoreable — camera plane not verified", html)
        self.assertIn('id="plane-block"', html)
        for needle in self.BLOCKED_NEEDLES:
            self.assertNotIn(needle, html, needle)
        # one blocking block, not the old red caveat plus a score
        self.assertEqual(html.count('id="plane-block"'), 1)

    def test_mismatch_blocks_the_score_and_states_the_ratio(self):
        _, html = self.build("mismatch")
        self.assert_blocked(html)
        self.assertIn("front/rear", html)
        self.assertIn("0.17 shoulder widths", html)

    def test_ambiguous_guess_blocks_the_score(self):
        _, html = self.build("ambiguous")
        self.assert_blocked(html)
        self.assertIn("ambiguous band", html)
        self.assertIn("0.42 shoulder widths", html)

    def test_missing_orientation_blocks_a_sagittal_session(self):
        _, html = self.build("missing")
        self.assert_blocked(html)
        self.assertIn("No camera-orientation check was stored", html)
        self.assertIn("not measurable", html)

    def test_verified_plane_still_renders_every_score(self):
        _, html = self.build("valid")
        self.assertNotIn("Not scoreable", html)
        for needle in ("Overall form score", "Injury risk screening",
                       "<h2>Joint angles</h2>", "Form metrics vs reference bands",
                       "Gait phases latched"):
            self.assertIn(needle, html, needle)
        # height/speed-derived cards are labelled estimates
        self.assertIn("Vertical osc. (est.)", html)
        self.assertIn("Step length (est.)", html)
        self.assertNotIn("(est.) (est.)", html)

    def test_frontal_session_is_not_blocked_by_missing_orientation(self):
        # the "missing block" rule is specified for sagittal-labelled sessions
        _, html = self.build("missing", plane="frontal")
        self.assertNotIn("Not scoreable", html)


class SessionStartPlaneCase(unittest.TestCase):
    """Drive the real start endpoint against a fake live rig."""

    ATTRS = ("active", "error", "session", "rig", "source", "source_label",
             "source2", "source2_label", "source3", "source3_label", "stalled",
             "stalled2", "stalled3", "warn2", "warn3", "view_plane", "thread",
             "stop_evt", "frames1", "frames2", "frames3")

    def setUp(self):
        self._saved = {k: getattr(appmod.L, k) for k in self.ATTRS}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._cfg = mock.patch.object(config, "SESSIONS_DIR", Path(self.tmp.name))
        self._cfg.start()
        self.addCleanup(self._cfg.stop)
        with appmod.L.lock:
            appmod.L.reset()
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()

    def tearDown(self):
        appmod.L.stop_evt.set()
        for k, v in self._saved.items():
            setattr(appmod.L, k, v)

    def rig(self, guess):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
            appmod.L.rig = {"1": {"plane_guess": guess, "ok": False,
                                  "plane": "sagittal"}}

    def post(self, **body):
        payload = {"name": "Dana", "phone": "444", "duration_s": 30}
        payload.update(body)
        with mock.patch.object(appmod.STORE, "upsert_patient", return_value=1), \
                mock.patch.object(appmod.STORE, "create_session", return_value=91):
            return self.client.post("/api/session/start", json=payload)

    def test_refuses_a_plane_that_contradicts_the_live_rig(self):
        self.rig("frontal")
        r = self.post(view_plane="sagittal")
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertFalse(d["ok"])
        self.assertIn("frontal", d["error"])
        self.assertIn("sagittal", d["error"])
        self.assertIn("confirm_plane_mismatch", d["error"])
        self.assertIsNone(appmod.L.session, "no session may be created on refusal")

    def test_confirm_override_records_the_mismatch_in_the_summary(self):
        self.rig("frontal")
        r = self.post(view_plane="sagittal", confirm_plane_mismatch=True)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        with appmod.L.lock:
            sess = appmod.L.session
            self.assertTrue(sess["plane_mismatch_confirmed"])
            sess["latch"] = None
            sess["stills"] = None
            sid = sess["id"]
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as fin, \
                mock.patch.object(appmod.R, "build"):
            self.assertEqual(appmod._finalize("stopped"), sid)
        summary = fin.call_args[0][3]
        self.assertTrue(summary["plane_mismatch_confirmed"])
        self.assertEqual(summary["plane_guess"], "frontal")
        self.assertIn("confirmed a camera-plane mismatch", summary["note"])

    def test_matching_plane_and_unknown_guess_start_normally(self):
        self.rig("sagittal")
        r = self.post(view_plane="sagittal")
        self.assertEqual(r.status_code, 200)
        with appmod.L.lock:
            self.assertFalse(appmod.L.session["plane_mismatch_confirmed"])
            appmod.L.session = None
        self.rig("unknown")
        r = self.post(view_plane="sagittal")
        self.assertEqual(r.status_code, 200)
        with appmod.L.lock:
            self.assertFalse(appmod.L.session["plane_mismatch_confirmed"])

    def test_no_rig_means_no_gate(self):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "USB camera 2"
            appmod.L.rig = {}
        self.assertEqual(self.post(view_plane="sagittal").status_code, 200)


if __name__ == "__main__":
    unittest.main()
