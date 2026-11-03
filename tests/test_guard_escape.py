"""F2 — clinical guard negation scope + escaping of patient/AI text.

The old guard skipped a whole sentence when any negation appeared in its first
60 characters, so ``"no swelling, and the runner has patellar tendinopathy"``
was accepted as a negated statement.  These tests pin clause-scoped negation,
the widened condition list, and the escaping of patient names and model replies
wherever they enter HTML.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import clinical           # noqa: E402
import config             # noqa: E402
import metrics as M       # noqa: E402
import report as R        # noqa: E402
from session_store import Store  # noqa: E402

TEMPLATE = BASE / "templates" / "index.html"


class TestGuardNegationScope(unittest.TestCase):
    def test_five_sentences_four_must_block(self):
        cases = [
            ("No swelling, and the runner has patellar tendinopathy.", True),
            ("There is no evidence of patellofemoral pain syndrome.", False),
            ("The runner has runner's knee on the right.", True),
            ("Treatment with daily loading is recommended.", True),
            ("Plantar fasciopathy explains the morning pain.", True),
        ]
        blocked = 0
        for text, expected in cases:
            got = bool(clinical.guard_text(text))
            self.assertEqual(got, expected, text)
            blocked += got
        self.assertEqual(blocked, 4)

    def test_genuine_negations_still_pass(self):
        for text in (
            "This is not a diagnosis of Achilles tendinopathy.",
            "No evidence of a stress reaction was found.",
            "A diagnosis cannot be made from this video.",
            "No treatment is recommended in this draft.",
        ):
            self.assertEqual(clinical.guard_text(text), [], text)

    def test_condition_names_are_covered(self):
        for text in (
            "The findings suggest greater trochanteric pain syndrome.",
            "This looks like ITB syndrome.",
            "A stress reaction of the tibia is present.",
            "The runner has gluteal tendinopathy.",
        ):
            self.assertTrue(clinical.guard_text(text), text)


class EscapingFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._patches = [
            mock.patch.object(config, "SESSIONS_DIR", self.root / "sessions"),
            mock.patch.object(appmod, "STORE", self.store),
        ]
        for p in self._patches:
            p.start()
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.tmp.cleanup()

    def seed(self, name="<b>x</b>", phone="9001"):
        summary = {
            "view_plane": "sagittal", "duration_s": 10.0, "labels": [],
            "form_score": None,
            "risk": {"level": "Low", "points": 0, "flags": [], "drills": [],
                     "caveat": "test caveat"},
            "data_quality": {"camera_orientation": {
                "guess": "sagittal", "guess_label": "sagittal (side)",
                "nose_offset_ratio": 1.4, "samples": 100, "plane_match": True}},
        }
        pid = self.store.upsert_patient(name, phone)
        sid = self.store.create_session(pid, {}, "side.mp4", view_plane="sagittal")
        self.store.finish_session(sid, 10.0, 100, summary, status="complete")
        return sid


class TestEscaping(EscapingFixture):
    def test_index_template_defines_and_uses_esc(self):
        html = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("function esc(", html)
        for frag in ("esc(s.patient_name)", "esc(s.patient_phone)",
                     "esc(p.name)", "esc(p.phone)", "esc(row.label)",
                     "esc(s.status)"):
            self.assertIn(frag, html)

    def test_patient_name_with_markup_renders_literally(self):
        sid = self.seed("<b>x</b>")
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", html)
        self.assertNotIn("<b>x</b>", html)
        # the served page still carries the helper the tables use
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("function esc(", page)

    def test_coach_reply_is_escaped_before_html(self):
        sid = self.seed("Nia")
        with mock.patch.object(appmod.ai_backend, "describe_images",
                               return_value="<b>x</b>\nline2"):
            r = self.client.post(f"/api/coach/{sid}")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(d["text"], "<b>x</b>\nline2")
        self.assertEqual(d["text_html"], "&lt;b&gt;x&lt;/b&gt;<br>line2")
        report_html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("escHtml(d.text)", report_html)
        self.assertNotIn("d.text.replace", report_html)


if __name__ == "__main__":
    unittest.main()
