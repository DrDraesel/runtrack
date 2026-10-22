"""Tests for the clinician-in-the-loop clinical rationale layer."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import clinical  # noqa: E402
import app as appmod  # noqa: E402
from session_store import Store  # noqa: E402

SUMMARY = {
    "discipline": "running",
    "view_plane": "sagittal",
    "duration_s": 60,
    "frames": 600,
    "form": {
        "cadence_spm": {"value": 158, "band": [160, 180], "in_band": False,
                        "label": "Cadence", "unit": "spm"},
        "vertical_osc_cm": {"value": 12.0, "band": [6.0, 9.5], "in_band": False,
                            "label": "Vertical oscillation", "unit": "cm"},
    },
    "risk": {
        "level": "Moderate", "points": 2,
        "flags": [{"id": "low_cadence", "label": "Low cadence", "value": 158,
                   "points": 1}],
        "caveat": "Screening aid only, not a diagnosis.",
    },
    "not_assessed": {"pelvic_drop_deg": "not assessed (wrong camera view)"},
}

VALID = {
    "summary": "Cadence measured 158 spm and vertical oscillation 12.0 cm in this session. These are descriptive findings only.",
    "findings": [{
        "finding": "Cadence 158 spm",
        "evidence_ids": ["RUN-CADENCE"],
        "interpretation": "The measured cadence is below the app's reference band; this is an association, not a diagnosis.",
    }],
    "clinical_questions": ["How do symptoms behave with changes in training load?"],
    "cannot_conclude": ["A diagnosis cannot be made from a video-derived cadence value."],
    "red_flag_note": None,
}


class ClinicalFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = Store(db_path=self.tmp / "t.db",
                           sessions_dir=self.tmp / "sessions")
        pid = self.store.upsert_patient("Synthetic Patient", "555-0100")
        self.sid = self.store.create_session(
            pid, {"weight_kg": 70, "height_cm": 175, "speed_kmh": 10},
            "synthetic", "sagittal", "running")
        self.store.finish_session(self.sid, 60, 600, SUMMARY)
        self.sess = self.store.get_session(self.sid)
        self.assertIsNotNone(self.sess)

    def _generate(self, obj, evidence=None):
        raw = json.dumps(obj)
        return clinical.generate_draft(self.sess, self.store, SUMMARY,
                                       completer=lambda prompt: raw)


class TestIntakeAndFindings(ClinicalFixture):
    def test_intake_does_not_keep_identifiers(self):
        state = clinical.save_intake(self.sess, self.store, {
            "age": 35, "sex": "female", "name": "Synthetic Patient",
            "phone": "555-0100", "pain_nrs": 3,
            "red_flags": [clinical.RED_FLAGS[0], "not-a-real-flag"],
            "proms": {"LEFS": 60},
        })
        self.assertEqual(state["intake"]["age"], 35)
        self.assertEqual(state["intake"]["red_flags"], [clinical.RED_FLAGS[0]])
        self.assertNotIn("name", state["intake"])
        self.assertNotIn("phone", state["intake"])
        self.assertEqual(state["intake"]["proms"]["LEFS"], 60)

    def test_findings_are_deterministic_and_do_not_invent(self):
        findings = clinical.build_findings(self.sess, SUMMARY, {"pain_nrs": 3})
        values = {m["key"]: m["value"] for m in findings["metrics"]}
        self.assertEqual(values["cadence_spm"], 158.0)
        self.assertIn("pelvic_drop_deg", findings["data_quality"]["not_assessed"])
        self.assertEqual(findings["risk"]["level"], "Moderate")
        text = json.dumps(findings)
        self.assertNotIn("Synthetic Patient", text)
        self.assertNotIn("555-0100", text)

    def test_red_flags_create_an_alert_not_a_diagnosis(self):
        findings = clinical.build_findings(self.sess, SUMMARY,
                                           {"red_flags": [clinical.RED_FLAGS[0]]})
        self.assertTrue(findings["red_flag_alert"])
        self.assertIn("red_flags", findings)

    def test_evidence_catalogue_is_citable(self):
        items = clinical.load_evidence()["items"]
        self.assertGreaterEqual(len(items), 10)
        ids = [item["id"] for item in items]
        self.assertEqual(len(ids), len(set(ids)))
        for item in items:
            self.assertTrue(item.get("claim"))
            self.assertTrue(item.get("source"))
            self.assertIn(item.get("grade"), ("A", "B", "C", "D"))


class TestPromptAndGuard(ClinicalFixture):
    def test_prompt_has_hard_rules_and_no_identifiers(self):
        findings = clinical.build_findings(self.sess, SUMMARY, {"age": 35})
        evidence = clinical.select_evidence(findings)
        prompt = clinical.build_prompt(findings, evidence)
        self.assertIn("NOT the treating clinician", prompt)
        self.assertIn("Do not diagnose", prompt)
        self.assertIn("FINDINGS JSON", prompt)
        self.assertIn("RUN-CADENCE", prompt)
        self.assertNotIn("Synthetic Patient", prompt)
        self.assertNotIn("555-0100", prompt)
        self.assertNotIn("treatment plan", prompt.lower().split("do not")[0])

    def test_guard_blocks_affirmative_clinical_claims(self):
        for text in (
            "The patient has patellofemoral pain syndrome.",
            "Start treatment with daily loading.",
            "Prescribe a return to running programme.",
            "The prognosis is excellent.",
        ):
            self.assertTrue(clinical.guard_text(text), text)

    def test_guard_allows_negated_limits(self):
        for text in (
            "This is not a diagnosis.",
            "No treatment is recommended in this draft.",
            "A diagnosis cannot be made from this video.",
        ):
            self.assertEqual(clinical.guard_text(text), [], text)

    def test_parse_rejects_unknown_evidence(self):
        bad = {**VALID, "findings": [{**VALID["findings"][0],
                                      "evidence_ids": ["MADE-UP-ID"]}]}
        parsed = clinical.parse_draft(json.dumps(bad), clinical.load_evidence()["items"])
        self.assertFalse(parsed["ok"])
        self.assertIn("evidence", parsed["error"])


class TestGenerateAndReview(ClinicalFixture):
    def test_valid_draft_is_saved_as_unsigned_draft(self):
        state = self._generate(VALID)
        draft = state["ai_draft"]
        self.assertEqual(draft["status"], "draft")
        self.assertEqual(draft["prompt_version"], clinical.PROMPT_VERSION)
        self.assertTrue(draft["input_hash"])
        self.assertEqual(state.get("review"), None)
        self.assertEqual(draft["findings"][0]["evidence_ids"], ["RUN-CADENCE"])

    def test_unsafe_model_output_is_withheld(self):
        bad = {**VALID, "summary": "The patient has patellofemoral pain syndrome and should start treatment."}
        state = self._generate(bad)
        self.assertEqual(state["ai_draft"]["status"], "blocked")
        self.assertTrue(state["ai_draft"]["issues"])
        public = clinical.public_state(self.sess, self.store, SUMMARY)
        self.assertNotIn("raw_text", public["ai_draft"])
        self.assertNotIn("start treatment", json.dumps(public).lower())

    def test_review_requires_attestation_and_clinician_fields(self):
        with self.assertRaises(ValueError):
            clinical.save_review(self.sess, self.store, {
                "clinician_name": "Dr Test", "credentials": "MD",
                "assessment": "A", "plan": "P"})
        with self.assertRaises(ValueError):
            clinical.save_review(self.sess, self.store, {
                "clinician_name": "", "credentials": "MD",
                "assessment": "A", "plan": "P", "attestation": True})
        state = clinical.save_review(self.sess, self.store, {
            "clinician_name": "Dr Test", "credentials": "MD",
            "assessment": "Clinician assessment", "plan": "Clinician plan",
            "attestation": True})
        self.assertEqual(state["review"]["status"], "signed")
        self.assertEqual(state["review"]["version"], 1)
        self.assertEqual(state["audit"][-1]["action"], "clinician_review_signed")

    def test_intake_change_invalidates_a_signed_review(self):
        clinical.save_review(self.sess, self.store, {
            "clinician_name": "Dr Test", "credentials": "MD",
            "assessment": "Assessment", "plan": "Plan", "attestation": True})
        state = clinical.save_intake(self.sess, self.store, {"age": 36})
        self.assertEqual(state["review"]["status"], "stale")
        self.assertIn("clinician_review_stale",
                      [entry["action"] for entry in state["audit"]])

    def test_ai_backend_failure_is_explicit(self):
        def boom(_prompt):
            raise RuntimeError("model unavailable")
        state = clinical.generate_draft(self.sess, self.store, SUMMARY, completer=boom)
        self.assertEqual(state["ai_draft"]["status"], "error")
        self.assertIn("model unavailable", state["ai_draft"]["issues"][0])


class TestClinicalApiWithSession(ClinicalFixture):
    def setUp(self):
        super().setUp()
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()

    def test_page_state_and_intake(self):
        with mock.patch.object(appmod, "STORE", self.store):
            r = self.client.get(f"/clinical/{self.sid}")
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Clinical rationale workspace", r.data)
            r = self.client.get(f"/api/clinical/{self.sid}")
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["ok"])
            self.assertIn("findings", r.get_json())
            r = self.client.post(f"/api/clinical/{self.sid}/intake",
                                 json={"age": 35, "pain_nrs": 3,
                                       "red_flags": [clinical.RED_FLAGS[0]]})
            self.assertEqual(r.status_code, 200)
            state = r.get_json()["state"]
            self.assertEqual(state["intake"]["age"], 35)
            self.assertEqual(len(state["intake"]["red_flags"]), 1)


class TestClinicalApi(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()

    def test_unknown_session_page_is_404(self):
        self.assertEqual(self.client.get("/clinical/999999").status_code, 404)

    def test_unknown_session_api_is_404(self):
        r = self.client.get("/api/clinical/999999")
        self.assertEqual(r.status_code, 404)
        self.assertFalse(r.get_json()["ok"])

    def test_review_endpoint_rejects_missing_signature(self):
        r = self.client.post("/api/clinical/999999/review", json={})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
