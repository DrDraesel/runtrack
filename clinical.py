"""Clinician-in-the-loop clinical rationale for RunTrack.

This module deliberately does **not** let the language model diagnose, prescribe,
clear an athlete, or write the clinical record.  It builds a de-identified,
deterministic findings package from the measured session, attaches a curated
evidence catalogue, asks the local model for a structured *draft*, validates the
draft against the same evidence catalogue, and stores it separately from the
clinician's own assessment and signature.

The signed clinician review - not the AI text - is the clinical record.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import advisor
import ai_backend

SCHEMA = "runtrack.clinical.v1"
PROMPT_VERSION = "runtrack-clinical-v1"
EVIDENCE_PATH = Path(__file__).with_name("clinical_evidence.json")
DISCLAIMER = (
    "AI-generated draft for clinician review only. It is not a diagnosis, not a "
    "treatment plan, and not part of the clinical record until reviewed, edited "
    "and signed by a licensed clinician."
)
REVIEW_DISCLAIMER = (
    "This clinical note was authored and signed by the clinician below. The "
    "AI draft is supporting material only and is not the clinical record."
)

RED_FLAGS = (
    "Chest pain, pressure or tightness during exertion",
    "Fainting or near-fainting during exercise",
    "Unexplained shortness of breath at low load",
    "Night pain or rest pain that wakes the patient",
    "Unexplained weight loss or fever",
    "Recent significant trauma / inability to bear weight",
    "Progressive neurological symptoms (numbness, weakness, bladder/bowel change)",
    "Calf pain with swelling, warmth or redness",
    "Locking, giving way or a hot swollen joint",
    "Known cardiac, renal or metabolic disease without clearance",
)

SEXES = ("", "female", "male", "other", "prefer not to say")
MAX_TEXT = 2000
MAX_SHORT = 240

# Evidence IDs used by the deterministic findings table.  The catalogue itself
# is data (clinical_evidence.json) so a clinician can review/update it without
# touching the engine.
_METRIC_EVIDENCE = {
    "cadence_spm": ("RUN-CADENCE", "RUN-METHOD"),
    "overstride_leg_frac": ("RUN-OVERSTRIDE", "RUN-METHOD"),
    "touchdown_shin_deg": ("RUN-OVERSTRIDE", "RUN-METHOD"),
    "knee_flexion_ic_deg": ("RUN-KINEMATICS", "RUN-METHOD"),
    "knee_flexion_ms_deg": ("RUN-KINEMATICS", "RUN-METHOD"),
    "pelvic_drop_deg": ("RUN-HIP", "RUN-METHOD"),
    "knee_valgus_deg": ("RUN-HIP", "RUN-METHOD"),
    "hip_extension_toeoff_deg": ("RUN-HIP", "RUN-METHOD"),
    "pushoff_ankle_deg": ("RUN-ANKLE", "RUN-METHOD"),
    "trunk_lean_fwd_deg": ("RUN-KINEMATICS", "RUN-METHOD"),
    "vertical_osc_cm": ("RUN-KINEMATICS", "RUN-METHOD"),
    "symmetry_mean_pct": ("RUN-METHOD",),
    "asymmetry": ("RUN-METHOD",),
    "knee_bdc_deg": ("CYC-KNEE", "CYC-METHOD"),
    "knee_tdc_min_deg": ("CYC-KNEE", "CYC-METHOD"),
    "hip_tdc_min_deg": ("CYC-HIP", "CYC-METHOD"),
    "torso_deg": ("CYC-TORSO", "CYC-METHOD"),
    "elbow_deg": ("CYC-ELBOW", "CYC-METHOD"),
    "cadence_rpm": ("CYC-METHOD",),
    "fa_m2": ("CYC-AERO", "CYC-METHOD"),
    "pelvic_obliquity_deg": ("RUN-HIP", "RUN-METHOD"),
    "hip_vertical_travel": ("RUN-HIP", "RUN-METHOD"),
    "knee_lateral_travel_left": ("RUN-HIP", "RUN-METHOD"),
    "knee_lateral_travel_right": ("RUN-HIP", "RUN-METHOD"),
}

_RISK_EVIDENCE = {
    "low_cadence": ("RUN-CADENCE",),
    "braking": ("RUN-OVERSTRIDE",),
    "overstride": ("RUN-OVERSTRIDE",),
    "overstride_frac": ("RUN-OVERSTRIDE",),
    "knee_flexion_ic": ("RUN-KINEMATICS",),
    "knee_flexion_ms": ("RUN-KINEMATICS",),
    "pelvic_drop": ("RUN-HIP",),
    "valgus": ("RUN-HIP",),
    "oscillation": ("RUN-KINEMATICS",),
    "asymmetry": ("RUN-METHOD",),
    "hip_extension": ("RUN-HIP",),
    "ankle_pushoff": ("RUN-ANKLE",),
    "cadence_low": ("CYC-METHOD",),
    "cadence_high": ("CYC-METHOD",),
    "knee_bdc_high": ("CYC-KNEE",),
    "knee_bdc_low": ("CYC-KNEE",),
    "hip_tdc_low": ("CYC-HIP",),
    "torso_high": ("CYC-TORSO",),
}

_BANNED = re.compile(
    r"\b("
    r"diagnos(?:e|es|ed|ing|is|tic)|"
    r"prescrib(?:e|es|ed|ing|ption)|"
    r"treatment|therapy|therapeutic|surgery|surgical|medication|"
    r"injection|inject(?:ed|ion)?|"
    r"return\s*to\s*(?:sport|play|running|training)|"
    r"clear(?:ed|ance)?\s*(?:for|to)\s*(?:sport|play|running|training)|"
    r"prognosis|cure|recover(?:y|ed)?|"
    r"differential\s+diagnosis|"
    r"recommend(?:ed|ing)?\s+(?:treatment|surgery|medication|injection|therapy)"
    r")\b",
    re.IGNORECASE,
)
_CONDITIONS = re.compile(
    r"\b("
    r"patellofemoral\s+pain(?:\s+syndrome)?|"
    r"iliotibial\s+band\s+syndrome|ITB\s+syndrome|ITBS|"
    r"medial\s+tibial\s+stress\s+syndrome|MTSS|shin\s+splints?|"
    r"Achilles\s+tendinopathy|plantar\s+fasci(?:itis|opathy)|"
    r"stress\s+fracture|fracture|hamstring\s+strain|"
    r"osteoarthritis|tendinopathy|compartment\s+syndrome|"
    r"cauda\s+equina|infection|tumou?r|"
    r"patellar\s+tendinopathy|gluteal\s+tendinopathy|"
    r"plantar\s+heel\s+pain"
    r")\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|no|never|cannot|can't|without|does not|do not|isn't|aren't|"
    r"must not|should not|unable to|rather than|instead of|free of)\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value, limit=MAX_TEXT) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _short(value) -> str:
    return _text(value, MAX_SHORT)


def _number(value, lo=None, hi=None):
    if value in (None, "", "null"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    if lo is not None and out < lo:
        return None
    if hi is not None and out > hi:
        return None
    return out


def load_evidence() -> dict:
    try:
        data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": "missing", "items": []}
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return {"version": "invalid", "items": []}
    items = [item for item in data["items"] if isinstance(item, dict) and item.get("id")]
    return {"version": str(data.get("version") or "unknown"), "items": items}


def evidence_by_id() -> dict:
    return {str(item["id"]): item for item in load_evidence()["items"]}


def _clean_intake(data: dict) -> dict:
    data = data if isinstance(data, dict) else {}
    flags = data.get("red_flags") if isinstance(data.get("red_flags"), list) else []
    allowed_flags = set(RED_FLAGS)
    proms = data.get("proms")
    if not isinstance(proms, dict):
        proms = {}
    clean_proms = {}
    limits = {"LEFS": (0, 80), "OSTRC": (0, 100), "NPRS": (0, 10), "VISA": (0, 100)}
    for key in ("LEFS", "OSTRC", "NPRS", "VISA", "other"):
        if key in proms:
            value = proms.get(key)
            clean_proms[key] = (_number(value, *limits[key]) if key in limits
                                else _short(value))
    return {
        "age": _number(data.get("age"), 10, 100),
        "sex": (data.get("sex") if data.get("sex") in SEXES else ""),
        "height_cm": _number(data.get("height_cm"), 100, 230),
        "weight_kg": _number(data.get("weight_kg"), 25, 250),
        "training_load_km_week": _number(data.get("training_load_km_week"), 0, 500),
        "training_change": _short(data.get("training_change")),
        "pain_nrs": _number(data.get("pain_nrs"), 0, 10),
        "pain_location": _short(data.get("pain_location")),
        "pain_duration_weeks": _number(data.get("pain_duration_weeks"), 0, 1040),
        "pain_onset": _short(data.get("pain_onset")),
        "injury_history": _text(data.get("injury_history")),
        "goals": _text(data.get("goals")),
        "red_flags": [flag for flag in flags if flag in allowed_flags],
        "red_flag_notes": _text(data.get("red_flag_notes")),
        "exam_findings": _text(data.get("exam_findings")),
        "imaging_labs": _text(data.get("imaging_labs")),
        "medications": _text(data.get("medications")),
        "other_conditions": _text(data.get("other_conditions")),
        "proms": clean_proms,
    }


def _deidentified_context(sess: dict, intake: dict) -> dict:
    """Only fields needed for the rationale; never name, phone or patient id."""
    ctx = {
        "age": intake.get("age"),
        "sex": intake.get("sex"),
        "height_cm": intake.get("height_cm") or _number(sess.get("height_cm"), 100, 230),
        "weight_kg": intake.get("weight_kg") or _number(sess.get("weight_kg"), 25, 250),
        "training_load_km_week": intake.get("training_load_km_week"),
        "training_change": intake.get("training_change"),
        "pain_nrs": intake.get("pain_nrs"),
        "pain_location": intake.get("pain_location"),
        "pain_duration_weeks": intake.get("pain_duration_weeks"),
        "pain_onset": intake.get("pain_onset"),
        "injury_history": intake.get("injury_history"),
        "goals": intake.get("goals"),
        "red_flags": intake.get("red_flags") or [],
        "red_flag_notes": intake.get("red_flag_notes"),
        "exam_findings": intake.get("exam_findings"),
        "imaging_labs": intake.get("imaging_labs"),
        "medications": intake.get("medications"),
        "other_conditions": intake.get("other_conditions"),
        "proms": intake.get("proms") or {},
    }
    return {key: value for key, value in ctx.items()
            if value not in (None, "", [], {})}


def _metric_entry(key, entry, discipline):
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    band = entry.get("band")
    if discipline == "cycling":
        ids = _METRIC_EVIDENCE.get(key) or ("CYC-METHOD",)
    else:
        ids = _METRIC_EVIDENCE.get(key) or ("RUN-METHOD",)
    return {
        "key": key,
        "label": _short(entry.get("label") or key.replace("_", " ")),
        "value": _number(value),
        "unit": _short(entry.get("unit")),
        "band": list(band) if isinstance(band, (list, tuple)) and len(band) == 2 else None,
        "in_band": entry.get("in_band") if isinstance(entry.get("in_band"), bool) else None,
        "evidence_ids": list(ids),
    }


def build_findings(sess: dict, summary: dict, intake: dict | None = None) -> dict:
    """Deterministic, de-identified findings used by the draft and the report."""
    summary = summary if isinstance(summary, dict) else {}
    intake = _clean_intake(intake or {})
    discipline = str(summary.get("discipline") or sess.get("discipline") or "running").lower()
    if discipline not in ("running", "cycling"):
        discipline = "running"
    metrics = []
    if discipline == "cycling":
        bike = summary.get("bike") or {}
        for key, entry in (bike.get("form") or {}).items():
            item = _metric_entry(key, entry, discipline)
            if item:
                metrics.append(item)
        for key in ("pelvic_obliquity_deg", "hip_vertical_travel",
                    "knee_lateral_travel_left", "knee_lateral_travel_right"):
            entry = (summary.get("rear") or {}).get(key)
            if isinstance(entry, dict):
                item = _metric_entry(key, entry, discipline)
                if item:
                    metrics.append(item)
    else:
        for key, entry in (summary.get("form") or {}).items():
            item = _metric_entry(key, entry, discipline)
            if item:
                metrics.append(item)
        # Scalar metrics that are not already in the form table.
        seen = {m.get("key") for m in metrics}
        for key in ("cadence_spm", "trunk_lean_fwd_deg", "vertical_osc_cm",
                    "symmetry_mean_pct", "touchdown_shin_deg",
                    "knee_flexion_ic_deg", "pelvic_drop_deg", "knee_valgus_deg"):
            if key in seen:
                continue
            value = advisor.metric(summary, key)
            if value is None:
                continue
            metrics.append({
                "key": key,
                "label": key.replace("_", " "),
                "value": _number(value),
                "unit": "spm" if key == "cadence_spm" else ("%" if "symmetry" in key else
                         ("cm" if key == "vertical_osc_cm" else "deg")),
                "band": None, "in_band": None,
                "evidence_ids": list(_METRIC_EVIDENCE.get(key, ("RUN-METHOD",))),
            })
    risk = summary.get("risk") or {}
    risk_flags = []
    for flag in risk.get("flags") or []:
        if not isinstance(flag, dict):
            continue
        flag_id = str(flag.get("id") or "")
        risk_flags.append({
            "id": flag_id,
            "finding": _short(flag.get("label") or flag.get("finding") or flag_id),
            "value": _number(flag.get("value")),
            "points": _number(flag.get("points"), 0, 100),
            "evidence_ids": list(_RISK_EVIDENCE.get(flag_id, ())),
        })
    red_flags = intake.get("red_flags") or []
    return {
        "discipline": discipline,
        "view_plane": str(summary.get("view_plane") or sess.get("view_plane") or "sagittal"),
        "bike_type": summary.get("bike_type") or sess.get("bike_type"),
        "metrics": metrics,
        "risk": {
            "level": _short(risk.get("level")),
            "points": _number(risk.get("points"), 0, 100),
            "flags": risk_flags,
            "caveat": _text(risk.get("caveat") or advisor.EVIDENCE_CAVEAT),
            "method": "deterministic screening heuristic; not a diagnosis",
        },
        "data_quality": {
            "frames": _number(summary.get("frames") or sess.get("frames"), 0, 10_000_000),
            "duration_s": _number(summary.get("duration_s") or sess.get("duration_s"), 0, 100000),
            "coverage": summary.get("form_score_detail", {}).get("coverage")
            if isinstance(summary.get("form_score_detail"), dict) else None,
            "not_assessed": summary.get("not_assessed") or {},
        },
        "patient_context": _deidentified_context(sess, intake),
        "red_flags": red_flags,
        "red_flag_alert": bool(red_flags),
    }


def select_evidence(findings: dict) -> list:
    discipline = findings.get("discipline") or "running"
    tags = {"METHOD", "RUN-METHOD" if discipline == "running" else "CYC-METHOD"}
    ids = set(tags)
    for metric in findings.get("metrics") or []:
        ids.update(metric.get("evidence_ids") or [])
    for flag in (findings.get("risk") or {}).get("flags") or []:
        ids.update(flag.get("evidence_ids") or [])
    if findings.get("red_flag_alert"):
        ids.add("RED-FLAGS")
    proms = (findings.get("patient_context") or {}).get("proms") or {}
    for key, ev_id in (("LEFS", "PROM-LEFS"), ("OSTRC", "PROM-OSTRC"),
                       ("NPRS", "PROM-NPRS"), ("VISA", "PROM-VISA")):
        if key in proms:
            ids.add(ev_id)
    if findings.get("discipline") == "cycling":
        ids.update({"CYC-METHOD", "CYC-SAFETY"})
    else:
        ids.update({"RUN-METHOD", "RUN-SAFETY"})
    out = []
    for item in load_evidence()["items"]:
        if str(item.get("id")) in ids:
            out.append(item)
    return out


def _evidence_text(items: list) -> str:
    lines = []
    for item in items:
        lines.append(
            f"- {item.get('id')}: {item.get('claim')} "
            f"({item.get('source')}, {item.get('year')}; grade {item.get('grade')}). "
            f"Caveat: {item.get('caveat')}"
        )
    return "\n".join(lines)


def build_prompt(findings: dict, evidence: list) -> str:
    payload = json.dumps(findings, separators=(",", ":"), default=str)[:12000]
    return (
        "You are a clinical documentation assistant preparing a draft for a licensed "
        "clinician. You are NOT the treating clinician. Do not diagnose or name a "
        "disease as present, recommend or prescribe treatment, medication, surgery, "
        "injections, therapy, clearance or return to sport, predict prognosis, or infer "
        "pain, emotion, attention or mental state. The clinician will complete the "
        "assessment, differential and plan and will sign the record.\n"
        "Use ONLY the FINDINGS and EVIDENCE below. Never invent a measurement, a normal "
        "range, a diagnosis or a citation. If a value is missing or marked not assessed, "
        "say 'not assessed'. Describe associations cautiously; do not turn a risk flag "
        "into a diagnosis. If red flags are present, state that the clinician must "
        "evaluate them promptly; do not attribute them to a diagnosis.\n"
        "Return STRICT JSON only, no markdown fences, with exactly these keys:\n"
        '{"summary": "2-3 sentences", '
        '"findings": [{"finding": "...", "evidence_ids": ["E-ID"], '
        '"interpretation": "descriptive, non-diagnostic"}], '
        '"clinical_questions": ["question for the clinician"], '
        '"cannot_conclude": ["..."], '
        '"red_flag_note": "..." or null}\n'
        "Every findings item must cite at least one evidence id from EVIDENCE. "
        "Do not include a treatment plan, diagnosis, differential diagnosis or "
        "prescription. The only allowed clinical actions are questions and statements "
        "of what cannot be concluded.\n\n"
        "FINDINGS JSON:\n" + payload +
        "\n\nEVIDENCE (cite by id only):\n" + _evidence_text(evidence)
    )


def _sentences(text: str):
    return re.split(r"(?<=[.!?])\s+", str(text or ""))


def guard_text(text: str) -> list:
    """Return human-readable issues for unsupported clinical claims."""
    issues = []
    for sentence in _sentences(text):
        match = _BANNED.search(sentence) or _CONDITIONS.search(sentence)
        if not match:
            continue
        prefix = sentence[:match.start()]
        if _NEGATION.search(prefix) or _NEGATION.search(sentence[:60]):
            continue
        issues.append(f"banned clinical claim: '{match.group(0)}' in: {sentence[:180]}")
    return issues


def _clean_list(value, key, max_items=12, max_len=500):
    out = []
    for item in list(value or [])[:max_items]:
        if isinstance(item, str):
            item = {key: item}
        if not isinstance(item, dict):
            continue
        cleaned = {}
        for k, v in item.items():
            if isinstance(v, str):
                cleaned[k] = _text(v, max_len)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                cleaned[k] = _number(v)
            elif isinstance(v, list):
                cleaned[k] = [str(x)[:80] for x in v[:8]]
        if cleaned:
            out.append(cleaned)
    return out


def parse_draft(raw: str, evidence: list) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"ok": False, "error": "model did not return a JSON object", "raw": raw}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as exc:
        return {"ok": False, "error": f"invalid JSON: {exc}", "raw": raw}
    if not isinstance(data, dict):
        return {"ok": False, "error": "JSON root is not an object", "raw": raw}
    allowed = {str(item.get("id")) for item in evidence}
    findings = _clean_list(data.get("findings"), "finding")
    if not findings:
        return {"ok": False, "error": "no findings were returned", "raw": raw}
    for item in findings:
        ids = [str(x) for x in (item.get("evidence_ids") or []) if str(x) in allowed]
        if not ids:
            return {"ok": False,
                    "error": "a finding has no valid evidence id from the supplied catalogue",
                    "raw": raw}
        item["evidence_ids"] = ids
    questions = [str(x)[:400] for x in (data.get("clinical_questions") or []) if str(x).strip()][:10]
    cannot = []
    for item in (data.get("cannot_conclude") or [])[:10]:
        text = str(item).strip()[:500]
        if not text:
            continue
        if not re.match(r"^(cannot|can't|not|no|never|unable)\b", text, re.IGNORECASE):
            text = "Cannot conclude: " + text
        cannot.append(text)
    draft = {
        "summary": _text(data.get("summary"), 1200),
        "findings": findings,
        "clinical_questions": questions,
        "cannot_conclude": cannot,
        "red_flag_note": _text(data.get("red_flag_note"), 500) or None,
        "disclaimer": DISCLAIMER,
    }
    issues = guard_text(draft["summary"])
    for item in findings:
        issues.extend(guard_text(item.get("finding", "")))
        issues.extend(guard_text(item.get("interpretation", "")))
    issues.extend(guard_text(draft["red_flag_note"] or ""))
    for question in questions:
        issues.extend(guard_text(question))
    for statement in cannot:
        issues.extend(guard_text(statement))
    if not draft["summary"]:
        return {"ok": False, "error": "empty summary", "raw": raw}
    return {"ok": True, "draft": draft, "issues": issues, "raw": raw}


def _clinical_path_state(sess, store):
    state = store.read_json(int(sess["id"]), "clinical.json") or {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("schema", SCHEMA)
    state.setdefault("session_id", int(sess["id"]))
    state.setdefault("intake", {})
    state.setdefault("ai_draft", None)
    state.setdefault("review", None)
    state.setdefault("audit", [])
    return state


def _audit(state, action, actor, detail=""):
    state.setdefault("audit", []).append({
        "ts": _now(), "action": _short(action), "actor": _short(actor or "system"),
        "detail": _text(detail, 1000),
    })


def save_intake(sess, store, data: dict, actor="clinician"):
    state = _clinical_path_state(sess, store)
    if (state.get("review") or {}).get("status") == "signed":
        state["review"]["status"] = "stale"
        _audit(state, "clinician_review_stale", actor,
               "intake changed after signature; re-review and re-sign required")
    intake = _clean_intake(data)
    state["intake"] = intake
    _audit(state, "intake_saved", actor,
           "red flags: %d; fields: %s" % (len(intake.get("red_flags") or []),
                                          ", ".join(sorted(intake.keys()))))
    store.save_json(int(sess["id"]), "clinical.json", state)
    return state


def _draft_state(sess, store, summary):
    state = _clinical_path_state(sess, store)
    intake = _clean_intake(state.get("intake") or {})
    findings = build_findings(sess, summary, intake)
    evidence = select_evidence(findings)
    return state, intake, findings, evidence


def generate_draft(sess, store, summary: dict, completer=None):
    """Generate a guard-railed draft.  ``completer`` is injectable for tests."""
    state, intake, findings, evidence = _draft_state(sess, store, summary)
    if not evidence:
        state["ai_draft"] = {
            "status": "blocked", "generated_at": _now(), "model": None,
            "prompt_version": PROMPT_VERSION, "evidence_version": load_evidence()["version"],
            "issues": ["No evidence catalogue is installed; AI draft not generated."],
            "disclaimer": DISCLAIMER,
        }
        _audit(state, "ai_draft_blocked", "system", "missing evidence catalogue")
        store.save_json(int(sess["id"]), "clinical.json", state)
        return state
    prompt = build_prompt(findings, evidence)
    try:
        raw = (completer or (lambda p: ai_backend.describe_images(
            p, [], json_mode=True, num_predict=1600)))(prompt)
    except Exception as exc:
        state["ai_draft"] = {
            "status": "error", "generated_at": _now(), "model": ai_backend.active_model(),
            "prompt_version": PROMPT_VERSION, "evidence_version": load_evidence()["version"],
            "issues": [f"AI backend failed: {type(exc).__name__}: {exc}"],
            "disclaimer": DISCLAIMER,
        }
        _audit(state, "ai_draft_error", "system", type(exc).__name__)
        store.save_json(int(sess["id"]), "clinical.json", state)
        return state
    parsed = parse_draft(raw, evidence)
    input_hash = hashlib.sha256(json.dumps(findings, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if not parsed.get("ok"):
        state["ai_draft"] = {
            "status": "blocked", "generated_at": _now(), "model": ai_backend.active_model(),
            "prompt_version": PROMPT_VERSION, "evidence_version": load_evidence()["version"],
            "issues": [parsed.get("error") or "invalid draft"],
            "input_hash": input_hash, "raw_text": _text(raw, 12000),
            "disclaimer": DISCLAIMER,
        }
        _audit(state, "ai_draft_blocked", "system", parsed.get("error") or "invalid draft")
    elif parsed.get("issues"):
        state["ai_draft"] = {
            "status": "blocked", "generated_at": _now(), "model": ai_backend.active_model(),
            "prompt_version": PROMPT_VERSION, "evidence_version": load_evidence()["version"],
            "issues": [f"AI draft withheld by the clinical safety guard "
                       f"({len(parsed['issues'])} disallowed claim(s))."],
            "guard_count": len(parsed["issues"]), "input_hash": input_hash,
            "raw_text": _text(raw, 12000), "disclaimer": DISCLAIMER,
        }
        _audit(state, "ai_draft_blocked", "system", "; ".join(parsed["issues"][:3]))
    else:
        state["ai_draft"] = {
            "status": "draft", "generated_at": _now(), "model": ai_backend.active_model(),
            "prompt_version": PROMPT_VERSION, "evidence_version": load_evidence()["version"],
            "input_hash": input_hash, **parsed["draft"],
            "evidence_count": len(evidence),
        }
        _audit(state, "ai_draft_generated", "system",
               f"model={ai_backend.active_model()} evidence={len(evidence)}")
    store.save_json(int(sess["id"]), "clinical.json", state)
    return state


def save_review(sess, store, data: dict):
    """Save the clinician's own assessment/plan and signature.  AI cannot call this."""
    data = data if isinstance(data, dict) else {}
    required = {
        "clinician_name": _short(data.get("clinician_name")),
        "credentials": _short(data.get("credentials")),
        "assessment": _text(data.get("assessment")),
        "plan": _text(data.get("plan")),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("clinician name, credentials, assessment and plan are required")
    if data.get("attestation") is not True:
        raise ValueError("the clinician must attest that they reviewed the data and draft")
    state = _clinical_path_state(sess, store)
    previous = state.get("review")
    if previous:
        state.setdefault("review_history", []).append(previous)
    version = 1 + len(state.get("review_history") or [])
    review = {
        "status": "signed",
        "version": version,
        "clinician_name": required["clinician_name"],
        "credentials": required["credentials"],
        "license_id": _short(data.get("license_id")),
        "assessment": required["assessment"],
        "differential": _text(data.get("differential")),
        "plan": required["plan"],
        "patient_education": _text(data.get("patient_education")),
        "follow_up": _text(data.get("follow_up")),
        "outcomes": _text(data.get("outcomes")),
        "attestation": True,
        "signed_at": _now(),
        "signature": required["clinician_name"],
        "disclaimer": REVIEW_DISCLAIMER,
    }
    state["review"] = review
    _audit(state, "clinician_review_signed", required["clinician_name"],
           f"version={version}")
    store.save_json(int(sess["id"]), "clinical.json", state)
    return state


def public_state(sess, store, summary: dict) -> dict:
    """JSON-safe clinical workspace state."""
    state = _clinical_path_state(sess, store)
    intake = _clean_intake(state.get("intake") or {})
    findings = build_findings(sess, summary, intake)
    evidence = select_evidence(findings)
    draft = state.get("ai_draft")
    if isinstance(draft, dict) and draft.get("status") == "blocked":
        draft = {k: v for k, v in draft.items() if k != "raw_text"}
    return {
        "schema": SCHEMA,
        "session_id": int(sess["id"]),
        "patient": {"name": sess.get("patient_name"), "phone": sess.get("patient_phone")},
        "session": {k: sess.get(k) for k in
                    ("id", "started_ts", "ended_ts", "duration_s", "view_plane",
                     "discipline", "bike_type", "status")},
        "intake": intake,
        "findings": findings,
        "evidence": evidence,
        "evidence_version": load_evidence()["version"],
        "ai_draft": draft,
        "review": state.get("review"),
        "disclaimer": DISCLAIMER,
        "review_disclaimer": REVIEW_DISCLAIMER,
        "red_flags_all": list(RED_FLAGS),
        "model": ai_backend.active_model(),
    }
