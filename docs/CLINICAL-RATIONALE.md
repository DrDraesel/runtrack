# Clinical rationale workspace — RunTrack

**Status:** local research/decision-support workflow with clinician review. Not a medical device; not a diagnosis; not validated for clinical decision-making. The signed clinician note — not the AI text — is the clinical record.

## Workflow

1. Finish a running session and open **Clinical** (session row or report).
2. Complete the intake: history, pain/NRS, training load, red flags, examination, imaging/labs, medications, goals, and PROMs.
   - Do not put names, phone numbers or other identifiers in free-text fields; the AI receives only de-identified fields.
3. Review the **deterministic findings** table. It contains only values the app measured or `not assessed`; the view plane and evidence ID are shown.
4. If any red flag is documented, arrange clinical assessment before interpreting gait or progressing exercise. The AI prompt does not permit attributing red flags to a diagnosis.
5. Press **Generate AI draft**. The local model receives findings + evidence only. It must return strict JSON with evidence IDs.
6. Review the draft critically. A safety guard withholds it if it diagnoses, names a condition as present, prescribes treatment, predicts outcome or clears an athlete. It is always labelled **not signed**.
7. Write your own **assessment**, **differential considerations**, **plan**, education and follow-up. Tick the attestation and sign. The signed note is the record; the AI draft is supporting material.
8. Re-open the report for the signed note and serial PROM table.

## Evidence

`clinical_evidence.json` is a curated catalogue; every draft finding must cite one of its IDs.

- Running: `research/RUNNING-CLINICAL-EVIDENCE.md` — 30 verified sources (R01–R30), evidence table, red flags, PROMs, prohibited claims, clinician checklist.
- Cycling: `research/CYCLING-TT-FIT-EVIDENCE.md` — 41 sources, fit bands, measurement error, aero limits.

Grades are pragmatic A–D certainty aids, not GRADE. Applicability to this camera, population and clinic must be reviewed by the clinician. The reports deliberately state that 2-D markerless measurement is reliable for repeated observation but has poor-to-moderate criterion validity against 3-D for most angles; frontal-plane hip/knee estimates are especially low confidence. The red-flag framework is graded C overall (the ESC cardiac guideline is grade A for exertional cardiac symptoms), and absence of a flag does not exclude serious pathology. Foot/ankle findings are descriptive only; the evidence does not rank them as more or less reliable than other joints.

## AI boundary

The model is a drafting assistant only. It may:

- summarise measured values;
- describe associations using the supplied evidence IDs;
- list clinical questions and what cannot be concluded.

It may **not**:

- diagnose or name a disease as present;
- prescribe treatment, medication, surgery, injections or therapy;
- clear an athlete or give a return-to-run date;
- predict injury or prognosis;
- invent a measurement, normal range, citation or PROM score;
- infer pain, emotion, attention or mental state.

The prompt encodes these rules and `clinical.guard_text()` checks every generated field. Unsafe drafts are stored locally for audit but never rendered; only a generic withholding message is shown.

## Data and privacy

- Default: local Ollama at `127.0.0.1:11434`; no cloud.
- The AI payload contains de-identified age/sex/height/weight, training context, pain, exam/labs free text, PROMs and measured findings — **not** name, phone or patient ID.
- Do not enter identifiers in free text.
- If `AI_API_URL` is configured, the de-identified payload and draft go to that provider. Do not use a cloud endpoint for PHI without consent, a BAA and a privacy review.
- `clinical.json` is stored in the session folder with the intake, draft, audit trail and signed note. Back it up with the rest of `data/`.

## Limitations and regulatory note

- No clinical validation of this workflow, the model, the risk strata or the composite form score.
- No diagnosis, no injury prediction, no return-to-sport clearance, no treatment guarantee.
- PROMs are not running-specific; use the same instrument, version and recall period. LEFS MCID/MDC ≈9; NPRS ≈2; VISA MCIDs are not established; OSTRC has no established MCID.
- “Medical grade” is a regulatory claim, not a model adjective. Clinical use/marketing requires a defined intended use, QMS, clinical validation and the applicable FDA/CE pathway.
- A local test–retest study of this camera pipeline is the next measurement priority; the current evidence is borrowed from other 2-D systems.

## Tests

```bash
.venv\Scripts\python.exe -m unittest tests.test_clinical -v
.venv\Scripts\python.exe -m unittest discover -s tests
```

Clinical tests cover: identifier stripping, deterministic findings, no invented values, prompt hard rules, evidence-ID validation, output-guard blocking/negation, unsafe-output withholding, explicit backend failure, signature/attestation requirements and report separation of draft vs signed note.
