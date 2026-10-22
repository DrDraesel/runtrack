# Running gait and clinical rationale — evidence review for the RunTrack running module

**Scope.** What the running metrics in `config.py` (`BANDS`, `FLAGS`), the rule-based risk strata and the PT drills in `advisor.py` can honestly claim; what a clinician-reviewed rationale module may say; and what it must never say. Written 2026-10-21 for `C:\Users\GAMEPOWER\projects\runtrack`. Markdown only; no code changes.

**Status.** Evidence review prepared for clinician review and sign-off. It is not a diagnostic protocol, not a medical device, and not a substitute for history, examination, imaging or clinical judgement.

**Grades.** **A** = high certainty: consistent systematic review/meta-analysis of RCTs, a large definitive RCT, or a directly applicable guideline; low risk of bias; consistent across populations. **B** = moderate certainty: systematic review/meta-analysis with heterogeneity, or one good-quality cohort/RCT with limitations; direct population. **C** = low certainty: cross-sectional, small sample, modelling, indirect population, or a consistent mechanism without a clinical outcome. **D** = very low certainty: expert opinion, narrative/commentary, convention, or evidence absent/conflicting. The grade applies to the claim as worded, not to the design of the best study in isolation.

**Sources.** 30 numbered sources (R01–R30). All were checked against PubMed and Crossref on 2026-10-21; every DOI/PMID used resolves. No preprint, vendor page or unverified source is used. Verification notes are in §9; the full reference list is in §10.

---

## 0. Bottom line

1. **2-D markerless gait analysis is not valid enough for absolute clinical inference, even though 2-D video is often reliable.** Two-dimensional video has good-to-excellent inter- and intra-rater reliability for many running angles, but only poor-to-moderate criterion validity against 3-D for most parameters [R01]. In runners with patellofemoral pain, a smartphone markerless system was invalid for peak hip adduction (ICC 0.06) and knee flexion (ICC 0.42) [R02].
2. **Frontal-plane hip adduction is a weak point, and the frontal metrics carry the app's heaviest weights.** `pelvic_drop_deg` and `knee_valgus_deg` each add 2 risk points, yet a running-specific smartphone markerless study found near-zero agreement for peak hip adduction (ICC 0.06) [R02]. The frontal-plane metrics therefore carry the app's highest weight despite weak or untested supporting agreement.
3. **Injury associations are moderate at best and usually limited.** The most consistent signal is greater peak hip adduction in female runners with patellofemoral pain and iliotibial band syndrome [R04–R07]; no single kinematic variable is a proven cause, and prospective evidence is sparse [R04].
4. **Cadence is the best-supported modification.** Increasing step rate by 5–10% reduces knee energy absorption, braking impulse, step length and peak knee flexion [R13]; 110% of preferred step rate reduced peak patellofemoral joint force by 14% [R14]; an in-field +7.5% cue reduced vertical load rates by about 18% and eccentric knee work by about 27% [R16].
5. **One RCT supports injury reduction, not treatment.** A 2-week feedback-based gait-retraining programme in 320 novice runners reduced 12-month injury from 38% to 16% (HR 0.38; 62% lower) [R15]. It does not show that the same programme treats an existing injury.
6. **Changing footstrike is not supported as injury prevention.** A mid/forefoot strike does not improve running economy, does not eliminate impact, and has not been shown to reduce running injury [R18].
7. **Red flags are referral triggers, not rule-outs.** High-quality diagnostic-accuracy evidence for most red flags is lacking [R19]; cauda equina has no single consensus feature [R20]; compartment-pressure thresholds vary [R21]; cardiac symptoms need medical evaluation [R22].
8. **PROMs make the note clinically useful.** LEFS (0–80; MCID/MDC 9) [R24], NPRS (0–10; MCID about 2) [R26], VISA-A and VISA-P (0–100) [R27,R30], and OSTRC-O (0–100 weekly) [R25] cover function, pain, region-specific tendon status and overuse burden.
9. **The module must never diagnose, predict injury, or prescribe a return timeline.** See §7 for the explicit prohibited claims.

---

## 1. What 2-D markerless gait analysis can and cannot support clinically

### 1.1 What the measurement literature shows

| Question | Finding | Source |
|---|---|---|
| Is 2-D video reliable? | Good-to-excellent inter-rater and intra-rater reliability for most running angles; inter-session reliability was good or excellent in the small number of studies that tested it. | R01 |
| Is 2-D video valid against 3-D? | Poor-to-moderate for most parameters; only a minority reached good or excellent agreement. 17 studies were included in the review. | R01 |
| Does a smartphone markerless app agree with 3-D in PFP? | No. In runners with PFP, the Hudl Technique app showed peak hip adduction ICC 0.06 (95% CI −0.35 to 0.47) and peak knee flexion ICC 0.42 (−0.10 to 0.75). The authors concluded that this 2-D approach "cannot be advocated" at present. | R02 |
| Do markerless kinetics match marker-based kinetics? | No. Joint moments and powers differed systematically, with larger differences at faster running speeds; the authors caution against clinical application without validation. | R03 |

**Practical conclusion.** Two-dimensional video is, in general, a *reliable observation tool* but a *weak absolute measurement tool*. Markerless systems must demonstrate their own reliability and validity; one smartphone system did not for frontal-plane hip adduction [R02]. Reliability means repeated measurements under the same conditions are consistent; validity means the number is close to the true 3-D value. RunTrack can use the first property only where it has been demonstrated for its own pipeline, not the second.

### 1.2 What it can support clinically

- **Standardised observation and documentation.** Same camera, same plane, same distance, same clothing, same treadmill/overground condition; record these in the session metadata.
- **Within-runner change over time.** Compare baseline and follow-up only when the setup is unchanged, and interpret the difference against the system's own test–retest error, not against a population norm.
- **Cadence and basic spatiotemporal events.** Cadence from foot-strike events is, in this review's judgement, the most defensible running metric the app produces, provided event-detection quality is reported (clean strikes per side, tracking coverage, gaps).
- **Hypothesis generation for the physical examination.** "Contact position is far ahead of the pelvis; assess hip and calf capacity" is a clinical question, not a finding.
- **Communication.** A shared visual frame for the runner and the clinician.
- **Adjunct documentation.** A record of what was observed, when, with what setup, and with what uncertainty.

### 1.3 What it cannot support

- **Absolute 3-D-equivalent angles.** Do not print a 2-D angle as if it were a 3-D laboratory value, and never compare across cameras, apps or clinics [R01,R02].
- **Frontal-plane hip adduction as a valid measure.** The running-specific smartphone study reviewed here found hip-adduction agreement near zero in this plane [R02]. Knee valgus was not separately tested in that study; treating a 2-D valgus angle as a validated measure is an extrapolation.
- **Joint-by-joint reliability rankings.** The sources reviewed do not establish that the ankle (or any other joint) is the least reliable markerless segment; do not rank joints unless a source does. R01 reports poor-to-moderate validity across parameters, and R02 tested hip adduction and knee flexion only.
- **Ground reaction force, vertical loading rate, joint moments, joint contact forces or "impact load".** The app estimates kinematics; it does not measure kinetics. Markerless-derived kinetics differ from marker-based estimates and the difference grows with speed [R03].
- **Muscle activity, muscle strength, tendon load, bone stress or tissue injury.** None of these are visible in a video.
- **Diagnosis** of patellofemoral pain, ITB syndrome, medial tibial stress syndrome, Achilles tendinopathy, plantar fasciopathy or hamstring strain. Diagnosis requires history, examination and, where indicated, imaging.
- **Injury-risk prediction or prognosis.** Prospective biomechanical evidence is sparse and inconsistent [R04]; the app's risk strata are heuristic flag counts, not a prediction model.
- **Treatment effect or return-to-run clearance.** Gait-retraining evidence is short-term and population-specific [R15–R17]; return-to-run decisions are clinical and shared [R28].
- **A validated composite "form score".** The current weights are heuristic; no study validates this composite or its cut-points.

### 1.4 Clinical boundary: questions the app may and may not answer

| Clinical question | Can the app answer it? | What it may output | What it must not output |
|---|---|---|---|
| Is cadence low for this runner? | Yes, with measurement caveat | "Cadence was X spm over N clean strides; the app's reference band is 160–190 spm." | "Low cadence is causing your injury." |
| Is the foot landing far ahead of the pelvis? | Descriptively, low confidence | "Contact position is X leg-lengths ahead of the pelvis; this is a descriptive observation." | "You are overstriding and damaging your knee." |
| Is frontal-plane control poor? | Only as a low-confidence observation | "A contralateral pelvic drop of X° was observed; 2-D frontal measurement is low confidence." | "You have gluteus medius weakness." |
| Is this patellofemoral pain? | No | "The findings are compatible with several causes; clinical assessment is required." | "You have patellofemoral pain." |
| Will this runner get injured? | No | "No validated prediction is possible from these metrics." | "Your injury risk is 62%." |
| Has gait changed since baseline? | Yes, within measurement error | "Cadence increased by X spm; the change exceeds/does not exceed the system's repeatability." | "Your gait is now safe." |
| Is the training load too high? | No | "Training load was not measured by the app; enter recent volume/intensity for clinical interpretation." | "Your mileage is safe." |
| Can the runner return to running? | No | "Return-to-run decisions require clinical assessment and shared decision-making." | "You can return in 4 weeks." |

### 1.5 Metric-by-metric claim audit

The table audits every running metric currently surfaced by `metrics.py`/`advisor.py`. "Evidence status" is the highest grade the evidence supports for the claim **as currently worded** in the app.

| Metric (current flag) | Plane | Current claim in the app | Evidence status | What may be said | What must not be said |
|---|---|---|---|---|---|
| `cadence_spm` (<160) | — | "longer stride time, higher impact per step and more vertical braking" | **C**: cadence manipulation changes loading [R13,R14,R16], but 160 spm is not a validated threshold | "Cadence is below the app's reference band; a 5% increase may reduce braking impulse and knee load; trial and reassess." | "Low cadence is causing injury"; "180 spm is optimal." |
| `touchdown_shin_deg` (>8) | Sagittal | "shin lands inclined forward, increases braking force through knee and hip" | **D**: shin angle at contact is a descriptive surrogate; no validated 8° threshold; force is not measured | "Shin angle at contact is forward of the app's reference; this is a braking-posture observation." | "Braking force is high"; "this is damaging the knee." |
| `overstride_leg_frac` (>0.15) | Sagittal | "longer braking phase and higher impact load" | **D**: step length and braking impulse change with cadence [R13], but the 0.15 threshold is not validated | "Contact is relatively far ahead of the pelvis; cadence/step-length work may shorten it." | "Overstriding is causing your injury." |
| `knee_flexion_ms_deg` (<35 or >45) | Sagittal | "stiffer absorption, load moved to the joint"; "deep collapse under load" | **C**: knee flexion predicts patellofemoral load in modelling [R14]; 35°/45° cut-points are heuristic | "Limited mid-stance knee flexion has been associated with higher modelled patellofemoral load; assess clinically." | "Your joint is taking the load instead of your muscles." |
| `pelvic_drop_deg` (>5) | Frontal | "potential gluteus-medius weakness" | **C**: contralateral pelvic drop is associated with PFP in some cohorts [R05]; 2-D frontal validity for hip adduction is poor [R02], and pelvic drop was not separately validated | "Contralateral pelvic drop was observed; it has been associated with PFP in some cohorts; clinical hip assessment is needed." | "This proves gluteus medius weakness." |
| `knee_valgus_deg` (>12) | Frontal | "primary patellofemoral / IT-band risk marker" | **D**: no source in this review validates a 2-D knee valgus angle; related 3-D findings concern hip adduction and knee internal rotation [R05,R09], not frontal knee valgus; the 12° threshold is not validated | "Frontal knee motion was observed; the 2-D estimate is low confidence and should be interpreted with the examination." | "Valgus is a primary risk marker"; "your valgus is 12°." |
| `vertical_osc_cm` (>10) | Sagittal | "extra vertical work is not converted into forward motion" | **D**: descriptive; no prospective evidence supports a 10 cm injury threshold [R04] | "Vertical oscillation is above the app's reference; no validated injury threshold exists." | "Excess bounce causes injury." |
| `symmetry_pct` (<90) | — | "persistent side-to-side difference is worth investigating" | **D**: no validated asymmetry cut-off; other spatiotemporal factors are under-studied [R04] | "A left–right difference was observed; repeat the measurement and assess clinically." | "Asymmetry below 90% is abnormal." |
| `trunk_lean_deg` (4–12) | Sagittal | reference band | **D**: no validated sagittal trunk-lean target; lateral trunk kinematics has a weak HSI signal but is a different measure [R10] | "Trunk lean is descriptive; no validated target exists." | "Trunk lean is causing injury." |
| `power_per_kg_w` (1.5–4.5) | Sagittal | "vertical power per kg" | **D**: derived estimate from video, not measured external power | "Estimated vertical power; not a measured training load." | "Running power" or "your load is 3 W/kg." |
| `arm_swing_sym_pct` (90–100) | Frontal | symmetry band | **D**: descriptive; no injury evidence | "Arm swing appears asymmetric; no validated injury threshold." | "Arm asymmetry causes injury." |

**Two of the highest-weighted metrics rest on weak or untested measurement evidence.** `pelvic_drop_deg` and `knee_valgus_deg` each add 2 points, yet the running-specific smartphone markerless study found near-zero agreement for frontal-plane hip adduction [R02]; pelvic drop and knee valgus were not separately validated. Until RunTrack has local repeatability data for those two metrics, they should be presented as "low-confidence observations" and should not outrank cadence, pain history, training load or examination findings in the clinical summary.

---

## 2. Evidence-graded associations between running kinematics/load and common injuries

**Read this first.** An association is not a cause. Most running-injury biomechanics is cross-sectional: injured runners differ from controls, but the difference may be a consequence of pain or adaptation rather than a cause. A systematic review of prospective studies found only limited evidence for greater peak hip adduction in female runners who developed PFP or ITBS, and inconsistent evidence for vertical loading rate [R04]. The grades below reflect certainty for the claim as worded, not the design of the best study. Where the cited reviews use their own labels ("strong", "moderate", "limited" evidence), those are the review authors' ratings, not this report's A–D grades.

### 2.1 Summary table

| Condition | Most consistent findings | Grade | What it supports | What it does not support |
|---|---|---|---|---|
| Patellofemoral pain (PFP) | Greater peak hip adduction, hip internal rotation, contralateral pelvic drop and reduced peak hip flexion in symptomatic runners; prospective hip adduction in female runners [R04–R06] | **B** | A modest, multi-factorial association; a rationale for assessing hip control and trialling retraining/strengthening | A single cause; a diagnostic test; a prediction rule |
| Iliotibial band syndrome (ITBS) | Higher peak knee internal rotation, lower rearfoot eversion and lower knee flexion at heel strike, greater hip adduction; female current ITBS: lower hip abductor strength and smaller peak hip internal rotation [R07,R09] | **B** | Sex- and status-specific associations; a rationale for hip strength and running-pattern assessment | That every runner with these features has ITBS, or that ITBS is caused by one variable |
| Medial tibial stress syndrome (MTSS) | Higher BMI (MD 0.79 kg/m²), navicular drop (1.19 mm), ankle plantarflexion ROM (5.94°) and hip external rotation ROM (3.95°) [R08] | **B** for risk factors; **C** for gait | A multi-factorial risk profile; examination of foot posture and ROM | A gait-only explanation; a validated 2-D marker |
| Achilles tendinopathy | No consistent kinematic risk factor; rearfoot eversion was not significant in Achilles tendinopathy; only 9 studies [R09] | **C** | That 2-D gait does not identify an Achilles risk pattern; load and calf capacity are clinical issues | That ankle or foot kinematics explain Achilles pain |
| Plantar fasciopathy | Higher plantarflexion ROM (7.04°), BMI (2.13 kg/m²) and body mass (4.52 kg); very limited evidence for higher ankle eversion [R09,R11] | **B** for risk factors; **C** for gait | A multi-factorial risk profile; examination of ankle ROM and body mass | That ankle eversion is a proven cause; that gait retraining prevents it |
| Hamstring strain | No consistent association; lateral trunk kinematics and horizontal propulsive forces were the only exceptions; mostly high-speed running/sprinting studies [R10] | **C** | That distance-running gait should not be used to predict hamstring strain | A hamstring "gait pattern"; a prevention rule |

### 2.2 Condition notes

**Patellofemoral pain.** A synthesis of 28 studies found moderate evidence of an association with increased peak hip adduction, hip internal rotation, contralateral pelvic drop and reduced peak hip flexion, plus very limited prospective evidence that increased hip adduction is a risk factor in female runners [R05]. In a prospective cohort of 400 female runners, the 15 who developed PFP had greater hip adduction at baseline (p = 0.007), but no significant difference in hip internal rotation or rearfoot eversion [R06]. The honest clinical statement is: "hip adduction is part of a multi-factorial picture in female runners; it is not a diagnosis and not a single cause." The same synthesis found limited evidence that running retraining and proximal strengthening both improve pain and function, with retraining reducing peak hip adduction [R05].

**Iliotibial band syndrome.** A 2023 meta-analysis of 17 articles found that female runners with current ITBS had lower isometric hip abductor strength and smaller peak hip internal rotation than controls, and concluded that risk factors differ by sex and injury status [R07]. A separate kinematic synthesis found strong evidence for higher peak knee internal rotation in ITBS, moderate evidence for lower peak rearfoot eversion and lower knee flexion at heel strike, and greater peak hip adduction [R09]. Do not apply the female current-ITBS findings to male runners or to recovered runners; the cohorts are not interchangeable.

**Medial tibial stress syndrome.** A systematic review and meta-analysis of 21 studies identified higher BMI (weighted MD 0.79, 95% CI 0.38–1.20), navicular drop (1.19 mm, 0.54–1.84), ankle plantarflexion ROM (5.94°, 3.65–8.24) and hip external rotation ROM (3.95°, 1.78–6.13) as associated factors [R08]. These are risk-factor associations, not gait features the app measures; navicular drop and ROM require physical examination.

**Achilles tendinopathy.** Only 9 Achilles studies contributed to the kinematic review, and rearfoot eversion was not a significant factor in Achilles tendinopathy [R09]. The evidence does not support a 2-D "Achilles gait pattern". Load management and calf capacity are clinical judgements; the app cannot measure tendon load.

**Plantar fasciopathy.** A systematic review and meta-analysis of 16 studies found higher plantarflexion ROM (7.04°, 95% CI 5.88–8.19), BMI (2.13 kg/m², 1.40–2.86) and body mass (4.52 kg, 0.55–8.49) in physically active people with plantar fasciitis [R11]. The kinematic review included only 2 plantar-fasciopathy studies and found very limited evidence for higher ankle eversion [R09]. Do not present ankle eversion as a proven cause, and do not present gait retraining as prevention.

**Hamstring strain.** Twelve studies were included; most found no difference or contradictory differences between runners with and without a history of hamstring strain, with the exceptions of lateral trunk kinematics and horizontal propulsive forces. The authors noted small samples and risk of bias, and concluded that definitive recommendations for biomechanical interventions cannot be made [R10]. The literature is dominated by high-speed running and sprinting, not recreational distance running; applicability to a 10 km runner is limited.

### 2.3 Load and training errors

The app does not measure training load. The running-specific synthesis of volume, duration, frequency and intensity found conflicting results and a mean quality score of 44%; it could not identify which training errors cause injury [R12]. A universal "safe mileage" or "10% rule" is not supported by that review. Load should be entered as clinical history (recent volume, long-run distance, intensity, surface, footwear, sleep, life stress) and interpreted by the clinician. For bone stress injury, load is part of a continuum from stress reaction to fracture, and management is modified loading plus progressive return, not a fixed mileage cap [R23].

### 2.4 What the app may say about associations

**May say:**
- "Greater peak hip adduction has been associated with PFP in female runners; this is not diagnostic and should be assessed clinically [R05,R06]."
- "MTSS has been associated with higher BMI, navicular drop and ankle plantarflexion ROM; these require examination [R08]."
- "The evidence for a kinematic Achilles or hamstring risk pattern is weak; do not infer one from this video [R09,R10]."

**Must not say:**
- "This finding explains your pain."
- "Your gait caused your injury."
- "You are at high risk of injury."
- "Your hip abductor strength is weak" (strength is not measured by video).

---

## 3. Gait retraining, cadence and overstride: evidence and effect sizes

### 3.1 Cadence

**Step-rate manipulation changes load.** In 45 healthy recreational runners, increasing step rate by 5% and 10% above preferred reduced mechanical energy absorption at the knee; the 10% condition also reduced hip energy absorption. Increasing step rate reduced step length, centre-of-mass vertical excursion, braking impulse and peak knee flexion. At +10%, peak hip adduction and hip adduction/internal rotation moments also decreased. Reducing step rate by 10% increased energy absorption at all joints [R13].

**Patellofemoral load falls, but load is redistributed.** In 30 healthy adults, running at 110% of preferred step rate reduced peak patellofemoral joint force by 14%; peak stance-phase knee flexion was the most important predictor of the reduction. Hip, knee and ankle extensor forces and hip abductor forces fell in midstance, but rectus femoris and hamstring loads increased during early and late swing [R14]. This is not a free lunch: higher step rates move load away from the patellofemoral joint and knee extensor mechanism, but not uniformly away from all tissues.

**In-field retraining is feasible and short-term effects persist.** Thirty at-risk runners were cued to increase preferred step rate by 7.5% over eight in-field sessions; actual step rate rose by 8.6%. Instantaneous vertical load rate fell 17.9%, average vertical load rate fell 18.9%, peak hip adduction fell 2.9°, and eccentric knee joint work fell 26.9% per stance and 21.1% per kilometre. The changes were maintained at 30 days [R16].

**One RCT shows injury reduction.** In 320 novice runners, a 2-week gait-retraining programme using real-time visual feedback reduced vertical loading rates (Cohen's d > 0.99) and reduced 12-month running-related injury from 38% in controls to 16% in the retrained group (HR 0.38, 95% CI 0.25–0.59; 62% lower) [R15]. This is the strongest clinical-outcome evidence in the review, but it is prevention in novice runners, the intervention was a feedback-based gait-retraining programme rather than a cadence prescription, and it does not establish a treatment effect in injured runners.

### 3.2 Overstride

There is no RCT of "overstride correction" as a discrete intervention. The evidence is for step-length/cadence manipulation: increasing cadence shortens step length and reduces braking impulse and knee energy absorption [R13]. The app's `overstride_leg_frac` is a descriptive estimate; the 0.15 threshold has no published validation. Say "contact is relatively far ahead of the pelvis", not "you are overstriding and this is causing injury".

### 3.3 Footstrike

Changing to a mid- or forefoot strike does not improve running economy, does not eliminate the impact at foot–ground contact, and has not been shown to reduce running injury [R18]. Footstrike should not be a default retraining target. If a clinician chooses it for a specific reason, document the rationale and the trade-offs.

### 3.4 Clinical retraining evidence

Barton et al. synthesised published evidence with 16 international experts: clinical-outcome evidence is limited, while evidence for immediate biomechanical effects is substantial (46 studies). Experts recommended considering retraining for ITBS, plantar fasciopathy, Achilles/patellar/proximal hamstring/gluteal tendinopathy, calf pain and MTSS, tailored to the individual [R17]. This is expert-supported practice, not a protocol with proven disease-modifying effect.

### 3.5 Effect-size table

| Intervention | Study | Population | Main effect | Caveats |
|---|---|---|---|---|
| Step rate +5% | Heiderscheit 2011 [R13] | 45 healthy recreational runners | Reduced knee energy absorption; shorter step length; lower centre-of-mass excursion; lower braking impulse; lower peak knee flexion | Acute treadmill study; no clinical outcome; individual response varies |
| Step rate +10% | Heiderscheit 2011 [R13] | 45 healthy recreational runners | Additional reduction in hip energy absorption; lower peak hip adduction and hip adduction/internal rotation moments | Same; higher step rates may not suit every runner |
| Step rate 110% of preferred | Lenhart 2014 [R14] | 30 healthy adults | 14% lower peak patellofemoral joint force; knee flexion the strongest predictor | Musculoskeletal model; rectus femoris and hamstring swing loads increased |
| In-field cue +7.5% | Willy 2016 [R16] | 30 at-risk runners | Actual +8.6%; vertical load rate −17.9% to −18.9%; peak hip adduction −2.9°; eccentric knee work −26.9% | Short-term (30 days); surrogate outcomes; selected at-risk sample |
| 2-week feedback-based retraining | Chan 2018 [R15] | 320 novice runners | Injury 16% vs 38%; HR 0.38 (0.25–0.59); 62% lower injury risk | Single RCT; novice runners; prevention, not treatment; supervised programme |
| Footstrike change | Hamill & Gruber 2017 [R18] | Review | No improvement in economy; impact not eliminated; no demonstrated injury reduction | Narrative critique; not a trial |
| Running retraining (mixed) | Barton 2016 [R17] | Lower-limb running injuries | Limited clinical-outcome evidence; substantial immediate biomechanical evidence | Mixed-methods; expert opinion; condition-specific effects uncertain |

### 3.6 How RunTrack should use this

- Offer cadence as a **trial**, not a prescription: a small increment (for example +5%) for a few weeks, with a metronome or external cue and the same camera/setup for reassessment. The increment and duration are clinical suggestions (grade D), not trial-defined protocol.
- Avoid a universal 180 spm target; baseline is the runner's preferred cadence, and the trials used preferred ±5–10% [R13,R14].
- Do not promise pain or injury reduction. The RCT is prevention in novice runners [R15]; the retraining literature in injured runners is limited [R17].
- Note load redistribution: higher step rates can increase swing-phase hamstring and rectus femoris loads [R14]; monitor symptoms and do not apply the same target to every runner.
- Footstrike change is not a default [R18].
- Retraining is an adjunct to strength, load management and clinical care, not a replacement [R17].

---

## 4. Red flags requiring medical referral

### 4.1 Principles

Red flags are **suspicion triggers**, not diagnostic tests. The international framework is explicit that high-quality evidence for the diagnostic accuracy of most red flags is lacking, and the framework is intended to support clinical reasoning rather than to rule disease in or out [R19]. The app must never say "no red flags, so this is safe"; it may say "no red flags were recorded; this does not exclude serious pathology."

### 4.2 Referral table

| Concern | Features that should trigger referral | Suggested urgency | Source | Grade |
|---|---|---|---|---|
| Bone stress injury / fracture | Focal bone pain or tenderness, pain at rest or at night, recent increase in load, prior stress fracture, low BMI or menstrual dysfunction, inability to bear weight, palpable callus or deformity | Urgent imaging/referral; emergency if unable to bear weight or gross deformity | R23 | C |
| Infection | Fever, malaise, hot/swollen joint, recent corticosteroid injection or surgery, immunosuppression, intravenous drug use, spreading erythema | Emergency/urgent medical assessment | R19 | C |
| Tumour | Unexplained weight loss, unremitting night pain, history of cancer, palpable mass, pain that is not load-related, pathological fracture | Urgent/emergency medical assessment | R19 | C |
| Cauda equina syndrome | Saddle anaesthesia, bladder or bowel dysfunction (retention/incontinence), bilateral sciatica, progressive motor weakness | Emergency same-day assessment and MRI | R20,R19 | C |
| Chronic exertional compartment syndrome | Exertional leg pain/tightness, reproducible at a certain distance or pace, relief with rest, paraesthesia or weakness during exercise | Urgent referral for compartment-pressure testing | R21 | C |
| Cardiac / exertional symptoms | Chest pain or pressure, exertional syncope or near-syncope, disproportionate dyspnoea, palpitations with dizziness, family history of sudden cardiac death at a young age | Emergency/urgent medical or cardiology assessment before continued strenuous exercise | R22 | A |
| Major trauma | Deformity, inability to bear weight, neurovascular compromise, open injury | Emergency department | R19,R23 (fracture red flags); basic first-aid principle | D |

**Scope note.** The red-flag framework in R19 is specifically about serious spinal pathology. Its infection and tumour features are used here as a clinical-reasoning template for peripheral presentations, which is an extrapolation (grade C/D); the cauda equina and fracture rows are more directly supported by R20 and R23. The cardiac symptom list is a clinical safety checklist, not a validated screening instrument; R22 supports the need for medical evaluation, not the accuracy of the list.

**Cauda equina nuance.** A review of 105 articles found no single feature achieved unanimity or consensus; bladder disturbance (74% of articles) and sensory disturbance (66%) were the most common majority views [R20]. The absence of one feature does not exclude the syndrome; any suspicion needs urgent assessment.

**Compartment syndrome nuance.** Diagnosis relies on intra-compartmental pressure measurement, but thresholds vary: 42% of surveyed UK surgeons used a maximal exercise pressure >35 mmHg for anterior CECS, while 35% used Pedowitz's modified criteria [R21]. The app cannot diagnose CECS; it can only flag the exertional pattern and refer.

**Cardiac nuance.** The ESC sports cardiology guideline addresses cardiovascular evaluation and exercise prescription in active individuals [R22]. It is not a running-injury guideline, and it is not a screening tool the app can apply; it supports the rule that exertional cardiac symptoms need medical evaluation.

### 4.3 What the app should say when a red flag is present

**May say:**
- "This symptom pattern needs medical assessment before gait interpretation or exercise progression."
- "I am not able to exclude a serious cause from video alone."
- "Please arrange urgent review; the gait report should not be used to reassure."

**Must not say:**
- "Probably nothing serious."
- "Your gait looks fine, so it is not a fracture."
- "You can keep running until the pain settles."
- "No red flags were found, so you are safe."

---

## 5. Patient-reported outcome measures suitable for running

### 5.1 Why PROMs

Gait metrics describe movement; they do not describe pain, function, participation or the runner's own priorities. A clinician-reviewed note should pair the video observation with at least one validated PROM, recorded at baseline and follow-up. The instruments below are the ones with direct relevance to running and lower-limb tendinopathy. They measure different constructs; do not average them into a single composite.

### 5.2 Instrument table

| Instrument | Construct | Items and scoring | Meaning | Change threshold | Use in running | Caveats | Source |
|---|---|---|---|---|---|---|---|
| **LEFS** (Lower Extremity Functional Scale) | Global lower-extremity function | 20 items, each 0–4; total 0–80 | Higher = better function | MCID 9 points; MDC 9 points (90% CI); typical error ±5.3 points | Baseline and follow-up for any lower-limb running injury | Not running-specific; 107 patients in the original study | R24 |
| **OSTRC-O** (Oslo Sports Trauma Research Centre Overuse Injury Questionnaire) | Overuse problem burden | 4 items per body area, each scored 0/8/17/25; total 0–100 | Higher = worse; a "substantial" problem causes moderate/severe reduction in performance or participation, or time loss | No established MCID; interpret change with the substantial-problem definition | Weekly monitoring of knee, lower leg, Achilles or foot problems | Designed for athletes across sports; 313 athletes in validation; not running-specific | R25 |
| **NPRS** (Numeric Pain Rating Scale) | Pain intensity | 0–10, single item | Higher = worse | MCID about 2 points; MDC 2 points; SEM 1.02 | Baseline and follow-up; specify recall period (e.g., 24 hours, 7 days) | Not running-specific; low back pain validation; recall period changes the score | R26 |
| **VISA-A** (Victorian Institute of Sport Assessment–Achilles) | Achilles symptoms and function | 8 items; total 0–100 | Higher = better; 100 = perfect score | No MCID established in the original paper; do not label a change meaningful without a validated threshold | Achilles tendinopathy | Not diagnostic; original validation in 45 non-surgical and 14 pre-surgical patients plus 87 controls | R27 |
| **VISA-P** (Victorian Institute of Sport Assessment–Patella) | Patellar tendon symptoms and function | 8 items; total 0–100 | Higher = better; 100 = asymptomatic | No MCID established in the original paper | Patellar tendinopathy | Not diagnostic; original development paper; use the validated language version | R30 |

### 5.3 Interpretation rules

- **Use the MCID, not the raw change.** A 4-point LEFS change is within measurement error; the MCID/MDC is 9 points [R24]. A 1-point NPRS change is within measurement error; the MCID is about 2 points [R26].
- **Do not invent a VISA MCID.** The original VISA-A and VISA-P papers establish reliability and construct validity, not a minimal important change [R27,R30]. Report the score and the direction; let the clinician judge meaningfulness.
- **Do not composite.** LEFS, NPRS, VISA and OSTRC measure function, pain, tendon-specific status and overuse burden respectively. A single "patient score" would be uninterpretable.
- **Record the recall period and the version/language.** PROM scores are not comparable across different recall periods or translations.
- **Do not infer PROM scores from gait.** A good-looking gait does not mean low pain, and a poor-looking gait does not mean high pain.

### 5.4 Suggested tracking schedule

Baseline, then at a clinically meaningful review point (commonly 2–4 weeks or after a treatment block), then at discharge or return-to-run. This is a clinical-practice suggestion, grade D; there is no running-specific trial establishing the optimal interval.

---

## 6. Clinical assessment, outcome tracking and return-to-run principles

### 6.1 Assessment sequence

1. **History.** Pain location and behaviour, onset, training change, prior injury, bone-health history, medications, footwear, surface, goals. Ask directly about red flags [R19].
2. **Red-flag screen.** If any feature in §4.2 is present, arrange medical assessment before interpreting gait or progressing exercise [R19–R23].
3. **Physical examination.** Palpation, range of motion, strength, single-leg control, functional testing, neurovascular examination as indicated. The video cannot replace this.
4. **Video gait observation.** Same camera, plane, distance and clothing; record the conditions; treat the output as a low-to-moderate-confidence observation [R01,R02].
5. **Integration.** Combine history, examination, PROMs, training load and video. The clinician writes the impression and plan.
6. **Shared decision.** Discuss uncertainty, the meaning of the findings, and the options.

### 6.2 What the video adds — and what it does not

The video adds a standardised visual record and a way to track change within a runner. It does not add diagnostic certainty, tissue-specific load, or prognosis [R01–R03]. A useful phrasing for the note: "Video observation is an adjunct to the examination; it does not establish a diagnosis."

### 6.3 Outcome tracking

- Record at least one PROM at baseline and follow-up: LEFS for function [R24], NPRS for pain [R26], VISA-A or VISA-P for the relevant tendon [R27,R30], OSTRC-O for weekly overuse burden [R25].
- Record the training load and the runner's main goal, not just the gait metrics.
- Re-assess gait only when the setup is unchanged; report the measurement confidence and the system's repeatability.
- Treat changes smaller than the MCID/MDC as uncertain, not as improvement or deterioration [R24,R26].
- Do not use the app's "form score" as an outcome measure; it is not validated.

### 6.4 Return-to-run principles

- **Return to sport is a continuum, not a single decision.** The Bern consensus describes shared decision-making, risk assessment and a biopsychosocial approach; evidence for specific return-to-sport criteria is scarce [R28].
- **Pain-guided loading can be appropriate in tendinopathy.** In a randomised trial of Achilles tendinopathy, continuing tendon-loading activity using a pain-monitoring model (pain allowed up to 5/10 during activity, settling by the next morning) produced no worse outcomes than active rest; both groups improved, with VISA-A rising from 57 to 85 in the exercise group and from 57 to 91 in the active-rest group at 12 months [R29]. This is specific to Achilles tendinopathy and a structured exercise programme; do not generalise it to fractures or acute muscle injuries.
- **Bone stress injury needs modified loading and progressive return.** Management addresses the load–bone balance, training programme, impact forces, strength and alignment; there is no universal week-by-week timeline [R23].
- **The app must not supply a return date.** It can record the clinician's plan and the criteria the clinician chooses; it cannot decide when running is safe.

### 6.5 Clinician sign-off template

```
CLINICIAN REVIEW AND SIGN-OFF

Reviewed by: ______________________________  Role: __________________
Date: __________________

I reviewed the video, history, examination, PROMs and the automated observations.
The automated text is decision support only and is not a diagnosis.

My clinical impression is:
______________________________________________________________________
______________________________________________________________________

Red flags screened:   [ ] Yes   [ ] No
If yes, action taken:
______________________________________________________________________

Automated text changed or rejected:
______________________________________________________________________

Plan / return-to-run decision:
______________________________________________________________________

Signature: ______________________________
```

### 6.6 Module contract for the clinician-reviewed rationale

**Inputs.** Metric values with plane and confidence; event-detection quality; clinician-entered history, examination and PROMs; training-load history; the evidence-catalog version.

**Outputs.** A draft set of "observation cards", each with: the observation, the plane, the measurement confidence, the evidence grade, the associated condition(s), what the observation does **not** mean, one or two clinical questions to consider, and the source ID(s).

**Invariants.**
- No diagnosis, no ICD code, no "impression" field populated by the model.
- No probability, percentage risk, prognosis or return date.
- No causal language ("causes", "leads to", "will injure").
- Every claim traceable to a catalog ID and a source in this review.
- Missing or low-confidence data are labelled "not assessed", never inferred.
- Red flags stop gait interpretation and route to medical review.

**Review.** The clinician edits, accepts or rejects each card, writes the impression and plan, and signs. The signed note is the clinical record; the model output is a draft.

**Audit.** Store the catalog version, prompt version, model version, timestamp and the clinician's edits. This makes the note reproducible and reviewable.

### 6.7 Required language changes in the existing report

These are documentation/UI wording changes, not code changes, but they determine whether the note overclaims.

| Current label | Problem | Recommended label |
|---|---|---|
| "Injury-risk strata: Low / Moderate / Elevated / High" | Sounds like a prognosis; the point sum is heuristic and unvalidated | "Observation flags: N (not an injury-risk prediction)" |
| "PT drill prescriptions" | Sounds like individualised treatment; the drills are generic examples | "Suggested movement practice for clinician consideration" |
| "Form score (0–100)" | Sounds like a validated composite; the weights are heuristic | "Form composite (experimental; not validated)" |
| "High braking force" | Force is not measured | "Forward shin angle at contact (descriptive)" |
| "Gluteus-medius weakness" | Strength is not measured | "Contralateral pelvic drop (low-confidence observation)" |
| "Primary patellofemoral / IT-band risk marker" | Overstates a 2-D frontal estimate | "Frontal knee motion (low-confidence observation)" |

---

## 7. Explicit claims the software must NOT make

Each item is prohibited because the cited evidence does not support it, or because it crosses the boundary from decision support into diagnosis, prognosis or prescription.

1. **"You have patellofemoral pain / ITB syndrome / MTSS / Achilles tendinopathy / plantar fasciopathy / hamstring strain."** Video cannot establish it [R01,R02]; diagnosis requires history, examination and, where indicated, imaging (clinical principle).
2. **"Your injury risk is X%."** No validated prediction model exists for these metrics; prospective biomechanical evidence is sparse and inconsistent [R04].
3. **"This gait pattern is causing your pain."** The evidence is associational, mostly cross-sectional; causal language overstates it [R04–R10].
4. **"Your pelvic drop proves gluteus medius weakness."** Contralateral pelvic drop is associated with PFP in some cohorts [R05], but video does not measure strength; 2-D frontal validity for hip adduction is poor [R02], and pelvic drop was not separately validated.
5. **"Your knee valgus is abnormal / is a primary risk marker."** No source in this review validates a 2-D knee valgus angle; the related 3-D findings concern hip adduction and knee internal rotation [R05,R09], not frontal knee valgus. The 12° threshold is not validated.
6. **"Your cadence should be 180 spm."** The trials used the runner's preferred cadence ±5–10%, not a universal 180 [R13,R14].
7. **"Switch to a forefoot strike."** Changing footstrike has not been shown to improve economy, eliminate impact or reduce injury [R18].
8. **"The app measured your loading rate / ground reaction force / joint force."** It estimates kinematics; markerless kinetics differ from marker-based and the difference grows with speed [R03].
9. **"Your 2-D angles are equivalent to 3-D laboratory angles."** Criterion validity is poor-to-moderate for most parameters [R01]; one smartphone system was invalid for hip adduction and knee flexion [R02].
10. **"You can return to running in X weeks."** Return-to-run is a clinical, shared decision; evidence for specific criteria is scarce [R28], and bone stress injury management is individualised [R23].
11. **"These drills will correct your gait / fix your injury."** The drills are examples, not individualised prescriptions; clinical-outcome evidence for retraining is limited [R17].
12. **"No red flags means no serious pathology."** Red flags are suspicion triggers with limited diagnostic accuracy; their absence does not exclude disease [R19].
13. **"Asymmetry below 90% is abnormal."** No validated asymmetry cut-off exists; other spatiotemporal factors are under-studied [R04].
14. **"Your form score of X means you are safe / unsafe."** The composite and its weights are heuristic and unvalidated; no study links this score to injury outcomes.
15. **"The app can clear you for running."** The app is not a medical device and cannot provide clearance; clearance is a clinician's decision (scope-of-practice principle).

---

## 8. Evidence table

| ID | Claim supported | Source | Year | DOI / PMID / URL | Grade | Applicability | Caveats |
|---|---|---|---|---|---|---|---|
| R01 | 2-D video is reliable (good–excellent) but only poor–moderate valid vs 3-D for most running angles | Leporace et al., *J Biomech* | 2023 | doi:10.1016/j.jbiomech.2023.111747; PMID 37541054 | B | 2-D vs 3-D lab; recreational runners | 17 studies; heterogeneous methods; few studies per angle |
| R02 | Smartphone markerless 2-D was invalid for peak hip adduction (ICC 0.06) and knee flexion (ICC 0.42) in PFP runners | Neal et al., *Phys Ther Sport* | 2020 | doi:10.1016/j.ptsp.2020.02.004; PMID 32066107 | B | Runners with PFP; one app | Single app; n=21; does not prove all markerless systems invalid |
| R03 | Markerless-derived joint kinetics differ from marker-based, with larger differences at faster speeds | Tang et al., *J Sport Health Sci* | 2024 | doi:10.1016/j.jshs.2024.01.002; PMID 38218372 | B | Treadmill 2.24–3.58 m/s | n=16; lab; joint moments/powers, not GRF |
| R04 | Prospective biomechanical evidence for RRI is sparse and inconsistent; limited evidence for hip adduction in female PFP/ITBS | Ceyssens et al., *Sports Med* | 2019 | doi:10.1007/s40279-019-01110-z; PMID 31028658 | B | Prospective studies; non-injured runners | 16 studies; narrative synthesis; inconsistent |
| R05 | PFP is associated with greater hip adduction, hip internal rotation, contralateral pelvic drop and reduced hip flexion; retraining/strengthening improve symptoms | Neal et al., *Gait Posture* | 2016 | doi:10.1016/j.gaitpost.2015.11.018; PMID 26979886 | B | Runners with PFP | 28 studies; mostly female; limited prospective evidence |
| R06 | Female runners who developed PFP had greater hip adduction prospectively (p=0.007) | Noehren et al., *Med Sci Sports Exerc* | 2013 | doi:10.1249/MSS.0b013e31828249d2; PMID 23274607 | B | 400 female runners; 15 PFP cases | Small case count; females only; no difference in hip IR or rearfoot eversion |
| R07 | Female runners with current ITBS have lower hip abductor strength and smaller peak hip internal rotation; risk factors differ by sex/status | Foch et al., *Gait Posture* | 2023 | doi:10.1016/j.gaitpost.2023.02.001; PMID 36758425 | B | ITBS; 17 articles | Majority cross-sectional; limited prospective data |
| R08 | MTSS risk factors: higher BMI (MD 0.79 kg/m²), navicular drop (1.19 mm), ankle plantarflexion ROM (5.94°), hip external rotation ROM (3.95°) | Hamstra-Wright et al., *Br J Sports Med* | 2015 | doi:10.1136/bjsports-2014-093462; PMID 25185588 | B | Runners and military personnel | 21 studies; risk factors are not gait-specific; ROM/navicular drop need examination |
| R09 | ITBS: strong evidence for higher knee internal rotation; moderate for lower rearfoot eversion and knee flexion at heel strike; greater hip adduction. Achilles/plantar kinematic evidence very limited | Mousavi et al., *Gait Posture* | 2019 | doi:10.1016/j.gaitpost.2019.01.011; PMID 30658311 | B | Distance runners; tendinopathy/ITBS | 28 studies; heterogeneous; only 9 Achilles and 2 plantar studies |
| R10 | No consistent association between running biomechanics and hamstring strain; exceptions lateral trunk kinematics and horizontal propulsive force | Wolski et al., *Sports Biomech* | 2024 | doi:10.1080/14763141.2021.1960418; PMID 34569907 | C | High-speed running/sprinting | 12 studies; small samples; not distance running |
| R11 | Plantar fasciitis risk factors: higher plantarflexion ROM (7.04°), BMI (2.13 kg/m²), body mass (4.52 kg) | Hamstra-Wright et al., *Sports Health* | 2021 | doi:10.1177/1941738120970976; PMID 33530860 | B | Physically active individuals | 16 studies; Level 3; inconsistent definitions |
| R12 | Training errors (volume, duration, frequency, intensity) are not consistently linked to RRI; study quality is low | Nielsen et al., *Int J Sports Phys Ther* | 2012 | PMID 22389869; https://pubmed.ncbi.nlm.nih.gov/22389869/ | C | Recreational and elite runners | 31 studies; mean quality 44%; conflicting results; no DOI |
| R13 | +5–10% step rate reduces knee energy absorption, braking impulse, step length and peak knee flexion; +10% reduces hip adduction and moments | Heiderscheit et al., *Med Sci Sports Exerc* | 2011 | doi:10.1249/MSS.0b013e3181ebedf4; PMID 20581720 | B | Healthy recreational runners; treadmill | 45 runners; acute; no clinical outcome |
| R14 | 110% of preferred step rate reduces peak patellofemoral joint force by 14%; knee flexion is the strongest predictor | Lenhart et al., *Med Sci Sports Exerc* | 2014 | doi:10.1249/MSS.0b013e3182a78c3a; PMID 23917470 | B | Healthy adults; musculoskeletal modelling | 30 adults; model-dependent; swing-phase hamstring/rectus femoris loads increase |
| R15 | 2-week feedback-based gait retraining reduced 12-month injury in novice runners: 16% vs 38%; HR 0.38 (62% lower) | Chan et al., *Am J Sports Med* | 2018 | doi:10.1177/0363546517736277; PMID 29065279 | A | Novice distance runners | Single RCT; specific intervention; prevention, not treatment |
| R16 | In-field +7.5% step-rate cue: load rates −17.9% to −18.9%, hip adduction −2.9°, knee work −26.9%, maintained 30 days | Willy et al., *Scand J Med Sci Sports* | 2016 | doi:10.1111/sms.12413; PMID 25652871 | B | At-risk healthy runners | n=30; short-term; surrogate outcomes |
| R17 | Running retraining has limited clinical-outcome evidence but substantial immediate biomechanical evidence; expert support for several conditions | Barton et al., *Br J Sports Med* | 2016 | doi:10.1136/bjsports-2015-095278; PMID 26884223 | C | Lower-limb running injuries | Mixed-methods; expert opinion; not a trial |
| R18 | Changing to mid/forefoot strike does not improve economy, eliminate impact or reduce injury risk | Hamill & Gruber, *J Sport Health Sci* | 2017 | doi:10.1016/j.jshs.2017.02.004; PMID 30356626 | C | Runners | Narrative critique; no new data |
| R19 | Red flags are suspicion triggers; high-quality diagnostic-accuracy evidence is lacking | Finucane et al., *J Orthop Sports Phys Ther* | 2020 | doi:10.2519/jospt.2020.9971; PMID 32438853 | C | Serious spinal pathology | Framework; not running-specific; not a rule-out |
| R20 | Cauda equina: bladder/sensory disturbance most common, but no single feature achieves consensus | Fraser et al., *Arch Phys Med Rehabil* | 2009 | doi:10.1016/j.apmr.2009.03.021; PMID 19887225 | C | Cauda equina literature | 105 articles; no unanimity; emergency referral |
| R21 | CECS diagnosis relies on compartment pressure; thresholds vary (>35 mmHg vs Pedowitz) | Tzortziou et al., *Clin J Sport Med* | 2006 | doi:10.1097/00042752-200605000-00004; PMID 16778540 | C | UK orthopaedic practice | Survey; not diagnostic accuracy; old data |
| R22 | ESC sports cardiology guidance addresses cardiovascular evaluation and exercise prescription in active individuals; exertional cardiac symptoms need medical assessment | Pelliccia et al., *Eur Heart J* | 2021 | doi:10.1093/eurheartj/ehaa605; PMID 32860412 | A | Active individuals with cardiovascular disease | Guideline; not running-specific; not a screening tool |
| R23 | Bone stress injury is a load-related continuum; management is modified loading, progressive return and addressing risk factors | Warden et al., *J Orthop Sports Phys Ther* | 2014 | doi:10.2519/jospt.2014.5334; PMID 25103133 | C | Long-distance runners | Clinical commentary (level 5); no trial |
| R24 | LEFS: 20 items, 0–80; MCID/MDC 9 points; typical error ±5.3 | Binkley et al., *Phys Ther* | 1999 | doi:10.1093/ptj/79.4.371; PMID 10201543 | B | Lower-extremity musculoskeletal dysfunction | 107 patients; not running-specific |
| R25 | OSTRC-O: 4 items, 0–100 weekly; captures more overuse burden than time-loss definitions | Clarsen et al., *Br J Sports Med* | 2013 | doi:10.1136/bjsports-2012-091524; PMID 23038786 | B | Athletes in five sports | 313 athletes; not running-specific; no MCID |
| R26 | NPRS: 0–10; MDC 2 points, MCID about 2 points | Childs et al., *Spine* | 2005 | doi:10.1097/01.brs.0000164099.92112.29; PMID 15928561 | B | Low back pain | Not running-specific; recall period matters |
| R27 | VISA-A: 8 items, 0–100; reliable and valid in Achilles tendinopathy; not diagnostic | Robinson et al., *Br J Sports Med* | 2001 | doi:10.1136/bjsm.35.5.335; PMID 11579069 | B | Achilles tendinopathy | No MCID in original; not diagnostic |
| R28 | Return to sport is a continuum with shared risk assessment; specific criteria evidence is scarce | Ardern et al., *Br J Sports Med* | 2016 | doi:10.1136/bjsports-2016-096278; PMID 27226389 | B | Athletes | Consensus; not running-specific; no timeline |
| R29 | Pain-monitoring model allowed continued Achilles loading without worse outcomes; VISA-A 57→85 at 12 months | Silbernagel et al., *Am J Sports Med* | 2007 | doi:10.1177/0363546506298279; PMID 17307888 | B | Achilles tendinopathy | n=38; specific exercise programme; not generalisable to fracture |
| R30 | VISA-P: 8 items, 0–100; patellar tendinopathy severity | Visentini et al., *J Sci Med Sport* | 1998 | doi:10.1016/S1440-2440(98)80005-4; PMID 9732118 | B | Patellar tendinopathy | Original development; no MCID; not diagnostic |

---

## 9. Verification, limitations and what would change this report

### 9.1 Verification

- **Search date:** 2026-10-21.
- **Databases:** PubMed (E-utilities) and Crossref.
- **Method:** Each source was identified by title/author search in PubMed, its PMID and bibliographic details were retrieved, and its DOI was resolved through the Crossref REST API. R12 (Nielsen 2012) has no DOI listed in its PubMed record; its PMID was verified. R24 (LEFS) has no DOI in PubMed; the DOI was verified through Crossref. R30 (VISA-P) is indexed in PubMed (PMID 9732118); both the PMID and the DOI were verified.
- **Result:** 30 of 30 sources verified. **No source was flagged as unverifiable.** No preprint, conference abstract, vendor page or invented citation is used.
- **What "verified" means:** the DOI resolves to a record whose title, journal, year and authors match the citation used here. It does not mean the study is high quality; quality is reflected in the grade and caveats.

### 9.2 Limitations

- This is a **narrative evidence review**, not a systematic review. Search terms, inclusion decisions and grading were done by one reviewer; a second reviewer could change some grades.
- The evidence base is **heterogeneous**. Many studies are cross-sectional, small, or use different angle definitions, camera setups and injury definitions. Pooled effect sizes are not available for most running-injury associations.
- **Applicability to RunTrack's camera and population is unknown.** No study validates this specific MediaPipe pipeline, camera, distance or population. The measurement literature is borrowed from other 2-D systems and laboratories.
- **The PROMs are generic or tendon-specific, not running-specific.** Their measurement properties were established in broader populations; using them in runners is reasonable but not the same as validation in runners.
- **Guidelines are not running-injury guidelines.** The ESC document is a cardiology guideline; the red-flag framework is spinal; CECS data are from a 2006 UK survey.
- **No formal GRADE process** was used. The A–D grades are a pragmatic certainty scale, defined above, not the GRADE system.

### 9.3 What would change this report

- A **local test–retest study** of the RunTrack pipeline (same runner, same camera, 10 repeats) would replace borrowed measurement error with the app's own repeatability and settle which flags are detectable at all.
- A **prospective validation** of the risk strata against injury outcomes would be needed before any risk label could be called a risk label.
- **Running-specific PROM validation** (LEFS, OSTRC, VISA in runners) would improve outcome tracking.
- **Better frontal-plane markerless validation** would be needed before pelvic drop or knee valgus could carry clinical weight.
- A **larger, injured-runner RCT of cadence retraining** would move §3 from prevention evidence to treatment evidence.

---

## 10. References

**R01.** Leporace G, Metsavaht L, Gonzalez FF, et al. Validity and reliability of two-dimensional video-based assessment to measure joint angles during running: a systematic review and meta-analysis. *J Biomech.* 2023;157:111747. doi:10.1016/j.jbiomech.2023.111747. PMID: 37541054.

**R02.** Neal BS, Lack SD, Barton CJ, et al. Is markerless, smart phone recorded two-dimensional video a clinically useful measure of relevant lower limb kinematics in runners with patellofemoral pain? A validity and reliability study. *Phys Ther Sport.* 2020;43:36–42. doi:10.1016/j.ptsp.2020.02.004. PMID: 32066107.

**R03.** Tang H, Munkasy B, Li L. Differences between lower extremity joint running kinetics captured by marker-based and markerless systems were speed dependent. *J Sport Health Sci.* 2024;13(4):569–578. doi:10.1016/j.jshs.2024.01.002. PMID: 38218372.

**R04.** Ceyssens L, Vanelderen R, Barton C, et al. Biomechanical risk factors associated with running-related injuries: a systematic review. *Sports Med.* 2019;49(7):1095–1115. doi:10.1007/s40279-019-01110-z. PMID: 31028658.

**R05.** Neal BS, Barton CJ, Gallie R, et al. Runners with patellofemoral pain have altered biomechanics which targeted interventions can modify: a systematic review and meta-analysis. *Gait Posture.* 2016;45:69–82. doi:10.1016/j.gaitpost.2015.11.018. PMID: 26979886.

**R06.** Noehren B, Hamill J, Davis I. Prospective evidence for a hip etiology in patellofemoral pain. *Med Sci Sports Exerc.* 2013;45(6):1120–1124. doi:10.1249/MSS.0b013e31828249d2. PMID: 23274607.

**R07.** Foch E, Brindle RA, Pohl MB. Lower extremity kinematics during running and hip abductor strength in iliotibial band syndrome: a systematic review and meta-analysis. *Gait Posture.* 2023;101:73–81. doi:10.1016/j.gaitpost.2023.02.001. PMID: 36758425.

**R08.** Hamstra-Wright KL, Bliven KC, Bay C. Risk factors for medial tibial stress syndrome in physically active individuals such as runners and military personnel: a systematic review and meta-analysis. *Br J Sports Med.* 2015;49(6):362–369. doi:10.1136/bjsports-2014-093462. PMID: 25185588.

**R09.** Mousavi SH, Hijmans JM, Rajabi R, et al. Kinematic risk factors for lower limb tendinopathy in distance runners: a systematic review and meta-analysis. *Gait Posture.* 2019;69:13–24. doi:10.1016/j.gaitpost.2019.01.011. PMID: 30658311.

**R10.** Wolski L, Pappas E, Hiller C, et al. Is there an association between high-speed running biomechanics and hamstring strain injury? A systematic review. *Sports Biomech.* 2024;23(10):1313–1339. doi:10.1080/14763141.2021.1960418. PMID: 34569907.

**R11.** Hamstra-Wright KL, Huxel Bliven KC, Bay RC, et al. Risk factors for plantar fasciitis in physically active individuals: a systematic review and meta-analysis. *Sports Health.* 2021;13(3):296–303. doi:10.1177/1941738120970976. PMID: 33530860.

**R12.** Nielsen RO, Buist I, Sørensen H, et al. Training errors and running related injuries: a systematic review. *Int J Sports Phys Ther.* 2012;7(1):58–75. PMID: 22389869. https://pubmed.ncbi.nlm.nih.gov/22389869/.

**R13.** Heiderscheit BC, Chumanov ES, Michalski MP, et al. Effects of step rate manipulation on joint mechanics during running. *Med Sci Sports Exerc.* 2011;43(2):296–302. doi:10.1249/MSS.0b013e3181ebedf4. PMID: 20581720.

**R14.** Lenhart RL, Thelen DG, Wille CM, et al. Increasing running step rate reduces patellofemoral joint forces. *Med Sci Sports Exerc.* 2014;46(3):557–564. doi:10.1249/MSS.0b013e3182a78c3a. PMID: 23917470.

**R15.** Chan ZYS, Zhang JH, Au IPH, et al. Gait retraining for the reduction of injury occurrence in novice distance runners: 1-year follow-up of a randomized controlled trial. *Am J Sports Med.* 2018;46(2):388–395. doi:10.1177/0363546517736277. PMID: 29065279.

**R16.** Willy RW, Buchenic L, Rogacki K, et al. In-field gait retraining and mobile monitoring to address running biomechanics associated with tibial stress fracture. *Scand J Med Sci Sports.* 2016;26(2):197–205. doi:10.1111/sms.12413. PMID: 25652871.

**R17.** Barton CJ, Bonanno DR, Carr J, et al. Running retraining to treat lower limb injuries: a mixed-methods study of current evidence synthesised with expert opinion. *Br J Sports Med.* 2016;50(9):513–526. doi:10.1136/bjsports-2015-095278. PMID: 26884223.

**R18.** Hamill J, Gruber AH. Is changing footstrike pattern beneficial to runners? *J Sport Health Sci.* 2017;6(2):146–153. doi:10.1016/j.jshs.2017.02.004. PMID: 30356626.

**R19.** Finucane LM, Downie A, Mercer C, et al. International framework for red flags for potential serious spinal pathologies. *J Orthop Sports Phys Ther.* 2020;50(7):350–372. doi:10.2519/jospt.2020.9971. PMID: 32438853.

**R20.** Fraser S, Roberts L, Murphy E. Cauda equina syndrome: a literature review of its definition and clinical presentation. *Arch Phys Med Rehabil.* 2009;90(11):1964–1968. doi:10.1016/j.apmr.2009.03.021. PMID: 19887225.

**R21.** Tzortziou V, Maffulli N, Padhiar N. Diagnosis and management of chronic exertional compartment syndrome (CECS) in the United Kingdom. *Clin J Sport Med.* 2006;16(3):209–213. doi:10.1097/00042752-200605000-00004. PMID: 16778540.

**R22.** Pelliccia A, Sharma S, Gati S, et al. 2020 ESC Guidelines on sports cardiology and exercise in patients with cardiovascular disease. *Eur Heart J.* 2021;42(1):17–96. doi:10.1093/eurheartj/ehaa605. PMID: 32860412.

**R23.** Warden SJ, Davis IS, Fredericson M. Management and prevention of bone stress injuries in long-distance runners. *J Orthop Sports Phys Ther.* 2014;44(10):749–765. doi:10.2519/jospt.2014.5334. PMID: 25103133.

**R24.** Binkley JM, Stratford PW, Lott SA, et al. The Lower Extremity Functional Scale (LEFS): scale development, measurement properties, and clinical application. *Phys Ther.* 1999;79(4):371–383. doi:10.1093/ptj/79.4.371. PMID: 10201543.

**R25.** Clarsen B, Myklebust G, Bahr R. Development and validation of a new method for the registration of overuse injuries in sports injury epidemiology: the Oslo Sports Trauma Research Centre (OSTRC) Overuse Injury Questionnaire. *Br J Sports Med.* 2013;47(8):495–502. doi:10.1136/bjsports-2012-091524. PMID: 23038786.

**R26.** Childs JD, Piva SR, Fritz JM. Responsiveness of the numeric pain rating scale in patients with low back pain. *Spine (Phila Pa 1976).* 2005;30(11):1331–1334. doi:10.1097/01.brs.0000164099.92112.29. PMID: 15928561.

**R27.** Robinson JM, Cook JL, Purdam C, et al. The VISA-A questionnaire: a valid and reliable index of the clinical severity of Achilles tendinopathy. *Br J Sports Med.* 2001;35(5):335–341. doi:10.1136/bjsm.35.5.335. PMID: 11579069.

**R28.** Ardern CL, Glasgow P, Schneiders A, et al. 2016 Consensus statement on return to sport from the First World Congress in Sports Physical Therapy, Bern. *Br J Sports Med.* 2016;50(14):853–864. doi:10.1136/bjsports-2016-096278. PMID: 27226389.

**R29.** Silbernagel KG, Thomeé R, Eriksson BI, et al. Continued sports activity, using a pain-monitoring model, during rehabilitation in patients with Achilles tendinopathy: a randomized controlled study. *Am J Sports Med.* 2007;35(6):897–906. doi:10.1177/0363546506298279. PMID: 17307888.

**R30.** Visentini PJ, Khan KM, Cook JL, Kiss ZS, Harcourt PR, Wark JD. The VISA score: an index of severity of symptoms in patients with jumper's knee (patellar tendinosis). *J Sci Med Sport.* 1998;1(1):22–28. doi:10.1016/S1440-2440(98)80005-4. PMID: 9732118.

---

## Appendix A — Suggested JSON evidence-catalog schema

The table in §8 is the human-readable catalog. For the local model, each row should become one machine-readable claim with the same ID, so that every generated sentence can be traced back to a source.

```json
{
  "catalog_version": "2026-10-21.1",
  "review_status": "clinician_review_required",
  "claims": [
    {
      "id": "RCL-001",
      "domain": "measurement",
      "claim": "Two-dimensional video is reliable for repeated running-angle measurement but has poor-to-moderate criterion validity against three-dimensional analysis for most parameters.",
      "grade": "B",
      "sources": ["R01"],
      "applicability": "Recreational runners; 2-D video vs 3-D laboratory analysis",
      "caveats": [
        "Heterogeneous methods across 17 studies",
        "Few studies per joint angle",
        "Not validated on the RunTrack camera pipeline"
      ],
      "allowed_phrasing": [
        "2-D estimates may differ from 3-D laboratory values",
        "Repeated measurements under the same setup are the most defensible use"
      ],
      "forbidden_phrasing": [
        "2-D angles are equivalent to 3-D",
        "This angle is accurate to one degree"
      ],
      "app_metrics": ["all joint angles"],
      "review": {
        "status": "pending_clinician_review",
        "reviewer": "",
        "date": ""
      }
    },
    {
      "id": "RCL-002",
      "domain": "injury_association",
      "claim": "Greater peak hip adduction has been associated with patellofemoral pain in female runners, but the evidence is not diagnostic and does not establish causation.",
      "grade": "B",
      "sources": ["R05", "R06"],
      "applicability": "Female runners with patellofemoral pain",
      "caveats": [
        "Mostly cross-sectional; prospective evidence is limited",
        "Small number of prospective PFP cases",
        "2-D frontal-plane measurement is low confidence"
      ],
      "allowed_phrasing": [
        "has been associated with",
        "should be assessed clinically"
      ],
      "forbidden_phrasing": [
        "causes patellofemoral pain",
        "proves hip weakness",
        "diagnostic of PFP"
      ],
      "app_metrics": ["pelvic_drop_deg", "knee_valgus_deg"],
      "review": {
        "status": "pending_clinician_review",
        "reviewer": "",
        "date": ""
      }
    }
  ]
}
```

**Catalog rules.**
- One claim per row; no compound claims that mix measurement, association and treatment.
- `grade` must be one of A/B/C/D and must match §8.
- `sources` must contain only IDs present in §10.
- `allowed_phrasing` and `forbidden_phrasing` are exemplars, not an exhaustive list; the prompt constraints in Appendix B apply in addition.
- `review.status` must be `clinician_reviewed` before the claim can appear in a signed note.

---

## Appendix B — Prompt constraints for the local model (Qwen3.8 27B, offline)

The model is a drafting aid. The clinician is the author. These constraints should be applied as system-level rules and checked after generation.

1. **Use only the evidence catalog.** Every factual sentence must map to a catalog `id`; if no catalog claim covers the sentence, do not write it. Do not use general medical knowledge to fill gaps.
2. **Never diagnose.** Do not output a diagnosis, ICD code, or "impression". If asked for a diagnosis, reply: "Diagnosis requires clinical assessment; I can summarise the observations and the evidence."
3. **Never predict.** Do not output a probability, percentage, risk category as a prognosis, or a statement that an injury will or will not happen.
4. **Use association language.** Prefer "associated with", "observed in", "consistent with"; avoid "causes", "leads to", "will injure", "proves".
5. **State the plane and the confidence.** Every kinematic statement must name the plane (sagittal/frontal) and the measurement confidence (low/moderate). Frontal-plane statements default to low confidence.
6. **Do not infer missing data.** If a metric is missing, low-confidence, or not assessable, write "not assessed" and ask for the missing information. Never fill a gap with a plausible value.
7. **Red flags stop the gait narrative.** If any red flag is present, do not interpret gait or recommend exercise progression; recommend medical review and document the trigger.
8. **No footstrike prescription.** Do not recommend switching to a forefoot or midfoot strike.
9. **No universal cadence target.** Do not say "180 spm". If discussing cadence, describe a +5% trial from the runner's preferred cadence and the need to reassess.
10. **No return-to-run timeline.** Do not give a week-by-week return plan or a date. Describe the clinician's criteria and the need for shared decision-making.
11. **No treatment guarantees.** Do not say a drill will "fix", "correct" or "cure" anything; describe the drill as an option for clinician consideration.
12. **No cross-system comparison.** Do not compare the runner's angles with 3-D laboratory norms, other apps, elite runners or published means.
13. **No PROM invention.** Do not generate or estimate a PROM score; only interpret a score the clinician entered, with its MCID/MDC caveat.
14. **Quote the caveat.** Every generated report must include: "Decision support only; not a diagnosis; clinician review required."
15. **Show the sources.** End each observation card with its catalog ID(s) and grade; do not cite a source that is not in the catalog.

**Forbidden sentence patterns (examples):** "You have…", "Your risk is…", "This causes…", "You should switch to…", "You can return in…", "This proves…", "Your gait is safe…", "No red flags means…".

---

## Appendix C — Clinician review checklist

Before signing a RunTrack note, the reviewing clinician should confirm:

- [ ] The runner's identity, date and session are correct.
- [ ] The camera plane, distance, frame rate and clothing are recorded.
- [ ] Event-detection quality is reported (clean strikes per side, tracking coverage).
- [ ] Every kinematic statement names its plane and confidence.
- [ ] No statement exceeds the evidence grade in §8.
- [ ] Red flags were screened and, if present, acted on.
- [ ] At least one PROM is recorded with its MCID/MDC caveat.
- [ ] Training load and the runner's goal are recorded.
- [ ] No diagnosis, probability, return date or treatment guarantee appears.
- [ ] The automated text was edited, accepted or rejected, and the edits are documented.
- [ ] The note includes the decision-support caveat.
- [ ] The clinician's signature and date are present.

---

*End of report. This document is an evidence review for clinician review; it is not a medical device and does not provide diagnosis or treatment.*
