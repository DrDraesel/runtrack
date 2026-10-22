# TT / triathlon bike fit — evidence review for the RunTrack cycling module

Scope: what the targets in `config.py` (`CYCLING_BANDS`, `CYCLING_FLAGS`) and `docs/CYCLING-TT-FIT.md`
can honestly claim. Written 2026-10-21 for `C:\Users\GAMEPOWER\projects\runtrack`.

**Grades:** **A** multiple primary studies/systematic review, direct to cycling · **B** one good primary study or several small consistent ones · **C** indirect/modelling/single small study/commercial range consistent with geometry · **D** coaching convention or vendor assertion, no published validation.

## 0. Bottom line

1. **2-D knee angle at BDC is valid but biased low ~2°:** +2.2° correction vs 3-D [4]; −2.4° at the knee and ~2° at the hip [5]; MediaPipe adds 3–9° typical error, ankle worst [6].
2. **Published knee targets disagree by ~10°** because static vs dynamic and included vs flexion are mixed. Holmes 25–35° flexion is a *static goniometer* rule [3,31]; dynamic BDC reads ~8° more flexed = 33–43° flexion = 137–147° included [3].
3. **The repo's road hip band looks wrong:** MyVeloFit's public road report uses closed hip 60–69° [13]; Velogic's road hip maximum (the open position) is 90–105° and says tri hip should "fairly closely match" road [10,11]. `config.py` uses 70–90° closed on road, above every published road range found; TT 45–75° is plausible but wide.
4. **Frontal area from a silhouette is not CdA:** area explained only 54% of variance in drag area [26], and smaller area did not automatically reduce drag across TT arm positions [27].
5. **A 5 mm saddle change is below single-measurement resolution:** ≈0.25°/mm at BDC (derived from the 2% saddle-height spans in [5]) → 5 mm ≈1.2°, 10 mm ≈2.5°, against 3–9° markerless error [6] and 2.15–6.55° Retül MDD [17]. Test 10 mm steps and repeat the baseline.

## 1. Targets and rationale

### 1.1 Knee at BDC — the one metric with real evidence

| Source | Convention | Range |
|---|---|---|
| Holmes, via [3,31] | static goniometry, **flexion** | 25–35° flexion (=145–155° included) |
| Velogic road & tri [10,11] | dynamic, **included** | max 140–145°, min >68° (=35–40° flexion) |
| MyVeloFit public report [13] | dynamic, **extension deficit** | extension 35–41° (=139–145° included) |
| Millour 2019 [3] | dynamic 2-D at BDC | 33–43° flexion (=137–147° included) |
| Swart & Holliday [1] | dynamic, intensity-dependent | 33–43° low intensity, 30–40° high intensity |

The disagreement is method, not physiology: dynamic BDC measured 8° greater than static and 3° greater than dynamic maximum extension [3]. **Use 138–147° included (33–42° flexion) for road and TT**, with 140–145° as the narrow commercial convention, and print "dynamic, included angle at BDC" beside every number. The present TT band being *narrower and more flexed* than road is unsupported — TT riders keep the leg–pedal relationship and rotate the torso forward [11,22].

### 1.2 Knee at TDC
Velogic gives a one-sided floor, min >68° included [10,11]; MyVeloFit shows 105–114° flexion at the top [13] = 66–75° included. The repo's 68–95° turns a floor into a band whose 95° upper bound (85° flexion) is implausible and will never bind. **Use "≥68°, flag below"; drop the 95.** Grade C.

### 1.3 Hip angle — closed at TDC, open at BDC
MyVeloFit road: **hip closed 60–69°**, hip open 95–104° [13]. Velogic road: hip **max** 90–105° (up to 120 with issues) [10], tri max 85–105° [11], and "hip angle and associated metrics should fairly closely match between the road drops and aerobars" [11]. Commercial TT guides put closed hip at 35–55° [22] or 35–45° base [23] (grade D). `hip_tdc_min_deg` legitimately differs from Velogic's maximum, but 70–90° road would pass a rider whose hip never closes. **Recommend road 55–70° closed (C), TT/tri 40–60° closed (D).** Holliday's 77±5° hip value [2] is definition-dependent; do not compare it with either.

### 1.4 Torso vs horizontal
Velogic road 35–40° elite, 40–50° general, 50°+ rare [10]; tri 10–15° elite, 20–30° general [11]. Pedalyze TT 10–25° [22]; BikeFittr tri back 20–35° [23]. The repo's 35–45 / 10–25 / 15–30 sit inside the envelope — **grade C**, convention, not a validated optimum. The real disagreement is at the aggressive end (10° vs 20°), where flexibility and power loss decide [24].

### 1.5 Elbow, forearm, shoulder
Road elbow: Velogic 155–170° [10], MyVeloFit 150–169° [13] — consistent, **C**. TT elbow 85–105° appears in Pedalyze's TT guide [22] but has no primary source, and `config.py` has no elbow band at all — **D, label convention**. TT forearm angle 0–20°, 5–10° good [11], is the better TT metric because it captures pad/extension tilt — **C**. Shoulder 75–100° [10,11] vs 112±7° [2] use different landmarks — **D, do not compare across tools**. Upper-arm angle is not separately defined by any source found. Head/neck position has no validated target and is unreliable from a side view (helmet/head occlusion) — **D, descriptive only; no band**.

### 1.6 Ankle, setback, sitting angle, pad stack
Velogic road ankle 100–130° (range <30°), tri 100–120° [10,11]; the repo's road 80–100° is outside that and needs a definition check (included foot–shank vs dorsiflexion). Markerless tracking is worst at the foot — MediaPipe correlations were as low as 0.25 and both networks performed poorly at the foot/ankle [6] — so **report ankle descriptively, do not flag** (grade C). Sitting angle (road 72–76°, tri 78–86°, aim 80°) and knee-over-foot (road 0 to −40 mm, tri 20–80 mm) need a bottom-bracket reference the app lacks [10,11]; pad Y/X is BB-to-pad-centre while Retül reports pad-rear, a systematic offset of roughly 1–2 cm depending on pad length [21]. **C/D, no universal target.**

### 1.7 Crank length
5–10 mm shorter reduced knee flexion 0–24% and hip flexion 0–13%, increased anterior pelvic tilt, with **no change in mean power** (199.1 ± 50.5 W) [35] — **B**. A 2026 preprint models **+1.42° minimum hip angle per 10 mm shorter** with little mechanical penalty, and reports triathletes choosing shorter relative cranks [37] — **C (preprint)**. 165/170/175 mm: no economy or sprint difference, less fatigue at 165/170 [36]. In boys 8–11, 170 mm vs 20% leg length gave equal power (364 vs 366 W), only higher optimal cadence with short cranks [38]. Crank length is an unmodelled confounder: 155 vs 170 mm moves hip ~2°, inside the app's own noise. Record it in session metadata.

## 2. Evidence quality: 2-D video, markerless pose, camera setup

| Study | Comparison | Error / finding |
|---|---|---|
| Fonda 2014 [4] | 2-D video vs 13-camera 3-D, 11 cyclists | 2-D valid; **add 2.2°**; all dynamic methods under-read vs 3-D |
| García-López 2017 [5] | 2-D Kinovea vs 3-D, 12 cyclists, 3 saddle heights | knee extension −2.4°, hip ~2° (2–3° in abstract), knee flexion −3.4°; trunk ns; detected 2% saddle changes |
| Bini 2023/24 [6,7] | MediaPipe/TransPose vs markers, 26 cyclists, 30 fps 640×480 | MediaPipe **3–9°**, r 0.25–0.96; poor foot/ankle; assumed 2-D error 2.2–10° |
| OpenCap 2024 [8] | markerless 3-D vs marker-based, 10 cyclists | knee RMSE **9.3–10.2°**, hip 7.9°, ankle 10.7–12.4°, despite r >0.9 |
| Umberger & Martin 2001 [9] | planar assumption, ergometer cycling | sagittal 2-D valid for hip/knee/ankle flexion–extension (as summarised in [5]) |
| Retül reliability [17] | Retül Vantage, 18 Hz, 10 cyclists | SE <3°, **MDD 2.15–6.55°**, linear MDD 15.5 mm |

**Bias is systematic, not random.** A band calibrated on 2-D video is internally usable, but absolute degrees must never be compared with a 3-D fit or another camera; same-setup repeated measures are the only defensible use of small differences.
- **Distance/height:** MyVeloFit specifies 2–3 m, square to the rider, at saddle height [41]; Bini used 4 m at saddle height [6]. Closer + wide-angle phone lens increases perspective error; 3–4 m preferred.
- **Frame rate:** at 90 rpm, 30 fps = 20 frames/rev = 18° crank/frame; 60 fps = 9°/frame (derived arithmetic). 60 fps halves BDC/TDC keyframe quantisation; 30 fps is usable for means, noisier per revolution.
- **Clothing/contrast:** snug, contrasting clothing and no mirrors/other people [41]; markerless gait found clothing did not change meaningful interpretation [39], but bar/saddle occlusion near the knee is a cycling-specific risk.
- **Frontal plane:** a side camera cannot see valgus/varus, hip drop or asymmetry. Velogic measures knee lateral travel (<40 mm avg, <20 good) and travel angle (<8°, 0–2° good) from front/rear [10]; the repo's rear targets copy those numbers — **C**, vendor convention.

## 3. Aero: what a silhouette can and cannot say

- >14 m/s: ~90% of mechanical power goes to overcoming drag [28]; upright→TT cuts drag ~35% (via [24]); the flow field is highly three-dimensional and non-linear [30].
- Photogrammetric area is reliable — CAD digitising vs weighing photographs CV 0.1% vs 1.26% [28] — but that validates the *area*, not the drag.
- **Area ≠ drag area:** projected area explained only **54% of variance in field drag area**; power/Ad correlated r = −0.92 with TT time vs −0.75 for power/area [26].
- Across eight TT arm positions, "a reduction in projected frontal area did not automatically reduce aerodynamic drag" [27]; Cd changed more than A.
- Velogic's aero module uses a depth camera for area and *estimates* Cd from torso angle; "changes due to e.g. different clothing won't be detected" [12]. If a depth-camera tool cannot measure Cd, a 2-D silhouette cannot.
- Field vs tunnel: velodrome vs wind tunnel r = 0.88, biased −0.0035 ± 0.0038 m² and not interchangeable; steady-state r = 0.99; detected 0.011 ± 0.007 m² (4.6 ± 2.9%) position changes [29]. Chung/VE is a ranking method, not an absolute measurement [40].
- TT trade-off: torso 24°→0° cost ~14% peak power while cutting drag; Cd in TT position spans 0.63–0.99 [24]; an optimum torso angle exists and moves with speed [24,25]. "Flatter is faster" is false. Hip closure is claimed to restrict breathing, but no validated threshold exists and the app cannot measure respiration — do not quantify it.

**May say:** relative area change, same camera/clothing/distance. **Must not say:** CdA, watts saved, predicted speed. `watts_delta_est` is a Cd = 0.80 scenario (`config.py`), not a measurement.

## 4. Clinical and safety

- Bike fit is not a medical device and cannot diagnose. A knee-pain review found patellofemoral pain is commonly attributed to a saddle "too low and/or too far forward", but "only a few of these strategies have been tested experimentally, and most of the advocated recommendations lack of solid evidence" [33].
- Saddle height: "strong evidence" for setting it from **dynamic** knee angle, but limited evidence that any method prevents injury [31]; an earlier review found methods "varied and not well established" [32].
- Knee load is not simply "lower saddle = more load": instrumented implants showed 119% BW peak tibiofemoral force at 60 W/40 rpm; a lower seat **did not increase resultant force** but increased posterior shear [34]. Load rose with power, fell with cadence.
- Paediatric: no validated angle norms found; the only relevant trial shows crank length barely affects boys' maximal power [38]. Growth, not adult targets, should drive changes.
- Refer out for pain with swelling/locking/giving way, neurological symptoms, prior knee surgery, or a request for injury treatment. Say plainly that the report is a measurement aid.

## 5. Platform review (public documentation only)

| Platform | Method | Metrics | Public targets | Proprietary / marketing |
|---|---|---|---|---|
| **Velogic Studio** [10,11,12] | depth camera + lateral camera | angles, joint motion, sitting angle, KOF, aero area | yes, explicit road/tri tables | Cd "estimated" from torso angle; clothing blind spot stated |
| **MyVeloFit** [13,14,15,41] | markerless 2-D, 2–3 m side view + mobility | knee flexion/extension, hip closed/open, back, shoulder, arm | per-report ranges, "calibrated" internally; full table not public | "accurate, reliable", 450k fits, no peer-reviewed validation published |
| **Retül** [16,17] | 3-D LED markers, 18 Hz, Zin bike map | 3-D angles, mm bike measurements | no public angle table | reliability published (SE <3°, MDD 2.15–6.55°); injury language is marketing |
| **BikeFit** [18] | fitter-led, qualitative; saddle first, then tilt/height | public articles reviewed are qualitative, no numeric angle table | none | SwitchIt tools proprietary |
| **AeroCoach** [19,20] | wind tunnel, velodrome, outdoor CdA | CdA, position/equipment deltas | no angle targets | claims 5–15% CdA / 15–35 W; data private |
| **Slowtwitch / F.I.S.T.** [21] | philosophy + frame geometry | Pad Y/X, stack/reach | no validated ranges | defines conventions, not targets |
| **Pedalyze / BikeFittr** [22,23] | commercial guides / AI photo fit | TT angle tables | Pedalyze: knee 25–35 flexion, hip 35–55, torso 10–25, elbow 85–105; BikeFittr: hip 35–45, back 20–35 | BikeFittr says closing the hip below 40° costs 5–15% power, citing "Retul data" with no published source |

## 6. Evidence-graded targets (included angles unless marked)

No individual target range earns an **A**: the A-grade finding is the *method* — dynamic knee angle is the best-supported way to set saddle height [31] — not any specific band. Ranges below are the best available bands for each metric.

| Metric | Road | TT | Tri | Grade | Type | Source |
|---|---|---|---|---|---|---|
| Knee included at BDC | 140–150 (test 138–147) | 140–147 | 140–147 | B | dynamic, validated method | [3,4,5,10,13] |
| Knee flexion at BDC | 30–40° | 33–42° | 33–42° | B | dynamic | [1,3] |
| Static knee flexion | 25–35° | 25–35° | 25–35° | C | static convention | [3,31] |
| Knee included at TDC (min) | ≥68 | ≥68 | ≥68 | C | floor, not a band | [10,11,13] |
| Hip closed at TDC | 55–70 | 40–60 | 40–60 | C/D | convention | [13,22,23] |
| Hip open at BDC | 90–105 | 85–105 | 85–105 | C | convention | [10,11,13] |
| Torso vs horizontal | 35–50 | 10–25 | 15–30 | C | convention | [10,11,22] |
| Elbow included | 150–170 | 85–105 | 85–105 | C/D | convention | [10,13,22] |
| Forearm vs horizontal | — | 0–20 (5–10) | 0–20 (5–10) | C | convention | [11] |
| Shoulder | 75–100 | 75–100 | 75–100 | D | definition-dependent | [10,11] |
| Ankle | 100–130 | 100–120 | 100–120 | C | convention; unreliable 2-D | [6,10,11] |
| Sitting angle (needs BB) | 72–76 | 78–86 | 78–86 | C | not measurable here | [10,11] |
| Knee-over-foot (needs BB) | 0 to −40 mm | 20–80 mm | 20–80 mm | D | not measurable here | [10,11] |
| Crank length | 10 mm shorter ≈ +1.4° hip | same | same | B/C | measured direction | [35,37] |

**Disagreements to state, not hide:** (i) knee BDC 25–35° vs 33–43° flexion — static vs dynamic; (ii) TT torso 10–15° elite vs 20–35° general — speed/flexibility; (iii) hip closed 35–45° vs 55–70° — commercial guides vs video-fit ranges; (iv) shoulder 75–100° vs 112±7° — landmark definitions.

## 7. What a single side camera cannot assess

Saddle setback/KOPS and sitting angle (no BB); pad Y/X and true stack/reach; frontal-plane knee tracking, valgus/varus, hip drop, pelvic obliquity, asymmetry (need rear/front); hip rocking; crank length; CdA or drag; saddle pressure; joint loads; breathing/oxygen cost; injury risk. Ankle/foot angles are measured but low-confidence [6]; elbow/shoulder are descriptive because no band is enforced in code.

## 8. Protocol for tomorrow's live test

**Setup.** Side camera perpendicular, 3–4 m, lens at hip/saddle height (~0.9–1.1 m), whole rider and both wheels in frame, tripod, 60 fps, 1080p, main 1× lens (not 0.5× ultra-wide), exposure/shutter locked, no zoom changes [6,41]. Rear camera 2.5 m behind at hip height [10]. Race kit, snug and contrasting; no mirrors/other people [41]. Mark floor positions for bike and cameras. Record saddle height (BB–saddle top), setback, pad stack/reach, crank length, tyre pressure, trainer setting.
**Warm-up and baseline.** 10–15 min warm-up [41], then 2 min at TT effort. Cadence self-selected 85–95 rpm held ±3 rpm; record actual power and HR. Capture 2–3 min (≈180–270 revs at 90 rpm; ≥60 s minimum). Repeat the baseline clip last to test drift.
**Positions, one change at a time, randomised:** (1) baseline; (2) saddle −10 mm; (3) saddle +10 mm; (4) pads −10 mm; (5) pads +10 mm; (6) optional pads ±10 mm reach. Do **not** use ±5 mm as the primary step — ≈1.2° knee angle [5], below markerless error [6] and Retül MDD [17]. Keep 5 mm only as a fine-tune once a 10 mm effect is confirmed.
**Record per trial:** knee BDC (mean, SD, per side), knee TDC, hip closed, torso, elbow, forearm, ankle, cadence, revolutions, power, HR, RPE, rider comments ("breathing", "hip pinch", "neck"), clip + session JSON.
**Acceptance criteria.** ≥20 clean revolutions per side; pose tracked >95% of frames; no bar/saddle occlusion of the knee; per-rev BDC SD ≤2.0°; cadence CV <3%; baseline repeat knee BDC within 2.5° (else session invalid — camera moved or fatigue); a difference must exceed 3° (knee/hip) or 5° (torso) to be called, otherwise "no detectable change"; any position changing power >5% or RPE ≥2 points is a performance trade-off, not a fit improvement. Report every §7 item as "not assessed", never blank.

## 9. What would change this report

A within-session repeatability study of this exact MediaPipe pipeline (same rider/camera, 10 repeats) would replace the borrowed 3–9° with a local error and settle whether 5 mm is ever detectable. A TT-specific closed-hip vs power/breathing dataset would upgrade §1.3 from C/D to B.

## References

1. Swart J, Holliday W. Cycling Biomechanics Optimization—the (R) Evolution of Bicycle Fitting. Curr Sports Med Rep 2019;18:490–496. https://pubmed.ncbi.nlm.nih.gov/31834181/
2. Holliday W, Swart J. Anthropometrics, flexibility and training history as determinants for bicycle configuration. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC9219349/
3. Millour G, et al. Comparison of static and dynamic methods based on knee kinematics to determine optimal saddle height. Acta Bioeng Biomech 2019;21:93–99. https://pubmed.ncbi.nlm.nih.gov/32022807/
4. Fonda B, Sarabon N, Li F-X. Validity and reliability of different kinematics methods used for bike fitting. J Sports Sci 2014;32:940–946. https://pubmed.ncbi.nlm.nih.gov/24499342/
5. García-López J, Abal del Blanco P. Kinematic analysis of bicycle pedalling using 2D and 3D motion capture systems. ISBS 2017. https://commons.nmu.edu/isbs/vol35/iss1/125/
6. Bini RR, Nascimento VB, Nibali A. Validity of neural networks in determining lower limb kinematics in stationary cycling. Sport Sci Health 2024;20:127–136. https://link.springer.com/article/10.1007/s11332-023-01075-7
7. Bini RR, et al. Criterion validity of neural networks to assess lower limb motion during cycling. 2023. https://pubmed.ncbi.nlm.nih.gov/36975046/
8. Kakavand R, et al. Comparison of kinematics and kinetics between OpenCap and a marker-based system in cycling. 2024. https://arxiv.org/pdf/2409.03766
9. Umberger BR, Martin PE. Testing the planar assumption during ergometer cycling. J Appl Biomech 2001;17:55–62. https://www.umass.edu/locomotion/pdfs/jab-2001.pdf
10. Velogic Studio docs — Road metrics. https://docs.velogicfit.com/section-02-using-velogic-studio/road-metrics
11. Velogic Studio docs — Triathlon metrics. https://docs.velogicfit.com/section-02-using-velogic-studio/triathlon-metrics
12. Velogic Studio docs — What you can (and can't) do with Aero. https://docs.velogicfit.com/section-04-aero/aero-capabilities
13. MyVeloFit — public sample fit report (road). https://videocdn.myvelofit.com/static/Fitreport/MyVeloFit+-+Justin+Fortune.pdf
14. MyVeloFit — AI bike fit / markerless motion capture. https://www.myvelofit.com/insights/ai-bike-fit-motion-capture-fit/
15. MyVeloFit — New Bike Fit Ranges (full table not public). https://www.myvelofit.com/fit-academy/new-bike-fit-ranges
16. Retül — Retül Fit. https://www.retul.com/retul-fit
17. Ribeiro Branco G, et al. Does the Retül System provide reliable kinematics information for cycling analysis? J Sci Cycling 2022;11:76–84. https://doi.org/10.28985/1322.jsc.15
18. BikeFit — How to Fit a Triathlon or TT Bike, Part 2: Pelvis/Saddle. https://bikefit.com/blogs/bikefit-blog/how-to-fit-a-triathlon-or-time-trial-tt-bike-part-2-the-pelvis-saddle-fitting
19. AeroCoach — Learn about aerodynamics. https://aero-coach.co.uk/learn
20. AeroCoach — Aero testing FAQ. https://aero-coach.co.uk/aero-testing-faq
21. Slowtwitch — What is Pad Y/X? https://slowtwitch.com/industry/what-is-pad-y-x
22. Pedalyze — TT & triathlon bike fit: angles and position. https://pedalyze.bike/en/guides/tt-bike-fit
23. BikeFittr — Balancing comfort and performance in triathlon bike fitting. https://www.bikefittr.com/blog/posts/triathlon-bike-fitting/comfort-performance-tri-bike-fit
24. Fintelman DM, et al. Optimal cycling time trial position models: aerodynamics versus power output and metabolic energy. J Biomech 2014. https://pubmed.ncbi.nlm.nih.gov/24726654/
25. Fintelman DM, et al. The effect of time trial cycling position on physiological and aerodynamic variables. J Sports Sci 2015;33:1730–1737. https://pubmed.ncbi.nlm.nih.gov/25658151
26. Peterman JE, et al. Field-measured drag area is a key correlate of level cycling time trial performance. PeerJ 2015;3:e1144. https://pmc.ncbi.nlm.nih.gov/articles/PMC4540006/
27. Maddocks P, Walker AD. Effect of arm position on the aerodynamic drag of an individual time trial position. Sports Eng 2025;28:11. https://doi.org/10.1007/s12283-025-00495-7
28. Debraux P, et al. New method to estimate the cycling frontal area. Int J Sports Med 2009;30:266–272. https://pubmed.ncbi.nlm.nih.gov/19199206/
29. Garcia-Lopez J, et al. The use of velodrome tests to evaluate aerodynamic drag in professional cyclists. Int J Sports Med 2014;35:451–455. https://pubmed.ncbi.nlm.nih.gov/24081618/
30. Crouch TN, et al. Riding against the wind: a review of competition cycling aerodynamics. Sports Eng 2017;20:81–110. https://doi.org/10.1007/s12283-017-0234-1
31. Bini RR, Priego-Quesada JI. Methods to determine saddle height in cycling and implications for performance and injury risk: a systematic review. J Sports Sci 2022. https://pubmed.ncbi.nlm.nih.gov/34706617
32. Bini RR, Hume PA, Croft JL. Effects of bicycle saddle height on knee injury risk and cycling performance. Sports Med 2011;41:463–476. https://pubmed.ncbi.nlm.nih.gov/21615188
33. Priego Quesada JI, et al. Potential factors associated with knee pain in cyclists: a systematic review. 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC5973630/
34. Kutzner I, et al. Loading of the knee joint during ergometer cycling: telemetric in vivo data. J Orthop Sports Phys Ther 2012;42:1032–1038. https://pubmed.ncbi.nlm.nih.gov/23346556/
35. The impact of minor crank length adjustments on lower body cycling kinematics. Sports Biomech 2026;25:454–467. https://doi.org/10.1080/14763141.2025.2511755
36. Li J, et al. Effects of crank length on cycling efficiency, sprint performance, and perceived fatigue in high-level amateur road cyclists. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12060448
37. Is shorter faster? The geometry and aerodynamics of crank length in cycling. bioRxiv preprint 2026. https://www.biorxiv.org/content/10.64898/2026.09.25.754230v1.full
38. Martin JC, Malina RM, Spirduso WW. Effects of crank length on maximal cycling power and optimal pedaling rate of boys aged 8–11 years. Eur J Appl Physiol 2002;86:215–217. https://link.springer.com/article/10.1007/s00421-001-0525-9
39. Clothing condition does not affect meaningful clinical interpretation in markerless motion capture. J Biomech 2022. https://pubmed.ncbi.nlm.nih.gov/35749889/
40. SILCA — Aero testing at home: a guide to the Chung method. https://silca.cc/blogs/silca/chunging-with-robert-chung
41. MyVeloFit — Setup instructions (distance, height, clothing, warm-up, cadence). https://www.myvelofit.com/fit-academy/myvelofit-setup-instructions/

---

**Three claims this report must NOT make**

1. **"The app measures CdA / predicts watts saved / estimates speed."** It measures a 2-D silhouette share; area explained 54% of drag-area variance [26] and smaller area did not always mean less drag [27]. `watts_delta_est` is a Cd = 0.80 scenario, not a measurement.
2. **"Fitting inside these bands prevents injury or treats knee pain."** Most advocated adjustments are untested [33]; saddle-height evidence is strong for the *method*, weak for injury outcomes [31]. Bike fit is not a medical device.
3. **"A hip angle below 40° costs 5–15% power" / "Holmes 25–35° is validated."** The 5–15% figure comes from an unsourced vendor claim [23]; Holmes is a static goniometer convention that reads ~8° different from dynamic BDC [3]. Neither is a validated threshold.
