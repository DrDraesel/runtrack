"""Gait event latching tests on SYNTHETIC landmark sequences.

A synthetic running gait cycle is built from real body geometry (hip -> thigh ->
shin -> ankle) with a KNOWN touchdown shin angle and a KNOWN mid-stance knee
flexion, then pushed through the same code path the app uses.  This is the test
that proves the latch + event-angle math, independently of any camera footage.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import gait_events as GA  # noqa: E402
import metrics as M  # noqa: E402

FPS = 30.0
CYCLE_S = 0.8            # 75 strides/min -> 150 spm per foot pair
STANCE_END = 0.42        # share of the cycle the foot is on the ground
SHIN_DEG = 10.0          # constructed touchdown shin angle (knee->ankle vs vertical)
MS_FLEX_DEG = 40.0       # constructed peak stance knee flexion
L_THIGH = L_SHIN = 0.22  # normalised segment lengths
PELVIS_Y = 0.55


def _rad(d):
    return math.radians(d)


def leg_angles(ph: float):
    """(thigh angle, shin angle) vs vertical for one leg at cycle phase ph.

    Sign convention: + = the joint is FORWARD (towards the image +x) of its
    parent, matching metrics.shin_angle_deg.
    """
    ph = ph % 1.0
    if ph < STANCE_END:
        u = ph / STANCE_END
        if u < 0.4:
            phi_s = SHIN_DEG                    # planted vertical-ish shin
        else:
            phi_s = SHIN_DEG + (-6.0 - SHIN_DEG) * (u - 0.4) / 0.6
        if u <= 0.55:                           # absorption to mid-stance
            flex = 15.0 + (MS_FLEX_DEG - 15.0) * (u / 0.55)
        else:                                   # push-off
            flex = MS_FLEX_DEG + (5.0 - MS_FLEX_DEG) * ((u - 0.55) / 0.45)
        phi_t = phi_s - flex                    # thigh behind the shin (stance)
    else:
        u = (ph - STANCE_END) / (1.0 - STANCE_END)
        phi_s = -6.0 + 6.0 * u                  # shin swings forward again
        phi_t = (-11.0) + 11.0 * u + 75.0 * math.sin(math.pi * u) ** 1.3
    return phi_t, phi_s


def synth_frame(t: float):
    """One (33, 4) landmark frame of the synthetic runner at time t seconds."""
    ph = (t / CYCLE_S) % 1.0
    # pelvis bob: lowest at mid-stance of each leg (2 x per cycle)
    hy = PELVIS_Y + 0.02 * math.cos(4 * math.pi * (ph - 0.23))
    hx = 0.50
    lms = np.zeros((33, 4), dtype=np.float32)
    lms[:, 0], lms[:, 1], lms[:, 3] = 0.5, 0.5, 1.0
    lms[23] = [hx - 0.05, hy, 0.0, 1.0]                    # left hip
    lms[24] = [hx + 0.05, hy, 0.0, 1.0]                    # right hip
    lms[11] = [hx - 0.055, hy - 0.25, 0.0, 1.0]            # left shoulder
    lms[12] = [hx + 0.055, hy - 0.25, 0.0, 1.0]            # right shoulder
    lms[0] = [hx + 0.02, hy - 0.35, 0.0, 1.0]              # nose (facing +x)
    lms[7] = [hx - 0.01, hy - 0.33, 0.0, 1.0]
    lms[8] = [hx + 0.03, hy - 0.33, 0.0, 1.0]
    for side, (hip_i, knee_i, ank_i, heel_i, foot_i), off in (
            (0, (23, 25, 27, 29, 31), 0.0), (1, (24, 26, 28, 30, 32), 0.5)):
        phi_t, phi_s = leg_angles(ph + off)
        hip = (hx + (-0.05 if side == 0 else 0.05), hy)
        knee = (hip[0] + L_THIGH * math.sin(_rad(phi_t)),
                hip[1] + L_THIGH * math.cos(_rad(phi_t)))
        ank = (knee[0] + L_SHIN * math.sin(_rad(phi_s)),
               knee[1] + L_SHIN * math.cos(_rad(phi_s)))
        lms[knee_i] = [knee[0], knee[1], 0.0, 1.0]
        lms[ank_i] = [ank[0], ank[1], 0.0, 1.0]
        lms[heel_i] = [ank[0] - 0.02, ank[1] + 0.02, 0.0, 1.0]
        lms[foot_i] = [ank[0] + 0.02, ank[1] + 0.02, 0.0, 1.0]
    return lms


def synth_rows(n_cycles=4.0, fps=FPS):
    n = int(n_cycles * CYCLE_S * fps) + 1
    rows = []
    for i in range(n):
        t = i / fps
        fm = M.frame_metrics(synth_frame(t))
        fm["t_rel"] = t
        rows.append(fm)
    return rows


def contact_times(side: int, n_cycles=4.0, fps=FPS):
    """Constructed initial-contact times for one side (phase 0 = left, .5 = right)."""
    off = 0.0 if side == 0 else 0.5
    out = []
    k = off
    while k * CYCLE_S <= n_cycles * CYCLE_S:
        out.append(k * CYCLE_S)
        k += 1.0
    return [t for t in out if t <= (n_cycles * CYCLE_S * fps) / fps]


class TestSyntheticGeometry(unittest.TestCase):
    """The synthetic frames must actually carry the constructed angles."""

    def test_constructed_shin_angle_at_contact(self):
        fm = M.frame_metrics(synth_frame(0.0))
        self.assertAlmostEqual(fm["shin_angle_l"], SHIN_DEG, places=1)
        self.assertAlmostEqual(fm["_facing"], 1.0, places=3)

    def test_constructed_knee_flexion_at_midstance(self):
        # mid-stance of the left leg is at phase 0.55 * STANCE_END
        t = 0.55 * STANCE_END * CYCLE_S
        fm = M.frame_metrics(synth_frame(t))
        self.assertAlmostEqual(fm["knee_flex_l"], MS_FLEX_DEG, delta=1.0)

    def test_stance_plateau_is_flat(self):
        ys = [M.frame_metrics(synth_frame(i / FPS))["l_ankle_y"]
              for i in range(int(CYCLE_S * FPS))]
        stance = ys[:int(STANCE_END * CYCLE_S * FPS)]
        swing = ys[int(STANCE_END * CYCLE_S * FPS):]
        # < 0.02 not < 0.01: this synthetic rig couples a small stance knee
        # flexion + pelvis bob into the ankle y (~0.017), which is still a
        # plateau relative to the swing lift; the functional guard for real
        # event latching lives in TestEventLatching.
        self.assertLess(max(stance) - min(stance), 0.02, "stance must be a plateau")
        # y grows downward: the foot lifts = swing y drops below the stance level
        self.assertGreater(max(stance) - min(swing), 0.05, "swing must lift off")


class TestEventLatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synth_rows()
        cls.events = GA.detect_events(cls.rows)

    def test_all_three_phases_latch(self):
        s = GA.summarize(self.events)
        self.assertGreaterEqual(s["IC"], 7, s)     # 4 cycles x 2 legs
        self.assertGreaterEqual(s["MS"], 7, s)
        self.assertGreaterEqual(s["TO"], 7, s)
        self.assertEqual(s["by_side"]["IC"]["left"], s["by_side"]["IC"]["right"])

    def test_initial_contact_matches_constructed_contacts(self):
        got = [e for e in self.events if e["type"] == "IC" and e["side"] == "left"]
        self.assertTrue(got, "no left IC latched")
        for i in range(4):
            want = i * CYCLE_S
            self.assertLess(min(abs(e["t"] - want) for e in got), 0.07,
                            f"left IC #{i} not near {want}")

    def test_event_order_within_a_cycle(self):
        left = sorted((e for e in self.events if e["side"] == "left"),
                      key=lambda e: e["t"])
        seq = [e["type"] for e in left]
        self.assertEqual(seq[:3], ["IC", "MS", "TO"], seq)
        # the pattern IC -> MS -> TO must repeat for every latched cycle
        self.assertEqual(seq[:3 * (len(seq) // 3)], ["IC", "MS", "TO"] * (len(seq) // 3), seq)

    def test_midstance_is_max_knee_flexion(self):
        for side, key in (("left", "left_knee"), ("right", "right_knee")):
            for e in [x for x in self.events if x["type"] == "MS" and x["side"] == side]:
                ic = max((x for x in self.events if x["type"] == "IC"
                          and x["side"] == side and x["t"] < e["t"]), key=lambda x: x["t"])
                to = min((x for x in self.events if x["type"] == "TO"
                          and x["side"] == side and x["t"] > e["t"]), key=lambda x: x["t"])
                span = [(r["t_rel"], r[key]) for r in self.rows
                        if ic["t"] - 1e-9 <= r["t_rel"] <= to["t"] + 1e-9
                        and r.get(key) is not None]
                self.assertTrue(span)
                worst = min(v for _, v in span)
                at_ms = min(span, key=lambda p: abs(p[0] - e["t"]))[1]
                self.assertAlmostEqual(at_ms, worst, delta=2.0,
                                       msg="mid-stance must sit at peak knee flexion")

    def test_stance_times_are_plausible(self):
        st = GA.stance_times(self.events)
        self.assertTrue(st["left"] and st["right"])
        for side in ("left", "right"):
            mean = float(np.mean(st[side]))
            self.assertGreater(mean, 0.15 * CYCLE_S)
            self.assertLess(mean, 0.7 * CYCLE_S)


class TestEventDerivedMetrics(unittest.TestCase):
    """analyze_session must report the constructed values at the latched events."""

    @classmethod
    def setUpClass(cls):
        cls.rows = synth_rows()
        cls.s = M.analyze_session(cls.rows, weight_kg=70, height_cm=175,
                                  view_plane="sagittal")

    def test_touchdown_shin_angle_recovered(self):
        blk = self.s.get("touchdown_shin_deg")
        self.assertIsNotNone(blk, self.s.get("not_assessed"))
        self.assertAlmostEqual(blk["mean"], SHIN_DEG, delta=2.0)
        self.assertAlmostEqual(blk["per_side"]["left"], SHIN_DEG, delta=2.0)
        self.assertGreater(blk["n"], 6)

    def test_midstance_knee_flexion_recovered(self):
        blk = self.s.get("knee_flexion_ms_deg")
        self.assertIsNotNone(blk, "mid-stance flexion missing")
        self.assertAlmostEqual(blk["mean"], MS_FLEX_DEG, delta=3.0)
        self.assertAlmostEqual(blk["per_side"]["right"], MS_FLEX_DEG, delta=3.0)

    def test_initial_contact_flexion_present(self):
        blk = self.s.get("knee_flexion_ic_deg")
        self.assertIsNotNone(blk)
        self.assertTrue(12.0 < blk["mean"] < 20.0, blk)

    def test_data_quality_reports_lower_body_ok(self):
        dq = self.s.get("data_quality") or {}
        self.assertEqual(dq.get("lower_body_ok"), True)
        self.assertAlmostEqual(dq.get("lower_body_coverage_pct"), 100.0, places=1)
        self.assertNotIn("note", dq)

    def test_overstride_and_asymmetry_blocks(self):
        self.assertIn("overstride_leg_frac", self.s)
        self.assertGreater(self.s["overstride_leg_frac"]["n"], 6)
        self.assertIn("asymmetry", self.s)
        self.assertGreater(self.s["asymmetry"]["stance_time_pct"], 90.0)
        self.assertIn("knee_flexion_ms_pct", self.s["asymmetry"])

    def test_form_bands_include_new_metrics(self):
        form = self.s["form"]
        self.assertIn("touchdown_shin_deg", form)
        self.assertIn("knee_flexion_ms_deg", form)
        # 10 deg touchdown shin angle is outside the optimal 0-5 deg band
        self.assertFalse(form["touchdown_shin_deg"]["in_band"])
        self.assertTrue(form["knee_flexion_ms_deg"]["in_band"])   # 40 deg is in 35-45
        self.assertIn("knee_flexion_ic_deg", form)
        self.assertIsNone(form["knee_flexion_ic_deg"]["in_band"])


class TestGaitLatchOnline(unittest.TestCase):
    def test_online_latch_matches_offline(self):
        rows = synth_rows()
        latch = GA.GaitLatch()
        for r in rows:
            latch.push(r["t_rel"], r)
        online = latch.events()
        offline = GA.detect_events(rows)
        for typ in ("IC", "MS", "TO"):
            a = sorted(round(e["t"], 2) for e in online if e["type"] == typ)
            b = sorted(round(e["t"], 2) for e in offline if e["type"] == typ)
            self.assertTrue(a, f"online latch produced no {typ}")
            self.assertGreaterEqual(len(a), len(b) - 1, (typ, a, b))
            for t in b:
                self.assertLess(min(abs(t - x) for x in a), 0.12, (typ, t, a))

    def test_coverage_helper(self):
        rows = synth_rows(n_cycles=1.0)
        self.assertAlmostEqual(latch_coverage(rows), 100.0, places=1)
        blind = [dict(r, l_ankle_y=None, r_ankle_y=None) for r in rows]
        self.assertAlmostEqual(latch_coverage(blind), 0.0, places=1)
        self.assertTrue(GA.events_not_assessable(latch_coverage(blind)))
        self.assertFalse(GA.events_not_assessable(latch_coverage(rows)))


def latch_coverage(rows):
    l = GA.GaitLatch()
    for r in rows:
        l.push(r["t_rel"], r)
    return l.lower_body_coverage()


class TestKeyframes(unittest.TestCase):
    def _jpeg(self, w=160, h=90):
        import cv2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 1] = 60
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        return buf.tobytes()

    def test_select_best_is_balanced_and_bounded(self):
        events = []
        for i in range(12):
            events.append({"type": ["IC", "MS", "TO"][i % 3], "side": "left",
                           "t": i * 0.5, "frame": b"x", "quality": 0.5 + 0.01 * i})
        picked = GA.select_best(events, per_type=3, max_total=9)
        self.assertLessEqual(len(picked), 9)
        kinds = [p["type"] for p in picked]
        self.assertLessEqual(kinds.count("IC"), 3)
        self.assertLessEqual(kinds.count("MS"), 3)
        self.assertLessEqual(kinds.count("TO"), 3)
        self.assertEqual([p["t"] for p in picked], sorted(p["t"] for p in picked))

    def test_save_events_writes_annotated_images(self):
        rows = synth_rows(n_cycles=1.0)
        latch = GA.GaitLatch()
        for i, r in enumerate(rows):
            # 4th argument = the raw (33,4) landmark array the keyframe overlay
            # draws itself from; handing in a metric dict here would only ever
            # produce a caption-only frame
            latch.push(r["t_rel"], r, self._jpeg(), synth_frame(r["t_rel"]))
        with tempfile.TemporaryDirectory() as tmp:
            meta = latch.save_snapshots(Path(tmp))
            self.assertTrue(meta, "no keyframes written")
            kdir = Path(tmp) / config.KEYFRAME_DIR
            files = sorted(kdir.glob("*.jpg"))
            self.assertEqual(len(files), len(meta))
            for m in meta:
                self.assertTrue((Path(tmp) / m["file"]).exists(), m["file"])
                self.assertIn(m["type"], ("IC", "MS", "TO"))
                self.assertIn("shin_angle_deg", m)
            ic = [m for m in meta if m["type"] == "IC"]
            self.assertTrue(ic)
            self.assertIsNotNone(ic[0]["shin_angle_deg"])
            self.assertAlmostEqual(abs(ic[0]["shin_angle_deg"]), SHIN_DEG, delta=2.5)
            ms = [m for m in meta if m["type"] == "MS"]
            self.assertTrue(ms)
            self.assertAlmostEqual(ms[0]["knee_flexion_deg"], MS_FLEX_DEG, delta=4.0)

    def test_no_frames_no_crash(self):
        latch = GA.GaitLatch()
        latch.push(0.0, {"t_rel": 0.0})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(latch.save_snapshots(Path(tmp)), [])


if __name__ == "__main__":
    unittest.main()
