"""Session stills: the report must ALWAYS carry pictures.

Regression behind this file: a session with bad framing latched zero gait events,
so `keyframes` was empty and the printed report had no image of the runner at
all. The still grabber keeps a periodic frame independent of events, and the
report renders it in its own "Session stills" section.

No camera here: frames are synthesised, the DB is a temp file.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import gait_events as GA  # noqa: E402
import metrics as M  # noqa: E402
import report as R  # noqa: E402
from session_store import Store  # noqa: E402


def fake_jpeg(w=96, h=64, shade=128):
    img = np.full((h, w, 3), shade, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def synth_rows(n=240, fps=30.0, cadence_spm=180.0, shin=3.0, osc=0.02):
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        rows.append({
            "t_rel": t,
            "ankle_y_l": 0.90 + 0.05 * math.sin(ph),
            "ankle_y_r": 0.90 + 0.05 * math.sin(ph + math.pi),
            "ankle_x_l": 0.30, "ankle_x_r": 0.62,
            "hip_y": 0.50, "hip_x": 0.45,
            "shoulder_y": 0.22, "shoulder_x": 0.45,
            "head_y": 0.10, "head_x": 0.45,
            "shin_angle_l": shin, "shin_angle_r": -shin,
            "left_knee": 160.0, "right_knee": 160.0,
            "left_hip": 170.0, "right_hip": 170.0,
            "vertical_osc_norm": osc,
            "pose_found": 1.0,
        })
    return rows


class TestStillCollector(unittest.TestCase):
    def test_keeps_one_frame_per_interval(self):
        c = GA.StillCollector(interval_s=5.0, max_total=10)
        kept = [c.push(t, fake_jpeg()) for t in (0.0, 1.0, 4.9, 5.0, 9.0)]
        self.assertEqual(kept, [True, False, False, True, False])
        self.assertEqual([round(i["t"], 1) for i in c.items], [0.0, 5.0])

    def test_no_jpeg_and_zero_cap_are_refused(self):
        c = GA.StillCollector(interval_s=5.0, max_total=3)
        self.assertFalse(c.push(0.0, None))
        self.assertEqual(c.items, [])
        z = GA.StillCollector(interval_s=5.0, max_total=0)
        self.assertFalse(z.push(0.0, fake_jpeg()))

    def test_cap_thins_evenly_instead_of_clustering_at_the_end(self):
        c = GA.StillCollector(interval_s=1.0, max_total=3)
        for t in range(40):
            c.push(float(t), fake_jpeg())
        self.assertLessEqual(len(c.items), 3)
        times = [i["t"] for i in c.items]
        # spread over the session, not the last three seconds
        self.assertLess(times[0], 10.0)
        self.assertGreater(times[-1], 20.0)

    def test_save_writes_files_with_zero_gait_events(self):
        c = GA.StillCollector(interval_s=5.0, max_total=2)
        c.push(0.0, fake_jpeg(), lms=None, fm={"left_knee": 150.0, "right_knee": 148.0})
        c.push(6.0, fake_jpeg(shade=60), lms=None, fm={})
        with tempfile.TemporaryDirectory() as tmp:
            meta = c.save(Path(tmp), ctx={"cadence_spm": 172, "trunk_lean_fwd_deg": 8.2})
            self.assertEqual(len(meta), 2)
            for m in meta:
                self.assertEqual(m["kind"], "periodic")
                self.assertTrue((Path(tmp) / m["file"]).exists(), m["file"])
                self.assertTrue(m["file"].endswith(".jpg"))
            self.assertEqual(meta[0]["knee_l_deg"], 150.0)

    def test_save_is_a_no_op_when_disabled_or_empty(self):
        empty = GA.StillCollector(interval_s=5.0, max_total=3)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(empty.save(Path(tmp)), [])
        full = GA.StillCollector(interval_s=5.0, max_total=3)
        full.push(0.0, fake_jpeg())
        with mock.patch.object(config, "STILLS_ALWAYS", False):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(full.save(Path(tmp)), [])

    def test_caption_scale_shrinks_so_long_lines_never_clip(self):
        # a caption wider than the frame used to run off the image edge
        self.assertLess(GA._fit_scale(["x" * 120], 200), 0.52)
        self.assertGreaterEqual(GA._fit_scale(["short line"], 200), 0.52)


class ReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "t.db", sessions_dir=self.root / "sessions")
        self._cfg = mock.patch.multiple(config, SESSIONS_DIR=self.root / "sessions")
        self._cfg.start()

    def tearDown(self):
        self._cfg.stop()
        self.tmp.cleanup()

    def seed(self, summary_extra=None, stills_on_disk=0, stills_meta=None):
        rows = synth_rows()
        summary = M.analyze_session(rows, weight_kg=75, height_cm=178,
                                    speed_kmh=10, view_plane="sagittal")
        summary.update(summary_extra or {})
        pid = self.store.upsert_patient("Ann", "111")
        sid = self.store.create_session(pid, {"weight_kg": 75, "height_cm": 178,
                                              "speed_kmh": 10}, "test source",
                                        view_plane="sagittal")
        self.store.save_csv(sid, rows)
        self.store.finish_session(sid, summary.get("duration_s"), len(rows), summary,
                                  status="complete")
        if stills_on_disk or stills_meta:
            kdir = self.root / "sessions" / str(sid) / config.KEYFRAME_DIR
            kdir.mkdir(parents=True, exist_ok=True)
            for i in range(stills_on_disk):
                cv2.imwrite(str(kdir / f"still_{i:02d}_t{i * 5:05.2f}.jpg".replace(" ", "0")),
                            np.full((64, 96, 3), 90, dtype=np.uint8))
            for m in stills_meta or []:
                cv2.imwrite(str(self.root / "sessions" / str(sid) / m["file"]),
                            np.full((64, 96, 3), 70, dtype=np.uint8))
        return sid


class TestReportRendersStills(ReportFixture):
    def test_stills_section_always_present(self):
        sid = self.seed()
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("Session stills (0)", html)
        self.assertIn("Periodic freeze-frames", html)

    def test_still_images_appear_and_are_not_event_keyframes(self):
        sid = self.seed(stills_on_disk=2)
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("Session stills (2)", html)
        self.assertIn("still_00", html)
        # the disk fallback must not promote stills into the event keyframe set
        self.assertIn("Gait event keyframes (0)", html)

    def test_hud_snapshots_are_the_last_resort_for_old_sessions(self):
        sid = self.seed()
        out = self.root / "sessions" / str(sid)
        for i in range(2):
            cv2.imwrite(str(out / f"frame_{i}.jpg"), np.full((64, 96, 3), 120, dtype=np.uint8))
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("Session stills (2)", html)
        self.assertIn("Session snapshot (live HUD)", html)
        self.assertIn("frame_0.jpg", html)

    def test_summary_stills_meta_renders_caption_with_metrics(self):
        meta = [{"file": f"{config.KEYFRAME_DIR}/still_00_t12.50.jpg", "t": 12.5,
                 "label": "Session still", "kind": "periodic",
                 "knee_l_deg": 152.0, "knee_r_deg": None, "pose_found": True}]
        sid = self.seed(summary_extra={"stills": meta}, stills_meta=meta)
        html = Path(R.build(sid, self.store)).read_text(encoding="utf-8")
        self.assertIn("Session stills (1)", html)
        self.assertIn("t = 12.50 s", html)
        self.assertIn("knee L 152°", html)


if __name__ == "__main__":
    unittest.main()
