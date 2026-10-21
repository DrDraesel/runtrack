"""Third (front) camera: direction label, plane gating, API and summary wiring.

The third view mirrors the second view's contract: its own source, its own
plane, its own time base.  ``direction`` is a report label only — frontal
metric math is mirror-invariant, so front and rear rows must score identically.
"""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import config             # noqa: E402
import second_view as SV  # noqa: E402


def ms_events(rows, win=6):
    """Mid-stance events from local ankle-y minima (the latch's own criterion)."""
    out = []
    for side, key in (("left", "l_ankle_y"), ("right", "r_ankle_y")):
        y = [r.get(key) for r in rows]
        t = [r.get("t_rel") for r in rows]
        for i in range(win, len(rows) - win):
            v = y[i]
            if v is None:
                continue
            seg = [x for x in y[i - win:i + win + 1] if x is not None]
            if v == min(seg):
                out.append({"type": "MS", "side": side, "t": t[i], "i": i})
    return out


def frontal_rows(n=240, fps=30.0, knee_valgus=4.0, cadence_spm=180.0):
    """Front/rear-view (frontal-plane) rows: alternating legs, knee collapse."""
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        valgus = knee_valgus + 2.0 * math.sin(ph)
        rows.append({
            "t_rel": t,
            "l_ankle_y": 0.90 + 0.05 * math.sin(ph),
            "r_ankle_y": 0.90 + 0.05 * math.sin(ph + math.pi),
            "l_hip_y": 0.62, "r_hip_y": 0.62,
            "_midhip_y": 0.62, "_midhip_x": 0.50,
            "_midsh_x": 0.50, "_midsh_y": 0.38, "_facing": 1.0,
            "l_hip_x": 0.48, "r_hip_x": 0.52,
            "l_knee_x": 0.47, "r_knee_x": 0.53,
            "l_ankle_x": 0.47, "r_ankle_x": 0.53,
            "l_sh_x": 0.44, "r_sh_x": 0.56,
            "left_knee": 160.0, "right_knee": 160.0,
            "pelvic_tilt_deg": 3.0 * math.sin(ph),
            "knee_valgus_l": valgus, "knee_valgus_r": valgus,
        })
    return rows


def sagittal_rows(n=240, fps=30.0, cadence_spm=180.0, shin=3.0, osc=0.02):
    """Side-view rows (same generator style as test_second_view.sagittal_rows)."""
    f_step = cadence_spm / 60.0 / 2.0
    rows = []
    for i in range(n):
        t = i / fps
        ph = 2 * math.pi * f_step * t
        dx = 0.2 * math.tan(math.radians(shin))
        rows.append({
            "t_rel": t,
            "l_ankle_y": 0.90 + 0.05 * math.sin(ph),
            "r_ankle_y": 0.90 + 0.05 * math.sin(ph + math.pi),
            "l_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "r_hip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_y": 0.62 - 0.5 * osc * math.sin(2 * ph),
            "_midhip_x": 0.50, "_midsh_x": 0.51, "_midsh_y": 0.38, "_facing": 1.0,
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


PARAMS = {"weight_kg": 75, "height_cm": 178, "speed_kmh": 10}


class TestThirdViewAnalysis(unittest.TestCase):
    """SV.analyze(..., direction=...) is a label passthrough, not a sign flip."""

    def test_front_direction_round_trips(self):
        sv = SV.analyze(frontal_rows(), PARAMS, "frontal",
                        events=ms_events(frontal_rows()), direction="front")
        self.assertIsNotNone(sv)
        self.assertEqual(sv["direction"], "front")

    def test_rear_direction_round_trips(self):
        sv = SV.analyze(frontal_rows(), PARAMS, "frontal", direction="rear")
        self.assertIsNotNone(sv)
        self.assertEqual(sv["direction"], "rear")

    def test_unknown_direction_is_not_invented(self):
        for bad in ("sideways", "", None, 3):
            sv = SV.analyze(frontal_rows(), PARAMS, "frontal", direction=bad)
            self.assertIsNotNone(sv)
            self.assertNotIn("direction", sv)

    def test_second_view_calls_without_direction_are_unchanged(self):
        # existing call sites pass no direction: the key must stay absent
        sv = SV.analyze(frontal_rows(), PARAMS, "frontal")
        self.assertIsNotNone(sv)
        self.assertNotIn("direction", sv)

    def test_direction_never_changes_frontal_metrics(self):
        # mirror invariance: front and rear cameras read the same numbers
        a = SV.analyze(frontal_rows(), PARAMS, "frontal", direction="front")
        b = SV.analyze(frontal_rows(), PARAMS, "frontal", direction="rear")
        c = SV.analyze(frontal_rows(), PARAMS, "frontal")
        for key in ("knee_valgus_deg", "form_score", "view_plane"):
            self.assertEqual(a[key], b[key])
            self.assertEqual(a[key], c[key])
        self.assertEqual(a["form"]["knee_valgus_deg"],
                         b["form"]["knee_valgus_deg"])

    def test_too_few_rows_returns_none(self):
        rows = frontal_rows(n=SV.MIN_ROWS - 1)
        self.assertIsNone(SV.analyze(rows, PARAMS, "frontal", direction="front"))
        self.assertIsNone(SV.analyze(frontal_rows(n=10), PARAMS, "frontal"))

    def test_min_rows_boundary_is_usable(self):
        rows = frontal_rows(n=SV.MIN_ROWS)
        self.assertIsNotNone(SV.analyze(rows, PARAMS, "frontal",
                                        direction="front"))

    def test_plane_gating_frontal(self):
        rows = frontal_rows()
        sv = SV.analyze(rows, PARAMS, "frontal", events=ms_events(rows),
                        direction="front")
        self.assertEqual(sv["view_plane"], "frontal")
        self.assertIn("knee_valgus_deg", sv)
        self.assertIn("knee_valgus_deg", sv["form"])
        # a frontal camera never invents sagittal metrics
        self.assertNotIn("touchdown_shin_deg", sv)
        self.assertIn("touchdown_shin_deg", sv["not_assessed"])
        self.assertEqual(sv["form_score_detail"]["plane"], "frontal")

    def test_plane_gating_sagittal_third_view(self):
        sv = SV.analyze(sagittal_rows(), PARAMS, "sagittal", direction="front")
        self.assertEqual(sv["view_plane"], "sagittal")
        self.assertIn("touchdown_shin_deg", sv)
        self.assertNotIn("knee_valgus_deg", sv)
        self.assertEqual(sv["direction"], "front")

    def test_cycling_has_no_third_view_block(self):
        self.assertIsNone(SV.analyze(frontal_rows(), PARAMS, "frontal",
                                     discipline="cycling", direction="front"))

    def test_bad_plane_falls_back_to_default(self):
        sv = SV.analyze(frontal_rows(), PARAMS, "diagonal", direction="front")
        self.assertEqual(sv["view_plane"], config.DEFAULT_PLANE3)


class TestThirdViewConfig(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(config.DEFAULT_PLANE3, "frontal")
        self.assertEqual(config.DEFAULT_DIRECTION3, "front")
        self.assertEqual(config.DEFAULT_DIRECTION2, "rear")
        self.assertEqual(config.DIRECTIONS, ("front", "rear"))
        self.assertEqual(config.MAX_VIEWS, 3)

    def test_third_plane_is_a_real_view_plane(self):
        self.assertIn(config.DEFAULT_PLANE3, config.VIEW_PLANES)


class _AppCase(unittest.TestCase):
    """Base: isolate the global Live state so tests never leak into each other."""

    ATTRS = ("source3", "source3_label", "plane3", "direction3", "seg3",
             "latest_jpeg3", "live3", "fps3", "warn3", "source", "source_label",
             "active", "session", "latest_jpeg", "live", "fps")

    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()
        self._saved = {k: getattr(appmod.L, k) for k in self.ATTRS}
        self._cams = appmod._CAM_CACHE.get("cams")

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(appmod.L, k, v)
        appmod._CAM_CACHE["cams"] = self._cams


class TestThirdViewCameraOpen(_AppCase):
    def test_missing_file_reports_third_camera(self):
        r = self.c.post("/api/camera/open", json={
            "kind": "url", "url": "http://127.0.0.1:9/none",
            "source3": {"kind": "file", "path": "Z:/nope_front.mp4"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("third camera", r.get_json()["error"])
        self.assertIn("file not found", r.get_json()["error"])

    def test_bad_kind_reports_third_camera(self):
        r = self.c.post("/api/camera/open", json={
            "kind": "url", "url": "http://127.0.0.1:9/none",
            "source3": {"kind": "nope"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("third camera", r.get_json()["error"])

    def test_pin_guard_applies_to_third_camera(self):
        appmod._CAM_CACHE["cams"] = [
            {"index": 0, "name": "Orbbec Femto Mega Depth Camera"},
            {"index": 2, "name": "Orbbec Femto Mega RGB Camera"},
        ]
        r = self.c.post("/api/camera/open", json={
            "kind": "usb", "index": 2,
            "source3": {"kind": "usb", "index": 0}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("third camera", r.get_json()["error"])
        self.assertIn("DEPTH", r.get_json()["error"])

    def test_missing_primary_still_reports_primary_error(self):
        r = self.c.post("/api/camera/open", json={
            "kind": "file", "path": "Z:/nope.mp4",
            "source3": {"kind": "file", "path": "Z:/nope_front.mp4"}})
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("third camera", r.get_json()["error"])


class TestThirdViewState(_AppCase):
    def test_state_exposes_third_view_keys(self):
        d = self.c.get("/api/state").get_json()
        for key in ("source3", "warning3", "fps3", "live3", "plane3",
                    "direction3", "has_camera3"):
            self.assertIn(key, d)
        self.assertIsNone(d["warning3"])
        self.assertIsInstance(d["fps3"], (int, float))
        self.assertIsInstance(d["live3"], dict)

    def test_state_reflects_third_camera_metadata(self):
        with appmod.L.lock:
            appmod.L.source3 = {"kind": "file", "path": "front.mp4"}
            appmod.L.source3_label = "File front.mp4"
            appmod.L.plane3 = "sagittal"
            appmod.L.direction3 = "rear"
            appmod.L.fps3 = 12.5
            appmod.L.live3 = {"pose_found": True}
        d = self.c.get("/api/state").get_json()
        self.assertEqual(d["source3"], "File front.mp4")
        self.assertEqual(d["plane3"], "sagittal")
        self.assertEqual(d["direction3"], "rear")
        self.assertEqual(d["fps3"], 12.5)
        self.assertEqual(d["live3"], {"pose_found": True})
        self.assertTrue(d["has_camera3"])

    def test_close_clears_third_camera(self):
        with appmod.L.lock:
            appmod.L.source3 = {"kind": "file", "path": "front.mp4"}
            appmod.L.source3_label = "File front.mp4"
            appmod.L.latest_jpeg3 = b"jpeg"
            appmod.L.live3 = {"pose_found": True}
            appmod.L.fps3 = 9.0
        with mock.patch.object(appmod, "_finalize", return_value=None):
            r = self.c.post("/api/camera/close")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(appmod.L.source3)
        self.assertIsNone(appmod.L.latest_jpeg3)
        self.assertEqual(appmod.L.live3, {})
        d = self.c.get("/api/state").get_json()
        self.assertEqual(d["source3"], "none")
        self.assertFalse(d["has_camera3"])

    def test_reset_clears_third_camera_state(self):
        with appmod.L.lock:
            appmod.L.source3 = {"kind": "file", "path": "front.mp4"}
            appmod.L.source3_label = "File front.mp4"
            appmod.L.plane3 = "sagittal"
            appmod.L.direction3 = "rear"
            appmod.L.seg3 = True
            appmod.L.latest_jpeg3 = b"jpeg"
            appmod.L.live3 = {"pose_found": True}
            appmod.L.fps3 = 30.0
            appmod.L.warn3 = "stale warning"
            appmod.L.reset()
            self.assertIsNone(appmod.L.source3)
            self.assertEqual(appmod.L.source3_label, "none")
            self.assertEqual(appmod.L.plane3, config.DEFAULT_PLANE3)
            self.assertEqual(appmod.L.direction3, config.DEFAULT_DIRECTION3)
            self.assertFalse(appmod.L.seg3)
            self.assertIsNone(appmod.L.latest_jpeg3)
            self.assertEqual(appmod.L.live3, {})
            self.assertEqual(appmod.L.fps3, 0.0)
            self.assertIsNone(appmod.L.warn3)

    def test_video_feed3_route_registered(self):
        rules = {r.rule: r.endpoint for r in appmod.app.url_map.iter_rules()}
        self.assertIn("/video_feed3", rules)
        self.assertEqual(rules["/video_feed3"], "video_feed3")


class TestThirdViewSession(_AppCase):
    def test_third_label_joins_the_source_string(self):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "File side.mp4"
            appmod.L.source2 = {"kind": "file", "path": "rear.mp4"}
            appmod.L.source2_label = "File rear.mp4"
            appmod.L.plane2 = "frontal"
            appmod.L.source3 = {"kind": "file", "path": "front.mp4"}
            appmod.L.source3_label = "File front.mp4"
            appmod.L.plane3 = "sagittal"
            appmod.L.direction3 = "front"
            appmod.L.session = None
        with mock.patch.object(appmod.STORE, "upsert_patient", return_value=1), \
                mock.patch.object(appmod.STORE, "create_session",
                                  return_value=42) as create:
            r = self.c.post("/api/session/start",
                            json={"name": "Dana", "phone": "444"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(create.call_args[0][2],
                         "File side.mp4 + File rear.mp4 + File front.mp4")
        sess = appmod.L.session
        self.assertEqual(sess["plane3"], "sagittal")
        self.assertEqual(sess["direction3"], "front")
        self.assertEqual(sess["source3_label"], "File front.mp4")
        self.assertEqual(sess["rows3"], [])
        self.assertIsNone(sess["t0_video3"])
        self.assertIsNone(sess["t0_3"])

    def test_primary_only_session_keeps_plain_label(self):
        with appmod.L.lock:
            appmod.L.active = True
            appmod.L.source_label = "File side.mp4"
            appmod.L.source2 = None
            appmod.L.source2_label = "none"
            appmod.L.source3 = None
            appmod.L.source3_label = "none"
            appmod.L.session = None
        with mock.patch.object(appmod.STORE, "upsert_patient", return_value=1), \
                mock.patch.object(appmod.STORE, "create_session",
                                  return_value=43) as create:
            r = self.c.post("/api/session/start",
                            json={"name": "Dana", "phone": "444"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(create.call_args[0][2], "File side.mp4")


class TestThirdViewFinalize(unittest.TestCase):
    """_finalize builds summary["third_view"] + multi_view_sync (STORE mocked)."""

    def setUp(self):
        self._saved = appmod.L.session
        appmod.L.session = {
            "id": 7, "patient": {"name": "Dana", "phone": "444"},
            "rows": sagittal_rows(n=180), "params": dict(PARAMS),
            "elapsed": 6.0, "duration": 30.0, "view_plane": "sagittal",
            "discipline": "running", "bike_type": config.DEFAULT_BIKE_TYPE,
            "latch": None, "seeks": 0, "finalized": False, "finalizing": False,
            "source1_label": "File side.mp4",
            "plane2": "frontal", "source2_label": "File rear.mp4",
            "rows2": frontal_rows(n=120), "t0_video2": None, "t0_2": None,
            "plane3": "frontal", "direction3": "front",
            "source3_label": "File front.mp4",
            "rows3": frontal_rows(n=120), "t0_video3": None, "t0_3": None,
        }

    def tearDown(self):
        appmod.L.session = self._saved

    def test_summary_has_third_view_and_sync_block(self):
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as finish, \
                mock.patch.object(appmod.R, "build"):
            sid = appmod._finalize("stopped")
        self.assertEqual(sid, 7)
        summary = finish.call_args[0][3]
        tv = summary.get("third_view")
        self.assertIsNotNone(tv)
        self.assertEqual(tv["view_plane"], "frontal")
        self.assertEqual(tv["direction"], "front")
        self.assertEqual(tv["rows"], 120)
        self.assertEqual(summary["plane3"], "frontal")
        self.assertEqual(summary["direction3"], "front")
        self.assertEqual(summary["source3_label"], "File front.mp4")
        sync = summary["multi_view_sync"]
        self.assertEqual(sync["clocks"], "software")
        self.assertIn("no hardware genlock", sync["note"])
        roles = [v["role"] for v in sync["views"]]
        self.assertEqual(roles, ["primary", "second", "third"])
        self.assertEqual([v["rows"] for v in sync["views"]], [180, 120, 120])
        self.assertEqual(sync["views"][2]["label"], "File front.mp4")

    def test_without_second_or_third_only_primary_is_listed(self):
        with appmod.L.lock:
            appmod.L.session["rows2"] = []
            appmod.L.session["rows3"] = []
            appmod.L.session["source2_label"] = "none"
            appmod.L.session["source3_label"] = "none"
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as finish, \
                mock.patch.object(appmod.R, "build"):
            appmod._finalize("stopped")
        summary = finish.call_args[0][3]
        self.assertNotIn("third_view", summary)
        self.assertEqual([v["role"] for v in summary["multi_view_sync"]["views"]],
                         ["primary"])

    def test_cycling_gets_no_third_view_block(self):
        with appmod.L.lock:
            appmod.L.session["discipline"] = "cycling"
        with mock.patch.object(appmod.STORE, "save_csv"), \
                mock.patch.object(appmod.STORE, "finish_session") as finish, \
                mock.patch.object(appmod.R, "build"):
            appmod._finalize("stopped")
        summary = finish.call_args[0][3]
        self.assertNotIn("third_view", summary)


if __name__ == "__main__":
    unittest.main()
