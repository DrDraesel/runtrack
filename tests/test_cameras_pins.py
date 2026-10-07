"""Femto Mega Depth/IR pin guard: those pins open but carry no colour video."""
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import cameras            # noqa: E402


class TestNonVideoPinReason(unittest.TestCase):
    def test_depth_pin_rejected_with_reason(self):
        reason = cameras.non_video_pin_reason("Orbbec Femto Mega Depth Camera")
        self.assertIsNotNone(reason)
        self.assertIn("DEPTH", reason)
        self.assertIn("RGB", reason)

    def test_ir_pin_rejected_with_reason(self):
        reason = cameras.non_video_pin_reason("Orbbec Femto Mega IR Camera")
        self.assertIsNotNone(reason)
        self.assertIn("IR", reason)
        self.assertIn("RGB", reason)

    def test_rgb_and_other_cameras_pass(self):
        self.assertIsNone(cameras.non_video_pin_reason("Orbbec Femto Mega RGB Camera"))
        self.assertIsNone(cameras.non_video_pin_reason("Logitech C920"))
        self.assertIsNone(cameras.non_video_pin_reason("USB2.0 HD UVC WebCam"))
        self.assertIsNone(cameras.non_video_pin_reason(""))


class TestOpenGuard(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()
        self._saved = appmod._CAM_CACHE.get("cams")
        appmod._CAM_CACHE["cams"] = [
            {"index": 0, "name": "Orbbec Femto Mega Depth Camera", "width": 640, "height": 576},
            {"index": 1, "name": "Orbbec Femto Mega IR Camera", "width": 320, "height": 288},
            {"index": 2, "name": "Orbbec Femto Mega RGB Camera", "width": 1280, "height": 720},
        ]

    def tearDown(self):
        appmod._CAM_CACHE["cams"] = self._saved

    def test_open_depth_pin_refused(self):
        r = self.c.post("/api/camera/open", json={"kind": "usb", "index": 0})
        self.assertEqual(r.status_code, 400)
        self.assertIn("DEPTH", r.get_json()["error"])

    def test_open_ir_pin_refused(self):
        r = self.c.post("/api/camera/open", json={"kind": "usb", "index": 1})
        self.assertEqual(r.status_code, 400)
        self.assertIn("IR", r.get_json()["error"])

    def test_second_camera_depth_pin_refused(self):
        r = self.c.post("/api/camera/open", json={
            "kind": "usb", "index": 2,
            "source2": {"kind": "usb", "index": 0}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("second camera", r.get_json()["error"])

    def test_guard_fails_open_without_scan(self):
        appmod._CAM_CACHE["cams"] = None
        r = self.c.post("/api/camera/open", json={
            "kind": "url", "url": "http://127.0.0.1:9/none"})
        # not a 400 from the pin guard; opening itself may fail, but the guard
        # must not refuse when no scan has populated the cache
        self.assertNotIn("DEPTH", (r.get_json() or {}).get("error", ""))


class TestScanSkipsNonVideoPins(unittest.TestCase):
    """The scanner must never OPEN depth/IR pins: opening the depth pin wedges
    the UVC family and the RGB pin then refuses to open."""

    def test_depth_and_ir_never_probed(self):
        probed = []

        def fake_probe(index, backend):
            probed.append(index)
            if index == 2:
                return {"index": 2, "backend": "dshow", "width": 1280, "height": 720, "fps": 30.0}
            return None

        saved_names = cameras.device_names
        saved_probe = cameras._probe_with
        cameras.device_names = lambda: ["Orbbec Femto Mega Depth Camera",
                                        "Orbbec Femto Mega IR Camera",
                                        "Orbbec Femto Mega RGB Camera"]
        cameras._probe_with = fake_probe
        try:
            found = cameras.list_usb_cameras()
        finally:
            cameras.device_names = saved_names
            cameras._probe_with = saved_probe
        self.assertEqual(probed, [2])                      # depth/IR untouched
        by_index = {f["index"]: f for f in found}
        self.assertTrue(by_index[0].get("non_video"))
        self.assertTrue(by_index[1].get("non_video"))
        self.assertIn("DEPTH", by_index[0]["non_video"])
        self.assertEqual(by_index[2]["width"], 1280)

    def test_scan_still_lists_real_cameras_without_names(self):
        def fake_probe(index, backend):
            if index == 0:
                return {"index": 0, "backend": "dshow", "width": 640, "height": 480, "fps": 30.0}
            return None

        saved_names = cameras.device_names
        saved_probe = cameras._probe_with
        cameras.device_names = lambda: None
        cameras._probe_with = fake_probe
        try:
            found = cameras.list_usb_cameras()
        finally:
            cameras.device_names = saved_names
            cameras._probe_with = saved_probe
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["index"], 0)

    def test_nameless_scan_never_opens_index_0_when_others_work(self):
        """Index 0 is the Femto depth pin on this hardware: opening it wedges
        the UVC family in-process.  With no names, 0 must be tried LAST and
        only when nothing else was found."""
        probed = []

        def fake_probe(index, backend):
            probed.append(index)
            if index == 2:
                return {"index": 2, "backend": "dshow", "width": 1280, "height": 720, "fps": 30.0}
            return None

        saved_names = cameras.device_names
        saved_probe = cameras._probe_with
        cameras.device_names = lambda: None
        cameras._probe_with = fake_probe
        try:
            found = cameras.list_usb_cameras()
        finally:
            cameras.device_names = saved_names
            cameras._probe_with = saved_probe
        self.assertEqual([f["index"] for f in found], [2])
        self.assertNotIn(0, probed)              # depth pin never touched
        self.assertEqual(probed[:2], [1, 2])     # non-zero indexes probed first

    def test_ffmpeg_device_name_parsing(self):
        sample = (
            '[in#0 @ 000001b9c4ff33c0] "Orbbec Femto Mega Depth Camera" (video)\n'
            '[in#0 @ 000001b9c4ff33c0]   Alternative name "@device_pnp_\\\\?\\\\usb#vid_2bc5"\n'
            '[in#0 @ 000001b9c4ff33c0] "Orbbec Femto Mega IR Camera" (video)\n'
            '[in#0 @ 000001b9c4ff33c0] "Orbbec Femto Mega RGB Camera" (video)\n'
            '[in#0 @ 000001b9c4ff33c0] "Headset Microphone (HK Onyx Studio 2)" (audio)\n'
            'Error opening input file dummy.\n')
        names = cameras._parse_ffmpeg_devices(sample)
        self.assertEqual(names, ["Orbbec Femto Mega Depth Camera",
                                 "Orbbec Femto Mega IR Camera",
                                 "Orbbec Femto Mega RGB Camera"])


if __name__ == "__main__":
    unittest.main()
