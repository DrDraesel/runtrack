"""Third-view (front camera) UI: controls ship in the page and the open payload.

Pure template assertions — no camera is opened, nothing is written under data/.
The page is rendered once through Flask's test client so Jinja must stay valid.
"""
import re
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402


class TestThirdViewUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config["TESTING"] = True
        r = appmod.app.test_client().get("/")
        cls.status = r.status_code
        cls.page = r.get_data(as_text=True)

    def assertMarkers(self, markers):
        missing = [m for m in markers if m not in self.page]
        self.assertEqual(missing, [], "missing from rendered page: %r" % (missing,))

    def test_index_renders(self):
        self.assertEqual(self.status, 200)
        self.assertIn("RunTrack", self.page)

    def test_third_view_controls_present(self):
        self.assertMarkers([
            'id="useCam3"', 'id="cam3Box"', 'id="cam3Kind"', 'id="feed3"',
            'id="feed3note"', 'id="url3Input"', 'id="file3Input"', 'id="seg3"',
            'id="usb3Index"', 'id="cam3Pill"',
        ])
        # the block starts collapsed, like the second-view block
        self.assertRegex(self.page, r'id="cam3Box"[^>]*display:\s*none')

    def test_third_view_handlers_wired(self):
        self.assertMarkers([
            "function kind3Changed(", "function setPlane3(", "function setDirection3(",
            'id="btn3Fro"', 'id="btn3Sag"', 'id="btn3Front"', 'id="btn3Rear"',
            "setPlane3('frontal')", "setPlane3('sagittal')",
            "setDirection3('front')", "setDirection3('rear')",
            "kind3Changed();",  # boot call
        ])

    def test_third_view_checkbox_toggles_box(self):
        self.assertRegex(
            self.page,
            r'\$\("useCam3"\)\.addEventListener\("change"[\s\S]{0,160}cam3Box',
        )

    def test_open_payload_carries_third_source(self):
        self.assertMarkers([
            "payload.source3", "payload.plane3", "payload.direction3", "payload.seg3",
            "plane3Planned", "direction3Planned", 'p3.url', 'p3.index', 'p3.path',
        ])

    def test_third_feed_polling(self):
        self.assertMarkers(["/video_feed3", "has_camera3", "warning3", "fps3"])

    def test_second_view_still_present(self):
        """Regression: the third view must not displace the second-view controls."""
        self.assertMarkers([
            'id="useCam2"', 'id="cam2Box"', 'id="cam2Kind"', 'id="feed2"',
            'id="feed2note"', 'id="url2Input"', 'id="file2Input"', 'id="segSecond"',
            'id="btn2Fro"', 'id="btn2Sag"', "payload.source2", "plane2Planned",
            'id="segPrimary"', 'id="cam2Pill"',
        ])
        self.assertMarkers(["dualVideoSetup"])


if __name__ == "__main__":
    unittest.main()
