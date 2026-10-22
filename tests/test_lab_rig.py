"""Lab-rig preset + capture-size request.

Covers the two 2026-10-22 additions:
  * `cameras.open_source(width=, height=)` asks DirectShow for a mode BEFORE the
    first read (the front RealSense used to open at 640x480 by default).
  * the one-click rig button that finds the Femto RGB (side) and the RealSense
    RGB (front) by NAME and opens both in one session.

No camera is opened here: cv2 is faked, and the page is only rendered.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod      # noqa: E402
import cameras            # noqa: E402


class FakeCap:
    """Minimal cv2.VideoCapture double: opens, honours set(), yields one frame."""

    def __init__(self, *a, **kw):
        self.sets = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        return True, object()

    def release(self):
        self.released = True

    def get(self, prop):
        return 0


class TestCaptureSizeRequest(unittest.TestCase):
    def _open(self, **kw):
        caps = []

        def factory(*a, **k):
            cap = FakeCap()
            caps.append(cap)
            return cap

        with mock.patch.object(cameras.cv2, "VideoCapture", factory):
            cap = cameras.open_source("usb", index=4, **kw)
        return cap, caps

    def test_requested_size_is_set_before_the_first_read(self):
        cap, caps = self._open(width=1280, height=720)
        self.assertIsNotNone(cap)
        self.assertEqual(cap, caps[0])
        sets = caps[0].sets
        self.assertIn((cameras.cv2.CAP_PROP_FRAME_WIDTH, 1280), sets)
        self.assertIn((cameras.cv2.CAP_PROP_FRAME_HEIGHT, 720), sets)

    def test_no_size_means_no_size_calls(self):
        cap, caps = self._open()
        self.assertIsNotNone(cap)
        sizes = [s for s in caps[0].sets
                 if s[0] in (cameras.cv2.CAP_PROP_FRAME_WIDTH, cameras.cv2.CAP_PROP_FRAME_HEIGHT)]
        self.assertEqual(sizes, [])

    def test_source_label_tolerates_a_size_carrying_source(self):
        # the open payload now can include width/height and is splatted into
        # source_label — it must not raise
        label = cameras.source_label(kind="usb", index=2, width=1280, height=720)
        self.assertEqual(label, "USB camera 2")


class TestLabRigUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config["TESTING"] = True
        r = appmod.app.test_client().get("/")
        cls.status = r.status_code
        cls.page = r.get_data(as_text=True)

    def test_page_renders(self):
        self.assertEqual(self.status, 200)

    def test_rig_button_and_handler_ship_in_the_page(self):
        for marker in ('id="btnLabRig"', "labRigSetup()", "async function labRigSetup"):
            self.assertIn(marker, self.page)

    def test_rig_writes_the_payload_shape_the_backend_expects(self):
        # the handler must post source2 + plane2 to the documented endpoint
        start = self.page.index("async function labRigSetup")
        block = self.page[start:start + 2600]
        self.assertIn('"/api/camera/open"', block)
        self.assertIn("source2", block)
        self.assertIn('plane2: "frontal"', block)
        self.assertIn("width: 1280", block)
        # must look cameras up by name, never by a hardcoded index
        self.assertIn("orbbec|femto", block)
        self.assertIn("realsense", block)

    def test_rig_syncs_the_form_after_opening(self):
        start = self.page.index("async function labRigSetup")
        block = self.page[start:start + 2600]
        for marker in ('$("usbIndex").value = side.index',
                       '$("useCam2").checked = true',
                       '$("usb2Index").value = front.index',
                       'setPlane2("frontal")'):
            self.assertIn(marker, block)


if __name__ == "__main__":
    unittest.main()
