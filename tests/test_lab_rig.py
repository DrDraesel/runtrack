"""Lab-rig preset + capture-size request.

Covers the two 2026-10-22 additions:
  * `cameras.open_source(width=, height=)` asks DirectShow for a mode BEFORE the
    first read (the front RealSense used to open at 640x480 by default).
  * the one-click rig button that finds the colour cameras by NAME and opens up
    to three in one session (best sensor on the side, second on the front,
    third on the rear), requesting a size per model.

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
        # the handler must post source2/source3 + plane2/plane3 to the endpoint
        start = self.page.index("async function labRigSetup")
        end = self.page.index("async function labRigDepth")
        block = self.page[start:end]
        self.assertIn('"/api/camera/open"', block)
        self.assertIn("source2", block)
        self.assertIn("source3", block)
        self.assertIn('plane2 = "frontal"', block)
        self.assertIn('plane3 = "frontal"', block)
        self.assertIn('direction3 = "rear"', block)
        # a size is requested per model, never one hardcoded pair for every camera
        self.assertIn("1920", block)
        self.assertIn("1080", block)
        self.assertIn("sizeFor", block)
        # must look cameras up by name, never by a hardcoded index
        for name_re in ("/azure/i", "/orbbec|femto/i", "/realsense/i"):
            self.assertIn(name_re, block)

    def test_rig_puts_the_best_sensor_on_the_side(self):
        start = self.page.index("async function labRigSetup")
        end = self.page.index("async function labRigDepth")
        block = self.page[start:end]
        # the Azure is the highest-resolution colour stream, so it takes the
        # side view (which carries most of the metrics) whenever it is present
        self.assertIn('const side = pick(/azure/i) || pick(/orbbec|femto/i)', block)

    def test_rig_syncs_the_form_after_opening(self):
        start = self.page.index("async function labRigSetup")
        end = self.page.index("async function labRigDepth")
        block = self.page[start:end]
        for marker in ('$("usbIndex").value = side.index',
                       '$("useCam2").checked = true',
                       '$("usb2Index").value = front.index',
                       'setPlane2("frontal")',
                       '$("useCam3").checked = true',
                       '$("usb3Index").value = rear.index',
                       'setDirection3("rear")'):
            self.assertIn(marker, block)


class TestCameraRolesUI(unittest.TestCase):
    """Scan → recommend → activate, with a per-camera side/front/rear picker.

    Roles are remembered by camera NAME (DirectShow indices move on replug), an
    explicit choice beats the recommendation, and Start opens every assigned
    camera in one call. Only the rendered page is inspected here — no camera is
    opened.
    """

    @classmethod
    def setUpClass(cls):
        appmod.app.config["TESTING"] = True
        r = appmod.app.test_client().get("/")
        cls.status = r.status_code
        cls.page = r.get_data(as_text=True)

    def _block(self, start_marker, end_marker):
        start = self.page.index(start_marker)
        end = self.page.index(end_marker, start)
        return self.page[start:end]

    def test_scan_button_and_role_panel_ship_in_the_page(self):
        self.assertEqual(self.status, 200)
        for marker in ('id="btnScanAssign"', "scanAndAssign()", 'id="rolePanel"', 'id="roleRows"',
                       'id="roleHint"', "function renderRolePanel", "async function openAssigned",
                       "⚡ Activate all assigned"):
            self.assertIn(marker, self.page)

    def test_roles_are_persisted_per_camera_name(self):
        block = self._block("function loadCamRoles", "function isColourCam")
        self.assertIn("localStorage", block)
        self.assertIn("runtrack.cameraRoles", self.page)
        store = self._block("function setRole(", "function forgetRoles")
        self.assertIn("camRoles[cam.name] = role", store)
        self.assertIn("saveCamRoles()", store)
        # assigning a role to one camera must free it from any other camera
        self.assertIn("delete camRoles[name]", store)

    def test_recommendation_puts_the_best_sensor_on_the_side(self):
        block = self._block("function recommendRoles", "function resolveAssignments")
        self.assertIn("pick(/azure/i) || pick(/orbbec|femto/i)", block)
        self.assertIn('out[side.name] = "side"', block)
        self.assertIn('out[front.name] = "front"', block)
        self.assertIn('out[rear.name] = "rear"', block)

    def test_explicit_role_beats_the_recommendation(self):
        block = self._block("function resolveAssignments", "function planRig")
        self.assertIn("camRoles[c.name]", block)
        self.assertIn("used.has(r)", block)
        # an explicit "off" must suppress the recommendation for that camera
        self.assertIn('camRoles[c.name] === "off"', block)

    def test_front_is_view_two_and_rear_is_view_three(self):
        block = self._block("async function openAssigned", "function rigMatches")
        self.assertIn('"/api/camera/open"', block)
        self.assertIn("payload.source2", block)
        self.assertIn('payload.plane2 = "frontal"', block)
        self.assertIn("payload.source3", block)
        self.assertIn('payload.plane3 = "frontal"', block)
        self.assertIn('payload.direction3 = "rear"', block)
        # a size is requested per model, never one hardcoded pair for every camera
        self.assertIn("sizeForCam(plan.primary)", block)

    def test_start_activates_every_assigned_camera(self):
        block = self._block("async function startSession", "async function stopSession")
        self.assertIn("rigMatches(plan)", block)
        self.assertIn("await openAssigned()", block)
        # file / URL sources must keep the old single-source path
        self.assertIn("await openCamera()", block)
        self.assertIn('$("camKind").value === "usb"', block)

    def test_rig_matches_compares_open_views_with_the_assignment(self):
        block = self._block("function rigMatches", "function setDiscipline")
        self.assertIn("has_camera2", block)
        self.assertIn("has_camera3", block)
        self.assertIn('"USB camera " + plan.primary.index', block)

    def test_depth_and_ir_pins_are_never_assignable(self):
        block = self._block("function isColourCam", "function sizeForCam")
        self.assertIn("c.non_video", block)
        self.assertIn("depth|infrared", block)


if __name__ == "__main__":
    unittest.main()
