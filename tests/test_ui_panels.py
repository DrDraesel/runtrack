"""Compact camera panel + Joint studies card (template markers only).

The Camera card used to be a wall of explanatory paragraphs (scan button, role
panel, import videos, two lab-rig presets, source blocks).  It is now:

    heading · one Scan button · one row per physical camera (status dot, name,
    role select, live fps) · a count of hidden depth/IR pins · Open rig ·
    Close cameras · one line saying what is actually open
    + a collapsed <details> holding every other source (file / phone / depth).

The right-hand column gained a Joint studies card: joint chips, a live angle
grid, one 90 s canvas drawn from the SAME /api/state poll, and an optional
labelled capture POSTed to /api/session/capture_angles.

Nothing here opens a camera or writes under data/: the page is rendered once
through Flask's test client and inspected as text, like tests/test_lab_rig.py.
"""
import re
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod  # noqa: E402


class PageCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config["TESTING"] = True
        r = appmod.app.test_client().get("/")
        cls.status = r.status_code
        cls.page = r.get_data(as_text=True)

    def markers(self, markers):
        missing = [m for m in markers if m not in self.page]
        self.assertEqual(missing, [], "missing from rendered page: %r" % (missing,))

    def block(self, start, end):
        i = self.page.index(start)
        j = self.page.index(end, i)
        return self.page[i:j]

    def details_block(self):
        i = self.page.index('<details class="adv" id="advSources">')
        j = self.page.index("</details>", i)
        return self.page[i:j]


class TestCameraPanel(PageCase):
    def test_page_renders(self):
        self.assertEqual(self.status, 200)
        self.assertIn("RunTrack", self.page)

    def test_compact_panel_ships(self):
        self.markers([
            'id="cameraCard"', 'id="btnScanAssign"', "🔍 Scan cameras",
            'id="cameraRows"', 'id="pinsHidden"',
            'id="btnOpenRig"', "⚡ Open rig", 'onclick="openAssigned()"',
            'id="btnCloseCams"', "■ Close cameras", 'onclick="closeCamera()"',
            'id="rigStatus"', 'id="roleHint"',
        ])

    def test_panel_is_compact_not_a_wall_of_paragraphs(self):
        """Everything above the Advanced block is the whole visible panel."""
        head = self.page[self.page.index('<div class="card" id="cameraCard">'):
                         self.page.index('<details class="adv" id="advSources">')]
        self.assertLess(len(head), 2000, "camera panel grew back into a wall of text")

    def test_verbose_paragraphs_are_gone(self):
        for gone in ("Finds every connected colour",
                     "Camera location — tell the app",
                     "Film the same run twice",
                     "One click: re-applies the",
                     "Same rig, but the RealSense is"):
            self.assertNotIn(gone, self.page, "old verbose camera paragraph still ships")
        self.assertIn("Scan first, pick where each", self.page)   # one short hint remains

    def test_one_row_per_camera_is_rendered_by_js(self):
        block = self.block("function camRoleSelect", "function camViewFor")
        self.assertIn('const box = $("cameraRows")', block)
        self.assertIn("for (const c of rgb)", block)
        self.assertIn("box.appendChild(row)", block)
        self.assertIn('document.createElement("select")', block)   # the role dropdown
        self.assertIn('for (const r of ["side", "front", "rear", "off"])', block)
        self.assertIn("camRoleSelect(c,", block)

    def test_hidden_depth_pins_are_reported_on_one_line(self):
        block = self.block("function camRoleSelect", "function camViewFor")
        self.assertIn("const pins = all.length - rgb.length", block)
        self.assertIn('$("pinsHidden")', block)
        self.assertIn("depth/IR pin", block)

    def test_status_dot_is_green_amber_or_grey(self):
        block = self.block("function updateCameraRows", "function camfpsClass")
        # green = open and streaming above 5 fps, amber = open but slow/stalled,
        # grey = closed (no class)
        self.assertIn("v.fps > 5", block)
        self.assertIn('good ? " ok" : (v ? " warn" : "")', block)
        self.assertRegex(self.page, r"\.camdot\.ok\s*\{[^}]*#4ade80")
        self.assertRegex(self.page, r"\.camdot\.warn\s*\{[^}]*#fbbf24")

    def test_status_line_reports_what_is_open(self):
        block = self.block("function openViewLine", "function updateCameraRows")
        self.assertIn("No camera open.", block)
        self.assertIn("roleForLabel(v.label)", block)   # "side" / "front" / "rear"
        for key in ("st.source2", "st.source3", "st.plane2", "st.plane3"):
            self.assertIn(key, block)
        # the documented shape: role · plane + size per view
        self.assertIn("side · sagittal 1920×1080 · front · frontal 1280×720", self.page)
        self.assertIn("sz.width + \"×\" + sz.height", block)   # requested per model
        self.assertIn('$("rigStatus")', self.page)

    def test_rows_update_without_a_second_poll(self):
        poll = self.block("async function poll", "/* --------------------------------------------------- audio")
        self.assertIn("updateCameraRows()", poll)
        self.assertIn("updateJointStudy(d)", poll)
        self.assertEqual(self.page.count("setInterval(poll"), 1)

    def test_open_and_close_keep_the_panel_line_current(self):
        opened = self.block("async function openAssigned", "function rigMatches")
        self.assertIn('$("rigStatus")', opened)
        close = self.block("async function closeCamera", "/* ------------------------------------------------------------ playback")
        self.assertIn('if ($("rigStatus"))', close)


class TestAdvancedSources(PageCase):
    def test_advanced_details_holds_every_other_source(self):
        block = self.details_block()
        for marker in ('id="camKind"', "RealSense depth", 'id="usbIndex"',
                       'id="urlInput"', 'id="fileInput"', 'id="useCam2"',
                       'id="useCam3"', 'id="btnDualVid"', 'id="btnLabRig"',
                       'id="btnLabRigDepth"', 'id="rigCard"'):
            self.assertIn(marker, block)
        self.assertIn("Advanced — file / phone / depth sources", block)
        self.assertIn('type="file"', block)                     # video file upload
        self.assertIn("http://192.168.1.50:8080/video", block)  # phone URL
        self.assertIn('<option value="depth">', block)           # RealSense depth

    def test_advanced_is_collapsed_by_default(self):
        match = re.search(r"<details[^>]*advSources[^>]*>", self.page)
        self.assertIsNotNone(match)
        tag = match.group(0)
        self.assertNotIn(" open", tag)
        self.assertIn("advSources", tag)

    def test_advanced_sits_below_the_compact_panel(self):
        self.assertLess(self.page.index('id="cameraRows"'),
                        self.page.index("Advanced — file / phone / depth sources"))

    def test_old_sources_are_no_longer_in_the_visible_panel(self):
        head = self.page[self.page.index('<div class="card" id="cameraCard">'):
                         self.page.index('<details class="adv" id="advSources">')]
        for moved in ('id="urlInput"', 'id="fileInput"', 'id="camKind"',
                      'id="btnLabRig"', 'id="btnDualVid"', 'id="usbIndex"'):
            self.assertNotIn(moved, head, "%s should live inside the Advanced block" % moved)

    def test_existing_handlers_are_still_wired(self):
        block = self.details_block()
        for handler in ('onchange="kindChanged()"', 'onclick="refreshCams()"',
                        'onclick="openCamera()"', 'onclick="closeCamera()"',
                        'onclick="dualVideoSetup()"', 'onclick="labRigSetup()"',
                        'onclick="labRigDepth()"', "onclick=\"setPlane2('frontal')\"",
                        "onclick=\"setDirection3('rear')\"", 'id="btn3Fro"',
                        "onclick=\"autoAssignCurrent()\"", "onclick=\"forgetRoles()\""):
            self.assertIn(handler, block)
        self.markers(["async function labRigSetup", "async function labRigDepth",
                      "async function openCamera", "function kindChanged"])


class TestJointStudies(PageCase):
    JOINTS = ("left_knee", "right_knee", "left_hip", "right_hip",
              "left_ankle", "right_ankle", "trunk_lean")

    def test_card_and_controls_ship(self):
        self.markers([
            'id="jointStudyCard"', "<h2>Joint studies</h2>",
            'id="jointChips"', 'id="jointGrid"', 'id="ch_joints"',
            'id="capLabel"', 'id="btnCapture"', "📌 Capture angles",
            'id="capList"', 'id="capCount"', 'id="capNote"',
            "/api/session/capture_angles",
        ])

    def test_one_chip_per_joint_with_the_expected_labels(self):
        chips = self.block('id="jointChips"', 'id="jointGrid"')
        for key in self.JOINTS:
            self.assertIn('data-joint="%s"' % key, chips)
        for label in ("LEFT KNEE", "RIGHT KNEE", "LEFT HIP", "RIGHT HIP",
                      "LEFT ANKLE", "RIGHT ANKLE", "TRUNK"):
            self.assertIn(label, chips)
        self.assertEqual(chips.count("data-joint="), 7)

    def test_both_knees_and_trunk_are_selected_by_default(self):
        chips = self.block('id="jointChips"', 'id="jointGrid"')
        found = re.findall(r'<button class="chip( on)?" data-joint="([a-z_]+)"', chips)
        on = {key for cls, key in found if cls.strip() == "on"}
        self.assertEqual(on, {"left_knee", "right_knee", "trunk_lean"})
        self.assertEqual({key for _, key in found}, set(self.JOINTS))

    def test_card_sits_below_the_metric_charts(self):
        charts = self.page.index('id="ch_trunk"')
        card = self.page.index('id="jointStudyCard"')
        chat = self.page.index("<h2>Live AI chat")
        self.assertLess(charts, card)
        self.assertLess(card, chat)

    def test_grid_colours_from_bands_only_when_one_exists(self):
        block = self.block("function updateJointValues", "function drawJointChart")
        self.assertIn("jointBand(j.key)", block)
        self.assertIn('probe >= band[0] && probe <= band[1] ? " ok" : " bad"', block)
        band = self.block("function jointBand", "function renderJointChips")
        self.assertIn("BANDS[JOINT_BAND_KEY[key] || key]", band)
        self.assertIn("Array.isArray(b)", band)

    def test_chart_reuses_the_existing_helpers_and_poll(self):
        block = self.block("function drawJointChart", "/* ------------------------------------------------------- angle capture")
        self.assertIn('drawChart("ch_joints"', block)
        self.assertIn("jointHist", block)
        charts = self.block("function drawCharts", "/* ------------------------------------------------------- joint studies")
        self.assertIn("drawJointChart()", charts)
        hist = self.block("function pushHist", "function setupCanvas")
        self.assertIn("jointHist[j.key].push(num(live[j.key]))", hist)
        self.assertEqual(self.page.count("setInterval(poll"), 1)

    def test_toggle_chips_rerender_grid_and_chart(self):
        block = self.block("function toggleJoint", "function renderJointGrid")
        for call in ("renderJointChips()", "renderJointGrid()", "drawJointChart()"):
            self.assertIn(call, block)
        self.assertIn("jointSel", block)

    def test_capture_posts_the_label_and_shows_the_record(self):
        block = self.block("async function captureAngles", "function updateJointStudy")
        self.assertIn('fetch("/api/session/capture_angles"', block)
        self.assertIn('method: "POST"', block)
        self.assertIn("JSON.stringify({label: label})", block)
        self.assertIn("d.capture", block)
        self.assertIn("addCapture(rec)", block)
        self.assertIn("catch (e)", block)              # never throws in the console
        self.assertIn("res.ok", block)

    def test_capture_is_disabled_without_a_running_session(self):
        match = re.search(r'<button[^>]*id="btnCapture"[^>]*>', self.page)
        self.assertIsNotNone(match)
        self.assertIn("disabled", match.group(0))
        block = self.block("function updateJointStudy", "function bandText")
        self.assertIn("const recording = !!(d && d.session)", block)
        self.assertIn("btn.disabled = !recording", block)
        self.assertIn("captureEnabled", block)

    def test_capture_count_comes_from_state(self):
        block = self.block("function updateJointStudy", "function bandText")
        self.assertIn("d.joint_study", block)
        self.assertIn('typeof js.captures === "number"', block)
        self.assertIn("js.last", block)

    def test_capture_list_renders_time_label_knee_and_hip(self):
        block = self.block("function renderCaptureList", "function addCapture")
        self.assertIn("rec.t", block)
        self.assertIn("rec.label", block)
        self.assertIn("capAnglesText(rec.angles)", block)
        pair = self.block("function capAnglesText", "function renderCaptureList")
        for joint in ("left_knee", "right_knee", "left_hip", "right_hip"):
            self.assertIn('"%s"' % joint, pair)
        self.assertIn("a.trunk_lean", pair)


class TestCameraContract(PageCase):
    def test_start_contract_is_still_referenced(self):
        self.markers([
            "scanCache", "camRoles", "let camRoles = loadCamRoles()",
            "runtrack.cameraRoles", "function resolveAssignments",
            "function planRig", "async function openAssigned", "function rigMatches",
            "async function scanAndAssign", 'id="startErr"',
        ])

    def test_start_still_auto_scans_and_activates_the_rig(self):
        block = self.block("async function startSession", "async function stopSession")
        self.assertIn('$("camKind").value === "usb"', block)
        self.assertIn("rigMatches(plan)", block)
        self.assertIn("await openAssigned()", block)
        self.assertIn("await openCamera()", block)
        self.assertIn('$("startErr").textContent', block)
        self.assertIn("rigMatches(plan)", self.page)

    def test_roles_are_still_persisted_per_camera_name(self):
        store = self.block("function setRole(", "function forgetRoles")
        self.assertIn("camRoles[cam.name] = role", store)
        self.assertIn("saveCamRoles()", store)
        self.assertIn("delete camRoles[name]", store)
        load = self.block("function loadCamRoles", "function isColourCam")
        self.assertIn("localStorage", load)
        self.assertIn("ROLE_KEY", load)

    def test_plan_and_open_keep_the_view_mapping_start_depends_on(self):
        plan = self.block("function resolveAssignments", "function planRig")
        self.assertIn("camRoles[c.name]", plan)
        self.assertIn("used.has(r)", plan)
        opened = self.block("async function openAssigned", "function rigMatches")
        self.assertIn('"/api/camera/open"', opened)
        self.assertIn("payload.source2", opened)
        self.assertIn('payload.plane2 = "frontal"', opened)
        self.assertIn("payload.source3", opened)
        self.assertIn('payload.direction3 = "rear"', opened)


if __name__ == "__main__":
    unittest.main()
