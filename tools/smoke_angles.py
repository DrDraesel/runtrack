"""End-to-end smoke test for the joint-angle capture lane.

Runs against a LIVE RunTrack server (start it with RUNTRACK_NO_BROWSER=1) and
drives the whole path with no camera involved:

    file source -> session start -> angle capture -> session end -> report

Checks, in order, and stops at the first hard failure:

1. server reachable, idle
2. file source opens (``assets/test_run.mp4`` by default)
3. session starts
4. rows accumulate (the capture loop is actually running)
5. ``POST /api/session/capture_angles`` stores a capture and it comes back in
   ``/api/state`` (``joint_study.captures``)  [WARN, not FAIL, when the clip
   shows no measurable joint: ``test_run.mp4`` is an upper-body close-up]
6. the session finalizes and ``summary.json`` carries ``joint_study`` /
   ``angle_captures``
7. the HTML report renders the new sections

Usage:
    .venv/Scripts/python.exe tools/smoke_angles.py [--base URL] [--clip PATH]
                                                   [--duration S] [--keep-open]

Exit code 0 = every hard check passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CLIP = REPO / "assets" / "test_run.mp4"


def _req(base, path, payload=None, timeout=30):
    url = f"{base.rstrip('/')}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:          # 400/403 are answers, not crashes
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def _text(base, path, timeout=30):
    with urllib.request.urlopen(f"{base.rstrip('/')}{path}", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


class Smoke:
    def __init__(self, base, clip, duration):
        self.base = base
        self.clip = clip
        self.duration = duration
        self.fails = 0
        self.warns = 0

    def ok(self, msg):
        print(f"  PASS  {msg}")

    def warn(self, msg):
        self.warns += 1
        print(f"  WARN  {msg}")

    def fail(self, msg):
        self.fails += 1
        print(f"  FAIL  {msg}")

    def check(self, cond, good, bad):
        self.ok(good) if cond else self.fail(bad)
        return cond

    # ---------------------------------------------------------------- steps
    def run(self):
        print(f"RunTrack angle-capture smoke against {self.base}")
        print(f"  clip: {self.clip}")

        status, state = _req(self.base, "/api/state")
        if not self.check(status == 200, "server answers /api/state",
                          f"/api/state returned HTTP {status}"):
            return
        if state.get("active"):
            print("  (an earlier source is open - closing it first)")
            _req(self.base, "/api/camera/close", {})
            time.sleep(1.5)

        # 1. file source -------------------------------------------------
        if not self.check(self.clip.exists(), "test clip present",
                          f"missing clip {self.clip}"):
            return
        status, body = _req(self.base, "/api/camera/open",
                            {"kind": "file", "path": str(self.clip),
                             "plane": "sagittal"})
        self.check(status == 200 and body.get("ok"),
                   f"file source opened ({body.get('source')})",
                   f"open failed: HTTP {status} {body}")
        if self.fails:
            return

        t0 = time.time()
        while time.time() - t0 < 15:
            _, st = _req(self.base, "/api/state")
            if st.get("active") and (st.get("media") or {}).get("is_file"):
                break
            time.sleep(0.5)
        _, st = _req(self.base, "/api/state")
        self.check(st.get("active"), "source is active", "source never became active")

        # 2. session ----------------------------------------------------
        status, body = _req(self.base, "/api/session/start",
                            {"name": "Smoke Angles", "phone": "000",
                             "duration_s": self.duration, "view_plane": "sagittal",
                             "discipline": "running"})
        if not self.check(status == 200 and body.get("ok"),
                          f"session started (id {body.get('session_id')})",
                          f"start failed: HTTP {status} {body}"):
            return
        sid = body.get("session_id")

        # 3. rows accumulate -------------------------------------------
        rows = 0
        t0 = time.time()
        while time.time() - t0 < 30:
            _, st = _req(self.base, "/api/state")
            rows = int((st.get("session") or {}).get("rows") or 0)
            if rows >= 20:
                break
            time.sleep(0.5)
        self.check(rows >= 20, f"capture loop advancing ({rows} rows)",
                   f"capture loop stalled at {rows} rows")

        # 4. capture angles --------------------------------------------
        status, body = _req(self.base, "/api/session/capture_angles",
                            {"label": "smoke mid-run"})
        if status == 200 and body.get("ok"):
            cap = body.get("capture") or {}
            self.ok(f"capture stored: t={cap.get('t')}s "
                    f"label={cap.get('label')!r} joints={len(cap.get('angles') or {})}")
        elif status == 400 and "no joint angles" in str(body.get("error", "")):
            self.warn("capture refused: this clip has no measurable joint "
                      "(expected for the upper-body test clip)")
        else:
            self.fail(f"capture failed: HTTP {status} {body}")

        _, st = _req(self.base, "/api/state")
        js = st.get("joint_study") or {}
        if js:
            self.ok(f"/api/state joint_study = {js.get('captures')} capture(s)")
        else:
            self.warn("/api/state has no joint_study key (older server?)")

        # 5. let the clip end / stop the session -----------------------
        t0 = time.time()
        while time.time() - t0 < self.duration + 30:
            _, st = _req(self.base, "/api/state")
            if not (st.get("session") or {}).get("id"):
                break
            time.sleep(1.0)
        _, st = _req(self.base, "/api/state")
        if (st.get("session") or {}).get("id"):
            _req(self.base, "/api/session/stop", {})
            time.sleep(2.0)
            _, st = _req(self.base, "/api/state")
        self.check(not (st.get("session") or {}).get("id"),
                   "session finalized", "session still running after stop")

        # 6. report ----------------------------------------------------
        report_sid = st.get("last_report_sid") or sid
        html = None
        last_code = None
        t0 = time.time()
        while time.time() - t0 < 25:          # report is built right after finalize
            try:
                html = _text(self.base, f"/report/{report_sid}/")
                break
            except urllib.error.HTTPError as exc:
                last_code = exc.code
                time.sleep(1.5)
        if html is None:
            self.fail(f"/report/{report_sid}/ -> HTTP {last_code} (gave up after 25 s)")
            return
        self.check("Captured angles" in html or "Joint study" in html,
                   "report renders the joint-study sections",
                   "report has no Captured angles / Joint study section")
        self.ok(f"report: {self.base}/report/{report_sid}/  ({len(html)} bytes)")

        # the running server keeps its own state; the JSON is on disk
        summary_path = REPO / "data" / "sessions" / str(report_sid) / "summary.json"
        if summary_path.exists():
            doc = json.loads(summary_path.read_text(encoding="utf-8"))
            has_caps = "angle_captures" in doc
            has_study = "joint_study" in doc
            self.check(has_caps and has_study,
                       f"summary.json carries angle_captures ({len(doc.get('angle_captures') or [])}) "
                       f"and joint_study ({len(doc.get('joint_study') or {})} joints)",
                       "summary.json is missing angle_captures / joint_study")
        else:
            self.warn(f"summary.json not found at {summary_path}")

        print()
        print(f"RESULT: {self.fails} failure(s), {self.warns} warning(s)")
        print(f"report: {self.base}/report/{report_sid}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8780")
    ap.add_argument("--clip", default=str(DEFAULT_CLIP))
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args(argv)
    smoke = Smoke(args.base, Path(args.clip), args.duration)
    smoke.run()
    return 1 if smoke.fails else 0


if __name__ == "__main__":
    sys.exit(main())
