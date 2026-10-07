"""RunTrack — local running-form tracker.

Live: click a camera (USB / phone IP stream / video file) → live joint
tracking → record a 1-minute run session → automatic report.

Local only: binds 127.0.0.1. No cloud upload.
"""
from __future__ import annotations

import base64
import collections
import json
import re
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

import advisor
import ai_backend
import cameras
import config
import cycling as CY
import gait_events as GA
import live_monitor
import metrics as M
import report as R
from filters import LandmarkFilter
from pose_engine import (PoseEngine, draw_skeleton, make_crop_refiner,
                         overlay_angles, person_crop_box)
from session_store import Store
from side_identity import SideIdentity

app = Flask(__name__)
STORE = Store(config.DB_PATH)


# ------------------------------------------------------------------ state --

class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_evt = threading.Event()
        self.active = False
        self.error = None
        self.source = None            # {"kind": ..., ...}
        self.source_label = "none"
        self.latest_jpeg = None
        self.live = {}                # latest frame metrics
        self.fps = 0.0
        self.session = None           # running session dict or None
        self.last_report_sid = None
        self.frames_seen = 0
        self.view_plane = config.DEFAULT_VIEW_PLANE
        # optional simultaneous second camera (own plane + own stream)
        self.source2 = None
        self.source2_label = "none"
        self.plane2 = config.DEFAULT_PLANE2
        self.seg = False
        self.seg2 = False
        self.latest_jpeg2 = None
        self.live2 = {}
        self.fps2 = 0.0
        self.warn2 = None
        self.discipline = config.DEFAULT_DISCIPLINE
        self.bike_type = config.DEFAULT_BIKE_TYPE
        # media playback (video files): slow-motion + scrubbing
        self.playback_speed = 1.0
        self.paused = False
        self.seek_t = None
        self.video_t = 0.0
        self.media = {}

    def reset(self):
        self.stop_evt = threading.Event()
        self.active = False
        self.error = None
        self.latest_jpeg = None
        self.live = {}
        self.fps = 0.0
        self.frames_seen = 0
        self.playback_speed = 1.0
        self.paused = False
        self.seek_t = None
        self.video_t = 0.0
        self.media = {}
        self.latest_jpeg2 = None
        self.live2 = {}
        self.fps2 = 0.0
        self.warn2 = None

    def playback_state(self) -> dict:
        return {
            "video_t": round(float(self.video_t or 0.0), 2),
            "video_duration": (self.media or {}).get("duration_s"),
            "speed": float(self.playback_speed),
            "paused": bool(self.paused),
            "can_control": bool(self.source and self.source.get("kind") == "file"),
            "speeds": list(config.PLAYBACK_SPEEDS),
        }


L = Live()


def _close_camera(join=True):
    if L.thread and L.thread.is_alive():
        L.stop_evt.set()
        if join:
            L.thread.join(timeout=4.0)
    L.thread = None


def _attach_hi(ev, ring):
    """Attach a native-resolution crop of the event frame to a latch event.

    The crop is what the HD refinement pass re-detects at session end (the
    full frame itself is long gone by then); the box lets the refined,
    crop-normalized landmarks be mapped back to frame coordinates.
    """
    i = ev.get("i")
    frame = None
    for j, fr in ring:
        if j == i:
            frame = fr
            break
    if frame is None:
        return
    box = person_crop_box(ev.get("lms"), pad=config.KEYFRAME_CROP_PAD)
    if box is None:
        return
    h, w = frame.shape[:2]
    x0 = int(max(0, round(box[0] * w)))
    y0 = int(max(0, round(box[1] * h)))
    x1 = int(min(w, round(box[2] * w)))
    y1 = int(min(h, round(box[3] * h)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return
    crop = frame[y0:y1, x0:x1]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        ev["hi"] = buf.tobytes()
        ev["hi_box"] = [float(b) for b in box]


def _finalize(reason="complete"):
    """Finalize the running session: CSV, keyframes, summary, charts, report.

    Safe to call from any thread; guarded against double-finalize.
    """
    with L.lock:
        sess = L.session
        if not sess or sess.get("finalized") or sess.get("finalizing"):
            return None
        sess["finalizing"] = True
        sid = sess["id"]
        rows = list(sess["rows"])
        params = dict(sess["params"])
        elapsed = sess.get("elapsed", 0.0)
        plane = sess.get("view_plane") or config.DEFAULT_VIEW_PLANE
        latch = sess.get("latch")
        side_id = sess.get("side_id")
        seeks = int(sess.get("seeks", 0))
        discipline = str(sess.get("discipline") or config.DEFAULT_DISCIPLINE).lower()
        bike_type = sess.get("bike_type") or config.DEFAULT_BIKE_TYPE
    status = "complete" if reason == "complete" else "incomplete"
    try:
        if discipline == "running":
            events = latch.events() if latch is not None else None
        else:
            # cycling: no gait events; the BDC/TDC latch feeds keyframes only
            events = []
        summary = M.analyze_session(rows, weight_kg=params.get("weight_kg"),
                                    height_cm=params.get("height_cm"),
                                    speed_kmh=params.get("speed_kmh"),
                                    view_plane=plane, events=events,
                                    discipline=discipline)
        CY.enrich(summary, rows, params, bike_type=bike_type, discipline=discipline)
        if side_id is not None:
            try:
                summary["side_identity"] = side_id.info()
            except Exception:
                pass
        summary["keyframes"] = []
        if latch is not None:
            ref = None
            refine = None
            try:
                ref = make_crop_refiner()      # heavy model, IMAGE mode
                refine = ref[1] if ref else None
            except Exception:
                ref = None                     # no HD pass — keyframes still saved
            try:
                summary["keyframes"] = latch.save_snapshots(
                    config.SESSIONS_DIR / str(sid), refine=refine)
            except Exception as e:
                summary.setdefault("labels", []).append(f"keyframe capture failed: {e}")
            finally:
                if ref is not None:
                    try:
                        ref[0].close()
                    except Exception:
                        pass
        if seeks:
            summary.setdefault("labels", []).append(
                f"video playback was scrubbed during this session "
                f"({seeks} seek jumps) — the time axis has gaps")
        advisor.enrich(summary)
        if reason == "stopped":
            summary["note"] = "session stopped early by operator"
            status = "complete" if elapsed >= 5 else "incomplete"
        STORE.save_csv(sid, rows)
        STORE.finish_session(sid, summary.get("duration_s", elapsed), len(rows),
                             summary, status=status)
        R.build(sid, STORE)
    except Exception as e:  # never leave a session half-written silently
        try:
            STORE.finish_session(sid, elapsed, len(rows), {"error": str(e)},
                                 status="incomplete")
        except Exception:
            pass
    with L.lock:
        sess["finalized"] = True
        sess["finalizing"] = False
        sess["latch"] = None
        if L.session is sess:
            L.session = None
        L.last_report_sid = sid
    return sid


# ------------------------------------------------------------ capture loop --

def _hud(frame, fm, fps, sess):
    h, w = frame.shape[:2]
    txt = f"{fps:.0f} fps"
    cv2.putText(frame, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 255, 200), 1, cv2.LINE_AA)
    if sess is not None:
        left = max(0.0, sess["duration"] - sess.get("elapsed", 0.0))
        mins, secs = divmod(int(left + 0.999), 60)
        ctxt = f"{mins:02d}:{secs:02d}"
        (tw, th), _ = cv2.getTextSize(ctxt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, ctxt, (w - tw - 16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, ctxt, (w - tw - 16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (80, 220, 255), 2, cv2.LINE_AA)
    return frame


def _loop(src: dict):
    cap = None
    cap2 = None
    engine = None
    engine2 = None
    lf = LandmarkFilter()
    lf2 = LandmarkFilter()
    si = SideIdentity() if config.SIDE_IDENTITY_ENABLED else None
    si2 = SideIdentity() if config.SIDE_IDENTITY_ENABLED else None
    ended_naturally = False
    try:
        cap = cameras.open_source(**{k: v for k, v in src.items()
                                     if k not in ("seg", "source2")})
        if cap is None:
            with L.lock:
                L.error = f"could not open source: {src}"
                L.active = False
            return
        engine = PoseEngine(config.MODEL_PATH, seg=bool(src.get("seg")))
        # ------------------------------ optional simultaneous second camera
        src2 = src.get("source2") or {}
        if src2:
            cap2 = cameras.open_source(**{k: v for k, v in src2.items()
                                          if k != "seg"})
            if cap2 is None:
                with L.lock:
                    L.warn2 = f"second camera could not be opened: {src2}"
            else:
                engine2 = PoseEngine(config.MODEL_PATH, seg=bool(src2.get("seg")))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_file = src.get("kind") == "file"
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        with L.lock:
            L.media = {
                "fps": round(float(src_fps), 2), "is_file": is_file,
                "duration_s": (round(float(n_frames) / src_fps, 2)
                               if (is_file and src_fps > 0 and n_frames > 0) else None),
            }
        base_pace = (1.0 / src_fps) if (is_file and 5 <= src_fps <= 120) else 0.0
        fps, fps_n, fps_t = 0.0, 0, time.time()
        fps2, fps2_n, fps2_t = 0.0, 0, time.time()
        while not L.stop_evt.is_set():
            t_frame = time.monotonic()
            with L.lock:
                seek_t = L.seek_t
                L.seek_t = None
                paused = bool(L.paused) and is_file
                speed = float(L.playback_speed) if is_file else 1.0
            if paused and seek_t is None:
                time.sleep(0.05)
                continue
            if is_file and seek_t is not None:
                for c_ in (cap, cap2):
                    if c_ is not None:
                        try:
                            c_.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seek_t)) * 1000.0)
                        except Exception:
                            pass
                lf.reset()                    # new time base -> restart smoothing
                if si is not None:
                    si.reset()                # and restart identity tracking
                if lf2 is not None:
                    lf2.reset()
                if si2 is not None:
                    si2.reset()
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    ended_naturally = True
                    break
                time.sleep(0.05)
                continue
            # ---- second camera (best effort: never blocks the primary)
            frame2 = None
            if cap2 is not None:
                try:
                    ok2, fr2 = cap2.read()
                    if ok2:
                        frame2 = fr2
                except Exception:
                    frame2 = None
            with L.lock:
                L.frames_seen += 1
            vt = None
            if is_file:
                try:
                    vt = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                except Exception:
                    vt = None
            h, w = frame.shape[:2]
            scale = config.PROC_WIDTH / w if w > config.PROC_WIDTH else 1.0
            proc = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            t_mono = time.monotonic()
            lms_raw = engine.detect(proc, int(t_mono * 1000))
            # left/right identity stays pinned across label flips (legs + arms)
            ident = None
            if si is not None:
                lms_raw, ident = si.update(lms_raw, t_mono)
            # 1-Euro smoothing before metric computation; raw landmarks are kept
            # in the keyframe buffer for event detection / re-analysis
            lms = lf(lms_raw, t_mono) if lms_raw is not None else None
            fm = M.frame_metrics(lms)
            if src.get("seg") and engine.seg_frac is not None:
                fm["fa_frac"] = engine.seg_frac
            annotated = draw_skeleton(proc.copy(), lms)
            overlay_angles(annotated, lms, fm)

            # ---- second camera: detect + metrics + its own annotated frame
            fm2, ident2, annotated2 = {}, None, None
            if frame2 is not None and engine2 is not None:
                h2, w2 = frame2.shape[:2]
                scale2 = config.PROC_WIDTH / w2 if w2 > config.PROC_WIDTH else 1.0
                proc2 = (cv2.resize(frame2, None, fx=scale2, fy=scale2)
                         if scale2 != 1.0 else frame2)
                lms_raw2 = engine2.detect(proc2, int(t_mono * 1000))
                if si2 is not None:
                    lms_raw2, ident2 = si2.update(lms_raw2, t_mono)
                lms2 = lf2(lms_raw2, t_mono) if lms_raw2 is not None else None
                fm2 = M.frame_metrics(lms2)
                if src2.get("seg") and engine2.seg_frac is not None:
                    fm2["fa_frac"] = engine2.seg_frac
                annotated2 = draw_skeleton(proc2.copy(), lms2)
                overlay_angles(annotated2, lms2, fm2)

            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now
            if frame2 is not None:
                fps2_n += 1
                if now - fps2_t >= 1.0:
                    fps2 = fps2_n / (now - fps2_t)
                    fps2_n, fps2_t = 0, now

            auto_done = False
            with L.lock:
                sess = L.session
                run = sess if (sess and not sess.get("finalized")
                               and not sess.get("finalizing")) else None
                annotated = _hud(annotated, fm, fps, run)
                ok_b, buf = cv2.imencode(".jpg", annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                jpeg_bytes = buf.tobytes() if ok_b else None
                if jpeg_bytes:
                    L.latest_jpeg = jpeg_bytes
                if annotated2 is not None:
                    ok_b2, buf2 = cv2.imencode(
                        ".jpg", annotated2,
                        [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                    if ok_b2:
                        L.latest_jpeg2 = buf2.tobytes()
                if is_file and vt is not None:
                    L.video_t = vt
                L.fps = fps
                L.live = {"pose_found": lms is not None, "fps": round(fps, 1),
                          "t": now, "view_plane": L.view_plane,
                          "video_t": L.video_t if is_file else None,
                          "playback_speed": speed if is_file else 1.0,
                          "paused": paused,
                          **({"identity": ident} if ident else {}),
                          **{k: v for k, v in fm.items() if not k.startswith("_")}}
                if frame2 is not None:
                    L.fps2 = fps2
                    L.live2 = {"pose_found": bool(fm2), "fps": round(fps2, 1),
                               "view_plane": L.plane2,
                               **({"identity": ident2} if ident2 else {}),
                               **{k: v for k, v in fm2.items()
                                  if not k.startswith("_")}}
                if run is not None:
                    if si is not None:
                        run.setdefault("side_id", si)
                    if is_file and vt is not None:
                        if run.get("t0_video") is None:
                            run["t0_video"] = vt
                        rel = max(0.0, vt - run["t0_video"])
                    else:
                        rel = time.monotonic() - run["t0"]
                    jump = run.get("last_rel") is not None and (
                        rel < run["last_rel"] - 1e-3 or rel - run["last_rel"] > 1.0)
                    if jump:
                        run["seeks"] = int(run.get("seeks", 0)) + 1
                    run["last_rel"] = rel
                    run["elapsed"] = rel
                    row = dict(fm)
                    # second-camera metrics ride along with an "r_" prefix
                    for k2, v2 in fm2.items():
                        row["r_" + k2] = v2
                    row["t_rel"] = rel
                    if is_file and vt is not None:
                        row["t_video"] = round(vt, 3)
                        row["playback_speed"] = float(speed)
                    if jump:
                        row["seek_jump"] = 1
                    run["rows"].append(row)
                    latch = run.get("latch")
                    if latch is not None:
                        # store the CLEAN frame for keyframes; the skeleton is
                        # (re)drawn at save time — refined when the HD pass runs
                        ok_c, cbuf = cv2.imencode(
                            ".jpg", proc,
                            [cv2.IMWRITE_JPEG_QUALITY, config.KEYFRAME_JPEG_QUALITY])
                        clean_jpeg = cbuf.tobytes() if ok_c else None
                        idx = getattr(latch, "_i", None)
                        ring = run.get("frame_ring")
                        if ring is None:
                            ring = run["frame_ring"] = collections.deque(
                                maxlen=config.FRAME_RING)
                        if idx is not None:
                            ring.append((idx, frame.copy()))
                        new_evs = []
                        if clean_jpeg is not None:
                            try:
                                new_evs = latch.push(rel, fm, clean_jpeg, lms)
                            except Exception:
                                new_evs = []
                        for ev in new_evs:
                            try:
                                # HD crop only for events with usable metrics
                                # (keeps session memory sane)
                                if float(ev.get("quality") or 0.0) >= 1.0:
                                    _attach_hi(ev, ring)
                            except Exception:
                                pass
                    if (rel - run["last_snap"] >= max(2.0, run["duration"] / 3.0)
                            and run["snap_count"] < 3):
                        sdir = config.SESSIONS_DIR / str(run["id"])
                        sdir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(sdir / f"frame_{run['snap_count']}.jpg"), annotated)
                        run["snap_count"] += 1
                        run["last_snap"] = rel
                    if rel >= run["duration"]:
                        auto_done = True
            if auto_done:
                _finalize("complete")
            if base_pace:
                dt = time.monotonic() - t_frame
                pace = base_pace / max(speed, 0.05)
                if dt < pace:
                    time.sleep(pace - dt)
    except Exception as e:
        with L.lock:
            L.error = f"capture error: {e}"
    finally:
        if engine is not None:
            engine.close()
        if engine2 is not None:
            engine2.close()
        if cap is not None:
            cap.release()
        if cap2 is not None:
            cap2.release()
        with L.lock:
            L.active = False
    # file ended while recording → finalize as complete
    if ended_naturally:
        _finalize("complete")


# ------------------------------------------------------------------ routes --

def _origin_ok():
    host = request.host.split(":")[0]
    return host in ("127.0.0.1", "localhost", "[::1]")


@app.get("/")
def index():
    return render_template("index.html", default_duration=config.DEFAULT_DURATION_S,
                           port=config.PORT, ollama_model=ai_backend.status_text(),
                           voice_cooldown=config.VOICE_COOLDOWN_S,
                           met_min=config.METRONOME_MIN_BPM,
                           met_max=config.METRONOME_MAX_BPM,
                           met_step=config.METRONOME_STEP_BPM,
                           met_presets=list(config.METRONOME_PRESETS),
                           playback_speeds=list(config.PLAYBACK_SPEEDS),
                           view_planes=list(config.VIEW_PLANES),
                           disciplines=list(config.DISCIPLINES),
                           bike_types=list(config.BIKE_TYPES),
                           cd_default=config.AERO_CD_DEFAULT,
                           bands={k: list(v) for k, v in config.BANDS.items()})


@app.get("/video_feed")
def video_feed():
    def gen():
        try:
            while True:
                with L.lock:
                    jpg = L.latest_jpeg
                    active = L.active
                if jpg is None or not active:
                    time.sleep(0.2)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / config.STREAM_FPS)
        except GeneratorExit:
            return
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed2")
def video_feed2():
    def gen():
        try:
            while True:
                with L.lock:
                    jpg = L.latest_jpeg2
                    active = L.active
                if jpg is None or not active:
                    time.sleep(0.2)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / config.STREAM_FPS)
        except GeneratorExit:
            return
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _build_state() -> dict:
    """Snapshot of the live state (shared by /api/state and the AI monitor chat)."""
    with L.lock:
        sess = L.session
        run = sess and not sess.get("finalizing")
        disc = str((sess or {}).get("discipline") or L.discipline or
                   config.DEFAULT_DISCIPLINE)
        latch = (sess or {}).get("latch")
        st = {
            "active": L.active,
            "source": L.source_label,
            "source2": L.source2_label,
            "error": L.error,
            "warning2": L.warn2,
            "fps": round(L.fps, 1),
            "fps2": round(L.fps2, 1),
            "live": L.live,
            "live2": L.live2,
            "view_plane": L.view_plane,
            "plane2": L.plane2,
            "seg": bool(L.seg),
            "seg2": bool(L.seg2),
            "has_camera2": bool(L.source2),
            "discipline": disc,
            "bike_type": L.bike_type,
            "playback": L.playback_state(),
            "media": dict(L.media or {}),
            "session": None if not (sess and run) else {
                "id": sess["id"],
                "elapsed": round(sess.get("elapsed", 0.0), 1),
                "duration": sess["duration"],
                "patient": sess["patient"],
                "view_plane": sess.get("view_plane"),
                "discipline": sess.get("discipline"),
                "bike_type": sess.get("bike_type"),
                "rows": len(sess["rows"]),
                **({"events_revolutions": latch.revolutions()}
                   if (disc == "cycling" and latch is not None
                       and hasattr(latch, "revolutions")) else {}),
                "events": (GA.summarize(latch.events())
                           if (disc == "running" and latch is not None) else None),
                "events_coverage_pct": (round(latch.lower_body_coverage(), 1)
                                        if (disc == "running" and latch is not None)
                                        else None),
                "events_note": (config.GAIT_NOT_ASSESSABLE_TEXT
                                if (disc == "running" and latch is not None
                                    and not latch.events()
                                    and GA.events_not_assessable(
                                        latch.lower_body_coverage()))
                                else None),
            },
            "session_finalizing": bool(sess and sess.get("finalizing")),
            "last_report_sid": L.last_report_sid,
        }
    return st


@app.get("/api/state")
def api_state():
    return jsonify(_build_state())


_CAM_CACHE = {"cams": None}


def _cached_pin_reason(src) -> str | None:
    """Guard: refuse opening a Femto Depth/IR pin (opens but shows nothing).

    Names come from the last /api/cameras scan; if no scan has happened yet the
    check fails open (the app still runs) rather than blocking camera opens.
    """
    cams = _CAM_CACHE.get("cams")
    if not cams or not isinstance(src, dict) or src.get("kind") != "usb":
        return None
    index = src.get("index")
    for cam in cams:
        if cam.get("index") == index:
            return cameras.non_video_pin_reason(cam.get("name"))
    return None


@app.get("/api/cameras")
def api_cameras():
    cams = cameras.list_usb_cameras()
    _CAM_CACHE["cams"] = cams
    return jsonify({"cameras": cams, "preferred": cameras.preferred_camera(cams)})


@app.post("/api/camera/open")
def api_camera_open():
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    kind = data.get("kind")

    def _parse_source(d, field="kind"):
        k = d.get(field)
        if k == "usb":
            return {"kind": "usb", "index": int(d.get("index", 0))}
        if k == "url":
            url = (d.get("url") or "").strip()
            if not url:
                return "no URL given"
            return {"kind": "url", "url": url}
        if k == "file":
            path = (d.get("path") or "").strip()
            if not Path(path).exists():
                return "file not found"
            return {"kind": "file", "path": path}
        return "bad kind"

    src = _parse_source(data)
    if isinstance(src, str):
        return jsonify({"ok": False, "error": src}), 400
    pin_reason = _cached_pin_reason(src)
    if pin_reason:
        return jsonify({"ok": False, "error": pin_reason}), 400
    src["seg"] = bool(data.get("seg"))
    # optional simultaneous second camera
    src2 = None
    d2 = data.get("source2") or {}
    if d2 and d2.get("kind"):
        s2 = _parse_source(d2)
        if isinstance(s2, str):
            return jsonify({"ok": False, "error": f"second camera: {s2}"}), 400
        s2_reason = _cached_pin_reason(s2)
        if s2_reason:
            return jsonify({"ok": False, "error": f"second camera: {s2_reason}"}), 400
        s2["seg"] = bool(data.get("seg2"))
        src2 = s2
    plane2 = str(data.get("plane2") or config.DEFAULT_PLANE2).lower()
    if plane2 not in config.VIEW_PLANES:
        plane2 = config.DEFAULT_PLANE2
    _close_camera()
    _finalize("stopped")
    if src2 is not None:
        src["source2"] = src2          # the capture loop opens both sources

    with L.lock:
        L.reset()
        L.source = src
        L.source_label = cameras.source_label(
            **{k: v for k, v in src.items() if k not in ("seg", "source2")})
        L.source2 = src2
        L.source2_label = (cameras.source_label(
            **{k: v for k, v in src2.items() if k != "seg"})
            if src2 else "none")
        L.plane2 = plane2
        L.seg = bool(src.get("seg"))
        L.seg2 = bool(src2 and src2.get("seg"))
        L.active = True
    L.thread = threading.Thread(target=_loop, args=(src,), daemon=True)
    L.thread.start()
    time.sleep(0.4)
    with L.lock:
        err = L.error
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({"ok": True, "source": L.source_label,
                    "source2": L.source2_label})


@app.post("/api/camera/close")
def api_camera_close():
    if not _origin_ok():
        abort(403)
    _close_camera()
    _finalize("stopped")
    with L.lock:
        L.active = False
        L.source = None
        L.source_label = "none"
        L.live = {}
        L.latest_jpeg = None
    return jsonify({"ok": True})


@app.post("/api/upload")
def api_upload():
    if not _origin_ok():
        abort(403)
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(f.filename).name)
    stamped = time.strftime("%Y%m%d_%H%M%S_") + safe
    dest = config.UPLOADS_DIR / stamped
    f.save(str(dest))
    mb = dest.stat().st_size / (1024 * 1024)
    if mb > config.MAX_UPLOAD_MB:
        dest.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": f"file too large ({mb:.0f} MB)"}), 400
    return jsonify({"ok": True, "path": str(dest), "mb": round(mb, 1)})


@app.post("/api/session/start")
def api_session_start():
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    if not name or not phone:
        return jsonify({"ok": False, "error": "name and phone are required"}), 400
    if not L.active:
        return jsonify({"ok": False, "error": "open a camera first"}), 400
    if L.session and not L.session.get("finalized"):
        return jsonify({"ok": False, "error": "a session is already running"}), 400

    def _num(key):
        v = data.get(key)
        try:
            return float(v) if v not in (None, "", "null") else None
        except (TypeError, ValueError):
            return None

    dur = _num("duration_s") or config.DEFAULT_DURATION_S
    dur = max(5.0, min(float(dur), config.MAX_DURATION_S))
    params = {"weight_kg": _num("weight_kg"), "height_cm": _num("height_cm"),
              "speed_kmh": _num("speed_kmh"), "shoulder_cm": _num("shoulder_cm"),
              "cd_est": _num("cd_est")}
    plane = str(data.get("view_plane") or config.DEFAULT_VIEW_PLANE).lower()
    if plane not in config.VIEW_PLANES:
        plane = config.DEFAULT_VIEW_PLANE
    discipline = str(data.get("discipline") or config.DEFAULT_DISCIPLINE).lower()
    if discipline not in config.DISCIPLINES:
        discipline = config.DEFAULT_DISCIPLINE
    bike_type = str(data.get("bike_type") or config.DEFAULT_BIKE_TYPE).lower()
    if bike_type not in config.BIKE_TYPES:
        bike_type = config.DEFAULT_BIKE_TYPE
    patient_id = STORE.upsert_patient(name, phone)
    sid = STORE.create_session(patient_id, params, L.source_label, view_plane=plane,
                               discipline=discipline,
                               bike_type=bike_type if discipline == "cycling" else None)
    latch = CY.CycleLatch() if discipline == "cycling" else GA.GaitLatch()
    with L.lock:
        L.view_plane = plane
        L.discipline = discipline
        L.bike_type = bike_type
        L.session = {"id": sid, "patient": {"name": name, "phone": phone},
                     "t0": time.monotonic(), "elapsed": 0.0, "duration": dur,
                     "rows": [], "params": params, "finalized": False,
                     "finalizing": False, "snap_count": 0, "last_snap": -1e9,
                     "view_plane": plane, "discipline": discipline,
                     "bike_type": bike_type, "latch": latch,
                     "seeks": 0, "last_rel": None, "t0_video": None}
        L.last_report_sid = None
    return jsonify({"ok": True, "session_id": sid, "duration": dur,
                    "view_plane": plane, "discipline": discipline,
                    "bike_type": bike_type})


@app.post("/api/session/stop")
def api_session_stop():
    if not _origin_ok():
        abort(403)
    sid = _finalize("stopped")
    if sid is None:
        return jsonify({"ok": False, "error": "no running session"}), 400
    return jsonify({"ok": True, "session_id": sid})


@app.post("/api/patient/<int:pid>/edit")
def api_patient_edit(pid: int):
    """Edit a saved patient's name/phone (Patients tab in the UI)."""
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    try:
        p = STORE.update_patient(pid, data.get("name"), data.get("phone"))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    if not p:
        return jsonify({"ok": False, "error": "unknown patient"}), 404
    return jsonify({"ok": True, "patient": p})


@app.post("/api/session/<int:sid>/edit")
def api_session_edit(sid: int):
    """Edit a saved session (weight/height/speed + patient name/phone)."""
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    sess = STORE.get_session(sid)
    if not sess:
        return jsonify({"ok": False, "error": "unknown session"}), 404

    def _num(key):
        v = data.get(key)
        try:
            return float(v) if v not in (None, "", "null") else None
        except (TypeError, ValueError):
            return None

    s = STORE.update_session(sid, _num("weight_kg"), _num("height_cm"),
                             _num("speed_kmh"))
    try:
        p = STORE.update_patient(int(sess["patient_id"]), data.get("name"),
                                 data.get("phone"))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "session": s, "patient": p})


@app.get("/api/sessions")
def api_sessions():
    return jsonify({"sessions": STORE.list_sessions(100)})


@app.get("/api/patients")
def api_patients():
    return jsonify({"patients": STORE.list_patients(),
                    "view_planes": list(config.VIEW_PLANES)})


@app.get("/api/patients/<int:pid>/history")
def api_patient_history(pid: int):
    p = STORE.get_patient(pid)
    if not p:
        return jsonify({"ok": False, "error": "unknown patient"}), 404
    sess = STORE.patient_sessions(pid, limit=30)
    return jsonify({"ok": True, "patient": p, "sessions": sess,
                    "limit": 30, "n": len(sess)})


@app.get("/api/history")
def api_history():
    return jsonify({"ok": True, "patients": STORE.history(per_patient_limit=30),
                    "limit_per_patient": 30})


@app.post("/api/playback")
def api_playback():
    """Slow-motion speed + scrubbing for video-file sources."""
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    with L.lock:
        kind = (L.source or {}).get("kind")
        if kind != "file":
            return jsonify({"ok": False,
                            "error": "playback controls need a video file source"}), 400
        if data.get("speed") is not None:
            try:
                sp = float(data["speed"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "bad speed"}), 400
            L.playback_speed = max(0.05, min(1.0, sp))
        if data.get("paused") is not None:
            L.paused = bool(data["paused"])
        if data.get("seek") is not None:
            try:
                L.seek_t = max(0.0, float(data["seek"]))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "bad seek"}), 400
        if data.get("step") is not None:
            # single-frame step (pauses playback and moves the video clock)
            try:
                step = float(data["step"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "bad step"}), 400
            L.paused = True
            L.seek_t = max(0.0, float(L.video_t or 0.0) + step)
        st = L.playback_state()
    return jsonify({"ok": True, **st})


@app.get("/api/compare")
def api_compare():
    """Baseline (a) vs follow-up (b) session comparison with delta badges."""
    a = request.args.get("a", type=int)
    b = request.args.get("b", type=int)
    if not a or not b:
        return jsonify({"ok": False,
                        "error": "pass ?a=<baseline session id>&b=<follow-up session id>"}), 400
    sa, sb = STORE.get_session(a), STORE.get_session(b)
    if not sa or not sb:
        return jsonify({"ok": False, "error": "unknown session"}), 404

    def _summ(s):
        try:
            return json.loads(s.get("summary_json") or "{}")
        except Exception:
            return {}

    def _meta(s):
        return {"id": s["id"], "date": s.get("started_ts"),
                "patient": s.get("patient_name"), "phone": s.get("patient_phone"),
                "plane": s.get("view_plane"), "source": s.get("source_label"),
                "discipline": s.get("discipline"), "bike_type": s.get("bike_type"),
                "speed_kmh": s.get("speed_kmh"), "status": s.get("status")}

    out = advisor.compare(_summ(sa), _summ(sb), _meta(sa), _meta(sb))
    out["ok"] = True
    return jsonify(out)


@app.get("/report/<int:sid>/")
def report_page(sid: int):
    p = config.SESSIONS_DIR / str(sid) / "report.html"
    if not p.exists():
        abort(404)
    return send_file(p)


@app.get("/report/<int:sid>/<path:name>")
def report_file(sid: int, name: str):
    base = (config.SESSIONS_DIR / str(sid)).resolve()
    p = (base / name).resolve()
    if not str(p).startswith(str(base)) or not p.exists() or p.is_dir():
        abort(404)
    return send_file(p)


@app.post("/api/coach/<int:sid>")
def api_coach(sid: int):
    if not _origin_ok():
        abort(403)
    sess = STORE.get_session(sid)
    if not sess:
        return jsonify({"ok": False, "error": "unknown session"}), 404
    sdir = config.SESSIONS_DIR / str(sid)
    try:
        summary = json.loads(sess.get("summary_json") or "{}")
    except Exception:
        summary = {}
    imgs = []
    for i in range(3):
        p = sdir / f"frame_{i}.jpg"
        if p.exists():
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > 512:
                img = cv2.resize(img, (512, int(h * 512 / w)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                imgs.append(base64.b64encode(buf.tobytes()).decode())
    disc = str(summary.get("discipline") or sess.get("discipline") or "running")
    bike = summary.get("bike") or {}
    rear = summary.get("rear") or {}
    aero = summary.get("aero") or {}
    if disc == "cycling":
        bt = summary.get("bike_type") or sess.get("bike_type") or "road"
        prompt = (
            "You are an assistant to a cycling coach and bike fitter. You receive "
            f"bike-position analysis data ({bt} position) from video pose estimation "
            "(side camera for angles, optional second camera for rear-view control, "
            "front-camera frontal area for aero) and up to 3 frames. Give concise, "
            "structured feedback: (1) position strengths, (2) fit adjustments to try "
            "with directions (saddle up/down, front end, reach), (3) one aero comment "
            "if frontal-area data is present, (4) one caveat about the data limits. "
            "120-200 words. Do not diagnose injuries. Values listed under 'not_assessed' "
            "were NOT measured - never comment on them.\n\nSummary JSON:\n"
            + json.dumps({"discipline": disc, "bike_type": bt,
                          "cadence_rpm": bike.get("cadence_rpm"),
                          "knee_bdc_deg": bike.get("knee_bdc_deg"),
                          "knee_bdc_max_deg": bike.get("knee_bdc_max_deg"),
                          "knee_tdc_min_deg": bike.get("knee_tdc_min_deg"),
                          "hip_tdc_min_deg": bike.get("hip_tdc_min_deg"),
                          "torso_deg": bike.get("torso_deg"),
                          "per_side": bike.get("per_side"),
                          "form": bike.get("form"),
                          "suggestions": bike.get("suggestions"),
                          "rear": rear, "aero": aero,
                          "symmetry": summary.get("symmetry"),
                          "risk": summary.get("risk"),
                          "not_assessed": summary.get("not_assessed")},
                         default=str))
    else:
        prompt = (
            "You are an assistant to a running coach. You receive biomechanics "
            "summary data (2D pose-estimated, single camera) and up to 3 frames "
            "of a person running on a treadmill. Give concise, structured form "
            "feedback: (1) three likely strengths, (2) three improvement "
            "suggestions, (3) one caveat about the data limits. 140-200 words. "
            "Do not diagnose injuries. Respect the camera view plane: metrics "
            "listed under 'not_assessed' were NOT measured and must not be "
            "commented on.\n\nSummary JSON:\n"
            + json.dumps({k: summary.get(k) for k in (
                "view_plane", "cadence_spm", "trunk_lean_deg", "trunk_lean_fwd_deg",
                "vertical_osc_cm", "symmetry", "symmetry_mean_pct", "asymmetry",
                "form", "form_index", "form_score", "risk", "step_length_m",
                "touchdown_shin_deg", "overstride_leg_frac", "knee_flexion_ic_deg",
                "knee_flexion_ms_deg", "pelvic_drop_deg", "knee_valgus_deg",
                "hip_extension_toeoff_deg", "pushoff_ankle_deg", "stance_time_s",
                "power_est_watts", "power_per_kg_watts", "gait_events", "rear",
                "not_assessed") if summary.get(k) is not None}, default=str)
        )
    try:
        text = ai_backend.describe_images(prompt, imgs)
        STORE.save_json(sid, "coach.json", {
            "model": ai_backend.active_model(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text, "images_used": len(imgs)})
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"AI request failed: {e}"}), 502


@app.post("/api/monitor/warm")
def api_monitor_warm():
    """Best-effort background preload of the local model for the live chat."""
    if not _origin_ok():
        abort(403)

    def _do_warm():
        try:
            ai_backend.warm()
        except Exception:
            pass                                  # warming must never break the app

    threading.Thread(target=_do_warm, daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/monitor/chat")
def api_monitor_chat():
    """Stream one live-monitor reply as newline-delimited JSON events.

    Body: {"mode": "auto"|"ask", "question": str?, "history": [...],
    "cadence_spm": float?}. The digest is built server-side from the current
    live state; only derived numbers reach the configured model (local Ollama
    or the API set with AI_API_URL).
    """
    if not _origin_ok():
        abort(403)
    data = request.get_json(force=True, silent=True) or {}
    mode = "ask" if str(data.get("mode") or "") == "ask" else "auto"
    question = str(data.get("question") or "").strip()
    if mode == "ask" and not question:
        return jsonify({"ok": False, "error": "empty question"}), 400
    bands = {k: list(v) for k, v in config.BANDS.items()
             if k in live_monitor.BAND_KEYS}
    digest = live_monitor.build_digest(_build_state(),
                                       cadence_spm=data.get("cadence_spm"),
                                       bands=bands)
    messages = live_monitor.build_live_messages(digest, history=data.get("history"),
                                                mode=mode, question=question)

    def gen():
        try:
            for event in ai_backend.chat_stream(messages):
                yield (json.dumps(event, ensure_ascii=False, allow_nan=False)
                       + "\n").encode("utf-8")
        except GeneratorExit:                     # client closed the chat: stop quietly
            return

    return Response(gen(), mimetype="application/x-ndjson")


def _open_browser():
    try:
        webbrowser.open(f"http://{config.HOST}:{config.PORT}/")
    except Exception:
        pass


if __name__ == "__main__":
    import os
    print(f"RunTrack starting on http://{config.HOST}:{config.PORT}/")
    print("Keep this window open. Ctrl+C to stop.")
    if not os.environ.get("RUNTRACK_NO_BROWSER"):
        threading.Timer(1.2, _open_browser).start()
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False)
