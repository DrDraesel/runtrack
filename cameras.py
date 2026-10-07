"""Camera discovery and source opening for RunTrack.

Sources:
  usb   - local USB / built-in camera via DirectShow (Media Foundation fallback)
  url   - phone IP-camera or any MJPEG/RTSP stream URL
  file  - a video file (e.g. recorded on a phone)

External USB/UVC cameras are supported (incl. Microsoft LifeCams). Up to
MAX_INDEX indexes are scanned; friendly device names come from pygrabber
(DirectShow) when available and are best-effort labels.
"""
from __future__ import annotations

from pathlib import Path
import re
import time

import cv2

MAX_INDEX = 8

# The Orbbec Femto Mega/Bolt exposes three UVC pins (RGB / depth / IR) named
# "Orbbec Femto Mega <pin> Camera".  Pose tracking needs the RGB pin, so it is
# auto-preferred in the camera list (zero-config "just press Open camera").
FEMTO_HINTS = ("femto", "orbbec")


def is_orbbec_femto(name) -> bool:
    n = (name or "").lower()
    return any(h in n for h in FEMTO_HINTS)


def preferred_camera(devices: list) -> int | None:
    """Index of the Femto Mega RGB camera, else any Femto pin, else None."""
    for d in devices:
        n = (d.get("name") or "").lower()
        if "rgb" in n and is_orbbec_femto(n):
            return d.get("index")
    for d in devices:
        if is_orbbec_femto(d.get("name")):
            return d.get("index")
    return None


def non_video_pin_reason(name) -> str | None:
    """Reason string when a UVC pin cannot provide usable colour video.

    The Femto Mega Depth/IR pins open successfully but deliver no frames the
    tracker can use, so picking one in the UI looks like "the camera is not
    opening".  Return a user-facing explanation, or None for normal cameras.
    """
    n = (name or "").lower()
    if not is_orbbec_femto(n):
        return None
    if "depth" in n:
        return ("That is the Orbbec Femto Mega DEPTH pin — it carries no colour video for "
                "pose tracking. Pick the RGB pin (marked ★ in the Camera list).")
    if " ir " in n or "infrared" in n:
        return ("That is the Orbbec Femto Mega IR pin — it carries no colour video. "
                "Pick the RGB pin (marked ★ in the Camera list).")
    return None


def _parse_ffmpeg_devices(text):
    """Video-device names from `ffmpeg -list_devices` output, in order."""
    names = []
    for line in (text or "").splitlines():
        if "Alternative name" in line:
            continue
        match = re.search(r'"([^"]+)"\s*\(video\)', line)
        if match:
            names.append(match.group(1))
    return names


def _ffmpeg_device_names():
    """Name fallback via ffmpeg: no COM, no device opens, safe from any thread."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "-hide_banner", "-list_devices", "true",
                               "-f", "dshow", "-i", "dummy"],
                              capture_output=True, text=True, timeout=20)
        text = (proc.stderr or "") + (proc.stdout or "")
    except Exception:
        return None
    return _parse_ffmpeg_devices(text) or None


def device_names():
    """Best-effort ordered list of DirectShow video-device names, or None.

    pygrabber needs COM initialised ON THE CALLING THREAD — from a Flask
    request thread it fails silently, and a nameless scan then probes index 0
    (the Femto depth pin) and wedges the UVC family. CoInitialize explicitly,
    then fall back to ffmpeg (no COM, callable from any thread).
    """
    try:
        import comtypes
        try:
            comtypes.CoInitialize()
        except Exception:
            pass
    except Exception:
        pass
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = list(FilterGraph().get_input_devices())
        if names:
            return names
    except Exception:
        pass
    return _ffmpeg_device_names()


def _probe_with(index: int, backend):
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    ok, frame = False, None
    try:
        ok, frame = cap.read()
    except Exception:
        ok = False
    info = None
    if ok and frame is not None:
        info = {
            "index": index,
            "backend": "dshow" if backend == cv2.CAP_DSHOW else "msmf",
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 1),
        }
    cap.release()
    return info


def probe_index(index: int):
    """Try DirectShow first, then Media Foundation. Returns info dict or None."""
    return _probe_with(index, cv2.CAP_DSHOW) or _probe_with(index, cv2.CAP_MSMF)


def list_usb_cameras(max_index: int = MAX_INDEX):
    """Probe camera indexes for usable colour video.

    Devices enumerated by DirectShow supply their index->name mapping, so
    Femto Mega Depth/IR pins are recorded as non-video WITHOUT opening them:
    opening the depth pin wedges the UVC family and the RGB pin then refuses
    to open ("camera is not opening").

    Returns a list of dicts: index, backend, width, height, fps, name
    (plus ``non_video`` reason for depth/IR pins).
    """
    names = device_names() or []
    if names:
        return _scan_with_names(names, max_index)
    return _scan_without_names(max_index)


def _safe_probe(index: int, backend):
    try:
        return _probe_with(index, backend)
    except Exception:
        return None


def _scan_with_names(names: list, max_index: int):
    found = []
    limit = min(len(names), max_index)
    for i in range(limit):
        name = names[i]
        reason = non_video_pin_reason(name)
        if reason:
            found.append({"index": i, "backend": "dshow", "width": 0, "height": 0,
                          "fps": -1.0, "name": name, "non_video": reason})
            continue
        info = _safe_probe(i, cv2.CAP_DSHOW)
        if info:
            info["name"] = name
            found.append(info)
    if not any(not f.get("non_video") for f in found):
        # Rare: device visible only through the Media Foundation stack.
        for i in range(limit):
            name = names[i]
            if non_video_pin_reason(name):
                continue
            info = _safe_probe(i, cv2.CAP_MSMF)
            if info:
                info["name"] = name
                found.append(info)
    return found


def _scan_without_names(max_index: int):
    """Fallback when device names are unavailable.

    Never open index 0 before the other indexes: on the Orbbec Femto Mega
    index 0 is the depth pin, and opening it wedges the UVC family inside
    this process (later opens fail until restart).  Index 0 is tried last
    and only when nothing else was found (single-camera machines).
    """
    found = []
    misses = 0
    for i in list(range(1, max_index)) + [0]:
        if i == 0 and found:
            break
        info = _safe_probe(i, cv2.CAP_DSHOW)
        if info:
            found.append(info)
            misses = 0
        else:
            misses += 1
            if i != 0 and misses >= 3:
                break
    if not found:
        for i in range(min(4, max_index)):
            info = _safe_probe(i, cv2.CAP_MSMF)
            if info:
                found.append(info)
    return found


def open_source(kind: str, index: int | None = None, url: str | None = None,
                path: str | None = None, attempts: int = 5, settle: float = 0.7):
    """Open a capture source. Returns cv2.VideoCapture or None.

    USB opens are retried: DirectShow can refuse a camera that was released
    moments ago (previous owner still tearing down), which used to leave the
    app stuck on "could not open source" until a page reload.  A capture is
    only accepted once it has delivered one real frame.
    """
    if kind == "usb":
        for attempt in range(attempts):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                try:
                    cap = cv2.VideoCapture(int(index), backend)
                except Exception:
                    cap = None
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    ok, frame = False, None
                    try:
                        for _ in range(6):
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                break
                            time.sleep(0.15)
                    except Exception:
                        ok = False
                    if ok and frame is not None:
                        return cap
                    cap.release()
                elif cap:
                    cap.release()
            if attempt < attempts - 1:
                time.sleep(settle)
        return None
    if kind == "url":
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap if cap.isOpened() else None
    if kind == "file":
        p = Path(path)
        if not p.exists():
            return None
        return cv2.VideoCapture(str(p)) if cv2.VideoCapture(str(p)).isOpened() else None
    return None


def source_label(kind: str, index=None, url=None, path=None) -> str:
    if kind == "usb":
        return f"USB camera {index}"
    if kind == "url":
        return f"Stream {url}"
    if kind == "file":
        return f"File {Path(path).name}"
    return "unknown"
