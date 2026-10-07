"""Live AI monitor for RunTrack — streaming local-model commentary.

While a session is recording, the web UI asks this module for a short
observation about the CURRENT derived numbers (live joint metrics, session
counters, discipline/view plane). The module queries the LOCAL Ollama server
(127.0.0.1:11434 by default) and streams the reply back chunk by chunk as
newline-delimited JSON events for the chat panel in the Live session tab.

Only derived numbers ever leave the process — never frames, video or patient
identity — and nothing is sent to the internet. The commentary describes the
measured numbers of the running session; it is an assistive observation for
the coach, not a diagnosis and not a substitute for an in-person assessment.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.8:latest"
DEFAULT_TIMEOUT_S = 300.0
KEEP_ALIVE = "30m"          # keep the model resident while a session is being watched

# Live frame metrics worth describing to the model (flat, 2D pose estimates).
LIVE_METRIC_KEYS = (
    "left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle",
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "trunk_lean", "trunk_lean_fwd", "head_tilt", "neck_trunk",
    "knee_flex_l", "knee_flex_r", "shin_angle_l", "shin_angle_r",
    "overstride_l", "overstride_r", "pelvic_tilt_deg",
    "knee_valgus_l", "knee_valgus_r",
    "power_est_watts", "power_per_kg_watts",
)
BAND_KEYS = ("cadence_spm", "knee_flexion_max_deg", "trunk_lean_deg")

LIVE_SYSTEM_PROMPT = (
    "You are a live assistant to a running/cycling coach watching a session in RunTrack, "
    "a local 2D video pose-estimation lab. You receive a stream of short updates with the "
    "current DERIVED numbers (video estimates only — not clinical measurements). Write "
    "brief, concrete observations the coach can act on.\n"
    "Hard rules:\n"
    "- Never diagnose injuries or conditions; never give medical or training-prescription "
    "advice beyond technique cues.\n"
    "- Use only the numbers in the JSON. Never invent, extrapolate or round-trip values "
    "that are missing; if a metric is absent, do not comment on it.\n"
    "- The view plane decides what is measured: sagittal shows knee/hip/ankle/elbow/shoulder "
    "angles, shin angle, overstride, trunk lean; frontal shows pelvic tilt, knee valgus/varus "
    "and arm/shoulder symmetry. Never comment on metrics of the other plane.\n"
    "- cadence_est_spm is a rolling video estimate that can lag or glitch — present it as an "
    "estimate. reference_bands are the app's heuristic targets, not clinical norms.\n"
    "- If pose_found is false or the numbers look impossible, say plainly that tracking is "
    "lost or the data unusable instead of discussing values.\n"
    "- 1-3 short sentences, under 70 words, plain text, no headings or lists; refer to left "
    "and right explicitly when both matter.\n"
    "- You are observing, not prescribing: no promises about the rest of the session."
)


def _number(value, places=2):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, places) if math.isfinite(number) else None


def build_digest(state, cadence_spm=None, bands=None):
    """Compact, JSON-safe digest of the live state for the local model.

    ``state`` is the app's /api/state-shaped dict; ``cadence_spm`` may carry the
    client-side rolling cadence estimate (the server does not compute it).
    """
    state = state if isinstance(state, dict) else {}
    live = state.get("live") if isinstance(state.get("live"), dict) else {}
    digest = {
        "discipline": str(state.get("discipline") or "running"),
        "view_plane": str(state.get("view_plane") or "sagittal"),
        "pose_found": bool(live.get("pose_found")),
        "fps": _number(state.get("fps"), 1),
    }
    cad = _number(cadence_spm, 0)
    if cad is None:
        cad = _number(live.get("cadence_spm"), 0)
    if cad is not None:
        digest["cadence_est_spm"] = cad
    metrics = {}
    for key in LIVE_METRIC_KEYS:
        value = _number(live.get(key))
        if value is not None:
            metrics[key] = value
    digest["metrics"] = metrics
    ident = live.get("identity")
    if isinstance(ident, dict):
        sides = {"label_corrections": int(ident.get("total_swaps") or 0)}
        hidden = ident.get("hidden")
        if isinstance(hidden, dict):
            hid = {str(g): [str(s) for s in (v or [])] for g, v in hidden.items() if v}
            if hid:
                sides["hidden_sides"] = hid
        digest["side_tracking"] = sides
    session = state.get("session")
    if isinstance(session, dict):
        digest["session"] = {
            "elapsed_s": _number(session.get("elapsed"), 1),
            "duration_s": _number(session.get("duration"), 0),
            "events": session.get("events"),
            "events_coverage_pct": _number(session.get("events_coverage_pct"), 1),
            "events_note": session.get("events_note"),
        }
    if isinstance(bands, dict):
        ref = {k: [_number(v[0], 1), _number(v[1], 1)]
               for k, v in bands.items()
               if isinstance(v, (list, tuple)) and len(v) >= 2}
        if ref:
            digest["reference_bands"] = ref
    return digest


def _clean_history(history, limit=8, cap=900):
    """Keep only well-formed, bounded user/assistant turns (most recent last)."""
    cleaned = []
    for item in list(history or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:cap]})
    return cleaned


def build_live_messages(digest, history=None, mode="auto", question=None):
    """Chat messages for one live-monitor request (system + history + prompt)."""
    messages = [{"role": "system", "content": LIVE_SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    payload = json.dumps(digest, separators=(",", ":"), allow_nan=False)
    if mode == "ask" and question:
        user = ("Live measurements JSON (only these values are real; an absent value means it "
                "was not measured):\n" + payload
                + "\n\nThe coach asks: " + str(question).strip()[:500]
                + "\nAnswer from these measured numbers only, briefly (under 90 words).")
    else:
        user = ("Live measurements JSON (only these values are real; an absent value means it "
                "was not measured) — this update arrives while the session is running:\n"
                + payload + "\n\nWrite the next short observation for the coach.")
    messages.append({"role": "user", "content": user})
    return messages


def stream_chat(messages, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT,
                timeout=DEFAULT_TIMEOUT_S, opener=None, keep_alive=KEEP_ALIVE,
                num_predict=260):
    """Stream one chat completion; yields NDJSON-friendly event dicts.

    Events: ``{"chunk": str}`` per content fragment, then exactly one terminal
    ``{"done": true, "model": ...}`` or ``{"error": str}``. Never raises.
    """
    opener = opener or urllib.request.urlopen
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        # A thinking model would otherwise spend the whole latency budget on
        # hidden reasoning before any visible text arrives.
        "think": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        yield {"error": f"local model HTTP {exc.code} — is '{model}' installed?"}
        return
    except OSError as exc:
        yield {"error": f"local model unreachable: {exc}"}
        return
    got_content = False
    try:
        with response:
            for raw in response:
                line = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray))
                        else str(raw)).strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue                      # ignore malformed stream lines
                if not isinstance(event, dict):
                    continue
                message = event.get("message")
                content = None
                if isinstance(message, dict):
                    content = message.get("content")
                if content:
                    got_content = True
                    yield {"chunk": str(content)}
                if event.get("done"):
                    break
    except (OSError, ValueError) as exc:
        yield {"error": f"local model stream failed: {exc}"}
        return
    if not got_content:
        yield {"error": "local model returned no text (empty or reasoning-only reply)"}
    else:
        yield {"done": True, "model": model}


def warm(model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT, timeout=180.0,
         opener=None, keep_alive=KEEP_ALIVE):
    """Best-effort model preload so the first live update is not delayed by the load."""
    opener = opener or urllib.request.urlopen
    body = json.dumps({"model": model, "prompt": "", "stream": False,
                       "keep_alive": keep_alive,
                       "options": {"num_predict": 1}}).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with opener(request, timeout=timeout) as response:
            response.read()
    except Exception:
        return False
    return True
