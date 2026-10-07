"""AI backend for RunTrack — one pluggable layer: local Ollama by default, or
ANY OpenAI-compatible API (cloud provider or a local server such as LM Studio).

Mode is chosen by `AI_API_URL` (see config.py / README):
  - empty          -> the local Ollama server (OLLAMA_URL / OLLAMA_MODEL)
  - set (any base) -> that endpoint's /chat/completions with AI_API_KEY
                      (optional) and AI_API_MODEL.

The live monitor only ever sends derived numbers. The form review additionally
sends the same up-to-three session frames that a local model already receives —
in API mode those go to the configured provider instead of staying on the PC.
Never put real key values in the repo; use the environment or the gitignored
ai_settings.env file.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import config
import live_monitor

DEFAULT_TIMEOUT_S = live_monitor.DEFAULT_TIMEOUT_S


def api_mode():
    """True when an OpenAI-compatible API endpoint is configured."""
    return bool(config.AI_API_URL)


def active_model():
    """Model name for the active backend (used in the UI and saved reviews)."""
    return config.AI_API_MODEL if api_mode() else config.OLLAMA_MODEL


def status_text():
    """Short label for the UI: which model the AI features are using."""
    if api_mode():
        return f"API model ({config.AI_API_MODEL or 'unset'})"
    return f"local model ({config.OLLAMA_MODEL})"


def _api_headers():
    headers = {"Content-Type": "application/json"}
    if config.AI_API_KEY:
        headers["Authorization"] = "Bearer " + config.AI_API_KEY
    return headers


def _api_url():
    return config.AI_API_URL.rstrip("/") + "/chat/completions"


def _http_detail(exc):
    """Short, safe snippet of an HTTP error body for the user-facing message."""
    try:
        raw = exc.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""
    return (" — " + raw[:200]) if raw else ""


def _sse_events(response):
    """Yield parsed JSON payloads from an OpenAI-style SSE byte stream."""
    for raw in response:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray))
                else str(raw)).strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except ValueError:
            continue                       # ignore malformed stream lines
        if isinstance(event, dict):
            yield event


def chat_stream(messages, timeout=DEFAULT_TIMEOUT_S, opener=None):
    """Stream one chat completion; same event dicts as live_monitor.stream_chat.

    Events: ``{"chunk": str}`` per content fragment, then exactly one terminal
    ``{"done": true, "model": ...}`` or ``{"error": str}``. Never raises.
    """
    opener = opener or urllib.request.urlopen
    if not api_mode():
        yield from live_monitor.stream_chat(messages, config.OLLAMA_MODEL,
                                            config.OLLAMA_URL, timeout=timeout,
                                            opener=opener)
        return
    model = config.AI_API_MODEL
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       "temperature": 0.2}, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(_api_url(), data=body, headers=_api_headers(),
                                     method="POST")
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        yield {"error": f"API HTTP {exc.code}{_http_detail(exc)} — "
                        "check AI_API_URL / AI_API_KEY / AI_API_MODEL"}
        return
    except OSError as exc:
        yield {"error": f"API unreachable: {exc}"}
        return
    got_content = False
    try:
        with response:
            for event in _sse_events(response):
                try:
                    delta = event["choices"][0].get("delta") or {}
                    content = delta.get("content")
                except (KeyError, IndexError, TypeError, AttributeError):
                    content = None
                if content:
                    got_content = True
                    yield {"chunk": str(content)}
    except (OSError, ValueError) as exc:
        yield {"error": f"API stream failed: {exc}"}
        return
    if not got_content:
        yield {"error": "API returned no text (check AI_API_MODEL and that the "
                        "model supports chat)"}
    else:
        yield {"done": True, "model": model}


def describe_images(prompt, images_b64, timeout=280.0, opener=None):
    """One-shot text/vision call for the form review.

    Returns the reply text. Raises RuntimeError with a user-readable message
    on any failure (the caller turns it into the JSON error reply).
    """
    opener = opener or urllib.request.urlopen
    if not api_mode():
        body = json.dumps({"model": config.OLLAMA_MODEL, "prompt": prompt,
                           "images": list(images_b64 or []), "stream": False}).encode()
        request = urllib.request.Request(
            config.OLLAMA_URL.rstrip("/") + "/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with opener(request, timeout=timeout) as response:
                out = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"local model HTTP {exc.code} — is "
                               f"'{config.OLLAMA_MODEL}' installed?") from None
        except OSError as exc:
            raise RuntimeError(f"local model unreachable: {exc}") from None
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"unreadable local model reply: {exc}") from None
        text = (out.get("response") or "").strip() if isinstance(out, dict) else ""
        if not text:
            raise RuntimeError("empty model response")
        return text
    # API mode: OpenAI vision message format (data URLs).  AI_API_MODEL must
    # name a vision-capable model for the form review to see the frames.
    parts = [{"type": "text", "text": prompt}]
    for image in list(images_b64 or [])[:3]:
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64," + str(image)}})
    body = json.dumps({"model": config.AI_API_MODEL,
                       "messages": [{"role": "user", "content": parts}],
                       "stream": False}, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(_api_url(), data=body, headers=_api_headers(),
                                     method="POST")
    try:
        with opener(request, timeout=timeout) as response:
            out = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API HTTP {exc.code}{_http_detail(exc)} — "
                           "check AI_API_URL / AI_API_KEY / AI_API_MODEL") from None
    except OSError as exc:
        raise RuntimeError(f"API unreachable: {exc}") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"unreadable API reply: {exc}") from None
    try:
        text = str(out["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        raise RuntimeError("API returned an unexpected reply shape") from None
    if not text:
        raise RuntimeError("empty model response")
    return text


def warm(opener=None):
    """Best-effort local model preload; nothing to preload for an API backend."""
    if api_mode():
        return False
    try:
        return live_monitor.warm(config.OLLAMA_MODEL, config.OLLAMA_URL,
                                 opener=opener)
    except Exception:
        return False
