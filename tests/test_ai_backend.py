"""AI backend tests: local-Ollama dispatch vs OpenAI-compatible API (faked).

No network, no Ollama server and no real API key is used — every response is
faked, mirroring the test_monitor.py conventions.
"""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import ai_backend as ab     # noqa: E402
import config               # noqa: E402


class FakeStream:
    """Line-iterable + read()-able response stand-in (no sockets)."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def read(self, *args):
        return b"".join(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_lines(lines, cls=FakeStream):
    def opener(request, timeout=None):
        opener.requests.append((request, timeout))
        return cls(lines)
    opener.requests = []
    return opener


def opener_error(exc):
    def opener(request, timeout=None):
        opener.calls += 1
        raise exc
    opener.calls = 0
    return opener


def sse(*chunks):
    lines = [("data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
              + "\n\n").encode("utf-8") for c in chunks]
    lines.append(b"data: [DONE]\n\n")
    return lines


class ApiMode:
    """Context manager: flip config into API mode, restore after."""

    def __init__(self, url="http://127.0.0.1:9999/v1", key="test-key",
                 model="mock-model"):
        self.values = (url, key, model)

    def __enter__(self):
        self.old = (config.AI_API_URL, config.AI_API_KEY, config.AI_API_MODEL)
        config.AI_API_URL, config.AI_API_KEY, config.AI_API_MODEL = self.values
        return self

    def __exit__(self, *exc):
        config.AI_API_URL, config.AI_API_KEY, config.AI_API_MODEL = self.old
        return False


class ModeTests(unittest.TestCase):
    def test_api_mode_flag_and_active_model(self):
        with ApiMode():
            self.assertTrue(ab.api_mode())
            self.assertEqual(ab.active_model(), "mock-model")
            self.assertIn("mock-model", ab.status_text())
        self.assertFalse(ab.api_mode())
        self.assertEqual(ab.active_model(), config.OLLAMA_MODEL)
        self.assertIn(config.OLLAMA_MODEL, ab.status_text())


class LocalDispatchTests(unittest.TestCase):
    def test_local_mode_calls_ollama_chat(self):
        opener = opener_lines([])          # empty stream -> terminal error event
        events = list(ab.chat_stream([{"role": "user", "content": "hi"}],
                                     opener=opener))
        request = opener.requests[0][0]
        self.assertTrue(request.full_url.endswith("/api/chat"))
        body = json.loads(request.data.decode())
        self.assertEqual(body["model"], config.OLLAMA_MODEL)
        self.assertEqual(
            events,
            [{"error": "local model returned no text (empty or reasoning-only reply)"}])

    def test_local_warm_is_noop_in_api_mode(self):
        with ApiMode():
            self.assertFalse(ab.warm())


class ApiStreamTests(unittest.TestCase):
    def test_api_stream_parses_sse_and_sends_auth(self):
        opener = opener_lines(sse("Hel", "lo"))
        with ApiMode():
            events = list(ab.chat_stream([{"role": "user", "content": "hi"}],
                                         opener=opener))
        request = opener.requests[0][0]
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
        self.assertEqual(request.headers.get("Authorization"), "Bearer test-key")
        self.assertEqual(events, [{"chunk": "Hel"}, {"chunk": "lo"},
                                  {"done": True, "model": "mock-model"}])

    def test_api_stream_without_key_sends_no_auth_header(self):
        opener = opener_lines(sse("x"))
        with ApiMode(key=""):
            list(ab.chat_stream([{"role": "user", "content": "hi"}], opener=opener))
        self.assertIsNone(opener.requests[0][0].headers.get("Authorization"))

    def test_api_stream_empty_reply_is_error(self):
        opener = opener_lines([b"data: [DONE]\n\n"])
        with ApiMode():
            events = list(ab.chat_stream([], opener=opener))
        self.assertEqual(len(events), 1)
        self.assertIn("no text", events[0]["error"])

    def test_api_http_error_names_status(self):
        http = urllib.error.HTTPError("http://x", 401, "Unauthorized", None,
                                      io.BytesIO(b'{"error": "bad key"}'))
        with ApiMode():
            events = list(ab.chat_stream([], opener=opener_error(http)))
        self.assertEqual(len(events), 1)
        self.assertIn("HTTP 401", events[0]["error"])

    def test_api_unreachable_is_reported(self):
        with ApiMode():
            events = list(ab.chat_stream([], opener=opener_error(OSError("refused"))))
        self.assertIn("unreachable", events[0]["error"])

    def test_api_stream_midstream_failure_keeps_partial_text(self):
        class Exploding(FakeStream):
            def __iter__(self):
                def gen():
                    yield self._lines[0]
                    raise OSError("reset")
                return gen()
        opener = opener_lines(sse("A"), cls=Exploding)
        with ApiMode():
            events = list(ab.chat_stream([], opener=opener))
        self.assertEqual(events[0], {"chunk": "A"})
        self.assertIn("stream failed", events[1]["error"])


class DescribeImagesTests(unittest.TestCase):
    def test_local_vision_call_uses_generate(self):
        reply = json.dumps({"response": "local review"}).encode()
        opener = opener_lines([reply])
        text = ab.describe_images("p", ["img1"], opener=opener)
        self.assertEqual(text, "local review")
        request = opener.requests[0][0]
        self.assertTrue(request.full_url.endswith("/api/generate"))
        body = json.loads(request.data.decode())
        self.assertEqual(body["images"], ["img1"])
        self.assertFalse(body["stream"])

    def test_api_vision_payload_is_openai_format(self):
        reply = json.dumps(
            {"choices": [{"message": {"content": "api review"}}]}).encode()
        opener = opener_lines([reply])
        with ApiMode():
            text = ab.describe_images("p", ["AAA", "BBB"], opener=opener)
        self.assertEqual(text, "api review")
        body = json.loads(opener.requests[0][0].data.decode())
        parts = body["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "p"})
        self.assertEqual(parts[1]["image_url"]["url"],
                         "data:image/jpeg;base64,AAA")
        self.assertEqual(len(parts), 3)

    def test_api_empty_response_raises(self):
        reply = json.dumps({"choices": [{"message": {"content": "  "}}]}).encode()
        with ApiMode():
            with self.assertRaises(RuntimeError):
                ab.describe_images("p", [], opener=opener_lines([reply]))

    def test_local_http_error_mentions_status_and_model(self):
        http = urllib.error.HTTPError("http://x", 404, "Not Found", None,
                                      io.BytesIO(b"model not found"))
        with self.assertRaises(RuntimeError) as ctx:
            ab.describe_images("p", [], opener=opener_error(http))
        self.assertIn("404", str(ctx.exception))


class SettingsFileTests(unittest.TestCase):
    def test_settings_file_sets_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_settings.env"
            path.write_text('# comment\nAI_API_URL=https://example.com/v1\n'
                            'not a line\nAI_API_KEY="sekret"\n', encoding="utf-8")
            saved = {k: os.environ.pop(k, None) for k in ("AI_API_URL", "AI_API_KEY")}
            try:
                config._load_ai_settings(path)
                self.assertEqual(os.environ["AI_API_URL"], "https://example.com/v1")
                self.assertEqual(os.environ["AI_API_KEY"], "sekret")
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_real_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_settings.env"
            path.write_text("AI_API_URL=https://file.example/v1\n", encoding="utf-8")
            saved = os.environ.get("AI_API_URL")
            os.environ["AI_API_URL"] = "https://env.example/v1"
            try:
                config._load_ai_settings(path)
                self.assertEqual(os.environ["AI_API_URL"], "https://env.example/v1")
            finally:
                if saved is None:
                    os.environ.pop("AI_API_URL", None)
                else:
                    os.environ["AI_API_URL"] = saved


if __name__ == "__main__":
    unittest.main()
