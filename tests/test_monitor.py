"""Live AI monitor tests: digest, prompts, NDJSON streaming, API surface.

No camera, no network, no Ollama server is used — responses are faked.
"""
import json
import sys
import unittest
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app as appmod          # noqa: E402
import live_monitor as lm     # noqa: E402


class FakeStream:
    """Line-iterable response stand-in (no sockets)."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ExplodingStream(FakeStream):
    """Yields the first line, then the connection dies mid-stream."""

    def __iter__(self):
        def gen():
            yield self._lines[0]
            raise OSError("connection reset")
        return gen()


def opener_lines(lines, cls=FakeStream):
    def opener(request, timeout=None):
        opener.requests.append((request, timeout))
        return cls(lines)
    opener.requests = []
    return opener


def ndjson(*events):
    return [(json.dumps(event) + "\n").encode("utf-8") for event in events]


def sample_state():
    return {
        "active": True, "fps": 29.6, "view_plane": "sagittal", "discipline": "running",
        "live": {"pose_found": True, "left_knee": 165.4, "right_knee": 161.2,
                 "trunk_lean_fwd": 4.2, "knee_flex_l": float("nan"),
                 "identity": {"total_swaps": 2, "hidden": {"leg": ["left"]}}},
        "session": {"id": 1, "elapsed": 12.5, "duration": 60,
                    "events": {"IC": 14, "MS": 7, "TO": 14},
                    "events_coverage_pct": 92.0, "events_note": None},
    }


class DigestTests(unittest.TestCase):
    def test_digest_keeps_only_finite_numbers(self):
        digest = lm.build_digest(sample_state())
        self.assertEqual(digest["discipline"], "running")
        self.assertEqual(digest["fps"], 29.6)
        self.assertTrue(digest["pose_found"])
        self.assertEqual(digest["metrics"]["left_knee"], 165.4)
        self.assertNotIn("knee_flex_l", digest["metrics"])       # NaN dropped, never faked
        self.assertEqual(digest["session"]["elapsed_s"], 12.5)
        self.assertEqual(digest["session"]["events"]["IC"], 14)
        self.assertEqual(digest["side_tracking"]["hidden_sides"], {"leg": ["left"]})
        json.dumps(digest, allow_nan=False)                      # JSON-safe by contract

    def test_digest_survives_absent_state(self):
        digest = lm.build_digest(None)
        self.assertEqual(digest["discipline"], "running")
        self.assertEqual(digest["view_plane"], "sagittal")
        self.assertFalse(digest["pose_found"])
        self.assertEqual(digest["metrics"], {})
        json.dumps(digest, allow_nan=False)

    def test_cadence_override_and_reference_bands(self):
        digest = lm.build_digest(sample_state(), cadence_spm=172.4,
                                 bands={"cadence_spm": (160, 180), "junk": "not-a-band"})
        self.assertEqual(digest["cadence_est_spm"], 172)
        self.assertEqual(digest["reference_bands"], {"cadence_spm": [160, 180]})


class MessageTests(unittest.TestCase):
    def test_system_prompt_rules(self):
        messages = lm.build_live_messages(lm.build_digest(sample_state()))
        self.assertEqual(messages[0]["role"], "system")
        system = messages[0]["content"]
        for rule in ("Never diagnose", "Never invent", "view plane",
                     "not clinical norms", "pose_found"):
            self.assertIn(rule, system)

    def test_auto_message_shape_and_history_cleaning(self):
        history = [{"role": "assistant", "content": "first update"},
                   {"role": "bogus", "content": "dropped"},
                   {"role": "user", "content": "x" * 2000}]
        messages = lm.build_live_messages(lm.build_digest(sample_state()),
                                          history=history, mode="auto")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([m["role"] for m in messages[1:-1]], ["assistant", "user"])
        self.assertEqual(len(messages[-2]["content"]), 900)      # clipped
        last = messages[-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("Live measurements JSON", last["content"])
        self.assertIn("next short observation", last["content"])
        self.assertIn("left_knee", last["content"])

    def test_history_capped_to_most_recent(self):
        history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
        messages = lm.build_live_messages({}, history=history)
        self.assertEqual(len(messages), 1 + 8 + 1)
        self.assertEqual(messages[1]["content"], "turn 12")

    def test_ask_message_carries_question(self):
        messages = lm.build_live_messages(lm.build_digest(sample_state()), mode="ask",
                                          question="Is the cadence too low?")
        self.assertIn("The coach asks: Is the cadence too low?", messages[-1]["content"])


class StreamTests(unittest.TestCase):
    def test_streams_content_and_done_event(self):
        opener = opener_lines(ndjson(
            {"message": {"role": "assistant", "content": "Cadence "}, "done": False},
            {"message": {"role": "assistant", "content": "steady."}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "model": "qwen"}))
        events = list(lm.stream_chat([{"role": "user", "content": "hi"}], model="qwen",
                                     opener=opener))
        self.assertEqual("".join(e["chunk"] for e in events if "chunk" in e),
                         "Cadence steady.")
        self.assertEqual(events[-1], {"done": True, "model": "qwen"})
        request, _timeout = opener.requests[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen")
        self.assertTrue(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertTrue(request.full_url.endswith("/api/chat"))
        self.assertEqual(request.get_method(), "POST")

    def test_thinking_only_reply_reports_error(self):
        opener = opener_lines(ndjson(
            {"message": {"thinking": "hmm"}, "done": False},
            {"message": {"content": ""}, "done": True}))
        events = list(lm.stream_chat([], opener=opener))
        self.assertFalse(any("chunk" in e for e in events))
        self.assertIn("no text", events[-1]["error"])

    def test_malformed_lines_are_skipped(self):
        lines = [b"not json\n"] + ndjson({"message": {"content": "ok"}, "done": False},
                                         {"message": {"content": ""}, "done": True})
        events = list(lm.stream_chat([], opener=opener_lines(lines)))
        self.assertIn("ok", "".join(e.get("chunk", "") for e in events))
        self.assertTrue(events[-1].get("done"))

    def test_http_error_is_reported_not_raised(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 500,
                                         "Server Error", {}, None)
        events = list(lm.stream_chat([], model="qwen3.8:latest", opener=opener))
        self.assertEqual(len(events), 1)
        self.assertIn("500", events[0]["error"])
        self.assertIn("qwen3.8:latest", events[0]["error"])

    def test_unreachable_endpoint_is_reported(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("connection refused")
        events = list(lm.stream_chat([], opener=opener))
        self.assertIn("unreachable", events[0]["error"])

    def test_stream_break_midway_reports_error_after_chunks(self):
        opener = opener_lines(ndjson({"message": {"content": "part"}, "done": False}),
                              cls=ExplodingStream)
        events = list(lm.stream_chat([], opener=opener))
        self.assertEqual(events[0], {"chunk": "part"})
        self.assertIn("stream failed", events[1]["error"])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()

    def _patch_stream(self, handler):
        original = appmod.live_monitor.stream_chat
        appmod.live_monitor.stream_chat = handler
        self.addCleanup(lambda: setattr(appmod.live_monitor, "stream_chat", original))

    def test_monitor_chat_streams_ndjson_events(self):
        seen = {}

        def fake_stream(messages, model, endpoint, **kwargs):
            seen["roles"] = [m["role"] for m in messages]
            yield {"chunk": "Cadence "}
            yield {"chunk": "looks steady."}
            yield {"done": True, "model": model}

        self._patch_stream(fake_stream)
        r = self.c.post("/api/monitor/chat",
                        json={"mode": "auto", "cadence_spm": 171.0, "history": []})
        self.assertEqual(r.status_code, 200)
        lines = [json.loads(l) for l in r.data.decode("utf-8").splitlines() if l.strip()]
        self.assertEqual("".join(e["chunk"] for e in lines if "chunk" in e),
                         "Cadence looks steady.")
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(seen["roles"], ["system", "user"])

    def test_ask_mode_requires_question(self):
        r = self.c.post("/api/monitor/chat", json={"mode": "ask"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("question", r.get_json()["error"])

    def test_foreign_host_rejected(self):
        r = self.c.post("/api/monitor/chat", json={"mode": "auto"},
                        headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 403)

    def test_warm_endpoint_returns_ok(self):
        r = self.c.post("/api/monitor/warm", json={})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_index_contains_monitor_panel(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        for marker in (b"btnMonitor", b"chatLog", b"Start live chat", b"/api/monitor"):
            self.assertIn(marker, r.data)


if __name__ == "__main__":
    unittest.main()
