"""Browser check for the RunTrack live AI chat monitor (uses the REAL local model).

Drives the actual UI: opens a video-file "camera", starts a session, clicks
"Start live chat", waits for the first streamed model update, then asks a
question in the chat box and waits for the streamed answer. Saves screenshots
to verification/.

Usage (server already running on :8780):
    .venv\\Scripts\\python.exe tools\\ui_monitor_check.py
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8780"
OUT = Path(__file__).resolve().parents[1] / "verification"
OUT.mkdir(exist_ok=True)
CLIP = "C:/Users/GAMEPOWER/AppData/Local/hermes/cache/scratch/long_run.mp4"


def _ensure_clip():
    """The 14.7 s asset auto-finalizes before a slow update can finish; make a longer copy."""
    import subprocess
    if Path(CLIP).exists():
        return
    src = Path(__file__).resolve().parents[1] / "assets" / "test_run.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "20",
                    "-i", str(src), "-c", "copy", CLIP], check=True)


def chat_text(page):
    return page.evaluate("document.getElementById('chatLog').innerText")


def wait_for(page, predicate, timeout=240, poll=2.0, label=""):
    deadline = time.time() + timeout
    last_status = ""
    while time.time() < deadline:
        last_status = page.inner_text("#monStatus")
        if predicate(last_status, chat_text(page)):
            return last_status
        time.sleep(poll)
    raise SystemExit(f"timeout waiting for {label}; last status: {last_status!r}")


def main():
    _ensure_clip()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1860, "height": 1080})
        page.goto(BASE, wait_until="networkidle")
        time.sleep(1.0)

        res = page.evaluate(
            """async () => (await (await fetch('/api/camera/open', {method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({kind:'file', path:'%s'})})).json())""" % CLIP)
        print("open camera:", res)
        page.evaluate("document.getElementById('feed').src = '/video_feed?ts=' + Date.now()")
        time.sleep(3)
        r = page.evaluate(
            """async () => (await (await fetch('/api/session/start', {method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({name:'Chat UI Test', phone:'000', view_plane:'sagittal',
                discipline:'running', duration_s:280})})).json())""")
        print("session start:", r)
        if not r.get("ok"):
            raise SystemExit("session did not start")
        time.sleep(3)

        # --- turn the live chat on (real button click) -----------------------
        page.click("#btnMonitor")
        print("after toggle:", page.inner_text("#monStatus"))

        # --- wait for the first streamed auto update -------------------------
        status = wait_for(
            page,
            lambda st, ct: "last reply" in st and ct.count("LOCAL MODEL") >= 1,
            label="first live update")
        time.sleep(0.5)
        text = chat_text(page)
        print("---- chat after first update ----")
        print(text[:1400])
        print("status:", status)
        page.screenshot(path=str(OUT / "10_monitor_fullpage.png"))
        page.locator("#chatLog").screenshot(path=str(OUT / "11_monitor_chatlog.png"))

        # --- ask a question through the chat box -----------------------------
        before = text.count("LOCAL MODEL")
        page.fill("#chatInput", "Is the cadence on target according to the reference band?")
        page.click("#btnAsk")
        status = wait_for(
            page,
            lambda st, ct: "last reply" in st and ct.count("LOCAL MODEL") > before
            and "THINKING" not in ct.upper(),
            label="streamed answer")
        time.sleep(0.5)
        text = chat_text(page)
        print("---- chat after question ----")
        print(text[:2200])
        print("status:", status)
        page.screenshot(path=str(OUT / "12_monitor_after_question.png"))
        page.locator("#chatLog").screenshot(path=str(OUT / "13_monitor_chatlog_qa.png"))

        # --- wait for the next automatic update (proves the ongoing loop) ----
        before2 = text.count("LOCAL MODEL")
        status = wait_for(
            page,
            lambda st, ct: ct.count("LOCAL MODEL") > before2
            and "THINKING" not in ct.upper(),
            timeout=180, label="next auto update")
        time.sleep(0.5)
        text = chat_text(page)
        print("---- chat after next auto update ----")
        print(text[:2600])
        print("status:", status)
        page.screenshot(path=str(OUT / "14_monitor_next_update.png"))
        page.locator("#chatLog").screenshot(path=str(OUT / "15_monitor_chatlog_loop.png"))

        # --- cleanup: monitor off, session stop ------------------------------
        page.click("#btnMonitor")
        print("monitor off:", page.inner_text("#monStatus"))
        try:
            page.evaluate(
                """async () => (await (await fetch('/api/session/stop', {method:'POST'
                   })).json())""")
        except Exception as exc:
            print("stop session note:", exc)
        browser.close()
    print("MONITOR UI CHECK DONE")


if __name__ == "__main__":
    main()
