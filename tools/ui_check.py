"""UI smoke test for RunTrack: renders the app in headless Chromium, drives the
same API endpoints the UI uses, and saves screenshots to verification/.

Usage: .venv\\Scripts\\python.exe tools\\ui_check.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8780"
OUT = Path(__file__).resolve().parents[1] / "verification"
OUT.mkdir(exist_ok=True)
CLIP = "C:/Users/GAMEPOWER/projects/runtrack/assets/test_run.mp4"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 950})
        page.goto(BASE, wait_until="networkidle")
        time.sleep(1.2)
        page.screenshot(path=str(OUT / "01_ui_empty.png"))
        print("title:", page.title())
        print("status:", page.inner_text("#statusPill"))

        res = page.evaluate(
            """async () => (await (await fetch('/api/camera/open', {method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({kind:'file', path:'%s'})})).json())""" % CLIP)
        print("open camera:", res)
        page.evaluate("document.getElementById('feed').src = '/video_feed?ts=' + Date.now()")
        time.sleep(8)
        page.screenshot(path=str(OUT / "02_ui_live.png"))
        st = page.evaluate("async () => (await (await fetch('/api/state')).json())")
        print("live: pose =", st["live"].get("pose_found"), "| fps =", st["fps"],
              "| knee =", st["live"].get("left_knee"), st["live"].get("right_knee"))

        r = page.evaluate(
            """async () => (await (await fetch('/api/session/start', {method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({name:'UI Test', phone:'000', weight_kg:80,
                height_cm:180, speed_kmh:10, duration_s:30})})).json())""")
        print("session start:", r)
        sid = None
        for _ in range(60):
            time.sleep(1)
            st = page.evaluate("async () => (await (await fetch('/api/state')).json())")
            if st.get("last_report_sid"):
                sid = st["last_report_sid"]
                break
        print("report sid:", sid)
        if sid:
            page.goto(f"{BASE}/report/{sid}/", wait_until="networkidle")
            time.sleep(1.2)
            page.screenshot(path=str(OUT / "03_report.png"))
            page.screenshot(path=str(OUT / "03_report_full.png"), full_page=True)
        # sessions table check
        page.goto(BASE, wait_until="networkidle")
        time.sleep(1.2)
        rows = page.evaluate("document.getElementById('sessRows').children.length")
        print("session rows in UI:", rows)
        page.screenshot(path=str(OUT / "04_sessions_list.png"))
        browser.close()
    print("UI CHECK DONE")


if __name__ == "__main__":
    main()
