"""
Automated Browser Screenshot Capture via Chrome DevTools Protocol (CDP)
========================================================================
Uses headless Chrome with CDP to capture pixel-perfect visual verification
screenshots for Phase 8.5.
"""

import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9222
BASE_URL = "http://localhost:8501"
SCREENSHOT_DIR = r"d:\VIT\SEMS\SEMESTER_7\PJT1\Digital_Twin\screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

class BrowserController:
    def __init__(self):
        self.proc = None
        self.ws = None
        self.msg_id = 0

    def start(self):
        cmd = [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={CDP_PORT}",
            "--window-size=1920,1080",
            "about:blank",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for debugging port
        for _ in range(20):
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1)
                if resp.status == 200:
                    break
            except Exception:
                time.sleep(0.3)
        print("Chrome started and CDP ready.")

    async def connect(self):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json")
        targets = json.loads(resp.read().decode())
        page_target = next(t for t in targets if t["type"] == "page")
        ws_url = page_target["webSocketDebuggerUrl"]
        self.ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)
        print("Connected to CDP websocket:", ws_url)
        await self.send("Page.enable")
        await self.send("DOM.enable")

    async def send(self, method, params=None):
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            raw = await self.ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == self.msg_id:
                return msg.get("result")

    async def navigate(self, url):
        await self.send("Page.navigate", {"url": url})
        # Wait for page to settle and Streamlit to render
        await asyncio.sleep(4.0)

    async def set_viewport(self, width, height):
        await self.send("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": width < 800,
        })
        await asyncio.sleep(0.5)

    async def evaluate(self, expression):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
        })
        return res

    async def screenshot(self, filename, full_page=False):
        res = await self.send("Page.captureScreenshot", {
            "format": "png",
            "captureBeyondViewport": full_page,
        })
        img_bytes = base64.b64decode(res["data"])
        out_path = os.path.join(SCREENSHOT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"Captured: {filename} ({len(img_bytes)} bytes)")
        return out_path

    async def close(self):
        if self.ws:
            await self.ws.close()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        print("Browser closed.")


async def run_visual_verification():
    browser = BrowserController()
    browser.start()
    try:
        await browser.connect()

        # 1. Initial application load (Desktop 1920x1080)
        print("1. Capturing 01_initial_load.png...")
        await browser.set_viewport(1920, 1080)
        await browser.navigate(f"{BASE_URL}/")
        for _ in range(30):
            res = await browser.evaluate("Boolean(document.querySelector('iframe, [data-testid=\"stMetric\"]'))")
            if res and res.get("value"):
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(2.5)
        await browser.screenshot("01_initial_load.png")

        # 2. Main floor plan with occupants present
        print("2. Capturing 02_floorplan_occupants.png...")
        # Step simulation so occupants populate rooms
        await browser.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const stepBtn = btns.find(b => b.textContent.includes('Step') || b.textContent.includes('+5 min'));
            if (stepBtn) { stepBtn.click(); }
        """)
        await asyncio.sleep(4.0)
        await browser.evaluate("""
            const ifr = document.querySelector('iframe');
            if (ifr) ifr.scrollIntoView({ behavior: 'instant', block: 'center' });
        """)
        await asyncio.sleep(1.5)
        await browser.screenshot("02_floorplan_occupants.png")

        # 3. Selected room with its detail panel (Select Room 2)
        print("3. Capturing 03_selected_room_detail.png...")
        await browser.navigate(f"{BASE_URL}/?selected_room=Room+2")
        await asyncio.sleep(4.0)
        await browser.evaluate("""
            const ifr = document.querySelector('iframe');
            if (ifr) ifr.scrollIntoView({ behavior: 'instant', block: 'center' });
        """)
        await asyncio.sleep(1.5)
        await browser.screenshot("03_selected_room_detail.png")

        # 4. A simulation step after occupants move
        print("4. Capturing 04_simulation_step_moved.png...")
        await browser.evaluate("""
            const btns = Array.from(document.querySelectorAll('button'));
            const stepBtn = btns.find(b => b.textContent.includes('Step') || b.textContent.includes('+5 min'));
            if (stepBtn) { stepBtn.click(); }
        """)
        await asyncio.sleep(4.0)
        await browser.evaluate("""
            const ifr = document.querySelector('iframe');
            if (ifr) ifr.scrollIntoView({ behavior: 'instant', block: 'center' });
        """)
        await asyncio.sleep(1.5)
        await browser.screenshot("04_simulation_step_moved.png")

        # 5. Narrow viewport (768x1024)
        print("5. Capturing 05_narrow_viewport.png...")
        await browser.set_viewport(768, 1024)
        await asyncio.sleep(1.0)
        await browser.evaluate("""
            const collapseBtn = document.querySelector('[data-testid=\"stSidebarCollapseButton\"] button');
            if (collapseBtn) collapseBtn.click();
            const ifr = document.querySelector('iframe');
            if (ifr) ifr.scrollIntoView({ behavior: 'instant', block: 'center' });
        """)
        await asyncio.sleep(2.0)
        await browser.screenshot("05_narrow_viewport.png")

        # 6. Optional Canvas view enabled
        print("6. Capturing 06_optional_canvas.png...")
        await browser.set_viewport(1920, 1080)
        await browser.evaluate("""
            const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
            if (radios.length >= 3) {
                radios[2].click();
            }
        """)
        await asyncio.sleep(4.0)
        await browser.evaluate("""
            const ifr = document.querySelector('iframe');
            if (ifr) ifr.scrollIntoView({ behavior: 'instant', block: 'center' });
        """)
        await asyncio.sleep(1.0)
        await browser.screenshot("06_optional_canvas.png")

        print("Visual verification screenshots complete!")


    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_visual_verification())

