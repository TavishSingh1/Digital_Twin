import asyncio
import json
import os
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9223

async def main():
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--remote-debugging-port={CDP_PORT}",
        "--window-size=1920,1080",
        "http://localhost:8501",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(2.0)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json")
        targets = json.loads(resp.read().decode())
        page = next(t for t in targets if t["type"] == "page")
        ws = await websockets.connect(page["webSocketDebuggerUrl"])
        print("Connected.")
        await asyncio.sleep(3.0)

        # Evaluate SVG elements in page DOM
        msg = {
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {
                "expression": """
                (() => {
                    const svg = document.querySelector('svg');
                    if (!svg) return 'NO SVG FOUND!';
                    const rects = Array.from(svg.querySelectorAll('rect')).map(r => ({
                        x: r.getAttribute('x'),
                        y: r.getAttribute('y'),
                        w: r.getAttribute('width'),
                        h: r.getAttribute('height'),
                        fill: r.getAttribute('fill'),
                        stroke: r.getAttribute('stroke')
                    }));
                    const texts = Array.from(svg.querySelectorAll('text')).map(t => ({
                        x: t.getAttribute('x'),
                        y: t.getAttribute('y'),
                        text: t.textContent.trim()
                    }));
                    return JSON.stringify({ rectsCount: rects.length, rects: rects.slice(0, 10), textsCount: texts.length, texts });
                })()
                """,
                "returnByValue": True
            }
        }
        await ws.send(json.dumps(msg))
        raw = await ws.recv()
        data = json.loads(raw)
        print("DOM EVAL RESULT:")
        print(data["result"]["result"]["value"])
        await ws.close()
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(main())

