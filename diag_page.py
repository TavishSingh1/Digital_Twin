import asyncio
import json
import subprocess
import time
import urllib.request
import websockets
import sys

sys.stdout.reconfigure(encoding='utf-8')

async def test_page():
    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--remote-debugging-port=9225',
        '--window-size=1920,1080',
        'http://localhost:8501'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(4.0)
        resp = urllib.request.urlopen('http://127.0.0.1:9225/json')
        targets = json.loads(resp.read().decode())
        page = next(t for t in targets if t['type'] == 'page')
        ws = await websockets.connect(page['webSocketDebuggerUrl'])
        expr = """
        Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, [data-testid="stMarkdownContainer"]'))
            .map(e => e.textContent.trim())
            .filter(t => t.length > 0 && t.length < 80)
            .slice(0, 30)
        """
        msg = {
            'id': 1,
            'method': 'Runtime.evaluate',
            'params': {
                'expression': expr,
                'returnByValue': True
            }
        }
        await ws.send(json.dumps(msg))
        raw = await ws.recv()
        print('HEADINGS:', json.loads(raw)['result']['result']['value'])
        await ws.close()
    finally:
        proc.terminate()

if __name__ == '__main__':
    asyncio.run(test_page())
