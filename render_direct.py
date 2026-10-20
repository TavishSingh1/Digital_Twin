import asyncio
import json
import base64
import subprocess
import urllib.request
import websockets

async def capture_local():
    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--remote-debugging-port=9226',
        '--window-size=1200,900',
        'file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/screenshots/floorplan.html'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(2.0)
        resp = urllib.request.urlopen('http://127.0.0.1:9226/json')
        targets = json.loads(resp.read().decode())
        page = next(t for t in targets if t['type'] == 'page')
        ws = await websockets.connect(page['webSocketDebuggerUrl'])
        res = await ws.recv() # wait for ready
        msg = {'id': 1, 'method': 'Page.captureScreenshot', 'params': {'format': 'png'}}
        await ws.send(json.dumps(msg))
        raw = await ws.recv()
        data = json.loads(raw)
        with open('screenshots/direct_floorplan.png', 'wb') as f:
            f.write(base64.b64decode(data['result']['data']))
        print('Captured screenshots/direct_floorplan.png successfully!')
        await ws.close()
    finally:
        proc.terminate()

if __name__ == '__main__':
    asyncio.run(capture_local())

