import os
import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACT_DIR = r"C:\Users\tavis\.gemini\antigravity\brain\7f9a58a8-78aa-46b5-bd28-ed39e24a9846"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8501"


def run():
    print("Capturing 07_auto_vs_eco_experiment.png with precise scroll position...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2500)

        # Switch to Smart HVAC & Energy tab
        page.get_by_text("Smart HVAC & Energy").first.click()
        page.wait_for_timeout(3500)

        # Scroll down to experiment section
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 2400")
        page.wait_for_timeout(1000)

        try:
            exp_btn = page.locator("button:has-text('Run Experiment Benchmark')").first
            if exp_btn.is_visible():
                print("Running isolated experiment...")
                exp_btn.click()
                page.wait_for_timeout(4000)
        except Exception as e:
            print(f"Experiment execution error: {e}")

        # Scroll so table and chart are centered
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 2650")
        page.wait_for_timeout(1200)

        print("Capturing 07_auto_vs_eco_experiment.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "07_auto_vs_eco_experiment.png"),
            clip={"x": 260, "y": 30, "width": 1320, "height": 920},
        )

        print("Screenshot 7 successfully captured!")
        browser.close()


if __name__ == "__main__":
    run()

