import os
import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACT_DIR = r"C:\Users\tavis\.gemini\antigravity\brain\7f9a58a8-78aa-46b5-bd28-ed39e24a9846"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=CHROME_PATH,
        headless=True,
    )
    context = browser.new_context(viewport={"width": 1600, "height": 1100})
    page = context.new_page()

    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(3000)

    # Step simulation 2 times
    step_btn = page.locator("button:has-text('Step')").first
    for _ in range(2):
        step_btn.click()
        page.wait_for_timeout(1500)

    # Switch to Smart HVAC & Energy tab
    print("Switching to Smart HVAC & Energy tab...")
    page.get_by_text("Smart HVAC & Energy").first.click()
    page.wait_for_timeout(3500)

    # 1. Scroll to Time-Series HVAC Analytics
    ts_title = page.locator("text=Time-Series HVAC Analytics").first
    ts_title.evaluate("el => el.scrollIntoView({behavior: 'instant', block: 'start'})")
    page.wait_for_timeout(1000)

    # Capture view of charts
    print("Capturing 05_temperature_vs_setpoint_chart.png...")
    page.screenshot(
        path=os.path.join(ARTIFACT_DIR, "05_temperature_vs_setpoint_chart.png"),
        clip={"x": 280, "y": 140, "width": 645, "height": 420},
    )

    print("Capturing 06_hvac_power_and_energy_chart.png...")
    page.screenshot(
        path=os.path.join(ARTIFACT_DIR, "06_hvac_power_and_energy_chart.png"),
        clip={"x": 930, "y": 140, "width": 645, "height": 420},
    )

    # 2. Scroll to Multi-Policy Simulation Experiment
    exp_title = page.locator("text=Reproducible Multi-Policy HVAC Simulation Experiment").first
    exp_title.evaluate("el => el.scrollIntoView({behavior: 'instant', block: 'start'})")
    page.wait_for_timeout(1500)

    print("Capturing 07_auto_vs_eco_experiment.png...")
    page.screenshot(
        path=os.path.join(ARTIFACT_DIR, "07_auto_vs_eco_experiment.png"),
        clip={"x": 280, "y": 100, "width": 1300, "height": 850},
    )

    print("All energy screenshots successfully captured!")
    browser.close()
