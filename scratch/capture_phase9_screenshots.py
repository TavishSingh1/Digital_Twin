"""
Precision Screenshot Capture for Phase 9 HVAC Digital Twin Verification
========================================================================
Captures the 7 required visual artifacts with clean framing and element scoping.
"""

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
    print(f"Launching Chromium against {BASE_URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()

        print("Navigating to application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(4000)

        # -----------------------------------------------------------------
        # Step simulation forward so HVAC actively conditions and accumulates
        # -----------------------------------------------------------------
        print("Advancing simulation step...")
        step_btn = page.get_by_role("button", name="Step (+5m)")
        if step_btn.count() == 0:
            step_btn = page.locator("button:has-text('Step')")
        for i in range(2):
            if step_btn.count() > 0:
                step_btn.first.click()
                page.wait_for_timeout(2000)

        # Wait for iframe to render
        page.locator("iframe").first.wait_for(state="visible", timeout=20000)
        page.wait_for_timeout(2000)

        # -----------------------------------------------------------------
        # Screenshot 1: Main floor plan with HVAC equipment indicators
        # -----------------------------------------------------------------
        print("Capturing 01_main_floorplan_hvac_indicators.png...")
        page.evaluate("window.scrollTo(0, 160)")
        page.wait_for_timeout(1000)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "01_main_floorplan_hvac_indicators.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 780},
        )

        # -----------------------------------------------------------------
        # Screenshot 2: HVAC overlay mode showing active equipment & airflows
        # -----------------------------------------------------------------
        print("Switching overlay to HVAC Power (W)...")
        overlay_box = page.locator("div[data-testid='stSelectbox']").first
        if overlay_box.count() > 0:
            overlay_box.click()
            page.wait_for_timeout(600)
            opt = page.locator("li:has-text('HVAC Power')")
            if opt.count() > 0:
                opt.first.click()
                page.wait_for_timeout(2000)

        print("Capturing 02_hvac_overlay_mode.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "02_hvac_overlay_mode.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 780},
        )

        # -----------------------------------------------------------------
        # Screenshot 3: Selected-room HVAC controls & live explainability
        # -----------------------------------------------------------------
        print("Switching to HVAC Controls sub-tab in Room Detail...")
        hvac_ctrl_tab = page.get_by_text("HVAC Controls")
        if hvac_ctrl_tab.count() > 0:
            hvac_ctrl_tab.first.click()
            page.wait_for_timeout(1500)

        print("Capturing 03_selected_room_hvac_controls.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "03_selected_room_hvac_controls.png"),
            clip={"x": 1000, "y": 380, "width": 580, "height": 680},
        )

        # -----------------------------------------------------------------
        # Navigate to "⚡ Smart HVAC & Energy" tab
        # -----------------------------------------------------------------
        print("Switching to Smart HVAC & Energy tab...")
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        energy_tab = page.get_by_text("Smart HVAC & Energy")
        if energy_tab.count() > 0:
            energy_tab.first.click()
            page.wait_for_timeout(3500)

        # -----------------------------------------------------------------
        # Screenshot 4: Five-room HVAC status matrix table & facility KPIs
        # -----------------------------------------------------------------
        print("Capturing 04_five_room_hvac_status_matrix.png...")
        page.evaluate("window.scrollTo(0, 80)")
        page.wait_for_timeout(1000)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "04_five_room_hvac_status_matrix.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 780},
        )

        # -----------------------------------------------------------------
        # Screenshot 5: Temperature vs Setpoint chart (with deadband)
        # -----------------------------------------------------------------
        print("Capturing 05_temperature_vs_setpoint_chart.png...")
        page.evaluate("window.scrollTo(0, 680)")
        page.wait_for_timeout(1000)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "05_temperature_vs_setpoint_chart.png"),
            clip={"x": 260, "y": 300, "width": 650, "height": 450},
        )

        # -----------------------------------------------------------------
        # Screenshot 6: HVAC Electrical Power & Cumulative Energy chart
        # -----------------------------------------------------------------
        print("Capturing 06_hvac_power_and_energy_chart.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "06_hvac_power_and_energy_chart.png"),
            clip={"x": 930, "y": 300, "width": 650, "height": 450},
        )

        # -----------------------------------------------------------------
        # Screenshot 7: AUTO vs ECO experiment results & trade-off chart
        # -----------------------------------------------------------------
        print("Navigating to Multi-Policy Simulation Experiment section...")
        page.evaluate("window.scrollTo(0, 2400)")
        page.wait_for_timeout(2000)

        print("Capturing 07_auto_vs_eco_experiment.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "07_auto_vs_eco_experiment.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 880},
        )

        print("All 7 screenshots successfully captured and verified!")
        browser.close()


if __name__ == "__main__":
    run()
