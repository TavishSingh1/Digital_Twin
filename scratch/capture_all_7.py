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
    print("Starting precision capture for Phase 9 HVAC Twin...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        # -------------------------------------------------------------
        # Part A: Floor Plan with HVAC Indicators (Default Overlay)
        # -------------------------------------------------------------
        print("1. Navigating to Overview...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        step_btn = page.locator("button:has-text('Step')").first
        for _ in range(2):
            step_btn.click()
            page.wait_for_timeout(1200)

        page.locator("iframe").first.wait_for(state="visible", timeout=20000)
        page.wait_for_timeout(1000)

        print("Capturing 01_main_floorplan_hvac_indicators.png...")
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 220")
        page.wait_for_timeout(800)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "01_main_floorplan_hvac_indicators.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 840},
        )

        # -------------------------------------------------------------
        # Part B: Floor Plan in HVAC Overlay Mode
        # -------------------------------------------------------------
        print("2. Navigating to HVAC Overlay with URL query param...")
        page.goto(f"{BASE_URL}/?overlay=HVAC+Power", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        step_btn = page.locator("button:has-text('Step')").first
        for _ in range(2):
            step_btn.click()
            page.wait_for_timeout(1200)

        page.locator("iframe").first.wait_for(state="visible", timeout=20000)
        page.wait_for_timeout(1000)

        print("Capturing 02_hvac_overlay_mode.png...")
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 220")
        page.wait_for_timeout(800)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "02_hvac_overlay_mode.png"),
            clip={"x": 260, "y": 80, "width": 1320, "height": 840},
        )

        # -------------------------------------------------------------
        # Part C: Selected-Room HVAC Controls & Explainability
        # -------------------------------------------------------------
        print("3. Switching to HVAC Controls sub-tab...")
        page.get_by_text("HVAC Controls").first.click()
        page.wait_for_timeout(1500)

        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 420")
        page.wait_for_timeout(800)

        print("Capturing 03_selected_room_hvac_controls.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "03_selected_room_hvac_controls.png"),
            clip={"x": 980, "y": 60, "width": 600, "height": 880},
        )

        # -------------------------------------------------------------
        # Part D: Smart HVAC & Energy Tab - Status Matrix & KPIs
        # -------------------------------------------------------------
        print("4. Switching to Smart HVAC & Energy tab...")
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 0")
        page.wait_for_timeout(500)
        page.get_by_text("Smart HVAC & Energy").first.click()
        page.wait_for_timeout(4000)

        print("Capturing 04_five_room_hvac_status_matrix.png...")
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 80")
        page.wait_for_timeout(800)
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "04_five_room_hvac_status_matrix.png"),
            clip={"x": 260, "y": 60, "width": 1320, "height": 840},
        )

        # -------------------------------------------------------------
        # Part E: Time-Series HVAC Analytics Charts (Plots 2 and 3)
        # -------------------------------------------------------------
        print("5 & 6. Targeting Time-Series Charts...")
        plots = page.locator("[data-testid='stPlotlyChart']")
        print(f"Detected {plots.count()} stPlotlyChart elements.")

        if plots.count() >= 4:
            print("Capturing 05_temperature_vs_setpoint_chart.png...")
            plots.nth(2).scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            plots.nth(2).screenshot(
                path=os.path.join(ARTIFACT_DIR, "05_temperature_vs_setpoint_chart.png")
            )

            print("Capturing 06_hvac_power_and_energy_chart.png...")
            plots.nth(3).scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            plots.nth(3).screenshot(
                path=os.path.join(ARTIFACT_DIR, "06_hvac_power_and_energy_chart.png")
            )

        # -------------------------------------------------------------
        # Part F: Multi-Policy Simulation Experiment (Plot 8 and Table)
        # -------------------------------------------------------------
        print("7. Scrolling to Experiment Benchmark Section...")
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 1950")
        page.wait_for_timeout(1000)

        try:
            exp_btn = page.locator("button:has-text('Run Experiment Benchmark')").first
            if exp_btn.is_visible():
                print("Running isolated experiment...")
                exp_btn.click()
                page.wait_for_timeout(3500)
        except Exception as e:
            print(f"Experiment execution error: {e}")

        # Scroll so both table and trade-off plot are captured
        page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 2100")
        page.wait_for_timeout(1000)

        print("Capturing 07_auto_vs_eco_experiment.png...")
        page.screenshot(
            path=os.path.join(ARTIFACT_DIR, "07_auto_vs_eco_experiment.png"),
            clip={"x": 260, "y": 40, "width": 1320, "height": 920},
        )

        print("All 7 screenshots successfully captured and framed!")
        browser.close()


if __name__ == "__main__":
    run()
