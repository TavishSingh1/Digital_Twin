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

    # 1. Step simulation forward
    step_btn = page.locator("button:has-text('Step')").first
    for _ in range(2):
        step_btn.click()
        page.wait_for_timeout(1500)

    # 1. Floor plan iframe
    print("1. Floor plan...")
    iframe_elem = page.locator("iframe").first
    iframe_elem.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    iframe_elem.screenshot(path=os.path.join(ARTIFACT_DIR, "01_main_floorplan_hvac_indicators.png"))

    # 2. Select overlay
    print("2. Overlay...")
    sel_box = page.locator("div[data-testid='stSelectbox']:has-text('Multi-Physics Overlay Layer')")
    if sel_box.count() > 0:
        sel_box.click()
        page.wait_for_timeout(500)
        page.locator("li:has-text('HVAC Power')").click()
        page.wait_for_timeout(1500)
    iframe_elem.screenshot(path=os.path.join(ARTIFACT_DIR, "02_hvac_overlay_mode.png"))

    # 3. Room detail HVAC controls
    print("3. Room detail HVAC controls...")
    page.get_by_text("HVAC Controls").click()
    page.wait_for_timeout(1000)
    # The right column containing room detail
    room_ctrl = page.locator("div[data-testid='stVerticalBlock']:has-text('Control Centre: Room 1')").first
    if room_ctrl.count() > 0:
        room_ctrl.screenshot(path=os.path.join(ARTIFACT_DIR, "03_selected_room_hvac_controls.png"))

    # Switch to Energy tab
    print("4. Switching to Energy tab...")
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    page.get_by_text("Smart HVAC & Energy").click()
    page.wait_for_timeout(3000)

    # 4. Five-room status matrix
    print("4. Five-room matrix...")
    matrix_elem = page.locator("div[data-testid='stVerticalBlock']:has-text('Five-Room HVAC Terminal Status Matrix')").first
    if matrix_elem.count() > 0:
        matrix_elem.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        matrix_elem.screenshot(path=os.path.join(ARTIFACT_DIR, "04_five_room_hvac_status_matrix.png"))

    # 5. Charts
    print("5 & 6. Time series charts...")
    charts_block = page.locator("div[data-testid='stVerticalBlock']:has-text('Time-Series HVAC Analytics & Thermostat Tracking')").first
    if charts_block.count() > 0:
        charts_block.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        plotly_charts = charts_block.locator("div[data-testid='stPlotlyChart']").all()
        print(f"Found {len(plotly_charts)} plotly charts in analytics section")
        if len(plotly_charts) >= 1:
            plotly_charts[0].screenshot(path=os.path.join(ARTIFACT_DIR, "05_temperature_vs_setpoint_chart.png"))
        if len(plotly_charts) >= 2:
            plotly_charts[1].screenshot(path=os.path.join(ARTIFACT_DIR, "06_hvac_power_and_energy_chart.png"))

    # 7. Experiment Benchmark
    print("7. Experiment...")
    exp_block = page.locator("div[data-testid='stVerticalBlock']:has-text('Reproducible Multi-Policy HVAC Simulation Experiment')").first
    if exp_block.count() > 0:
        exp_block.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        exp_block.screenshot(path=os.path.join(ARTIFACT_DIR, "07_auto_vs_eco_experiment.png"))

    print("Element screenshots completed successfully!")
    browser.close()

