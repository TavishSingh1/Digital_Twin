import os
import sys
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(3000)

    # Click Smart HVAC & Energy
    print("Clicking Smart HVAC & Energy...")
    page.get_by_text("Smart HVAC & Energy").first.click()
    page.wait_for_timeout(4000)

    # Scroll down to time-series section
    page.evaluate("document.querySelector('section[data-testid=\"stMain\"]').scrollTop = 700")
    page.wait_for_timeout(2000)

    # Check for errors on page
    errors = page.locator(".stException").all_text_contents()
    if errors:
        print("FOUND EXCEPTIONS ON PAGE:")
        for err in errors:
            print("---")
            print(err)
    else:
        print("No stException found on page! Page rendered cleanly.")

    # Find plotly elements
    plots = page.locator("[data-testid='stPlotlyChart']")
    print(f"Total stPlotlyChart elements: {plots.count()}")
    for i in range(plots.count()):
        box = plots.nth(i).bounding_box()
        visible = plots.nth(i).is_visible()
        print(f"Plot {i}: visible={visible}, box={box}")

    browser.close()

