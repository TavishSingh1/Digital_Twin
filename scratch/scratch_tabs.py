import sys
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=True,
    )
    page = browser.new_page()
    page.goto("http://localhost:8501")
    page.wait_for_timeout(4000)

    # Click Smart HVAC & Energy tab
    print("Clicking Smart HVAC & Energy tab...")
    page.get_by_text("Smart HVAC & Energy").click()
    page.wait_for_timeout(3000)

    # Check for text in energy view
    has_matrix = page.locator("text=Five-Room HVAC Terminal Status Matrix").count()
    has_analytics = page.locator("text=Time-Series HVAC Analytics").count()
    has_exp = page.locator("text=Reproducible Multi-Policy HVAC Simulation Experiment").count()
    print(f"Energy View Loaded: Matrix={has_matrix}, Analytics={has_analytics}, Experiment={has_exp}")
    browser.close()

