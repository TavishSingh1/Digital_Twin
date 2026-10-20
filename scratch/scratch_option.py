import sys
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=True,
    )
    page = browser.new_page(viewport={"width": 1600, "height": 900})
    page.goto("http://localhost:8501")
    page.wait_for_timeout(3000)

    # Click Smart HVAC & Energy tab
    page.get_by_text("Smart HVAC & Energy").first.click()
    page.wait_for_timeout(3000)

    scrollables = page.evaluate("""() => {
        const results = [];
        document.querySelectorAll('*').forEach(el => {
            if (el.scrollHeight > el.clientHeight + 100 && el.clientHeight > 200) {
                results.push({
                    tag: el.tagName,
                    class: el.className,
                    testid: el.getAttribute('data-testid'),
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                    overflowY: window.getComputedStyle(el).overflowY
                });
            }
        });
        return results;
    }""")
    print("Scrollables on Energy tab:", scrollables)
    browser.close()
