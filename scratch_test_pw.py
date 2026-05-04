import os
from playwright.sync_api import sync_playwright

def test_screenshot():
    print("Testing playwright screenshot...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            
            print("Navigating to example.com...")
            page.goto("https://example.com", timeout=30000, wait_until="networkidle")
            
            output_path = "/tmp/test_screenshot.png"
            page.screenshot(path=output_path, full_page=True)
            
            browser.close()
            print(f"Success! Saved to {output_path}")
            if os.path.exists(output_path):
                print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            else:
                print("Error: File not found.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_screenshot()
