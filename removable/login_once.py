"""
Run this ONCE (and again whenever your session expires) to log in manually
via a real browser and save your session cookies to disk. The main scraper
(scrape_historical.py) reuses these cookies with plain HTTP requests -
no browser needed for the actual scraping, since the APIs we use
(partner-delivery-data, pagemeta) are plain authenticated JSON endpoints.

Usage:
    python login_once.py
"""

import json
import sys
from playwright.sync_api import sync_playwright

from config import COOKIES_PATH


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://indianexpress.com", wait_until="domcontentloaded")

        print("\n" + "=" * 70)
        print("A browser window has opened.")
        print("1. Log in with your Indian Express subscriber account.")
        print("2. Navigate to https://indianexpress.com/express-plus/epaper/previous-issues/?productId=4288")
        print("   and confirm you can see past issues listed.")
        print("3. Come back here and press Enter to save your session cookies.")
        print("=" * 70 + "\n")
        input("Press Enter once logged in and confirmed... ")

        cookies = context.cookies()
        with open(COOKIES_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)

        print(f"\nSaved {len(cookies)} cookies to: {COOKIES_PATH}")
        print("You can now run scrape_historical.py.")
        print("NOTE: this cookie file grants access to your logged-in session.")
        print("Keep it private - do not share it or commit it to version control.")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
