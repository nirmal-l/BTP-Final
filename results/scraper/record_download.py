#!/usr/bin/env python
"""Developer tool: opens the e-paper with your saved login and records the network
traffic while YOU download one page as PDF, so the download request can be studied.

    .venv/bin/python scraper/record_download.py

Writes scraper/debug/download.har (private: contains your session) and saves the PDF
you download to scraper/debug/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright  # noqa: E402

from scraper import config  # noqa: E402


def main():
    config.DEBUG_DIR.mkdir(exist_ok=True)
    har = config.DEBUG_DIR / "download.har"
    seen = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(storage_state=str(config.STORAGE_FILE),
                                  user_agent=config.USER_AGENT, accept_downloads=True,
                                  viewport={"width": 1400, "height": 900},
                                  record_har_path=str(har), record_har_content="embed",
                                  record_har_url_filter="**/*")
        page = ctx.new_page()

        def on_download(d):
            target = config.DEBUG_DIR / (d.suggested_filename or "downloaded.pdf")
            d.save_as(str(target))
            seen.append(target.name)
            print(f"  >> captured download: {target.name}")

        page.on("download", on_download)
        ctx.on("page", lambda pg: pg.on("download", on_download))
        page.goto(config.EPAPER_HOME)
        print("\n1. In the browser: open any 2025 Jaipur issue (log in again if asked).")
        print("2. Download ONE page as PDF, exactly as you normally do")
        print("   (solve the CAPTCHA if one appears).")
        input("3. When the download has finished, press Enter here... ")
        ctx.close()
        browser.close()
    print(f"\nRecorded network traffic to {har.relative_to(config.HERE.parent)}")
    print("Downloads captured:", seen or "none - did the download start in this window?")


if __name__ == "__main__":
    main()
