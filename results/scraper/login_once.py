#!/usr/bin/env python
"""One-time login.  Opens a real browser; YOU log in; your session is saved locally.

    .venv/bin/python scraper/login_once.py

Saves scraper/session_cookies.json and scraper/storage_state.json (private - do not
share or commit them).  Re-run whenever the scraper reports an expired session.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright  # noqa: E402

from scraper import config  # noqa: E402


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent=config.USER_AGENT,
                                      viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.goto(config.EPAPER_HOME)
        print("\n1. Log in to your Indian Express account in the browser window.")
        print("2. Open the e-paper (Jaipur edition) and check a 2025 issue loads.")
        input("3. Then come back here and press Enter to save your session... ")
        context.storage_state(path=str(config.STORAGE_FILE))
        cookies = context.cookies()
        config.COOKIES_FILE.write_text(json.dumps(cookies, indent=1))
        for f in (config.COOKIES_FILE, config.STORAGE_FILE):
            os.chmod(f, 0o600)
        names = sorted({c["name"] for c in cookies})
        print(f"\nSaved {len(cookies)} cookies to {config.COOKIES_FILE.name}.")
        has_token = any(n in names for n in ("access_token", "exp_token_jwt"))
        print("Login token cookie found." if has_token else
              "WARNING: no access_token cookie found - are you logged in?")
        browser.close()


if __name__ == "__main__":
    main()
