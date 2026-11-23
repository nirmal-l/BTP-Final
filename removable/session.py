"""
Builds an authenticated requests.Session from cookies saved by login_once.py,
including acquiring the necessary JWT Bearer token for e-commerce and issue APIs.
"""

import json
import logging
import sys
from typing import Optional

import requests

from config import COOKIES_PATH, OAUTH_USER_URL, REQUEST_TIMEOUT_SECONDS

log = logging.getLogger("ie_historical")


def get_authenticated_session(cookies_path: Optional[str] = None) -> requests.Session:
    target_path = COOKIES_PATH if cookies_path is None else cookies_path
    if not target_path.exists():
        print(
            f"ERROR: No saved session found at {target_path}.\n"
            "Run `python login_once.py` first to log in and save your session cookies.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            cookies = json.load(f)
    except Exception as e:
        log.error("Failed to read cookies from %s: %s", target_path, e)
        raise

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://indianexpress.com/",
    })

    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    access_token = None
    for cookie in cookies:
        name = cookie.get("name")
        val = cookie.get("value")
        domain = cookie.get("domain", "").lstrip(".")
        if name and val:
            session.cookies.set(name, val, domain=domain)
            if name == "access_token" and not access_token:
                access_token = val

    if not access_token:
        # Check exp_token_jwt as fallback
        for cookie in cookies:
            if cookie.get("name") == "exp_token_jwt":
                access_token = cookie.get("value")
                break

    if not access_token:
        log.warning(
            "No 'access_token' or 'exp_token_jwt' found in cookies. "
            "Re-run `python login_once.py` if authentication fails."
        )
    else:
        # Acquire the Bearer token for the ecommerce API
        oauth_url = OAUTH_USER_URL.format(access_token=access_token)
        try:
            resp = session.get(oauth_url, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                bearer_token = data.get("token")
                if bearer_token:
                    session.headers["Authorization"] = f"Bearer {bearer_token}"
                    log.debug("Successfully acquired Bearer token for session.")
                else:
                    log.warning("OAuth response did not contain 'token' field: %s", data)
            else:
                log.warning(
                    "OAuth user token request returned status %d. "
                    "Session may have expired. If scraping fails, run `python login_once.py`.",
                    resp.status_code,
                )
        except Exception as e:
            log.warning("Could not obtain Bearer token from OAuth endpoint: %s", e)

    return session
