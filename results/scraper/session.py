"""Authenticated HTTP session built from the cookies saved by login_once.py."""
from __future__ import annotations

import json
import logging
import time

import requests

from . import config

log = logging.getLogger("scraper")


class SessionExpired(RuntimeError):
    pass


class CaptchaOrBlocked(RuntimeError):
    pass


def _token_from_storage() -> str | None:
    if not config.COOKIES_FILE.exists():
        raise SessionExpired("No saved session - run: .venv/bin/python scraper/login_once.py")
    cookies = json.loads(config.COOKIES_FILE.read_text())
    for name in ("access_token", "exp_token_jwt"):
        for c in cookies:
            if c["name"] == name and c.get("value"):
                return c["value"]
    if config.STORAGE_FILE.exists():           # some sites keep the token in localStorage
        st = json.loads(config.STORAGE_FILE.read_text())
        for origin in st.get("origins", []):
            for item in origin.get("localStorage", []):
                if item["name"] in ("access_token", "exp_token_jwt", "token"):
                    return item["value"]
    return None


def _find_bearer(obj):
    """The OAuth response shape isn't documented - look for a token-like field."""
    if isinstance(obj, dict):
        for k in ("bearer", "bearer_token", "access_token", "token", "jwt", "id_token"):
            v = obj.get(k)
            if isinstance(v, str) and len(v) > 20:
                return v
        for v in obj.values():
            found = _find_bearer(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_bearer(v)
            if found:
                return found
    return None


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT,
                      "Accept": "application/json, text/plain, */*",
                      "Origin": config.EPAPER_HOME.rstrip("/"),
                      "Referer": config.EPAPER_HOME})
    for c in json.loads(config.COOKIES_FILE.read_text()) if config.COOKIES_FILE.exists() else []:
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
    token = _token_from_storage()
    if not token:
        raise SessionExpired("No login token in saved session - run scraper/login_once.py")
    r = s.get(config.OAUTH_URL.format(token=token), timeout=config.TIMEOUT)
    if r.status_code != 200:
        raise SessionExpired(f"Login token rejected (HTTP {r.status_code}) - "
                             "run scraper/login_once.py again")
    bearer = _find_bearer(r.json())
    if not bearer:
        raise SessionExpired("Could not find a bearer token in the OAuth response")
    s.headers["Authorization"] = f"Bearer {bearer}"
    return s


def get(s: requests.Session, url: str, expect="json"):
    """GET with retries; distinguishes expiry and CAPTCHA/blocking from network errors."""
    delay = 3
    for attempt in range(config.RETRIES + 1):
        try:
            r = s.get(url, timeout=config.TIMEOUT)
        except requests.RequestException as e:
            if attempt == config.RETRIES:
                raise
            log.info("  network error (%s), retrying in %ds", e.__class__.__name__, delay)
            time.sleep(delay)
            delay *= 2
            continue
        ctype = r.headers.get("content-type", "")
        if r.status_code in (401,):
            raise SessionExpired(f"HTTP 401 on {url.split('?')[0]}")
        if r.status_code in (403, 429) or (expect != "html" and "text/html" in ctype
                                           and "captcha" in r.text.lower()):
            raise CaptchaOrBlocked(f"HTTP {r.status_code} / CAPTCHA on {url.split('?')[0]}")
        if r.status_code >= 500 and attempt < config.RETRIES:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        time.sleep(config.DELAY_BETWEEN_REQUESTS)
        return r.json() if expect == "json" else r.content
    raise RuntimeError("unreachable")
