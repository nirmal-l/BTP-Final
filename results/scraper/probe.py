#!/usr/bin/env python
"""Developer check: confirms login works and saves the raw API responses for one
month / one issue to scraper/debug/ so the parser can be matched to them.
Prints only structure (keys, counts), never token values."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper import config, session  # noqa: E402


def shape(o, depth=0, max_depth=4):
    pad = "  " * depth
    if depth > max_depth:
        return pad + "..."
    if isinstance(o, dict):
        return "\n".join(f"{pad}{k}: {type(v).__name__}" + (
            "\n" + shape(v, depth + 1) if isinstance(v, (dict, list)) else "")
            for k, v in list(o.items())[:25])
    if isinstance(o, list):
        return f"{pad}[list of {len(o)}]" + ("\n" + shape(o[0], depth + 1) if o else "")
    return pad + type(o).__name__


def main():
    config.DEBUG_DIR.mkdir(exist_ok=True)
    s = session.build_session()
    print("Login OK - bearer token obtained.")
    issues = session.get(s, config.ISSUES_URL.format(pid=config.PRODUCT_ID, month=1, year=2025))
    (config.DEBUG_DIR / "issues_2025_01.json").write_text(json.dumps(issues, indent=1))
    print("Issues response structure:\n" + shape(issues))
    print("\nSaved to scraper/debug/. Paste nothing here - tell Claude 'probe done'.")


if __name__ == "__main__":
    main()
