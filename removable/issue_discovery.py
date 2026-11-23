"""
Discovers e-paper issues for a given month/year via the authenticated API:

    GET https://ecommerce.indianexpress.com/api/partner-delivery-data/
        ?productId=4288&month=M&year=Y&size=31

Returns a list of Issue objects with issueId, human date string, and parsed datetime.date.
"""

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    ISSUE_LIST_URL,
    PRODUCT_ID,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
)

log = logging.getLogger("ie_historical")


@dataclass
class Issue:
    issue_id: str
    name: str            # e.g. "March 31, 2025"
    date: dt.date


def _parse_issue_date(item: dict) -> Optional[dt.date]:
    """Parse date from either 'name' string or 'publishDate' unix timestamp."""
    name_str = item.get("name", "").strip()
    if name_str:
        try:
            return dt.datetime.strptime(name_str, "%B %d, %Y").date()
        except ValueError:
            pass

    # Fallback to publishDate timestamp if available
    publish_ts = item.get("publishDate")
    if publish_ts:
        try:
            return dt.datetime.fromtimestamp(int(publish_ts)).date()
        except (ValueError, OSError):
            pass

    return None


def get_issues_for_month(session: requests.Session, year: int, month: int) -> list[Issue]:
    """
    Fetch all issues for a given month/year.
    """
    params = {
        "productId": PRODUCT_ID,
        "month": month,
        "year": year,
        "size": 31,
    }
    last_exc = None
    data = {}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(ISSUE_LIST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code >= 500:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code} Server Error: {resp.text[:200]}")
                raise last_exc
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            last_exc = e
            log.warning(
                "Issue list request failed for %04d-%02d (attempt %d/%d): %s",
                year,
                month,
                attempt,
                RETRY_ATTEMPTS,
                e,
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                log.error("Giving up fetching issues for %04d-%02d: %s", year, month, last_exc)
                return []

    issues = []
    results = data.get("results", []) if isinstance(data, dict) else []
    for item in results:
        issue_date = _parse_issue_date(item)
        if not issue_date:
            log.warning("Could not parse issue date from item: %s", item)
            continue
        issue_id = str(item.get("issueId") or item.get("id"))
        issues.append(Issue(issue_id=issue_id, name=item.get("name", str(issue_date)), date=issue_date))

    log.info("Found %d issues for %04d-%02d", len(issues), year, month)
    return issues


def get_issues_for_range(
    session: requests.Session, start_date: dt.date, end_date: dt.date
) -> list[Issue]:
    """
    Fetch issues across a date range spanning potentially multiple months,
    filtered strictly within [start_date, end_date] and sorted chronologically.
    """
    all_issues: list[Issue] = []
    seen_issue_ids = set()

    # Walk month by month from start to end
    current = dt.date(start_date.year, start_date.month, 1)
    while current <= end_date:
        month_issues = get_issues_for_month(session, current.year, current.month)
        for issue in month_issues:
            if issue.issue_id in seen_issue_ids:
                continue
            if start_date <= issue.date <= end_date:
                seen_issue_ids.add(issue.issue_id)
                all_issues.append(issue)

        # advance to first of next month
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)

    all_issues.sort(key=lambda i: i.date)
    return all_issues
