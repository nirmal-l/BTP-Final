"""
Main entry point and Python API for the historical Indian Express e-paper scraper.

Usage via CLI:
    python scrape_historical.py --start 2025-03-01 --end 2025-08-31 --skip-existing

Usage in Topic Modeling Pipeline:
    from scrape_historical import scrape_date_range
    df = scrape_date_range("2025-03-01", "2025-03-31", as_dataframe=True)
"""

import pandas
import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import requests

from config import (
    DELAY_BETWEEN_ISSUES_SECONDS,
    LOG_DIR,
    OCR_IMAGE_LEVEL,
    OUTPUT_DIR,
)
from issue_discovery import Issue, get_issues_for_range
from page_extraction import (
    find_editorial_page,
    find_front_page,
    get_front_page,
    get_page_manifest,
    ocr_full_page_text,
    parse_pages,
)
from session import get_authenticated_session
from writer import append_to_corpus_files, is_issue_already_saved, save_issue_text

log = logging.getLogger("ie_historical")


def setup_logging(verbose: bool = False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"historical_{dt.date.today().isoformat()}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def process_single_issue(
    session: requests.Session,
    issue: Issue,
    preferred_level: str = OCR_IMAGE_LEVEL,
    skip_existing: bool = True,
    save_files: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Process one issue: fetch manifest, locate front & editorial pages,
    stitch composite images, run OCR, and optionally save to disk.
    Returns a structured dictionary of results.
    """
    target_output_dir = output_dir or OUTPUT_DIR

    # Check if already processed
    if skip_existing and is_issue_already_saved(issue.date, output_dir=target_output_dir):
        log.info("Skipping %s - already saved on disk", issue.date.isoformat())
        day_dir = target_output_dir / issue.date.isoformat()
        fp_file = day_dir / "front_page.txt"
        ed_file = day_dir / "editorial_page.txt"
        fp_text = fp_file.read_text(encoding="utf-8") if fp_file.exists() else None
        ed_text = ed_file.read_text(encoding="utf-8") if ed_file.exists() else None
        return {
            "date": issue.date.isoformat(),
            "issue_id": issue.issue_id,
            "status": "skipped_existing",
            "front_page_text": fp_text,
            "editorial_page_text": ed_text,
        }

    manifest = get_page_manifest(session, issue.issue_id)
    pages = parse_pages(manifest, preferred_level=preferred_level)
    if not pages:
        log.warning("No pages parsed for issue %s (%s)", issue.issue_id, issue.date)
        return {
            "date": issue.date.isoformat(),
            "issue_id": issue.issue_id,
            "status": "no_pages",
            "front_page_text": None,
            "editorial_page_text": None,
        }

    front_page = find_front_page(session, pages)
    editorial_page = find_editorial_page(session, pages)

    log.info("Running OCR on Front Page (page %s)...", front_page.pagenum if front_page else "N/A")
    front_text = ocr_full_page_text(session, front_page) if front_page else None

    log.info("Running OCR on Editorial Page (page %s)...", editorial_page.pagenum if editorial_page else "N/A")
    editorial_text = ocr_full_page_text(session, editorial_page) if editorial_page else None

    if save_files:
        save_issue_text(
            issue_date=issue.date,
            front_page_text=front_text,
            editorial_page_text=editorial_text,
            front_page_num=front_page.pagenum if front_page else 1,
            editorial_page_num=editorial_page.pagenum if editorial_page else None,
            issue_id=issue.issue_id,
            output_dir=target_output_dir,
        )

    return {
        "date": issue.date.isoformat(),
        "issue_id": issue.issue_id,
        "status": "success",
        "front_page_text": front_text,
        "front_page_num": front_page.pagenum if front_page else 1,
        "editorial_page_text": editorial_text,
        "editorial_page_num": editorial_page.pagenum if editorial_page else None,
    }


def scrape_date_range(
    start_date: Union[str, dt.date],
    end_date: Union[str, dt.date],
    preferred_level: str = OCR_IMAGE_LEVEL,
    skip_existing: bool = True,
    save_files: bool = True,
    export_corpus: bool = True,
    as_dataframe: bool = False,
    session: Optional[requests.Session] = None,
    output_dir: Optional[Path] = None,
) -> Union[list[dict], "pandas.DataFrame"]:
    """
    High-level Python API to scrape all issues within [start_date, end_date].
    
    Can return either a list of record dicts or a pandas DataFrame ready for Topic Modeling.
    """
    if isinstance(start_date, str):
        start_date = dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = dt.date.fromisoformat(end_date)

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    sess = session or get_authenticated_session()

    log.info("Discovering issues from %s to %s...", start_date, end_date)
    issues = get_issues_for_range(sess, start_date, end_date)
    log.info("Found %d issues to process", len(issues))

    results = []
    corpus_records = []

    for idx, issue in enumerate(issues, 1):
        log.info("[%d/%d] Processing %s (Issue ID: %s)", idx, len(issues), issue.date, issue.issue_id)
        try:
            res = process_single_issue(
                session=sess,
                issue=issue,
                preferred_level=preferred_level,
                skip_existing=skip_existing,
                save_files=save_files,
                output_dir=output_dir,
            )
            results.append(res)

            # Prepare records for corpus export
            if res.get("front_page_text"):
                corpus_records.append({
                    "date": res["date"],
                    "issue_id": res["issue_id"],
                    "page_type": "front_page",
                    "page_num": res.get("front_page_num", 1),
                    "text": res["front_page_text"],
                })
            if res.get("editorial_page_text"):
                corpus_records.append({
                    "date": res["date"],
                    "issue_id": res["issue_id"],
                    "page_type": "editorial_page",
                    "page_num": res.get("editorial_page_num"),
                    "text": res["editorial_page_text"],
                })

        except Exception as e:
            log.error("Failed processing issue %s (%s): %s", issue.issue_id, issue.date, e)
            results.append({
                "date": issue.date.isoformat(),
                "issue_id": issue.issue_id,
                "status": "error",
                "error": str(e),
            })

        if idx < len(issues) and DELAY_BETWEEN_ISSUES_SECONDS > 0:
            time.sleep(DELAY_BETWEEN_ISSUES_SECONDS)

    if export_corpus and corpus_records:
        append_to_corpus_files(corpus_records, output_dir=output_dir)

    if as_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(corpus_records)
        except ImportError:
            log.warning("pandas is not installed. Returning list of records instead.")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill historical Indian Express front and editorial pages.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--level", default=OCR_IMAGE_LEVEL, choices=["level3", "level2", "level1", "leveldefault"], help="Image resolution level")
    parser.add_argument("--force", action="store_true", help="Overwrite already downloaded dates")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    try:
        start_date = dt.date.fromisoformat(args.start)
        end_date = dt.date.fromisoformat(args.end)
    except ValueError as e:
        log.error("Invalid date format, use YYYY-MM-DD: %s", e)
        return 1

    skip_existing = not args.force
    results = scrape_date_range(
        start_date=start_date,
        end_date=end_date,
        preferred_level=args.level,
        skip_existing=skip_existing,
        save_files=True,
        export_corpus=True,
    )

    n_ok = sum(1 for r in results if r.get("status") in ("success", "skipped_existing"))
    n_failed = sum(1 for r in results if r.get("status") in ("error", "no_pages"))

    log.info("Finished scraping range [%s to %s]. Processed: %d OK, %d Failed.", start_date, end_date, n_ok, n_failed)
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
