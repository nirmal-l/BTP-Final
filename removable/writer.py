"""
Writes OCR'd front-page and editorial-page text to disk, organized by date,
and provides helpers for dataset export (JSONL / CSV / metadata) and resume checks.
"""

import csv
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from config import OUTPUT_DIR

log = logging.getLogger("ie_historical")


def is_issue_already_saved(issue_date: dt.date, output_dir: Optional[Path] = None) -> bool:
    """Check if the text files for the given date already exist on disk."""
    base_dir = output_dir or OUTPUT_DIR
    day_dir = base_dir / issue_date.isoformat()
    fp_path = day_dir / "front_page.txt"
    ed_path = day_dir / "editorial_page.txt"
    return fp_path.exists() and ed_path.exists()


def save_issue_text(
    issue_date: dt.date,
    front_page_text: Optional[str],
    editorial_page_text: Optional[str],
    front_page_num: Optional[int] = 1,
    editorial_page_num: Optional[int] = None,
    issue_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save front-page and editorial-page text to output/YYYY-MM-DD/,
    along with a structured meta.json file.
    """
    base_dir = output_dir or OUTPUT_DIR
    day_dir = base_dir / issue_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    is_sunday = issue_date.weekday() == 6

    # 1. Save Front Page
    if front_page_text:
        (day_dir / "front_page.txt").write_text(
            f"Front Page - {issue_date.isoformat()} (page {front_page_num})\n"
            f"{'-' * 50}\n\n{front_page_text}\n",
            encoding="utf-8",
        )
    else:
        (day_dir / "front_page.txt").write_text(
            f"[FRONT PAGE OCR FAILED OR IMAGE UNAVAILABLE - {issue_date.isoformat()}]\n",
            encoding="utf-8",
        )

    # 2. Save Editorial Page
    if editorial_page_text:
        (day_dir / "editorial_page.txt").write_text(
            f"Editorial Page - {issue_date.isoformat()} (page {editorial_page_num})\n"
            f"{'-' * 50}\n\n{editorial_page_text}\n",
            encoding="utf-8",
        )
    elif is_sunday:
        (day_dir / "editorial_page.txt").write_text(
            f"[NO EDITORIAL PAGE - {issue_date.isoformat()} is a Sunday, this is expected]\n",
            encoding="utf-8",
        )
    else:
        (day_dir / "editorial_page.txt").write_text(
            f"[EDITORIAL PAGE NOT FOUND - {issue_date.isoformat()} - "
            "header OCR did not match any page; may need manual check]\n",
            encoding="utf-8",
        )

    # 3. Save Structured Metadata
    meta = {
        "date": issue_date.isoformat(),
        "issue_id": issue_id,
        "is_sunday": is_sunday,
        "front_page_num": front_page_num,
        "front_page_chars": len(front_page_text) if front_page_text else 0,
        "editorial_page_num": editorial_page_num,
        "editorial_page_chars": len(editorial_page_text) if editorial_page_text else 0,
    }
    with open(day_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return day_dir


def append_to_corpus_files(records: list[dict], output_dir: Optional[Path] = None):
    """
    Appends processed issue records to central output/corpus.jsonl and output/corpus.csv
    to make feeding topic modeling algorithms effortless.
    """
    base_dir = output_dir or OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = base_dir / "corpus.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = base_dir / "corpus.csv"
    write_header = not csv_path.exists()
    fieldnames = ["date", "issue_id", "page_type", "page_num", "char_count", "text"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow({
                "date": rec.get("date"),
                "issue_id": rec.get("issue_id"),
                "page_type": rec.get("page_type"),
                "page_num": rec.get("page_num"),
                "char_count": len(rec.get("text", "")),
                "text": rec.get("text", ""),
            })
