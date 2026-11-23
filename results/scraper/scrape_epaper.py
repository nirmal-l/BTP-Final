#!/usr/bin/env python
"""Download the front page and editorial page of each Indian Express (Jaipur) issue.

    .venv/bin/python scraper/scrape_epaper.py --start 2025-01-01 --end 2025-12-31

Writes raw_scraped/YYYY-MM-DD_front.png and YYYY-MM-DD_editorial.png (no editorial on
Sundays) plus raw_scraped/manifest.csv.  Already-downloaded days are skipped, so a
stopped run can simply be started again.  Then run:

    .venv/bin/python run_pipeline.py --input raw_scraped
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper import config, detect, session  # noqa: E402

log = logging.getLogger("scraper")
FIELDS = ["date", "weekday", "issue_id", "page_type", "page_num", "level", "width", "height",
          "found_by", "status", "file", "headlines"]


def issues_between(s, start: dt.date, end: dt.date):
    out, seen = [], set()
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        data = session.get(s, config.ISSUES_URL.format(pid=config.PRODUCT_ID,
                                                       month=cur.month, year=cur.year))
        items = data.get("results", []) if isinstance(data, dict) else data
        for it in items:
            try:
                d = dt.datetime.strptime(it["name"].strip(), "%B %d, %Y").date()
            except (KeyError, ValueError):
                d = dt.datetime.fromtimestamp(int(it["publishDate"])).date()
            if start <= d <= end and it["issueId"] not in seen:
                seen.add(it["issueId"])
                out.append((d, str(it["issueId"]), it.get("description", "")))
        cur = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return sorted(out)


def _headlines(desc: str) -> str:
    import html
    import re
    parts = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip(" -* \r\n")
             for p in desc.split("</p>")]
    return " || ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=config.HERE.parent / "raw_scraped")
    ap.add_argument("--force", action="store_true", help="re-download days already done")
    args = ap.parse_args()
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out / "scrape.log")])

    man_path = args.out / "manifest.csv"
    done = set()
    if man_path.exists():
        with man_path.open() as fh:
            for r in csv.DictReader(fh):
                if r["status"] in ("ok", "not_expected"):
                    done.add((r["date"], r["page_type"]))
    new_file = not man_path.exists()
    man = man_path.open("a", newline="")
    writer = csv.DictWriter(man, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    try:
        s = session.build_session()
        issues = issues_between(s, start, end)
    except session.SessionExpired as e:
        sys.exit(f"Login problem: {e}")
    log.info("%d issues between %s and %s", len(issues), start, end)

    for n, (d, issue_id, desc) in enumerate(issues, 1):
        sunday = d.weekday() == 6
        want = ["front"] + ([] if sunday else ["editorial"])
        todo = [t for t in want if args.force or (d.isoformat(), t) not in done]
        if not todo:
            continue
        log.info("[%d/%d] %s %s (issue %s)", n, len(issues), d, d.strftime("%a"), issue_id)
        base = {"date": d.isoformat(), "weekday": d.strftime("%a"), "issue_id": issue_id,
                "headlines": _headlines(desc)}
        try:
            pages = detect.parse_pages(session.get(s, config.PAGEMETA_URL.format(issue_id=issue_id)))
            front_page = None
            for ptype in ["front"] + [t for t in todo if t != "front"]:
                if ptype == "front":
                    front_page, how = detect.find_front(pages)
                    if "front" not in todo:
                        continue
                    page = front_page
                else:
                    page, how = detect.find_editorial(pages, front_page)
                row = {**base, "page_type": ptype, "found_by": how}
                if page is None:
                    writer.writerow({**row, "status": "not_found"})
                    log.warning("  %s page not found", ptype)
                    continue
                img = detect.stitch(page)
                if img is None:
                    writer.writerow({**row, "page_num": page.pagenum, "status": "tiles_missing"})
                    continue
                f = args.out / f"{d.isoformat()}_{ptype}.png"
                img.save(f, optimize=True)
                writer.writerow({**row, "page_num": page.pagenum, "level": page.level,
                                 "width": page.width, "height": page.height, "status": "ok",
                                 "file": f.name})
                log.info("  %s = page %d (%s, %s %dx%d)", ptype, page.pagenum, how, page.level,
                         page.width, page.height)
                time.sleep(config.DELAY_BETWEEN_PAGES)
            if sunday and not args.force and (d.isoformat(), "editorial") not in done:
                writer.writerow({**base, "page_type": "editorial", "status": "not_expected",
                                 "found_by": "Sunday - no editorial page"})
        except session.SessionExpired as e:
            log.error("Session expired (%s). Run scraper/login_once.py, then re-run this "
                      "command - finished days are kept.", e)
            break
        except session.CaptchaOrBlocked as e:
            log.error("The site is blocking requests (%s). Stopping; wait a while and re-run.", e)
            break
        except Exception as e:  # noqa: BLE001
            log.exception("  failed on %s: %s - will retry next run", d, e)
        man.flush()
        time.sleep(config.DELAY_BETWEEN_DAYS)
    man.close()
    log.info("Done. Pages are in %s; next: run_pipeline.py --input %s", args.out, args.out)


if __name__ == "__main__":
    main()
