#!/usr/bin/env python
"""Check every downloaded page by OCR-ing its own top strip (no network).

    .venv/bin/python scraper/verify_pages.py [--out raw_scraped]

Prints and writes raw_scraped/verify.csv: editorial files whose title does not read
'The Editorial Page', and editorial files that are pixel-identical to that day's front."""
import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scraper import config, detect  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def top_text(f: Path) -> str:
    im = Image.open(f).convert("L")
    strip = im.crop((0, 0, im.width, int(im.height * 0.11)))
    scale = 2800 / im.width
    strip = strip.resize((2800, max(1, int(strip.height * scale))), Image.Resampling.LANCZOS)
    txt = pytesseract.image_to_string(strip).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.HERE.parent / "raw_scraped")
    a = ap.parse_args()
    rows = []
    for f in sorted(a.out.glob("*_editorial.png")):
        d = f.name[:10]
        front = a.out / f"{d}_front.png"
        same = front.exists() and hashlib.md5(front.read_bytes()).digest() == hashlib.md5(f.read_bytes()).digest()
        t = top_text(f)
        ok = detect._is_edit_title(t) and not same
        rows.append({"date": d, "file": f.name, "same_as_front": same, "title_ok": detect._is_edit_title(t),
                     "ok": ok, "top_text": t[:120]})
        if not ok:
            print(d, "same_as_front" if same else "", "|", t[:90], flush=True)
    with (a.out / "verify.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows)} editorial files checked, {len(bad)} need re-download")


if __name__ == "__main__":
    main()
