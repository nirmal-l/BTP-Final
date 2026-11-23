"""Finding the front page and the editorial page in an issue.

Ported from the earlier BTP_WebScrapping project (page_extraction.py): each page's
small header thumbnail is OCR'd twice (raw, and 2x upscaled + contrast), then
  front     = first non-advert page whose NEXT page is 'The Second Page'
              -> else first non-advert page with the masthead
              -> else first non-advert page -> else page 1
  editorial = page whose header reads 'The Editorial Page' / 'Opinion'
              -> else the page just before 'The Ideas Page'
"""
from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass, field

import pytesseract
import requests
from PIL import Image, ImageEnhance

from . import config

log = logging.getLogger("scraper")

AD_RE = re.compile(r"\b(?:advt|advertisement|advertorial|consumer\s+connect|brand\s+connect|"
                   r"impact\s+feature|promotional\s+feature|sponsored\s+feature|marketing\s+initiative)\b")
MAST_RE = re.compile(r"\b(?:indian\s+express|journalism\s+of\s+courage)\b")
SECOND_RE = re.compile(r"\b(?:the\s+)?sec[o0]nd\s+pa[gce]{2,3}\b")
EDIT_RE = re.compile(r"\b(?:the\s+)?edit[o0]rial\s+pa[gce]{2,3}\b|\be[a-z]{5,9}\s+pa[gce]{2,3}\b"
                     r"|\bed[i1l]t[o0]r[i1l]al\b|\bopinio?n\b")
IDEAS_RE = re.compile(r"\bi[dp]eas?\s+pa[gce]{2,3}\b|\bidea\s+exchange\b")
# OCR often merges words ("jtheepitorialpage"), so also match with spaces removed
EDIT_COMPACT = re.compile(r"e[dpb][i1l]?t[o0]r[i1l]a[l1]pa[gce]{2}|w[o0]r[a-z0-9]{1,5}w[i1l]se")
IDEAS_COMPACT = re.compile(r"the[i1l][dp]easpa[gce]{2}")


def _is_edit(t: str) -> bool:
    return bool(EDIT_RE.search(t) or EDIT_COMPACT.search(t.replace(" ", "")))


def _is_edit_title(t: str) -> bool:
    """Strict: the page's own title 'The Editorial Page' or its fixed 'Wordly Wise'
    column - not just any story that mentions an editorial or opinion."""
    return bool(EDIT_COMPACT.search(t.replace(" ", "")))


def _is_ideas(t: str) -> bool:
    return bool(IDEAS_RE.search(t) or IDEAS_COMPACT.search(t.replace(" ", "")))


@dataclass
class PageInfo:
    pagenum: int
    header_url: str
    level: str
    width: int
    height: int
    jpg: list = field(default_factory=list)
    png: list = field(default_factory=list)
    header_text: str = ""
    top_text: str = ""
    top_chunks: list = field(default_factory=list)   # level1 tiles of the top row
    top_size: tuple = (0, 0)


_img = requests.Session()
_img.headers.update({"User-Agent": config.USER_AGENT,
                     "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})


def download_image(url: str) -> Image.Image | None:
    """Image server rejects requests carrying the login header - use a clean session."""
    for attempt in range(config.RETRIES):
        try:
            r = _img.get(url, timeout=config.TIMEOUT)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001
            log.debug("image retry %d %s: %s", attempt + 1, url, e)
            time.sleep(2 * (attempt + 1))
    log.warning("  could not download image %s", url.split("?")[0])
    return None


def parse_pages(manifest: dict) -> list[PageInfo]:
    pages = []
    for entry in manifest.values():
        if not isinstance(entry, dict) or entry.get("pagenum") is None:
            continue
        levels = entry.get("levels", {})
        name = next((l for l in config.RESOLUTION_LEVELS if l in levels), None)
        if not name:
            continue
        lv = levels[name]
        chunks = lv.get("chunks", [])
        head = levels.get("header", {}).get("chunks", [])
        small = levels.get("level1") or levels.get("leveldefault") or {}
        pages.append(PageInfo(
            pagenum=int(entry["pagenum"]), header_url=head[0]["url"] if head else "",
            level=name, width=lv["width"], height=lv["height"],
            jpg=[c for c in chunks if c["url"].endswith(".jpg")],
            png=[c for c in chunks if c["url"].endswith(".png")],
            top_chunks=[c for c in small.get("chunks", []) if c.get("ty", 0) == 0],
            top_size=(small.get("width", 0), small.get("height", 0))))
    return sorted(pages, key=lambda p: p.pagenum)


def header_text(page: PageInfo) -> str:
    if page.header_text or not page.header_url:
        return page.header_text
    im = download_image(page.header_url)
    if im is None:
        return ""
    im = im.convert("L")
    texts = []
    for img in (im, ImageEnhance.Contrast(im.resize((im.width * 2, im.height * 2),
                                                    Image.Resampling.LANCZOS)).enhance(2.0)):
        try:
            texts.append(pytesseract.image_to_string(img))
        except Exception:  # noqa: BLE001
            pass
    page.header_text = re.sub(r"[^a-z0-9\s]", " ", " | ".join(texts).lower())
    page.header_text = re.sub(r"\s+", " ", page.header_text)
    time.sleep(0.3)
    return page.header_text


def page_top_text(page: PageInfo) -> str:
    """OCR the section title printed at the top of the page itself (more reliable than
    the separate header thumbnail, which is blank or mismatched on some issues)."""
    if page.top_text or not page.top_chunks:
        return page.top_text
    w = page.top_size[0]
    h = max(c["height"] for c in page.top_chunks)
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    for c in sorted(page.top_chunks, key=lambda c: c["url"].endswith(".png")):
        tile = download_image(c["url"])
        if tile is None:
            continue
        tile = tile.convert("RGBA")
        if c["url"].endswith(".png"):
            canvas.alpha_composite(tile, (c["tx"], 0))
        else:
            canvas.paste(tile, (c["tx"], 0))
        time.sleep(config.DELAY_BETWEEN_TILES)
    strip = canvas.convert("L").crop((0, 0, w, int(page.top_size[1] * 0.11)))
    strip = strip.resize((strip.width * 2, strip.height * 2), Image.Resampling.LANCZOS)
    try:
        txt = pytesseract.image_to_string(strip)
    except Exception:  # noqa: BLE001
        txt = ""
    page.top_text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", txt.lower()))
    return page.top_text


def find_front(pages: list[PageInfo]) -> tuple[PageInfo | None, str]:
    if not pages:
        return None, "no pages"
    scan = pages[:min(config.FRONT_SCAN_DEPTH, len(pages))]
    texts = [header_text(p) for p in scan]
    is_ad = [bool(AD_RE.search(t)) for t in texts]
    for i in range(len(scan) - 1):
        if not is_ad[i] and SECOND_RE.search(texts[i + 1]):
            return scan[i], "next page is 'The Second Page'"
    for i, p in enumerate(scan):
        if not is_ad[i] and MAST_RE.search(texts[i]):
            return p, "masthead"
    for i, p in enumerate(scan):
        if not is_ad[i]:
            return p, "first non-advert page"
    return pages[0], "fallback page 1"


def find_editorial(pages: list[PageInfo], front: PageInfo | None = None
                   ) -> tuple[PageInfo | None, str]:
    """Never returns the front page.  Tries the title OCR'd from the top of each page first,
    then the header thumbnails."""
    cands = [p for p in pages if front is None or p.pagenum != front.pagenum]
    cands = [p for p in cands if p.pagenum > 1] or cands       # editorial is never page 1
    # the page's own top strip first (header thumbnails are sometimes blank); the front
    # page is excluded because from Nov 2025 it carries a "The Editorial Page: ..." teaser
    for get, how, is_edit in ((page_top_text, "page title", _is_edit_title),
                              (header_text, "header", _is_edit)):
        ideas = None
        for i, p in enumerate(cands):
            t = get(p)
            if _is_ideas(t) and not is_edit(t):
                ideas = i if ideas is None else ideas
                continue
            if is_edit(t):
                return p, f"{how} says editorial"
        if ideas is not None and ideas > 0 and get is page_top_text:
            return cands[ideas - 1], "page before 'The Ideas Page'"
    return None, "not found"


def stitch(page: PageInfo) -> Image.Image | None:
    canvas = Image.new("RGBA", (page.width, page.height), (255, 255, 255, 255))
    missing = 0
    for group, is_text in ((page.jpg, False), (page.png, True)):
        for c in group:
            tile = download_image(c["url"])
            if tile is None:
                missing += 1
                continue
            tile = tile.convert("RGBA")
            if is_text:
                canvas.alpha_composite(tile, (c["tx"], c["ty"]))
            else:
                canvas.paste(tile, (c["tx"], c["ty"]))
            time.sleep(config.DELAY_BETWEEN_TILES)
    if missing:
        log.warning("  page %d: %d tiles missing - page not saved", page.pagenum, missing)
        return None
    return canvas.convert("RGB")
