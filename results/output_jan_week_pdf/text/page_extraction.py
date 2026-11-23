"""
Fetches page manifests, reconstructs high-resolution composite pages from tiled
chunks (combining JPEG raster backgrounds with PNG text overlay layers),
detects editorial pages via enhanced header OCR, and extracts full article text.
"""

import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pytesseract
import requests
from PIL import Image, ImageEnhance

from config import (
    AD_HEADER_MARKERS,
    DELAY_BETWEEN_CHUNKS_SECONDS,
    EDITORIAL_HEADER_MARKERS,
    FRONT_PAGE_MASTHEAD_MARKERS,
    IDEAS_HEADER_MARKERS,
    MAX_FRONT_PAGE_SCAN_DEPTH,
    MAX_PAGES_GUESS,
    OCR_IMAGE_LEVEL,
    PAGEMETA_URL_TEMPLATE,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    SECOND_PAGE_HEADER_MARKERS,
    TESSERACT_CONFIG,
    TESSERACT_LANG,
)

log = logging.getLogger("ie_historical")


@dataclass
class PageInfo:
    pagenum: int
    header_url: str
    level_name: str
    width: int
    height: int
    jpg_chunks: list[dict] = field(default_factory=list)
    png_chunks: list[dict] = field(default_factory=list)


def get_page_manifest(session: requests.Session, issue_id: str) -> dict:
    """Fetch the raw pagemeta JSON for an issue."""
    url = PAGEMETA_URL_TEMPLATE.format(issue_id=issue_id, max_pages=MAX_PAGES_GUESS)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning("Pagemeta request failed for issue %s (attempt %d/%d): %s", issue_id, attempt, RETRY_ATTEMPTS, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                log.error("Failed to fetch page manifest for issue %s: %s", issue_id, e)
                raise


def parse_pages(manifest: dict, preferred_level: str = OCR_IMAGE_LEVEL) -> list[PageInfo]:
    """
    Extract PageInfo for every page in the manifest.
    Collects all JPEG raster and PNG text overlay chunks with their (tx, ty) coordinates.
    """
    pages = []
    for key, page_data in manifest.items():
        try:
            pagenum = page_data.get("pagenum")
            if pagenum is None:
                continue

            levels = page_data.get("levels", {})
            header_entry = levels.get("header", {})
            header_chunks = header_entry.get("chunks", [])
            header_url = header_chunks[0]["url"] if header_chunks else ""

            # Pick preferred resolution level (e.g. level3 -> level2 -> leveldefault)
            selected_level_name = preferred_level
            level_entry = levels.get(preferred_level)
            if not level_entry:
                for fallback in ["level3", "level2", "level1", "leveldefault"]:
                    if fallback in levels:
                        selected_level_name = fallback
                        level_entry = levels[fallback]
                        break

            if not level_entry:
                level_entry = next(iter(levels.values()), {})
                selected_level_name = "unknown"

            width = level_entry.get("width", 2400)
            height = level_entry.get("height", 3339)
            all_chunks = level_entry.get("chunks", [])

            jpg_chunks = [c for c in all_chunks if c.get("url", "").endswith(".jpg")]
            png_chunks = [c for c in all_chunks if c.get("url", "").endswith(".png")]

            # If no .jpg chunks found by extension, treat all chunks as base chunks
            if not jpg_chunks and not png_chunks and all_chunks:
                jpg_chunks = all_chunks

            pages.append(
                PageInfo(
                    pagenum=pagenum,
                    header_url=header_url,
                    level_name=selected_level_name,
                    width=width,
                    height=height,
                    jpg_chunks=jpg_chunks,
                    png_chunks=png_chunks,
                )
            )
        except Exception as e:
            log.warning("Skipping malformed page entry %r: %s", key, e)
            continue

    pages.sort(key=lambda p: p.pagenum)
    return pages


_IMG_SESSION = requests.Session()
_img_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
_IMG_SESSION.mount("https://", _img_adapter)
_IMG_SESSION.mount("http://", _img_adapter)
_IMG_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
})


def _download_image(session: Optional[requests.Session], url: str) -> Optional[Image.Image]:
    """Download an image with retries and return as PIL Image using clean CDN connection."""
    if not url:
        return None
    
    # Use clean image session to avoid sending massive cookie payload/Auth headers to CDN
    fetcher = _IMG_SESSION
    last_exc = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = fetcher.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        except Exception as e:
            last_exc = e
            log.debug("Image download failed (attempt %d/%d) %s: %s", attempt, RETRY_ATTEMPTS, url, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)

    log.warning("Giving up downloading image %s: %s", url, last_exc)
    return None


def stitch_page_image(session: requests.Session, page: PageInfo) -> Optional[Image.Image]:
    """
    Reconstructs the full broadsheet page image by compositing:
    1. Base white canvas
    2. Background JPEG tiles at (tx, ty)
    3. Foreground PNG vector/text overlay tiles at (tx, ty) with transparency
    """
    try:
        # Create blank white background
        canvas = Image.new("RGBA", (page.width, page.height), (255, 255, 255, 255))

        # 1. Paste JPEG background tiles
        for chunk in page.jpg_chunks:
            tile_url = chunk.get("url")
            tx = chunk.get("tx", 0)
            ty = chunk.get("ty", 0)
            tile_img = _download_image(session, tile_url)
            if tile_img:
                tile_rgba = tile_img.convert("RGBA")
                canvas.paste(tile_rgba, (tx, ty))
            if DELAY_BETWEEN_CHUNKS_SECONDS > 0:
                time.sleep(DELAY_BETWEEN_CHUNKS_SECONDS)

        # 2. Alpha-composite PNG text overlay tiles
        for chunk in page.png_chunks:
            tile_url = chunk.get("url")
            tx = chunk.get("tx", 0)
            ty = chunk.get("ty", 0)
            tile_img = _download_image(session, tile_url)
            if tile_img:
                tile_rgba = tile_img.convert("RGBA")
                canvas.paste(tile_rgba, (tx, ty), mask=tile_rgba)
            if DELAY_BETWEEN_CHUNKS_SECONDS > 0:
                time.sleep(DELAY_BETWEEN_CHUNKS_SECONDS)

        # Convert composite to high-contrast grayscale for OCR
        return canvas.convert("L")
    except Exception as e:
        log.error("Failed to stitch composite image for page %d: %s", page.pagenum, e)
        return None


def _ocr_page_header(session: requests.Session, page: PageInfo) -> tuple[str, str, str]:
    """
    Downloads and runs dual-pass OCR on a page's small header crop:
    - Pass 1: Raw image OCR
    - Pass 2: 2x Lanczos upscaled + contrast enhanced image OCR
    Returns (raw_text, enhanced_text, combined_text).
    """
    if not page.header_url:
        return ("", "", "")

    raw_img = _download_image(session, page.header_url)
    if raw_img is None:
        return ("", "", "")

    # Pass 1: Raw OCR
    try:
        raw_text = pytesseract.image_to_string(raw_img, lang=TESSERACT_LANG).strip().replace("\n", " ").lower()
    except Exception:
        raw_text = ""

    # Pass 2: Enhanced OCR (upscale 2x + high contrast)
    try:
        w, h = raw_img.size
        enhanced_img = raw_img.resize((w * 2, h * 2), Image.Resampling.LANCZOS).convert("L")
        enhanced_img = ImageEnhance.Contrast(enhanced_img).enhance(2.0)
        enhanced_text = pytesseract.image_to_string(enhanced_img, lang=TESSERACT_LANG).strip().replace("\n", " ").lower()
    except Exception:
        enhanced_text = ""

    combined_header = f"{raw_text} | {enhanced_text}"
    return (raw_text, enhanced_text, combined_header)


def _is_ad_header_text(txt: str) -> bool:
    """Matches advertisement, sponsored, or brand connect markers."""
    if not txt:
        return False
    t = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    for m in AD_HEADER_MARKERS:
        if m in t:
            return True
    if re.search(r"\b(?:advt|advertisement|consumer\s+connect|brand\s+connect|impact\s+feature|promotional\s+feature|sponsored\s+feature)\b", t):
        return True
    return False


def _is_front_page_masthead_text(txt: str) -> bool:
    """Matches 'The Indian Express' / 'Journalism of Courage' front-page masthead markers."""
    if not txt:
        return False
    t = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    for m in FRONT_PAGE_MASTHEAD_MARKERS:
        if m in t:
            return True
    if re.search(r"\b(?:indian\s+express|journalism\s+of\s+courage)\b", t):
        return True
    return False


def _is_second_page_header_text(txt: str) -> bool:
    """Matches 'The Second Page' section header on Page 2."""
    if not txt:
        return False
    t = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    for m in SECOND_PAGE_HEADER_MARKERS:
        if m in t:
            return True
    if re.search(r"\b(?:the\s+)?sec[o0]nd\s+pa[gce]{2,3}\b|\bpage\s*2\b|\b2\s+(?:the\s+)?indian\s+express\b", t):
        return True
    return False


def _is_editorial_header_text(txt: str) -> bool:
    """Matches direct and common OCR misreadings of 'The Editorial Page' / 'Opinion'."""
    if not txt:
        return False
    t = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    if any(m in t for m in EDITORIAL_HEADER_MARKERS):
        return True
    if re.search(r"\b(?:e[a-z]{5,9}\s+pa[gce]{2,3}|e[dp][it1l][ro0]{1,3}[a-z]{0,4}|opinio?n)\b", t):
        if not re.search(r"\bi[dp]eas?\s+pa[gce]{2,3}\b", t):
            return True
    return False


def _is_ideas_header_text(txt: str) -> bool:
    """Matches 'The Ideas Page'."""
    if not txt:
        return False
    t = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    if any(m in t for m in IDEAS_HEADER_MARKERS):
        return True
    return bool(re.search(r"\bi[dp]eas?\s+pa[gce]{2,3}\b|ideas\s+page|idea\s+exchange", t))


def find_front_page(session: requests.Session, pages: list[PageInfo]) -> Optional[PageInfo]:
    """
    Identifies the genuine news Front Page by:
    1. Skipping full-page / jacket advertisement pages (detecting 'Advt', 'Brand Connect', etc.).
    2. Looking ahead to verify if the subsequent page carries 'The Second Page' header.
    3. Matching 'The Indian Express' / 'Journalism of Courage' masthead on candidate page.
    Falls back to the first non-ad page or page 1 if inconclusive.
    """
    if not pages:
        return None

    scan_depth = min(MAX_FRONT_PAGE_SCAN_DEPTH, len(pages))
    header_texts: list[tuple[str, str, str]] = []

    # Pre-fetch and OCR headers for the first candidate pages
    for i in range(scan_depth):
        header_texts.append(_ocr_page_header(session, pages[i]))

    # Step 1: Lookahead check for "The Second Page"
    # If pages[i+1] is "The Second Page", then pages[i] is guaranteed to be the Front Page.
    for i in range(scan_depth - 1):
        curr_page = pages[i]
        next_page = pages[i + 1]
        curr_raw, curr_enh, curr_comb = header_texts[i]
        next_raw, next_enh, next_comb = header_texts[i + 1]

        # Ensure current page is not an ad jacket
        if _is_ad_header_text(curr_raw) or _is_ad_header_text(curr_enh) or _is_ad_header_text(curr_comb):
            log.info("Page %d is an advertisement jacket. Skipping...", curr_page.pagenum)
            continue

        # Check if next page is "The Second Page"
        if _is_second_page_header_text(next_raw) or _is_second_page_header_text(next_enh) or _is_second_page_header_text(next_comb):
            log.info(
                "Front Page detected on page %d (Confirmed by 'The Second Page' on page %d; Next Header: %r)",
                curr_page.pagenum,
                next_page.pagenum,
                next_comb[:90],
            )
            return curr_page

    # Step 2: Direct Masthead Match
    for i in range(scan_depth):
        curr_page = pages[i]
        curr_raw, curr_enh, curr_comb = header_texts[i]

        if _is_ad_header_text(curr_raw) or _is_ad_header_text(curr_enh) or _is_ad_header_text(curr_comb):
            log.info("Page %d is an advertisement jacket. Skipping...", curr_page.pagenum)
            continue

        if _is_front_page_masthead_text(curr_raw) or _is_front_page_masthead_text(curr_enh) or _is_front_page_masthead_text(curr_comb):
            log.info("Front Page detected on page %d via Masthead Match (Header: %r)", curr_page.pagenum, curr_comb[:90])
            return curr_page

    # Step 3: Fallback - First non-ad page among initial pages
    for i in range(scan_depth):
        curr_page = pages[i]
        curr_raw, curr_enh, curr_comb = header_texts[i]
        if not (_is_ad_header_text(curr_raw) or _is_ad_header_text(curr_enh) or _is_ad_header_text(curr_comb)):
            log.info("Front Page fallback selected on page %d (first non-ad page)", curr_page.pagenum)
            return curr_page

    # Ultimate fallback: Page 1
    log.warning("Could not definitively detect front page via header OCR; falling back to page 1")
    return get_front_page(pages)


def find_editorial_page(session: requests.Session, pages: list[PageInfo]) -> Optional[PageInfo]:
    """
    Identify the editorial page by running dual-pass OCR on each page's small header crop:
    - Pass 1: Raw image OCR
    - Pass 2: 2x Lanczos upscaled + contrast enhanced image OCR
    Matches markers like 'the editorial page', 'editorial', or 'opinion'.
    If header OCR is inconclusive, falls back to the page immediately preceding 'The Ideas Page'.
    """
    ideas_page_idx = None

    for idx, page in enumerate(pages):
        raw_text, enhanced_text, combined_header = _ocr_page_header(session, page)
        if not combined_header.strip():
            continue

        # 1. Match Editorial Page directly
        if _is_editorial_header_text(raw_text) or _is_editorial_header_text(enhanced_text) or _is_editorial_header_text(combined_header):
            log.info("Editorial page detected on page %d (Header: %r)", page.pagenum, combined_header[:90])
            return page

        # 2. Track Ideas Page position for fallback
        if _is_ideas_header_text(raw_text) or _is_ideas_header_text(enhanced_text) or _is_ideas_header_text(combined_header):
            ideas_page_idx = idx

    # Fallback: In Indian Express, the Editorial Page always immediately precedes the Ideas Page
    if ideas_page_idx is not None and ideas_page_idx > 0:
        fallback_page = pages[ideas_page_idx - 1]
        log.info("Editorial page detected via Ideas Page predecessor on page %d", fallback_page.pagenum)
        return fallback_page

    log.info("No editorial page header match found across %d pages", len(pages))
    return None


def get_front_page(pages: list[PageInfo]) -> Optional[PageInfo]:
    """Fallback helper: returns page where pagenum == 1 or first available page."""
    for page in pages:
        if page.pagenum == 1:
            return page
    return pages[0] if pages else None


def clean_ocr_text(text: str) -> str:
    """
    Clean raw OCR text to optimize for NLP / Topic Modeling:
    - Reconnects hyphenated line splits (e.g. 'insti-\\ntutes' -> 'institutes')
    - Cleans isolated non-ASCII punctuation/noise
    - Normalizes multi-spaces while preserving paragraph breaks
    """
    if not text:
        return ""

    # Rejoin hyphenated linebreaks: "govern-\nment" -> "government"
    text = re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean multiple trailing spaces per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Collapse 3+ consecutive newlines to 2 (paragraph separator)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def ocr_full_page_text(session: requests.Session, page: PageInfo) -> Optional[str]:
    """
    Stitches the high-resolution composite page image and runs Tesseract OCR.
    Returns cleaned article text.
    """
    stitched_img = stitch_page_image(session, page)
    if stitched_img is None:
        return None

    try:
        raw_text = pytesseract.image_to_string(
            stitched_img,
            lang=TESSERACT_LANG,
        )
        cleaned_text = clean_ocr_text(raw_text)
        return cleaned_text if cleaned_text else None
    except Exception as e:
        log.error("OCR extraction failed for page %d: %s", page.pagenum, e)
        return None
