"""Finding input pages, parsing dates from filenames, and rendering them to pixels."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from . import config

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}

# 2026-11-10_front.jpg
_ISO = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?:_(?P<ptype>front|editorial))?", re.I)
# Jaipur-January-01--2025-page-1 (1).pdf   (Indian Express e-paper downloads)
_EPAPER = re.compile(
    r"^(?P<city>[A-Za-z]+)-(?P<month>[A-Za-z]+)-(?P<d>\d{1,2})--(?P<y>\d{4})"
    r"(?:-page-(?P<page>\d+))?(?:\s*\((?P<dup>\d+)\))?", re.I)


@dataclass
class PageInfo:
    page_id: str
    source_file: str
    path: str
    date: str | None
    filename_page_type: str | None   # only if the filename states it
    edition: str | None

    def to_dict(self):
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def parse_filename(path: Path) -> PageInfo:
    stem = path.stem
    d, ptype, edition = None, None, None
    m = _ISO.match(stem)
    if m:
        d = date(int(m["y"]), int(m["m"]), int(m["d"])).isoformat()
        ptype = m["ptype"].lower() if m["ptype"] else None
    else:
        m = _EPAPER.match(stem)
        if m and m["month"].lower() in _MONTHS:
            d = date(int(m["y"]), _MONTHS[m["month"].lower()], int(m["d"])).isoformat()
            edition = m["city"]
    return PageInfo(page_id=_slug(stem), source_file=path.name, path=str(path),
                    date=d, filename_page_type=ptype, edition=edition)


def discover(input_dir: Path) -> list[PageInfo]:
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS | config.PDF_EXTS)
    return [parse_filename(p) for p in files]


def render(page: PageInfo, width_px: int = config.RENDER_WIDTH_PX) -> np.ndarray:
    """Return the page as a BGR uint8 image, scaled to ``width_px`` wide.

    The source file is opened read-only and never modified.
    """
    path = Path(page.path)
    if path.suffix.lower() in config.PDF_EXTS:
        import fitz  # PyMuPDF
        with fitz.open(path) as doc:
            pg = doc[0]
            zoom = width_px / pg.rect.width
            pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False,
                                colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    scale = width_px / img.shape[1]
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
