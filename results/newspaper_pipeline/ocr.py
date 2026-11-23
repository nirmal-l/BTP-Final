"""OCR each article separately: headline crops and body blocks in reading order."""
from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from . import config


def _crop(img, box, pad=8):
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    return img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def ocr_box(binary: np.ndarray, box) -> tuple[str, list]:
    """OCR one block (psm 6 = a single uniform block of text). Returns text, word confs."""
    crop = _crop(binary, box)
    if crop.size == 0:
        return "", []
    d = pytesseract.image_to_data(crop, config=config.TESS_BLOCK_CONFIG,
                                  output_type=Output.DICT)
    lines: dict = {}
    confs = []
    for i in range(len(d["level"])):
        if d["level"][i] != 5:
            continue
        t = d["text"][i].strip()
        c = float(d["conf"][i])
        if not t or c < 0:
            continue
        confs.append(c)
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(t)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text, confs


def _drop_repeats(parts, min_overlap=0.6):
    """Remove pull quotes: blocks whose text mostly repeats another block of the same article."""
    from .textclean import shingles
    sh = [shingles(p) for p in parts]
    keep = []
    for i, p in enumerate(parts):
        others = set().union(*(sh[j] for j in range(len(parts)) if j != i and len(sh[j]) > len(sh[i])))
        if sh[i] and len(sh[i] & others) / len(sh[i]) >= min_overlap:
            continue
        keep.append(p)
    return keep


def ocr_article(binary: np.ndarray, art: dict) -> dict:
    head_parts, confs = [], []
    for box in sorted(art["headline_boxes"], key=lambda b: b[1]):
        t, c = ocr_box(binary, box)
        head_parts.append(t)
        confs += c
    deck_parts = []
    for box in sorted(art["deck_boxes"], key=lambda b: b[1]):
        t, c = ocr_box(binary, box)
        deck_parts.append(t)
    body_parts, body_confs = [], []
    for box in art["body_boxes"]:              # already in reading order
        t, c = ocr_box(binary, box)
        body_parts.append(t)
        body_confs += c
    body_parts = _drop_repeats(body_parts)
    return {
        "headline_raw": "\n".join(head_parts),
        "deck_raw": "\n".join(deck_parts),
        "body_raw": "\n\n".join(body_parts),
        "ocr_conf": float(np.mean(body_confs)) if body_confs else None,
        "ocr_low_conf_share": float(np.mean(np.array(body_confs) < 60)) if body_confs else None,
    }
