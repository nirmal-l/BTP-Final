"""Deskew, grayscale, photo/halftone removal, adaptive threshold and rule detection."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from . import config


@dataclass
class Preprocessed:
    color: np.ndarray        # deskewed BGR page (for overlays)
    gray: np.ndarray         # deskewed grayscale
    binary: np.ndarray       # uint8, 255 = paper, 0 = ink; photos whited out
    photos: list             # [x, y, w, h] boxes of photos / halftone / graphics
    hrules: list             # [x0, y, x1] horizontal rules
    vrules: list             # [x, y0, y1] vertical rules
    skew_deg: float


# --------------------------------------------------------------------------- deskew
def estimate_skew(gray: np.ndarray, max_deg: float = config.MAX_SKEW_DEG) -> float:
    """Projection-profile deskew: the angle at which text rows are sharpest."""
    small = cv2.resize(gray, (1200, int(gray.shape[0] * 1200 / gray.shape[1])),
                       interpolation=cv2.INTER_AREA)
    ink = (small < 160).astype(np.float32)
    h, w = ink.shape
    centre = (w / 2, h / 2)

    def score(angle):
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = rot.sum(axis=1)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.25)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05)
    return float(max(fine, key=score))


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# --------------------------------------------------------------------------- photos
def detect_photos(gray: np.ndarray, color: np.ndarray) -> list:
    """Find photos, halftones and coloured graphics.

    Printed/rendered text is almost pure black on white, so a window of text
    has few mid-grey pixels.  Photos (continuous tone) and halftone screens are
    dominated by mid-tones or by very dense ink.  We measure both on a 4x
    reduced grid and keep only big, roughly rectangular blobs, which drops
    coloured headline text and kickers.
    """
    mid = ((gray > 50) & (gray < 205)).astype(np.uint8) * 255
    ink = (gray < 205).astype(np.uint8) * 255
    # strongly coloured pixels (e.g. colour photos printed light) also count
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    sat = ((hsv[..., 1] > 60) & (hsv[..., 2] > 40)).astype(np.uint8) * 255
    f = 4
    size = (gray.shape[1] // f, gray.shape[0] // f)
    mid_s = cv2.resize(mid, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    ink_s = cv2.resize(ink, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sat_s = cv2.resize(sat, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    k = 15  # ~60 px window at full resolution
    mid_f = cv2.blur(mid_s, (k, k))
    ink_f = cv2.blur(ink_s, (k, k))
    sat_f = cv2.blur(sat_s, (k, k))
    cand = ((mid_f > config.PHOTO_MIDTONE_FRAC) | (ink_f > config.PHOTO_INK_FRAC)
            | (sat_f > 0.5)).astype(np.uint8)
    # open removes thin structures (coloured text lines), close fills photo holes
    side = max(3, config.PHOTO_MIN_SIDE_PX // f // 2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((side, side), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        X, Y, W, H = x * f, y * f, w * f, h * f
        if min(W, H) < config.PHOTO_MIN_SIDE_PX or W * H < config.PHOTO_MIN_AREA_PX:
            continue
        if area / float(w * h) < 0.45:      # sprawling, non-rectangular -> not a photo
            continue
        pad = 6
        boxes.append([max(0, X - pad), max(0, Y - pad), W + 2 * pad, H + 2 * pad])
    return boxes


# --------------------------------------------------------------------------- binarise
def binarize(gray: np.ndarray) -> np.ndarray:
    thr = threshold_sauvola(gray, window_size=config.SAUVOLA_WINDOW, k=config.SAUVOLA_K)
    return np.where(gray > thr, 255, 0).astype(np.uint8)


def detect_rules(binary: np.ndarray):
    """Long thin horizontal and vertical printed lines (column and article rules)."""
    ink = (binary == 0).astype(np.uint8) * 255
    L, T = config.RULE_MIN_LEN_PX, config.RULE_MAX_THICK_PX
    # horizontal: close tiny gaps (dotted rules) then keep long runs
    hz = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
    hz = cv2.morphologyEx(hz, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    vt = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    vt = cv2.morphologyEx(vt, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    # dotted / dashed rules: rows of many tiny, flat components.  Full stops and
    # i-dots are too sparse along a text line to form a long run, so they survive.
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    tiny = np.zeros(n, bool)
    tiny[1:] = (st[1:, cv2.CC_STAT_HEIGHT] <= 8) & (st[1:, cv2.CC_STAT_WIDTH] <= 80)
    tiny_mask = tiny[lab].astype(np.uint8) * 255
    run = cv2.dilate(tiny_mask, np.ones((3, 41), np.uint8))
    run = cv2.morphologyEx(run, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    dotted = cv2.bitwise_and(tiny_mask, run)
    hz = cv2.bitwise_or(hz, cv2.dilate(dotted, np.ones((3, 3), np.uint8)))
    hrules, vrules = [], []
    for img, out, horiz in ((hz, hrules, True), (vt, vrules, False)):
        n, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if horiz and h <= T and w >= L:
                out.append([int(x), int(y + h // 2), int(x + w)])
            elif not horiz and w <= T and h >= L:
                out.append([int(x + w // 2), int(y), int(y + h)])
    return hrules, vrules, hz, vt


def preprocess(color: np.ndarray) -> Preprocessed:
    gray0 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    skew = estimate_skew(gray0)
    color = rotate(color, skew)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    photos = detect_photos(gray, color)
    binary = binarize(gray)
    for x, y, w, h in photos:
        binary[y:y + h, x:x + w] = 255
    hrules, vrules, hz, vt = detect_rules(binary)
    # remove rule pixels so they don't confuse OCR
    binary[(hz > 0) | (vt > 0)] = 255
    return Preprocessed(color=color, gray=gray, binary=binary, photos=photos,
                        hrules=hrules, vrules=vrules, skew_deg=round(skew, 2))
