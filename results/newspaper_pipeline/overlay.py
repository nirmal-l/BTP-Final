"""Draw segmentation results on the page so they can be checked by eye."""
from __future__ import annotations

import cv2
import numpy as np

# reference categorical palette (BGR), fixed order
PALETTE = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237),
           (164, 123, 232), (0, 131, 0), (214, 80, 98), (72, 73, 227)]
SKIP = (60, 60, 200)      # masthead / footer / captions
PHOTO = (150, 150, 150)
ORPHAN = (180, 0, 180)


def _fill(img, box, color, alpha):
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    if sub.size:
        img[y0:y1, x0:x1] = (sub * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def draw(color_img, layout, articles_meta, photos, out_path, width=1800):
    img = color_img.copy()
    s = width / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def sc(b):
        return [int(round(v * s)) for v in b]

    for x, y, w, h in photos:
        bx = sc([x, y, x + w, y + h])
        _fill(img, bx, PHOTO, 0.45)
        cv2.rectangle(img, bx[:2], bx[2:], PHOTO, 2)
    for b in layout["blocks"]:
        if b["role"] in ("masthead", "footer", "caption"):
            bx = sc(b["bbox"])
            _fill(img, bx, SKIP, 0.25)
            cv2.rectangle(img, bx[:2], bx[2:], SKIP, 1)
    for meta in articles_meta:
        a = meta["layout"]
        kept = meta["kept"]
        col = ORPHAN if a.get("orphan") else PALETTE[(meta["n"] - 1) % len(PALETTE)]
        thick = 3 if kept else 1
        for box in a["body_boxes"]:
            bx = sc(box)
            _fill(img, bx, col, 0.12 if kept else 0.04)
            cv2.rectangle(img, bx[:2], bx[2:], col, thick)
        for box in a["headline_boxes"] + a["deck_boxes"]:
            bx = sc(box)
            _fill(img, bx, col, 0.30 if kept else 0.08)
            cv2.rectangle(img, bx[:2], bx[2:], col, thick + 1)
        anchor = (a["headline_boxes"] or a["body_boxes"])[0]
        ax, ay = sc(anchor)[:2]
        tag = f"{meta['n']}" + ("" if kept else " x")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ay = max(0, ay - th - 12)          # label sits just above the box
        cv2.rectangle(img, (ax, ay), (ax + tw + 10, ay + th + 10), col, -1)
        cv2.putText(img, tag, (ax + 5, ay + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    # legend
    legend = [("article (number = id, 'x' = dropped)", PALETTE[0]),
              ("no headline found (orphan)", ORPHAN), ("photo / graphic removed", PHOTO),
              ("skipped: masthead, footer, caption", SKIP)]
    y = 10
    cv2.rectangle(img, (5, 5), (520, 20 + 26 * len(legend)), (255, 255, 255), -1)
    for text, c in legend:
        cv2.rectangle(img, (12, y + 4), (32, y + 20), c, -1)
        cv2.putText(img, text, (40, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)
        y += 26
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
