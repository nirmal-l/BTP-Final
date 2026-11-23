"""Page segmentation: text lines -> blocks -> headline groups -> articles.

Strategy (rule based, no trained layout model):

1. Tesseract (psm 3) finds text lines and their heights on the cleaned page.
   Lines that cross a vertical column rule are split at the rule.
2. Lines are chained into blocks: a line joins the block directly above it if
   they overlap horizontally, have similar type size, sit close together and no
   horizontal rule separates them.
3. Each block gets a type-size class relative to the page's body text height H:
   head (>= 1.75 H), deck (kickers / standfirsts), body, small (credits, tables).
4. Masthead, footer strip, text inside photos and photo captions are skipped.
5. Adjacent head/deck blocks form a headline group.  Each body block is joined
   to whatever sits directly above it in its column (a headline group or an
   already-assigned body block), never across a horizontal rule.
6. Blocks of an article are put in reading order: columns left -> right, each
   column top -> bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytesseract
from pytesseract import Output

from . import config


# --------------------------------------------------------------------------- data
@dataclass
class Line:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    conf: float
    n_words: int

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0


@dataclass
class Block:
    id: int
    lines: list = field(default_factory=list)
    kind: str = "body"           # head | deck | body | small | dropcap
    role: str = "text"           # text | masthead | footer | caption | in_photo
    article: int | None = None
    group: int | None = None

    @property
    def x0(self): return min(l.x0 for l in self.lines)
    @property
    def y0(self): return min(l.y0 for l in self.lines)
    @property
    def x1(self): return max(l.x1 for l in self.lines)
    @property
    def y1(self): return max(l.y1 for l in self.lines)
    @property
    def w(self): return self.x1 - self.x0
    @property
    def line_h(self):
        # upper quartile: lines with ascenders+descenders give the true type size
        return float(np.percentile([l.h for l in self.lines], 75))
    @property
    def n_words(self): return sum(l.n_words for l in self.lines)
    @property
    def text(self): return "\n".join(l.text for l in self.lines)

    def bbox(self):
        return [self.x0, self.y0, self.x1, self.y1]


def _xoverlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _rule_between(hrules, x0, x1, ytop, ybot, min_frac=0.5, ignore=()):
    """True if a horizontal rule lies between ytop and ybot covering the x-range.

    ``ignore`` is a list of (y0, y1) spans whose rules don't count (rules that
    close off a photo + caption, which sit *inside* an article)."""
    width = max(1, x1 - x0)
    for rx0, ry, rx1 in hrules:
        if any(a <= ry <= b for a, b in ignore):
            continue
        if ytop - 2 <= ry <= ybot + 2 and _xoverlap(x0, x1, rx0, rx1) >= min_frac * width:
            return True
    return False


# --------------------------------------------------------------------------- step 1
def ocr_lines(binary: np.ndarray, vrules: list) -> list[Line]:
    d = pytesseract.image_to_data(binary, config=config.TESS_LAYOUT_CONFIG,
                                  output_type=Output.DICT)
    words_by_line: dict = {}
    line_boxes: dict = {}
    for i in range(len(d["level"])):
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if d["level"][i] == 4:
            line_boxes[key] = (d["top"][i], d["height"][i])
        if d["level"][i] != 5:
            continue
        txt = d["text"][i].strip()
        conf = float(d["conf"][i])
        if not txt or conf < 0:
            continue
        words_by_line.setdefault(key, []).append(
            (d["left"][i], d["top"][i], d["left"][i] + d["width"][i],
             d["top"][i] + d["height"][i], txt, conf))

    raw = []
    for key, words in words_by_line.items():
        words.sort(key=lambda w: w[0])
        top, height = line_boxes.get(key, (min(w[1] for w in words),
                                           max(w[3] for w in words) - min(w[1] for w in words)))
        raw.append((top, height, words))
    H = body_height([_make_line(t, h, w) for t, h, w in raw])

    lines = []
    for top, height, words in raw:
        for seg in _split_words(top, height, words, vrules, H):
            lines.append(_make_line(top, height, seg))
    lines += _recover_missed(binary, lines, H)
    # drop junk: low confidence and almost no letters
    keep = []
    for l in lines:
        alpha = sum(c.isalpha() for c in l.text)
        if l.conf < config.MIN_LINE_CONF and alpha < 6:
            continue
        if alpha == 0:
            continue
        keep.append(l)
    return keep


def _recover_missed(binary, lines, H) -> list[Line]:
    """Second chance for large type the page-level layout pass skipped.

    Tesseract's page layout analysis occasionally discards a whole headline
    (e.g. one sitting under a dotted rule).  Find big patches of ink that no
    detected line covers and OCR each patch on its own.
    """
    import cv2
    h, w = binary.shape
    ink = (binary == 0).astype(np.uint8)
    for l in lines:
        ink[max(0, l.y0 - 6):l.y1 + 6, max(0, l.x0 - 6):l.x1 + 6] = 0
    f = 4
    small = cv2.resize(ink * 255, (w // f, h // f), interpolation=cv2.INTER_AREA) > 20
    kx, ky = max(3, int(H / f)), max(1, int(H / f / 3))
    blob = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ky, kx), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    found = []
    for i in range(1, n):
        x, y, bw, bh = (int(v) * f for v in st[i, :4])
        if bh < config.DECK_RATIO * H or bw < 4 * H or bh > 8 * H:
            continue
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        crop = np.full((min(h, y + bh + pad) - y0, min(w, x + bw + pad) - x0), 255, np.uint8)
        crop[ink[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] > 0] = 0
        d = pytesseract.image_to_data(crop, config=config.TESS_BLOCK_CONFIG,
                                      output_type=Output.DICT)
        by_line, boxes = {}, {}
        for j in range(len(d["level"])):
            key = (d["block_num"][j], d["par_num"][j], d["line_num"][j])
            if d["level"][j] == 4:
                boxes[key] = (d["top"][j] + y0, d["height"][j])
            if d["level"][j] == 5 and d["text"][j].strip() and float(d["conf"][j]) >= 50:
                by_line.setdefault(key, []).append(
                    (d["left"][j] + x0, d["top"][j] + y0, d["left"][j] + d["width"][j] + x0,
                     d["top"][j] + d["height"][j] + y0, d["text"][j].strip(), float(d["conf"][j])))
        for key, words in by_line.items():
            words.sort(key=lambda t: t[0])
            top, height = boxes[key]
            if sum(c.isalpha() for t in words for c in t[4]) >= 4:
                found.append(_make_line(top, height, words))
    return found


def _make_line(top, height, words) -> Line:
    return Line(x0=min(w[0] for w in words), y0=top, x1=max(w[2] for w in words),
                y1=top + height, text=" ".join(w[4] for w in words),
                conf=float(np.mean([w[5] for w in words])), n_words=len(words))


def _split_words(top, height, words, vrules, H):
    """Split a Tesseract line that runs across a column boundary.

    * text-size lines: cut where a vertical column rule crosses the line.
    * display lines (headlines): cut at an unusually wide gap between words -
      two headlines side by side are often read as one line.  Vertical rules are
      not used here because column rules sometimes run up into a headline.
    """
    cuts = sorted(rx for rx, ry0, ry1 in vrules
                  if ry0 <= top + height / 2 <= ry1 and words[0][0] < rx < words[-1][2])
    segs, cur, ci = [], [], 0
    for w in words:
        while ci < len(cuts) and w[0] > cuts[ci]:
            if cur:
                segs.append(cur)
            cur, ci = [], ci + 1
        cur.append(w)
    segs += [cur] if cur else []
    if height < config.DECK_RATIO * H:
        return segs
    out = []
    for seg in segs:
        gaps = [b[0] - a[2] for a, b in zip(seg, seg[1:])]
        if not gaps:
            out.append(seg)
            continue
        limit = max(0.45 * height, 2.5 * float(np.median(gaps)))
        cur = [seg[0]]
        for g, w in zip(gaps, seg[1:]):
            if g > limit:
                out.append(cur)
                cur = []
            cur.append(w)
        out.append(cur)
    return out


def body_height(lines: list[Line]) -> float:
    hs = [l.h for l in lines if l.n_words >= 3]
    ws = [l.n_words for l in lines if l.n_words >= 3]
    if not hs:
        return float(np.median([l.h for l in lines]))
    order = np.argsort(hs)
    cum = np.cumsum(np.array(ws)[order])
    return float(np.array(hs)[order][np.searchsorted(cum, cum[-1] / 2)])


# --------------------------------------------------------------------------- step 2/3
def size_class(h: float, H: float) -> str:
    r = h / H
    if r >= config.HEADLINE_RATIO:
        return "head"
    if r >= config.DECK_RATIO:
        return "deck"
    if r < config.SMALL_RATIO:
        return "small"
    return "body"


def _coarse(h: float, H: float) -> str:
    """Line-level size class.  A line's box height varies ~25% with ascenders and
    descenders (e.g. 'a rise in us' vs 'High Court'), so only coarse classes."""
    c = size_class(h, H)
    return "text" if c in ("body", "small") else c


def chain_blocks(lines: list[Line], H: float, hrules: list) -> list[Block]:
    lines = sorted(lines, key=lambda l: (l.y0, l.x0))
    blocks: list[Block] = []
    for ln in lines:
        best, best_gap = None, None
        for b in blocks:
            last = max(b.lines, key=lambda l: l.y1)
            gap = ln.y0 - last.y1
            if gap < -0.5 * min(ln.h, last.h) or gap > 0.9 * max(H, min(ln.h, last.h)):
                continue
            ov = _xoverlap(ln.x0, ln.x1, last.x0, last.x1)
            if ov < 0.6 * min(ln.w, last.w):
                continue
            # don't merge across different type sizes
            ratio = ln.h / max(1, last.h)
            if not (0.67 <= ratio <= 1.5):
                continue
            if _rule_between(hrules, max(ln.x0, last.x0), min(ln.x1, last.x1),
                             last.y1 - 4, ln.y0 + 4):
                continue
            if best is None or gap < best_gap:
                best, best_gap = b, gap
        if best is None:
            best = Block(id=len(blocks))
            blocks.append(best)
        best.lines.append(ln)
    for b in blocks:
        b.kind = size_class(b.line_h, H)
        alpha = sum(c.isalpha() for c in b.text)
        if b.kind == "head" and alpha <= 2:
            b.kind = "dropcap"
        elif b.kind in ("head", "deck") and b.w < 1.5 * b.line_h:
            b.kind = "small"          # rotated / vertical label
        elif b.kind == "deck" and len(b.lines) >= 4:
            b.kind = "body"          # pull quotes / large-type intros read as body
    return blocks


# --------------------------------------------------------------------------- step 4
def mark_furniture(blocks, H, page_w, page_h, hrules, photos):
    zone = config.MASTHEAD_ZONE * page_h
    wide = [r for r in hrules if (r[2] - r[0]) >= 0.6 * page_w and r[1] < zone]
    masthead_y = max((r[1] for r in wide), default=0)
    if masthead_y == 0:
        big = [b for b in blocks if b.y1 < zone and b.line_h >= 4 * H]
        masthead_y = max((b.y1 for b in big), default=0)
    footer_y = (1 - config.FOOTER_ZONE) * page_h
    for b in blocks:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        if b.y1 <= masthead_y + 2:
            b.role = "masthead"
        elif b.y0 >= footer_y:
            b.role = "footer"
        elif any(px <= cx <= px + pw and py <= cy <= py + ph for px, py, pw, ph in photos):
            b.role = "in_photo"
    for b in blocks:
        if b.role != "text" or b.kind == "head" or len(b.lines) > 4:
            continue
        for px, py, pw, ph in photos:
            below = (py + ph - H) <= b.y0 <= (py + ph + 2.5 * H)
            if below and _xoverlap(b.x0, b.x1, px, px + pw) >= 0.5 * b.w:
                b.role = "caption"
                break
    return masthead_y


# --------------------------------------------------------------------------- step 5
def group_headlines(blocks, H, photos=()):
    heads = [b for b in blocks if b.role == "text" and b.kind == "head"]
    # kickers: a single short capitalised text-size line sitting right on top of a headline
    for b in blocks:
        if b.role != "text" or b.kind not in ("body", "small") or len(b.lines) != 1:
            continue
        letters = [c for c in b.text if c.isalpha()]
        if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.7:
            continue
        for h in heads:
            if -0.3 * H <= h.y0 - b.y1 <= 1.5 * H and _xoverlap(b.x0, b.x1, h.x0, h.x1) >= 0.5 * b.w:
                b.kind = "deck"
                break

    disp = [b for b in blocks if b.role == "text" and b.kind in ("head", "deck")]
    parent = {b.id: b.id for b in disp}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(disp):
        for b in disp[i + 1:]:
            gap = max(b.y0 - a.y1, a.y0 - b.y1)
            ov = _xoverlap(a.x0, a.x1, b.x0, b.x1)
            if gap > 4 * H or ov < 0.3 * min(a.w, b.w):
                continue
            # kicker / headline / deck are stacked left-aligned or centred on each
            # other; a different story's headline below is offset to one side
            left_aligned = abs(a.x0 - b.x0) <= 1.5 * H
            centred = abs((a.x0 + a.x1) - (b.x0 + b.x1)) / 2 <= 2 * H
            if not (left_aligned or centred):
                continue
            parent[find(a.id)] = find(b.id)
    groups: dict = {}
    for b in disp:
        groups.setdefault(find(b.id), []).append(b)

    # A group with no headline-size type (a standfirst, an author's name, a
    # pull-quote label) belongs to the headline above it if nothing but a photo
    # or white space separates them - the usual op-ed layout.
    others = [b for b in blocks if b.role == "text" and b.kind in ("head", "deck", "body")]

    def box(m):
        return (min(b.x0 for b in m), min(b.y0 for b in m), max(b.x1 for b in m),
                max(b.y1 for b in m))

    def byline_like(m):
        """Op-ed author name: one short all-caps line right under a head-shot."""
        if len(m) != 1 or len(m[0].lines) != 1:
            return False
        b = m[0]
        letters = [c for c in b.text if c.isalpha()]
        if len(b.text.split()) > 7 or not letters or \
                sum(c.isupper() for c in letters) / len(letters) < 0.85:
            return False
        return any(_xoverlap(b.x0, b.x1, px, px + pw) > 0 and -H <= b.y0 - (py + ph) <= 3 * H
                   for px, py, pw, ph in photos)

    def is_headline(m):
        return any(b.kind == "head" for b in m) and not byline_like(m)

    changed = True
    while changed:
        changed = False
        for key, members in sorted(groups.items(), key=lambda kv: box(kv[1])[1]):
            if is_headline(members):
                continue
            dx0, dy0, dx1, dy1 = box(members)
            best = None
            for k2, m2 in groups.items():
                if k2 == key or not is_headline(m2):
                    continue
                gx0, gy0, gx1, gy1 = box(m2)
                if gy1 > dy0 + 0.5 * H or dy0 - gy1 > 12 * H:
                    continue
                side_gap = max(gx0 - dx1, dx0 - gx1, 0)
                if _xoverlap(dx0, dx1, gx0, gx1) < 0.3 * min(dx1 - dx0, gx1 - gx0) and \
                        side_gap > 10 * H:
                    continue
                ids = {b.id for b in members} | {b.id for b in m2}
                ux0, ux1 = min(dx0, gx0), max(dx1, gx1)
                if any(o.id not in ids and o.y0 >= gy1 - 2 and o.y1 <= dy0 + 2 and
                       _xoverlap(ux0, ux1, o.x0, o.x1) > 0 for o in others):
                    continue
                if best is None or gy1 > best[0]:
                    best = (gy1, k2)
            if best:
                groups[best[1]].extend(members)
                del groups[key]
                changed = True
                break
    out = []
    for gi, members in enumerate(sorted(groups.values(), key=lambda m: min(b.y0 for b in m))):
        for b in members:
            b.group = gi
        out.append(members)
    return out


def assign_articles(blocks, groups, H, hrules, photos=()):
    """Give every body block an article id; returns number of articles."""
    # rules that end a photo/caption unit don't separate a headline from its text
    caps = [b for b in blocks if b.role == "caption"]
    ignore = []
    for px, py, pw, ph in photos:
        bottom = py + ph
        for c in caps:
            if _xoverlap(c.x0, c.x1, px, px + pw) > 0 and bottom - H <= c.y0 <= bottom + 2.5 * H:
                bottom = max(bottom, c.y1)
        ignore.append((py, bottom + 1.5 * H))
    def anchors(b):
        """For each headline group: the x-span and bottom of its blocks lying
        above body block b (a group may also have members beside b, e.g. a
        chart title, which must not count)."""
        out = []
        for gi, members in enumerate(groups):
            above = [m for m in members if m.y1 <= b.y0 + 0.5 * H]
            if above:
                out.append((gi, min(m.x0 for m in above), max(m.x1 for m in above),
                            max(m.y1 for m in above)))
        return out
    body = sorted([b for b in blocks if b.role == "text" and b.kind in ("body", "deck")
                   and b.group is None], key=lambda b: (b.y0, b.x0))
    for b in body:
        best = None   # (bottom_y, ("group", gi) | ("block", block))
        for gi, x0, x1, y1 in anchors(b):
            if _xoverlap(b.x0, b.x1, x0, x1) < 0.5 * b.w:
                continue
            if _rule_between(hrules, b.x0, b.x1, y1, b.y0, ignore=ignore):
                continue
            if best is None or y1 > best[0]:
                best = (y1, ("group", gi))
        for c in body:
            if c is b or c.article is None or c.y1 > b.y0 + 0.5 * H:
                continue
            if _xoverlap(b.x0, b.x1, c.x0, c.x1) < 0.5 * min(b.w, c.w):
                continue
            if b.y0 - c.y1 > 3 * H:
                continue
            if _rule_between(hrules, max(b.x0, c.x0), min(b.x1, c.x1), c.y1, b.y0,
                             ignore=ignore):
                continue
            if best is None or c.y1 > best[0]:
                best = (c.y1, ("block", c))
        if best is None:
            continue
        kind, ref = best[1]
        b.article = ref if kind == "group" else ref.article
    # second pass for body text that found nothing: accept a headline above it
    # even across a rule (op-eds put a rule under the author byline), as long as
    # no other text sits in between.
    text_blocks = [b for b in blocks if b.role == "text" and b.kind in ("head", "deck", "body")]
    for b in body:
        if b.article is not None:
            continue
        best = None
        for gi, x0, x1, y1 in anchors(b):
            if _xoverlap(b.x0, b.x1, x0, x1) < 0.5 * b.w:
                continue
            blocked = any(o is not b and o.group != gi and o.article != gi and o.y0 >= y1 - 2 and o.y1 <= b.y0 + 2
                          and _xoverlap(b.x0, b.x1, o.x0, o.x1) >= 0.3 * b.w
                          for o in text_blocks)
            if not blocked and (best is None or y1 > best[0]):
                best = (y1, gi)
        if best is None:
            for c in body:     # or continue the column of an article adopted above
                if c is b or c.article is None or c.y1 > b.y0 + 0.5 * H:
                    continue
                if _xoverlap(b.x0, b.x1, c.x0, c.x1) < 0.5 * min(b.w, c.w) or b.y0 - c.y1 > 3 * H:
                    continue
                if _rule_between(hrules, max(b.x0, c.x0), min(b.x1, c.x1), c.y1, b.y0,
                                 ignore=ignore):
                    continue
                if best is None or c.y1 > best[0]:
                    best = (c.y1, c.article)
        if best is not None:
            b.article = best[1]
    for members in groups:
        for m in members:
            m.article = m.group
    return len(groups)


def reading_order(blocks: list[Block]) -> list[Block]:
    cols: list[list[Block]] = []
    for b in sorted(blocks, key=lambda b: b.x0):
        for col in cols:
            cx0 = min(c.x0 for c in col)
            cx1 = max(c.x1 for c in col)
            if _xoverlap(b.x0, b.x1, cx0, cx1) >= 0.5 * min(b.w, cx1 - cx0):
                col.append(b)
                break
        else:
            cols.append([b])
    cols.sort(key=lambda col: min(c.x0 for c in col))
    return [b for col in cols for b in sorted(col, key=lambda c: c.y0)]


# --------------------------------------------------------------------------- driver
def detect_page_type(blocks, page_h, filename_type=None) -> str:
    top = " ".join(b.text for b in blocks if b.y1 < 0.15 * page_h).upper()
    top = " ".join(top.split())
    if "JOURNALISM" in top or "COURAGE" in top:     # front-page masthead slogan
        return "front"
    if "EDITORIAL" in top:                            # 'THE EDITORIAL PAGE' banner
        return "editorial"
    page = " ".join(" ".join(b.text for b in blocks).upper().split())
    if "RAMNATH GOENKA" in page or "FOUNDED BY" in page:   # editorial-page flag
        return "editorial"
    if filename_type:
        return filename_type
    return "other"


def segment(pre, filename_type=None) -> dict:
    page_h, page_w = pre.binary.shape
    lines = ocr_lines(pre.binary, pre.vrules)
    if not lines:
        return {"H": None, "articles": [], "blocks": [], "page_type": filename_type or "other",
                "masthead_y": 0}
    H = body_height(lines)
    blocks = chain_blocks(lines, H, pre.hrules)
    masthead_y = mark_furniture(blocks, H, page_w, page_h, pre.hrules, pre.photos)
    page_type = detect_page_type(blocks, page_h, filename_type)
    groups = group_headlines(blocks, H, pre.photos)
    assign_articles(blocks, groups, H, pre.hrules, pre.photos)

    articles = []
    for gi, members in enumerate(groups):
        head_blocks = sorted(members, key=lambda b: b.y0)
        headline = [b for b in head_blocks if b.kind == "head"] or head_blocks[:1]
        deck = [b for b in head_blocks if b not in headline]
        body = [b for b in blocks if b.article == gi and b.group is None]
        if not body:
            continue
        articles.append({
            "headline_boxes": [b.bbox() for b in headline],
            "deck_boxes": [b.bbox() for b in deck],
            "body_boxes": [b.bbox() for b in reading_order(body)],
            "layout_words": sum(b.n_words for b in body),
        })
    # body text nobody claimed: becomes "untitled" articles, one per column chain
    orphans = [b for b in blocks if b.role == "text" and b.kind in ("body", "deck")
               and b.article is None]
    orphan_words = sum(b.n_words for b in orphans)
    for b in orphans:
        articles.append({"headline_boxes": [], "deck_boxes": [],
                         "body_boxes": [b.bbox()], "layout_words": b.n_words,
                         "orphan": True})
    # order articles top-left first
    def key(a):
        boxes = a["headline_boxes"] or a["body_boxes"]
        return (min(b[1] for b in boxes) // (3 * H), min(b[0] for b in boxes))
    articles.sort(key=key)
    body_words_total = sum(b.n_words for b in blocks
                           if b.role == "text" and b.kind in ("body", "deck"))
    return {
        "H": H,
        "page_type": page_type,
        "masthead_y": masthead_y,
        "n_lines": len(lines),
        "mean_line_conf": float(np.mean([l.conf for l in lines])),
        "orphan_words": orphan_words,
        "body_words_total": body_words_total,
        "articles": articles,
        "blocks": [{"bbox": b.bbox(), "kind": b.kind, "role": b.role,
                    "article": b.article, "group": b.group} for b in blocks],
    }
