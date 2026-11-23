"""Cleaning OCR text: hyphenation, junk, page furniture, bylines, sentence splitting."""
from __future__ import annotations

import re
from pathlib import Path

_DICT_PATHS = ["/usr/share/dict/words", "/usr/dict/words"]
_base: set | None = None
_extra: set = set()


def _base_vocab() -> set:
    global _base
    if _base is None:
        _base = set()
        for p in _DICT_PATHS:
            if Path(p).exists():
                _base = {w.strip().lower() for w in Path(p).read_text().splitlines()}
                break
    return _base


class _Vocab:
    def __contains__(self, w):
        return w in _extra or w in _base_vocab()


def vocab():
    return _Vocab()


def set_page_vocab(texts):
    """Words seen unbroken on the current page (plurals, names, jargon) count as
    known words when deciding whether to rejoin a line-end hyphen.  Reset per
    page so results don't depend on processing order."""
    global _extra
    _extra = set()
    for t in texts:
        for w in re.findall(r"(?<![-\w])[A-Za-z]+(?:-[A-Za-z]+)*(?![-\w])", t):
            _extra.add(w.lower())


_CHAR_FIXES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"', "—": " - ",
    "–": "-", "­": "", "|": " ", "•": " ",
}

_FURNITURE = [
    re.compile(r"^\s*(?:continued|contd\.?|cont'd)\b.*$", re.I),
    re.compile(r"^.*\b(?:continued|see|more|full report|details|report)\s+on\s+page\s*\d+.*$", re.I),
    re.compile(r"^\s*(?:page|p\.)\s*\d+\s*$", re.I),
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^.*epaper\.indianexpress\.com.*$", re.I),
    re.compile(r"^.*www\.indianexpress\.com.*$", re.I),
    re.compile(r"^\s*inside\s*$", re.I),
]

_MONTHS = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*"


def _dehyphenate(text: str) -> str:
    v = vocab()

    def repl(m):
        left, right = m.group(1), m.group(2)
        if not right[0].islower():
            return f"{left}-{right}"
        joined = left + right
        if joined.lower() in v:
            return joined
        if f"{left}-{right}".lower() in v:
            return f"{left}-{right}"
        if left.lower() in v and right.lower() in v and len(left) > 3 and len(right) > 3:
            return f"{left}-{right}"   # probably a real compound (e.g. long-term)
        return joined                    # most line-end hyphens split one word

    return re.sub(r"([A-Za-z]+)[-­‐]\s*\n\s*([A-Za-z]+)", repl, text)


def _junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    letters = sum(c.isalpha() for c in s)
    if letters / max(1, len(s)) < 0.5 and len(s.split()) < 5:
        return True
    tokens = s.split()
    # lines made mostly of 1-2 char fragments are OCR noise from graphics
    if len(tokens) >= 3 and sum(len(t) <= 2 for t in tokens) / len(tokens) > 0.7:
        return True
    return False


def _strip_byline(lines: list[str]) -> tuple[list[str], str]:
    """Remove leading author / dateline lines, e.g. 'RAVI DUTTA MISHRA' 'NEW DELHI, DECEMBER 31'."""
    byline = []
    while lines and len(byline) < 5:
        s = lines[0].strip()
        letters = [c for c in s if c.isalpha()]
        upper = sum(c.isupper() for c in letters) / max(1, len(letters))
        is_dateline = re.search(rf"\b{_MONTHS}\b\s*\d{{1,2}}\b", s.upper()) and len(s.split()) <= 7
        if (upper > 0.85 and len(s.split()) <= 8) or is_dateline:
            byline.append(s)
            lines.pop(0)
        else:
            break
    return lines, "; ".join(byline)


def clean_block(raw: str, strip_byline=True) -> tuple[str, str]:
    for k, v in _CHAR_FIXES.items():
        raw = raw.replace(k, v)
    raw = _dehyphenate(raw)
    lines = []
    for ln in raw.split("\n"):
        if any(p.match(ln) for p in _FURNITURE):
            continue
        if _junk_line(ln):
            continue
        lines.append(ln.strip())
    byline = ""
    if strip_byline:
        lines, byline = _strip_byline(lines)
    text = " ".join(lines)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text, byline


def clean_article(headline_raw: str, deck_raw: str, body_raw: str) -> dict:
    headline, _ = clean_block(headline_raw, strip_byline=False)
    deck, _ = clean_block(deck_raw, strip_byline=False)
    paras, bylines = [], []
    for i, block in enumerate(body_raw.split("\n\n")):
        t, b = clean_block(block, strip_byline=(i < 2))
        if t:
            paras.append(t)
        if b:
            bylines.append(b)
    text = "\n\n".join(paras)
    # a block ending mid-word with "-" followed by the next block
    text = re.sub(r"([a-z])-\n\n([a-z])", r"\1\2", text)
    return {"headline": headline, "deck": deck, "byline": "; ".join(bylines),
            "text": text, "n_words": len(text.split())}


_SPLIT = re.compile(r'(?<=[.!?])["\')]?\s+(?=["\'(]?[A-Z0-9])')
_ABBREV_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|No|vs|Co|Jr|Gen|Lt|Col|Gov|Sen|Rep|Prof)|\b[A-Z])\.$")


def sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        pieces = _SPLIT.split(para)
        merged = []
        for p in pieces:
            if merged and _ABBREV_END.search(merged[-1]):
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)
        for s in merged:
            s = s.strip()
            if len(s.split()) >= 3:
                out.append(s)
    return out


def shingles(text: str, n=5) -> set:
    w = re.findall(r"[a-z]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
