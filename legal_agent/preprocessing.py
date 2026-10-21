"""Text cleaning, clause segmentation and chunking."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List

_NUM_START = re.compile(
    r"^(?:(?:article|section|clause|schedule|annex(?:ure)?|exhibit)\s+[\w.]+"
    r"|\d{1,3}(?:\.\d{1,3})*[.)]\s*\S"
    r"|\d{1,3}(?:\.\d{1,3})+\s+\S"
    r"|\(?[a-zA-Z]{1,2}\)\s+\S"
    r"|\(?[ivxIVX]{1,5}\)\s+\S"
    r"|[•\-–*]\s+\S)",
    re.IGNORECASE,
)
_GLUED = re.compile(r"^([A-Z][A-Za-z ]{2,40}?[a-z])([A-Z][a-z]{2,}\s.{40,})$")
_SMALL = {"of", "and", "by", "for", "to", "the", "in", "on", "or", "a", "an", "at", "with"}
_PAGE_NO = re.compile(r"^(?:page\s*)?\d+\s*(?:of|/)\s*\d+$|^page\s*\d+$|^-\s*\d+\s*-$", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[\"“(\[]?[A-Z0-9])")
_ABBR = ["Mr.", "Mrs.", "Ms.", "Dr.", "No.", "Inc.", "Ltd.", "Co.", "Corp.", "vs.", "etc.", "i.e.", "e.g.",
         "Sec.", "Art.", "Rs.", "St.", "Pvt.", "approx.", "Hon'ble"]


def clean_text(raw: str) -> str:
    """Normalise whitespace/characters but keep line structure (needed for clause detection)."""
    if not raw:
        return ""
    t = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").replace("\u00ad", "")
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    t = re.sub(r"(\w)-\n([a-z])", r"\1\2", t)  # de-hyphenate words broken across lines
    lines = []
    for ln in t.split("\n"):
        s = re.sub(r"[ \t]+", " ", ln).strip()
        if _PAGE_NO.match(s):
            continue
        g = _GLUED.match(s)  # "Entry by LandlordThe Landlord may..." -> heading + body
        if g and len(g.group(1).split()) <= 5:
            lines.append(g.group(1).strip())
            s = g.group(2)
        lines.append(s)
    # drop running headers / footers (short lines repeated many times)
    counts = Counter(l for l in lines if 3 <= len(l) <= 80)
    repeated = {l for l, c in counts.items() if c >= 4 and not _NUM_START.match(l) and "_" not in l}
    lines = [l for l in lines if l not in repeated]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def split_sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text or "")
    for a in _ABBR:
        t = t.replace(a, a.replace(".", "\u2024"))
    parts = _SENT_SPLIT.split(t)
    return [p.replace("\u2024", ".").strip() for p in parts if p.strip()]


_NUM_HEADING = re.compile(r"^(?:article|section|clause)?\s*\d{1,2}(?:\.\d{1,2})*[.)]?\s+[A-Z][A-Za-z ,/&'\-]{2,55}$")


def _is_heading(line: str) -> bool:
    s = line.strip().rstrip(":")
    if 3 <= len(s) <= 80 and s.isupper() and any(c.isalpha() for c in s):
        return True
    # short numbered title such as "3. Liability and Indemnity" (no sentence-ending period)
    return bool(_NUM_HEADING.match(s)) and not s.endswith(".") and len(s.split()) <= 8


def _is_title_heading(s: str) -> bool:
    """Short Title-Case line without sentence punctuation, e.g. 'Security Deposit', 'Entry by Landlord'."""
    s = s.strip()
    if not (3 <= len(s) <= 60) or s[-1] in ".,;:!?)" or _NUM_START.match(s):
        return False
    words = s.split()
    if len(words) > 6 or not s[0].isupper():
        return False
    return all(w[0].isupper() or w.lower() in _SMALL or not w[0].isalpha() for w in words)


def segment_clauses(text: str, max_chars: int = 700) -> List[Dict]:
    """Split a cleaned document into clause-sized units: {id, heading, text}."""
    paras: List[Dict] = []
    cur: List[str] = []
    heading = ""
    cur_heading = ""

    def flush():
        nonlocal cur, cur_heading
        if cur:
            body = " ".join(cur).strip()
            if len(body) >= 15:
                paras.append({"heading": cur_heading, "text": body})
        cur = []

    raw_lines = text.split("\n")
    prev_ok = True  # previous line ended a sentence / heading / blank
    for i, ln in enumerate(raw_lines):
        s = ln.strip()
        if not s:
            flush()
            prev_ok = True
            continue
        nxt = next((l.strip() for l in raw_lines[i + 1:i + 4] if l.strip()), "")
        if _is_heading(s) or (prev_ok and nxt and _is_title_heading(s)):
            flush()
            heading = s.rstrip(":")
            prev_ok = True
            continue
        prev_ok = s[-1] in '.;:!?)"”'
        if _NUM_START.match(s):
            flush()
            prev_ok = prev_ok
        if not cur:
            cur_heading = heading
        cur.append(s)
    flush()

    clauses: List[Dict] = []
    for p in paras:
        if len(p["text"]) <= max_chars * 1.4:
            pieces = [p["text"]]
        else:
            pieces, buf = [], ""
            for sent in split_sentences(p["text"]):
                if buf and len(buf) + len(sent) + 1 > max_chars:
                    pieces.append(buf)
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                pieces.append(buf)
        for piece in pieces:
            clauses.append({"id": len(clauses) + 1, "heading": p["heading"], "text": piece})
    return clauses


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Greedy chunking on sentence boundaries."""
    chunks, buf = [], ""
    for sent in split_sentences(text):
        while len(sent) > max_chars:  # pathological very long "sentence"
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:]
        if buf and len(buf) + len(sent) + 1 > max_chars:
            chunks.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        chunks.append(buf)
    return chunks
