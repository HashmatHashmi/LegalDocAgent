"""Dependency-free fallbacks (no ML models, no network): extractive summary, keyword QA."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from .preprocessing import split_sentences

_STOP = set(
    "the a an and or of to in on for by with as at from that this these those is are was were be been being it its "
    "which who whom shall will may any all such not no nor but if then than so into upon under over between "
    "hereby herein thereof thereto hereof hereto said each other their his her they them party parties agreement".split()
)
_TOKEN = re.compile(r"[a-z][a-z'-]{2,}")
_MODAL = re.compile(r"\b(shall|must|agrees?|obligat\w+|liable|entitled|may\s+not|prohibited|responsible|terminat\w+|payable|pay)\b", re.I)
_FACT = re.compile(r"(?:\$|USD|PKR|Rs\.?|€|£)\s?\d|\b\d{1,3}(?:,\d{3})+\b|\b\d+\s*(?:days?|months?|years?|%)|\b(?:19|20)\d{2}\b", re.I)


def tokens(text: str) -> List[str]:
    return [w for w in _TOKEN.findall((text or "").lower()) if w not in _STOP]


def _score_sentences(sents: List[str]) -> List[float]:
    freq = Counter(w for s in sents for w in tokens(s))
    n = len(sents)
    scores = []
    for i, s in enumerate(sents):
        toks = tokens(s)
        base = sum(freq[w] for w in toks) / (len(toks) ** 0.7 + 1) if toks else 0.0
        boost = 1.0
        if _MODAL.search(s):
            boost *= 1.25
        if _FACT.search(s):
            boost *= 1.2
        if i < max(2, int(n * 0.08)):
            boost *= 1.25
        scores.append(base * boost)
    return scores


def _strip_headings(text: str) -> str:
    """Remove section headings ("RENT", "3. Termination", "Security Deposit") so they don't glue onto sentences."""
    from .preprocessing import _is_title_heading

    lines = (text or "").split("\n")
    keep = []
    prev_ok = True
    for i, ln in enumerate(lines):
        t = ln.strip().rstrip(":")
        if not t:
            prev_ok = True
            continue
        nxt = next((l.strip() for l in lines[i + 1:i + 4] if l.strip()), "")
        if len(t) <= 80 and t.isupper():
            prev_ok = True
            continue
        if re.fullmatch(r"\d{1,2}[.)]?\s+[A-Za-z /&-]{2,40}", t):  # "3. Termination"
            prev_ok = True
            continue
        if prev_ok and nxt and _is_title_heading(t):  # "Security Deposit"
            continue
        prev_ok = ln.strip()[-1:] in '.;:!?)"\u201d'
        keep.append(ln)
    return "\n".join(keep)


strip_headings = _strip_headings


def summarize(text: str, max_words: int = 180) -> Dict:
    text = _strip_headings(text) or text
    sents = [s for s in split_sentences(text) if 6 <= len(s.split()) <= 70]
    if not sents:
        flat = re.sub(r"\s+", " ", text or "").strip()
        return {"overview": " ".join(flat.split()[:max_words]), "key_points": []}
    scores = _score_sentences(sents)
    order = sorted(range(len(sents)), key=lambda i: -scores[i])
    chosen, words = [], 0
    for i in order:
        w = len(sents[i].split())
        if chosen and words + w > max_words:
            continue
        chosen.append(i)
        words += w
        if words >= max_words * 0.9:
            break
    chosen.sort()
    overview = " ".join(sents[i] for i in chosen)
    rest = [i for i in order if i not in chosen]
    key = []
    for i in rest:
        if _MODAL.search(sents[i]) or _FACT.search(sents[i]):
            key.append(sents[i][:260])
        if len(key) >= 6:
            break
    return {"overview": overview, "key_points": key}


def answer_from_passages(question: str, passages: List[Dict]) -> str:
    if not passages:
        return "I could not find anything relevant to that question in the document."
    qset = set(tokens(question))
    best = []
    for p in passages[:3]:
        sents = split_sentences(p["text"]) or [p["text"]]
        ranked = sorted(sents, key=lambda s: -len(qset & set(tokens(s))))
        best.append(f"[{p['id']}] {ranked[0]}")
    return "Most relevant passages from the document:\n\n" + "\n\n".join(best)
