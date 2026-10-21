"""Rule-based extraction of parties, dates, money, durations and governing law."""
from __future__ import annotations

import re
from typing import Dict, List

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_PATTERNS = [
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:{_MONTHS}),?\s+\d{{4}}\b", re.I),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
_MONEY = re.compile(
    r"(?<![A-Za-z])(?:USD|US\$|\$|PKR|Rs\.?|INR|EUR|€|£|AED|GBP)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|lakh|lac|crore|k))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|PKR|rupees|dollars|euros)\b",
    re.I,
)
_NUMW = r"one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty-five|sixty|ninety"
_DURATION = re.compile(rf"\b(?:\d{{1,3}}|{_NUMW})\s*(?:\(\d{{1,3}}\)\s*)?(?:business\s+)?(?:days?|weeks?|months?|years?)\b", re.I)
_DEFINED = re.compile(
    r"([A-Z][\w&.,'’()\- ]{2,120}?)\s*\(\s*(?:hereinafter\s+)?(?:referred\s+to\s+as\s+|called\s+|the\s+)?[“\"']([^”\"']{2,30})[”\"']\s*\)"
)
_ROLE_WORDS = ("landlord|tenant|lessor|lessee|employer|employee|company|client|provider|contractor|consultant|buyer|seller|"
               "purchaser|vendor|borrower|lender|licensor|licensee|disclosing party|receiving party|principal|agent|customer|supplier")
_PLAIN_ROLE = re.compile(r"([A-Z][\w&.,'’\- ]{2,80}?)\s*\(\s*(?:the\s+)?(" + _ROLE_WORDS + r")\s*\)", re.I)
_GENERIC = {"services", "service", "agreement", "this agreement", "effective date", "parties", "party", "deliverables",
            "confidential information", "term", "fees", "work", "products", "software", "premises", "property", "affiliates"}
_BETWEEN = re.compile(r"\bbetween\s+(.{3,80}?)\s+and\s+(.{3,80}?)(?:[,.(;]|\s+dated|\s+on\s)", re.I)
_GOV_LAW = re.compile(
    r"governed\s+by(?:\s+and\s+construed\s+in\s+accordance\s+with)?\s+the\s+laws?\s+of\s+(?:the\s+)?([A-Z][A-Za-z ,]{2,50}?)(?=[,.;]|\s+without|\s+and\b|\s+excluding)"
)
_JURIS = re.compile(r"\bcourts?\s+(?:of|at|in)\s+([A-Z][A-Za-z ]{2,40}?)(?=[,.;]|\s+shall|\s+and\b)")


def _uniq(items: List[str], limit: int) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = re.sub(r"\s+", " ", it.strip()).lower()
        if k and k not in seen:
            seen.add(k)
            out.append(re.sub(r"\s+", " ", it.strip()))
        if len(out) >= limit:
            break
    return out


def extract_entities(text: str) -> Dict:
    flat = re.sub(r"\s+", " ", text or "")
    parties: List[Dict[str, str]] = []
    for m in _DEFINED.finditer(flat):
        name = m.group(1).strip()
        if re.search(r"\bbetween\s+", name, re.I):
            name = re.split(r"\bbetween\s+", name, flags=re.I)[-1]
        name = re.sub(r"^(?:and|by)\s+", "", name.strip(), flags=re.I).strip(" ,")
        role = m.group(2).strip()
        if len(name) < 3 or len(role.split()) > 4 or role.lower() in _GENERIC or name.lower().startswith(("schedule", "annex", "exhibit")):
            continue
        if not any(p["role"].lower() == role.lower() for p in parties):
            parties.append({"name": name[:80], "role": role})
    for m in _PLAIN_ROLE.finditer(flat):  # "Muhammad Ali (Landlord)" without quotes
        name = m.group(1).strip()
        name = re.split(r"\b(?:between|and)\s+", name, flags=re.I)[-1].strip(" ,")
        role = m.group(2).strip().title()
        if len(name) >= 3 and not any(p["role"].lower() == role.lower() for p in parties):
            parties.append({"name": name[:80], "role": role})
    if not parties:
        m = _BETWEEN.search(flat[:3000])
        if m:
            parties = [{"name": m.group(1).strip()[:80], "role": "Party A"}, {"name": m.group(2).strip()[:80], "role": "Party B"}]

    dates: List[str] = []
    for pat in _DATE_PATTERNS:
        dates += [m.group(0) for m in pat.finditer(flat)]
    gov = _GOV_LAW.search(flat)
    return {
        "parties": parties[:6],
        "dates": _uniq(dates, 8),
        "amounts": _uniq([m.group(0).rstrip(" ,.;") for m in _MONEY.finditer(flat)], 10),
        "durations": _uniq([m.group(0) for m in _DURATION.finditer(flat)], 10),
        "governing_law": gov.group(1).strip() if gov else "",
        "jurisdictions": _uniq([m.group(1) for m in _JURIS.finditer(flat)], 4),
    }
