"""Safer-wording suggestions: LLM rewrite -> validated -> rule-based auto-edit -> guidance."""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Sequence

from .backends import Backend
from .config import Settings
from .risk import SEV, _suggestion_for
from .risk_rules import RULES

# Deliberately conservative: only edits that stay grammatical and non-contradictory in almost any sentence.
_GENERIC_ALSO = {"Action without notice", "Vague / discretionary standards", "Time is of the essence", "Unilateral / sole discretion"}

_AUTO_EDITS = [
    (re.compile(r"\b(?:sole|absolute|unfettered)\s+(?:and\s+(?:absolute|unfettered)\s+)?discretion\b", re.I),
     "reasonable discretion, exercised in good faith"),
    (re.compile(r"\bunlimited\s+liabilit\w*", re.I),
     "liability capped at the fees paid under this Agreement in the preceding twelve (12) months"),
    (re.compile(r"\bwithin\s+(?:24|48|72|twenty[\s-]four|forty[\s-]eight|seventy[\s-]two)\s*(?:\(\d+\)\s*)?hours?\b", re.I),
     "within ten (10) business days"),
    (re.compile(r"\bwithout\s+(?:any\s+|prior\s+|written\s+|advance\s+|further\s+)*notice\s+or\s+(?:cause|reason)\b", re.I),
     "upon thirty (30) days' prior written notice"),
]
_PROMPT_LEAK = ("keep these party names", "keep party names", "problem to fix", "rewritten clause", "rewrite the following",
                "also fix these", "fix every problem")
_BAD_START = re.compile(r"^(?:sorry|as an ai|i cannot|i can't|rewritten clause:?$)", re.I)


def auto_edit(text: str) -> Optional[str]:
    out = text
    for pat, rep in _AUTO_EDITS:
        out = pat.sub(rep, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out if out != text.strip() else None


def _roles(entities: Dict) -> List[str]:
    roles = [p["role"] for p in (entities or {}).get("parties", []) if p.get("role")]
    return [r for r in roles if len(r) > 2 and not r.lower().startswith("party")]


def rewrite_problem(original: str, revised: Optional[str], roles: List[str], strict: bool = False,
                    rule_ids: Sequence[str] = (), echo: Sequence[str] = ()) -> Optional[str]:
    """Return None if the rewrite is acceptable, otherwise a short reason.
    strict=True is used for small local models, whose output is much less reliable."""
    if not revised or not revised.strip():
        return "empty output"
    r = revised.strip()
    low = r.lower()
    if _BAD_START.match(r):
        return "model refused or added commentary"
    if strict and (any(p in low for p in _PROMPT_LEAK) or any(len(e) >= 25 and e[:40].lower() in low and e[:40].lower() not in original.lower() for e in echo)):
        return "model repeated the instructions instead of rewriting"  # small local models only
    lo, lr = len(original), len(r)
    if lr < (0.8 if strict else 0.35) * lo:
        return "too short (dropped content)"
    if lr > 4.0 * lo + 200:
        return "too long (likely merged other clauses)"
    if re.sub(r"\W+", "", low) == re.sub(r"\W+", "", original.lower()):
        return "unchanged"
    present = [x for x in roles if x.lower() in original.lower()]
    if present and not any(x.lower() in low for x in present) and not re.search(r"\b(?:either|each|both|any)\s+part(?:y|ies)\b|\bthe\s+parties\b", r, re.I):
        return f"party name '{present[0]}' missing"
    words = low.split()
    if len(words) > 12 and len(set(words)) / len(words) < 0.4:
        return "repetitive text"
    if strict:
        if difflib.SequenceMatcher(None, original.lower(), low).ratio() > 0.88:
            return "barely changed"
        for rule in RULES:
            if rule.id in rule_ids and rule.pattern.search(r):
                return "the risky wording is still present"
    return None


def valid_rewrite(original: str, revised: Optional[str], roles: List[str]) -> bool:
    return rewrite_problem(original, revised, roles) is None


def draft_revisions(findings: List[Dict], clauses: List[Dict], backend: Backend, settings: Settings,
                    entities: Dict, perspective: str = "") -> List[Dict]:
    by_id = {c["id"]: c for c in clauses}
    picks, seen = [], set()
    for f in sorted(findings, key=lambda f: (-SEV[f["severity"]], -f.get("score", 0), f["clause_id"])):
        if f["severity"] == "Low" or f["clause_id"] in seen or f["clause_id"] not in by_id:
            continue
        seen.add(f["clause_id"])
        picks.append(f)
        if len(picks) >= settings.max_rewrites:
            break

    roles = _roles(entities)
    out: List[Dict] = []
    for f in picks:
        original = by_id[f["clause_id"]]["text"]
        issue = f"{f['category']}: {f['explanation']}"
        extra = [f"{c}: {_suggestion_for(c)}" for c in f.get("also", []) if c not in _GENERIC_ALSO and _suggestion_for(c).startswith(("Make", "Limit", "Tie", "Require", "Replace", "Remove", "Delete", "Give", "State"))]
        if extra:
            issue += " ALSO fix these related problems in the same clause: " + " | ".join(extra[:4])
        if roles:
            issue += f" Keep these party names exactly as written: {', '.join(roles)}."
        revised, method, note = None, "", ""
        if backend.name != "offline":
            cand = backend.rewrite(original, issue, f["suggestion"], perspective)
            strict = backend.name == "local"
            rids = f.get("rule_ids", [])
            echo = (f["category"], f["explanation"])
            problem = rewrite_problem(original, cand, roles, strict, rids, echo) if cand is not None else "no model output"
            if problem and cand is not None and not strict:  # one retry with the reason spelled out
                cand = backend.rewrite(original, issue + f" A previous attempt was rejected because: {problem}. "
                                       "Rewrite ONLY this one clause.", f["suggestion"], perspective)
                problem = rewrite_problem(original, cand, roles, strict, rids, echo) if cand is not None else "no model output"
            if problem is None and cand:
                revised, method = cand.strip(), f"{backend.name} model"
            else:
                note = problem or ""
                backend.warn(f"Clause {f['clause_id']}: model rewrite rejected ({problem}); used a rule-based edit or guidance instead.")
        if not revised:
            ae = auto_edit(original)
            if ae and valid_rewrite(original, ae, roles):
                revised, method = ae, "rule-based edit"
        out.append({
            "clause_id": f["clause_id"],
            "category": f["category"],
            "severity": f["severity"],
            "original": original,
            "revised": revised or "",
            "method": method or "guidance only",
            "guidance": f["suggestion"],
            "note": note,
        })
    return out
