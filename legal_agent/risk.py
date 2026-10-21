"""Hybrid risk detector: rules + structural checks + semantic similarity + (optional) LLM review."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends import Backend
from .config import Settings
from .tfidf import TfidfModel, cosine
from .risk_rules import MISSING_CHECKS, PROTOTYPES, RULES, tags_for

SEV = {"Low": 1, "Medium": 2, "High": 3}
SEV_NAME = {1: "Low", 2: "Medium", 3: "High"}
SRC_PRI = {"llm": 0, "rule": 1, "structure": 1, "semantic": 2}
MUTUAL = re.compile(
    r"\b(?:each\s+party|either\s+party|both\s+parties|mutual(?:ly)?|reciprocal(?:ly)?|the\s+parties\s+(?:shall|agree|may))\b", re.I
)


def _sentence_around(text: str, s: int, e: int) -> str:
    left = max(text.rfind(". ", 0, s), text.rfind("; ", 0, s), text.rfind("\n", 0, s))
    left = 0 if left < 0 else left + 2
    cands = [i for i in (text.find(". ", e), text.find("; ", e), text.find("\n", e)) if i != -1]
    right = min(cands) + 1 if cands else len(text)
    out = text[left:right].strip()
    return out[:420]


def _finding(clause: Dict, category: str, severity: str, source: str, explanation: str, suggestion: str,
             excerpt: str = "", rule_id: str = "", score: float = 1.0) -> Dict:
    return {
        "id": 0,
        "clause_id": clause["id"],
        "heading": clause.get("heading", ""),
        "excerpt": excerpt or clause["text"][:420],
        "category": category,
        "rule_id": rule_id,
        "rule_ids": [rule_id] if rule_id else [],
        "also": [],
        "severity": severity,
        "source": source,
        "explanation": explanation,
        "suggestion": suggestion,
        "score": round(float(score), 3),
    }


# ----------------------------------------------------------------------------- rules
def rule_findings(clauses: List[Dict], tags: set) -> List[Dict]:
    out: List[Dict] = []
    for c in clauses:
        hits = []
        for r in RULES:
            if "any" not in r.applies and not (r.applies & tags):
                continue
            m = r.pattern.search(c["text"])
            if not m:
                continue
            sent = _sentence_around(c["text"], m.start(), m.end())
            sev = SEV[r.severity]
            note = ""
            if r.escalate and r.escalate.search(sent):
                sev = min(3, sev + 1)
            if MUTUAL.search(sent) and sev > 1:
                sev -= 1
                note = " (The wording looks mutual, so the severity was reduced.)"
            hits.append((r, sev, sent, note))
        ids = {h[0].id for h in hits}
        for r, sev, sent, note in hits:
            if any(s in ids for s in r.superseded_by):
                continue
            out.append(_finding(c, r.category, SEV_NAME[sev], "rule", r.explanation + note, r.suggestion,
                                excerpt=sent, rule_id=r.id, score=1.0))
    return out


# ------------------------------------------------------------------ structural asymmetry
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10, "fifteen": 15,
         "thirty": 30, "sixty": 60, "ninety": 90}
_PERIOD = re.compile(r"\b(\d{1,3}|one|two|three|four|five|six|seven|ten|fifteen|thirty|sixty|ninety)\s*(?:\(\d{1,3}\)\s*)?(day|week|month|year)s?\b", re.I)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_STRONG = ("landlord", "lessor", "company", "employer", "licensor", "lender", "franchisor", "owner")
_WEAK = ("tenant", "lessee", "employee", "licensee", "borrower", "franchisee", "contractor", "consultant")
_ROLE_RE = re.compile(r"\b(" + "|".join(_STRONG + _WEAK) + r")\b", re.I)


def asymmetry_findings(clauses: List[Dict]) -> List[Dict]:
    strong: List[Tuple[int, Dict, str]] = []
    weak: List[Tuple[int, Dict, str]] = []
    for c in clauses:
        for sent in re.split(r"(?<=[.;])\s+", c["text"]):
            if not re.search(r"\bnotice\b", sent, re.I) or not re.search(r"terminat|vacat|resign|cancel|quit|surrender|\bend\b", sent, re.I):
                continue
            role = _ROLE_RE.search(sent)
            per = _PERIOD.search(sent)
            if not role or not per:
                continue
            n = per.group(1).lower()
            n = int(n) if n.isdigit() else _NUMW.get(n, 0)
            days = n * _UNIT_DAYS[per.group(2).lower()]
            if days <= 0:
                continue
            (strong if role.group(1).lower() in _STRONG else weak).append((days, c, sent))
    out: List[Dict] = []
    if strong and weak:
        s_min = min(strong, key=lambda x: x[0])
        w_max = max(weak, key=lambda x: x[0])
        s_max = max(strong, key=lambda x: x[0])
        w_min = min(weak, key=lambda x: x[0])
        if w_max[0] >= 3 * s_min[0]:
            out.append(_finding(w_max[1], "Unequal notice periods", "Medium", "structure",
                                f"The weaker party must give about {w_max[0]} days' notice while the stronger party needs only about {s_min[0]} days.",
                                "Use the same notice period for both parties.", excerpt=w_max[2], rule_id="notice_asymmetry"))
        elif s_max[0] >= 3 * w_min[0]:
            out.append(_finding(s_max[1], "Unequal notice periods", "Low", "structure",
                                f"Notice periods differ materially (about {s_max[0]} days vs {w_min[0]} days).",
                                "Check that the difference is intended and use the same notice period for both parties.",
                                excerpt=s_max[2], rule_id="notice_asymmetry"))
    return out


# ------------------------------------------------------------------ semantic detector
def semantic_findings(clauses: List[Dict], backend: Backend, already: set) -> Tuple[List[Dict], str]:
    cand = [c for c in clauses if c["id"] not in already and len(c["text"].split()) >= 8 and not MUTUAL.search(c["text"])]
    if not cand:
        return [], "none"
    ptexts = [p[2] for p in PROTOTYPES]
    ctexts = [c["text"] for c in cand]
    emb = backend.embed(ptexts + ctexts)
    if emb is not None and len(emb) == len(ptexts) + len(ctexts):
        sims = emb[len(ptexts):] @ emb[: len(ptexts)].T
        thr = backend.embed_threshold or 0.5
        method = "embeddings"
    else:
        mats = TfidfModel().fit_transform(ptexts + ctexts)
        protos, cl = mats[: len(ptexts)], mats[len(ptexts):]
        sims = np.array([[cosine(c, p) for p in protos] for c in cl])
        thr = 0.40
        method = "tfidf"
    out: List[Dict] = []
    for i, c in enumerate(cand):
        j = int(np.argmax(sims[i]))
        score = float(sims[i][j])
        if score >= thr:
            cat, sev = PROTOTYPES[j][0], PROTOTYPES[j][1]
            if sev == "High" and backend.name == "local":
                sev = "Medium"  # local-embedding similarity is only a hint, never a "High" verdict
            out.append(_finding(c, cat, sev, "semantic",
                                f"This clause is semantically similar to a known risky pattern ({cat.lower()}); similarity {score:.2f}. Review it manually.",
                                _suggestion_for(cat), score=score))
    return out, method


def _suggestion_for(category: str) -> str:
    for r in RULES:
        if r.category == category:
            return r.suggestion
    return "Review this clause and negotiate more balanced wording."


# ------------------------------------------------------------------ missing clauses
def missing_clauses(text: str, tags: set) -> List[Dict]:
    flat = re.sub(r"\s+", " ", text or "")
    out = []
    if "contract" not in tags or len(flat.split()) < 60:
        return out
    for name, pat, sev, applies, expl, sugg in MISSING_CHECKS:
        if not (applies & tags):
            continue
        if not re.search(pat, flat, re.I):
            out.append({"name": name, "severity": sev, "explanation": expl, "suggestion": sugg})
    return out


# ------------------------------------------------------------------ merge + score
def _norm(s: str) -> str:
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def merge_findings(findings: List[Dict]) -> List[Dict]:
    """One card per issue: if the LLM already flagged a clause, rule hits on that clause are folded into
    the LLM finding ("also matched") instead of being listed again under a different name."""
    by: Dict[int, List[Dict]] = {}
    for f in findings:
        by.setdefault(f["clause_id"], []).append(f)
    out: List[Dict] = []
    for _, fs in by.items():
        llm = sorted([f for f in fs if f["source"] == "llm"], key=lambda f: -SEV[f["severity"]])
        rest = sorted([f for f in fs if f["source"] != "llm"], key=lambda f: (-SEV[f["severity"]], SRC_PRI.get(f["source"], 3)))
        kept: List[Dict] = []
        seen = set()
        for f in llm:
            k = _norm(f["category"])
            if k not in seen:
                seen.add(k)
                kept.append(f)
        if kept:
            primary = kept[0]
            for f in rest:
                if f["category"] not in primary["also"]:
                    primary["also"].append(f["category"])
                primary["rule_ids"] = sorted(set(primary["rule_ids"]) | set(f["rule_ids"]))
                if SEV[f["severity"]] > SEV[primary["severity"]]:
                    primary["severity"] = f["severity"]
            if rest:
                primary["source"] = "llm + rules"
        else:
            for f in rest:
                k = _norm(f["category"])
                if k in seen:
                    continue
                seen.add(k)
                kept.append(f)
        out += kept[:6]
    out.sort(key=lambda f: (-SEV[f["severity"]], f["clause_id"]))
    for i, f in enumerate(out, 1):
        f["id"] = i
    return out


def score_risk(findings: List[Dict], missing: List[Dict]) -> Tuple[int, str]:
    w = {3: 10, 2: 4, 1: 1}
    mw = {3: 6, 2: 3, 1: 1}
    raw = sum(w[SEV[f["severity"]]] for f in findings) + sum(mw[SEV[m["severity"]]] for m in missing)
    score = int(round(100 * (1 - math.exp(-raw / 35.0))))
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    return score, level


def detect_risks(clauses: List[Dict], text: str, doc_type: str, backend: Backend, settings: Settings,
                 perspective: str = "") -> Dict:
    tags = tags_for(doc_type)
    findings = rule_findings(clauses, tags) + asymmetry_findings(clauses)
    layers = ["rules", "structure"]

    llm = backend.review_risks(clauses, perspective, doc_type) if clauses else None
    by_id = {c["id"]: c for c in clauses}
    if llm is not None:
        layers.append("llm")
        for r in llm:
            c = by_id.get(r["clause_id"])
            if c:
                findings.append(_finding(c, r["category"], r["severity"], "llm", r["explanation"], r["suggestion"], score=0.9))
    elif settings.enable_semantic and clauses:
        try:
            sem, method = semantic_findings(clauses, backend, {f["clause_id"] for f in findings})
            findings += sem
            layers.append(f"semantic ({method})")
        except Exception as e:  # noqa: BLE001
            backend.warn(f"Semantic risk layer skipped ({type(e).__name__}: {str(e)[:100]}).")

    merged = merge_findings(findings)
    missing = missing_clauses(text, tags)
    score, level = score_risk(merged, missing)
    return {"findings": merged, "missing": missing, "score": score, "level": level, "layers": layers}
