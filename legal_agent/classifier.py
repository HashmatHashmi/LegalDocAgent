"""Document-type classification: keyword evidence blended with a model (LLM or zero-shot NLI)."""
from __future__ import annotations

import math
import re
from typing import Dict, List

from .backends import Backend
from .taxonomy import ALL_LABELS, DOC_TYPES, OTHER


def keyword_probs(text: str, temperature: float = 2.2) -> Dict[str, float]:
    """Return a probability-like distribution over ALL_LABELS from keyword evidence."""
    low = re.sub(r"\s+", " ", (text or "").lower())
    head = low[:600]
    scores: Dict[str, float] = {}
    for label, terms in DOC_TYPES.items():
        s = 0.0
        for t in terms["title"]:
            if t in head:
                s += 6.0
            elif t in low:
                s += 1.5
        for t in terms["body"]:
            c = low.count(t)
            if c:
                s += min(math.log1p(c), 2.2)
        scores[label] = s
    best = max(scores.values()) if scores else 0.0
    scores[OTHER] = 2.0 if best < 3.0 else 0.0
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / temperature) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def classify_document(text: str, backend: Backend, top_k: int = 8) -> Dict:
    kw = keyword_probs(text)
    ranked_kw = sorted(kw, key=lambda k: -kw[k])
    candidates = ranked_kw[:top_k] if backend.name != "openai" else list(ALL_LABELS)
    if OTHER not in candidates:
        candidates.append(OTHER)

    model = backend.classify_scores(text, list(ALL_LABELS), candidates) if text.strip() else None
    method = "keywords"
    if model:
        w = backend.classify_weight
        kw_c = {c: kw.get(c, 0.0) for c in candidates}
        zk = sum(kw_c.values()) or 1.0
        mo_c = {c: model.get(c, 0.0) for c in candidates}
        zm = sum(mo_c.values()) or 1.0
        blended = {c: w * mo_c[c] / zm + (1 - w) * kw_c[c] / zk for c in candidates}
        method = f"keywords + {backend.name} model"
    else:
        blended = {c: kw[c] for c in candidates}
        zb = sum(blended.values()) or 1.0
        blended = {c: v / zb for c, v in blended.items()}

    ranking = sorted(blended.items(), key=lambda kv: -kv[1])
    label, conf = ranking[0]
    if not text.strip():
        label, conf = OTHER, 0.0
    return {
        "label": label,
        "confidence": round(float(conf), 3),
        "ranking": [(l, round(float(p), 3)) for l, p in ranking[:5]],
        "method": method,
    }
