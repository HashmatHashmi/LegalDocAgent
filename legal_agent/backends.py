"""Pluggable AI backends.

    OpenAIBackend  - best quality, cheap with gpt-4o-mini, has a hard spend cap
    LocalBackend   - free Hugging Face models (needs torch + transformers)
    OfflineBackend - no models at all; rule/statistics based

Every public method is *guarded*: if a backend fails (no credit, no internet, missing
library, model download error) the call silently falls back to the offline method and
a warning is recorded, so the pipeline never crashes.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from . import offline
from .config import Settings
from .preprocessing import chunk_text

_CATEGORY_HINT = (
    "one-sided or unlimited indemnity/liability, unilateral termination or modification, termination or "
    "action without notice, unequal notice periods, penalties/forfeiture, auto-renewal/lock-in, perpetual or "
    "irrevocable obligations, broad waivers, restrictive non-compete/exclusivity, broad IP assignment, "
    "warranty disclaimers, unilateral price/rent increases, entry/access without notice, personal guarantees, "
    "very short cure periods, unfair dispute forums, vague discretionary standards"
)


def _ns(obj):
    """dict/list -> attribute-style objects, so HTTP responses look like SDK responses."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


def _http_json(path: str, payload: dict, api_key: str, timeout: float = 60.0, base_url: Optional[str] = None) -> dict:
    """Plain-HTTPS call to the OpenAI REST API (standard library only: no compiled dependencies).
    Used only when the `openai` package cannot run (e.g. Windows Application Control blocks one of its DLLs)."""
    base = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    data = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last: Optional[Exception] = None
    for attempt in range(3):
        req = urllib.request.Request(base + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:500]
            retryable = e.code in (429, 500, 502, 503, 504) and "insufficient_quota" not in body
            last = RuntimeError(f"OpenAI HTTP {e.code}: {body}")
            if not (retryable and attempt < 2):
                raise last
        except urllib.error.URLError as e:
            last = RuntimeError(f"Network error: {e.reason}")
            if attempt >= 2:
                raise last
        time.sleep(1.5 * (attempt + 1))
    raise last or RuntimeError("OpenAI request failed")


class BudgetExceeded(Exception):
    pass


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            return json.loads(t[s : e + 1])
        raise


class Backend:
    name = "offline"
    embed_threshold: Optional[float] = None
    classify_weight = 0.0  # how much the classifier trusts this backend vs. keywords

    def __init__(self, settings: Settings):
        self.s = settings
        self.warnings: List[str] = []

    # ---------------- bookkeeping ----------------
    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def usage(self) -> Dict:
        return {"backend": self.name, "estimated_cost_usd": 0.0}

    def _try(self, hook: str, *args, **kw):
        fn = getattr(self, hook, None)
        if fn is None:
            return None
        try:
            return fn(*args, **kw)
        except BudgetExceeded as e:
            self.warn(str(e))
        except Exception as e:  # noqa: BLE001 - we really want to catch everything here
            self.warn(f"{self.name}: {hook.lstrip('_')} failed ({type(e).__name__}: {str(e)[:140]}). Used a free fallback.")
        return None

    # ---------------- public API ----------------
    def summarize(self, text: str, max_words: int) -> Dict:
        out = self._try("_summarize", text, max_words)
        if isinstance(out, dict) and out.get("overview"):
            out.setdefault("key_points", [])
            return out
        return offline.summarize(text, max_words)

    def classify_scores(self, text: str, all_labels: List[str], candidates: List[str]) -> Optional[Dict[str, float]]:
        return self._try("_classify", text, all_labels, candidates)

    def review_risks(self, clauses: List[Dict], perspective: str, doc_type: str) -> Optional[List[Dict]]:
        return self._try("_review_risks", clauses, perspective, doc_type)

    def rewrite(self, clause: str, issue: str, suggestion: str, perspective: str) -> Optional[str]:
        return self._try("_rewrite", clause, issue, suggestion, perspective)

    def answer(self, question: str, passages: List[Dict]) -> str:
        out = self._try("_answer", question, passages)
        return out if out else offline.answer_from_passages(question, passages)

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        return self._try("_embed", texts)

    # ---------------- hooks (None = unsupported) ----------------
    def _summarize(self, text, max_words):
        return None

    def _classify(self, text, all_labels, candidates):
        return None

    def _review_risks(self, clauses, perspective, doc_type):
        return None

    def _rewrite(self, clause, issue, suggestion, perspective):
        return None

    def _answer(self, question, passages):
        return None

    def _embed(self, texts):
        return None


class OfflineBackend(Backend):
    name = "offline"


# =====================================================================================
# OpenAI
# =====================================================================================
class OpenAIBackend(Backend):
    name = "openai"
    classify_weight = 0.75
    embed_threshold = 0.42

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self._use_http = False
        self.client = None
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)
        except Exception as e:  # noqa: BLE001 - package missing or blocked on this PC
            self._use_http = True
            self.warn(f"OpenAI Python package unavailable ({type(e).__name__}); using direct HTTPS calls instead.")
        self.calls = 0
        self.tok_in = 0
        self.tok_out = 0
        self.tok_embed = 0
        self._spent = 0.0
        self._no_temperature = False

    # ---- transport: SDK first, plain HTTPS if the SDK cannot run ----
    def _chat_create(self, kwargs: dict):
        if not self._use_http and self.client is not None:
            try:
                return self.client.chat.completions.create(**kwargs)
            except ImportError as e:  # e.g. Windows blocks the SDK's compiled `jiter` module
                self._use_http = True
                self.warn(f"The OpenAI package is blocked on this PC ({str(e)[:80]}); switched to direct HTTPS calls.")
        return _ns(_http_json("/chat/completions", kwargs, self.s.openai_api_key))

    def _embed_create(self, model: str, batch: List[str]):
        if not self._use_http and self.client is not None:
            try:
                return self.client.embeddings.create(model=model, input=batch)
            except ImportError as e:
                self._use_http = True
                self.warn(f"The OpenAI package is blocked on this PC ({str(e)[:80]}); switched to direct HTTPS calls.")
        return _ns(_http_json("/embeddings", {"model": model, "input": batch}, self.s.openai_api_key))

    # ---- cost control ----
    def cost(self) -> float:
        return self._spent + self.tok_embed * self.s.price_embed_per_m / 1e6

    def usage(self) -> Dict:
        label = f"openai ({self.s.openai_model}"
        if self.s.openai_rewrite_model:
            label += f"; rewrites: {self.s.openai_rewrite_model}"
        return {
            "backend": label + ")",
            "calls": self.calls,
            "prompt_tokens": self.tok_in,
            "completion_tokens": self.tok_out,
            "embedding_tokens": self.tok_embed,
            "estimated_cost_usd": round(self.cost(), 5),
        }

    def _check_budget(self) -> None:
        if self.cost() >= self.s.max_spend_usd:
            raise BudgetExceeded(
                f"Spend cap of ${self.s.max_spend_usd:.2f} reached (estimate); remaining steps used free methods."
            )

    def _chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 900, temperature: float = 0.2,
              model: Optional[str] = None) -> str:
        self._check_budget()
        model_name = model or self.s.openai_model
        kwargs = {
            "model": model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if not self._no_temperature:
            kwargs["temperature"] = temperature
        try:
            resp = self._chat_create(kwargs)
        except Exception as e:  # some models reject a custom temperature
            if "temperature" in str(e).lower() and not self._no_temperature:
                self._no_temperature = True
                kwargs.pop("temperature", None)
                resp = self._chat_create(kwargs)
            else:
                raise
        self.calls += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            t_in = getattr(u, "prompt_tokens", 0) or 0
            t_out = getattr(u, "completion_tokens", 0) or 0
            self.tok_in += t_in
            self.tok_out += t_out
            if model_name != self.s.openai_model:  # separate (usually pricier) rewrite model
                p_in = self.s.price_in_rewrite_per_m or 20 * self.s.price_in_per_m
                p_out = self.s.price_out_rewrite_per_m or 20 * self.s.price_out_per_m
            else:
                p_in, p_out = self.s.price_in_per_m, self.s.price_out_per_m
            self._spent += (t_in * p_in + t_out * p_out) / 1e6
        return resp.choices[0].message.content or ""

    # ---- tasks ----
    def _summarize(self, text, max_words):
        system = "You are a careful legal analyst. Write plain English. Never invent facts that are not in the text."
        chunks = chunk_text(text, 14000)[:14]
        if len(chunks) <= 1:
            material = text[:16000]
        else:
            parts = []
            for c in chunks:
                parts.append(
                    self._chat(
                        system,
                        "Summarise this part of a legal document in about 150 words. Keep parties, obligations, amounts, "
                        "dates, deadlines, termination and liability terms.\n\n" + c,
                        max_tokens=450,
                    )
                )
            material = "\n\n".join(parts)
        user = (
            f'Return JSON with keys "overview" (plain-English summary, about {max_words} words) and "key_points" '
            "(list of 5-8 short strings: parties, purpose, money, duration, main obligations, termination, notable risks).\n\n"
            "Document:\n" + material
        )
        data = parse_json(self._chat(system, user, json_mode=True, max_tokens=1000))
        overview = str(data.get("overview", "")).strip()
        if not overview:
            return None
        return {"overview": overview, "key_points": [str(x).strip() for x in data.get("key_points", [])][:10]}

    def _classify(self, text, all_labels, candidates):
        user = (
            "Classify this legal document into exactly ONE of these types:\n- " + "\n- ".join(all_labels)
            + '\n\nReturn JSON: {"label": "<type copied exactly>", "confidence": <0..1>, "runner_up": "<type or null>"}\n\n'
            "Document excerpt:\n" + text[:6000]
        )
        data = parse_json(self._chat("You classify legal documents.", user, json_mode=True, max_tokens=120, temperature=0))
        label = str(data.get("label", "")).strip()
        if label not in all_labels:
            return None
        conf = max(0.05, min(0.98, float(data.get("confidence", 0.7))))
        scores = {l: 0.0 for l in all_labels}
        scores[label] = conf
        ru = data.get("runner_up")
        rest = 1.0 - conf
        if isinstance(ru, str) and ru in scores and ru != label:
            scores[ru] = rest * 0.6
            rest *= 0.4
        others = [l for l in all_labels if scores[l] == 0.0]
        for l in others:
            scores[l] = rest / max(1, len(others))
        return scores

    def _review_risks(self, clauses, perspective, doc_type):
        who = (
            f'The reader is the "{perspective}". Flag terms that disadvantage them.'
            if perspective
            else "The reader is a neutral reviewer. Flag terms that are one-sided, unusually harsh, ambiguous or missing safeguards."
        )
        system = (
            "You are a meticulous contract-risk reviewer. Only flag genuinely problematic or unbalanced terms; "
            "do not flag standard, balanced boilerplate. Be concise and never invent clause numbers."
        )
        valid = {c["id"] for c in clauses}
        findings: List[Dict] = []
        batch, size = [], 0

        def run(batch_clauses):
            listing = "\n".join(f"[{c['id']}] {c['text']}" for c in batch_clauses)
            user = (
                f"Document type: {doc_type}. {who}\nTypical issues: {_CATEGORY_HINT}.\n\n"
                'Return JSON: {"risks":[{"clause_id":<int>,"category":"<short label>","severity":"High|Medium|Low",'
                '"explanation":"<one sentence why it is risky>","suggestion":"<one sentence how to fix it>"}]}\n'
                'If nothing is risky return {"risks":[]}.\n\nClauses:\n' + listing
            )
            data = parse_json(self._chat(system, user, json_mode=True, max_tokens=1400, temperature=0))
            for r in data.get("risks", []):
                try:
                    cid = int(r["clause_id"])
                except Exception:
                    continue
                sev = str(r.get("severity", "Medium")).capitalize()
                if cid not in valid or sev not in ("High", "Medium", "Low"):
                    continue
                findings.append(
                    {
                        "clause_id": cid,
                        "category": str(r.get("category", "Risky clause"))[:60],
                        "severity": sev,
                        "explanation": str(r.get("explanation", ""))[:400],
                        "suggestion": str(r.get("suggestion", ""))[:400],
                    }
                )

        for c in clauses:
            if size + len(c["text"]) > 9000 and batch:
                run(batch)
                batch, size = [], 0
            batch.append(c)
            size += len(c["text"])
        if batch:
            run(batch)
        return findings

    def _rewrite(self, clause, issue, suggestion, perspective):
        who = f" Protect the interests of the {perspective} while staying fair to both sides." if perspective else " Make it balanced and fair to both sides."
        user = (
            f"Rewrite this contract clause to fix the issue.{who} Keep party names, defined terms, amounts and dates "
            "unless the fix requires changing them. Do NOT add new rights, duties, exceptions, notice periods or cure periods "
            "that are not needed to fix the listed problems, and do not touch topics outside this clause. Fix EVERY problem listed, not just one; if the problem is structural "
            "(for example a penalty on top of full damages, or a trigger based on one party's belief) restructure the clause "
            "instead of only changing a number or adding notice. Keep roughly the same length and legal tone. "
            "Output ONLY the rewritten clause.\n\n"
            f"Issue: {issue}\nHint: {suggestion}\n\nClause:\n{clause}"
        )
        return self._chat("You are a careful legal drafter.", user, max_tokens=500, temperature=0.2,
                          model=self.s.openai_rewrite_model or None).strip()

    def _answer(self, question, passages):
        ctx = "\n\n".join(f"[{p['id']}] {p['text']}" for p in passages)
        user = (
            "Answer the question using ONLY the passages below. Cite passage numbers like [3]. "
            "If the answer is not in the passages, say you cannot find it.\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {question}"
        )
        return self._chat("You answer questions about a legal document accurately and concisely.", user, max_tokens=400, temperature=0).strip()

    def _embed(self, texts):
        self._check_budget()
        vecs: List[List[float]] = []
        for i in range(0, len(texts), 96):
            batch = [t[:2000] or " " for t in texts[i : i + 96]]
            r = self._embed_create(self.s.openai_embed_model, batch)
            self.tok_embed += getattr(getattr(r, "usage", None), "total_tokens", 0) or 0
            vecs += [d.embedding for d in r.data]
        a = np.asarray(vecs, dtype="float32")
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


# =====================================================================================
# Local Hugging Face models (free)
# =====================================================================================
_CACHE: Dict[tuple, tuple] = {}


def _load_seq2seq(name: str):
    key = ("s2s", name)
    if key not in _CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.eval()
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def _load_nli(name: str):
    key = ("nli", name)
    if key not in _CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        ent = [i for i, l in id2label.items() if "entail" in l]
        ent_idx = ent[0] if ent else (len(id2label) - 1)  # BART-style: last index = entailment
        _CACHE[key] = (tok, model, ent_idx)
    return _CACHE[key]


def _load_encoder(name: str):
    key = ("enc", name)
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        _CACHE[key] = (tok, model)
    return _CACHE[key]


class LocalBackend(Backend):
    name = "local"
    classify_weight = 0.5
    embed_threshold = 0.62  # MiniLM scored 0.52-0.55 on harmless clauses (it ignores negation like 'refundable')

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import torch  # noqa: F401
        import transformers  # noqa: F401

    def _gen(self, model_name: str, prompt: str, max_new_tokens: int = 128, min_new_tokens: int = 0, max_input: int = 1024) -> str:
        import torch

        tok, model = _load_seq2seq(model_name)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
        kw = dict(max_new_tokens=max_new_tokens, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
        if min_new_tokens:
            kw["min_new_tokens"] = min_new_tokens
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    def _summarize(self, text, max_words):
        text = offline.strip_headings(text) or text  # local models only; the OpenAI summariser is unchanged
        chunks = chunk_text(text, 2800)
        if len(chunks) > 18:  # keep CPU time sane: sample evenly
            step = len(chunks) / 18
            chunks = [chunks[int(i * step)] for i in range(18)]
        level = chunks
        for _ in range(3):
            parts = [self._gen(self.s.local_summarizer, c, max_new_tokens=140, min_new_tokens=25) for c in level if c.strip()]
            combined = " ".join(p for p in parts if p)
            if len(combined.split()) <= max_words * 1.6 or len(parts) <= 1:
                break
            level = chunk_text(combined, 2800)
        words = combined.split()
        overview = " ".join(words[: int(max_words * 1.4)])
        if not overview:
            return None
        kp = offline.summarize(text, max_words)["key_points"]
        return {"overview": overview, "key_points": kp}

    def _classify(self, text, all_labels, candidates):
        import torch

        tok, model, ent_idx = _load_nli(self.s.local_nli)
        premise = text[:1800]
        hyps = [f"This document is a {c.split('/')[0].strip().lower()}." for c in candidates]
        scores = []
        for i in range(0, len(hyps), 4):
            enc = tok([premise] * len(hyps[i : i + 4]), hyps[i : i + 4], truncation="only_first", padding=True,
                      max_length=384, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, ent_idx]
            scores += probs.tolist()
        total = sum(scores) or 1.0
        return {c: s / total for c, s in zip(candidates, scores)}

    def _rewrite(self, clause, issue, suggestion, perspective):
        who = f" in favour of the {perspective}" if perspective else ""
        prompt = (
            f"Rewrite the following contract clause so it is fair and balanced{who}. Problem to fix: {issue}. "
            f"Keep party names unchanged.\n\nClause: {clause}\n\nRewritten clause:"
        )
        return self._gen(self.s.local_generator, prompt, max_new_tokens=220, max_input=768)

    def _answer(self, question, passages):
        ctx = " ".join(p["text"] for p in passages[:3])
        prompt = f"Answer the question using only the context.\n\nContext: {ctx}\n\nQuestion: {question}\n\nAnswer:"
        out = self._gen(self.s.local_generator, prompt, max_new_tokens=120, max_input=900)
        if len(out.split()) < 2:
            return None
        cites = ", ".join(f"[{p['id']}]" for p in passages[:3])
        return f"{out}\n\n(Sources: {cites})"

    def _embed(self, texts):
        import torch

        tok, model = _load_encoder(self.s.local_embedder)
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i : i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.vstack(out)


# =====================================================================================
def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def make_backend(settings: Settings) -> Backend:
    """Create the requested backend, degrading gracefully if it cannot be built."""
    choice = (settings.backend or "auto").lower()
    notes: List[str] = []
    if choice == "auto":
        if settings.openai_api_key:
            choice = "openai"
        elif _has("torch") and _has("transformers"):
            choice = "local"
        else:
            choice = "offline"

    backend: Optional[Backend] = None
    if choice == "openai":
        try:
            backend = OpenAIBackend(settings)
        except Exception as e:  # noqa: BLE001
            notes.append(f"OpenAI backend unavailable ({type(e).__name__}: {str(e)[:100]}). Falling back to free methods.")
            choice = "local" if (_has("torch") and _has("transformers")) else "offline"
    if backend is None and choice == "local":
        try:
            backend = LocalBackend(settings)
        except Exception as e:  # noqa: BLE001
            notes.append(f"Local models unavailable ({type(e).__name__}). Install requirements-local.txt to enable them. Using offline mode.")
    if backend is None:
        backend = OfflineBackend(settings)
    for n in notes:
        backend.warn(n)
    return backend
