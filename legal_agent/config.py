"""Central settings. Everything can be overridden from a .env file or the Streamlit sidebar."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # auto | openai | local | offline
    backend: str = "auto"

    # --- OpenAI (optional, paid) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    # USD per 1M tokens. Used ONLY for the cost estimate/spend cap. Check openai.com/pricing.
    price_in_per_m: float = 0.15
    price_out_per_m: float = 0.60
    price_embed_per_m: float = 0.02
    # Hard stop: once the estimated spend of one backend instance reaches this, the
    # agent silently falls back to free methods instead of calling OpenAI again.
    max_spend_usd: float = 0.75
    # Optional stronger model used ONLY for the rewrite step (few, short calls). Empty = same model as above.
    openai_rewrite_model: str = ""
    # USD per 1M tokens for that model; 0 = unknown -> estimate at 20x the base price (conservative).
    price_in_rewrite_per_m: float = 0.0
    price_out_rewrite_per_m: float = 0.0

    # --- Local (free) Hugging Face models ---
    local_summarizer: str = "sshleifer/distilbart-cnn-12-6"  # or facebook/bart-large-cnn
    local_nli: str = "typeform/distilbert-base-uncased-mnli"  # or facebook/bart-large-mnli
    local_generator: str = "google/flan-t5-base"
    local_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"  # or nlpaueb/legal-bert-base-uncased

    # --- Behaviour ---
    summary_words: int = 180
    max_rewrites: int = 8
    perspective: str = ""  # e.g. "Tenant", "Employee", "Small business owner"
    enable_rewrites: bool = True
    enable_semantic: bool = True
    enable_ocr: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend=os.getenv("LEGAL_AGENT_BACKEND", "auto").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip(),
            price_in_per_m=_float("PRICE_IN_PER_M", 0.15),
            price_out_per_m=_float("PRICE_OUT_PER_M", 0.60),
            price_embed_per_m=_float("PRICE_EMBED_PER_M", 0.02),
            max_spend_usd=_float("MAX_SPEND_USD", 0.75),
            openai_rewrite_model=os.getenv("OPENAI_REWRITE_MODEL", "").strip(),
            price_in_rewrite_per_m=_float("PRICE_IN_REWRITE_PER_M", 0.0),
            price_out_rewrite_per_m=_float("PRICE_OUT_REWRITE_PER_M", 0.0),
            local_summarizer=os.getenv("LOCAL_SUMMARIZER", "sshleifer/distilbart-cnn-12-6"),
            local_nli=os.getenv("LOCAL_NLI", "typeform/distilbert-base-uncased-mnli"),
            local_generator=os.getenv("LOCAL_GENERATOR", "google/flan-t5-base"),
            local_embedder=os.getenv("LOCAL_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
            summary_words=_int("SUMMARY_WORDS", 180),
            max_rewrites=_int("MAX_REWRITES", 8),
            perspective=os.getenv("PERSPECTIVE", "").strip(),
            enable_rewrites=_bool("ENABLE_REWRITES", True),
            enable_semantic=_bool("ENABLE_SEMANTIC", True),
            enable_ocr=_bool("ENABLE_OCR", True),
        )
