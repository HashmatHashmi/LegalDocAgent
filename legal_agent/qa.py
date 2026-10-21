"""Question answering over the analysed document (TF-IDF retrieval + optional LLM answer)."""
from __future__ import annotations

from typing import Dict, List

from .backends import Backend
from .tfidf import TfidfModel, cosine


class DocumentIndex:
    def __init__(self, clauses: List[Dict]):
        self.clauses = [c for c in clauses if c.get("text")]
        self.model = TfidfModel()
        self.vectors = self.model.fit_transform([c["text"] for c in self.clauses]) if self.clauses else []

    def search(self, question: str, k: int = 4) -> List[Dict]:
        if not self.vectors or not question.strip():
            return []
        q = self.model.transform([question])[0]
        if not q:
            return []
        sims = [cosine(v, q) for v in self.vectors]
        order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        return [{"id": self.clauses[i]["id"], "text": self.clauses[i]["text"], "score": float(sims[i])} for i in order if sims[i] > 0]


def ask(index: DocumentIndex, question: str, backend: Backend) -> Dict:
    passages = index.search(question)
    answer = backend.answer(question, passages)
    return {"answer": answer, "sources": passages}
