"""LangGraph orchestration (with an automatic sequential fallback if LangGraph is unavailable)."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

from .backends import Backend, make_backend
from .classifier import classify_document
from .config import Settings
from .drafting import draft_revisions
from .entities import extract_entities
from .extraction import ExtractionError, extract_text
from .preprocessing import clean_text, segment_clauses
from .report import build_markdown
from .risk import detect_risks
from .taxonomy import OTHER


class AgentState(TypedDict, total=False):
    filename: str
    file_bytes: bytes
    raw_text: str
    page_count: int
    clean_text: str
    clauses: List[dict]
    entities: dict
    classification: dict
    summary: dict
    risks: List[dict]
    missing_clauses: List[dict]
    risk_score: int
    risk_level: str
    risk_layers: List[str]
    revisions: List[dict]
    report_markdown: str
    warnings: List[str]
    errors: List[str]
    timings: dict


class Pipeline:
    STEP_LABELS = [
        ("extract", "Extracting text"),
        ("clean", "Cleaning and splitting into clauses"),
        ("entities", "Finding parties, dates and amounts"),
        ("classify", "Classifying document type"),
        ("summarize", "Summarising"),
        ("risk", "Detecting risky clauses"),
        ("draft", "Drafting safer wording"),
        ("report", "Building report"),
    ]

    def __init__(self, settings: Optional[Settings] = None, backend: Optional[Backend] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None, use_langgraph: bool = True):
        self.settings = settings or Settings.from_env()
        self.backend = backend or make_backend(self.settings)
        self.progress = progress
        self.fns = {
            "extract": self._extract, "clean": self._clean, "entities": self._entities, "classify": self._classify,
            "summarize": self._summarize, "risk": self._risk, "draft": self._draft, "report": self._report,
        }
        self.graph = self._compile() if use_langgraph else None
        self.engine = "langgraph" if self.graph is not None else "sequential"

    # ------------------------------------------------------------------ nodes
    def _extract(self, st: AgentState) -> dict:
        if st.get("raw_text"):
            return {"page_count": st.get("page_count", 0)}
        try:
            text, warns, pages = extract_text(st.get("file_bytes", b""), st.get("filename", "document.txt"), self.settings.enable_ocr)
        except ExtractionError as e:
            return {"raw_text": "", "errors": list(st.get("errors", [])) + [str(e)]}
        upd: dict = {"raw_text": text, "page_count": pages, "warnings": list(st.get("warnings", [])) + warns}
        if len(text.strip()) < 20:
            upd["errors"] = list(st.get("errors", [])) + ["No readable text was found in this document."]
        return upd

    def _clean(self, st: AgentState) -> dict:
        txt = clean_text(st.get("raw_text", ""))
        return {"clean_text": txt, "clauses": segment_clauses(txt)}

    def _entities(self, st: AgentState) -> dict:
        return {"entities": extract_entities(st.get("clean_text", ""))}

    def _classify(self, st: AgentState) -> dict:
        return {"classification": classify_document(st.get("clean_text", ""), self.backend)}

    def _summarize(self, st: AgentState) -> dict:
        return {"summary": self.backend.summarize(st.get("clean_text", ""), self.settings.summary_words)}

    def _risk(self, st: AgentState) -> dict:
        label = st.get("classification", {}).get("label", OTHER)
        res = detect_risks(st.get("clauses", []), st.get("clean_text", ""), label, self.backend, self.settings, self.settings.perspective)
        return {"risks": res["findings"], "missing_clauses": res["missing"], "risk_score": res["score"],
                "risk_level": res["level"], "risk_layers": res["layers"]}

    def _draft(self, st: AgentState) -> dict:
        if not self.settings.enable_rewrites:
            return {"revisions": []}
        revs = draft_revisions(st.get("risks", []), st.get("clauses", []), self.backend, self.settings,
                               st.get("entities", {}), self.settings.perspective)
        return {"revisions": revs}

    def _report(self, st: AgentState) -> dict:
        return {"report_markdown": build_markdown(dict(st))}

    # ------------------------------------------------------------------ wiring
    def _wrap(self, idx: int, name: str, label: str):
        fn = self.fns[name]

        def node(st: AgentState) -> dict:
            if self.progress:
                try:
                    self.progress(label, idx, len(self.STEP_LABELS))
                except Exception:  # noqa: BLE001
                    pass
            t0 = time.time()
            try:
                upd = fn(st) or {}
            except Exception as e:  # noqa: BLE001 - one failing step must not kill the run
                upd = {"errors": list(st.get("errors", [])) + [f"'{label}' failed: {type(e).__name__}: {str(e)[:160]}"]}
            timings = dict(st.get("timings", {}))
            timings[name] = round(time.time() - t0, 2)
            upd["timings"] = timings
            return upd

        return node

    def _compile(self):
        try:
            from langgraph.graph import END, START, StateGraph

            g = StateGraph(AgentState)
            names = [n for n, _ in self.STEP_LABELS]
            for i, (n, label) in enumerate(self.STEP_LABELS):
                g.add_node(f"step_{n}", self._wrap(i, n, label))
            g.add_edge(START, "step_extract")
            g.add_conditional_edges(
                "step_extract",
                lambda s: "step_clean" if len((s.get("raw_text") or "").strip()) >= 20 else "stop",
                {"step_clean": "step_clean", "stop": END},
            )
            for a, b in zip(names[1:-1], names[2:]):
                g.add_edge(f"step_{a}", f"step_{b}")
            g.add_edge(f"step_{names[-1]}", END)
            return g.compile()
        except Exception:  # noqa: BLE001
            return None

    def _run_sequential(self, state: AgentState) -> AgentState:
        for i, (n, label) in enumerate(self.STEP_LABELS):
            state.update(self._wrap(i, n, label)(state))
            if n == "extract" and len((state.get("raw_text") or "").strip()) < 20:
                break
        return state

    # ------------------------------------------------------------------ public
    def _finish(self, out: dict) -> Dict:
        out.pop("file_bytes", None)
        out.setdefault("warnings", [])
        out.setdefault("errors", [])
        for w in self.backend.warnings:
            if w not in out["warnings"]:
                out["warnings"].append(w)
        out["backend"] = self.backend.usage().get("backend", self.backend.name)
        out["usage"] = self.backend.usage()
        out["workflow_engine"] = self.engine
        out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        out["ok"] = bool(out.get("clean_text")) and not any("No readable text" in e or "Unsupported" in e for e in out["errors"])
        if out.get("clean_text"):
            out["report_markdown"] = build_markdown(out)
        return out

    def run(self, data: bytes, filename: str = "document.pdf") -> Dict:
        state: AgentState = {"filename": filename, "file_bytes": data, "warnings": [], "errors": [], "timings": {}}
        return self._invoke(state)

    def run_text(self, text: str, filename: str = "pasted_text.txt") -> Dict:
        state: AgentState = {"filename": filename, "raw_text": text, "page_count": 0, "warnings": [], "errors": [], "timings": {}}
        return self._invoke(state)

    def _invoke(self, state: AgentState) -> Dict:
        try:
            out = self.graph.invoke(state) if self.graph is not None else self._run_sequential(state)
        except Exception as e:  # noqa: BLE001 - last-resort safety net
            self.backend.warn(f"LangGraph failed ({type(e).__name__}); re-ran sequentially.")
            self.engine = "sequential"
            out = self._run_sequential(state)
        return self._finish(dict(out))
