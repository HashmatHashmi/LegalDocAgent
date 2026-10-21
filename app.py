"""Streamlit UI for the Legal Document Analysis Agent.   Run:  streamlit run app.py"""
from __future__ import annotations

import html
from dataclasses import replace

import streamlit as st

from legal_agent.backends import make_backend
from legal_agent.config import Settings
from legal_agent.graph import Pipeline
from legal_agent.qa import DocumentIndex, ask
from legal_agent.report import DISCLAIMER, build_json, build_markdown, build_pdf

st.set_page_config(page_title="Legal Document Analysis Agent", page_icon="⚖️", layout="wide")

BACKENDS = {
    "Auto (best available)": "auto",
    "OpenAI - best quality (cheap, needs API key)": "openai",
    "Local models - free (needs requirements-local.txt)": "local",
    "Offline rules - free, instant, no downloads": "offline",
}
ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
base = Settings.from_env()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Analysis engine", list(BACKENDS), index=0)
    api_key = st.text_input("OpenAI API key (optional)", type="password", value="",
                            help="Only kept in memory for this session. You can also put it in a .env file.")
    model = st.text_input("OpenAI model", value=base.openai_model)
    rw_model = st.text_input("Model for rewrites only (optional)", value=base.openai_rewrite_model,
                             help="A stronger model for the few rewrite calls, e.g. a bigger GPT model. Leave empty to use the model above. "
                                  "Set PRICE_IN_REWRITE_PER_M / PRICE_OUT_REWRITE_PER_M in .env for an accurate cost estimate.")
    cap = st.number_input("Spend cap per analysis (USD, estimate)", min_value=0.05, max_value=5.0, value=float(base.max_spend_usd), step=0.05)
    st.divider()
    perspective = st.text_input("Which party are you? (optional)", value=base.perspective,
                                placeholder="Tenant, Employee, Client...",
                                help="Used by the AI review and rewrites to protect your side.")
    words = st.slider("Summary length (words)", 80, 400, int(base.summary_words), 20)
    rewrites = st.checkbox("Suggest safer wording", value=base.enable_rewrites)
    semantic = st.checkbox("Semantic risk layer", value=base.enable_semantic)
    st.caption("Tip: with no key and no local models, the app still works in offline mode.")

st.title("⚖️ Legal Document Analysis Agent")
st.caption("Upload a contract or legal document to get a summary, document type, risky clauses, safer wording and a downloadable report.")

tab_file, tab_text = st.tabs(["Upload file", "Paste text"])
with tab_file:
    upload = st.file_uploader("PDF, Word (.docx) or text file", type=["pdf", "docx", "txt", "md"])
with tab_text:
    pasted = st.text_area("Paste the document text", height=200, key="pasted")

go = st.button("Analyze document", type="primary")

if go:
    if upload is None and not (pasted or "").strip():
        st.warning("Please upload a file or paste some text first.")
    else:
        settings = replace(
            base, backend=BACKENDS[choice], openai_api_key=(api_key.strip() or base.openai_api_key), openai_model=model.strip() or base.openai_model,
            openai_rewrite_model=rw_model.strip(),
            max_spend_usd=float(cap), perspective=perspective.strip(), summary_words=int(words),
            enable_rewrites=bool(rewrites), enable_semantic=bool(semantic),
        )
        with st.status("Analyzing...", expanded=True) as status:
            bar = st.progress(0.0)

            def on_progress(label: str, i: int, n: int) -> None:
                status.update(label=f"Step {i + 1}/{n}: {label}")
                bar.progress(min(1.0, i / n))

            backend = make_backend(settings)
            pipe = Pipeline(settings, backend, progress=on_progress)
            if upload is not None:
                result = pipe.run(upload.getvalue(), upload.name)
            else:
                result = pipe.run_text(pasted)
            bar.progress(1.0)
            status.update(label="Done" if result.get("ok") else "Could not analyze this document",
                          state="complete" if result.get("ok") else "error")
        st.session_state["result"] = result
        st.session_state["backend"] = backend
        try:
            st.session_state["index"] = DocumentIndex(result.get("clauses", [])) if result.get("ok") else None
        except Exception as e:  # noqa: BLE001 - Q&A is optional; never lose the analysis because of it
            st.session_state["index"] = None
            result.setdefault("warnings", []).append(f"Question answering is unavailable ({type(e).__name__}).")
        st.session_state["qa"] = []
        try:
            st.session_state["pdf"] = build_pdf(result, "compact") if result.get("ok") else None
            st.session_state["pdf_full"] = build_pdf(result, "detailed") if result.get("ok") else None
        except Exception as e:  # noqa: BLE001
            st.session_state["pdf"] = None
            st.session_state["pdf_full"] = None
            result.setdefault("warnings", []).append(f"PDF report could not be built ({type(e).__name__}).")

res = st.session_state.get("result")
if res:
    if not res.get("ok"):
        for e in res.get("errors", []):
            st.error(e)
        st.stop()
    for e in res.get("errors", []):
        st.error(e)
    for w in res.get("warnings", []):
        st.warning(w)

    cls = res["classification"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Document type", cls["label"])
    c2.metric("Type confidence", f"{cls['confidence']:.0%}")
    c3.metric("Overall risk", f"{res['risk_level']} ({res['risk_score']}/100)")
    c4.metric("Risky clauses", len(res["risks"]))
    c5.metric("Clauses read", len(res["clauses"]))
    u = res.get("usage", {})
    st.caption(f"Engine: {res['backend']} | workflow: {res.get('workflow_engine')} | risk layers: {', '.join(res.get('risk_layers', []))}"
               + (f" | estimated OpenAI cost: ${u.get('estimated_cost_usd', 0):.4f}" if "calls" in u else ""))

    d1, d2, d3, d4 = st.columns(4)
    stem = res["filename"].rsplit(".", 1)[0]
    if st.session_state.get("pdf"):
        d1.download_button("PDF - short report", st.session_state["pdf"], f"{stem}_short.pdf", "application/pdf")
    if st.session_state.get("pdf_full"):
        d2.download_button("PDF - full report", st.session_state["pdf_full"], f"{stem}_full.pdf", "application/pdf")
    d3.download_button("Markdown", build_markdown(res), f"{stem}_analysis.md", "text/markdown")
    d4.download_button("JSON", build_json(res), f"{stem}_analysis.json", "application/json")

    t_sum, t_risk, t_rev, t_facts, t_qa, t_text = st.tabs(["Summary", "Risks", "Suggested wording", "Key facts", "Ask a question", "Clauses"])

    with t_sum:
        st.write(res["summary"].get("overview", ""))
        for p in res["summary"].get("key_points", []):
            st.markdown(f"- {p}")
        with st.expander("Other possible document types"):
            for lab, p in cls["ranking"]:
                st.write(f"{lab}: {p:.0%}")

    with t_risk:
        if not res["risks"]:
            st.success("No risky clauses were detected by the active detectors.")
        for f in res["risks"]:
            with st.expander(f"{ICON[f['severity']]} {f['severity']} - {f['category']} (clause {f['clause_id']})"):
                st.markdown(f"> {f['excerpt']}")
                st.markdown(f"**Why it matters:** {f['explanation']}")
                st.markdown(f"**Suggested fix:** {f['suggestion']}")
                if f.get("also"):
                    st.caption("Also matched rules: " + ", ".join(f["also"]))
                st.caption(f"Detected by: {f['source']}" + (f" | section: {f['heading']}" if f.get("heading") else ""))
        if res["missing_clauses"]:
            st.subheader("Possibly missing clauses")
            for m in res["missing_clauses"]:
                st.markdown(f"- **{m['name']}** ({m['severity']}): {m['explanation']} _{m['suggestion']}_")

    with t_rev:
        if not res["revisions"]:
            st.info("No revisions were generated (turn on 'Suggest safer wording' or no medium/high risks were found).")
        if res["revisions"]:
            st.caption("Figures in suggested wording (days, caps, percentages) are proposals to negotiate, not legal advice.")
        for v in res["revisions"]:
            with st.expander(f"Clause {v['clause_id']} - {v['category']} ({v['method']})"):
                st.markdown("**Original**")
                st.info(v["original"])
                if v["revised"]:
                    st.markdown("**Suggested wording**")
                    st.success(v["revised"])
                else:
                    st.markdown(f"**Guidance:** {v['guidance']}")

    with t_facts:
        ent = res["entities"]
        if ent.get("parties"):
            st.markdown("**Parties**")
            for p in ent["parties"]:
                st.markdown(f"- {p['name']} - *{p['role']}*")
        for label, key in (("Dates", "dates"), ("Amounts", "amounts"), ("Durations / deadlines", "durations"), ("Jurisdictions", "jurisdictions")):
            if ent.get(key):
                st.markdown(f"**{label}:** " + ", ".join(ent[key]))
        if ent.get("governing_law"):
            st.markdown(f"**Governing law:** {ent['governing_law']}")

    with t_qa:
        idx = st.session_state.get("index")
        with st.form("qa_form", clear_on_submit=True):
            q = st.text_input("Ask something about this document", placeholder="What is the notice period? Who pays for repairs?")
            asked = st.form_submit_button("Ask")
        if asked and q.strip() and idx is not None:
            out = ask(idx, q.strip(), st.session_state["backend"])
            st.session_state.setdefault("qa", []).insert(0, (q.strip(), out))
        for question, out in st.session_state.get("qa", []):
            st.markdown(f"**Q: {question}**")
            st.write(out["answer"])
            with st.expander("Source clauses"):
                for s in out["sources"]:
                    st.markdown(f"[{s['id']}] {s['text']}")

    with t_text:
        st.caption("The document split into clauses. Risky clauses are highlighted: red = high, orange = medium, yellow = low.")
        worst, issues = {}, {}
        rank = {"High": 3, "Medium": 2, "Low": 1}
        for f in res["risks"]:
            cid = f["clause_id"]
            if rank[f["severity"]] > rank.get(worst.get(cid, ""), 0):
                worst[cid] = f["severity"]
            issues.setdefault(cid, []).append(f["category"])
        tint = {"High": "#7f1d1d55", "Medium": "#9a340055", "Low": "#71350055"}
        for c in res["clauses"]:
            head = f"<b>[{c['id']}]</b> " + (f"<i>{html.escape(c['heading'])}</i> - " if c["heading"] else "")
            text = head + html.escape(c["text"])
            sev = worst.get(c["id"])
            if sev:
                tag = "<br/><small>" + html.escape("; ".join(issues[c["id"]])) + "</small>"
                st.markdown(f"<div style='background:{tint[sev]};padding:6px 10px;border-radius:6px;margin:4px 0'>{text}{tag}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:6px 10px;margin:4px 0'>{text}</div>", unsafe_allow_html=True)

st.divider()
st.caption(DISCLAIMER)
