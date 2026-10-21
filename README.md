# Legal Document Analysis Agent (v2)

Upload a contract (PDF / Word / text) and get: a plain-English **summary**, the **document type**, **risky clauses**
with explanations, **possibly missing clauses**, **safer wording**, key facts (parties, dates, amounts, governing law),
a **question-answering box**, and a downloadable **PDF / Markdown / JSON report**. It is orchestrated with **LangGraph**.

## Daily use & updating (Windows) - do NOT recreate the .venv each time
* Start: double-click `start.bat` (uses your existing `.venv`, installs nothing).
* Update to a newer zip: `update.bat "C:\path\to\LegalDocAgent.zip"` replaces only the code; your `.venv` and `.env` stay.
  (Manual alternative: copy the `legal_agent` folder and `app.py` from the new zip over the old ones.)
* Run `pip install -r requirements.txt` again only if a release note says new packages were added.

## Quick start (5 minutes, no paid account needed)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows      (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Or just double-click `run.bat` (Windows) / run `./run.sh` (Linux/Mac).
With no key and no extra installs the app runs in **Offline mode**: instant, free, nothing downloaded.

## Three engines - pick in the sidebar

| Engine | Cost | Quality | Needs |
|---|---|---|---|
| **OpenAI** (default model `gpt-4o-mini`) | tiny per document (see below) | **Best.** Understands paraphrased/new risks, writes real summaries and rewrites | API key |
| **Local** (Hugging Face) | free | Medium. Summaries/classification from small models; rewrites are validated and fall back to rule edits | `pip install -r requirements-local.txt` (1-2 GB download, CPU is fine) |
| **Offline** | free | Basic. Extractive summary, keyword classification, rule-based risks | nothing |

Set the key in `.env` (copy `.env.example`) or paste it in the sidebar. **Auto** picks OpenAI if a key exists, else Local
if installed, else Offline.

### Your $5 OpenAI budget
* Only the OpenAI engine uses credit. A typical 10-20 page contract is roughly tens of thousands of tokens across all calls,
  which at `gpt-4o-mini` rates should cost a **fraction of a cent to a few cents** - $5 should cover hundreds of documents.
  Rates change, so check https://openai.com/api/pricing and update `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` in `.env`.
* **Spend cap:** `MAX_SPEND_USD` (default $0.75 per analysis, estimated from token counts). When reached, the agent stops calling
  OpenAI and finishes with the free methods. The sidebar and report show the estimated cost.
* If a call fails (no credit, no internet, rate limit) the step automatically falls back to the free method and a note appears.
* The API is billed separately from a ChatGPT subscription - make sure the $5 is API credit.

## How it works

```
Upload -> extract -> clean + clause split -> entities -> classify -> summarise -> risk -> draft -> report
                       (LangGraph nodes, each isolated: one failing step never crashes the run)
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI (upload/paste, tabs, downloads, Q&A) |
| `legal_agent/graph.py` | LangGraph state + nodes + edges (automatic sequential fallback) |
| `legal_agent/extraction.py` | PDF (pdfplumber -> pypdf fallback, optional OCR), DOCX, TXT |
| `legal_agent/preprocessing.py` | cleaning, header/footer removal, clause segmentation |
| `legal_agent/entities.py` | parties, dates, amounts, durations, governing law |
| `legal_agent/classifier.py` + `taxonomy.py` | 18 document types: keyword evidence blended with LLM / zero-shot NLI |
| `legal_agent/risk.py` + `risk_rules.py` | **hybrid risk detector** (below) |
| `legal_agent/drafting.py` | safer wording: LLM rewrite -> validation -> rule edit -> guidance |
| `legal_agent/backends.py` | OpenAI / Local / Offline engines, all with graceful fallback |
| `legal_agent/qa.py` | question answering over the document |
| `legal_agent/report.py` | Markdown / JSON / PDF reports |

### Hybrid risk detection (fixes the ~18% recall problem)
1. **36 rules** with severity, mutual-wording downgrade, escalation words, and per-document-type rules (lease, employment, loan, ToS...).
2. **Structural checks**: unequal notice periods between the parties.
3. **Missing-clause checks**: governing law, liability cap, termination, notices, force majeure, etc.
4. **Semantic layer** (24 risky-clause prototypes): embeddings in OpenAI/Local mode, TF-IDF offline.
5. **LLM review** (OpenAI mode): reads every clause from *your* side ("I am the Tenant") and flags unbalanced terms that no keyword list can catch.

### What changed from your old design (and why)
* `flan-t5-base` could not reliably keep party names -> OpenAI rewrites + a **validator** (party roles preserved, no repetition, not unchanged) + rule-based fallback.
* zero-shot classifier confidence 0.30-0.45 -> keyword evidence blended with the model, 18 clearer labels; confidence now typically 0.8-0.95 on clear documents.
* regex-only risks -> hybrid detector above. `pdf_report.py` is now actually connected (download button).
* mid-word truncation / chunking -> sentence-aware clause segmentation; duplicates merged; no stale state between runs.
* new: DOCX/TXT/paste input, OCR hook, Q&A, perspective, missing clauses, JSON output, cost control, tests.

## Be realistic about accuracy
* `python evaluate.py` scores the rules on 4 seeded sample documents (100% there) - **but the author wrote those samples next to the rules,
  so that number is optimistic.** (`sample_lease_titlecase.pdf` is a real user document; four rules were added after seeing it, so it is not an independent test either.) On paraphrased wording the rules do not know, **Offline mode finds nothing** (we tested this).
  For real contracts use **OpenAI mode** (or at least Local with embeddings) and add your own labelled documents to `samples/` + `samples/expected.json`.
* Summaries/classification in Offline mode are basic by design.
* Local mode was written carefully but could not be run end-to-end where this project was built (no model downloads possible);
  its failure path (falls back to Offline with a notice) is tested. If a local model misbehaves, change it in `.env`.
* Scanned PDFs need OCR: `pip install pytesseract` + install the Tesseract program, then they are read automatically.
* This tool is not legal advice.

## If Windows blocks a compiled package
`ImportError: DLL load failed ... An Application Control policy has blocked this file` means Windows (Smart App Control / an admin policy)
refuses a compiled file in a freshly installed package. The agent avoids two common victims: the `openai` package (it falls back to plain HTTPS) and
scikit-learn (removed; `legal_agent/tfidf.py` is a pure-Python copy verified against it). The remaining dependencies are mostly pure Python.

## If Windows blocks the OpenAI package
Error such as `DLL load failed while importing jiter: An Application Control policy has blocked this file` means Windows (Smart App Control / an
admin policy) blocked a compiled file used by the `openai` Python package. The agent now detects this and sends the same requests to OpenAI
over plain HTTPS (standard library only), so OpenAI mode keeps working; the report notes "switched to direct HTTPS calls".

## Engine isolation (important)
Changes made to improve one engine must not alter another. The extra safeguards for small local models (instruction-echo detection,
"barely changed" / "risky wording still present" checks, stricter length rule, High->Medium cap for semantic hints, higher embedding threshold)
apply **only** to `Local`. OpenAI results are unaffected; `tests/test_backends.py` enforces this.

## Reports: short vs full
* **Short PDF** (default download, about 2 pages): summary, key facts, a compact risk table (clause no., level, short excerpt, issue + fix), missing clauses.
* **Full PDF / Markdown**: adds the explanation for every issue and the suggested rewording of each risky clause (quotes each clause again, so it is long).
* In the app, the **Clauses** tab shows the whole document with risky clauses highlighted (red/orange/yellow).

## Command line & tests
```bash
python -m legal_agent.cli samples/sample_lease.txt --backend offline --perspective Tenant --out reports/
python evaluate.py offline
pytest -q            # 28 tests: pipeline, PDF/DOCX input, fake-OpenAI + spend cap + failure fallback, Streamlit app
```

## For your viva
* **Why LangGraph?** explicit state machine; each node is independently testable and failure-isolated.
* **Why hybrid?** rules = precise and explainable; embeddings = catch paraphrases; LLM = reads context and perspective.
* **Evaluation:** per-module metrics (classification confidence/margin, risk recall/precision, rewrite validity). Extend `evaluate.py`.
* **Limitations to state honestly:** rule brittleness, small-model summaries, OCR dependence, no jurisdiction-specific legal knowledge.
