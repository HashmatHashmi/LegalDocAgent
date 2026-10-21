import io
import json
import os

import pytest

from legal_agent.config import Settings
from legal_agent.graph import Pipeline
from legal_agent.qa import DocumentIndex, ask
from legal_agent.report import build_json, build_markdown, build_pdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")


def read(name):
    with open(os.path.join(SAMPLES, name), "rb") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(Settings(backend="offline"))


@pytest.mark.parametrize("use_lg", [True, False])
def test_both_workflow_engines(use_lg):
    p = Pipeline(Settings(backend="offline"), use_langgraph=use_lg)
    r = p.run(read("sample_lease.txt"), "sample_lease.txt")
    assert r["ok"] and r["workflow_engine"] == ("langgraph" if use_lg else "sequential")
    assert r["classification"]["label"].startswith("Lease")


def test_seeded_risks_are_found(pipe):
    exp = json.load(open(os.path.join(SAMPLES, "expected.json")))
    for fname, want in exp.items():
        r = pipe.run(read(fname), fname)
        got = {f["rule_id"] for f in r["risks"] if f["rule_id"]}
        assert set(want) <= got, f"{fname} missed {set(want) - got}"


def test_clean_mutual_nda_has_no_findings(pipe):
    r = pipe.run(read("sample_nda_mutual.txt"), "sample_nda_mutual.txt")
    assert r["risks"] == []
    assert r["risk_level"] == "Low"
    assert r["classification"]["label"].startswith("Non-Disclosure")
    assert r["entities"]["governing_law"] == "Pakistan"


def test_entities(pipe):
    r = pipe.run(read("sample_employment.txt"), "e.txt")
    roles = {p["role"]: p["name"] for p in r["entities"]["parties"]}
    assert roles["Company"].startswith("Zenith")
    assert "Rs. 120,000" in r["entities"]["amounts"]


def test_reports_build(pipe):
    r = pipe.run(read("sample_service.txt"), "s.txt")
    assert "Legal Document Analysis Report" in build_markdown(r)
    assert json.loads(build_json(r))["classification"]["label"]
    assert build_pdf(r)[:5] == b"%PDF-"


def test_qa(pipe):
    r = pipe.run(read("sample_lease.txt"), "l.txt")
    out = ask(DocumentIndex(r["clauses"]), "How much is the monthly rent?", pipe.backend)
    assert "45,000" in out["answer"]
    out = ask(DocumentIndex(r["clauses"]), "zzzz qqqq", pipe.backend)
    assert "could not find" in out["answer"].lower()


def test_empty_and_garbage_inputs_do_not_crash(pipe):
    assert pipe.run_text("")["ok"] is False
    assert pipe.run(b"", "x.pdf")["ok"] is False
    assert pipe.run(b"not a pdf at all", "x.pdf")["ok"] is False
    assert pipe.run(b"\x00\x01\x02", "x.xyz")["ok"] is False
    r = pipe.run_text("Hello world. This is a short note about lunch plans for tomorrow, nothing legal here at all.")
    assert r["ok"] and r["classification"]["label"].startswith("Other")


def test_pdf_and_docx_input(pipe):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    text = read("sample_lease.txt").decode()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.splitlines():
        c.drawString(40, y, line[:110])
        y -= 14
        if y < 40:
            c.showPage()
            y = 800
    c.save()
    r = pipe.run(buf.getvalue(), "lease.pdf")
    assert r["ok"] and r["classification"]["label"].startswith("Lease") and r["risks"]

    import docx

    d = docx.Document()
    for line in text.splitlines():
        d.add_paragraph(line)
    b = io.BytesIO()
    d.save(b)
    r = pipe.run(b.getvalue(), "lease.docx")
    assert r["ok"] and r["classification"]["label"].startswith("Lease") and r["risks"]


def test_offline_summary_has_no_glued_headings():
    p = Pipeline(Settings(backend="offline"))
    r = p.run(read("sample_lease_titlecase.pdf"), "x.pdf")
    blob = r["summary"]["overview"] + " ".join(r["summary"]["key_points"])
    for glued in ("Property The Landlord", "Late Payment If rent", "Subletting The Tenant", "Utilities The Tenant", "Maintenance All repairs"):
        assert glued not in blob


def test_short_report_is_shorter_than_full_report(pipe):
    from pypdf import PdfReader

    r = pipe.run(read("sample_service.txt"), "s.txt")
    short, full = build_pdf(r, "compact"), build_pdf(r, "detailed")
    assert short[:5] == full[:5] == b"%PDF-"
    n_short, n_full = len(PdfReader(io.BytesIO(short)).pages), len(PdfReader(io.BytesIO(full)).pages)
    assert n_short < n_full and n_short <= 3
