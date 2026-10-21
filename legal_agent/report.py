"""Report builders: Markdown, JSON and PDF (ReportLab)."""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict
from xml.sax.saxutils import escape

DISCLAIMER = (
    "This report is generated automatically to help you understand a document. It is not legal advice and can miss or "
    "misjudge clauses. Have a qualified lawyer review important documents before you sign."
)
_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}


def build_markdown(r: Dict) -> str:
    cls = r.get("classification", {})
    summ = r.get("summary", {})
    ent = r.get("entities", {})
    lines = ["# Legal Document Analysis Report", ""]
    lines.append(f"- **File:** {r.get('filename', '-')}")
    lines.append(f"- **Generated:** {r.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M'))}")
    lines.append(f"- **Document type:** {cls.get('label', '-')} (confidence {cls.get('confidence', 0):.0%})")
    lines.append(f"- **Overall risk:** {r.get('risk_level', '-')} ({r.get('risk_score', 0)}/100)")
    lines.append(f"- **Analysis engine:** {r.get('backend', '-')}")
    lines += ["", "## Summary", "", summ.get("overview", "_No summary available._"), ""]
    if summ.get("key_points"):
        lines += ["### Key points", ""] + [f"- {p}" for p in summ["key_points"]] + [""]
    lines += ["## Key facts", ""]
    parties = ent.get("parties", [])
    if parties:
        lines.append("- **Parties:** " + "; ".join(f"{p['name']} ({p['role']})" for p in parties))
    for label, key in (("Dates", "dates"), ("Amounts", "amounts"), ("Durations / deadlines", "durations"), ("Jurisdictions", "jurisdictions")):
        if ent.get(key):
            lines.append(f"- **{label}:** " + ", ".join(ent[key]))
    if ent.get("governing_law"):
        lines.append(f"- **Governing law:** {ent['governing_law']}")
    lines += ["", "## Risky clauses", ""]
    findings = r.get("risks", [])
    if not findings:
        lines.append("_No risky clauses were detected by the active detectors._")
    for f in findings:
        lines.append(f"### {_ICON.get(f['severity'], '')} {f['severity']} - {f['category']} (clause {f['clause_id']})")
        lines.append(f"> {f['excerpt']}")
        lines.append("")
        lines.append(f"**Why it matters:** {f['explanation']}  ")
        lines.append(f"**Suggested fix:** {f['suggestion']}  ")
        if f.get("also"):
            lines.append(f"_Also matched rules: {', '.join(f['also'])}_  ")
        lines.append(f"_Detected by: {f['source']}_")
        lines.append("")
    miss = r.get("missing_clauses", [])
    if miss:
        lines += ["## Possibly missing clauses", ""]
        for m in miss:
            lines.append(f"- **{m['name']}** ({m['severity']}): {m['explanation']} _{m['suggestion']}_")
        lines.append("")
    revs = r.get("revisions", [])
    if revs:
        lines += ["## Suggested revisions", "", "_Figures in suggested wording (days, caps, percentages) are proposals to negotiate, not legal advice._", ""]
        for v in revs:
            lines.append(f"### Clause {v['clause_id']} - {v['category']}")
            lines.append(f"**Original:** {v['original']}")
            lines.append("")
            if v["revised"]:
                lines.append(f"**Suggested wording** ({v['method']}): {v['revised']}")
            else:
                lines.append(f"**Guidance:** {v['guidance']}")
            lines.append("")
    if r.get("warnings"):
        lines += ["## Notes", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    lines += ["---", f"_{DISCLAIMER}_"]
    return "\n".join(lines)


def build_json(r: Dict) -> str:
    keep = {k: v for k, v in r.items() if k not in ("raw_text",)}
    return json.dumps(keep, indent=2, ensure_ascii=False, default=str)


_REPL = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "•": "-", "…": "...", "\u2022": "-", "🔴": "", "🟠": "", "🟡": ""}


def _t(text) -> str:
    s = str(text if text is not None else "")
    for a, b in _REPL.items():
        s = s.replace(a, b)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return escape(s)


def build_pdf_compact(r: Dict) -> bytes:
    """One-to-two page summary: header, summary, a compact risk table, missing clauses. No repeated clause text."""
    from collections import Counter

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    def clip(text: str, n: int) -> str:
        text = " ".join(str(text or "").split())
        return text if len(text) <= n else text[: n - 1].rstrip() + "..."

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Legal Document Analysis - Short Report")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
    sev_col = {"High": "#B91C1C", "Medium": "#C2410C", "Low": "#A16207"}

    cls, summ, ent = r.get("classification", {}), r.get("summary", {}), r.get("entities", {})
    s = [Paragraph("Legal Document Analysis - Short Report", ss["Title"])]
    meta = [
        ["File", _t(r.get("filename", "-"))],
        ["Document type", _t(f"{cls.get('label', '-')} (confidence {cls.get('confidence', 0):.0%})")],
        ["Overall risk", _t(f"{r.get('risk_level', '-')} ({r.get('risk_score', 0)}/100)")],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", body), Paragraph(b, body)] for a, b in meta], colWidths=[35 * mm, 135 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [t, Spacer(1, 8), Paragraph("Summary", ss["Heading2"]), Paragraph(_t(summ.get("overview", "No summary available.")), body)]

    facts = []
    if ent.get("parties"):
        facts.append("<b>Parties:</b> " + _t("; ".join(f"{p['name']} ({p['role']})" for p in ent["parties"])))
    if ent.get("amounts"):
        facts.append("<b>Amounts:</b> " + _t(", ".join(ent["amounts"][:4])))
    if ent.get("dates"):
        facts.append("<b>Dates:</b> " + _t(", ".join(ent["dates"][:3])))
    if facts:
        s += [Spacer(1, 4), Paragraph(" &nbsp;|&nbsp; ".join(facts), body)]

    risks = r.get("risks", [])
    c = Counter(f["severity"] for f in risks)
    s.append(Paragraph("Risky clauses", ss["Heading2"]))
    if not risks:
        s.append(Paragraph("No risky clauses were detected by the active detectors.", body))
    else:
        s.append(Paragraph(f"{len(risks)} issues: {c.get('High', 0)} high, {c.get('Medium', 0)} medium, {c.get('Low', 0)} low.", body))
        s.append(Spacer(1, 4))
        rows = [[Paragraph("<b>No.</b>", cell), Paragraph("<b>Level</b>", cell), Paragraph("<b>What it says</b>", cell),
                 Paragraph("<b>Issue and suggested fix</b>", cell)]]
        for f in risks:
            rows.append([
                Paragraph(str(f["clause_id"]), cell),
                Paragraph(f"<font color='{sev_col.get(f['severity'], '#000000')}'><b>{f['severity']}</b></font>", cell),
                Paragraph(_t(clip(f["excerpt"], 120)), cell),
                Paragraph(f"<b>{_t(f['category'])}</b><br/>Fix: {_t(clip(f['suggestion'], 170))}", cell),
            ])
        tbl = Table(rows, colWidths=[13 * mm, 16 * mm, 60 * mm, 81 * mm], repeatRows=1)
        tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        s.append(tbl)

    if r.get("missing_clauses"):
        s.append(Paragraph("Possibly missing clauses", ss["Heading2"]))
        s.append(Paragraph(_t(", ".join(f"{m['name']} ({m['severity']})" for m in r["missing_clauses"])), body))
    s += [Spacer(1, 8), Paragraph("Explanations and suggested rewording for each issue are in the full report.", small),
          Spacer(1, 4), Paragraph(_t(DISCLAIMER), small)]
    doc.build(s)
    return buf.getvalue()


def build_pdf(r: Dict, style: str = "detailed") -> bytes:
    if style == "compact":
        return build_pdf_compact(r)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Legal Document Analysis Report")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
    quote = ParagraphStyle("quote", parent=body, leftIndent=8, textColor=colors.HexColor("#333333"), backColor=colors.HexColor("#F3F4F6"))
    h1, h2, h3 = ss["Title"], ss["Heading2"], ParagraphStyle("h3", parent=ss["Heading4"], spaceBefore=6)
    sev_col = {"High": "#B91C1C", "Medium": "#C2410C", "Low": "#A16207"}

    cls, summ, ent = r.get("classification", {}), r.get("summary", {}), r.get("entities", {})
    s = [Paragraph("Legal Document Analysis Report", h1)]
    meta = [
        ["File", _t(r.get("filename", "-"))],
        ["Document type", _t(f"{cls.get('label', '-')} (confidence {cls.get('confidence', 0):.0%})")],
        ["Overall risk", _t(f"{r.get('risk_level', '-')} ({r.get('risk_score', 0)}/100)")],
        ["Engine", _t(r.get("backend", "-"))],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", body), Paragraph(b, body)] for a, b in meta], colWidths=[35 * mm, 135 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [t, Spacer(1, 8), Paragraph("Summary", h2), Paragraph(_t(summ.get("overview", "No summary available.")), body)]
    for p in summ.get("key_points", []):
        s.append(Paragraph("- " + _t(p), body))

    facts = []
    if ent.get("parties"):
        facts.append("<b>Parties:</b> " + _t("; ".join(f"{p['name']} ({p['role']})" for p in ent["parties"])))
    for label, key in (("Dates", "dates"), ("Amounts", "amounts"), ("Durations", "durations"), ("Jurisdictions", "jurisdictions")):
        if ent.get(key):
            facts.append(f"<b>{label}:</b> " + _t(", ".join(ent[key])))
    if ent.get("governing_law"):
        facts.append("<b>Governing law:</b> " + _t(ent["governing_law"]))
    if facts:
        s.append(Paragraph("Key facts", h2))
        s += [Paragraph(f, body) for f in facts]

    s.append(Paragraph("Risky clauses", h2))
    if not r.get("risks"):
        s.append(Paragraph("No risky clauses were detected by the active detectors.", body))
    for f in r.get("risks", []):
        s.append(Paragraph(f"<font color='{sev_col.get(f['severity'], '#000000')}'><b>{f['severity']}</b></font> - "
                           f"<b>{_t(f['category'])}</b> (clause {f['clause_id']})", h3))
        s.append(Paragraph(_t(f["excerpt"]), quote))
        s.append(Paragraph("<b>Why:</b> " + _t(f["explanation"]), body))
        s.append(Paragraph("<b>Fix:</b> " + _t(f["suggestion"]), body))
        if f.get("also"):
            s.append(Paragraph("Also matched rules: " + _t(", ".join(f["also"])), small))
        s.append(Spacer(1, 3))

    if r.get("missing_clauses"):
        s.append(Paragraph("Possibly missing clauses", h2))
        for m in r["missing_clauses"]:
            s.append(Paragraph(f"<b>{_t(m['name'])}</b> ({m['severity']}): {_t(m['explanation'])} <i>{_t(m['suggestion'])}</i>", body))

    if r.get("revisions"):
        s.append(Paragraph("Suggested revisions", h2))
        s.append(Paragraph("Figures in suggested wording (days, caps, percentages) are proposals to negotiate, not legal advice.", small))
        for v in r["revisions"]:
            s.append(Paragraph(f"Clause {v['clause_id']} - {_t(v['category'])}", h3))
            s.append(Paragraph("<b>Original:</b> " + _t(v["original"]), quote))
            if v["revised"]:
                s.append(Paragraph(f"<b>Suggested ({_t(v['method'])}):</b> " + _t(v["revised"]), body))
            else:
                s.append(Paragraph("<b>Guidance:</b> " + _t(v["guidance"]), body))
            s.append(Spacer(1, 3))
    if r.get("warnings"):
        s.append(Paragraph("Notes", h2))
        s += [Paragraph("- " + _t(w), small) for w in r["warnings"]]
    s += [Spacer(1, 10), Paragraph(_t(DISCLAIMER), small)]
    doc.build(s)
    return buf.getvalue()
