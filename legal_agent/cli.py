"""Command-line interface:  python -m legal_agent.cli contract.pdf --backend offline --out reports/"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from .config import Settings
from .graph import Pipeline
from .report import build_json, build_markdown, build_pdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse a legal document (PDF / DOCX / TXT).")
    ap.add_argument("file")
    ap.add_argument("--backend", choices=["auto", "openai", "local", "offline"], default=None)
    ap.add_argument("--perspective", default=None, help='Which party are you? e.g. "Tenant"')
    ap.add_argument("--out", default="reports")
    ap.add_argument("--no-rewrite", action="store_true")
    a = ap.parse_args(argv)

    s = Settings.from_env()
    if a.backend:
        s = replace(s, backend=a.backend)
    if a.perspective is not None:
        s = replace(s, perspective=a.perspective)
    if a.no_rewrite:
        s = replace(s, enable_rewrites=False)

    with open(a.file, "rb") as fh:
        data = fh.read()
    res = Pipeline(s, progress=lambda l, i, n: print(f"[{i + 1}/{n}] {l}...")).run(data, os.path.basename(a.file))
    if not res.get("ok"):
        print("Analysis failed:", "; ".join(res.get("errors", [])) or "unknown error")
        return 1

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.file))[0]
    with open(os.path.join(a.out, f"{stem}_report.md"), "w", encoding="utf-8") as fh:
        fh.write(build_markdown(res))
    with open(os.path.join(a.out, f"{stem}_report.json"), "w", encoding="utf-8") as fh:
        fh.write(build_json(res))
    with open(os.path.join(a.out, f"{stem}_report.pdf"), "wb") as fh:
        fh.write(build_pdf(res, "compact"))
    with open(os.path.join(a.out, f"{stem}_report_full.pdf"), "wb") as fh:
        fh.write(build_pdf(res, "detailed"))
    cls = res["classification"]
    print(f"\nType: {cls['label']} ({cls['confidence']:.0%}) | Risk: {res['risk_level']} ({res['risk_score']}/100) | "
          f"{len(res['risks'])} risky clauses | engine: {res['backend']}")
    for w in res["warnings"]:
        print("note:", w)
    print(f"Reports saved in {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
