"""Smoke-test the risk detector against the seeded sample documents.

    python evaluate.py [offline|openai|local]

NOTE: these 4 samples were written by the project author with known risks seeded in. This measures
that the detector works on them - it is NOT a benchmark of real-world recall. Add your own
labelled documents to samples/ and samples/expected.json to measure what matters to you.
"""
import json
import os
import sys
from dataclasses import replace

from legal_agent.config import Settings
from legal_agent.graph import Pipeline

backend = sys.argv[1] if len(sys.argv) > 1 else "offline"
settings = replace(Settings.from_env(), backend=backend)
pipe = Pipeline(settings)
exp = json.load(open(os.path.join("samples", "expected.json"), encoding="utf-8"))

tp = fn = fp = 0
for fname, want in exp.items():
    res = pipe.run(open(os.path.join("samples", fname), "rb").read(), fname)
    got = {rid for f in res["risks"] for rid in f.get("rule_ids", [])}
    want = set(want)
    tp += len(want & got)
    fn += len(want - got)
    print(f"{fname:28s} type={res['classification']['label'][:28]:28s} conf={res['classification']['confidence']:.2f} "
          f"recall={len(want & got)}/{len(want)} missed={sorted(want - got)} risk={res['risk_level']}")
    if not want:
        fp += len(res["risks"])
print(f"\nSeeded-risk recall: {tp}/{tp + fn} = {tp / max(1, tp + fn):.0%}   |   findings on the clean mutual NDA: {fp}")
print("Engine:", pipe.backend.usage())
