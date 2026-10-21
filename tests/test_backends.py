import json
from types import SimpleNamespace

import numpy as np

from legal_agent.backends import OpenAIBackend, make_backend
from legal_agent.config import Settings
from legal_agent.graph import Pipeline

LEASE = open(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "samples", "sample_lease.txt"), "rb").read()


class FakeOpenAI:
    """Mimics the parts of the OpenAI client we use. Counts calls and can fail on demand."""

    def __init__(self, fail=False, tokens=(1000, 200)):
        self.calls = 0
        self.fail = fail
        self.tokens = tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._emb)

    def _chat(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("insufficient_quota")
        prompt = kw["messages"][-1]["content"]
        if "Classify this legal document" in prompt:
            out = {"label": "Lease / Rental Agreement", "confidence": 0.93, "runner_up": None}
        elif '"risks"' in prompt:
            out = {"risks": [
                {"clause_id": 3, "category": "Unilateral rent increase", "severity": "High", "explanation": "Rent can rise at will.", "suggestion": "Fix the rent."},
                {"clause_id": 999, "category": "bogus", "severity": "High", "explanation": "", "suggestion": ""},
                {"clause_id": 4, "category": "x", "severity": "Catastrophic", "explanation": "", "suggestion": ""},
            ]}
        elif '"overview"' in prompt:
            out = {"overview": "A residential lease with several one-sided terms.", "key_points": ["Rent Rs. 45,000", "11 months"]}
        elif "Rewrite this contract clause" in prompt:
            return self._resp("The Landlord may adjust the rent once per year with sixty (60) days written notice, capped at 10%.")
        else:
            out = {}
        return self._resp(json.dumps(out))

    def _resp(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1]))

    def _emb(self, **kw):
        n = len(kw["input"])
        return SimpleNamespace(data=[SimpleNamespace(embedding=list(np.random.rand(8))) for _ in range(n)], usage=SimpleNamespace(total_tokens=10 * n))


def make_openai(fake, **kw):
    s = Settings(backend="openai", openai_api_key="sk-test", **kw)
    b = OpenAIBackend(s)
    b.client = fake
    return b


def test_openai_happy_path_uses_llm_everywhere():
    fake = FakeOpenAI()
    b = make_openai(fake)
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert r["ok"]
    assert r["summary"]["overview"].startswith("A residential lease")
    assert "llm" in r["risk_layers"]
    assert any(f["source"].startswith("llm") and f["clause_id"] == 3 for f in r["risks"])
    assert all(f["clause_id"] != 999 for f in r["risks"])          # invalid ids dropped
    assert all(f["severity"] in ("High", "Medium", "Low") for f in r["risks"])
    assert r["classification"]["confidence"] > 0.8
    assert any(v["method"].startswith("openai") for v in r["revisions"])
    assert r["usage"]["estimated_cost_usd"] > 0 and r["usage"]["calls"] == fake.calls


def test_openai_failure_falls_back_cleanly():
    b = make_openai(FakeOpenAI(fail=True))
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert r["ok"] and r["risks"] and r["summary"]["overview"]
    assert any("failed" in w for w in r["warnings"])


def test_spend_cap_stops_calls():
    fake = FakeOpenAI(tokens=(2_000_000, 0))  # one call = $0.30 at default prices
    b = make_openai(fake, max_spend_usd=0.25)
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert r["ok"]
    assert fake.calls == 1
    assert any("Spend cap" in w for w in r["warnings"])


def test_make_backend_degrades():
    b = make_backend(Settings(backend="openai", openai_api_key=""))
    assert b.name == "offline" and b.warnings
    b = make_backend(Settings(backend="local"))
    assert b.name in ("offline", "local")


def test_llm_and_rule_findings_are_merged_per_clause():
    b = make_openai(FakeOpenAI())
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    clause3 = [f for f in r["risks"] if f["clause_id"] == 3]
    assert len(clause3) == 1                       # LLM finding + rule hits folded into one card
    assert clause3[0]["source"] == "llm + rules" and clause3[0]["also"]
    assert "price_increase" in clause3[0]["rule_ids"]


def test_bad_rewrite_is_retried_then_explained():
    from legal_agent.drafting import rewrite_problem

    assert rewrite_problem("The Landlord may enter at any time.", "", ["Landlord"]) == "empty output"
    assert "Landlord" in rewrite_problem("The Landlord may enter at any time.", "The owner may enter with notice to the occupant.", ["Landlord"])

    class BadRewrite(FakeOpenAI):
        def _chat(self, **kw):
            if "Rewrite this contract clause" in kw["messages"][-1]["content"]:
                self.calls += 1
                return self._resp("Sorry, I cannot do that.")
            return super()._chat(**kw)

    b = make_openai(BadRewrite())
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert r["ok"] and r["revisions"]
    assert any("rewrite rejected" in w for w in r["warnings"])


def test_mutual_rewrite_is_accepted_but_name_dropping_is_not():
    from legal_agent.drafting import rewrite_problem

    orig = "The Landlord may terminate this lease at any time without notice if the Landlord believes the Tenant has violated the agreement."
    mutual = "Either party may terminate this lease on thirty (30) days written notice, and a party alleged to be in breach may cure within fifteen (15) days of that notice."
    assert rewrite_problem(orig, mutual, ["Landlord", "Tenant"]) is None
    assert rewrite_problem(orig, "The owner may end the arrangement with notice to the occupant after a documented breach.", ["Landlord", "Tenant"])


def test_separate_rewrite_model_is_used_and_priced_higher():
    models = []

    class Rec(FakeOpenAI):
        def _chat(self, **kw):
            models.append((kw["model"], "Rewrite this contract clause" in kw["messages"][-1]["content"]))
            return super()._chat(**kw)

    b = make_openai(Rec(), openai_rewrite_model="big-model")
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert any(m == "big-model" and is_rw for m, is_rw in models)
    assert all(m == "gpt-4o-mini" for m, is_rw in models if not is_rw)
    assert "rewrites: big-model" in r["backend"]
    base_only = make_openai(FakeOpenAI())
    Pipeline(base_only.s, base_only).run(LEASE, "lease.txt")
    assert b.cost() > base_only.cost()


def test_strict_validation_rejects_bad_small_model_output():
    from legal_agent.drafting import rewrite_problem

    orig = "The Landlord may terminate this lease at any time without notice if the Landlord believes the Tenant has violated any condition of this agreement."
    roles = ["Landlord", "Tenant"]
    echo = ("One-sided termination right", "One party can end the agreement freely, leaving the other without security or time to react.")
    leak = "One-sided termination right: One party can end the agreement freely, leaving the other without security or time to react. Keep these party names exactly as written: Landlord, Tenant.."
    assert "repeated the instructions" in rewrite_problem(orig, leak, roles, True, ["one_sided_termination"], echo)
    assert rewrite_problem(orig, "The Landlord may terminate this lease at any time if the Landlord believes the Tenant has violated the agreement.", roles, True, ["one_sided_termination"])
    assert rewrite_problem("The Tenant shall pay all repairs including structural repairs, plumbing and roof damage regardless of cause.", "All repairs shall be the Tenant's.", roles, True)  # dropped content
    good = "Either party may terminate this lease on thirty (30) days written notice if the other party is in material breach and fails to remedy it within that period."
    assert rewrite_problem(orig, good, roles, True, ["one_sided_termination"], echo) is None


def test_local_embedding_threshold_and_semantic_severity_cap():
    from legal_agent.backends import LocalBackend, OfflineBackend
    from legal_agent.risk import semantic_findings

    assert LocalBackend.embed_threshold >= 0.6

    class LocalLike(OfflineBackend):  # same TF-IDF path, but reports itself as "local"
        name = "local"

    clauses = [{"id": 1, "heading": "", "text": "The Landlord may enter the premises at any time and without any prior notice whatsoever for any purpose."}]
    sem, _ = semantic_findings(clauses, LocalLike(Settings(backend="offline")), set())
    assert sem and all(f["severity"] != "High" for f in sem)
    sem2, _ = semantic_findings(clauses, OfflineBackend(Settings(backend="offline")), set())
    assert any(f["severity"] == "High" for f in sem2)  # offline behaviour is unchanged


def test_local_only_checks_do_not_touch_the_openai_path():
    """Guarantee: the extra strictness for small local models never applies to OpenAI rewrites (strict=False)."""
    from legal_agent.drafting import rewrite_problem

    orig = "The Landlord may terminate this lease at any time without notice if the Landlord believes the Tenant has violated any condition."
    roles = ["Landlord", "Tenant"]
    echo = ("One-sided termination right", "One party can end the agreement freely, leaving the other without security or time to react.")
    # near-identical text and instruction-like wording: rejected only in strict (local) mode
    near = "The Landlord may terminate this lease at any time without notice if the Landlord believes the Tenant has violated any condition of it."
    leak = "Either party may terminate on thirty (30) days written notice if the other is in material breach. Keep these party names exactly as written: Landlord, Tenant."
    assert rewrite_problem(orig, near, roles, False, ["one_sided_termination"], echo) is None
    assert rewrite_problem(orig, leak, roles, False, ["one_sided_termination"], echo) is None
    assert rewrite_problem(orig, near, roles, True, ["one_sided_termination"], echo) is not None
    assert rewrite_problem(orig, leak, roles, True, ["one_sided_termination"], echo) is not None


def test_openai_still_works_when_the_sdk_is_blocked_by_windows(monkeypatch):
    """Simulates: ImportError: DLL load failed while importing jiter (Application Control policy)."""
    import legal_agent.backends as be

    reference = FakeOpenAI()

    class Blocked(FakeOpenAI):
        def _chat(self, **kw):
            raise ImportError("DLL load failed while importing jiter: An Application Control policy has blocked this file.")

    seen = []

    def fake_http(path, payload, api_key, timeout=60.0, base_url=None):
        seen.append(path)
        assert api_key == "sk-test" and path == "/chat/completions"
        r = reference._chat(**payload)
        return {"choices": [{"message": {"content": r.choices[0].message.content}}],
                "usage": {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}}

    monkeypatch.setattr(be, "_http_json", fake_http)
    b = make_openai(Blocked())
    r = Pipeline(b.s, b).run(LEASE, "lease.txt")
    assert r["ok"] and seen
    assert "llm" in r["risk_layers"]                      # real GPT review, not the free fallback
    assert r["summary"]["overview"].startswith("A residential lease")
    assert any(v["method"].startswith("openai") for v in r["revisions"])
    assert any("direct HTTPS" in w for w in r["warnings"])
    assert r["usage"]["estimated_cost_usd"] > 0


def test_http_helper_reports_api_errors_clearly(monkeypatch):
    import io
    import urllib.error

    import legal_agent.backends as be

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"message":"Incorrect API key"}}'))

    monkeypatch.setattr(be.urllib.request, "urlopen", boom)
    try:
        be._http_json("/chat/completions", {"model": "x"}, "sk-bad")
        assert False, "should raise"
    except RuntimeError as e:
        assert "401" in str(e) and "Incorrect API key" in str(e)


def test_http_fallback_over_a_real_socket():
    """Runs the plain-HTTPS path against a tiny local server that imitates /v1/chat/completions."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import legal_agent.backends as be

    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got["auth"], got["path"], got["model"] = self.headers["Authorization"], self.path, body["model"]
            out = json.dumps({"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = be._http_json("/chat/completions", {"model": "gpt-4o-mini", "messages": []}, "sk-abc",
                             base_url=f"http://127.0.0.1:{srv.server_port}/v1")
    finally:
        srv.shutdown()
    assert data["choices"][0]["message"]["content"] == "hello"
    assert got == {"auth": "Bearer sk-abc", "path": "/v1/chat/completions", "model": "gpt-4o-mini"}
    assert be._ns(data).choices[0].message.content == "hello"
