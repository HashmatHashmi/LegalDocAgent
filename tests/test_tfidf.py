import os

import pytest

from legal_agent.config import Settings
from legal_agent.graph import Pipeline
from legal_agent.risk_rules import PROTOTYPES
from legal_agent.tfidf import TfidfModel, cosine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_matches_sklearn_numerically():
    try:
        import sklearn.feature_extraction.text as sk
    except ImportError:  # e.g. blocked by Windows Application Control, or not installed
        pytest.skip("scikit-learn not importable here (that is exactly why tfidf.py exists)")
    r = Pipeline(Settings(backend="offline")).run(open(os.path.join(ROOT, "samples", "sample_lease_titlecase.pdf"), "rb").read(), "l.pdf")
    docs = [p[2] for p in PROTOTYPES] + [c["text"] for c in r["clauses"]]
    queries = ["How much is the monthly rent?", "Who pays for repairs and maintenance?", "notice period to terminate"]

    vec = sk.TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(docs)
    m = TfidfModel()
    mine = m.fit_transform(docs)
    n = len(docs)
    for i in range(0, n, 3):
        for j in range(0, n, 5):
            ref = float((mat[i] @ mat[j].T).toarray()[0, 0])
            assert abs(cosine(mine[i], mine[j]) - ref) < 1e-9
    # query side (transform with the fitted vocabulary)
    qm = vec.transform(queries)
    mq = m.transform(queries)
    for qi in range(len(queries)):
        for j in range(n):
            ref = float((qm[qi] @ mat[j].T).toarray()[0, 0])
            assert abs(cosine(mq[qi], mine[j]) - ref) < 1e-9


def test_edge_cases():
    m = TfidfModel()
    assert m.fit_transform(["", "the and of"]) == [{}, {}]
    assert m.transform(["anything"]) == [{}]
