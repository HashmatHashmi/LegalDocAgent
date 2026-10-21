import os

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_runs_with_pasted_text():
    text = open(os.path.join(ROOT, "samples", "sample_lease.txt"), encoding="utf-8").read()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
    assert not at.exception
    at.selectbox[0].set_value("Offline rules - free, instant, no downloads")
    at.text_area[0].set_value(text)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Document type" in labels and "Overall risk" in labels
    assert any("Lease" in m.value for m in at.metric)
