"""REST API and Streamlit dashboard smoke tests."""
import io

import pytest

pytest.importorskip("torch")

from xgnn_ids.data.synthetic import generate_traffic  # noqa: E402


def test_api(trained):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.main as api_main

    api_main.MODELS_DIR = str(trained["models"])
    api_main.detector.cache_clear()
    client = TestClient(api_main.app)
    assert client.get("/health").json()["status"] == "ok"
    assert "egraphsage" in client.get("/models").json()

    csv = generate_traffic("enterprise", duration_s=120, seed=5).to_csv(index=False).encode()
    r = client.post("/detect", files={"file": ("flows.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["flows"] > 0 and body["alerts"] > 0
    row = body["alert_rows"][0]["row"]
    r = client.post("/explain", json={"row": row})
    assert r.status_code == 200, r.text
    assert "narrative" in r.json()

    bad = client.post("/detect", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 400


def test_dashboard_renders(trained):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 5
