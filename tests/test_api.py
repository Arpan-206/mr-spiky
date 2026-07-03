"""HTTP-level tests for the /analyze endpoint (schema + language gating)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

PY_SAMPLE = "def f(x):\n    if x:\n        return x\n"


def test_analyze_defaults_to_python() -> None:
    r = client.post("/analyze", json={"code": PY_SAMPLE})
    assert r.status_code == 200
    body = r.json()
    assert {"verdict", "lines", "top_flagged", "dominant_axis"}.issubset(body.keys())


def test_analyze_accepts_python_language() -> None:
    r = client.post("/analyze", json={"code": PY_SAMPLE, "language": "python"})
    assert r.status_code == 200


def test_analyze_python_case_insensitive() -> None:
    r = client.post("/analyze", json={"code": PY_SAMPLE, "language": "Python"})
    assert r.status_code == 200


def test_analyze_rejects_unsupported_language() -> None:
    r = client.post("/analyze", json={"code": "int main() {}", "language": "cpp"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "cpp" in detail
    assert "python" in detail.lower()


def test_health_reports_supported_languages() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "python" in body["supported_languages"]


def test_health_reports_mode() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in {"snn", "mock"}
    assert isinstance(body["threshold"], (int, float))
    if body["mode"] == "snn":
        # Real SNN mode should expose diagnostic fields
        assert isinstance(body["hidden_size"], int) and body["hidden_size"] > 0
        assert isinstance(body["output_size"], int) and body["output_size"] > 0
        assert isinstance(body["hidden_baselines_distinct"], int)
        assert isinstance(body["ecdf_reference_size"], int)
    else:
        # Mock mode should say why
        assert isinstance(body.get("reason"), str) and body["reason"]
