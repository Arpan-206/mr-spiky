"""Smoke test: infer.analyze returns the fixed response schema for a snippet."""

from __future__ import annotations

from src.infer import analyze

SAMPLE = """
def gnarly(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    return 
    x
"""


def _assert_schema(result: dict) -> None:
    assert set(result.keys()) == {"verdict", "lines", "top_flagged"}
    assert isinstance(result["verdict"], str) and result["verdict"]
    assert isinstance(result["lines"], list)
    assert isinstance(result["top_flagged"], list)
    for entry in result["lines"]:
        assert set(entry.keys()) == {"line", "score", "flag"}
        assert isinstance(entry["line"], int) and entry["line"] >= 1
        assert isinstance(entry["score"], float) and 0.0 <= entry["score"] <= 1.0
        assert isinstance(entry["flag"], bool)
    for ln in result["top_flagged"]:
        assert isinstance(ln, int)


def test_analyze_returns_valid_schema() -> None:
    _assert_schema(analyze(SAMPLE))


def test_analyze_handles_empty_input() -> None:
    result = analyze("")
    _assert_schema(result)
    assert result["lines"] == []
    assert result["top_flagged"] == []


def test_analyze_handles_syntax_error() -> None:
    result = analyze("def broken(:\n    return")
    _assert_schema(result)  # must still return valid JSON, not raise