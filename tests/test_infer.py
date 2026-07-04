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
    # Required keys (dominant_axis is None-able but always present).
    required = {"verdict", "lines", "top_flagged", "dominant_axis"}
    assert required.issubset(result.keys())
    assert isinstance(result["verdict"], str) and result["verdict"]
    assert isinstance(result["lines"], list)
    assert isinstance(result["top_flagged"], list)
    assert result["dominant_axis"] is None or isinstance(result["dominant_axis"], str)
    for entry in result["lines"]:
        assert {"line", "score", "flag", "axes"}.issubset(entry.keys())
        assert isinstance(entry["line"], int) and entry["line"] >= 1
        assert isinstance(entry["score"], float) and 0.0 <= entry["score"] <= 1.0
        assert isinstance(entry["flag"], bool)
        assert isinstance(entry["axes"], dict)
        # Each axis is a float in [0, 1]-ish (occasionally slightly above due
        # to the p95 rescaler, so allow up to 1.05).
        for name, val in entry["axes"].items():
            assert isinstance(name, str)
            assert isinstance(val, float) and 0.0 <= val <= 1.05
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


def test_analyze_syntax_error_line_carries_malformed_axis() -> None:
    result = analyze("if boolean = True:\n    pass\n")
    _assert_schema(result)
    line1 = next(e for e in result["lines"] if e["line"] == 1)
    assert line1["axes"].get("malformed", 0.0) > 0.0
    other_lines = [e for e in result["lines"] if e["line"] != 1]
    assert all(e["axes"].get("malformed", 0.0) == 0.0 for e in other_lines)