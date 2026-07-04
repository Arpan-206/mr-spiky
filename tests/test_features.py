"""Unit tests for the parse_error feature: deterministic, no trained model involved."""

from __future__ import annotations

from src.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_function_features,
    extract_line_features,
)

_IDX = FEATURE_NAMES.index("parse_error")


def test_num_features_includes_parse_error() -> None:
    assert NUM_FEATURES == 10
    assert "parse_error" in FEATURE_NAMES


def test_syntax_error_line_gets_parse_error_flag() -> None:
    result = extract_line_features("if boolean = True:\n    pass\n")
    by_line = {lf.line: lf.vector[_IDX] for lf in result}
    assert by_line[1] == 1.0
    assert by_line[2] == 0.0


def test_clean_code_has_no_parse_error() -> None:
    src = (
        "def f(x):\n"
        "    if x == True:\n"
        "        return 1\n"
        "    return 0\n"
    )
    result = extract_line_features(src)
    assert result  # sanity: lines were actually extracted
    assert all(lf.vector[_IDX] == 0.0 for lf in result)


def test_function_features_never_carry_parse_error() -> None:
    result = extract_function_features("def f(x):\n    return x + 1\n")
    assert result
    for fn in result:
        assert fn.vector[_IDX] == 0.0
