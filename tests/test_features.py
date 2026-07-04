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
    # Bumped from 10 → 13 when cross-function features (global_reach,
    # attr_reach, call_graph_depth) landed. Test is a wire-check: fails
    # loudly whenever the schema shape changes so downstream trainers /
    # calibrators know to rebuild.
    assert NUM_FEATURES == 13
    assert "parse_error" in FEATURE_NAMES
    assert "global_reach" in FEATURE_NAMES
    assert "attr_reach" in FEATURE_NAMES
    assert "call_graph_depth" in FEATURE_NAMES


def test_cross_function_features_fire_on_real_reach() -> None:
    """Synthetic test: verify the three new features actually activate on
    code that has cross-function reach (global refs, self.attr reach,
    call chains). Prevents silent regressions if the AST walk changes."""
    src = (
        "GLOBAL_STATE = {}\n"
        "\n"
        "def helper_a(x):\n"
        "    return helper_b(x)\n"
        "\n"
        "def helper_b(x):\n"
        "    return x * 2\n"
        "\n"
        "class Config:\n"
        "    def __init__(self):\n"
        "        self.value = None\n"
        "\n"
        "    def load(self, key):\n"
        "        return helper_a(self.value or GLOBAL_STATE.get(key))\n"
    )
    result = extract_line_features(src)
    by_line = {lf.line: dict(zip(FEATURE_NAMES, lf.vector)) for lf in result}
    # The call inside `load` reaches back to GLOBAL_STATE (line 1),
    # self.value (line 11), and calls helper_a which chains to helper_b.
    load_line = by_line[14]
    assert load_line["global_reach"] > 0.0, "should reach back to GLOBAL_STATE"
    assert load_line["attr_reach"] > 0.0, "should reach back to self.value"
    assert load_line["call_graph_depth"] >= 1.0, "helper_a → helper_b is depth 1+"


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
