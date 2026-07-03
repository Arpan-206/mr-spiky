"""Structural feature extraction from Python source using the stdlib `ast` module.

Two entry points:
- `extract_function_features(src)` — one vector per top-level function/method
- `extract_line_features(src)`     — one vector per source line (used by infer)

Feature vector (order is contractual — encode.py and model.py depend on it):
    0. nesting_depth       — max indentation depth at this scope/line
    1. length              — number of AST nodes (function) or tokens (line)
    2. token_entropy       — Shannon entropy over token/name multiset
    3. naming_entropy      — Shannon entropy over identifier char distribution
    4. cyclomatic_proxy    — 1 + count of branching AST nodes in scope
"""

from __future__ import annotations

import ast
import math
import tokenize
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

FEATURE_NAMES: tuple[str, ...] = (
    "nesting_depth",
    "length",
    "token_entropy",
    "naming_entropy",
    "cyclomatic_proxy",
)
NUM_FEATURES = len(FEATURE_NAMES)

# Rough per-feature caps used for min-max normalization. Deliberately loose —
# these are just to squash to [0, 1] for encoding, not to be statistically tight.
_FEATURE_CAPS: dict[str, float] = {
    "nesting_depth": 8.0,
    "length": 200.0,
    "token_entropy": 6.0,
    "naming_entropy": 5.0,
    "cyclomatic_proxy": 20.0,
}

_BRANCHING_NODES: tuple[type, ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.BoolOp,
    ast.IfExp,
    ast.Match,
)


@dataclass
class FunctionFeatures:
    name: str
    lineno: int
    end_lineno: int
    vector: list[float]  # length == NUM_FEATURES


@dataclass
class LineFeatures:
    line: int
    vector: list[float]  # length == NUM_FEATURES


def _shannon(items: Iterable) -> float:
    counts = Counter(items)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _naming_entropy(names: Iterable[str]) -> float:
    chars: list[str] = []
    for n in names:
        chars.extend(n)
    return _shannon(chars)


def _nesting_depth(node: ast.AST) -> int:
    """Deepest nesting of branching/scope-forming nodes inside `node`."""
    best = 0

    def walk(n: ast.AST, depth: int) -> None:
        nonlocal best
        best = max(best, depth)
        inc = 1 if isinstance(n, _BRANCHING_NODES + (ast.FunctionDef, ast.AsyncFunctionDef)) else 0
        for child in ast.iter_child_nodes(n):
            walk(child, depth + inc)

    walk(node, 0)
    return best


def _cyclomatic_proxy(node: ast.AST) -> int:
    count = 1
    for child in ast.walk(node):
        if isinstance(child, _BRANCHING_NODES):
            count += 1
    return count


def _identifiers(node: ast.AST) -> list[str]:
    out: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.arg):
            out.append(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(n.name)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
    return out


def _function_vector(fn: ast.AST) -> list[float]:
    idents = _identifiers(fn)
    length = sum(1 for _ in ast.walk(fn))
    return [
        float(_nesting_depth(fn)),
        float(length),
        _shannon(idents),
        _naming_entropy(idents),
        float(_cyclomatic_proxy(fn)),
    ]


def normalize(vector: list[float]) -> list[float]:
    """Min-max normalize a raw feature vector into [0, 1]."""
    out: list[float] = []
    for name, val in zip(FEATURE_NAMES, vector):
        cap = _FEATURE_CAPS[name]
        v = 0.0 if cap <= 0 else max(0.0, min(1.0, val / cap))
        out.append(v)
    return out


def extract_function_features(src: str) -> list[FunctionFeatures]:
    """Extract one feature vector per top-level function definition (recursive
    into classes, but not into nested defs — those roll up into the parent)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out: list[FunctionFeatures] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(
                    FunctionFeatures(
                        name=child.name,
                        lineno=child.lineno,
                        end_lineno=child.end_lineno or child.lineno,
                        vector=_function_vector(child),
                    )
                )
            elif isinstance(child, ast.ClassDef):
                visit(child)

    visit(tree)
    return out


def extract_line_features(src: str) -> list[LineFeatures]:
    """One feature vector per non-blank source line.

    Line-level features are cheaper approximations of the function-level ones:
    they exist so the API can score arbitrary snippets even when there's no
    enclosing function. If a line sits inside a function we also fold in that
    function's nesting depth and cyclomatic proxy so `if`/`for` bodies inherit
    "risk" from their scope.
    """
    lines = src.splitlines()

    # Map line -> enclosing function's (nesting_depth, cyclomatic_proxy).
    scope: dict[int, tuple[float, float]] = {}
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nd = float(_nesting_depth(node))
                cx = float(_cyclomatic_proxy(node))
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    scope[ln] = (nd, cx)
    except SyntaxError:
        pass

    out: list[LineFeatures] = []
    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        tokens: list[str] = []
        try:
            for tok in tokenize.tokenize(BytesIO(raw.encode("utf-8")).readline):
                if tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING):
                    tokens.append(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            tokens = stripped.split()

        indent = (len(raw) - len(raw.lstrip(" \t"))) // 2
        names = [t for t in tokens if t.isidentifier()]
        nd_scope, cx_scope = scope.get(i, (0.0, 1.0))

        vec = [
            max(float(indent), nd_scope),
            float(len(tokens)),
            _shannon(tokens),
            _naming_entropy(names),
            cx_scope,
        ]
        out.append(LineFeatures(line=i, vector=vec))

    return out
