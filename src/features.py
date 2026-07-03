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
    5. use_def_distance    — max lines between a variable's def and its use (line-level)
                             or mean use-def gap across the function (fn-level)
    6. name_flow           — distinct identifiers touched on this line / in this fn
    7. call_graph_shape    — outgoing calls in scope, weighted by library-vs-local
    8. exception_density   — try/except/raise count / body_size in scope

Tiers 1's four new features add channels the AST already has but the original
five ignored: data-flow (5), name-flow (6), call graph (7), exceptions (8).
"""

from __future__ import annotations

import ast
import math
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

FEATURE_NAMES: tuple[str, ...] = (
    "nesting_depth",
    "length",
    "token_entropy",
    "naming_entropy",
    "cyclomatic_proxy",
    "use_def_distance",
    "name_flow",
    "call_graph_shape",
    "exception_density",
)
NUM_FEATURES = len(FEATURE_NAMES)

# Human-readable axes for the multi-axis /analyze output. Each axis is the
# weighted sum of a subset of the normalized features, then re-scaled so that
# a value of ~1.0 means "extreme along this axis relative to senior code."
#
# This is what turns "score: 0.73" into "high on complexity + tangled_state,
# low on naming" — the reasoning-style output that shows *why* the SNN
# would flag the line.
AXIS_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "complexity":         ("nesting_depth", "cyclomatic_proxy", "length"),
    "tangled_state":      ("use_def_distance", "name_flow"),
    "hidden_calls":       ("call_graph_shape",),
    "exception_surface":  ("exception_density",),
    "naming":             ("token_entropy", "naming_entropy"),
}
AXIS_NAMES: tuple[str, ...] = tuple(AXIS_DEFINITIONS.keys())

# Per-axis empirical p95 over the senior corpus. Values above ~1.0 after
# dividing by these correspond to "top 5% along this axis" — i.e. genuinely
# extreme relative to how senior code looks. Measured once, hardcoded so
# inference is deterministic.
_AXIS_P95: dict[str, float] = {
    "complexity":         0.50,   # (nesting_depth p95=1.0 + cyclomatic p95=0.85 + length p95=0.08) / 3
    "tangled_state":      0.50,   # (use_def p95=1.0 + name_flow p95=0.75) / 2 — but real p95 is lower
    "hidden_calls":       0.60,   # call_graph p95=1.0, but most lines have moderate call surface
    "exception_surface":  0.10,   # exception_density p95=0.06; multiplied for headroom
    "naming":             0.70,   # (token_entropy p95=0.6 + naming_entropy p95=0.8) / 2
}


def compute_axes(normalized_vector: list[float]) -> dict[str, float]:
    """Map a normalized (already in [0,1]) feature vector to human-readable axes.
    Each axis is the mean of its constituent features, rescaled by the corpus
    p95 so all axes live on the same 0-to-~1 scale where 1.0 = 'extreme for
    senior code.'"""
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out: dict[str, float] = {}
    for axis, members in AXIS_DEFINITIONS.items():
        vals = [normalized_vector[idx[m]] for m in members]
        raw = sum(vals) / len(vals) if vals else 0.0
        rescaled = min(1.0, raw / max(_AXIS_P95[axis], 1e-6))
        out[axis] = round(rescaled, 4)
    return out

# Rough per-feature caps used for min-max normalization. Deliberately loose —
# these are just to squash to [0, 1] for encoding, not to be statistically tight.
_FEATURE_CAPS: dict[str, float] = {
    "nesting_depth": 8.0,
    "length": 200.0,
    "token_entropy": 6.0,
    "naming_entropy": 5.0,
    "cyclomatic_proxy": 20.0,
    "use_def_distance": 30.0,   # 30 lines between def and use is already spooky
    "name_flow": 12.0,          # 12 distinct names touched on one line is dense
    "call_graph_shape": 15.0,   # 15 outgoing calls in scope
    "exception_density": 0.5,   # exception nodes / body_size ratio; capped at 0.5
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

_EXCEPTION_NODES: tuple[type, ...] = (ast.Try, ast.ExceptHandler, ast.Raise)

# Rough heuristic: identifiers that start with these prefixes are "library-ish"
# and cost less than a local call (unknown-implementation risk).
_STDLIB_HINTS = frozenset({
    "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple",
    "map", "filter", "zip", "enumerate", "sorted", "min", "max", "sum", "abs",
    "round", "any", "all", "isinstance", "type", "iter", "next", "open", "input",
})


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


# ---- Tier 1 new features -------------------------------------------------

def _use_def_map(node: ast.AST) -> dict[str, int]:
    """Return {var_name -> lineno of first Store}. Used to compute the gap to
    later Loads of the same name."""
    defs: dict[str, int] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defs.setdefault(n.id, n.lineno)
        elif isinstance(n, ast.arg):
            defs.setdefault(n.arg, n.lineno)
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            defs.setdefault(n.target.id, n.lineno)
    return defs


def _mean_use_def_distance(node: ast.AST) -> float:
    """Function-level: mean (lineno_of_use - lineno_of_first_def) over all
    Load references. Longer chains = more nonlocal action = worth flagging."""
    defs = _use_def_map(node)
    gaps: list[int] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            d = defs.get(n.id)
            if d is not None and n.lineno > d:
                gaps.append(n.lineno - d)
    return sum(gaps) / len(gaps) if gaps else 0.0


def _line_use_def_gap(line_no: int, loads_on_line: Iterable[str], defs: dict[str, int]) -> float:
    """Line-level: max gap between this line and the first def of any Load on
    this line. Captures "line 42 reads x defined on line 5" spookiness."""
    best = 0.0
    for name in loads_on_line:
        d = defs.get(name)
        if d is not None and line_no > d:
            best = max(best, float(line_no - d))
    return best


def _call_graph_shape(node: ast.AST) -> float:
    """Weighted count of outgoing calls: local calls (unresolved names) count
    2x, library-hint calls count 0.5x. Higher = more delegation to unknowns."""
    total = 0.0
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                total += 0.5 if func.id in _STDLIB_HINTS else 2.0
            elif isinstance(func, ast.Attribute):
                total += 1.0  # method call, medium risk
            else:
                total += 1.5  # weirder call shape
    return total


def _exception_density(node: ast.AST) -> float:
    body_size = max(1, sum(1 for _ in ast.walk(node)))
    ex = sum(1 for n in ast.walk(node) if isinstance(n, _EXCEPTION_NODES))
    return ex / body_size


def _name_flow(node_or_names: ast.AST | Iterable[str]) -> float:
    """Count of distinct identifiers touched. AST node → walk; iterable → set."""
    if isinstance(node_or_names, ast.AST):
        return float(len({n for n in _identifiers(node_or_names)}))
    return float(len(set(node_or_names)))


# ---- Vector assembly -----------------------------------------------------

def _function_vector(fn: ast.AST) -> list[float]:
    idents = _identifiers(fn)
    length = sum(1 for _ in ast.walk(fn))
    return [
        float(_nesting_depth(fn)),
        float(length),
        _shannon(idents),
        _naming_entropy(idents),
        float(_cyclomatic_proxy(fn)),
        _mean_use_def_distance(fn),
        _name_flow(fn),
        _call_graph_shape(fn),
        _exception_density(fn),
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


def _skip_lines(tree: ast.AST) -> set[int]:
    """Line numbers that shouldn't be scored: docstrings + top-level imports.

    Docstrings look like prose (high naming entropy, many distinct tokens) so
    they saturate our features but a reviewer would never flag them. Imports
    are structurally trivial and don't carry the kind of complexity signal
    Mr. Spiky is trying to detect. Skipping both is what a senior reviewer
    would do implicitly when reading a file.
    """
    skip: set[int] = set()

    def _collect_docstring(scope_body: list[ast.stmt]) -> None:
        if scope_body and isinstance(scope_body[0], ast.Expr):
            e = scope_body[0].value
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                for ln in range(e.lineno, (e.end_lineno or e.lineno) + 1):
                    skip.add(ln)

    # Module-level docstring
    if isinstance(tree, ast.Module):
        _collect_docstring(tree.body)

    for node in ast.walk(tree):
        # Class/function docstrings
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect_docstring(node.body)
        # Import statements at any scope
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                skip.add(ln)

    return skip


def _line_scoped_stats(tree: ast.AST) -> tuple[
    dict[int, tuple[float, float, float, float]],  # (nd, cx, call_shape, exc_dens) per line
    dict[int, dict[str, int]],                      # per-function line-set → local use-def defs
    dict[int, int],                                 # line -> function's start line (for def lookup)
]:
    """Precompute per-function scope stats, mapped from line number to values.

    Returns:
      scope_stats[line] = (nesting_depth, cyclomatic, call_shape, exception_density)
      fn_defs[fn_start_line] = {var: first_def_line} (for use-def gap computation)
      line_to_fn[line] = fn_start_line
    """
    scope_stats: dict[int, tuple[float, float, float, float]] = {}
    fn_defs: dict[int, dict[str, int]] = {}
    line_to_fn: dict[int, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nd = float(_nesting_depth(node))
            cx = float(_cyclomatic_proxy(node))
            cs = _call_graph_shape(node)
            ed = _exception_density(node)
            defs = _use_def_map(node)
            fn_start = node.lineno
            fn_defs[fn_start] = defs
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                scope_stats[ln] = (nd, cx, cs, ed)
                line_to_fn[ln] = fn_start

    return scope_stats, fn_defs, line_to_fn


def extract_line_features(src: str) -> list[LineFeatures]:
    """One feature vector per non-blank source line.

    Line-level features are cheaper approximations of the function-level ones:
    they exist so the API can score arbitrary snippets even when there's no
    enclosing function. If a line sits inside a function we also fold in that
    function's nesting depth and cyclomatic proxy so `if`/`for` bodies inherit
    "risk" from their scope.
    """
    lines = src.splitlines()

    scope_stats: dict[int, tuple[float, float, float, float]] = {}
    fn_defs: dict[int, dict[str, int]] = {}
    line_to_fn: dict[int, int] = {}
    skip_lines: set[int] = set()
    try:
        tree = ast.parse(src)
        scope_stats, fn_defs, line_to_fn = _line_scoped_stats(tree)
        skip_lines = _skip_lines(tree)
    except SyntaxError:
        pass

    out: list[LineFeatures] = []
    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if i in skip_lines:
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
        nd_scope, cx_scope, cs_scope, ed_scope = scope_stats.get(i, (0.0, 1.0, 0.0, 0.0))
        fn_start = line_to_fn.get(i)
        defs = fn_defs.get(fn_start, {}) if fn_start is not None else {}
        use_def_gap = _line_use_def_gap(i, names, defs)

        vec = [
            max(float(indent), nd_scope),
            float(len(tokens)),
            _shannon(tokens),
            _naming_entropy(names),
            cx_scope,
            use_def_gap,
            _name_flow(names),
            cs_scope,
            ed_scope,
        ]
        out.append(LineFeatures(line=i, vector=vec))

    return out
