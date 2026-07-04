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
    9. parse_error         — 1.0 on the line where ast.parse choked, else 0.0

Tiers 1's four new features add channels the AST already has but the original
five ignored: data-flow (5), name-flow (6), call graph (7), exceptions (8).

Tier 2 adds parse_error (9): senior code always parses, so this is a constant
zero across the entire training corpus — it's inert during STDP pretraining
and whitening, but gives the SNN a dedicated, previously-nonexistent channel
to react to at inference when code doesn't even compile (e.g. `if x = True:`,
a typo for `==`). See CLAUDE.md for why richer features (not threshold
tuning) are the intended way to close detection gaps.
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
    "parse_error",
    # Cross-function data-flow features (added after 10-dim baseline). These
    # capture reach that `use_def_distance` (function-local) misses:
    "global_reach",       # max line distance to a module-level def a line references
    "attr_reach",         # max line distance to a self.attr def in the same class
    "call_graph_depth",   # depth of transitive same-file call chain reachable from a line's calls
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
    # tangled_state now also covers *cross-function* state reach — a line
    # that reads a module-level global or a self.attr defined far away is
    # tangled in the way this axis is trying to name, even if the local
    # use_def_distance is small.
    "tangled_state":      ("use_def_distance", "name_flow", "global_reach", "attr_reach"),
    # hidden_calls now includes the transitive depth of same-file calls
    # this line reaches — one function call that recurses two levels down
    # is more "hidden" than a single library call.
    "hidden_calls":       ("call_graph_shape", "call_graph_depth"),
    "exception_surface":  ("exception_density",),
    "naming":             ("token_entropy", "naming_entropy"),
    "malformed":          ("parse_error",),
}
AXIS_NAMES: tuple[str, ...] = tuple(AXIS_DEFINITIONS.keys())

# Per-axis empirical p95 of the RAW (pre-rescale) axis value over the current
# senior corpus. Rescaling by these means a value of ~1.0 corresponds to
# "top 5% along this axis" — the axes are then comparable across dimensions.
# Recomputed whenever features.py changes: see `_measure_axis_p95` in this
# module (run manually against data/senior_corpus.json).
_AXIS_P95: dict[str, float] = {
    "complexity":         0.37,   # was 0.50; features got more sparse per-line
    "tangled_state":      0.63,   # was 0.50; use_def+name_flow p95 higher than assumed
    "hidden_calls":       0.50,   # was 0.60; per-line call weight is smaller
    "exception_surface":  0.06,   # was 0.10; exception_density is naturally sparse
    "naming":             0.68,   # was 0.70; small correction
    # parse_error is 0.0 for every senior-corpus example (it always parses),
    # so there's no real p95 to measure. Hardcode a small constant so any
    # actual occurrence (1.0) saturates the axis to 1.0.
    "malformed":          0.05,
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
    "cyclomatic_proxy": 3.0,    # LOCAL: branching nodes starting on this line; usually 0 or 1
    "use_def_distance": 30.0,   # 30 lines between def and use is already spooky
    "name_flow": 12.0,          # 12 distinct names touched on one line is dense
    "call_graph_shape": 4.0,    # LOCAL: call weight on this line; multi-call lines are gnarly
    "exception_density": 0.5,   # exception nodes / body_size ratio; capped at 0.5
    "parse_error": 1.0,         # already 0/1, cap is a no-op
    "global_reach": 100.0,      # a line reading a global defined 100 lines away is very tangled
    "attr_reach": 80.0,         # self.attr defined 80 lines away in the class body is similarly tangled
    "call_graph_depth": 4.0,    # depth-4 call chain within the same file is the ceiling we count
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


def _node_label(node: ast.AST) -> str | None:
    """Short human-readable label for an AST node — used in lineage strings
    that show up in review comments. Returns None for nodes we don't want
    to advertise (e.g. Module, plain Name loads).

    The labels read like a reviewer describing structure:
        for i in range(x) at L12
        if x > 0 at L14
        try at L10
        function parse_config at L5
    """
    if isinstance(node, ast.FunctionDef):
        return f"function `{node.name}`"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async function `{node.name}`"
    if isinstance(node, ast.ClassDef):
        return f"class `{node.name}`"
    if isinstance(node, ast.For):
        try:
            tgt = ast.unparse(node.target)
            it = ast.unparse(node.iter)
            return f"`for {tgt} in {it}`"
        except Exception:  # noqa: BLE001
            return "`for` loop"
    if isinstance(node, ast.AsyncFor):
        return "`async for` loop"
    if isinstance(node, ast.While):
        try:
            return f"`while {ast.unparse(node.test)}`"
        except Exception:  # noqa: BLE001
            return "`while` loop"
    if isinstance(node, ast.If):
        try:
            return f"`if {ast.unparse(node.test)}`"
        except Exception:  # noqa: BLE001
            return "`if` block"
    if isinstance(node, ast.Try):
        return "`try` block"
    if isinstance(node, ast.ExceptHandler):
        try:
            return f"`except {ast.unparse(node.type)}`" if node.type else "`except` handler"
        except Exception:  # noqa: BLE001
            return "`except` handler"
    if isinstance(node, ast.With):
        return "`with` block"
    if isinstance(node, ast.AsyncWith):
        return "`async with` block"
    if isinstance(node, ast.Match):
        return "`match` statement"
    return None


_LINEAGE_NODE_TYPES: tuple[type, ...] = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.For, ast.AsyncFor, ast.While, ast.If,
    ast.Try, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Match,
)


def line_lineage(src: str, line: int, max_depth: int = 3) -> list[dict[str, object]]:
    """Return the innermost `max_depth` AST-node ancestors that contain `line`.

    Each entry is `{"kind": "<node class name>", "label": "<human string>",
    "line": <starting line of the node>}`, ordered innermost-first. Used by
    the review pipeline to build reason strings like `"nested for at L47
    inside if at L45 inside function parse_config at L38"`.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    hits: list[tuple[int, ast.AST]] = []  # (start_line, node)
    for node in ast.walk(tree):
        if not isinstance(node, _LINEAGE_NODE_TYPES):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        if start is None:
            continue
        if start <= line <= end:
            hits.append((start, node))

    # Sort by start line descending — deepest (highest lineno within `line`) first.
    hits.sort(key=lambda p: -p[0])
    out: list[dict[str, object]] = []
    seen: set[int] = set()
    for start, node in hits:
        label = _node_label(node)
        if label is None:
            continue
        # Skip functions/classes that already appear (rare) or exact-line dupes.
        if start in seen:
            continue
        seen.add(start)
        out.append({
            "kind": type(node).__name__,
            "label": label,
            "line": start,
        })
        if len(out) >= max_depth:
            break
    return out


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


# ---- Cross-function reach helpers ------------------------------------------
# `use_def_distance` only sees state defined inside the enclosing function.
# In real code, tangled state usually crosses function boundaries: a module-
# level global that's mutated in one place and read in another, or a
# `self.attr` set in `__init__` and read three methods later. These helpers
# build per-file lookups so lines can score by how far they reach across the
# file to touch state defined elsewhere.


def _global_def_map(tree: ast.AST) -> dict[str, int]:
    """Return {name -> lineno} for module-level (top-of-file) assignments.
    Only stmts directly under the Module count — assignments inside a class
    or function belong to their scope, not to the global namespace."""
    defs: dict[str, int] = {}
    if not isinstance(tree, ast.Module):
        return defs
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    defs.setdefault(target.id, target.lineno)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            defs.setdefault(stmt.target.id, stmt.target.lineno)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Top-level `def`/`class` bind a name too — a line that references
            # a helper defined at the top of the file "reaches" back to that
            # def, and this is a common tangled-state pattern (mutually-
            # recursive helpers, module-level state machines).
            defs.setdefault(stmt.name, stmt.lineno)
    return defs


def _attr_def_map(tree: ast.AST) -> dict[str, dict[str, int]]:
    """Return {class_name -> {attr_name -> first_lineno_of_self.attr=<...>}}.
    We only track assignments to `self.<attr>` inside a class's methods —
    that's the pattern that produces attribute reach at the class level."""
    per_class: dict[str, dict[str, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: dict[str, int] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        attrs.setdefault(target.attr, target.lineno)
            elif isinstance(sub, ast.AnnAssign):
                t = sub.target
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    attrs.setdefault(t.attr, t.lineno)
        if attrs:
            per_class[node.name] = attrs
    return per_class


def _line_to_class_map(tree: ast.AST) -> dict[int, str]:
    """{lineno -> enclosing class name}. A method inside class Foo scoped
    self.bar assignments live in Foo's namespace, so we need to know which
    class a line is inside to look up the right attr map."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = node.end_lineno or node.lineno
            for ln in range(node.lineno, end + 1):
                out[ln] = node.name
    return out


def _call_graph(tree: ast.AST) -> dict[str, set[str]]:
    """Return {caller_fn_name -> set(callee_fn_names_defined_in_same_file)}.
    Only edges to functions that are themselves defined in this file — a
    call to `open()` or `os.getenv(...)` doesn't contribute (that's what
    `call_graph_shape` already measures).

    We treat methods as if they lived in a flat namespace: `self.foo()` maps
    to `foo` if `foo` is defined anywhere in the file. Not perfect (name
    collisions across classes get merged), but good enough for the
    approximation this feature is going for."""
    # Collect every function/method name defined in the file.
    all_fn_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_fn_names.add(node.name)

    graph: dict[str, set[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        callees: set[str] = set()
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            called: str | None = None
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            if called and called in all_fn_names and called != fn.name:
                # Skip self-recursion for the graph — it's a call but not
                # a step deeper into a different function's code.
                callees.add(called)
        graph[fn.name] = callees
    return graph


def _call_depths(graph: dict[str, set[str]], max_depth: int = 4) -> dict[str, int]:
    """For each function, return the depth of the deepest same-file call chain
    reachable from its body. Cycle-safe via a per-source-node visited set.
    Bounded at `max_depth` — anything deeper is just "very deep."""
    memo: dict[str, int] = {}

    def depth_from(root: str) -> int:
        # Per-root visited set — same node visited twice from the same root
        # would be a cycle we don't want to follow further.
        stack: list[tuple[str, int]] = [(root, 0)]
        visited: set[str] = {root}
        best = 0
        while stack:
            node, d = stack.pop()
            if d > best:
                best = d
            if d >= max_depth:
                continue
            for nxt in graph.get(node, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append((nxt, d + 1))
        return best

    for fn_name in graph:
        memo[fn_name] = depth_from(fn_name)
    return memo


def _line_global_reach(
    line_no: int,
    loads_on_line: Iterable[str],
    globals_map: dict[str, int],
) -> float:
    """Max distance to a module-level def of any name referenced on this line."""
    best = 0.0
    for name in loads_on_line:
        d = globals_map.get(name)
        if d is not None and line_no > d:
            best = max(best, float(line_no - d))
    return best


def _line_attr_reach(
    line_no: int,
    attrs_referenced: Iterable[str],
    class_attrs: dict[str, int],
) -> float:
    """Max distance to a self.attr assignment of any attribute referenced on
    this line. Class scope only — attrs_referenced should be pre-filtered to
    the attr names on this line (see the tokenize path in extract_line_features)."""
    best = 0.0
    for attr in attrs_referenced:
        d = class_attrs.get(attr)
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
    """Function-level 13-dim vector.

    The three cross-function features (global_reach, attr_reach,
    call_graph_depth) can only be computed with access to the *whole file*,
    which this helper doesn't have. When called from `extract_function_features`
    without that context, they're set to 0.0 — this path is used for STDP
    training vectors, where the loss of these signals is a known tradeoff
    (we're already scoring the function in isolation for training purposes).
    Line-level scoring at inference time has full-file context and gets
    the real values.
    """
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
        0.0,  # parse_error: a function only gets a vector if it parsed at all
        0.0,  # global_reach: not computable in function-only scope
        0.0,  # attr_reach: same
        0.0,  # call_graph_depth: same — needs the whole file's call graph
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
    dict[int, tuple[float, float]],                 # (nesting_depth, exception_density) per line — scope-level
    dict[int, dict[str, int]],                      # per-function → local use-def defs
    dict[int, int],                                 # line -> function's start line
    dict[int, float],                               # per-line LOCAL cyclomatic (branch nodes starting on this line)
    dict[int, float],                               # per-line LOCAL call_graph (call nodes starting on this line)
    dict[int, str],                                 # line -> enclosing function's name (for call-graph depth lookup)
]:
    """Precompute stats indexed by line number.

    Historically we copied the enclosing function's *total* cyclomatic and
    call-graph counts to every line inside it — meaning a trivial `except:`
    inside a big function scored the same as the function's most complex
    line. That inflated every line and produced false positives on plain
    code sitting inside a gnarly function.

    Now we compute those two features *locally* — a line's cyclomatic
    contribution is the number of branching AST nodes whose `lineno` equals
    that line; same for calls. Nesting depth and exception density stay
    scope-level (they're inherently "context" signals).

    Returns:
      scope_stats[line] = (nesting_depth, exception_density) — scope-level context
      fn_defs[fn_start]  = {var: first_def_line}              — for use-def gap
      line_to_fn[line]   = fn_start                           — function lookup
      line_cx[line]      = local branching count              — per-line
      line_cg[line]      = local call weight                  — per-line
    """
    scope_stats: dict[int, tuple[float, float]] = {}
    fn_defs: dict[int, dict[str, int]] = {}
    line_to_fn: dict[int, int] = {}
    line_to_fn_name: dict[int, str] = {}

    # Per-line local counts: walk the whole tree once and bucket by lineno.
    line_cx: dict[int, float] = {}
    line_cg: dict[int, float] = {}
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        if isinstance(node, _BRANCHING_NODES):
            line_cx[lineno] = line_cx.get(lineno, 0.0) + 1.0
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                weight = 0.5 if func.id in _STDLIB_HINTS else 2.0
            elif isinstance(func, ast.Attribute):
                weight = 1.0
            else:
                weight = 1.5
            line_cg[lineno] = line_cg.get(lineno, 0.0) + weight

    # Function-scope context: nesting depth + exception density, still
    # inherited by all lines inside the function.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nd = float(_nesting_depth(node))
            ed = _exception_density(node)
            defs = _use_def_map(node)
            fn_start = node.lineno
            fn_defs[fn_start] = defs
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                scope_stats[ln] = (nd, ed)
                line_to_fn[ln] = fn_start
                line_to_fn_name[ln] = node.name

    return scope_stats, fn_defs, line_to_fn, line_cx, line_cg, line_to_fn_name


def _line_self_attrs(tree: ast.AST) -> dict[int, set[str]]:
    """{lineno -> set(self.attr names LOADED on this line)}. Only Load
    context — a `self.foo = ...` on the line is a def, which is what
    _attr_def_map already picked up, not a reference back to it."""
    out: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            v = node.value
            if isinstance(v, ast.Name) and v.id == "self":
                out.setdefault(node.lineno, set()).add(node.attr)
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

    scope_stats: dict[int, tuple[float, float]] = {}
    fn_defs: dict[int, dict[str, int]] = {}
    line_to_fn: dict[int, int] = {}
    line_cx: dict[int, float] = {}
    line_cg: dict[int, float] = {}
    line_to_fn_name: dict[int, str] = {}
    skip_lines: set[int] = set()
    syntax_error_line: int | None = None
    globals_map: dict[str, int] = {}
    attr_map: dict[str, dict[str, int]] = {}
    line_to_class: dict[int, str] = {}
    call_depths: dict[str, int] = {}
    line_self_attrs: dict[int, set[str]] = {}
    try:
        tree = ast.parse(src)
        (scope_stats, fn_defs, line_to_fn, line_cx, line_cg,
         line_to_fn_name) = _line_scoped_stats(tree)
        skip_lines = _skip_lines(tree)
        # Cross-function reach lookups. Compute once per file — these are
        # module-scoped so it would be wasteful to recompute per line.
        globals_map = _global_def_map(tree)
        attr_map = _attr_def_map(tree)
        line_to_class = _line_to_class_map(tree)
        call_depths = _call_depths(_call_graph(tree))
        line_self_attrs = _line_self_attrs(tree)
    except SyntaxError as e:
        if lines:
            syntax_error_line = max(1, min(e.lineno or 1, len(lines)))

    out: list[LineFeatures] = []
    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        is_error_line = i == syntax_error_line
        if not is_error_line:
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
        nd_scope, ed_scope = scope_stats.get(i, (0.0, 0.0))
        fn_start = line_to_fn.get(i)
        defs = fn_defs.get(fn_start, {}) if fn_start is not None else {}
        use_def_gap = _line_use_def_gap(i, names, defs)

        # Cross-function reach:
        # - global_reach: how far back does this line reach into module-level state?
        # - attr_reach: for methods, how far back into the class's self.attr defs?
        # - call_graph_depth: for the enclosing function, how deep does its
        #   same-file call chain go? A cheap proxy for "how much of the file's
        #   logic does this line potentially execute."
        global_reach = _line_global_reach(i, names, globals_map)
        class_name = line_to_class.get(i)
        attrs_here = line_self_attrs.get(i, set())
        attr_reach = 0.0
        if class_name and attrs_here:
            attr_reach = _line_attr_reach(i, attrs_here, attr_map.get(class_name, {}))
        fn_name_here = line_to_fn_name.get(i)
        call_depth = float(call_depths.get(fn_name_here, 0)) if fn_name_here else 0.0

        # nesting_depth = 70% local (this line's own indentation) + 30%
        # scope (enclosing function's max nesting). Local dominates so a
        # trivial `return x` at column 0 of a nested function doesn't
        # inherit the function's deepest branch as its own score, but scope
        # still contributes so lines *inside* a gnarly function score a
        # bit higher than the same line in isolation.
        local_nd = float(indent)
        blended_nd = 0.7 * local_nd + 0.3 * nd_scope

        vec = [
            blended_nd,
            float(len(tokens)),
            _shannon(tokens),
            _naming_entropy(names),
            line_cx.get(i, 0.0),           # LOCAL cyclomatic count
            use_def_gap,
            _name_flow(names),
            line_cg.get(i, 0.0),           # LOCAL call-graph weight
            ed_scope,
            1.0 if is_error_line else 0.0,
            global_reach,                  # NEW: cross-function reach to module state
            attr_reach,                    # NEW: cross-function reach to class self.attrs
            call_depth,                    # NEW: depth of same-file call chain from enclosing fn
        ]
        out.append(LineFeatures(line=i, vector=vec))

    return out
