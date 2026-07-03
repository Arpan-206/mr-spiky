"""Mine senior-annotated suspicion markers from the senior corpus.

Every occurrence of `# noqa`, `# type: ignore`, `# pyright: ignore`,
`# pragma: no cover`, or `# nosec` in a real Python codebase is a senior
developer saying: "yes, this line looks off, and I've deliberately allowed
it." That's the closest signal we have to *real senior judgment applied to
individual lines*.

For a Mr. Spiky pitch that claims to encode senior intuition, this is the
right validation dataset: does Mr. Spiky's per-line score elevate on lines
that seniors themselves marked as exceptions?

We build a labeled dataset:
    label=1  →  a line with a senior-annotation marker
    label=0  →  a randomly-sampled line from the same source, without any marker

Result: data/annotations_labeled.json
    [{"label": 0|1, "code": "<full function containing the line>",
      "line_index": <1-based line in the code>, "marker": "<matched token>",
      "repo": "...", "path": "..."}]

The "code" field contains the enclosing function (not the whole file) so
sequence-mode inference has real context. `line_index` is the position of
the annotated line within that function.
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
import sys
from pathlib import Path

log = logging.getLogger("mrspiky.data.annotations")

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "senior_corpus.json"
MANIFEST_PATH = ROOT / "senior_corpus_manifest.json"
OUT_PATH = ROOT / "annotations_labeled.json"

# Regexes for senior-annotation markers. Matches inline comments only.
_MARKERS: list[tuple[str, re.Pattern]] = [
    ("noqa",         re.compile(r"#\s*noqa\b")),
    ("type_ignore",  re.compile(r"#\s*type:\s*ignore\b")),
    ("pyright",      re.compile(r"#\s*pyright:\s*ignore\b")),
    ("pragma_no_cover", re.compile(r"#\s*pragma:\s*no\s*cover\b")),
    ("nosec",        re.compile(r"#\s*nosec\b")),
    ("mypy_ignore",  re.compile(r"#\s*mypy:\s*ignore\b")),
    ("todo_refactor", re.compile(r"#\s*(TODO|FIXME|XXX).*(refactor|cleanup|simplify|complex|nested)", re.IGNORECASE)),
]


def _find_enclosing_function(tree: ast.AST, line: int) -> ast.AST | None:
    """Deepest FunctionDef/AsyncFunctionDef that spans `line`."""
    best: ast.AST | None = None
    best_span = float("inf")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= line <= end:
            span = end - node.lineno
            if span < best_span:
                best = node
                best_span = span
    return best


def _function_source(source_lines: list[str], fn: ast.AST) -> tuple[str, int]:
    """Return (function source, 1-based line offset of fn.lineno within the source).
    Also normalizes indentation so the function is left-flush and parseable."""
    end = fn.end_lineno or fn.lineno
    fn_lines = source_lines[fn.lineno - 1 : end]
    if not fn_lines:
        return "", 0
    # Dedent
    first = fn_lines[0]
    indent = len(first) - len(first.lstrip())
    dedented = [ln[indent:] if len(ln) >= indent else ln for ln in fn_lines]
    fn_src = "\n".join(dedented) + "\n"
    return fn_src, fn.lineno


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not CORPUS_PATH.exists():
        log.error("senior corpus not found. run download_senior_corpus.py first.")
        return 1
    if not MANIFEST_PATH.exists():
        log.error("senior corpus manifest not found. run download_senior_corpus.py first.")
        return 1

    corpus: list[str] = json.loads(CORPUS_PATH.read_text())
    manifest: list[dict] = json.loads(MANIFEST_PATH.read_text())
    if len(corpus) != len(manifest):
        log.warning("corpus and manifest length mismatch (%d vs %d)", len(corpus), len(manifest))

    positives: list[dict] = []
    negatives: list[dict] = []
    marker_counts: dict[str, int] = {}

    for i, src in enumerate(corpus):
        meta = manifest[i] if i < len(manifest) else {}
        source_lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        annotated_line_nums: list[tuple[int, str]] = []
        for lineno, ln in enumerate(source_lines, start=1):
            for name, pat in _MARKERS:
                if pat.search(ln):
                    annotated_line_nums.append((lineno, name))
                    marker_counts[name] = marker_counts.get(name, 0) + 1
                    break  # one marker per line max

        for lineno, marker in annotated_line_nums:
            fn = _find_enclosing_function(tree, lineno)
            if fn is None:
                continue
            fn_src, fn_start = _function_source(source_lines, fn)
            if not fn_src:
                continue
            positives.append({
                "label": 1,
                "code": fn_src,
                "line_index": lineno - fn_start + 1,
                "marker": marker,
                "repo": meta.get("repo"),
                "path": meta.get("path"),
            })

        # Build negatives: pick random FunctionDef spans from the same source
        # that contain NO annotation markers. We match roughly 1 negative
        # per positive so classes stay balanced.
        clean_functions: list[ast.AST] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end = node.end_lineno or node.lineno
            has_marker = any(node.lineno <= ln <= end for ln, _ in annotated_line_nums)
            if not has_marker:
                clean_functions.append(node)
        if clean_functions and annotated_line_nums:
            for _ in range(len(annotated_line_nums)):
                fn = random.choice(clean_functions)
                fn_src, fn_start = _function_source(source_lines, fn)
                if not fn_src:
                    continue
                # Pick a middle line as the "line of interest" for parity
                # with positives (we still score all lines but this lets us
                # frame it consistently in the labeled block).
                mid_line = 1 + (fn.end_lineno - fn.lineno) // 2
                negatives.append({
                    "label": 0,
                    "code": fn_src,
                    "line_index": mid_line,
                    "marker": None,
                    "repo": meta.get("repo"),
                    "path": meta.get("path"),
                })

    log.info("marker counts: %s", marker_counts)
    log.info("positives: %d  negatives: %d", len(positives), len(negatives))
    if not positives:
        log.error("no annotated lines found — corpus may be missing senior repos")
        return 1

    random.seed(0)
    random.shuffle(positives)
    random.shuffle(negatives)
    balanced_n = min(len(positives), len(negatives))
    combined = positives[:balanced_n] + negatives[:balanced_n]
    random.shuffle(combined)

    OUT_PATH.write_text(json.dumps(combined, indent=2))
    log.info("wrote %d samples (%d positive + %d negative) -> %s",
             len(combined), balanced_n, balanced_n, OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
