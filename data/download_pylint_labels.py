"""Mine cross-function-tangled labels from the senior corpus via pylint.

Our existing annotations set (`# noqa` markers) captures per-line senior
judgments — but those markers are almost always on single expressions with
local reasoning. They don't measure "this function is tangled across scope,"
which is exactly what the 13-dim cross-function features are trying to
detect. Without a labeled set that exercises that signal, we can't measure
whether the new features are useful.

pylint's "refactoring" family (`too-many-branches`, `too-many-locals`,
`too-many-instance-attributes`, `too-many-statements`, `too-many-return-
statements`, `too-many-nested-blocks`, `too-complex`) tracks a different
notion of complexity from ours — thresholds are ad-hoc heuristics developed
over a decade of Python style debates, not spiking-neural-network features.
Using pylint as a *second opinion* gives us a labeled set that isn't just
re-encoding what we already measure.

Positives: functions in the senior corpus that trip any of the target
pylint checks. Negatives: random functions from the same files that don't
trip any check.

Writes: `data/pylint_labeled.json` — list of `{label, code, function_name,
pylint_checks, source_path}`.
"""

from __future__ import annotations

import ast
import json
import logging
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("mrspiky.data.pylint")

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "senior_corpus.json"
MANIFEST_PATH = ROOT / "senior_corpus_manifest.json"
OUT_PATH = ROOT / "pylint_labeled.json"

# pylint checks that flag cross-scope tangled state. `too-many-locals` and
# `too-many-branches` are on by default (message IDs R0914 / R0912); the
# rest need to be enabled explicitly. We skip style/convention (C-codes)
# and error (E-codes) — those don't correlate with our features.
_TARGET_CHECKS = {
    "too-many-branches",              # R0912
    "too-many-locals",                # R0914
    "too-many-statements",            # R0915
    "too-many-return-statements",     # R0911
    "too-many-nested-blocks",         # R1702
    "too-many-instance-attributes",   # R0902
    "too-many-arguments",             # R0913
    "too-complex",                    # C0901 (McCabe cyclomatic)
    "cyclic-import",                  # R0401 — cross-module tangled
}

# Cap so we don't over-count huge files; also keeps runtime reasonable.
_MAX_POSITIVES_PER_FILE = 3
_TARGET_POSITIVE_COUNT = 300  # ~matches our annotations set size for direct comparison


def _load_corpus() -> list[tuple[str, str]]:
    """Return [(path, source), ...] for each file in the corpus."""
    if not CORPUS_PATH.exists():
        raise SystemExit(f"missing {CORPUS_PATH}; run data-pretrain first")
    corpus = json.loads(CORPUS_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    out: list[tuple[str, str]] = []
    for i, src in enumerate(corpus):
        path = manifest[i]["path"] if i < len(manifest) else f"corpus_{i}.py"
        out.append((path, src))
    return out


def _run_pylint(src: str) -> list[dict]:
    """Run pylint on a temp file, return the parsed JSON messages. pylint
    exits with a non-zero code if it finds issues — that's expected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp_path = f.name
    try:
        # --enable=all makes sure the "too-many-*" checks are on (they're
        # in the R category which is on by default, but some pylint configs
        # disable them). --disable=all first + only re-enable ours = fastest.
        result = subprocess.run(
            [
                "pylint",
                "--output-format=json",
                "--disable=all",
                "--enable=" + ",".join(_TARGET_CHECKS),
                "--load-plugins=pylint.extensions.mccabe",  # for too-complex
                "--max-complexity=10",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # pylint prints JSON to stdout even on non-zero exit code (which
        # just signals "issues found"). Parse defensively.
        if not result.stdout.strip():
            return []
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
    except subprocess.TimeoutExpired:
        return []
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _function_at(tree: ast.AST, line: int) -> ast.AST | None:
    """Return the innermost FunctionDef/AsyncFunctionDef containing `line`."""
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


def _extract_function_source(src: str, fn: ast.AST) -> str:
    """Dedent and return the function's source. Same idea as
    mine_annotations.py: normalize the indentation so the snippet is
    self-contained and parseable in isolation."""
    src_lines = src.splitlines()
    start = fn.lineno - 1
    end = fn.end_lineno or fn.lineno
    fn_lines = src_lines[start:end]
    if not fn_lines:
        return ""
    first = fn_lines[0]
    indent = len(first) - len(first.lstrip())
    dedented = [ln[indent:] if len(ln) >= indent else ln for ln in fn_lines]
    return "\n".join(dedented) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    corpus = _load_corpus()
    log.info("scanning %d files with pylint...", len(corpus))

    positives: list[dict] = []
    check_counts: dict[str, int] = defaultdict(int)

    # Collect all flagged functions across the corpus first, then sample.
    for i, (path, src) in enumerate(corpus):
        if len(positives) >= _TARGET_POSITIVE_COUNT * 2:  # headroom for sampling
            break
        messages = _run_pylint(src)
        if not messages:
            continue

        # Group messages by their line's enclosing function; we want to
        # count all target-checks-per-function so the pylint_checks field
        # in the output reflects why we flagged it.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        by_fn: dict[int, list[str]] = defaultdict(list)
        for msg in messages:
            check = msg.get("symbol") or msg.get("message-id", "unknown")
            if check not in _TARGET_CHECKS:
                continue
            line = msg.get("line", 0)
            fn = _function_at(tree, line)
            if fn is None:
                continue
            check_counts[check] += 1
            by_fn[fn.lineno].append(check)

        # Emit one positive per flagged function, up to per-file cap.
        emitted = 0
        for fn_start, checks in by_fn.items():
            if emitted >= _MAX_POSITIVES_PER_FILE:
                break
            fn = _function_at(tree, fn_start)
            if fn is None:
                continue
            code = _extract_function_source(src, fn)
            if not code.strip():
                continue
            positives.append({
                "label": 1,
                "code": code,
                "function_name": fn.name,
                "pylint_checks": sorted(set(checks)),
                "source_path": path,
            })
            emitted += 1

        if (i + 1) % 25 == 0:
            log.info("  scanned %d/%d files, %d positives so far", i + 1, len(corpus), len(positives))

    log.info("positive check distribution: %s", dict(check_counts))
    log.info("collected %d positive samples", len(positives))

    # Sample negatives: random functions from the corpus that pylint DIDN'T
    # flag on any target check. We already have the pylint output cached in
    # the loop above but not saved — cheaper to just re-collect the clean
    # functions from AST here, since we don't care which pylint checks
    # DIDN'T fire.
    negatives: list[dict] = []
    flagged_key = {(p["source_path"], p["function_name"]) for p in positives}
    random.seed(0)
    corpus_shuf = list(corpus)
    random.shuffle(corpus_shuf)
    target_negatives = min(len(positives), _TARGET_POSITIVE_COUNT)

    for path, src in corpus_shuf:
        if len(negatives) >= target_negatives:
            break
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (path, node.name) in flagged_key:
                continue  # skip functions flagged by pylint
            code = _extract_function_source(src, node)
            if not code.strip():
                continue
            # Skip trivially small functions — they're neither tangled nor
            # a useful contrast (they'd all score ~0 on every feature).
            if code.count("\n") < 3:
                continue
            negatives.append({
                "label": 0,
                "code": code,
                "function_name": node.name,
                "pylint_checks": [],
                "source_path": path,
            })
            if len(negatives) >= target_negatives:
                break

    log.info("collected %d negative samples", len(negatives))

    # Balance and shuffle.
    n = min(len(positives), len(negatives))
    positives = positives[:n]
    negatives = negatives[:n]
    combined = positives + negatives
    random.shuffle(combined)

    OUT_PATH.write_text(json.dumps(combined, indent=2))
    log.info("wrote %d samples (%d positive + %d negative) -> %s",
             len(combined), n, n, OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
