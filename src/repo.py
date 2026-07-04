"""Whole-repo scoring — score every Python file under a directory, aggregate
the top-N flagged lines across the entire tree.

This is the "run Mr. Spiky on my whole codebase, tell me the 20 gnarliest
lines" surface. It's a thin batching layer over `analyze()` — no new
scoring logic, just walking + aggregation + a tidy summary.

Usage:
    python -m src.repo <root> [--top-n 20] [--min-score 0.9] [--format json|human]

Or via the justfile:
    just repo-review <root>
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .infer import analyze

log = logging.getLogger("mrspiky.repo")

# Directories we always skip. A repo could add its own via .gitignore-style
# rules later — these are just the common cases where scoring is guaranteed
# to be noise (vendored deps, caches, generated output).
_SKIP_DIR_PATTERNS = (
    ".git", ".venv", "venv", ".env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build", ".idea",
    ".vscode", "site-packages", ".tox", ".nox",
)
_SKIP_FILE_PATTERNS = (
    "setup.py",
    "conftest.py",
    "*_pb2.py", "*_pb2_grpc.py",  # generated protobuf
)


@dataclass
class FlaggedEntry:
    """One aggregated finding across the repo. `path` is relative to the
    scanned root — the frontend needs stable paths, not absolute."""
    path: str
    line: int
    score: float
    reason: str
    axes: dict[str, float]
    function: str | None
    function_score: float | None


def _should_skip_dir(path: Path) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pat) for pat in _SKIP_DIR_PATTERNS)


def _should_skip_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".py"):
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in _SKIP_FILE_PATTERNS)


def iter_python_files(root: Path) -> Iterable[Path]:
    """Depth-first walk yielding scoreable .py files. Prunes at any dir
    matching _SKIP_DIR_PATTERNS so we don't recurse into .venv, etc."""
    if root.is_file() and root.suffix == ".py":
        yield root
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if _should_skip_dir(entry):
                continue
            yield from iter_python_files(entry)
        elif entry.is_file() and not _should_skip_file(entry):
            yield entry


def score_repo(
    root: Path,
    min_score: float = 0.9,
    top_n: int = 20,
) -> dict[str, Any]:
    """Walk `root`, analyze every .py file, return aggregated top-N flagged
    lines across the whole tree.

    Files that fail to read, parse, or analyze are skipped with a warning —
    one broken file shouldn't kill the whole scan.
    """
    all_flagged: list[FlaggedEntry] = []
    per_file_totals: dict[str, dict[str, Any]] = {}
    axis_totals: dict[str, float] = {}
    axis_counts: dict[str, int] = {}
    scanned = 0
    failed: list[str] = []

    files = list(iter_python_files(root))
    log.info("scanning %d Python files under %s", len(files), root)

    for py in files:
        try:
            code = py.read_text(errors="replace")
        except OSError as e:
            failed.append(f"{py}: {e}")
            continue
        try:
            result = analyze(code)
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't kill the scan
            failed.append(f"{py}: analyze failed ({type(e).__name__}: {e})")
            continue

        rel = str(py.relative_to(root)) if py.is_relative_to(root) else str(py)
        n_scored = len(result.get("lines", []))
        flagged_in_file = 0
        top_score_in_file = 0.0
        scanned += 1

        for entry in result.get("lines", []):
            score = entry.get("score", 0.0)
            top_score_in_file = max(top_score_in_file, score)
            if not entry.get("flag") or score < min_score:
                continue
            flagged_in_file += 1
            axes = entry.get("axes", {})
            for a, v in axes.items():
                axis_totals[a] = axis_totals.get(a, 0.0) + v
                axis_counts[a] = axis_counts.get(a, 0) + 1
            ctx = entry.get("context") or {}
            all_flagged.append(FlaggedEntry(
                path=rel,
                line=entry["line"],
                score=score,
                reason=entry.get("reason", ""),
                axes=axes,
                function=ctx.get("function"),
                function_score=ctx.get("function_score"),
            ))

        per_file_totals[rel] = {
            "n_scored": n_scored,
            "n_flagged": flagged_in_file,
            "top_score": round(top_score_in_file, 4),
        }

    all_flagged.sort(key=lambda f: -f.score)
    top = all_flagged[:top_n]

    # Repo-wide dominant axis: which axis fired most on average across
    # flagged lines. Different from per-line dominant — answers "if you
    # had to give this repo one label, what is it?"
    dominant_axis: str | None = None
    if axis_counts:
        axis_means = {a: axis_totals[a] / axis_counts[a] for a in axis_counts}
        dominant_axis = max(axis_means, key=axis_means.get)

    return {
        "summary": {
            "root": str(root),
            "files_scanned": scanned,
            "files_failed": len(failed),
            "total_flagged": len(all_flagged),
            "dominant_axis": dominant_axis,
            "min_score": min_score,
            "top_n": top_n,
        },
        "top_flagged": [
            {
                "path": f.path,
                "line": f.line,
                "score": round(f.score, 4),
                "reason": f.reason,
                "axes": f.axes,
                "function": f.function,
                "function_score": f.function_score,
            }
            for f in top
        ],
        "per_file": per_file_totals,
        "failed": failed,
    }


def _human_render(result: dict[str, Any]) -> str:
    """Terminal rendering of the repo scan, grouped so a reviewer can scan
    by hot spot rather than reading a flat list."""
    lines: list[str] = []
    s = result["summary"]
    lines.append(f"Mr. Spiky scanned {s['files_scanned']} Python file(s) under {s['root']}")
    if s["files_failed"]:
        lines.append(f"  ({s['files_failed']} file(s) failed to analyze; see .failed)")
    lines.append(
        f"  {s['total_flagged']} line(s) crossed the flag threshold "
        f"(min_score={s['min_score']:.2f}); dominant axis: {s['dominant_axis'] or '-'}"
    )
    lines.append("")

    per_file = sorted(
        result["per_file"].items(),
        key=lambda kv: (-kv[1]["top_score"], -kv[1]["n_flagged"]),
    )
    lines.append("--- Per-file summary (top 15) ---")
    lines.append(f"{'file':<50} {'scored':>7} {'flagged':>8} {'top':>7}")
    for path, stats in per_file[:15]:
        lines.append(
            f"{path:<50} {stats['n_scored']:>7} {stats['n_flagged']:>8} "
            f"{stats['top_score']:>7.3f}"
        )
    lines.append("")

    lines.append(f"--- Top {len(result['top_flagged'])} flagged lines across the repo ---")
    for i, entry in enumerate(result["top_flagged"], 1):
        fn = f"  ({entry['function']})" if entry.get("function") else ""
        lines.append(f"{i:>3}. {entry['path']}:{entry['line']}  score={entry['score']:.3f}{fn}")
        if entry.get("reason"):
            first_sentence = entry["reason"].split(". ")[0]
            lines.append(f"     {first_sentence}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory (or file) to scan")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.9)
    ap.add_argument("--format", choices=("human", "json"), default="human")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    result = score_repo(root, min_score=args.min_score, top_n=args.top_n)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(_human_render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
