"""Curate a large 'senior-approved' Python corpus for STDP pretraining.

The pitch claim of Mr. Spiky is that the SNN encodes senior developer
intuition. That claim only holds if we trained on code written by senior
developers — code that shipped through review by maintainers at organizations
with strong review culture.

We fetch source files from a handful of such repos (CPython, Django, FastAPI,
Flask, requests, black, httpx, pydantic, sqlalchemy, poetry) using GitHub's
tree API to enumerate .py files, then downloading via raw content. Target
~150 files, which yields ~15k functions — enough for STDP to differentiate
neurons on the fine-grained variance within senior code.

Result:
    data/senior_corpus.json           list[str] of file contents
    data/senior_corpus_manifest.json  list of {repo, path, sha} pairs
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

import requests

OUT_PATH = Path(__file__).resolve().parent / "senior_corpus.json"
MANIFEST_PATH = Path(__file__).resolve().parent / "senior_corpus_manifest.json"

log = logging.getLogger("mrspiky.data.senior")

# Each entry: (repo, branch, path prefix to include, max files to sample).
# We pick source directories (Lib/, src/, lib/) and exclude tests/docs/vendor.
_SOURCES: list[tuple[str, str, str, int]] = [
    ("python/cpython", "main", "Lib/", 25),
    ("django/django", "main", "django/", 20),
    ("tiangolo/fastapi", "master", "fastapi/", 15),
    ("pallets/flask", "main", "src/flask/", 10),
    ("psf/requests", "main", "src/requests/", 10),
    ("psf/black", "main", "src/black/", 10),
    ("encode/httpx", "master", "httpx/", 15),
    ("pydantic/pydantic", "main", "pydantic/", 15),
    ("sqlalchemy/sqlalchemy", "main", "lib/sqlalchemy/", 20),
    ("python-poetry/poetry", "main", "src/poetry/", 10),
]

# Files whose paths match these substrings are skipped — tests, docs, migrations
# aren't representative of the "senior implementation" surface we want.
_SKIP_PATH_SUBSTRINGS = (
    "/test", "test_", "_test.", "/tests/", "/testing/",
    "/docs/", "/doc/", "/example", "/examples/",
    "/migrations/", "__pycache__", ".pyi",
    "conftest",
)

# Skip very small (probably __init__ stubs) or very large (usually generated).
_MIN_CHARS = 500
_MAX_CHARS = 300_000


def _list_files(repo: str, branch: str, prefix: str) -> list[dict]:
    """Get all .py files under `prefix` in the repo tree."""
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        log.warning("tree fetch failed for %s@%s: %d", repo, branch, r.status_code)
        return []
    tree = r.json().get("tree", [])
    out = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path.startswith(prefix):
            continue
        if not path.endswith(".py"):
            continue
        if any(s in path for s in _SKIP_PATH_SUBSTRINGS):
            continue
        out.append({"path": path, "sha": item.get("sha"), "size": item.get("size", 0)})
    return out


def _fetch_file(repo: str, branch: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:  # noqa: BLE001
        log.warning("net %s: %s", url, e)
        return None
    if r.status_code != 200:
        return None
    text = r.text
    if not (_MIN_CHARS <= len(text) <= _MAX_CHARS):
        return None
    try:
        ast.parse(text)
    except SyntaxError:
        return None
    return text


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    corpus: list[str] = []
    manifest: list[dict] = []
    total_planned = 0
    total_kept = 0

    for repo, branch, prefix, max_files in _SOURCES:
        log.info("enumerating %s %s/%s (max %d)…", repo, branch, prefix, max_files)
        candidates = _list_files(repo, branch, prefix)
        # Prefer mid-sized files: the sweet spot is real implementation, not
        # tiny stubs or huge generated bundles. Sort by size, take a stride.
        candidates.sort(key=lambda f: f["size"])
        # Filter to plausibly-sized files then evenly stride to sample diversity
        sized = [f for f in candidates if _MIN_CHARS <= f["size"] <= _MAX_CHARS]
        if len(sized) > max_files:
            stride = len(sized) / max_files
            sized = [sized[int(i * stride)] for i in range(max_files)]
        total_planned += len(sized)
        for f in sized:
            text = _fetch_file(repo, branch, f["path"])
            if text is None:
                continue
            corpus.append(text)
            manifest.append({
                "repo": repo, "branch": branch, "path": f["path"], "sha": f["sha"], "chars": len(text),
            })
            total_kept += 1
        log.info("  %s: %d/%d files kept (running total %d)", repo, len(sized), len(candidates), total_kept)

    if not corpus:
        log.error("no files fetched")
        return 1

    # Rough function count using a cheap ast.walk (avoids depending on src/).
    def _count_functions(source: str) -> int:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0
        return sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

    n_functions = sum(_count_functions(src) for src in corpus)

    OUT_PATH.write_text(json.dumps(corpus))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info(
        "wrote %d files (%d functions, ~%d MB) -> %s",
        len(corpus), n_functions, sum(len(c) for c in corpus) // 1_000_000, OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
