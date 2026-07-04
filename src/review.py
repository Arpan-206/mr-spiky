"""Mr. Spiky review CLI — score a GitHub PR (or any diff) as a code reviewer.

Two entry points depending on how you got the diff:

    python -m src.review --pr owner/repo#N          # uses `gh api`
    python -m src.review --diff path/to.patch       # unified-diff file

Emits, per invocation, three payloads to stdout as JSON:

    {
      "summary": {
        "verdict": "...",
        "flagged_count": 3,
        "files_touched": 5,
        "top_flagged_lines": [{"path": "...", "line": 42, "score": 0.94,
                               "reason": "...", "axes": {...}}]
      },
      "inline_comments": [
        {"path": "src/foo.py", "line": 42, "side": "RIGHT",
         "body": "**Mr. Spiky flag** (score 0.94)…"}
      ],
      "summary_comment": "…markdown blob…"
    }

The GitHub Action workflow reads this JSON and POSTs the two payloads. A
human running the CLI locally can also pipe stdout into `jq` or through
`.temp/*.py` for pretty printing.

The bot uses a *stricter* threshold than the interactive API by default —
review noise burns reviewer trust fast. `--min-score` and `--max-comments`
give you knobs. Only lines inside the diff's changed-line hunks are ever
considered; unchanged context lines are skipped even if they'd otherwise
score high.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .infer import analyze

log = logging.getLogger("mrspiky.review")

# Defaults tuned for a review bot, not the interactive API:
#   - min-score 0.95 (vs 0.9 default) → very high bar to comment inline
#   - max-comments 5 per PR → prevents diff-spam on large refactors
DEFAULT_MIN_SCORE = 0.95
DEFAULT_MAX_COMMENTS = 5


@dataclass
class ChangedLines:
    """Line ranges inside a file's new-side (post-change) content."""
    path: str
    added_lines: set[int]  # 1-based line numbers on the RIGHT (post) side
    new_content: str


# ---------------------------------------------------------------------------
# Diff fetching + parsing
# ---------------------------------------------------------------------------

def _fetch_pr_diff(pr_ref: str) -> str:
    """`pr_ref` is 'owner/repo#N'. Uses `gh` CLI which respects auth already."""
    m = re.match(r"^(?P<repo>[^/]+/[^#]+)#(?P<num>\d+)$", pr_ref)
    if not m:
        raise SystemExit(f"invalid --pr {pr_ref!r}: expected owner/repo#N")
    repo = m["repo"]
    num = m["num"]
    # `gh pr diff` prints a unified diff to stdout. It uses whatever auth the
    # user has configured (no separate token needed for the CLI mode).
    try:
        out = subprocess.check_output(
            ["gh", "pr", "diff", num, "--repo", repo],
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit("`gh` CLI not found. Install it or use --diff <path> instead.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"`gh pr diff` failed: {e}")
    return out


def _fetch_pr_file_contents(pr_ref: str) -> dict[str, str]:
    """Fetch full post-change contents of every file touched by the PR.

    We need this because the diff alone is not enough for the SNN — the
    scoring is contextual (line N's score depends on lines 1..N-1). So we
    grab each file at the PR's head commit and score the whole thing, then
    filter to only lines actually inside the diff hunks.
    """
    m = re.match(r"^(?P<owner>[^/]+)/(?P<repo>[^#]+)#(?P<num>\d+)$", pr_ref)
    owner, repo, num = m["owner"], m["repo"], m["num"]

    # Use `gh api` to get the PR + list-of-files + head SHA.
    pr_json = subprocess.check_output(
        ["gh", "api", f"repos/{owner}/{repo}/pulls/{num}"], text=True,
    )
    pr = json.loads(pr_json)
    head_sha = pr["head"]["sha"]
    head_repo = pr["head"]["repo"]["full_name"]  # may be a fork

    files_json = subprocess.check_output(
        ["gh", "api", f"repos/{owner}/{repo}/pulls/{num}/files", "--paginate"], text=True,
    )
    # `--paginate` concatenates JSON arrays; parse leniently.
    try:
        files = json.loads(files_json)
    except json.JSONDecodeError:
        # `gh api --paginate` sometimes joins arrays as ][ — normalize.
        joined = "[" + files_json.replace("][", ",") + "]"
        files = json.loads(joined) if joined.startswith("[[") else json.loads(files_json)

    out: dict[str, str] = {}
    for f in files:
        path = f["filename"]
        if not path.endswith(".py"):
            continue
        if f.get("status") == "removed":
            continue
        # Fetch file at head SHA via contents API.
        try:
            content_json = subprocess.check_output(
                ["gh", "api", f"repos/{head_repo}/contents/{path}?ref={head_sha}"],
                text=True,
            )
            meta = json.loads(content_json)
            import base64
            raw = base64.b64decode(meta["content"]).decode("utf-8", errors="replace")
            out[path] = raw
        except subprocess.CalledProcessError as e:
            log.warning("could not fetch %s: %s", path, e)
    return out


def _parse_diff_added_lines(unified_diff: str) -> dict[str, set[int]]:
    """Given a unified diff, return {path -> set of RIGHT-side line numbers
    that are added or modified}. Deletions are ignored since they have no
    RIGHT-side location to comment on.
    """
    changed: dict[str, set[int]] = {}
    current_path: str | None = None
    new_line = 0

    for line in unified_diff.splitlines():
        # File header: `+++ b/path/to/file.py`
        if line.startswith("+++ "):
            m = re.match(r"\+\+\+ (?:b/)?(.*)", line)
            current_path = m.group(1) if m else None
            if current_path == "/dev/null":
                current_path = None
            continue
        # Hunk header: `@@ -a,b +c,d @@`
        if line.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                new_line = int(m.group(1))
            continue
        if current_path is None or not current_path.endswith(".py"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changed.setdefault(current_path, set()).add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # removed lines have no new-side line number
        else:
            new_line += 1

    return changed


# ---------------------------------------------------------------------------
# Reviewing
# ---------------------------------------------------------------------------

def _line_body(entry: dict[str, Any], path: str) -> str:
    """Markdown body for an inline review comment."""
    score = entry["score"]
    axes = entry.get("axes", {})
    top_axes = sorted(axes.items(), key=lambda kv: -kv[1])[:3]
    reason = entry.get("reason") or "SNN flagged this line."
    ctx = entry.get("context") or {}
    fn = ctx.get("function")

    lines = [
        f"**Mr. Spiky** — score `{score:.2f}` (top 5% for senior Python code)",
        "",
        reason,
        "",
    ]
    if fn:
        fn_score = ctx.get("function_score")
        fn_msg = f"Inside `{fn}`"
        if fn_score is not None:
            fn_msg += f" — function-level score `{fn_score:.2f}`"
        lines.append(fn_msg)
    lines.append("")
    lines.append("<sub>Top axes: " + ", ".join(f"`{n}` {v:.2f}" for n, v in top_axes) + "</sub>")
    lines.append(
        "<sub>Not a bug detector — see the [Mr. Spiky README]"
        "(https://github.com/Arpan-206/mr-spiky) for what the axes mean.</sub>"
    )
    return "\n".join(lines)


def _summary_body(
    flagged: list[dict[str, Any]],
    files_touched: int,
    file_verdicts: dict[str, str],
) -> str:
    if not flagged:
        return (
            "**Mr. Spiky** reviewed this PR and had no comments.\n\n"
            f"Scanned {files_touched} Python file(s); nothing crossed the "
            f"review threshold. This is a *complexity* review — Mr. Spiky "
            f"can't see semantic bugs (see the README)."
        )
    lines = [
        f"**Mr. Spiky** reviewed this PR — **{len(flagged)} line(s) flagged** across "
        f"{files_touched} Python file(s).",
        "",
        "| file | line | score | reason |",
        "| --- | ---: | ---: | --- |",
    ]
    for e in flagged:
        r = (e.get("reason") or "").split(" — ")[0]  # first clause only
        lines.append(f"| `{e['path']}` | {e['line']} | {e['score']:.2f} | {r} |")
    lines.append("")
    lines.append(
        "<sub>Mr. Spiky is a spiking neural network trained on ~2600 functions "
        "written by maintainers at CPython/Django/FastAPI/etc. It flags lines "
        "that look structurally *unusual* compared to that senior code — deep "
        "nesting, heavy delegation, tangled state. It cannot detect semantic "
        "bugs (validated on PyResBugs at chance-level, honestly).</sub>",
    )
    return "\n".join(lines)


def review(
    file_contents: dict[str, str],
    changed_lines: dict[str, set[int]],
    min_score: float,
    max_comments: int,
) -> dict[str, Any]:
    """Score each file, filter to changed lines, produce inline + summary payloads."""
    all_flagged: list[dict[str, Any]] = []
    file_verdicts: dict[str, str] = {}
    files_actually_scored = 0

    for path, content in file_contents.items():
        result = analyze(content)
        file_verdicts[path] = result.get("verdict", "")
        files_actually_scored += 1

        changed_set = changed_lines.get(path, set())
        if not changed_set:
            continue

        for entry in result.get("lines", []):
            if not entry.get("flag"):
                continue
            if entry.get("score", 0.0) < min_score:
                continue
            if entry["line"] not in changed_set:
                continue
            e = dict(entry)
            e["path"] = path
            all_flagged.append(e)

    all_flagged.sort(key=lambda e: -e["score"])
    top = all_flagged[:max_comments]

    inline_comments = [
        {
            "path": e["path"],
            "line": e["line"],
            "side": "RIGHT",
            "body": _line_body(e, e["path"]),
        }
        for e in top
    ]
    summary_comment = _summary_body(top, files_actually_scored, file_verdicts)

    return {
        "summary": {
            "verdict": ("no suspicious spikes" if not top
                        else f"{len(top)} high-intensity spike(s) inside the diff"),
            "flagged_count": len(top),
            "total_flagged_before_cap": len(all_flagged),
            "files_touched": files_actually_scored,
            "min_score": min_score,
            "max_comments": max_comments,
            "top_flagged_lines": [
                {"path": e["path"], "line": e["line"], "score": e["score"],
                 "reason": e.get("reason"), "axes": e.get("axes")}
                for e in top
            ],
        },
        "inline_comments": inline_comments,
        "summary_comment": summary_comment,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--pr", help="owner/repo#N — fetches diff + files via `gh`")
    src_group.add_argument("--diff", help="path to a unified-diff file (use with --root)")
    ap.add_argument(
        "--root",
        help="local checkout root (only for --diff); files are read from disk",
    )
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS)
    ap.add_argument(
        "--format",
        choices=("json", "human"),
        default="json",
        help="`json` for the Action to consume; `human` for local reading",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.pr:
        diff = _fetch_pr_diff(args.pr)
        file_contents = _fetch_pr_file_contents(args.pr)
    else:
        diff = Path(args.diff).read_text()
        if not args.root:
            raise SystemExit("--root required with --diff")
        root = Path(args.root)
        file_contents = {}
        # Only include files whose head-side content is on local disk.
        for touched in _parse_diff_added_lines(diff):
            p = root / touched
            if p.exists():
                file_contents[touched] = p.read_text()

    changed = _parse_diff_added_lines(diff)
    result = review(
        file_contents=file_contents,
        changed_lines=changed,
        min_score=args.min_score,
        max_comments=args.max_comments,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        # Human mode: print the summary_comment as markdown and each inline
        # body as a boxed block — useful for local demos.
        print(result["summary_comment"])
        print()
        for c in result["inline_comments"]:
            print(f"--- {c['path']}:{c['line']} ---")
            print(c["body"])
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
