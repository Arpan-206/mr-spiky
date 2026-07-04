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

from .infer import analyze, function_scores

log = logging.getLogger("mrspiky.review")

# Defaults tuned for a review bot, not the interactive API:
#   - min-score 0.90 matches the analyze flag threshold (top 10% vs
#     senior code). 0.95 was tuned before we had ground truth on where
#     realistic junior code lands and left most multi-axis gnarl silent.
#   - max-comments 5 per PR → prevents diff-spam on large refactors.
DEFAULT_MIN_SCORE = 0.90
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


def _fetch_pr_file_contents(pr_ref: str) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch full head-side AND base-side contents of every file touched.

    Head content is what the SNN scores to produce the "after" per-line
    output. Base content is what we score to get the "before" function
    scores, so we can report `function_score_delta` per flagged line — the
    concrete "your changes made this function 20% gnarlier" signal.

    Returns (head_files, base_files). base_files may be missing keys where
    the file was newly added by this PR.
    """
    m = re.match(r"^(?P<owner>[^/]+)/(?P<repo>[^#]+)#(?P<num>\d+)$", pr_ref)
    owner, repo, num = m["owner"], m["repo"], m["num"]

    pr_json = subprocess.check_output(
        ["gh", "api", f"repos/{owner}/{repo}/pulls/{num}"], text=True,
    )
    pr = json.loads(pr_json)
    head_sha = pr["head"]["sha"]
    head_repo = pr["head"]["repo"]["full_name"]  # may be a fork
    base_sha = pr["base"]["sha"]
    base_repo = pr["base"]["repo"]["full_name"]

    files_json = subprocess.check_output(
        ["gh", "api", f"repos/{owner}/{repo}/pulls/{num}/files", "--paginate"], text=True,
    )
    try:
        files = json.loads(files_json)
    except json.JSONDecodeError:
        joined = "[" + files_json.replace("][", ",") + "]"
        files = json.loads(joined) if joined.startswith("[[") else json.loads(files_json)

    import base64

    def _fetch_at(repo_full_name: str, path: str, sha: str) -> str | None:
        try:
            content_json = subprocess.check_output(
                ["gh", "api", f"repos/{repo_full_name}/contents/{path}?ref={sha}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            meta = json.loads(content_json)
            return base64.b64decode(meta["content"]).decode("utf-8", errors="replace")
        except subprocess.CalledProcessError:
            return None

    head_files: dict[str, str] = {}
    base_files: dict[str, str] = {}
    for f in files:
        path = f["filename"]
        if not path.endswith(".py"):
            continue
        if f.get("status") == "removed":
            continue
        head = _fetch_at(head_repo, path, head_sha)
        if head is None:
            log.warning("could not fetch head-side %s", path)
            continue
        head_files[path] = head
        # Newly added files won't exist on base — that's fine, we just
        # won't have a "before" score for functions in them.
        if f.get("status") != "added":
            base = _fetch_at(base_repo, path, base_sha)
            if base is not None:
                base_files[path] = base
    return head_files, base_files


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
    fn_score = ctx.get("function_score")
    fn_score_before = ctx.get("function_score_before")
    fn_delta = ctx.get("function_score_delta")
    lineage = ctx.get("lineage") or []

    lines = [
        f"**Mr. Spiky** — score `{score:.2f}` (top 10% for senior Python code)",
        "",
        reason,
        "",
    ]
    if fn:
        fn_msg = f"Inside `{fn}`"
        if fn_score is not None:
            fn_msg += f" — function-level score `{fn_score:.2f}`"
        if fn_score_before is not None and fn_delta is not None:
            arrow = "↑" if fn_delta > 0 else "↓"
            sign = "+" if fn_delta > 0 else ""
            fn_msg += (
                f" ({arrow} {sign}{fn_delta:.2f} — was `{fn_score_before:.2f}`"
                f" before your changes)"
            )
        lines.append(fn_msg)
    if lineage:
        # Compact structural breadcrumb; the full sentence version is in the
        # main reason above, this is the at-a-glance version.
        crumb = " ⟶ ".join(str(entry.get("label", "")) for entry in reversed(lineage))
        lines.append("")
        lines.append(f"<sub>Structure: {crumb}</sub>")
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
    function_deltas: list[dict[str, Any]] | None = None,
) -> str:
    if not flagged and not function_deltas:
        return (
            "**Mr. Spiky** reviewed this PR and had no comments.\n\n"
            f"Scanned {files_touched} Python file(s); nothing crossed the "
            f"review threshold. This is a *complexity* review — Mr. Spiky "
            f"can't see semantic bugs (see the README)."
        )
    lines: list[str] = []
    if flagged:
        lines.append(
            f"**Mr. Spiky** reviewed this PR — **{len(flagged)} line(s) flagged** "
            f"across {files_touched} Python file(s)."
        )
        lines.append("")
        lines.append("| file | line | score | reason |")
        lines.append("| --- | ---: | ---: | --- |")
        for e in flagged:
            r = (e.get("reason") or "").split(" — ")[0]
            lines.append(f"| `{e['path']}` | {e['line']} | {e['score']:.2f} | {r} |")
        lines.append("")
    if function_deltas:
        lines.append("**Functions that got structurally gnarlier with this PR:**")
        lines.append("")
        lines.append("| function | file | before | after | Δ |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for d in function_deltas:
            delta = d["function_score_delta"]
            arrow = "↑" if delta > 0 else "↓"
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| `{d['function']}` | `{d['path']}` | "
                f"{d['function_score_before']:.2f} | {d['function_score']:.2f} | "
                f"{arrow} {sign}{delta:.2f} |"
            )
        lines.append("")
    lines.append(
        "<sub>Mr. Spiky is a spiking neural network trained on ~2680 functions "
        "written by maintainers at CPython/Django/FastAPI/etc. It flags lines "
        "that look structurally *unusual* compared to that senior code — deep "
        "nesting, heavy delegation, tangled state. It cannot detect semantic "
        "bugs (validated on PyResBugs at chance-level, honestly).</sub>"
    )
    return "\n".join(lines)


def _base_function_scores(base_files: dict[str, str]) -> dict[tuple[str, str], float]:
    """Map (path, function_name) → per-function SNN score on the BASE branch.

    Uses the public `function_scores` helper from infer, which scores every
    function regardless of whether any of its lines flag — critical because
    a function that's clean on base won't have any per-line contexts to
    read a score from, but we still need its baseline for the delta.
    """
    out: dict[tuple[str, str], float] = {}
    for path, content in base_files.items():
        for fn_name, score in function_scores(content).items():
            out[(path, fn_name)] = score
    return out


def review(
    file_contents: dict[str, str],
    changed_lines: dict[str, set[int]],
    min_score: float,
    max_comments: int,
    base_file_contents: dict[str, str] | None = None,
    use_claude_rephrase: bool = True,
) -> dict[str, Any]:
    """Score each file, filter to changed lines, produce inline + summary payloads.

    When `base_file_contents` is provided, we also score the base-side and
    attach `function_score_before` + `function_score_delta` to each flagged
    line's context, plus a "functions that got gnarlier" table to the
    summary. This is what turns the bot from "line 42 is complex" into
    "line 42's function went from 0.71 to 0.89 with your changes."
    """
    all_flagged: list[dict[str, Any]] = []
    file_verdicts: dict[str, str] = {}
    files_actually_scored = 0

    base_fn_scores = _base_function_scores(base_file_contents or {})
    # Head-side per-function scores. Computed via the same public helper as
    # base so both sides are apples-to-apples, and so we capture functions
    # that don't have any flagged lines above min_score.
    head_fn_scores: dict[tuple[str, str], float] = {}
    for path, content in file_contents.items():
        for fn_name, score in function_scores(content).items():
            head_fn_scores[(path, fn_name)] = score

    for path, content in file_contents.items():
        result = analyze(content)
        file_verdicts[path] = result.get("verdict", "")
        files_actually_scored += 1

        changed_set = changed_lines.get(path, set())

        for entry in result.get("lines", []):
            if not entry.get("flag"):
                continue
            if entry.get("score", 0.0) < min_score:
                continue
            if entry["line"] not in changed_set:
                continue
            e = dict(entry)
            e["path"] = path
            # Stash the actual source-line text so the optional Claude
            # rephrase has something concrete to anchor on. Guarded because
            # a line number occasionally overshoots the file length by 1
            # (analyze() reports on trailing-newline sentinels).
            src_lines = content.splitlines()
            line_no = entry["line"]
            if 0 < line_no <= len(src_lines):
                e["line_text"] = src_lines[line_no - 1]
            ctx = e.get("context") or {}
            fn = ctx.get("function")
            fn_score = ctx.get("function_score")
            # Attach before/after fn scores if we have them.
            if fn is not None:
                before = base_fn_scores.get((path, fn))
                if before is not None and fn_score is not None:
                    e.setdefault("context", {})
                    e["context"] = dict(e["context"])
                    e["context"]["function_score_before"] = round(before, 4)
                    e["context"]["function_score_delta"] = round(fn_score - before, 4)
            all_flagged.append(e)

    all_flagged.sort(key=lambda e: -e["score"])
    top = all_flagged[:max_comments]

    # Optional: rewrite each top-N line's `reason` into reviewer voice via
    # Claude Haiku. Silently falls back to the templated reason if the
    # ANTHROPIC_API_KEY env var is unset, the SDK is missing, or the call
    # fails. Only runs on the top N — no point paying for lines we won't
    # comment on. Preserves the original templated reason under
    # `raw_reason` in case downstream consumers want it.
    if use_claude_rephrase and top:
        from .rephrase import RephraseInput, rephrase_batch

        inputs: list[RephraseInput] = []
        for entry in top:
            ctx = entry.get("context") or {}
            inputs.append({
                "score": entry.get("score", 0.0),
                "axes": entry.get("axes", {}),
                "lineage": ctx.get("lineage") or [],
                "line_text": entry.get("line_text", ""),
                "function_name": ctx.get("function"),
                "fallback": entry.get("reason") or "SNN flagged this line.",
            })
        rephrased = rephrase_batch(inputs)
        for entry, new_reason in zip(top, rephrased):
            entry["raw_reason"] = entry.get("reason")
            entry["reason"] = new_reason

    # "Functions that got gnarlier" — independent of the flagged-line list.
    # Threshold on the delta so we don't spam every function with a small
    # score fluctuation. 0.05 is roughly one std of noise across runs.
    _MIN_DELTA = 0.05
    function_deltas: list[dict[str, Any]] = []
    for (path, fn), after in head_fn_scores.items():
        before = base_fn_scores.get((path, fn))
        if before is None:
            continue
        delta = after - before
        if delta < _MIN_DELTA:
            continue
        function_deltas.append({
            "path": path,
            "function": fn,
            "function_score": round(after, 4),
            "function_score_before": round(before, 4),
            "function_score_delta": round(delta, 4),
        })
    function_deltas.sort(key=lambda d: -d["function_score_delta"])
    function_deltas = function_deltas[:5]  # cap to keep the summary short

    inline_comments = [
        {
            "path": e["path"],
            "line": e["line"],
            "side": "RIGHT",
            "body": _line_body(e, e["path"]),
        }
        for e in top
    ]
    summary_comment = _summary_body(top, files_actually_scored, file_verdicts, function_deltas)

    return {
        "summary": {
            "verdict": ("no suspicious spikes" if not top
                        else f"{len(top)} high-intensity spike(s) inside the diff"),
            "flagged_count": len(top),
            "total_flagged_before_cap": len(all_flagged),
            "files_touched": files_actually_scored,
            "function_deltas": function_deltas,
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
    ap.add_argument(
        "--base-root",
        help="optional base-branch checkout root (--diff mode). When present, "
             "review computes function_score_delta for each flagged line.",
    )
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS)
    ap.add_argument(
        "--no-claude",
        action="store_true",
        help="Skip the optional Claude Haiku reason-rephrase step and use the "
             "templated reasons only. Also implied when ANTHROPIC_API_KEY is unset.",
    )
    ap.add_argument(
        "--format",
        choices=("json", "human"),
        default="json",
        help="`json` for the Action to consume; `human` for local reading",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base_file_contents: dict[str, str] = {}
    if args.pr:
        diff = _fetch_pr_diff(args.pr)
        file_contents, base_file_contents = _fetch_pr_file_contents(args.pr)
    else:
        diff = Path(args.diff).read_text()
        if not args.root:
            raise SystemExit("--root required with --diff")
        root = Path(args.root)
        file_contents = {}
        for touched in _parse_diff_added_lines(diff):
            p = root / touched
            if p.exists():
                file_contents[touched] = p.read_text()
        if args.base_root:
            base_root = Path(args.base_root)
            for touched in _parse_diff_added_lines(diff):
                p = base_root / touched
                if p.exists():
                    base_file_contents[touched] = p.read_text()

    changed = _parse_diff_added_lines(diff)
    result = review(
        file_contents=file_contents,
        changed_lines=changed,
        min_score=args.min_score,
        max_comments=args.max_comments,
        use_claude_rephrase=not args.no_claude,
        base_file_contents=base_file_contents,
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
