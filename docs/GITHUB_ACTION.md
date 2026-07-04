# Mr. Spiky as a PR review bot

Drop-in GitHub Action that runs Mr. Spiky on every pull request and posts a
review with inline comments on flagged lines plus a summary block at the top.

## What it does

For each Python line added or modified by a PR:

1. Fetches the full post-change file contents (needed because Mr. Spiky's
   scoring is context-aware — line N's score depends on lines 1..N-1).
2. Runs `analyze()` on each file to get per-line scores + axes.
3. Filters to lines actually inside the diff hunks (unchanged context lines
   are never commented on, even if they'd otherwise score high).
4. Applies a stricter-than-API threshold (`--min-score 0.95`, `--max-comments 5`)
   so the bot doesn't spam reviewers.
5. Posts a single GitHub review containing:
   - A **summary comment** at the top of the PR with a table of flagged lines.
   - **Inline review comments** on the top-N flagged lines with reason, top
     axes, and enclosing-function context.

## Adopt in another repo

Copy this file into `.github/workflows/mr-spiky-review.yml`:

```yaml
name: Mr. Spiky code review
on:
  pull_request_target:
    types: [opened, synchronize, reopened]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    uses: Arpan-206/mr-spiky/.github/workflows/mr-spiky-review.yml@main
```

That's it — no secrets, no configuration. The action uses the default
`GITHUB_TOKEN` to post the review.

## Configuration

Pass inputs to the reusable workflow via `with:`:

```yaml
jobs:
  review:
    uses: Arpan-206/mr-spiky/.github/workflows/mr-spiky-review.yml@main
    with:
      min_score: 0.92        # score threshold to flag a line (0.0–1.0)
      max_comments: 10       # hard cap on inline comments per PR
      mr_spiky_ref: v0.2.0   # pin to a specific mr-spiky version
```

All three inputs are optional. Defaults: `min_score=0.95`, `max_comments=5`,
`mr_spiky_ref=main`. Lower `min_score` = more comments = noisier. In prod,
pin `mr_spiky_ref` to a tag so bot behavior is stable.

## What the bot can and can't detect

**Can:** structurally unusual code — deep nesting, high cyclomatic
complexity, dense use-def chains, heavy call-graph delegation, unusual
identifier density, syntax errors.

**Can't:** semantic bugs. This is a deliberate design honesty — Mr. Spiky's
AST features don't see the difference between `x = y` and `x == y` at the
logic level. Validated on PyResBugs at chance accuracy (see the main
README's Validation section). If your PR is a subtle bug fix, Mr. Spiky
will almost certainly stay quiet on it — that's not what it's for.

## Running locally against a PR

```bash
python -m src.review --pr owner/repo#N --format human
```

Uses your `gh` auth, doesn't post anything, prints the same summary +
inline bodies to stdout. Good for kicking the tires before you enable
the bot on a real repo.

## Troubleshooting

- **"no release asset found — running in mock mode"** — the workflow can't
  find trained SNN artifacts, so it falls back to the linear-feature
  scoring path. Comments still post; scores are less refined. Fix: create
  a GitHub release on `Arpan-206/mr-spiky` with `models.tar.gz` attached.
- **Bot commented on the wrong lines** — Mr. Spiky filters comments to
  changed lines inside diff hunks. If you're seeing it comment on unchanged
  code, that's a bug — please open an issue with the diff and workflow log.
- **Bot didn't comment at all** — either no lines cleared the 0.95
  threshold (common on small clean PRs, which is the intended behavior)
  or the file wasn't Python.
