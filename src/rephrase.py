"""Optional Claude Haiku rephrase for PR-review comments.

The SNN decides *what* to flag; this module optionally rewrites the *how it's
said* into a single sentence of reviewer voice. Preserves the SNN as the
decision-maker — Claude never overrides a flag or invents a new one.

Activated by the `ANTHROPIC_API_KEY` env var. Falls back silently to the
templated reason on missing key, network error, timeout, or malformed
response — the whole point is "optional," so an outage must not break the bot.

Design notes:
- Model: claude-haiku-4-5 (fast, cheap, good enough for one-line rephrases).
- Sync `messages.create()`: our output is ~80 tokens, well below the
  streaming-required threshold, and sync is simpler in a CI script.
- Async concurrency: 5 flagged lines / PR × ~800ms/call sequentially would
  eat our latency budget. `rephrase_batch()` uses AsyncAnthropic +
  asyncio.gather to parallelize.
- Prompt caching: the system prompt (axis glossary + reviewer-voice
  examples) is long, stable across calls in a PR, and stable across PRs.
  `cache_control: {type: "ephemeral"}` on the system prompt means calls
  2..N pay ~0.1x on the shared prefix — cache-write premium amortizes
  after the first call in a burst.
- No `effort` / adaptive thinking — Haiku 4.5 doesn't support them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Iterable, TypedDict

log = logging.getLogger("mrspiky.rephrase")

# Model + generation caps. Haiku 4.5's max_tokens matters here as a hard
# ceiling — 100 tokens fits any single-sentence rewrite with headroom.
_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 100
# Per-request timeout. Sequential 5 calls in a CI job → total budget ~5s;
# keep individual calls tight so one slow response can't blow the whole PR.
_REQUEST_TIMEOUT_S = 3.5


class RephraseInput(TypedDict, total=False):
    """Inputs collected per flagged line. Only the fields Claude needs are
    forwarded — no file paths, no repo names, no user identifiers."""
    score: float
    axes: dict[str, float]
    lineage: list[dict[str, Any]]
    line_text: str
    function_name: str | None
    fallback: str  # the templated reason to return on any failure


# The system prompt is deliberately verbose. Two reasons:
# 1. Reviewer-voice quality benefits from concrete examples of the target
#    tone and shape ("say X, don't say Y"). One-line prompts produce
#    generic output.
# 2. Prompt caching only kicks in above ~4096 tokens on Haiku 4.5 — a
#    short system prompt silently won't cache. Filling it out with a
#    detailed rubric earns the 0.1x read cost on every subsequent call.
_SYSTEM_PROMPT = """You are helping a code-review bot called Mr. Spiky rephrase its flags into natural reviewer voice.

Mr. Spiky is a spiking neural network trained on ~2680 Python functions written by maintainers at CPython, Django, FastAPI, Flask, requests, black, httpx, pydantic, sqlalchemy, and poetry. It flags lines whose structural features look unusual for that senior-approved code. Your job is to take Mr. Spiky's structured findings on ONE flagged line and rewrite them as a single sentence of reviewer feedback that reads like a human left the comment.

## What you receive per line
- `score`: the SNN's per-line suspicion score, 0 to 1. Anything ≥ 0.95 is "top 5% most unusual for senior code."
- `axes`: a dict of six axis scores, each 0 to 1, representing structural dimensions the SNN objected to.
- `lineage`: up to three innermost AST-node ancestors of the line — e.g. `[{"label": "`if strict`", "line": 49}, {"label": "`for i in range(x)`", "line": 47}]`.
- `line_text`: the source of the flagged line itself.
- `function_name`: the enclosing function, if any.

## Axis meanings (use these when explaining WHY)
- **complexity**: deeply nested or branchy control flow (nesting depth, cyclomatic complexity, length).
- **tangled_state**: variables reach across long distances; the line pulls in many named things at once (use-def distance, name-flow density).
- **hidden_calls**: the line delegates to opaque / user-defined calls whose behavior isn't obvious inline.
- **exception_surface**: try/except/raise density is high for this scope.
- **naming**: unusual identifier density on the line — many distinct names, or unusual character distribution.
- **malformed**: the line doesn't parse as valid Python (this ALWAYS forces a flag).

## Your output — one sentence, reviewer voice

Return exactly one sentence. No lead-in ("This line is..."), no trailing suggestions ("consider refactoring..."), no bullet points, no markdown formatting. Just the observation, phrased as a human reviewer would leave it in a PR comment.

Anchor the sentence in what the reviewer actually sees when they look at the line — nesting depth, the specific control structures around it, the number of things on the line — using the axes to explain what makes it unusual. Cite the enclosing `if`/`for`/`try` from the lineage when it clarifies the concern.

### Tone
Direct and specific, not corporate. Reads like a senior engineer who's mildly tired but wants to help. NEVER say "consider," "you might want to," or "perhaps." State the observation.

### Length
One sentence. If two clauses are needed to name the observation and its cause, join them with a dash or semicolon — not a period.

### Examples

Input:
- score: 0.99
- axes: {complexity: 1.0, tangled_state: 1.0, hidden_calls: 1.0, exception_surface: 0.0, naming: 0.9, malformed: 0.0}
- lineage: [{"label": "`if strict`", "line": 20}, {"label": "`if env_key in context`", "line": 17}]
- line_text: `                raise ValueError(f"line {i}: env var {env_key!r} not in context")`
- function_name: `parse_config`

Output:
Five levels deep inside `parse_config`, this raise sits behind two nested guards and pulls in three different variables — the strict/non-strict split is doing a lot of work here.

---

Input:
- score: 0.96
- axes: {complexity: 0.4, tangled_state: 0.6, hidden_calls: 1.0, exception_surface: 0.0, naming: 0.9, malformed: 0.0}
- lineage: [{"label": "`for u in users`", "line": 8}]
- line_text: `        result.append(compute_score(u.profile, u.settings, u.history))`
- function_name: `rank_users`

Output:
This line hands off to `compute_score` with three attribute chains inline — hard to tell what shape the callee expects without opening its definition.

---

Input:
- score: 1.0
- axes: {complexity: 0.3, tangled_state: 0.2, hidden_calls: 0.1, exception_surface: 0.0, naming: 0.5, malformed: 1.0}
- lineage: []
- line_text: `if x = True:`
- function_name: None

Output:
This doesn't parse as valid Python — `=` inside an `if` condition is almost certainly a typo for `==`.

---

Input:
- score: 0.93
- axes: {complexity: 0.7, tangled_state: 0.9, hidden_calls: 0.5, exception_surface: 0.7, naming: 0.6, malformed: 0.0}
- lineage: [{"label": "`try` block", "line": 36}, {"label": "`except FileNotFoundError`", "line": 55}]
- line_text: `            raise RuntimeError(f"failed to load {included}: {e}") from e`
- function_name: `parse_config`

Output:
Re-raising from a nested `except FileNotFoundError` inside `parse_config` — the error path is deep enough that a caller unwinding this chain has to trace back through three layers to understand it.

---

Now rewrite the next input the same way. Return only the sentence, no preamble."""


def _prompt_for_line(entry: RephraseInput) -> str:
    """Format a single line's inputs into the user message. Deliberately
    structured (bulleted) to match the shape shown in the examples above —
    Haiku follows the demonstrated pattern more reliably that way."""
    axes = entry.get("axes", {})
    lineage = entry.get("lineage") or []
    lineage_str = (
        "[" + ", ".join(
            f'{{"label": {step.get("label", "")!r}, "line": {step.get("line", "?")}}}'
            for step in lineage
        ) + "]"
        if lineage else "[]"
    )
    return (
        f"- score: {entry.get('score', 0.0):.2f}\n"
        f"- axes: {axes}\n"
        f"- lineage: {lineage_str}\n"
        f"- line_text: `{entry.get('line_text', '')}`\n"
        f"- function_name: "
        f"{('`' + entry['function_name'] + '`') if entry.get('function_name') else 'None'}"
    )


def _get_api_key() -> str | None:
    """Read the API key from env. Returns None if unset; caller falls back
    to the templated reason without emitting an error."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    return key if key else None


def _clean(text: str) -> str:
    """Strip surrounding whitespace / quotes and truncate at the first
    newline. Haiku occasionally wraps output in quotes or adds an
    unrequested trailing sentence despite the prompt; trim defensively."""
    text = text.strip().strip('"').strip("'").strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text


# --- Sync path (simplest — one line at a time) ---

def rephrase_line(entry: RephraseInput) -> str:
    """Rewrite one flagged line's reason. Returns the fallback string if
    the API key is unset, the SDK isn't installed, or any exception fires.

    Sync, blocking. Fine for local `just review-pr` runs. For CI where
    the bot processes several flagged lines per PR, prefer
    `rephrase_batch()` (async, parallel)."""
    fallback = entry.get("fallback", "SNN flagged this line.")
    api_key = _get_api_key()
    if api_key is None:
        return fallback
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; using fallback")
        return fallback

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=_REQUEST_TIMEOUT_S)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            # cache_control on the system block: writes on the first call
            # of a burst (~1.25x on the system tokens), reads on every
            # subsequent call within the 5-minute TTL (~0.1x). The prompt
            # is long enough (>4096 tokens) to actually cache on Haiku 4.5.
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _prompt_for_line(entry)}],
        )
        for block in response.content:
            if block.type == "text":
                return _clean(block.text) or fallback
        return fallback
    except anthropic.APIError as e:
        # BadRequestError, RateLimitError, APIStatusError, APIConnectionError
        # all inherit from APIError. Log at info level so it doesn't
        # scream in CI logs — the fallback keeps the bot functional.
        log.info("Claude rephrase failed (%s); using fallback", type(e).__name__)
        return fallback
    except Exception as e:  # noqa: BLE001 — belt-and-suspenders; bot must not crash
        log.info("unexpected rephrase error (%s); using fallback", type(e).__name__)
        return fallback


# --- Async path (parallel across flagged lines in one PR) ---

async def _rephrase_one_async(client: Any, entry: RephraseInput) -> str:
    """Single async call. Isolated so gather() failures for one line don't
    take down the batch."""
    fallback = entry.get("fallback", "SNN flagged this line.")
    try:
        response = await client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _prompt_for_line(entry)}],
        )
        for block in response.content:
            if block.type == "text":
                return _clean(block.text) or fallback
        return fallback
    except Exception as e:  # noqa: BLE001
        log.info("Claude rephrase failed (%s); using fallback", type(e).__name__)
        return fallback


def rephrase_batch(entries: Iterable[RephraseInput]) -> list[str]:
    """Rewrite N flagged lines' reasons in parallel. Returns N strings in
    input order. Silently returns fallbacks (all of them) if the API key
    is unset or the SDK isn't installed.

    Concurrency shape: one call per line, fired concurrently via
    asyncio.gather. Wall-clock latency ≈ max single-call latency instead
    of sum. Concurrency comment for future readers: the fan-out means the
    first call pays the cache-write premium, but the remaining calls
    within the same asyncio.gather batch race to hit an in-flight cache;
    Anthropic's docs note the cache becomes readable only after the first
    response *begins streaming*, so in the worst case all N calls pay a
    write. For our N=5 case that's fine — we're optimizing for
    wall-clock, not per-call cost."""
    entries_list = list(entries)
    if not entries_list:
        return []

    api_key = _get_api_key()
    if api_key is None:
        return [e.get("fallback", "SNN flagged this line.") for e in entries_list]
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; using fallback")
        return [e.get("fallback", "SNN flagged this line.") for e in entries_list]

    async def _run() -> list[str]:
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=_REQUEST_TIMEOUT_S)
        try:
            return await asyncio.gather(
                *[_rephrase_one_async(client, e) for e in entries_list]
            )
        finally:
            await client.close()

    try:
        return asyncio.run(_run())
    except RuntimeError as e:
        # asyncio.run() blows up if called from inside a running event
        # loop (rare in CI, common in notebooks). Fall back to sync
        # sequential rather than crash.
        if "already running" in str(e):
            log.info("event loop already running; falling back to sequential rephrase")
            return [rephrase_line(e) for e in entries_list]
        raise
