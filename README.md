# Mr. Spiky — Intuition Compiler

> Can intuition be translated into something executable?
> A senior engineer looks at code and instantly knows "this is wrong."
> Mr. Spiky is a first attempt at encoding that intuition.

A spiking neural network (SNN) trained unsupervised via STDP on **~2680 Python
functions written by maintainers at CPython, Django, FastAPI, Flask, requests,
black, httpx, pydantic, sqlalchemy, and poetry** — code that shipped through
review at organizations with strong review culture. At inference the SNN reads
your code line-by-line as a temporal stream and fires on lines that light up
neurons which are usually quiet on senior-approved code.

**Four ways to use it:**

- **HTTP API** — `POST /analyze` returns per-line scores plus a multi-axis
  reasoning breakdown. Ship as a docker container, wire up to a frontend.
- **GitHub PR review bot** — a reusable Actions workflow that comments on
  structurally-unusual lines a pull request adds. See [PR review
  bot](#pr-review-bot) below, or [`docs/GITHUB_ACTION.md`][ga] for adoption
  in another repo.
- **Whole-repo scan** — `just repo-review path/to/repo` walks every `.py`
  file and prints the top-N gnarliest lines across the tree. Also
  available as structured JSON for CI dashboards.
- **Per-team axis weights** — add `axis_weights` to any `/analyze` request
  to bias the score toward the axes your team cares about (or away from
  the ones it doesn't). No retraining needed.

[ga]: docs/GITHUB_ACTION.md

---

## Contents

- [What it detects (and honestly doesn't)](#what-it-detects-and-honestly-doesnt)
- [The biological claim in 60 seconds](#the-biological-claim-in-60-seconds)
- [Setup + Run](#setup--run)
- [HTTP API for a frontend](#http-api-for-a-frontend)
- [PR review bot](#pr-review-bot)
- [Whole-repo scan](#whole-repo-scan)
- [Per-team axis weights](#per-team-axis-weights)
- [Architecture](#architecture)
- [Validation](#validation)
- [Why an SNN?](#why-an-snn)

## What it detects (and honestly doesn't)

**Detects:** structurally unusual code relative to senior Python. Deep
nesting, high cyclomatic complexity, dense use-def chains, heavy call-graph
delegation, unusual identifier density, syntax errors, exception-handling
pressure. The kinds of things a reviewer circles in a PR.

**Doesn't detect:** semantic bugs. Mr. Spiky can't tell `x = y` from `x == y`
at the logic level — the AST features are structural, not semantic. This is
validated honestly: on PyResBugs (4000 real buggy/fixed Python pairs), it
performs at 49.6% balanced accuracy, chance. That negative result is
recorded in `models/threshold.json` and shown here, not swept aside.
The one exception is *syntax* errors — the `parse_error` feature force-flags
any line that doesn't parse, so `x = y` inside an `if` gets caught.

## The biological claim in 60 seconds

"Intuition Compiler" isn't a metaphor. Mr. Spiky runs on three concrete
mechanisms that your brain also uses:

**1. A neuron reads code like you do.** LIF (leaky integrate-and-fire)
neurons accumulate potential from their inputs, leak between them, and
spike when they cross threshold. Feed a function's lines in one at a
time — the neuron carries context across lines and fires on the ones a
reviewer would flinch at.

![LIF membrane accumulating across code lines](docs/deck/assets/lif_membrane_over_lines.mp4)

*A spiking neuron scans a real 14-line function from the CPython stdlib
(`linecache.checkcache`) and flinches at the same bare-except line a
reviewer would. ~30s.*

**2. It learned by Hebbian rule — "neurons that fire together, wire
together."** STDP (spike-timing-dependent plasticity) is the actual
learning rule the mammalian brain uses. Feed the SNN 2,680 senior-
authored Python functions and let neurons wire themselves according to
which patterns co-fire. No gradient descent. No labels. No teacher.
Different neurons end up specialized on different structural patterns.

![STDP: A→B strengthens on exception handling, A→C stays weak](docs/deck/assets/stdp_learning_rule.mp4)

*Three neurons; over six exposures A→B strengthens on exception-handling
patterns while A→C stays weak. Visual proof of differential
specialization from the same Hebbian rule. ~30s.*

**3. It reads temporally, not all-at-once.** A senior developer's gut
fires at line 47 because of what they read through line 46. LIF membrane
potential accumulates over time in exactly that way. Feeding code as a
temporal stream (one line = one timestep) gives per-line scores that
depend on prior context — which no per-line MLP can produce.

These three ideas are what make "encoding senior intuition" a concrete
architectural claim, not a marketing line. The clips above (rendered
with Manim) are what we use in the live pitch — you can render them
locally with `just deck-clips` if you want to iterate.

## Setup + Run

Requires [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).
Python 3.12 (torch/snntorch don't ship 3.13+ wheels).

```bash
uv sync
```

```bash
just api            # FastAPI on :8000 (mock mode until weights are trained)
just smoke          # POST a sample snippet to the running API
just test           # pytest — 18 tests

just data-pretrain  # download senior corpus (~30s–3min, 149 files, ~2MB)
just data-calib     # download PyResBugs + CodeComplex + mine annotations
just train          # fit whitening → STDP pretraining
just calibrate      # baselines + ECDF + threshold + labeled-set stats
just all            # data + train + calibrate end-to-end

just docker-build   # multi-stage build with CPU-only torch (~350MB image)
just docker-run     # run at :8000

just review-pr owner/repo 42   # score any open PR locally (uses `gh` auth)
just repo-review path/to/repo  # score every .py file, print top-N gnarliest lines
just release-models            # tar the trained artifacts for `gh release upload`
```

Local dev helpers (system-Python, gitignored) live under `.temp/`. See
`.temp/README.md` if you want pretty-printed API output.

## HTTP API for a frontend

### Endpoints

- `POST /analyze` — body `{code: string, language: "python"}`. Non-Python
  languages return HTTP 400.
- `GET /health` — status + mode diagnostics so the frontend can gate features:

  ```json
  {
    "status": "ok",
    "supported_languages": ["python"],
    "mode": "snn",
    "threshold": 0.9,
    "hidden_size": 128,
    "output_size": 32,
    "hidden_baselines_distinct": 7,
    "ecdf_reference_size": 39942
  }
  ```

  When `mode == "mock"` the payload has `mode`, `threshold` (0.55 in mock),
  and a `reason` string explaining why. Use it to render a "running against
  mock scores" banner instead of silently presenting weaker output.

CORS is enabled for `http://localhost:3000` and
`https://mr-spiky.crnicholson.com` — edit `src/api.py::app.add_middleware`
to add your own frontend origin.

### Response shape

Two line shapes: **flagged** lines carry extra reasoning fields (`reason`,
`context`, `raw_features`); **unflagged** lines stay lean.

Flagged line (all fields present):

```json
{
  "line": 167,
  "score": 0.99,
  "flag": true,
  "axes": {
    "complexity": 1.0,
    "tangled_state": 1.0,
    "hidden_calls": 1.0,
    "exception_surface": 0.0,
    "naming": 0.97,
    "malformed": 0.0
  },
  "reason": "high on complexity (1.00) + tangled_state (1.00) — deeply nested / branchy control flow; variables reach across long distances",
  "context": {
    "function": "per_timestep_attribution",
    "span": [121, 189],
    "function_score": 0.99,
    "function_score_before": 0.71,
    "function_score_delta": 0.28,
    "lineage": [
      {"kind": "If",  "label": "`if hidden_baseline is not None`", "line": 165},
      {"kind": "FunctionDef", "label": "function `per_timestep_attribution`", "line": 121}
    ]
  },
  "raw_features": {
    "nesting_depth": 1.0, "length": 0.1, "token_entropy": 0.59,
    "naming_entropy": 0.77, "cyclomatic_proxy": 0.55,
    "use_def_distance": 1.0, "name_flow": 0.33,
    "call_graph_shape": 1.0, "exception_density": 0.0,
    "parse_error": 0.0,
    "global_reach": 0.42, "attr_reach": 0.0, "call_graph_depth": 0.5
  }
}
```

Unflagged line (lean shape):

```json
{
  "line": 168,
  "score": 0.41,
  "flag": false,
  "axes": {
    "complexity": 0.55,
    "tangled_state": 0.3,
    "hidden_calls": 0.1,
    "exception_surface": 0.0,
    "naming": 0.62,
    "malformed": 0.0
  }
}
```

Top-level envelope:

```json
{
  "verdict": "3 high-intensity spikes detected — dominant axis: complexity",
  "dominant_axis": "complexity",
  "top_flagged": [167, 166, 165],
  "lines": []
}
```

### How to render each field

| Field | Type | Rendering suggestion |
| :-- | :-- | :-- |
| `verdict` | string | Banner at top of the result panel. Already includes the dominant axis. |
| `dominant_axis` | string \| null | Highlight this axis on your per-line axis chart so the user sees what the SNN is objecting to overall. |
| `top_flagged` | list<int> | Line numbers, worst first. Great for a "jump to next hot line" button. |
| `lines[i].line` | int (1-based) | Match to your source line numbering. |
| `lines[i].score` | float ∈ [0,1] | Percentile rank against senior code. `0.9` = "top 10% most unusual line vs Django/CPython/etc." Great for background-color intensity. |
| `lines[i].flag` | bool | Whether `score ≥ threshold`. Threshold varies by mode — read it from `/health`. Use for the gutter marker / underline. |
| `lines[i].axes` | dict<string, float> | Six axes, each ∈ [0,1] (may briefly exceed by ~5%). Radar chart or horizontal-bar breakdown per line. See axis glossary. |
| `lines[i].reason` | string *(flagged only)* | Ready-to-display tooltip / hover text. Reads like reviewer feedback. |
| `lines[i].context` | object *(flagged only)* | `{function, span, function_score, function_score_before?, function_score_delta?, lineage}`. `function_score_before` / `function_score_delta` are only populated by the PR-review path (when a base branch is provided). `lineage` lists up to 3 innermost AST-node ancestors (function/for/if/try/etc.) with their labels and start lines — great for "this line sits inside `if x > 0` at L14, inside `for i in …` at L12." |
| `lines[i].raw_features` | dict<string, float> *(flagged only)* | The 13 normalized inputs. Only render if you want a debug/expert view; the axes are what humans read. |

### Axis glossary

Each axis is derived from a subset of the 13 input features. A single
feature can contribute to multiple axes.

| Axis | Plain-English meaning | Features that drive it |
| :-- | :-- | :-- |
| **complexity** | Deeply nested / branchy control flow. | `nesting_depth`, `cyclomatic_proxy`, `length` |
| **tangled_state** | Variables reach across long distances; line reads state defined far away (locally, at module scope, or on `self`). | `use_def_distance`, `name_flow`, `global_reach`, `attr_reach` |
| **hidden_calls** | Delegates to opaque calls; single call chains into transitive same-file calls. | `call_graph_shape`, `call_graph_depth` |
| **exception_surface** | Try/except/raise density is high for the scope. | `exception_density` |
| **naming** | Unusual identifier density. Not always bad — flags information-dense lines. | `token_entropy`, `naming_entropy` |
| **malformed** | Line doesn't parse as valid Python. Dedicated channel — forces a flag regardless of SNN score. | `parse_error` |

### Score cheat-sheet

- `score < 0.5` — comfortably normal for senior code. Don't draw attention.
- `0.5 ≤ score < 0.7` — moderately unusual. Consider a subtle marker.
- `0.7 ≤ score < 0.9` — noticeably above senior baseline. "Warm" line;
  tooltip only, no flag.
- `score ≥ 0.9` — flagged. Comes with `reason`, `context`, `raw_features`.
  Show prominently.
- `score = 1.0` on a `malformed`-axis line — forced by parse-error override.
  Always flagged in both SNN and mock modes.

### Empty / error states

- **Blank input** → `{verdict: "no suspicious spikes detected", lines: [], top_flagged: [], dominant_axis: null}`. Render a neutral state.
- **Python syntax error inside a snippet** → the offending line is scored
  and gets `axes.malformed = 1.0`. Its `flag` is set to `true` regardless
  of SNN score. Other lines score normally.
- **Non-Python language** → HTTP 400 with `detail` naming the unsupported
  language. Surface as an error toast; disable the analyze button.
- **Mock mode** → same `/analyze` schema. Detect via `/health.mode == "mock"`
  and show a banner. Threshold drops to 0.55 in this path; scores use a
  hand-picked linear combination of raw features.

## PR review bot

Mr. Spiky runs as a reusable GitHub Actions workflow that reviews any pull
request against your default branch. It posts:

1. A **summary comment** at the top of the PR with:
   - A table of flagged lines.
   - A **"Functions that got structurally gnarlier"** table — before/after
     SNN scores for functions whose score jumped by ≥0.05.
2. **Inline review comments** on the top-N most-flagged lines, each with:
   - The axis breakdown.
   - A reasoning sentence. By default it's a templated summary that names
     *specific enclosing AST nodes* (e.g. *"this line sits inside `if
     strict` at L20, inside `if env_key in context` at L17"*). When an
     optional `ANTHROPIC_API_KEY` secret is set, Claude Haiku rewrites
     each reason into reviewer-voice English before posting (e.g. *"this
     bare `pass` sits three levels deep in nested try blocks and
     silences all exceptions without logging"*). Rephrasing falls back
     silently to the template on unset key or API failure.
   - Function-level score with a **↑/↓ delta** vs the base branch's score
     for the same function.
   - A one-line structural breadcrumb (`outer ⟶ middle ⟶ inner`).

### Adopt in your repo

Drop this into `.github/workflows/mr-spiky-review.yml`:

```yaml
name: Mr. Spiky review

on:
  pull_request_target:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    uses: Arpan-206/mr-spiky/.github/workflows/mr-spiky-review.yml@main
    with:
      min_score: 0.90      # threshold to comment on a line (default 0.90)
      max_comments: 5      # hard cap on comments per PR (default 5)
    secrets:
      # Optional. When set, Claude Haiku rewrites each flag's templated
      # reason into reviewer-voice English. Falls back silently to the
      # template on unset key or API failure.
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

That's the whole integration. `GITHUB_TOKEN` (provided automatically) is
used to post reviews via the GitHub API. See
[`docs/GITHUB_ACTION.md`][ga] for a full reference including the Docker
variant, release-pinning, and troubleshooting.

### Live demos — Arpan-206/mr-spiky-testbed

A test repo lives at
[Arpan-206/mr-spiky-testbed](https://github.com/Arpan-206/mr-spiky-testbed),
with a set of PRs each engineered to showcase one Mr. Spiky axis:

| PR | Axis showcased | What the bot did |
| :-- | :-- | :-- |
| [#4](https://github.com/Arpan-206/mr-spiky-testbed/pull/4) — nested try/except pyramid | `exception_surface` | 4 inline comments, Haiku-rephrased into reviewer voice (e.g. *"silencing all exceptions without logging or re-raising makes it impossible to debug what actually failed"*). |
| [#6](https://github.com/Arpan-206/mr-spiky-testbed/pull/6) — pipeline with 6 opaque hooks | `hidden_calls`, `naming` | 3 inline comments citing hook-chain depth and information density. |
| [#7](https://github.com/Arpan-206/mr-spiky-testbed/pull/7) — `=` typo inside `if` | `malformed` | Parse-error force-flag: `score = 1.00`, plain-English *"this doesn't parse as valid Python — `=` inside an `if` condition is a typo for `==`"*. |
| [#3](https://github.com/Arpan-206/mr-spiky-testbed/pull/3) — deeply-nested arrow code | `complexity` | Silent — pure nesting alone doesn't cross the 0.90 review threshold; it's inside the senior distribution. |
| [#5](https://github.com/Arpan-206/mr-spiky-testbed/pull/5) — session state on module globals | `tangled_state`, `global_reach` | Silent — globals-heavy needs another axis co-firing to cross the review bar. |

The silent PRs (#3, #5) are honest surfacing of a limit: single-axis
junior code can resemble legitimate long senior functions. Multi-axis
gnarl (nesting + exceptions, or delegation + dense naming) is where Mr.
Spiky's comments land clearly.

When `ANTHROPIC_API_KEY` is set as a repo secret, each templated axis
reason is rewritten by Claude Haiku into reviewer-voice English before
posting — see PR #4 and #6 comments for the difference. Falls back
silently to templates on unset key or API failure.

### Local dry runs (no comments posted)

```bash
just review-pr owner/repo 42
```

Reads your `gh` auth, fetches the diff, scores it, prints the same summary
and inline bodies to stdout. Good for tuning `--min-score` before enabling
the bot on a real repo.

## Whole-repo scan

Score every `.py` file under a directory and print the top-N gnarliest
lines across the tree:

```bash
just repo-review path/to/repo               # top 20, flag threshold 0.90
just repo-review path/to/repo 50 0.85       # top 50, threshold 0.85
```

Prunes vendored dirs (`.venv`, `node_modules`, `__pycache__`, etc.),
reports a per-file summary (lines scored, lines flagged, top score), then
lists the top-N flagged lines with `path:line`, score, function, and a
one-sentence reason. The repo-wide dominant axis tells you what kind of
code this repo has the most of — a codebase where `naming` dominates
tends to have information-dense, hard-to-scan lines; one where
`exception_surface` dominates tends to over-swallow errors.

JSON output (`--format json`) is stable and suitable for CI dashboards
or scoring drift over time.

## Per-team axis weights

Every team has a different sense of what "gnarly" means. Add an optional
`axis_weights` object to any `/analyze` request to multiply each axis's
contribution to the suspicion score:

```json
{
  "code": "...",
  "language": "python",
  "axis_weights": {
    "exception_surface": 1.5,
    "naming": 0.6
  }
}
```

- Missing axes default to `1.0` (no change). You only specify the axes
  you want to bias.
- Identity weights (all `1.0`) are a strict no-op — byte-identical scores
  to a request without `axis_weights`.
- Scores are clamped to `[0, 1]` after the reshape, so `weight = 3.0`
  doesn't force-flag flat lines. It's a reshape, not an override.
- Parse errors are un-silenceable: `axis_weights: {"malformed": 0.0}`
  still returns `score = 1.0` on any line that doesn't parse. Syntax
  errors are the one thing a team can't opt out of.

Useful for tuning per-repo without retraining: a security-focused team
can boost `exception_surface` and `hidden_calls`; a code-clarity team can
boost `naming` and `complexity`; a team drowning in dense-but-legitimate
one-liners can dampen `naming` to under 1.0.

## Architecture

**1. Pretrain (unsupervised STDP)** — 2-layer LIF SNN with 13 input dims →
**128** hidden → **32** output. Depression-dominated multiplicative STDP with
weight decay prevents saturation. Trained on 2680 functions from
`data/senior_corpus.json` (auto-fetched from 10 respected Python repos, 149
source files, ~40k line vectors).

**2. Preprocess (ZCA whitening)** — before any encoding, feature vectors are
whitened against a covariance matrix fitted on the senior corpus's per-line
features. Raw features are heavily correlated on real Python (e.g. `length` ↔
`token_entropy` at r=0.84, `cyclomatic_proxy` ↔ `call_graph_shape` at r=0.92),
so without whitening the SNN saw only ~4 truly independent dimensions and
collapsed 128 neurons into 2 clusters. With whitening the same corpus
produces multiple distinct neuron baselines per training run — real
specialization. Live count exposed as `hidden_baselines_distinct` in
`/health`.

**3. Calibrate** — feed the training corpus through the trained SNN in
sequence mode (each line = one timestep, membranes carry across lines) and
compute:

- **Per-neuron baseline** firing rates (what "normal" looks like per neuron).
- **ECDF over corpus scores** (turns raw SNN output into a percentile rank).
- **Anomaly threshold** at p90 = top-10%-most-unusual for senior code.

**4. Infer** — for new code:

1. Extract per-line features (skipping docstrings and imports).
2. Whiten with the calibrated transform.
3. Run through the SNN as a temporal sequence.
4. Score each line by *continuous membrane activation excess over the
   per-neuron baseline* — not binary spike output. Continuous membranes are
   what makes per-line scores smooth instead of collapsing to a few discrete
   values.
5. ECDF-rescale so the score is a percentile rank vs senior code.
6. **parse_error override**: any line where `ast.parse` chokes gets
   `score=1.0` and `axes.malformed=1.0` regardless of SNN output — the SNN
   can't have learned to react to parse errors since they're absent from
   senior training code.

### The 13 features

Function-local (10): `nesting_depth`, `length`, `token_entropy`,
`naming_entropy`, `cyclomatic_proxy`, `use_def_distance`, `name_flow`,
`call_graph_shape`, `exception_density`, `parse_error`.

Cross-function (3, added when `use_def_distance` — which is function-local
— proved blind to tangled state that reaches across scopes):

- `global_reach` — max line distance to a module-level def a line references.
- `attr_reach` — max line distance to a `self.attr` def in the same class.
- `call_graph_depth` — depth of transitive same-file call chain reachable
  from a line's outgoing calls (`f() → g() → h()` scores 2 on the `f()` line).

Full definitions in `src/features.py`. `parse_error` is dead (0.0) across
the entire training corpus by definition (senior code parses), so it's
inert during STDP and whitening — but at inference it becomes the SNN's
dedicated "this doesn't even compile" channel.

### Mock mode

Until `models/snn_weights.pt` exists, `infer.py` falls back to scoring lines
from a hand-picked linear combination of the 13 raw features. The API stays
up and returns the same `/analyze` schema (same fields, same axes, same
reason strings on flagged lines). `/health` reports `mode: "mock"` plus a
`reason` field so the frontend can render a banner. A warning is also
logged server-side on each request. This lets a frontend integrate against
a live API before training completes; the PR-review workflow uses the same
fallback if it can't download a release asset.

## Validation

Recorded in `models/threshold.json`. Four labeled datasets, ordered by
relevance to the pitch:

| Dataset | n | What it measures | Balanced accuracy (optimal) | Mean-score gap |
| :-- | --: | :-- | --: | --: |
| **Pylint refactor family** (functions in the senior corpus that trip `too-many-branches`, `too-many-locals`, `too-complex`, etc.) | 258 | **Cross-function structural gnarl** — the signal the 13-dim features were built to catch | **77.5%** | +0.273 |
| **Annotations** (`# noqa`, `# type: ignore`, `# pragma: no cover` from the same senior repos) | 596 | Real senior judgments — lines seniors themselves marked as exceptions | **67.4%** | +0.207 |
| **CodeComplex** (Codeforces Python, 7 algorithmic-complexity classes) | 4900 | Algorithmic complexity | **69.2%** | +0.208 |
| **PyResBugs** (buggy vs fixed Python pairs from real CVEs) | 4000 | Semantic bugs | 50.4% (chance, by design) | −0.004 |

The PyResBugs result is a **feature, not a bug**: AST-structural features
can't see semantic bugs, and the tool honestly reports that. What the SNN
*does* catch is what it claims — the tangled, deeply-nested,
high-delegation lines that seniors would circle in review.

The pylint-mined set is the strongest supervised signal in the project.
It's the ground truth the 13-dim cross-function features were reaching
for: functions whose complexity spans scopes, which `use_def_distance`
(function-local) couldn't see. The annotations set stays flat under the
13-dim retrain, which is what "these features add signal without
regressing what already worked" looks like.

STDP training is stochastic, so the trained SNN produces roughly **5–15
distinct per-neuron baselines** across the 128 hidden units on repeated
runs. The important thing is that it stably clears the ~2-cluster
degenerate case that un-whitened training got stuck in. Live count
exposed as `hidden_baselines_distinct` in `/health`.

## Why an SNN?

Senior developers read code sequentially and their gut fires at line 47
because of what they've absorbed through line 46. LIF membrane potentials
accumulate over time in exactly that way — a neuron that hasn't crossed
threshold yet is still *carrying* the influence of prior lines. Feeding a
function's lines as a temporal stream into the SNN and reading per-line
membrane state gives us context-dependent per-line scores that neither a
per-line MLP nor an average over rate-coded spikes can produce cleanly.

The temporal architecture is what earns the "intuition" framing. STDP on
the senior corpus is what gives the SNN *whose* intuition.
