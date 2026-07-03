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

Feature vectors are ZCA-whitened against the training corpus before entering
the SNN so its 128 hidden neurons see 9 orthogonal input dimensions instead of
~4 tangled ones — that fix alone broke the SNN out of a 2-cluster collapse
(neurons converged to only 2 distinct baseline firing rates) into typically
10–17 distinct baselines, and lifted CodeComplex balanced accuracy from 58%
to 70%.

## What it does

Given Python source, `POST /analyze` (body: `{code, language}`) returns per-line
scores plus a multi-axis breakdown that explains *why* the SNN flagged
something.

## API reference (frontend integration)

### Endpoints

- `POST /analyze` — body `{code: string, language: "python"}`. Non-Python
  languages return HTTP 400.
- `GET /health` — returns backend status plus a `mode` flag so the frontend
  can tell whether the trained SNN is loaded or the server is running in
  the linear-scoring fallback:

  ```json
  {
    "status": "ok",
    "supported_languages": ["python"],
    "mode": "snn",
    "threshold": 0.9,
    "hidden_size": 128,
    "output_size": 32,
    "hidden_baselines_distinct": 17,
    "ecdf_reference_size": 39942
  }
  ```

  When `mode == "mock"` the payload contains `mode`, `threshold`, and a
  `reason` string explaining why (missing weights file). Use this to show a
  "running against mock scores — retrain to enable" banner on the frontend
  instead of silently presenting weaker output. Also handy for the
  language-selector gate: read `supported_languages` from the same call.

### Response shape

Two line shapes: **flagged** lines carry extra reasoning fields (`reason`,
`context`, `raw_features`); **unflagged** lines stay lean to keep long-file
responses small.

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
    "naming": 0.97
  },
  "reason": "high on complexity (1.00) + tangled_state (1.00) — deeply nested / branchy control flow; variables reach across long distances",
  "context": {
    "function": "per_timestep_attribution",
    "span": [121, 189],
    "function_score": 0.99
  },
  "raw_features": {
    "nesting_depth": 1.0, "length": 0.1, "token_entropy": 0.59,
    "naming_entropy": 0.77, "cyclomatic_proxy": 0.55,
    "use_def_distance": 1.0, "name_flow": 0.33,
    "call_graph_shape": 1.0, "exception_density": 0.0
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
    "naming": 0.62
  }
}
```

Top-level response envelope (fields wrap the `lines` array; each entry follows
one of the two shapes above):

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
| `verdict` | string | Banner at top of the result panel. Already includes the dominant axis, no extra formatting needed. |
| `dominant_axis` | string \| null | If not null, highlight this axis on your per-line axis chart so the user knows what the SNN is objecting to *overall*. |
| `top_flagged` | list<int> | Line numbers, worst first. Perfect for a "jump to next hot line" button or a table of contents. |
| `lines[i].line` | int (1-based) | Match to your source line numbering. |
| `lines[i].score` | float ∈ [0,1] | The main scalar. Interpret as *percentile rank against senior code*: `0.9` = "top 10% most unusual line vs Django/CPython/etc." Great for background-color intensity. |
| `lines[i].flag` | bool | Whether `score ≥ threshold`. Read the exact threshold from `/health` (0.9 in SNN mode, 0.55 in mock mode). Use for the gutter marker / underline. |
| `lines[i].axes` | dict<string, float> | Five axes, each ∈ [0,1] roughly (may briefly exceed 1.0 by ~5%). Radar chart or horizontal-bar breakdown per line. See the axis glossary below. |
| `lines[i].reason` | string *(flagged only)* | Ready-to-display tooltip / hover text. Reads like reviewer feedback. |
| `lines[i].context` | object *(flagged only)* | `{function, span: [start, end], function_score}`. `function_score` is the SNN's score for the *enclosing function as a whole* — useful to show "this line is inside a function that's also gnarly," or to fold flagged lines by function. |
| `lines[i].raw_features` | dict<string, float> *(flagged only)* | The 9 normalized inputs. Only bother rendering if you want a debug/expert view — the axes are what humans read. |

### Axis glossary (for tooltips / legends)

Each axis is a normalized 0-to-1 signal derived from the AST features that
went into the SNN. Same feature can contribute to multiple axes.

| Axis | Plain-English meaning | Features that drive it |
| :-- | :-- | :-- |
| **complexity** | Deeply nested / branchy control flow. | `nesting_depth`, `cyclomatic_proxy`, `length` |
| **tangled_state** | Variables reach across long distances; the line pulls in many named things at once. | `use_def_distance`, `name_flow` |
| **hidden_calls** | Delegates to opaque calls (user-defined, non-stdlib). Reviewer would ask "what does that function do?" | `call_graph_shape` |
| **exception_surface** | Try/except/raise density is high for the scope. | `exception_density` |
| **naming** | Unusual identifier density (many distinct names or unusual character distribution). Not always bad — flags very information-dense lines. | `token_entropy`, `naming_entropy` |

### Score interpretation cheat-sheet

- `score < 0.5` — comfortably normal for senior code. Don't draw attention.
- `0.5 ≤ score < 0.7` — moderately unusual. Consider a subtle marker (light
  color) but no flag.
- `0.7 ≤ score < 0.9` — noticeably above senior baseline. Not flagged by
  default but a "warm" line. Good for hover tooltips only.
- `score ≥ 0.9` — flagged. Renders with `flag: true` and includes `reason`,
  `context`, and `raw_features`. Show prominently.

### Empty / error states

- Blank input → `{verdict: "no suspicious spikes detected", lines: [], top_flagged: [], dominant_axis: null}`. Render a neutral "nothing to analyze" state.
- Python syntax error → same shape, empty `lines`. **Don't** show an error banner — the tool just returned nothing to flag. Show your own parser feedback if you have one.
- Non-Python language → HTTP 400 with `detail: "language 'X' not supported. Mr. Spiky's AST features are Python-only. Supported: ['python']"`. Surface as an error toast; disable the analyze button until the user switches back to Python.
- Mock mode (no trained weights on the backend) → same `/analyze` response schema, scores derived from a linear combination of raw features. Detect it by checking `mode` in the `/health` response (`"snn"` vs `"mock"`) — the mock payload also carries a `reason` string you can surface directly. In mock mode the flag threshold is 0.55 (not 0.9) and the axes are the same but reflect only what the AST-feature linear scoring saw.

## Architecture

**1. Pretrain (unsupervised STDP)** — 2 layer LIF SNN with 9 input dims →
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
produces 10–17 distinct neuron baselines per training run — real
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
   what makes per-line scores smooth instead of collapsing to 4-5 discrete
   values.
5. ECDF-rescale so the score is a percentile rank vs senior code.

## Validation (recorded in `models/threshold.json`)

Three labeled datasets, ordered by relevance to the pitch:

| Dataset | n | What it measures | Balanced accuracy (optimal) | Mean-score gap (positive − negative) |
| :-- | --: | :-- | --: | --: |
| **Annotations** (`# noqa`, `# type: ignore`, `# pragma: no cover` from the same senior repos) | 596 | **Real senior judgments** — lines seniors themselves marked as exceptions | **67.4%** | +0.253 |
| **CodeComplex** (Codeforces Python, 7 algorithmic-complexity classes) | 4900 | Algorithmic complexity | **70.1%** | +0.271 |
| **PyResBugs** (buggy vs fixed Python pairs from real CVEs) | 4000 | Semantic bugs | 50.0% (chance, by design) | −0.013 |

The PyResBugs result is a **feature, not a bug**: AST-structural features can't
see semantic bugs, and the tool honestly reports that. What the SNN *does*
catch is what it claims — the tangled, deeply-nested, high-delegation lines
that seniors would circle in review.

STDP training is stochastic, so the trained SNN produces **10–17 distinct
per-neuron baselines** across the 128 hidden units on repeated runs — the
important thing is that it stably clears the ~2-cluster degenerate case that
un-whitened training got stuck in. Live diagnostics available at `/health`.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).
Python 3.12 (torch/snntorch don't ship 3.13+ wheels).

```bash
uv sync
```

## Run

```bash
just api            # FastAPI on :8000 (mock mode until weights are trained)
just smoke          # POST a sample snippet to the running API
just test           # pytest — 9 tests

just data-pretrain  # download senior corpus (~30s, 149 files, ~2MB)
just data-calib     # download PyResBugs + CodeComplex + mine annotations
just train          # fit whitening → STDP pretraining → models/whitening.pt + snn_weights.pt
just calibrate      # baselines + ECDF + threshold → models/threshold.json + snn_baselines.pt + snn_ecdf.pt
just all            # data + train + calibrate end-to-end

just docker-build   # multi-stage build with CPU-only torch (~350MB image)
just docker-run     # run at :8000
```

The API accepts `{code, language}`; non-Python languages return 400 (the
AST features are Python-only).

## Mock mode

Until `models/snn_weights.pt` exists, `infer.py` falls back to scoring lines
directly from normalized AST features via a hand-picked linear combination.
The API stays up and returns the same `/analyze` schema (same fields, same
axes, same reason strings on flagged lines), and `/health` reports
`mode: "mock"` plus a `reason` field so the frontend can render a banner or
disable confidence badges. A warning is also logged server-side on each
request. This lets a frontend integrate against a live API before training
completes.

## Why an SNN?

Senior developers read code sequentially and their gut fires at line 47
because of what they've absorbed through line 46. LIF membrane potentials
accumulate over time in exactly that way — a neuron that hasn't crossed
threshold yet is still *carrying* the influence of prior lines. Feeding a
function's lines as a temporal stream into the SNN and reading per-line
membrane state gives us context-dependent per-line scores that neither a
per-line MLP nor an average over rate-coded spikes can produce cleanly.

The temporal architecture is what earns the "intuition" framing. STDP on the
senior corpus is what gives the SNN *whose* intuition.
