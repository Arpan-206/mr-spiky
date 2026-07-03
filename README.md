# Mr. Spiky — Intuition Compiler

> Can intuition be translated into something executable?
> A senior engineer looks at code and instantly knows "this is wrong."
> Mr. Spiky is a first attempt at encoding that intuition.

A spiking neural network (SNN) trained unsupervised via STDP on **~2500 Python
functions written by maintainers at CPython, Django, FastAPI, Flask, requests,
black, httpx, pydantic, sqlalchemy, and poetry** — code that shipped through
review at organizations with strong review culture. At inference the SNN reads
your code line-by-line as a temporal stream and fires on lines that light up
neurons which are usually quiet on senior-approved code.

## What it does

Given Python source, `/analyze` returns per-line scores plus a multi-axis
breakdown that explains *why* the SNN flagged something:

```json
{
  "verdict": "3 high-intensity spikes detected — dominant axis: complexity",
  "dominant_axis": "complexity",
  "lines": [
    {
      "line": 12,
      "score": 0.94,
      "flag": true,
      "axes": {
        "complexity": 0.99,
        "tangled_state": 0.25,
        "hidden_calls": 0.78,
        "exception_surface": 0.63,
        "naming": 0.81
      }
    }
  ],
  "top_flagged": [12, 45, 88]
}
```

The five axes correspond to feature groups a reviewer would recognize:
**complexity** (nesting, cyclomatic, length), **tangled_state** (use-def
distance, name flow), **hidden_calls** (delegation to opaque calls),
**exception_surface** (try/except/raise density), **naming** (token/name
entropy).

## Architecture

**1. Pretrain (unsupervised STDP)** — 2 layer LIF SNN with 9 input dims →
64 hidden → 16 output. Depression-dominated multiplicative STDP with weight
decay prevents saturation. Trained on ~2500 functions from `data/senior_corpus.json`
(auto-fetched from 10 respected Python repos).

**2. Calibrate** — feed the training corpus through the trained SNN in
sequence mode (each line = one timestep, membranes carry across lines) and
compute:
- **Per-neuron baseline** firing rates (what "normal" looks like per neuron).
- **ECDF over corpus scores** (turns raw SNN output into a percentile rank).
- **Anomaly threshold** at p90 = top-10%-most-unusual for senior code.

**3. Infer** — for new code:
1. Extract per-line features (skipping docstrings and imports).
2. Run through the SNN as a temporal sequence.
3. Score each line by *continuous membrane activation excess over the
   per-neuron baseline* — not binary spike output. Continuous membranes are
   what makes per-line scores smooth instead of collapsing to 4-5 discrete
   values.
4. ECDF-rescale so the score is a percentile rank vs senior code.

## Validation (recorded in `models/threshold.json`)

Three labeled datasets, ordered by relevance to the pitch:

| Dataset | n | What it measures | Balanced accuracy (optimal) |
| :-- | --: | :-- | --: |
| **Annotations** (`# noqa`, `# type: ignore`, `# pragma: no cover` from the same senior repos) | 596 | **Real senior judgments** — lines seniors themselves marked as exceptions | **66.1%** |
| **CodeComplex** (Codeforces Python, 7 algorithmic-complexity classes) | 4900 | Algorithmic complexity | **63.7%** |
| **PyResBugs** (buggy vs fixed Python pairs from real CVEs) | 4000 | Semantic bugs | 50.0% (chance, by design) |

The PyResBugs result is a **feature, not a bug**: AST-structural features can't
see semantic bugs, and the tool honestly reports that. What the SNN *does*
catch is what it claims — the tangled, deeply-nested, high-delegation lines
that seniors would circle in review.

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
just test           # pytest — 8 tests

just data-pretrain  # download senior corpus (~30s, 149 files, ~2MB)
just data-calib     # download PyResBugs + CodeComplex + mine annotations
just train          # STDP pretraining → models/snn_weights.pt
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
The API stays up and returns the same JSON schema — a warning is logged on
each request so it's obvious you're not running the trained SNN yet. This
lets a frontend integrate against a live API before training completes.

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
