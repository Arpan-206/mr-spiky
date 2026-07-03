# Mr. Spiky — Intuition Compiler

Backend for a hackathon tool that flags "suspicious" lines of Python code using a
Spiking Neural Network trained unsupervised via STDP (spike-timing-dependent
plasticity), plus per-line Temporal Spike Attribution at inference.

## Pipeline

1. **Pretrain (unsupervised)** — sample 150–300 Python functions from
   CodeSearchNet, extract structural features (nesting depth, length, token/naming
   entropy, cyclomatic-complexity proxy), rate-code them into spike trains, and
   train a small LIF SNN via STDP. Weights → `models/snn_weights.pt`.
2. **Calibrate** — compute per-line TSA scores over the same CodeSearchNet
   corpus, take `mean + 1·std` as the anomaly cutoff, and validate against a
   labeled Python bug dataset (PyResBugs from HuggingFace, ~5000 pairs) purely
   for provenance. Threshold + full stats → `models/threshold.json`.

   **What the tool actually is, honestly:** a *structural complexity anomaly
   detector*, not a bug detector. On PyResBugs (real Python buggy/fixed pairs),
   flag rates for buggy and fixed versions are nearly identical (~57% vs ~62%)
   because semantic bugs don't change AST structure. What the tool *does* catch
   reliably: unusual nesting, cyclomatic complexity, and length — the kinds of
   lines a reviewer would circle. The `labeled_stats` block in `threshold.json`
   records this so it's transparent to the judges/reader.
3. **Infer** — for new code, extract features per line, encode as spikes, run
   through the SNN, compute TSA, and return per-line scores as JSON.

Response schema (always the same, even in mock mode):

```json
{
  "verdict": "3 high-intensity spikes detected",
  "lines": [{"line": 12, "score": 0.91, "flag": true}],
  "top_flagged": [12, 45, 88]
}
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).

```bash
uv sync
```

Python is pinned to 3.12 because `torch`/`snntorch` don't yet publish wheels for
3.13+.

## Run

```bash
just api                 # start FastAPI on :8000 (mock mode if no weights yet)
just smoke               # POST a sample snippet to the running API

just data-pretrain       # 250 Python functions from CodeSearchNet (HF, ~2-10 min)
just data-calib          # 300 labeled Python samples from PyResBugs (HF, seconds)
just train               # STDP pretraining → models/snn_weights.pt
just calibrate           # anomaly threshold → models/threshold.json
just all                 # data + train + calibrate end-to-end

just test                # pytest
```

## Mock mode

Until `models/snn_weights.pt` exists, `infer.py` falls back to scoring lines
directly from normalized AST features. The API stays up and returns the same
JSON schema — a warning is logged on each request so it's obvious you're not
running the trained SNN yet. This is intentional so the frontend can be wired up
before training finishes.