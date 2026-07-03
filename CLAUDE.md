# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is (and what it isn't)

Mr. Spiky is a **structural complexity anomaly detector** for Python, dressed up as a "code suspicion" tool. It uses a small LIF spiking neural network trained unsupervised via STDP on CodeSearchNet, plus Temporal Spike Attribution (TSA) at inference to score each line.

**Important honesty caveat baked into the design.** The features (`src/features.py`) are AST-structural only: nesting depth, cyclomatic proxy, token/naming entropy, length. These features are provably blind to semantic bugs — the PyResBugs validation (`data/download_pyresbugs.py`, 4000 samples) records ~49% balanced accuracy, i.e. chance. What the tool *does* pick up cleanly is algorithmic/structural complexity — CodeComplex validation (4900 samples) shows ~58% balanced accuracy with complex-class means 64% higher than simple-class means. Both stats are recorded in `models/threshold.json` as `pyresbugs_stats` and `codecomplex_stats`. **Do not "fix" this by tuning the threshold to make PyResBugs numbers look better** — the negative result is the honest finding, and richer features (data-flow, use-def) would be the only real fix.

## Common commands

Everything runs through `just`. Python is pinned to `3.11–3.12` in `pyproject.toml` because torch/snntorch have no wheels for 3.13+.

```bash
just sync            # uv sync (installs deps into .venv)
just all             # data-pretrain + data-calib + train + calibrate
just api             # FastAPI on :8000 with --reload
just test            # pytest tests/ -v
just docker-build    # build mrspiky:latest (assumes `just all` ran first)
just docker-run      # docker run -p 8000:8000
```

Single test: `uv run python -m pytest tests/test_api.py::test_analyze_rejects_unsupported_language -v`.

CLI-style run of a script (never `python src/foo.py` — package-relative imports fail): `uv run python -m src.train_stdp`, `uv run python -m src.calibrate`, `uv run python -m src.infer` (reads code from stdin).

## Pipeline architecture

Three stages, each producing an artifact the next stage reads. The API works even if only stage 1 is done — later artifacts unlock better behavior but the response schema is stable throughout.

```
CodeSearchNet ──► train_stdp.py ──► models/snn_weights.pt
        │                                │
        └───────────► calibrate.py ◄─────┘        ◄── PyResBugs + CodeComplex (provenance only)
                            │
                            ▼
                   models/threshold.json
                            │
                            ▼
                     infer.py / api.py
```

**Stage 1 (unsupervised)**: `src/train_stdp.py` extracts function-level features from CodeSearchNet Python, encodes them as rate-coded spike trains (`src/encode.py`), and trains `SpikyNet` (`src/model.py`, 5→16→4 LIF) with **depression-dominated multiplicative STDP** (`A− = 0.020 > A+ = 0.005`, `Δw⁻ ∝ w`, plus weight decay). This specific STDP flavor is load-bearing — earlier additive STDP with `A+ > A−` saturated every weight to `W_MAX` after ~750 updates; don't revert to it.

**Stage 2 (unsupervised anomaly cutoff)**: `src/calibrate.py` runs line-level features from the same CodeSearchNet corpus through the trained SNN, computes TSA intensity per line, and takes `mean + K·std` as the threshold (`ANOMALY_K = 1.0` currently, giving ~85th percentile). This is *not* fit to labeled data — the labeled datasets are diagnostic-only. `ANOMALY_K` is the main sensitivity knob: lower flags more, higher flags fewer.

**Stage 3 (inference)**: `src/infer.py` extracts line-level features from arbitrary code, runs them through the SNN, and flags lines with TSA ≥ threshold. `src/api.py` wraps this in FastAPI. Both must run at line granularity — mixing function-level calibration with line-level inference silently mis-scales scores (I hit this bug, don't rediscover it).

## Two invariants that break easily

1. **Line-level and function-level TSA live on different scales.** `calibrate.py::_batch_intensities` uses `extract_line_features`, and `infer.py::_snn_scores` runs on line-level vectors too. If you switch one to function-level for any reason, the threshold no longer applies — you'll either flag nothing or flag everything. Keep them symmetric.

2. **`infer.py` must never crash on bad input.** Mock mode (no `models/snn_weights.pt`) is a supported runtime state, not a build error — the frontend integrates against the API before training finishes. Missing weights → linear fallback + `WARNING` log. Syntax errors → `extract_line_features` returns `[]`, which produces a valid empty response. Both paths have tests in `tests/test_infer.py` and `tests/test_api.py`.

## API contract

`POST /analyze` with `{"code": "...", "language": "python"}`. Non-Python languages get HTTP 400 (case-insensitive check; see `SUPPORTED_LANGUAGES` in `src/api.py`). The response schema is fixed and used by the frontend:

```json
{"verdict": "<human sentence>", "lines": [{"line": int, "score": 0..1, "flag": bool}], "top_flagged": [int, ...]}
```

`/health` reports `supported_languages` so the frontend can gate its selector. If you add a new language, update `SUPPORTED_LANGUAGES` *and* implement its feature extractor (currently `ast`-based, so any new language needs its own parser — tree-sitter would be the natural choice).

## Data files and gitignore contract

`data/*.json` (downloaded corpora) and `models/*` (trained artifacts) are `.gitignore`d. Only the `data/download_*.py` scripts and empty directory-holding files are committed. Re-hydrate with `just all` (~2–10 min for the first CodeSearchNet stream — `datasets` streaming pulls a full parquet row group even to read 250 rows; this is a known-slow one-time cost, not a fixable bug).

`data/download_mlcq.py` is retained for provenance but the dataset is Java-only and unused — the "labeled calibration set" is actually PyResBugs, written to `data/mlcq_labeled.json` for backward compatibility. If you touch calibration, prefer `LABELED_PATH` naming even though the content is PyResBugs.

## Docker notes

Multi-stage build in `Dockerfile`; `uv` is installed via `pip` (not the `ghcr.io/astral-sh/uv` image) because GHCR anonymous pulls have hit quota 401s. The runtime image bakes `models/` and `data/` in — you must run `just all` before `just docker-build`, or the container starts in mock mode.
