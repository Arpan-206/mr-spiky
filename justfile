# Mr. Spiky — Intuition Compiler
# Run `just` to see recipes.

default:
    @just --list

# Install / sync deps
sync:
    uv sync

# Download the CodeSearchNet Python sample used for STDP pretraining
data-pretrain:
    uv run python data/download_codesearchnet.py

# Download labeled datasets for calibration provenance:
#   - PyResBugs: buggy vs fixed Python functions (semantic bugs)
#   - CodeComplex: algorithmic complexity-labeled Python (structural complexity)
data-calib:
    uv run python data/download_pyresbugs.py
    uv run python data/download_codecomplex.py

# Download both
data-all: data-pretrain data-calib

# Stage 1: unsupervised STDP pretraining → models/snn_weights.pt
train:
    uv run python -m src.train_stdp

# Stage 2: calibrate suspicion threshold → models/threshold.json
calibrate:
    uv run python -m src.calibrate

# Run the FastAPI server (auto-reload); falls back to mock scoring if no weights yet
api:
    uv run uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# Quick smoke test: POST a snippet at the running API
smoke:
    curl -s -X POST http://127.0.0.1:8000/analyze \
        -H 'content-type: application/json' \
        -d '{"code":"def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n"}' | python -m json.tool

# Tests
test:
    uv run python -m pytest tests/ -v

# End-to-end: fetch data, train, calibrate
all: data-pretrain data-calib train calibrate

# --- Docker ---
IMAGE := "mrspiky:latest"

# Build the runtime image. Assumes `just all` has already produced models/*.
docker-build:
    docker build -t {{IMAGE}} .

# Run the API in a container on :8000
docker-run:
    docker run --rm -p 8000:8000 --name mrspiky {{IMAGE}}

# Smoke-test a running container
docker-smoke:
    curl -s -X POST http://127.0.0.1:8000/analyze \
        -H 'content-type: application/json' \
        -d '{"code":"def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n","language":"python"}' | python -m json.tool