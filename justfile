# Mr. Spiky — Intuition Compiler
# Run `just` to see recipes.

default:
    @just --list

# Install / sync deps
sync:
    uv sync

# Download the CodeSearchNet Python sample used for STDP pretraining
data-pretrain:
    uv run python3 data/download_codesearchnet.py

# Download labeled datasets for calibration provenance:
#   - PyResBugs: buggy vs fixed Python functions (semantic bugs)
#   - CodeComplex: algorithmic complexity-labeled Python (structural complexity)
data-calib:
    uv run python3 data/download_pyresbugs.py
    uv run python3 data/download_codecomplex.py

# Download both
data-all: data-pretrain data-calib

# Stage 1: unsupervised STDP pretraining → models/snn_weights.pt
train:
    uv run python3 -m src.train_stdp

# Stage 2: calibrate suspicion threshold → models/threshold.json
calibrate:
    uv run python3 -m src.calibrate

# Run the FastAPI server (auto-reload); falls back to mock scoring if no weights yet
api:
    uv run uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# Quick smoke test: POST a snippet at the running API
smoke:
    curl -s -X POST http://127.0.0.1:8000/analyze \
        -H 'content-type: application/json' \
        -d '{"code":"def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n"}' | python3 -m json.tool

# Tests
test:
    uv run python3 -m pytest tests/ -v

# End-to-end: fetch data, train, calibrate
all: data-pretrain data-calib train calibrate

# --- Review CLI ---
# Score a PR locally (uses `gh` auth, does NOT post):
#   just review-pr owner/repo 42
review-pr repo pr:
    uv run python3 -m src.review --pr {{repo}}#{{pr}} --format human

# Score a diff file locally against a checkout:
#   just review-diff path/to.patch path/to/checkout
review-diff diff root:
    uv run python3 -m src.review --diff {{diff}} --root {{root}} --format human

# Score every Python file under a directory; prints the top-N gnarliest lines.
#   just repo-review path/to/repo
#   just repo-review path/to/repo 30 0.85
repo-review root top_n="20" min_score="0.9":
    uv run python3 -m src.repo {{root}} --top-n {{top_n}} --min-score {{min_score}}

# --- Release packaging ---
# Package trained model artifacts for the GitHub Action to download.
# Produces models.tar.gz + a SHA-256 checksum. Attach both to a release with
#   gh release create v0.1.0 models.tar.gz models.tar.gz.sha256
release-models:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f models/snn_weights.pt ]; then
        echo "no models/snn_weights.pt — run \`just all\` first" >&2
        exit 1
    fi
    tar -czvf models.tar.gz \
        models/snn_weights.pt \
        models/snn_baselines.pt \
        models/snn_ecdf.pt \
        models/whitening.pt \
        models/threshold.json
    shasum -a 256 models.tar.gz > models.tar.gz.sha256
    ls -lh models.tar.gz models.tar.gz.sha256

# --- Pitch deck ---
# Serves the reveal.js deck at :5555 and opens it. `s` in the deck for speaker notes.
present:
    #!/usr/bin/env bash
    set -euo pipefail
    cd docs/deck
    # Python's built-in server is enough — the deck loads reveal.js from a CDN
    # and only needs to serve the static index.html + style.css + assets/.
    python3 -m http.server 5555 --bind 127.0.0.1 &
    PID=$!
    trap "kill $PID 2>/dev/null" EXIT
    sleep 0.5
    open "http://127.0.0.1:5555/" 2>/dev/null || xdg-open "http://127.0.0.1:5555/" 2>/dev/null || true
    echo "deck served at http://127.0.0.1:5555/  (Ctrl-C to stop)"
    wait $PID

# Render both Manim clips into docs/deck/assets/. Requires system-python `manim`.
# The .temp/ pattern applies here — don't run via uv, Manim needs its own env.
deck-clips:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v manim >/dev/null; then
        echo "manim not on PATH — install with: pip install manim" >&2
        exit 1
    fi
    cd docs/deck/manim
    manim -qm lif_membrane_over_lines.py LIFMembraneOverLines
    manim -qm stdp_learning_rule.py STDPLearningRule
    # Manim writes to media/videos/<file>/720p30/<Scene>.mp4 by default.
    cp media/videos/lif_membrane_over_lines/720p30/LIFMembraneOverLines.mp4 ../assets/lif_membrane_over_lines.mp4
    cp media/videos/stdp_learning_rule/720p30/STDPLearningRule.mp4 ../assets/stdp_learning_rule.mp4
    echo "rendered → docs/deck/assets/{lif_membrane_over_lines,stdp_learning_rule}.mp4"

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
        -d '{"code":"def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n","language":"python"}' | python3 -m json.tool