# ---- Stage 1: install deps with uv into a virtualenv ---------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install uv from PyPI. Using pip (instead of the ghcr.io/astral-sh/uv image)
# avoids the GHCR anonymous-pull quota that periodically 401s in CI/local.
RUN pip install --no-cache-dir uv==0.11.15

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install runtime deps only into /opt/venv.
# On Linux, `[tool.uv.sources]` in pyproject.toml routes torch to the CPU-only
# wheel index (drops ~3GB of NVIDIA transitive deps). On macOS local dev it
# stays on PyPI. `--frozen` is dropped because the lockfile was generated on
# macOS where the default source applies.
RUN uv sync --no-dev --no-install-project


# ---- Stage 2: runtime image ----------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the ready-to-run venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# App code + baked-in artifacts. Datasets and models are included so the
# container starts in real-SNN mode with no external downloads.
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
