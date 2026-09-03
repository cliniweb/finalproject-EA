# syntax=docker/dockerfile:1
# CliniAI v2 — production image.
#
# SECURITY: this image contains NO secrets. The OpenAI key (and any other
# API key) must be injected at RUNTIME via environment variables or an
# env-file — never via COPY, ARG or ENV in this file. `.env` files are
# excluded by docker/.dockerignore as a second line of defense.
#
# Build (from the project root):
#   docker build -f docker/Dockerfile -t cliniai_v2 .
# Run:
#   docker run --rm -p 8001:8001 -e OPENAI_API_KEY=sk-... cliniai_v2

# --- builder: install deps into a venv --------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# uv is the fastest way to resolve + install; pinned for reproducibility.
RUN pip install --no-cache-dir uv==0.4.30

# Reproducible install: uv.lock pins every transitive dependency version.
COPY pyproject.toml uv.lock ./
# Create the venv and install runtime deps only (no dev group), exactly as
# locked — the build FAILS if uv.lock is missing or out of sync.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-dev --no-install-project

# --- runtime: slim image, non-root user --------------------------------------
FROM python:3.11-slim AS runtime

# Never run as root in production.
RUN useradd --create-home --shell /usr/sbin/nologin cliniai
USER cliniai
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Application code only — no .env, no tests, no docs (see docker/.dockerignore).
COPY --chown=cliniai:cliniai app ./app
COPY --chown=cliniai:cliniai evals ./evals

EXPOSE 8001

# Health check uses stdlib only (curl is not in slim images).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
