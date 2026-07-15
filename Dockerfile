# syntax=docker/dockerfile:1

# ---- Builder ----
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv sync --frozen --no-dev

# ---- Runtime ----
FROM python:3.11-slim AS runtime

# Matches the common default host UID/GID (1000:1000) so bind-mounted volumes
# (see docker-compose.yml) stay writable without a runtime --user override.
# Override at build time (--build-arg APP_UID=...) if the host user differs.
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid ${APP_GID} app \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --home-dir /app app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv .venv
COPY --chown=app:app src/ src/

ENV PATH="/app/.venv/bin:${PATH}"

USER app

EXPOSE 8000

CMD ["uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
