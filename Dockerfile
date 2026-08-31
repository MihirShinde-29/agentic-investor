# syntax=docker/dockerfile:1.7

# Stage 1: build the Vite dashboard bundle.
FROM node:20-slim AS frontend
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build


# Stage 2: install Python deps with uv into a project venv.
FROM python:3.12-slim AS pydeps
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


# Stage 3: runtime image with source + built frontend + prewarmed venv.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY --from=pydeps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev
COPY --from=frontend /app/dashboard/dist /app/dashboard/dist
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["agentic-investor", "paper-loop", \
     "--auto", "--amount", "50000", "--universe", "sp500_large", \
     "--top-n", "8", "--interval", "5m", "--regen-mode", "event", \
     "--serve-dashboard", "--dashboard-port", "8000"]
