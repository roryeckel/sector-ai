# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# GIFs are decoded into still frames; without ffmpeg the bot keeps the still
# thumbnail and its markers say motion was not sampled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pyproject.toml, README.md and LICENSE are the packaging inputs; src holds the
# code. Copying them before installing keeps the dependency layer cacheable via
# the BuildKit mount below.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . \
    && rm -rf build src

RUN useradd --system --uid 10001 --create-home sector

USER sector

# Configuration is environment-only, so the image needs no volume and no
# config file: `docker run -e TELEGRAM_BOT_TOKEN=... -e OLLAMA_URL=... image`.
# For an OpenAI-compatible backend, set SECTOR_AI_LLM_PROVIDER=openai and
# SECTOR_AI_OPENAI_BASE_URL instead.
ENTRYPOINT ["sector-ai"]
