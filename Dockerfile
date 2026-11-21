# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

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
ENTRYPOINT ["sector-ai"]
