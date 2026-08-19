# syntax=docker/dockerfile:1
# Universal AI Provider SDK (uai-sdk) — container image.
# Works with Podman (`podman build`) and Docker; push to Docker Hub with
# `podman push` / `docker push`.

# --- Build stage: build the wheel (poetry-core backend) ---
FROM python:3.12-slim AS builder

WORKDIR /src

COPY . .

RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip wheel --no-deps --wheel-dir /wheels .

# --- Runtime stage: install wheel, run as non-root ---
FROM python:3.12-slim

LABEL org.opencontainers.image.title="uai-sdk" \
      org.opencontainers.image.description="Universal AI Provider SDK for Python" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.source="https://github.com/GenAIwithMS/uai-sdk-python" \
      org.opencontainers.image.licenses="MIT"

# Non-root user; the CLI only prints to stdout, no filesystem writes needed.
RUN useradd --create-home --shell /usr/sbin/nologin uai

WORKDIR /home/uai

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

USER uai

ENTRYPOINT ["uai"]
CMD ["list-providers"]