# Serve-only image: the base dependencies, no `scan` extra.
#
# The models are ~2 GB of torch, insightface and friends, and the target is a
# Celeron J4125 with 7.6 GB of RAM that also runs the house's DNS. Everything
# the web UI does apart from drawing a face by hand works without them —
# clustering included, since that reads embeddings already in the database.
# The Mac keeps the models and does the scanning.
#
# Build for the server from an Apple Silicon machine:
#   docker buildx build --platform linux/amd64 -t ritrova:latest --load .

FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer survives every
# change that does not touch them.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.14-slim

# opencv-python-headless still wants libGL's companions for video frames.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# UID/GID must match the owner of the photo tree on the host, or a read-only
# bind mount of it is unreadable. Override at build time to suit the server.
ARG RITROVA_UID=1000
ARG RITROVA_GID=1000
RUN groupadd --gid "${RITROVA_GID}" ritrova \
 && useradd --uid "${RITROVA_UID}" --gid "${RITROVA_GID}" --create-home ritrova

WORKDIR /app
COPY --from=builder --chown=ritrova:ritrova /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    RITROVA_DATA_DIR=/data \
    RITROVA_PHOTOS_DIR=/photos

# D15: thumbnails and frame caches are written here, so it must be a writable
# volume. Without one the app starts, serves pages, and fails on every image.
RUN mkdir -p /data && chown ritrova:ritrova /data
VOLUME ["/data"]

USER ritrova
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=4).status==200 else 1)"

# Binds to every interface inside the container; what the host actually
# exposes is the published port's business, not this image's.
CMD ["sh", "-c", "ritrova --data-dir \"$RITROVA_DATA_DIR\" --photos-dir \"$RITROVA_PHOTOS_DIR\" serve --host 0.0.0.0 --port 8787"]
