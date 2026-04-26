# Multi-stage build keeps the runtime image small. The build stage
# compiles wheels and any C extensions; the runtime stage copies just
# the installed packages and the app code.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential needed for any source-only wheels (older xgboost
# versions, occasionally pyarrow on uncommon archs). Most deps will
# pull prebuilt wheels and skip compilation entirely.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DATA_DIR=/data

# libgomp is required by xgboost; everything else runs on glibc alone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root UID. Unraid's default "nobody" user is 99:100;
# we use the same so bind-mounted /data has expected ownership.
# Note: GID 100 already exists in debian:slim as "users", so we skip
# group creation if the GID is taken and just attach our user to it.
ARG PUID=99
ARG PGID=100
RUN if ! getent group ${PGID} >/dev/null; then groupadd -g ${PGID} app; fi \
 && useradd -u ${PUID} -g ${PGID} -m -s /bin/bash app

COPY --from=builder /install /usr/local

WORKDIR /app
# Numeric IDs in --chown so we don't depend on the group's *name*
# (which is "users" when GID 100 already exists, "app" otherwise).
COPY --chown=99:100 app.py ./
COPY --chown=99:100 src ./src
COPY --chown=99:100 docs ./docs
COPY --chown=99:100 scripts ./scripts

# Persistent volume for raw exports, processed parquets, and the
# trained model. Bind-mount this on Unraid (e.g. to
# /mnt/user/appdata/t1d-pipeline) so re-running the container doesn't
# wipe your work.
RUN mkdir -p /data/raw /data/processed /data/models /data/plots \
 && chown -R ${PUID}:${PGID} /data /app

USER app

# Symlink so the existing relative `data/...` paths in the code still work
# when the persistent volume is mounted at /data.
RUN ln -s /data /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
