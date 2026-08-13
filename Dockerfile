# syntax=docker/dockerfile:1.7

FROM python:3.10-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/core
COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /opt/wheels ".[web]"


FROM python:3.10-slim-bookworm AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglfw3 \
        libglx0 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /tmp/core-runs \
    && chown user:user /tmp/core-runs

COPY --from=builder /opt/wheels /tmp/wheels
RUN python -m pip install \
        --no-cache-dir \
        --no-index \
        --find-links=/tmp/wheels \
        "core-retarget[web]==0.1.0" \
    && rm -rf /tmp/wheels

# Keep the browser quick-start motions outside the Python wheel while making
# them available to the Space at the WebConfig default path.
COPY --chown=user:user examples/LICENSE.md /home/user/app/examples/LICENSE.md
COPY --chown=user:user examples/motions/kimodo/SOURCE.yaml /home/user/app/examples/motions/kimodo/SOURCE.yaml
COPY --chown=user:user examples/motions/kimodo/soma_rp_v11/foot_walk_stop.npz /home/user/app/examples/motions/kimodo/soma_rp_v11/foot_walk_stop.npz
COPY --chown=user:user examples/motions/gem-x/SOURCE.yaml /home/user/app/examples/motions/gem-x/SOURCE.yaml
COPY --chown=user:user examples/motions/gem-x/scurry_walk.pt /home/user/app/examples/motions/gem-x/scurry_walk.pt

USER user
WORKDIR /home/user/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["core-retarget", "serve", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--runs-dir", "/tmp/core-runs", \
     "--max-upload-mb", "32", \
     "--max-frames", "1800", \
     "--max-active-jobs", "3", \
     "--result-ttl-minutes", "30", \
     "--max-video-width", "1280", \
     "--max-video-height", "720", \
     "--disable-stage-archives"]
