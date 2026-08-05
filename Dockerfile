# syntax=docker/dockerfile:1
#
# Evaluation-friendly package for the audio-watermarking robustness demo.
#   docker build -t sok-audio-watermark-demo .
#   docker run -p 7860:7860 sok-audio-watermark-demo
# or simply run ./install.sh
#
# The image is CPU-only and self-contained: it bundles AudioSeal, WavMark,
# SilentCipher, the audiowmark CLI (built from source), and the in-repo
# FSVC / Patchwork / Norm-space methods, plus the Gradio demo UI.

# ── Stage 1: build the audiowmark CLI from source ────────────────────────────
FROM debian:bookworm-slim AS audiowmark-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential autoconf automake libtool pkg-config git ca-certificates \
        libfftw3-dev libsndfile1-dev libgcrypt20-dev libmpg123-dev \
        libzita-resampler-dev \
    && rm -rf /var/lib/apt/lists/*
ARG AUDIOWMARK_REF=0.6.5
RUN git clone --depth 1 --branch "${AUDIOWMARK_REF}" \
        https://github.com/swesterfeld/audiowmark.git /src/audiowmark \
    && cd /src/audiowmark \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install-strip

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libfftw3-single3 libgcrypt20 libmpg123-0 \
        libzita-resampler1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=audiowmark-build /usr/local/bin/audiowmark /usr/local/bin/audiowmark

WORKDIR /app

# Python dependencies first (cache-friendly).  torch is pinned to 2.0.0
# (CPU wheels) because silentcipher requires torch<=2.0.0; audioseal and
# wavmark are compatible with it.  g++ is needed only while pip builds the
# libsvm-official sdist (a visqol-python dependency, no prebuilt wheel); it
# is purged in the same layer, while libgomp1 (installed above) stays for
# the compiled extension's OpenMP runtime.
COPY demo/requirements.txt /app/demo/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.0.0 torchaudio==2.0.1 \
    && pip install -r /app/demo/requirements.txt \
    && apt-get purge -y g++ \
    && rm -rf /var/lib/apt/lists/*

# Isolated environments for the methods whose dependencies are incompatible
# with the main env (see requirements/README.md):
#   /opt/venvs/torch27 — Timbre + AWARE workers (torch 2.7, librosa 0.9.2)
#   /opt/venvs/tf212   — RobustDNN worker (TensorFlow 2.12, numpy<1.24)
# g++ is transient again: webrtcvad (AWARE) compiles a C extension.
COPY demo/requirements-torch27.txt demo/requirements-tf212.txt /app/demo/
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && python -m venv /opt/venvs/torch27 \
    && /opt/venvs/torch27/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.0 torchaudio==2.7.0 \
    && /opt/venvs/torch27/bin/pip install --no-cache-dir \
        -r /app/demo/requirements-torch27.txt \
    && python -m venv /opt/venvs/tf212 \
    && /opt/venvs/tf212/bin/pip install --no-cache-dir \
        -r /app/demo/requirements-tf212.txt \
    && apt-get purge -y g++ \
    && rm -rf /var/lib/apt/lists/*

# Bake model checkpoints into the image (best effort — the demo lazily
# re-downloads at runtime if a prefetch failed at build time).
COPY demo/prefetch_models.py /app/demo/prefetch_models.py
RUN python /app/demo/prefetch_models.py

# Project code (heavy data dirs are excluded via .dockerignore)
COPY . /app

ENV SOK_AUDIOWMARK_BIN=/usr/local/bin/audiowmark \
    SOK_TIMBRE_PYTHON=/opt/venvs/torch27/bin/python \
    SOK_AWARE_PYTHON=/opt/venvs/torch27/bin/python \
    SOK_DNN_PYTHON=/opt/venvs/tf212/bin/python \
    SOK_DEMO_TMPDIR=/tmp/sok_demo \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    NO_TORCH_COMPILE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('GRADIO_SERVER_PORT','7860'))" || exit 1

CMD ["python", "demo/app.py"]
