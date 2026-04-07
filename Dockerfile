# ── Stage 1: Base ─────────────────────────────────────────────────────────────
# CUDA 12.4 + cuDNN 9 runtime on Ubuntu 22.04 — required for paddlepaddle-gpu 3.x
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Python 3.11 + system deps: Tesseract + language packs + OpenCV headless libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-hin \
    tesseract-ocr-pan \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Dependencies ──────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip

# Step 1: install all deps from PyPI — paddleocr will pull paddlepaddle-gpu 2.x here
RUN pip install -r requirements.txt

# Step 2: force-upgrade paddle to 3.x GPU build from PaddlePaddle's own index.
# PyPI only has paddlepaddle-gpu up to 2.6.2; 3.x is on PaddlePaddle's index only.
# cu123 is forward-compatible with CUDA 12.4.
RUN pip install "paddlepaddle-gpu==3.1.0" \
    --force-reinstall --no-deps \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# ── Stage 3: App ───────────────────────────────────────────────────────────────
FROM deps AS app

COPY . .

# Pre-download PaddleOCR models at build time (avoids first-request latency)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', ocr_version='PP-OCRv4')" || true
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ar', ocr_version='PP-OCRv5')" || true
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='hi', ocr_version='PP-OCRv5')" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
