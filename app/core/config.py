from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

# Resolve .env relative to this file's location so it is found regardless
# of the working directory uvicorn is launched from.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    APP_NAME: str = "OCR Benchmark API"
    DEBUG: bool = False
    HF_TOKEN: Optional[str] = None
    QWEN_MODEL_SIZE: str = "7B"
    QWEN_LOAD_IN_4BIT: bool = False
    MMOCR_DET_MODEL: str = "DBNet"
    # Single checkpoint used for all languages (fallback if per-language not set)
    CALAMARI_CHECKPOINT: Optional[str] = None
    # Per-language checkpoints — take priority over CALAMARI_CHECKPOINT
    CALAMARI_CHECKPOINT_EN: Optional[str] = None
    CALAMARI_CHECKPOINT_AR: Optional[str] = None
    CALAMARI_CHECKPOINT_HI: Optional[str] = None
    KRAKEN_RECOGNITION_MODEL: Optional[str] = None

    # Which models to load on startup (comment out to disable)
    ENABLED_MODELS: List[str] = [
        "paddleocr_v4",       # Tier 2 — fast CPU/GPU, good baseline
        "easyocr",            # Tier 2 — easy multilingual
        "tesseract",          # Tier 2 — classic baseline
        # "mmocr_crnn",       # Tier 2 — MMOCR end-to-end OCR (English)
        # "mmocr_sar",        # Tier 2 — MMOCR end-to-end OCR (English)
        # "mmocr_nrtr",       # Tier 2 — MMOCR end-to-end OCR (English)
        # "keras_ocr",        # Tier 2 — Keras-OCR detector + recognizer (English)
        # "calamari",         # Tier 2 — requires CALAMARI_CHECKPOINT
        # "kraken",           # Tier 2 — requires KRAKEN_RECOGNITION_MODEL
        # "paddleocr_vl",     # Tier 1 — requires ~4GB VRAM
        # "got_ocr2",         # Tier 1 — requires ~8GB VRAM
        # "qwen25_vl",        # Tier 1 — requires ~16GB VRAM
        # "olmocr2",          # Tier 1 — requires ~16GB VRAM
    ]

    # Pure OCR engines only. These models do not use LLM/VLM decoding.
    OCR_WITHOUT_LLM_CAPABILITIES: List[str] = [
        "paddleocr_v4",
        "easyocr",
        "tesseract",
        "mmocr_crnn",
        "mmocr_sar",
        "mmocr_nrtr",
        "keras_ocr",
        "calamari",
        "kraken",
    ]

    # GPU settings
    USE_GPU: bool = False          # GPU for PyTorch-based models (qwen25_vl, olmocr2, got_ocr2)
    PADDLE_USE_GPU: bool = False   # GPU for PaddleOCR models — set False for Blackwell (SM 120) until PaddlePaddle adds support
    GPU_DEVICE_ID: int = 0

    # F2 pipeline debug visualisations
    # Set to a directory path to save stage images on every inference call.
    # Leave empty to disable (default — no overhead in production).
    PADDLE_DEBUG_OUTPUT_DIR: str = ""
    # Optional Arabic fallback in F2 mode: run PP-OCRv3 only when
    # quality heuristics indicate PP-OCRv5 output may be weak.
    PADDLE_ARABIC_V3_FALLBACK: bool = False
    # If true, always run both Arabic engines (v5 + v3) and pick best.
    PADDLE_ARABIC_RUN_BOTH_ENGINES: bool = False
    PADDLE_F2_FALLBACK_MIN_LINES: int = 4
    PADDLE_F2_FALLBACK_MIN_CHARS: int = 80
    PADDLE_F2_FALLBACK_MIN_AVG_CONF: float = 0.65
    PADDLE_F2_FALLBACK_MIN_ARABIC_RATIO: float = 0.60
    # Alt pass must beat primary by this relative margin to replace it.
    PADDLE_F2_FALLBACK_REPLACE_MARGIN: float = 0.03

    # Input pipeline: document ROI (reduces background clutter for det+rec).
    # Finds largest quadrilateral via edges → approxPoly → perspective warp.
    PADDLE_INPUT_ROI_WARP: bool = False
    # Contour area must be ≥ this fraction of image area to accept the quad.
    PADDLE_ROI_MIN_AREA_RATIO: float = 0.15
    # Extra border after warp (fraction of max side) for detector margin.
    PADDLE_ROI_PAD_RATIO: float = 0.02

    # Inference timeouts (seconds)
    MODEL_TIMEOUT: int = 60
    BENCHMARK_TIMEOUT: int = 300

    # File upload limits
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"]

    # Metrics
    ENABLE_PROMETHEUS: bool = False

    CORS_ORIGINS: List[str] = ["*"]

settings = Settings()
