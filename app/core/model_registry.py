"""
Model Registry — central manager for all OCR model instances.
Handles lazy loading, GPU allocation, and graceful shutdown.
"""

import logging
from typing import Dict, List, Optional
from app.models.base import BaseOCRModel
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.loaded_models: Dict[str, BaseOCRModel] = {}
        self.failed_models: Dict[str, str] = {}

    async def initialize(self, model_names: List[str]):
        """Load all enabled models at startup."""
        self.loaded_models.clear()
        self.failed_models.clear()
        for name in model_names:
            try:
                model = self._build_model(name)
                if not model:
                    self.failed_models[name] = "Unknown model name or optional dependency missing"
                    logger.error(f"[Registry] Failed to load {name}: {self.failed_models[name]}")
                    continue

                await model.load()
                self.loaded_models[name] = model
                logger.info(f"[Registry] Loaded: {name}")
            except Exception as e:
                self.failed_models[name] = str(e)
                logger.error(f"[Registry] Failed to load {name}: {e}")

        loaded_names = sorted(self.loaded_models.keys())
        failed_names = sorted(self.failed_models.keys())
        requested_count = len(model_names)
        loaded_count = len(loaded_names)
        failed_count = len(failed_names)

        if failed_count == 0:
            logger.info(
                f"[Registry] Startup complete: all {loaded_count}/{requested_count} models loaded successfully: "
                f"{', '.join(loaded_names)}"
            )
            return

        logger.warning(
            f"[Registry] Startup complete with partial success: loaded {loaded_count}/{requested_count}, failed {failed_count}"
        )
        logger.warning(
            f"[Registry] Loaded models: {', '.join(loaded_names) if loaded_names else 'none'}"
        )
        for name in failed_names:
            logger.error(f"[Registry] Startup failure detail | model={name} | reason={self.failed_models[name]}")

    def _build_model(self, name: str) -> Optional["BaseOCRModel"]:
        """Factory: maps model name → implementation class."""
        if name == "paddleocr_v4":
            from app.models.paddleocr_v4 import PaddleOCRv4Model

            return PaddleOCRv4Model(
                use_gpu=settings.PADDLE_USE_GPU,
                debug_output_dir=settings.PADDLE_DEBUG_OUTPUT_DIR or None,
                enable_arabic_v3_fallback=settings.PADDLE_ARABIC_V3_FALLBACK,
                always_run_both_arabic_engines=settings.PADDLE_ARABIC_RUN_BOTH_ENGINES,
                fallback_min_lines=settings.PADDLE_F2_FALLBACK_MIN_LINES,
                fallback_min_chars=settings.PADDLE_F2_FALLBACK_MIN_CHARS,
                fallback_min_avg_conf=settings.PADDLE_F2_FALLBACK_MIN_AVG_CONF,
                fallback_min_ar_ratio=settings.PADDLE_F2_FALLBACK_MIN_ARABIC_RATIO,
                fallback_replace_margin=settings.PADDLE_F2_FALLBACK_REPLACE_MARGIN,
                input_roi_warp=settings.PADDLE_INPUT_ROI_WARP,
                roi_min_area_ratio=settings.PADDLE_ROI_MIN_AREA_RATIO,
                roi_pad_ratio=settings.PADDLE_ROI_PAD_RATIO,
            )

        if name == "easyocr":
            from app.models.easyocr_model import EasyOCRModel

            return EasyOCRModel(use_gpu=settings.USE_GPU)

        if name == "tesseract":
            from app.models.tesseract_model import TesseractModel

            return TesseractModel()

        if name == "mmocr_crnn":
            from app.models.mmocr_model import MMOCRModel

            return MMOCRModel(
                name="mmocr_crnn",
                rec_model="CRNN",
                det_model=settings.MMOCR_DET_MODEL,
                use_gpu=settings.USE_GPU,
            )

        if name == "mmocr_sar":
            from app.models.mmocr_model import MMOCRModel

            return MMOCRModel(
                name="mmocr_sar",
                rec_model="SAR",
                det_model=settings.MMOCR_DET_MODEL,
                use_gpu=settings.USE_GPU,
            )

        if name == "mmocr_nrtr":
            from app.models.mmocr_model import MMOCRModel

            return MMOCRModel(
                name="mmocr_nrtr",
                rec_model="NRTR",
                det_model=settings.MMOCR_DET_MODEL,
                use_gpu=settings.USE_GPU,
            )

        if name == "keras_ocr":
            from app.models.keras_ocr_model import KerasOCRModel

            return KerasOCRModel()

        if name == "calamari":
            from app.models.calamari_model import CalamariModel

            return CalamariModel(
                checkpoint_path=settings.CALAMARI_CHECKPOINT,
                checkpoint_en=settings.CALAMARI_CHECKPOINT_EN,
                checkpoint_ar=settings.CALAMARI_CHECKPOINT_AR,
                checkpoint_hi=settings.CALAMARI_CHECKPOINT_HI,
            )

        if name == "kraken":
            from app.models.kraken_model import KrakenModel

            return KrakenModel(recognition_model_path=settings.KRAKEN_RECOGNITION_MODEL)

        # Tier 1 VLMs — import only when explicitly requested
        if name == "paddleocr_vl":
            from app.models.paddleocr_vl import PaddleOCRVLModel

            return PaddleOCRVLModel(use_gpu=settings.PADDLE_USE_GPU)

        if name == "got_ocr2":
            from app.models.got_ocr2 import GOTOcr2Model

            return GOTOcr2Model(use_gpu=settings.USE_GPU)

        if name == "qwen25_vl":
            from app.models.qwen25_vl import Qwen25VLModel

            return Qwen25VLModel(
                use_gpu=settings.USE_GPU,
                model_size=settings.QWEN_MODEL_SIZE,
                load_in_4bit=settings.QWEN_LOAD_IN_4BIT,
            )

        if name == "olmocr2":
            from app.models.olmocr2 import OlmOCR2Model

            return OlmOCR2Model(use_gpu=settings.USE_GPU)

        logger.warning(f"[Registry] Unknown model: {name}")
        return None

    def get(self, name: str) -> Optional["BaseOCRModel"]:
        return self.loaded_models.get(name)

    def all(self) -> Dict[str, "BaseOCRModel"]:
        return self.loaded_models

    async def shutdown(self):
        for name, model in self.loaded_models.items():
            try:
                await model.unload()
                logger.info(f"[Registry] Unloaded: {name}")
            except Exception as e:
                logger.warning(f"[Registry] Error unloading {name}: {e}")
