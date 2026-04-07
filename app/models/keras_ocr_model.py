"""
Keras-OCR wrapper.

Uses the public keras-ocr detector + recognizer pipeline, which is a traditional OCR
stack and does not include LLM/VLM generation.
"""

import logging
import os

# keras-ocr 0.9.3 uses the Keras 2 API. TensorFlow ≥ 2.16 ships Keras 3 by
# default, which removed the `weights` constructor argument that keras-ocr
# depends on. Setting TF_USE_LEGACY_KERAS=1 before any TF/Keras import makes
# tensorflow.keras route to the legacy Keras 2 implementation (tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

from app.core.document import load_document_as_rgb_images
from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage

logger = logging.getLogger(__name__)


class KerasOCRModel(BaseOCRModel):
    name = "keras_ocr"
    supported_languages = [SupportedLanguage.ENGLISH]
    tier = 2

    def __init__(self):
        self._pipeline = None

    @staticmethod
    def _ensure_numpy_sctypes_compat() -> None:
        """
        imgaug (used by keras-ocr) still accesses np.sctypes in some releases.
        NumPy 2 removed this attribute, so we provide a minimal compatibility map.
        """
        if hasattr(np, "sctypes"):
            return

        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.str_, np.bytes_],
        }

    async def load(self) -> None:
        self._ensure_numpy_sctypes_compat()
        import keras_ocr

        logger.info("[Keras-OCR] Loading pretrained pipeline")
        self._pipeline = keras_ocr.pipeline.Pipeline()

    async def unload(self) -> None:
        self._pipeline = None

    @staticmethod
    def _to_xyxy_bbox(box) -> list[int]:
        if box is None or len(box) == 0:
            return [0, 0, 0, 0]
        xs = [int(float(point[0])) for point in box]
        ys = [int(float(point[1])) for point in box]
        return [min(xs), min(ys), max(xs), max(ys)]

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language not in self.supported_languages:
            return OCRResult.from_error(self.name, language.value, f"Language {language.value} not supported")

        if self._pipeline is None:
            return OCRResult.from_error(self.name, language.value, "Model is not loaded")

        try:
            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                image_array = np.array(image)

                t0 = self._timer()
                prediction_groups = self._pipeline.recognize([image_array])
                total_elapsed += self._elapsed_ms(t0)

                lines = []
                for text, box in (prediction_groups[0] if prediction_groups else []):
                    text = str(text).strip()
                    if not text:
                        continue
                    words.append(
                        OCRWord(
                            text=text,
                            confidence=0.0,
                            bbox=self._to_xyxy_bbox(box),
                        )
                    )
                    lines.append(text)

                page_texts.append("\n".join(lines))

            raw_text = "\n\n".join(text for text in page_texts if text)

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=0.0,
                metadata={
                    "page_count": len(pages),
                    "confidence_available": False,
                },
            )
        except Exception as e:
            logger.exception("[Keras-OCR] Inference error: %s", e)
            return OCRResult.from_error(self.name, language.value, str(e))
