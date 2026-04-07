"""
Kraken OCR wrapper.

This uses Kraken's legacy page segmentation and a user-supplied recognition model.
Kraken is a neural OCR engine, but it does not use LLM/VLM generation.
"""

import logging

from app.core.document import load_document_as_rgb_images
from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage

logger = logging.getLogger(__name__)


class KrakenModel(BaseOCRModel):
    name = "kraken"
    supported_languages = [SupportedLanguage.ENGLISH]
    tier = 2

    def __init__(self, recognition_model_path: str | None):
        self.recognition_model_path = recognition_model_path
        self._recognizer = None

    async def load(self) -> None:
        if not self.recognition_model_path:
            raise RuntimeError(
                "KRAKEN_RECOGNITION_MODEL is not configured. Point it to a Kraken recognition model."
            )

        from kraken.lib import models

        logger.info("[Kraken] Loading recognition model from %s", self.recognition_model_path)
        self._recognizer = models.load_any(self.recognition_model_path)

    async def unload(self) -> None:
        self._recognizer = None

    @staticmethod
    def _to_xyxy_bbox(line_or_boundary) -> list[int]:
        if not line_or_boundary:
            return [0, 0, 0, 0]

        if len(line_or_boundary) == 4 and not isinstance(line_or_boundary[0], (list, tuple)):
            return [int(float(v)) for v in line_or_boundary]

        xs = [int(float(point[0])) for point in line_or_boundary]
        ys = [int(float(point[1])) for point in line_or_boundary]
        return [min(xs), min(ys), max(xs), max(ys)]

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language not in self.supported_languages:
            return OCRResult.from_error(self.name, language.value, f"Language {language.value} not supported")

        if self._recognizer is None:
            return OCRResult.from_error(self.name, language.value, "Model is not loaded")

        try:
            from kraken import binarization, pageseg, rpred

            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)
            recognizer_fn = rpred.rpred if hasattr(rpred, "rpred") else rpred

            for image in pages:
                bw_image = binarization.nlbin(image.convert("L"))
                segmentation = pageseg.segment(bw_image)

                t0 = self._timer()
                records = list(recognizer_fn(self._recognizer, image, segmentation))
                total_elapsed += self._elapsed_ms(t0)

                lines = []
                for record in records:
                    text = str(getattr(record, "prediction", "")).strip()
                    if not text:
                        continue

                    confidences = list(getattr(record, "confidences", []) or [])
                    avg_conf = (
                        float(sum(confidences) / len(confidences))
                        if confidences
                        else 0.0
                    )

                    bbox_source = getattr(record, "line", None) or getattr(record, "boundary", None)
                    words.append(
                        OCRWord(
                            text=text,
                            confidence=avg_conf,
                            bbox=self._to_xyxy_bbox(bbox_source),
                        )
                    )
                    lines.append(text)

                page_texts.append("\n".join(lines))

            raw_text = "\n\n".join(text for text in page_texts if text)
            avg_conf = sum(word.confidence for word in words) / len(words) if words else 0.0

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=round(avg_conf, 4),
                metadata={
                    "page_count": len(pages),
                    "recognition_model": self.recognition_model_path,
                    "line_segmentation": "kraken.pageseg",
                },
            )
        except Exception as e:
            logger.exception("[Kraken] Inference error: %s", e)
            return OCRResult.from_error(self.name, language.value, str(e))
