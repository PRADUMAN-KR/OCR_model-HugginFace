"""
MMOCR end-to-end OCR wrappers without language-model decoding.

These wrappers expose pure OCR pipelines using MMOCR's detector + recognizer stack.
They are configured as separate runtime model names so CRNN, SAR, and NRTR can be
benchmarked independently.
"""

import logging

import numpy as np

from app.core.document import load_document_as_rgb_images
from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage

logger = logging.getLogger(__name__)


class MMOCRModel(BaseOCRModel):
    supported_languages = [SupportedLanguage.ENGLISH]
    tier = 2

    def __init__(
        self,
        *,
        name: str,
        rec_model: str,
        det_model: str = "DBNet",
        use_gpu: bool = False,
    ):
        self.name = name
        self.rec_model = rec_model
        self.det_model = det_model
        self.use_gpu = use_gpu
        self._inferencer = None

    async def load(self) -> None:
        from mmocr.apis import MMOCRInferencer

        device = "cuda:0" if self.use_gpu else "cpu"
        logger.info(
            "[MMOCR] Loading %s with det=%s rec=%s on %s",
            self.name,
            self.det_model,
            self.rec_model,
            device,
        )
        self._inferencer = MMOCRInferencer(
            det=self.det_model,
            rec=self.rec_model,
            device=device,
        )

    async def unload(self) -> None:
        self._inferencer = None

    @staticmethod
    def _to_xyxy_bbox(poly_or_box) -> list[int]:
        if poly_or_box is None:
            return [0, 0, 0, 0]

        if len(poly_or_box) == 4 and not isinstance(poly_or_box[0], (list, tuple)):
            return [int(float(v)) for v in poly_or_box]

        points = []
        if poly_or_box and not isinstance(poly_or_box[0], (list, tuple)):
            for idx in range(0, len(poly_or_box), 2):
                if idx + 1 < len(poly_or_box):
                    points.append((poly_or_box[idx], poly_or_box[idx + 1]))
        else:
            points = poly_or_box

        xs = [int(float(point[0])) for point in points]
        ys = [int(float(point[1])) for point in points]
        return [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language not in self.supported_languages:
            return OCRResult.from_error(self.name, language.value, f"Language {language.value} not supported")

        if self._inferencer is None:
            return OCRResult.from_error(self.name, language.value, "Model is not loaded")

        try:
            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                image_bgr = np.array(image)[:, :, ::-1].copy()

                t0 = self._timer()
                result = self._inferencer(image_bgr, progress_bar=False)
                total_elapsed += self._elapsed_ms(t0)

                prediction = ((result or {}).get("predictions") or [{}])[0] or {}
                texts = prediction.get("rec_texts") or []
                scores = prediction.get("rec_scores") or []
                bboxes = prediction.get("det_bboxes") or []
                polys = prediction.get("det_polygons") or []

                lines = []
                total = max(len(texts), len(scores), len(bboxes), len(polys))
                for idx in range(total):
                    text = str(texts[idx]).strip() if idx < len(texts) else ""
                    if not text:
                        continue

                    conf = float(scores[idx]) if idx < len(scores) else 0.0
                    bbox_source = bboxes[idx] if idx < len(bboxes) else (polys[idx] if idx < len(polys) else None)
                    words.append(
                        OCRWord(
                            text=text,
                            confidence=conf,
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
                    "det_model": self.det_model,
                    "rec_model": self.rec_model,
                },
            )
        except Exception as e:
            logger.exception("[MMOCR] Inference error for %s: %s", self.name, e)
            return OCRResult.from_error(self.name, language.value, str(e))
