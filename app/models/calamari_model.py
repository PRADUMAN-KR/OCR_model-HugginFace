"""
Calamari OCR wrapper.

Calamari is a script-agnostic line-level OCR recognizer. It supports any script
(Latin, Arabic, Devanagari, etc.) provided an appropriate trained checkpoint.

This wrapper performs simple page-to-line segmentation with Kraken's legacy
segmenter, then recognizes each cropped line with the per-language checkpoint.

Checkpoint config (set in .env):
  - CALAMARI_CHECKPOINT_EN  — English checkpoint
  - CALAMARI_CHECKPOINT_AR  — Arabic checkpoint
  - CALAMARI_CHECKPOINT_HI  — Hindi (Devanagari) checkpoint
  - CALAMARI_CHECKPOINT     — Universal fallback when a per-language key is unset

Recommended checkpoints:
  English: Calamari-OCR/calamari_models → uw3-modern-english/0.ckpt.json
  Arabic:  use a valid public/community Arabic Calamari checkpoint
  Hindi:   community Devanagari models or self-trained
"""

import logging

import numpy as np

from app.core.document import load_document_as_rgb_images
from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage

logger = logging.getLogger(__name__)

# Maps language → config key suffix used in Settings
_LANG_CHECKPOINT_ATTR = {
    SupportedLanguage.ENGLISH: "CALAMARI_CHECKPOINT_EN",
    SupportedLanguage.ARABIC:  "CALAMARI_CHECKPOINT_AR",
    SupportedLanguage.HINDI:   "CALAMARI_CHECKPOINT_HI",
}


class CalamariModel(BaseOCRModel):
    name = "calamari"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
    ]
    tier = 2

    @staticmethod
    def _autodetect_english_checkpoint() -> str | None:
        """Scan a set of well-known local paths for a Calamari English checkpoint.

        Calamari models are not on PyPI — they must be cloned from:
          git clone https://github.com/Calamari-OCR/calamari_models ~/checkpoints/calamari_models
        Then set CALAMARI_CHECKPOINT_EN in .env, or drop the clone at one of
        the paths below for zero-config auto-detection.
        """
        import os
        candidates = [
            # Most common manual clone locations
            os.path.expanduser("~/checkpoints/calamari_models/uw3-modern-english/0.ckpt.json"),
            os.path.expanduser("~/calamari_models/uw3-modern-english/0.ckpt.json"),
            "/opt/calamari_models/uw3-modern-english/0.ckpt.json",
            # Fallback: any .ckpt.json directly in checkpoints dir
            os.path.expanduser("~/checkpoints/calamari_models/0.ckpt.json"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def __init__(self, checkpoint_path: str | None,
                 checkpoint_en: str | None = None,
                 checkpoint_ar: str | None = None,
                 checkpoint_hi: str | None = None):
        # Per-language checkpoints; fall back to universal checkpoint_path,
        # then auto-detect English from the calamari_models package.
        self._checkpoint_map: dict[SupportedLanguage, str] = {}
        en_resolved = (
            checkpoint_en
            or checkpoint_path
            or self._autodetect_english_checkpoint()
        )
        for lang, value in [
            (SupportedLanguage.ENGLISH, en_resolved),
            (SupportedLanguage.ARABIC,  checkpoint_ar or checkpoint_path),
            (SupportedLanguage.HINDI,   checkpoint_hi or checkpoint_path),
        ]:
            if value:
                self._checkpoint_map[lang] = value

        self._predictors: dict[SupportedLanguage, object] = {}

    async def load(self) -> None:
        if not self._checkpoint_map:
            raise RuntimeError(
                "No Calamari checkpoints configured. Set CALAMARI_CHECKPOINT "
                "(universal) or CALAMARI_CHECKPOINT_EN / _AR / _HI in .env."
            )

        from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams

        for lang, ckpt in self._checkpoint_map.items():
            logger.info("[Calamari] Loading %s checkpoint from %s", lang.value, ckpt)
            self._predictors[lang] = Predictor.from_checkpoint(
                params=PredictorParams(),
                checkpoint=ckpt,
            )

        # Restrict supported_languages to what actually has a checkpoint
        self.__class__.supported_languages = list(self._predictors.keys())
        logger.info(
            "[Calamari] Ready for languages: %s",
            [l.value for l in self._predictors],
        )

    async def unload(self) -> None:
        self._predictors.clear()

    @staticmethod
    def _segment_lines(image):
        from kraken import binarization, pageseg

        bw_image = binarization.nlbin(image)
        segmentation = pageseg.segment(bw_image)
        return segmentation.get("boxes") or []

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language not in self._predictors:
            configured = [l.value for l in self._predictors]
            return OCRResult.from_error(
                self.name,
                language.value,
                f"Language '{language.value}' not supported — "
                f"no checkpoint configured. Configured: {configured}",
            )

        predictor = self._predictors[language]

        try:
            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                gray = image.convert("L")
                line_boxes = self._segment_lines(gray)
                line_images = []
                kept_boxes = []

                for box in line_boxes:
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = [int(float(v)) for v in box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = gray.crop((x1, y1, x2, y2))
                    line_images.append(np.array(crop))
                    kept_boxes.append([x1, y1, x2, y2])

                if not line_images:
                    page_texts.append("")
                    continue

                t0 = self._timer()
                predictions = list(predictor.predict_raw(line_images))
                total_elapsed += self._elapsed_ms(t0)

                lines = []
                for box, sample in zip(kept_boxes, predictions):
                    prediction = sample.outputs
                    text = str(prediction.sentence).strip()
                    if not text:
                        continue

                    confidences = getattr(prediction, "char_confidences", None) or []
                    avg_conf = (
                        float(sum(confidences) / len(confidences))
                        if confidences
                        else 0.0
                    )
                    words.append(OCRWord(text=text, confidence=avg_conf, bbox=box))
                    lines.append(text)

                page_texts.append("\n".join(lines))

            raw_text = "\n\n".join(text for text in page_texts if text)
            avg_conf = sum(w.confidence for w in words) / len(words) if words else 0.0

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=round(avg_conf, 4),
                metadata={
                    "page_count": len(pages),
                    "checkpoint": self._checkpoint_map[language],
                    "line_segmentation": "kraken.pageseg",
                },
            )
        except Exception as e:
            logger.exception("[Calamari] Inference error: %s", e)
            return OCRResult.from_error(self.name, language.value, str(e))
