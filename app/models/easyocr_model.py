"""
EasyOCR — Tier 2
Supports: English, Arabic, Hindi, Punjabi
Install: pip install easyocr
"""

import logging
from io import BytesIO
import numpy as np
from PIL import Image

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)

# EasyOCR uses ISO codes but some scripts need multiple codes
LANG_MAP = {
    SupportedLanguage.ENGLISH: ["en"],
    SupportedLanguage.ARABIC: ["ar"],
    SupportedLanguage.HINDI: ["hi"],
    SupportedLanguage.PUNJABI: ["pa"],  # Punjabi Gurmukhi
}


class EasyOCRModel(BaseOCRModel):
    name = "easyocr"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 2

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._readers = {}

    def supports_all_languages(self) -> bool:
        return True

    async def load(self) -> None:
        import easyocr
        for lang_enum, lang_codes in LANG_MAP.items():
            logger.info(f"[EasyOCR] Loading reader for {lang_enum.value}")
            try:
                self._readers[lang_enum] = easyocr.Reader(
                    lang_codes,
                    gpu=self.use_gpu,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(
                    f"[EasyOCR] Skipping unsupported language {lang_enum.value} "
                    f"(codes={lang_codes}): {e}"
                )
        combined_codes = []
        for lang_codes in LANG_MAP.values():
            for code in lang_codes:
                if code not in combined_codes:
                    combined_codes.append(code)
        try:
            logger.info("[EasyOCR] Loading combined multilingual reader")
            self._readers[SupportedLanguage.ALL] = easyocr.Reader(
                combined_codes,
                gpu=self.use_gpu,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"[EasyOCR] Unable to load combined multilingual reader: {e}")
        logger.info("[EasyOCR] All readers loaded.")

    async def unload(self) -> None:
        self._readers.clear()

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        reader = self._readers.get(language)
        if not reader:
            return OCRResult.from_error(self.name, language.value, f"Language {language} not loaded")

        try:
            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                img_array = np.array(image)

                t0 = self._timer()
                result = reader.readtext(img_array, detail=1)
                total_elapsed += self._elapsed_ms(t0)

                lines = []
                for (bbox, text, conf) in result:
                    flat_bbox = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])]
                    words.append(OCRWord(text=text, confidence=float(conf), bbox=flat_bbox))
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
                metadata={"page_count": len(pages)},
            )

        except Exception as e:
            logger.exception(f"[EasyOCR] Inference error: {e}")
            return OCRResult.from_error(self.name, language.value, str(e))
