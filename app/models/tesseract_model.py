"""
Tesseract 5 LSTM — Tier 2 (baseline)
Install: sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-hin tesseract-ocr-pan
         pip install pytesseract
"""

import logging
import platform
from io import BytesIO
from PIL import Image

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)

LANG_MAP = {
    SupportedLanguage.ENGLISH: "eng",
    SupportedLanguage.ARABIC: "ara",
    SupportedLanguage.HINDI: "hin",
    SupportedLanguage.PUNJABI: "pan",  
}


class TesseractModel(BaseOCRModel):
    name = "tesseract"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 2

    def supports_all_languages(self) -> bool:
        return True

    async def load(self) -> None:
        import pytesseract
        # Verify tesseract binary is accessible
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"[Tesseract] Version: {version}")
        except Exception as e:
            os_name = platform.system()
            if os_name == "Darwin":
                hint = "Install with: brew install tesseract tesseract-lang"
            else:
                hint = "Install with: sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-hin tesseract-ocr-pan"
            raise RuntimeError(f"tesseract is not installed or not in PATH. {hint}") from e

    async def unload(self) -> None:
        pass  # No persistent state

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        import pytesseract

        if language == SupportedLanguage.ALL:
            tess_lang = "+".join(LANG_MAP[lang] for lang in self.supported_languages)
        else:
            tess_lang = LANG_MAP.get(language)
        if not tess_lang:
            return OCRResult.from_error(self.name, language.value, f"Language {language} not supported")

        try:
            words = []
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                t0 = self._timer()

                data = pytesseract.image_to_data(
                    image,
                    lang=tess_lang,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 1 --psm 6",
                )
                total_elapsed += self._elapsed_ms(t0)

                lines = []
                n = len(data["text"])
                for i in range(n):
                    text = data["text"][i].strip()
                    conf = int(data["conf"][i])
                    if text and conf > 0:
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        words.append(OCRWord(
                            text=text,
                            confidence=conf / 100.0,
                            bbox=[x, y, x + w, y + h],
                        ))
                        lines.append(text)
                page_texts.append(" ".join(lines))

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
                    "tesseract_lang": tess_lang,
                },
            )

        except Exception as e:
            logger.exception(f"[Tesseract] Inference error: {e}")
            return OCRResult.from_error(self.name, language.value, str(e))
