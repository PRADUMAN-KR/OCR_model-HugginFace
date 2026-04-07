"""
GOT-OCR 2.0 — Tier 1 VLM
Model: stepfun-ai/GOT-OCR2_0 (~8GB VRAM)
Install: pip install transformers torch torchvision
"""

import logging
import tempfile
import os
from io import BytesIO
from PIL import Image

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.config import settings
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)


class GOTOcr2Model(BaseOCRModel):
    name = "got_ocr2"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 1

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._model = None
        self._tokenizer = None

    async def load(self) -> None:
        from transformers import AutoTokenizer, AutoModel
        import torch

        model_id = "stepfun-ai/GOT-OCR2_0"
        device = "cuda" if self.use_gpu else "cpu"
        hf_kwargs = {"token": settings.HF_TOKEN} if settings.HF_TOKEN else {}

        logger.info(f"[GOT-OCR2] Loading model on {device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            **hf_kwargs,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device,
            use_safetensors=True,
            pad_token_id=self._tokenizer.eos_token_id,
            **hf_kwargs,
        )
        self._model = self._model.eval()
        logger.info("[GOT-OCR2] Model loaded.")

    async def unload(self) -> None:
        import torch
        del self._model
        del self._tokenizer
        if self.use_gpu:
            torch.cuda.empty_cache()

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if not self._model:
            return OCRResult.from_error(self.name, language.value, "Model not loaded")

        try:
            page_texts = []
            total_elapsed = 0.0
            pages = load_document_as_rgb_images(image_bytes)

            for image in pages:
                # GOT-OCR2 requires an image file path (not bytes) via its API
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path)

                try:
                    t0 = self._timer()
                    result_text = self._model.chat(self._tokenizer, tmp_path, ocr_type="ocr")
                    total_elapsed += self._elapsed_ms(t0)
                    page_texts.append(result_text.strip())
                finally:
                    os.unlink(tmp_path)

            raw_text = "\n\n".join(text for text in page_texts if text)
            words = [OCRWord(text=w, confidence=1.0) for w in raw_text.split() if w]

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=1.0,  # GOT-OCR2 doesn't expose per-word confidence
                metadata={"note": "Confidence is model-level, not per-word", "page_count": len(pages)},
            )

        except Exception as e:
            logger.exception(f"[GOT-OCR2] Inference error: {e}")
            return OCRResult.from_error(self.name, language.value, str(e))
