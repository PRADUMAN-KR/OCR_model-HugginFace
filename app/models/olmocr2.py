"""
olmOCR-2 — Tier 1 VLM
Model: allenai/olmOCR-2-0724  (~16GB VRAM, built on Qwen2.5-VL-7B)
Best-in-class for Arabic based on independent benchmarks.
Install: pip install transformers torch
"""

import logging
import base64
import importlib.util
from io import BytesIO
from PIL import Image

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.config import settings
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)

LANG_PROMPTS = {
    SupportedLanguage.ENGLISH: "Extract all text from this image. Output only the raw text.",
    SupportedLanguage.ARABIC: "استخرج كل النص من هذه الصورة. أخرج النص الخام فقط.",
    SupportedLanguage.HINDI: "इस छवि से सभी पाठ निकालें। केवल कच्चा पाठ आउटपुट करें।",
    SupportedLanguage.PUNJABI: "ਇਸ ਚਿੱਤਰ ਵਿੱਚੋਂ ਸਾਰਾ ਟੈਕਸਟ ਕੱਢੋ। ਸਿਰਫ਼ ਕੱਚਾ ਟੈਕਸਟ ਆਉਟਪੁੱਟ ਕਰੋ।",
}


class OlmOCR2Model(BaseOCRModel):
    name = "olmocr2"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 1

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.model_id = "allenai/olmOCR-2-7B-1025"
        self._model = None
        self._processor = None

    async def load(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  
        import torch

        device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"[olmOCR-2] Loading {self.model_id} on {device}...")
        hf_kwargs = {"token": settings.HF_TOKEN} if settings.HF_TOKEN else {}
        has_accelerate = importlib.util.find_spec("accelerate") is not None

        self._processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs)
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.use_gpu else torch.float32,
            **hf_kwargs,
        }
        if has_accelerate:
            model_kwargs["device_map"] = device
        elif self.use_gpu:
            logger.warning("[olmOCR-2] accelerate not found; loading without device_map and moving model to cuda.")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.use_gpu and not has_accelerate:
            self._model = self._model.to(device)
        self._model = self._model.eval()
        logger.info("[olmOCR-2] Loaded.")

    async def unload(self) -> None:
        import torch
        del self._model
        del self._processor
        if self.use_gpu:
            torch.cuda.empty_cache()

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if not self._model:
            return OCRResult.from_error(self.name, language.value, "Model not loaded")

        try:
            import torch

            pages = load_document_as_rgb_images(image_bytes)
            prompt = LANG_PROMPTS.get(language, LANG_PROMPTS[SupportedLanguage.ENGLISH])
            page_texts = []
            total_elapsed = 0.0

            for image in pages:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text_input = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(self._model.device)

                t0 = self._timer()
                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs, max_new_tokens=2048)
                total_elapsed += self._elapsed_ms(t0)

                trimmed = [
                    out[len(inp):]
                    for inp, out in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                page_texts.append(output_text.strip())

            output_text = "\n\n".join(text for text in page_texts if text)
            words = [OCRWord(text=w, confidence=1.0) for w in output_text.split() if w]

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=output_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=1.0,
                metadata={"model_id": self.model_id, "page_count": len(pages)},
            )

        except Exception as e:
            logger.exception(f"[olmOCR-2] Inference error: {e}")
            return OCRResult.from_error(self.name, language.value, str(e))
