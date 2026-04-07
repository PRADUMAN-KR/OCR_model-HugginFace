"""
Qwen2.5-VL — Tier 1 VLM
Model: Qwen/Qwen2.5-VL-7B-Instruct (~16GB VRAM) or 3B variant
Install: pip install transformers torch qwen-vl-utils
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

# Language-specific prompts for better extraction
LANG_PROMPTS = {
    SupportedLanguage.ENGLISH: "Extract all text from this image exactly as written. Output only the extracted text, nothing else.",
    SupportedLanguage.ARABIC: "استخرج جميع النصوص من هذه الصورة كما هي مكتوبة. أخرج النص المستخرج فقط.",
    SupportedLanguage.HINDI: "इस छवि से सभी पाठ बिल्कुल वैसे ही निकालें जैसे लिखा गया है। केवल निकाला गया पाठ आउटपुट करें।",
    SupportedLanguage.PUNJABI: "ਇਸ ਚਿੱਤਰ ਤੋਂ ਸਾਰਾ ਟੈਕਸਟ ਉਸੇ ਤਰ੍ਹਾਂ ਕੱਢੋ ਜਿਵੇਂ ਲਿਖਿਆ ਗਿਆ ਹੈ। ਸਿਰਫ਼ ਕੱਢਿਆ ਗਿਆ ਟੈਕਸਟ ਆਉਟਪੁੱਟ ਕਰੋ।",
}


class Qwen25VLModel(BaseOCRModel):
    name = "qwen25_vl"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 1

    def __init__(self, use_gpu: bool = True, model_size: str = "7B", load_in_4bit: bool = False):
        self.use_gpu = use_gpu
        self.model_id = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        self.model_size = model_size
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    async def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        from qwen_vl_utils import process_vision_info  # validate dependency at startup

        device = "cuda" if self.use_gpu else "cpu"
        precision_label = "4-bit" if self.load_in_4bit else ("bf16" if self.use_gpu else "fp32")
        logger.info(f"[Qwen2.5-VL] Loading {self.model_id} on {device} ({precision_label})...")
        hf_kwargs = {"token": settings.HF_TOKEN} if settings.HF_TOKEN else {}
        has_accelerate = importlib.util.find_spec("accelerate") is not None

        self._process_vision_info = process_vision_info

        self._processor = AutoProcessor.from_pretrained(self.model_id, **hf_kwargs)
        model_kwargs = {**hf_kwargs}

        if self.load_in_4bit:
            if not self.use_gpu:
                raise RuntimeError("Qwen 4-bit loading requires GPU mode")
            if not has_accelerate:
                raise RuntimeError("Qwen 4-bit loading requires accelerate")
            if importlib.util.find_spec("bitsandbytes") is None:
                raise RuntimeError("Qwen 4-bit loading requires bitsandbytes")

            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = device
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if self.use_gpu else torch.float32
            if has_accelerate:
                model_kwargs["device_map"] = device
            elif self.use_gpu:
                logger.warning("[Qwen2.5-VL] accelerate not found; loading without device_map and moving model to cuda.")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.use_gpu and not has_accelerate and not self.load_in_4bit:
            self._model = self._model.to(device)
        logger.info("[Qwen2.5-VL] Model loaded.")

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

            prompt = LANG_PROMPTS.get(language, LANG_PROMPTS[SupportedLanguage.ENGLISH])
            pages = load_document_as_rgb_images(image_bytes)
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

                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self._process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self._model.device)

                t0 = self._timer()
                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs, max_new_tokens=2048)
                total_elapsed += self._elapsed_ms(t0)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
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
                metadata={
                    "model_id": self.model_id,
                    "page_count": len(pages),
                    "model_size": self.model_size,
                    "load_in_4bit": self.load_in_4bit,
                },
            )

        except Exception as e:
            logger.exception(f"[Qwen2.5-VL] Inference error: {e}")
            return OCRResult.from_error(self.name, language.value, str(e))
