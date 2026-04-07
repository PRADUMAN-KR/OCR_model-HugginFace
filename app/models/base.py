"""
Base class all OCR model wrappers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import time


class SupportedLanguage(str, Enum):
    ALL = "all"
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    PUNJABI = "pa"


@dataclass
class OCRWord:
    text: str
    confidence: float          # 0.0 – 1.0
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]


@dataclass
class OCRResult:
    model_name: str
    language: str
    raw_text: str
    words: List[OCRWord]
    inference_time_ms: float
    avg_confidence: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_error(cls, model_name: str, language: str, error: str) -> "OCRResult":
        return cls(
            model_name=model_name,
            language=language,
            raw_text="",
            words=[],
            inference_time_ms=0.0,
            avg_confidence=0.0,
            error=error,
        )


class BaseOCRModel(ABC):
    name: str = "base"
    supported_languages: List[SupportedLanguage] = []
    tier: int = 2  # 1 = VLM, 2 = Traditional

    @abstractmethod
    async def load(self) -> None:
        """Initialize model weights, tokenizer, etc."""

    @abstractmethod
    async def unload(self) -> None:
        """Free GPU/CPU memory."""

    @abstractmethod
    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        """Run inference and return structured OCRResult."""

    def supports_language(self, lang: SupportedLanguage) -> bool:
        if lang == SupportedLanguage.ALL:
            return self.supports_all_languages()
        return lang in self.supported_languages

    def supports_all_languages(self) -> bool:
        """
        Whether the model can run in a best-effort multilingual mode.
        """
        return False

    def _timer(self):
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
