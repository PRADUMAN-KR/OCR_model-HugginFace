from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class Language(str, Enum):
    ALL = "all"
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    PUNJABI = "pa"


class WordDetail(BaseModel):
    text: str
    confidence: float
    bbox: Optional[List[int]] = None


class ModelResult(BaseModel):
    model_name: str
    language: str
    raw_text: str
    words: List[WordDetail]
    inference_time_ms: float
    avg_confidence: float
    error: Optional[str] = None
    metadata: Dict = {}


class OCRRequest(BaseModel):
    language: Language
    models: Optional[List[str]] = Field(
        default=None,
        description="Specific model names to run. If null, runs all loaded models.",
    )


class OCRResponse(BaseModel):
    filename: str
    language: str
    results: List[ModelResult]
    models_run: int
    total_time_ms: float


# --- Benchmark (with ground truth) ---

class MetricsDetail(BaseModel):
    cer: float = Field(description="Character Error Rate (lower=better)")
    wer: float = Field(description="Word Error Rate (lower=better)")
    ned: float = Field(description="Normalized Edit Distance (lower=better)")
    char_precision: float
    char_recall: float
    char_f1: float
    word_precision: float
    word_recall: float
    word_f1: float
    overall_accuracy: float = Field(description="Blended OCR accuracy score from 0 to 1 (higher=better)")
    exact_match: bool


class BenchmarkModelResult(BaseModel):
    model_name: str
    language: str
    raw_text: str
    inference_time_ms: float
    avg_confidence: float
    metrics: MetricsDetail
    error: Optional[str] = None


class BenchmarkResponse(BaseModel):
    filename: str
    language: str
    ground_truth: str
    results: List[BenchmarkModelResult]
    best_model_cer: str
    best_model_wer: str
    best_model_f1: str
    total_time_ms: float


# --- Model info ---

class ModelInfo(BaseModel):
    name: str
    tier: int
    supported_languages: List[str]
    loaded: bool


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    total_loaded: int


class OCRRunOptionsResponse(BaseModel):
    loaded_models: List[ModelInfo]
    available_model_names: List[str]
    available_languages: List[Language]
    preset_model_groups: Dict[str, List[str]]