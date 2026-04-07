"""
/ocr endpoints — run loaded OCR models on an uploaded image.
"""

import time
import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from app.schemas import (
    OCRResponse,
    ModelResult,
    WordDetail,
    ModelsListResponse,
    OCRRunOptionsResponse,
    ModelInfo,
    Language,
)
from app.models.base import SupportedLanguage
from app.core.config import settings
from app.core.model_selection import (
    find_unknown_requested_models,
    preset_model_groups,
    resolve_requested_models,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_registry(request: Request):
    return request.app.state.model_registry


def _validate_image(file: UploadFile):
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {settings.ALLOWED_IMAGE_TYPES}",
        )


def _model_request_languages(model) -> List[str]:
    languages = [lang.value for lang in model.supported_languages]
    if model.supports_all_languages():
        languages.append(Language.ALL.value)
    return languages


@router.get("/models", response_model=ModelsListResponse, summary="List loaded models")
async def list_models(registry=Depends(get_registry)):
    """Returns all models currently loaded in memory."""
    models = []
    for name, model in registry.all().items():
        models.append(ModelInfo(
            name=name,
            tier=model.tier,
            supported_languages=_model_request_languages(model),
            loaded=True,
        ))
    return ModelsListResponse(models=models, total_loaded=len(models))


@router.get("/options", response_model=OCRRunOptionsResponse, summary="Runtime OCR options")
async def run_options(registry=Depends(get_registry)):
    """Returns the models currently loaded at startup and the supported language options."""
    loaded_models = []
    for name, model in registry.all().items():
        loaded_models.append(ModelInfo(
            name=name,
            tier=model.tier,
            supported_languages=_model_request_languages(model),
            loaded=True,
        ))

    return OCRRunOptionsResponse(
        loaded_models=loaded_models,
        available_model_names=[m.name for m in loaded_models],
        available_languages=list(Language),
        preset_model_groups=preset_model_groups(),
    )


@router.post("/run", response_model=OCRResponse, summary="Run OCR on an image")
async def run_ocr(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    language: Language = Form(
        ...,
        description=(
            "Target language. Use 'all' to run best-effort multilingual OCR and extract all detectable text content."
        ),
    ),
    models: Optional[str] = Form(
        default=None,
        description=(
            "Comma-separated model names. Leave blank to run all loaded models. "
            "You can also pass OCR_WITHOUT_LLM_CAPABILITIES to run all currently loaded pure OCR models."
        ),
    ),
    registry=Depends(get_registry),
):
    """
    Upload an image and run one or all OCR models on it.
    Returns extracted text, word-level details, confidence, and timing per model.
    """
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    lang_enum = SupportedLanguage(language.value)

    # Resolve which models to run
    loaded_model_names = list(registry.all().keys())
    requested_models = resolve_requested_models(models, loaded_model_names)
    loaded_model_names = set(registry.all().keys())
    unknown_requested = find_unknown_requested_models(models, loaded_model_names)
    if unknown_requested:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some requested models are not loaded at startup.",
                "requested_models": requested_models,
                "unknown_models": unknown_requested,
                "loaded_models": sorted(loaded_model_names),
                "available_presets": preset_model_groups(),
            },
        )

    selected = {
        name: model
        for name, model in registry.all().items()
        if name in requested_models and model.supports_language(lang_enum)
    }

    if not selected:
        raise HTTPException(
            status_code=404,
            detail=f"No loaded models support language '{language.value}' among: {requested_models}",
        )

    start = time.perf_counter()
    results = []

    for name, model in selected.items():
        logger.info(f"[OCR] Running {name} | lang={language.value} | file={file.filename}")
        result = await model.run(image_bytes, lang_enum)
        results.append(ModelResult(
            model_name=result.model_name,
            language=result.language,
            raw_text=result.raw_text,
            words=[WordDetail(text=w.text, confidence=w.confidence, bbox=w.bbox) for w in result.words],
            inference_time_ms=result.inference_time_ms,
            avg_confidence=result.avg_confidence,
            error=result.error,
            metadata=result.metadata,
        ))

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return OCRResponse(
        filename=file.filename,
        language=language.value,
        results=results,
        models_run=len(results),
        total_time_ms=total_ms,
    )
