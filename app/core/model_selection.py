from typing import Iterable, List

from app.core.config import settings


PURE_OCR_PRESET_ALIASES = {
    "OCR_WITHOUT_LLM_CAPABILITIES",
    "ocr_without_llm_capabilities",
    "pure_ocr",
    "pure-ocr",
}


def preset_model_groups() -> dict[str, List[str]]:
    return {
        "OCR_WITHOUT_LLM_CAPABILITIES": settings.OCR_WITHOUT_LLM_CAPABILITIES,
    }


def resolve_requested_models(models: str | None, loaded_model_names: Iterable[str]) -> List[str]:
    """
    Expand comma-separated model names and preset aliases into a deduplicated list.

    Preset-derived model names are filtered to the currently loaded models so users can
    ask for "all pure OCR" in one go without getting a validation error for models that
    are configured in the preset but not loaded in the current process.
    """
    loaded = list(loaded_model_names)
    loaded_set = set(loaded)
    if not models or not models.strip():
        return loaded

    resolved: List[str] = []
    seen: set[str] = set()

    for raw_token in models.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if token in PURE_OCR_PRESET_ALIASES:
            for model_name in settings.OCR_WITHOUT_LLM_CAPABILITIES:
                if model_name in loaded_set and model_name not in seen:
                    resolved.append(model_name)
                    seen.add(model_name)
            continue

        if token not in seen:
            resolved.append(token)
            seen.add(token)

    return resolved


def find_unknown_requested_models(models: str | None, loaded_model_names: Iterable[str]) -> List[str]:
    """
    Return explicitly requested model names that are not loaded.

    Preset aliases are intentionally excluded because they may legitimately expand to a
    subset of the currently loaded pure OCR models.
    """
    loaded_set = set(loaded_model_names)
    unknown: List[str] = []
    seen: set[str] = set()

    if not models or not models.strip():
        return unknown

    for raw_token in models.split(","):
        token = raw_token.strip()
        if not token or token in PURE_OCR_PRESET_ALIASES or token in seen:
            continue
        seen.add(token)
        if token not in loaded_set:
            unknown.append(token)

    return unknown
