from app.core import model_selection


def test_resolve_requested_models_expands_pure_ocr_preset_to_loaded_models_only():
    loaded_models = ["tesseract", "easyocr", "got_ocr2"]

    resolved = model_selection.resolve_requested_models(
        "OCR_WITHOUT_LLM_CAPABILITIES",
        loaded_models,
    )

    assert resolved == ["easyocr", "tesseract"]


def test_resolve_requested_models_preserves_explicit_models_alongside_preset():
    loaded_models = ["tesseract", "easyocr", "got_ocr2"]

    resolved = model_selection.resolve_requested_models(
        "got_ocr2, OCR_WITHOUT_LLM_CAPABILITIES",
        loaded_models,
    )

    assert resolved == ["got_ocr2", "easyocr", "tesseract"]


def test_find_unknown_requested_models_ignores_preset_aliases():
    loaded_models = ["tesseract", "easyocr"]

    unknown = model_selection.find_unknown_requested_models(
        "OCR_WITHOUT_LLM_CAPABILITIES, not_loaded_model",
        loaded_models,
    )

    assert unknown == ["not_loaded_model"]
