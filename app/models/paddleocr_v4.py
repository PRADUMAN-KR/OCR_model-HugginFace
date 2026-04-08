"""
PaddleOCR PP-OCRv4 — Tier 2  (accuracy-first mode)
Supports: English, Arabic, Hindi
Install: pip install paddlepaddle paddleocr

2-Draft Method
──────────────
Draft 1  Full-page OCR pass: collect text, bounding boxes, confidence scores.
         Every word with confidence < SECOND_PASS_THRESHOLD is flagged for
         refinement — short words AND long phrases alike.

Draft 2  For every flagged region, two enhancement strategies are tried:
           Strategy A  gentle  — upscale 2×, moderate contrast/sharpness
           Strategy B  strong  — upscale 3×, higher contrast/sharpness
         For each strategy the crop is enhanced and re-run through OCR.
         All detected words are joined (not just the max) so long phrases
         are reconstructed whole rather than replaced by a fragment.
         The strategy with the highest average confidence that also passes
         both acceptance guards is kept:
           • avg_conf  >  original_conf + MIN_CONFIDENCE_GAIN  (real gain)
           • len(new)  ≥  len(original)  × MIN_TEXT_LENGTH_RATIO  (no truncation)
         Hard binarization is intentionally omitted — it collapses the
         pixel gradients that connected scripts (Arabic, Hindi) rely on.

Final output merges Draft-1 high-conf words with Draft-2 improved words.
Per-page stats are stored in OCRResult.metadata["draft2"].
"""

import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)

# ── Draft 2 tuning knobs ────────────────────────────────────────────────────
SECOND_PASS_THRESHOLD = 0.80   # confidence below this triggers a refinement pass
CROP_PADDING          = 6      # pixels to pad around each low-confidence region

# Two enhancement strategies tried on every low-confidence region.
# Both are attempted; the one with the highest average confidence that also
# passes the acceptance guards is used.
ENHANCEMENT_STRATEGIES = [
    {"upscale": 2, "contrast": 1.5, "sharpness": 1.8},  # gentle
    {"upscale": 3, "contrast": 2.2, "sharpness": 2.8},  # strong
]

# Acceptance guards — prevent Draft 2 from replacing text with a worse result.
MIN_CONFIDENCE_GAIN   = 0.05   # Draft 2 avg conf must exceed Draft 1 by this margin
MIN_TEXT_LENGTH_RATIO = 0.60   # Draft 2 text must be ≥ 60 % of Draft 1 char count

# ── PaddleOCR language map ───────────────────────────────────────────────────
LANG_CONFIG = {
    SupportedLanguage.ENGLISH: {"lang": "en", "ocr_version": "PP-OCRv4"},
    SupportedLanguage.ARABIC:  {"lang": "ar", "ocr_version": "PP-OCRv5"},
    SupportedLanguage.HINDI:   {"lang": "hi", "ocr_version": "PP-OCRv5"},
}


class PaddleOCRv4Model(BaseOCRModel):
    name = "paddleocr_v4"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
    ]
    tier = 2

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._engines: dict = {}   # one engine per language

    def supports_all_languages(self) -> bool:
        return True

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def load(self) -> None:
        from paddleocr import PaddleOCR
        device = "gpu" if self.use_gpu else "cpu"
        for lang_enum, config in LANG_CONFIG.items():
            paddle_lang  = config["lang"]
            ocr_version  = config["ocr_version"]
            logger.info(f"[PaddleOCR] Loading engine lang={paddle_lang} ocr_version={ocr_version}")
            self._engines[lang_enum] = PaddleOCR(
                use_textline_orientation=True,
                lang=paddle_lang,
                device=device,
                ocr_version=ocr_version,
            )
        logger.info("[PaddleOCR] All language engines loaded.")

    async def unload(self) -> None:
        self._engines.clear()

    # ── Parsing helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_xyxy_bbox(poly) -> list[int]:
        """Convert polygon/box output to [x1, y1, x2, y2]."""
        if poly is None:
            return [0, 0, 0, 0]
        try:
            if len(poly) == 0:
                return [0, 0, 0, 0]
        except TypeError:
            return [0, 0, 0, 0]
        xs = [int(float(p[0])) for p in poly]
        ys = [int(float(p[1])) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _parse_ocr_page(self, page_result) -> list[tuple]:
        """
        Parse one PaddleOCR page result into [(text, conf, bbox), ...].
        Supports PaddleOCR 3.x dict format and 2.x legacy list format.
        """
        parsed: list[tuple] = []

        if isinstance(page_result, dict):
            texts  = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            polys  = page_result.get("dt_polys") or page_result.get("rec_polys") or []
            count  = min(len(texts), len(scores), len(polys))
            for i in range(count):
                parsed.append((str(texts[i]), float(scores[i]), self._to_xyxy_bbox(polys[i])))
            return parsed

        if isinstance(page_result, list):
            for line in page_result:
                if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                    continue
                bbox, text_conf = line[0], line[1]
                if not (isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2):
                    continue
                text, conf = text_conf[0], text_conf[1]
                parsed.append((str(text), float(conf), self._to_xyxy_bbox(bbox)))
            return parsed

        return parsed

    @staticmethod
    def _page_score(parsed_page: list) -> float:
        if not parsed_page:
            return 0.0
        return sum(conf for _, conf, _ in parsed_page)

    # ── Draft 2 — image enhancement helpers ─────────────────────────────────

    def _enhance_crop(
        self,
        img_array: np.ndarray,
        bbox: list[int],
        upscale: int   = 2,
        contrast: float = 1.5,
        sharpness: float = 1.8,
    ) -> np.ndarray:
        """
        Crop a low-confidence region and enhance it for re-recognition.
        Parameters come from ENHANCEMENT_STRATEGIES so callers can try
        multiple presets on the same region.

        Pipeline: pad → upscale (LANCZOS) → contrast → sharpness → denoise.
        Hard binarization is intentionally omitted — it collapses the pixel
        gradients that connected scripts (Arabic, Hindi) depend on.
        """
        h, w = img_array.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(w, x2 + CROP_PADDING)
        y2 = min(h, y2 + CROP_PADDING)

        crop = img_array[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return crop

        pil = Image.fromarray(crop)
        pil = pil.resize((pil.width * upscale, pil.height * upscale), Image.LANCZOS)
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
        pil = ImageEnhance.Sharpness(pil).enhance(sharpness)
        pil = pil.filter(ImageFilter.MedianFilter(size=3))
        return np.array(pil)

    def _refine_region(
        self,
        engine,
        img_array: np.ndarray,
        text: str,
        conf: float,
        bbox: list[int],
    ) -> tuple[str, float]:
        """
        Try every enhancement strategy on the low-confidence region.

        For each strategy:
          1. Crop + enhance the region.
          2. Re-run OCR and JOIN all detected words (preserving the engine's
             reading order) so long phrases are reconstructed whole.
          3. Compute average confidence across all joined words.
          4. Apply acceptance guards (gain + length retention).

        Returns the best (text, conf) that passes the guards, or the
        original pair if no strategy improves on it.
        """
        original_len = len(text.strip())
        best_text, best_conf = text, conf

        for strategy in ENHANCEMENT_STRATEGIES:
            enhanced = self._enhance_crop(img_array, bbox, **strategy)
            if enhanced.size == 0:
                continue

            try:
                result  = engine.ocr(enhanced)
                if not result or not result[0]:
                    continue
                refined = self._parse_ocr_page(result[0])
                if not refined:
                    continue

                # Join ALL detected words in the crop (engine's order = correct
                # reading direction for the script). This prevents replacing a
                # full sentence with a single high-confidence fragment.
                candidate_text = " ".join(r[0] for r in refined).strip()
                candidate_conf = sum(r[1] for r in refined) / len(refined)

                # Guard 1: meaningful confidence gain
                if candidate_conf < conf + MIN_CONFIDENCE_GAIN:
                    logger.debug(
                        "[PaddleOCR][Draft2] strategy%s gain too small for '%s': "
                        "%.3f → %.3f (need +%.2f)",
                        strategy, text, conf, candidate_conf, MIN_CONFIDENCE_GAIN,
                    )
                    continue

                # Guard 2: must retain most of the original character count
                if original_len > 0 and len(candidate_text) < original_len * MIN_TEXT_LENGTH_RATIO:
                    logger.debug(
                        "[PaddleOCR][Draft2] strategy%s text too short for '%s': "
                        "%d → %d chars (need ≥%.0f%%)",
                        strategy, text, original_len, len(candidate_text),
                        MIN_TEXT_LENGTH_RATIO * 100,
                    )
                    continue

                if candidate_conf > best_conf:
                    best_text = candidate_text
                    best_conf = candidate_conf
                    logger.debug(
                        "[PaddleOCR][Draft2] strategy%s candidate '%s'(%.3f) → '%s'(%.3f)",
                        strategy, text, conf, candidate_text, candidate_conf, 
                    )

            except Exception as exc:
                logger.debug(
                    "[PaddleOCR][Draft2] strategy%s error for '%s': %s",
                    strategy, text, exc,
                )
                continue

        if best_text != text:
            logger.debug(
                "[PaddleOCR][Draft2] final: '%s'(%.3f) → '%s'(%.3f)",
                text, conf, best_text, best_conf,
            )
        return best_text, best_conf

    def _second_pass(
        self,
        engine,
        img_array: np.ndarray,
        parsed_page: list,
    ) -> tuple[list, dict]:
        """
        Draft 2: walk every low-confidence word, enhance its crop, and re-run OCR.
        Accepts the refined result only when confidence improves.
        Returns the updated parsed_page and a stats dict that includes a word-level
        diff (changes) so callers can compare Draft 1 vs Draft 2 side-by-side.
        """
        candidates = sum(1 for _, conf, _ in parsed_page if conf < SECOND_PASS_THRESHOLD)
        improved   = 0
        changes: list[dict] = []
        t0 = self._timer()

        refined_page: list[tuple] = []
        for text, conf, bbox in parsed_page:
            if conf < SECOND_PASS_THRESHOLD:
                new_text, new_conf = self._refine_region(engine, img_array, text, conf, bbox)
                refined_page.append((new_text, new_conf, bbox))
                if new_conf > conf:
                    improved += 1
                    changes.append({
                        "bbox":        bbox,
                        "draft1_text": text,       "draft1_conf": round(conf, 4),
                        "draft2_text": new_text,   "draft2_conf": round(new_conf, 4),
                    })
                    logger.debug(
                        "[PaddleOCR][Draft2] '%s'(%.3f) → '%s'(%.3f)",
                        text, conf, new_text, new_conf,
                    )
            else:
                refined_page.append((text, conf, bbox))

        stats = {
            "threshold":     SECOND_PASS_THRESHOLD,
            "candidates":    candidates,
            "improved":      improved,
            "refinement_ms": round(self._elapsed_ms(t0), 2),
            "changes":       changes,
        }
        logger.info(
            "[PaddleOCR][Draft2] %d/%d low-conf words improved (threshold=%.2f)",
            improved, candidates, SECOND_PASS_THRESHOLD,
        )
        return refined_page, stats

    # ── Language-selection helper (ALL mode) ─────────────────────────────────

    def _select_all_language_page(self, img_array: np.ndarray):
        """
        Run Draft 1 with every loaded language engine; return the engine/page
        whose cumulative confidence score is highest.
        Returns (best_lang, best_page, total_elapsed_ms).
        """
        best_lang  = None
        best_page: list  = []
        best_score = -1.0
        total_elapsed = 0.0

        for lang_enum, engine in self._engines.items():
            t0     = self._timer()
            result = engine.ocr(img_array)
            total_elapsed += self._elapsed_ms(t0)

            parsed_page = self._parse_ocr_page(result[0]) if result and result[0] else []
            score = self._page_score(parsed_page)
            if score > best_score:
                best_score = score
                best_lang  = lang_enum
                best_page  = parsed_page

        return best_lang, best_page, total_elapsed

    # ── Main inference entry point ───────────────────────────────────────────

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language != SupportedLanguage.ALL:
            engine = self._engines.get(language)
        else:
            engine = None

        if language != SupportedLanguage.ALL and not engine:
            return OCRResult.from_error(
                self.name, language.value, f"Language {language} not loaded"
            )

        try:
            # final (Draft 2) accumulators
            words: list[OCRWord]      = []
            page_texts: list[str]     = []
            # Draft 1 snapshot accumulators
            draft1_words: list[dict]  = []
            draft1_texts: list[str]   = []

            total_elapsed  = 0.0
            pages          = load_document_as_rgb_images(image_bytes)
            resolved_page_languages: list = []
            second_pass_stats: list[dict] = []

            for image in pages:
                img_array = np.array(image)

                # ── Draft 1: full-page OCR ────────────────────────────────────
                if language == SupportedLanguage.ALL:
                    page_language, parsed_page, page_elapsed = self._select_all_language_page(img_array)
                    total_elapsed += page_elapsed
                    resolved_page_languages.append(page_language.value if page_language else None)
                    page_engine = self._engines.get(page_language) if page_language else None
                else:
                    t0 = self._timer()
                    result = engine.ocr(img_array)
                    total_elapsed += self._elapsed_ms(t0)
                    parsed_page = self._parse_ocr_page(result[0]) if result and result[0] else []
                    page_engine = engine

                # Snapshot Draft 1 output before any refinement
                d1_lines: list[str] = []
                for text, conf, bbox in parsed_page:
                    draft1_words.append({"text": text, "confidence": round(conf, 4), "bbox": bbox})
                    d1_lines.append(text)
                draft1_texts.append("\n".join(d1_lines))

                # ── Draft 2: refine low-confidence regions ────────────────────
                if page_engine and parsed_page:
                    parsed_page, pass_stats = self._second_pass(page_engine, img_array, parsed_page)
                    second_pass_stats.append(pass_stats)
                    total_elapsed += pass_stats["refinement_ms"]

                # Build final (merged) output from Draft 2 result
                lines: list[str] = []
                for text, conf, flat_bbox in parsed_page:
                    words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
                    lines.append(text)
                page_texts.append("\n".join(lines))

            raw_text      = "\n\n".join(t for t in page_texts if t)
            d1_raw_text   = "\n\n".join(t for t in draft1_texts if t)
            avg_conf      = sum(w.confidence for w in words) / len(words) if words else 0.0
            d1_avg_conf   = sum(w["confidence"] for w in draft1_words) / len(draft1_words) if draft1_words else 0.0

            metadata: dict = {
                "page_count": len(pages),
                # ── Draft 1 full snapshot ─────────────────────────────────────
                "draft1": {
                    "raw_text":     d1_raw_text,
                    "words":        draft1_words,
                    "avg_confidence": round(d1_avg_conf, 4),
                },
                # ── Draft 2 refinement stats + word-level diff ────────────────
                "draft2": {
                    "threshold":    SECOND_PASS_THRESHOLD,
                    "pages":        second_pass_stats,   # per-page: candidates/improved/changes/ms
                },
            }
            if language == SupportedLanguage.ALL:
                metadata["best_effort_language_mode"] = "best_single_language_engine_per_page"
                metadata["resolved_page_languages"]   = resolved_page_languages

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,           # final best result (Draft 2 merged)
                words=words,                 # final best words
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=round(avg_conf, 4),
                metadata=metadata,
            )

        except Exception as e:
            logger.exception("[PaddleOCR] Inference error: %s", e)
            return OCRResult.from_error(self.name, language.value, str(e))
