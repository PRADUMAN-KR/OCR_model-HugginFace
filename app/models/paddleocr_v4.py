"""
PaddleOCR PP-OCRv4/v5 — Tier 2

Standard pipeline + F2 MAX-ACCURACY pipeline (enabled when max_accuracy=True):

  Stage 1 — Generate full-page image variants (original, 2x upscale, CLAHE,
             sharpened, denoised)
  Stage 2 — Run full engine.ocr() on each variant × each engine.
             Arabic: PP-OCRv5 primary + PP-OCRv3 secondary → 5 × 2 = 10 passes
  Stage 3 — Align results across passes by bbox IoU and pick best candidate
             per text line
  Stage 4 — Candidate scoring (model confidence × Arabic char ratio ×
             repeated-char penalty × length sanity)
  Stage 5 — Arabic language-model correction (diacritic stripping, Alef /
             Teh-marbuta / Alef-maqsura normalisation, zero-width cleanup)

Supports: English, Arabic, Hindi
Install:  pip install paddlepaddle paddleocr opencv-python
"""

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → PaddleOCR engine config
# ---------------------------------------------------------------------------

LANG_CONFIG: dict[SupportedLanguage, dict] = {
    SupportedLanguage.ENGLISH: {"lang": "en", "ocr_version": "PP-OCRv4"},
    SupportedLanguage.ARABIC:  {"lang": "ar", "ocr_version": "PP-OCRv5"},
    SupportedLanguage.HINDI:   {"lang": "hi", "ocr_version": "PP-OCRv5"},
}

# Alternate ocr_version engines for F2 multi-engine diversity.
# Only versions that have models for the given language are listed.
_ALT_ENGINES: dict[SupportedLanguage, list[str]] = {
    SupportedLanguage.ARABIC:  ["PP-OCRv3"],   # PP-OCRv5 primary + PP-OCRv3 secondary
    SupportedLanguage.HINDI:   [],
    SupportedLanguage.ENGLISH: [],
}

# ---------------------------------------------------------------------------
# Arabic normalisation table (applied in Stage 5)
# ---------------------------------------------------------------------------

# Only strip invisible / zero-width characters that add no readable value.
# Diacritics, Hamza variants, Taa-marbuta etc. are intentionally KEPT
# because the OCR engine already produces correct Arabic and normalising
# them degrades the output.
_ARABIC_CLEANUP_MAP = str.maketrans({
    "\u200C": "",   # ZWNJ
    "\u200D": "",   # ZWJ
    "\u200B": "",   # zero-width space
    "\u200E": "",   # LRM
    "\u200F": "",   # RLM
    "\uFEFF": "",   # BOM / ZWNBSP
})

_RE_REPEATED = re.compile(r"(.)\1{4,}")


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PaddleOCRv4Model(BaseOCRModel):
    """
    PaddleOCR wrapper with an optional F2 MAX-ACCURACY pipeline.

    The F2 pipeline runs the full PaddleOCR det+rec engine on multiple
    preprocessed versions of the entire page (not isolated crops), then
    merges/scores results at the text-line level via bbox IoU alignment.
    """

    name = "paddleocr_v4"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
    ]
    tier = 2

    def __init__(
        self,
        use_gpu: bool = False,
        max_accuracy: bool = True,
        debug_output_dir: str | None = None,
    ):
        self.use_gpu = use_gpu
        self.max_accuracy = max_accuracy
        self.debug_output_dir = debug_output_dir
        # Primary full-pipeline engines (one per language)
        self._engines: dict[SupportedLanguage, object] = {}
        # F2: alternate-version full-pipeline engines per language
        self._alt_engines: dict[SupportedLanguage, list[tuple[str, object]]] = {}

    def supports_all_languages(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    async def load(self) -> None:
        from paddleocr import PaddleOCR

        device = "gpu" if self.use_gpu else "cpu"
        logger.info("[PaddleOCR] Device: %s | max_accuracy(F2): %s", device.upper(), self.max_accuracy)

        for lang_enum, config in LANG_CONFIG.items():
            paddle_lang = config["lang"]
            ocr_version = config["ocr_version"]
            logger.info("[PaddleOCR] Loading engine: lang=%s version=%s", paddle_lang, ocr_version)
            self._engines[lang_enum] = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                lang=paddle_lang,
                device=device,
                ocr_version=ocr_version,
                text_det_thresh=0.5,
                text_det_box_thresh=0.7,
                text_rec_score_thresh=0.5,
            )

            if not self.max_accuracy:
                continue

            # Load alternate-version engines for multi-engine diversity
            alt_list: list[tuple[str, object]] = []
            for alt_ver in _ALT_ENGINES.get(lang_enum, []):
                try:
                    logger.info("[PaddleOCR F2] Loading alt engine: lang=%s version=%s", paddle_lang, alt_ver)
                    alt_engine = PaddleOCR(
                        use_textline_orientation=True,
                        use_doc_orientation_classify=True,
                        use_doc_unwarping=True,
                        lang=paddle_lang,
                        device=device,
                        ocr_version=alt_ver,
                        text_det_thresh=0.5,
                        text_det_box_thresh=0.7,
                        text_rec_score_thresh=0.5,
                    )
                    alt_list.append((alt_ver, alt_engine))
                except Exception as exc:
                    logger.warning("[PaddleOCR F2] Alt engine %s unavailable for %s: %s", alt_ver, paddle_lang, exc)

            if alt_list:
                self._alt_engines[lang_enum] = alt_list
                logger.info(
                    "[PaddleOCR F2] %d alt engine(s) for %s: %s",
                    len(alt_list), paddle_lang, [v for v, _ in alt_list],
                )

        n_engines = 1 + len(self._alt_engines.get(SupportedLanguage.ARABIC, []))
        logger.info("[PaddleOCR] All engines loaded. F2 passes/variant: %d", n_engines)

    async def unload(self) -> None:
        self._engines.clear()
        self._alt_engines.clear()

    # ------------------------------------------------------------------
    # Stage 1 — Full-page preprocessing variants
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_page_variants(img_rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        """
        Generate full-page preprocessing variants.
        Returns list of (name, scale_factor, image_rgb).
        scale_factor is relative to the original (1.0 = same size, 2.0 = 2x).
        """
        variants: list[tuple[str, float, np.ndarray]] = []

        variants.append(("original", 1.0, img_rgb))

        h, w = img_rgb.shape[:2]
        up = cv2.resize(img_rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled_2x", 2.0, up))

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        variants.append((
            "clahe", 1.0,
            cv2.cvtColor(cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
        ))

        sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(bgr, -1, sharpen_k)
        variants.append(("sharpened", 1.0, cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)))

        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
        variants.append(("denoised", 1.0, cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)))

        return variants

    # ------------------------------------------------------------------
    # Stage 2 — Run full engine.ocr() on each variant × engine
    # ------------------------------------------------------------------

    def _run_all_passes(
        self, img_rgb: np.ndarray, lang_enum: SupportedLanguage
    ) -> list[tuple[str, str, list[tuple[str, float, list[int]]]]]:
        """
        Run every (variant × engine) combination.
        All bboxes are normalized back to the original image scale so that
        IoU alignment works across upscaled variants.
        Returns list of (variant_name, engine_name, parsed_page).
        """
        variants = self._generate_page_variants(img_rgb)

        primary_engine = self._engines.get(lang_enum)
        if not primary_engine:
            return []

        primary_ver = LANG_CONFIG[lang_enum]["ocr_version"]
        engines: list[tuple[str, object]] = [(primary_ver, primary_engine)]
        engines.extend(self._alt_engines.get(lang_enum, []))

        all_passes: list[tuple[str, str, list[tuple[str, float, list[int]]]]] = []

        for variant_name, scale, variant_img in variants:
            for engine_name, engine in engines:
                try:
                    result = engine.ocr(variant_img)
                    parsed = self._parse_ocr_page(result[0]) if result and result[0] else []
                    if scale != 1.0 and parsed:
                        inv = 1.0 / scale
                        parsed = [
                            (text, conf, [
                                int(bbox[0] * inv), int(bbox[1] * inv),
                                int(bbox[2] * inv), int(bbox[3] * inv),
                            ])
                            for text, conf, bbox in parsed
                        ]
                    all_passes.append((variant_name, engine_name, parsed))
                except Exception as exc:
                    logger.warning(
                        "[PaddleOCR F2] Pass failed (%s × %s): %s",
                        variant_name, engine_name, exc,
                    )

        logger.info(
            "[PaddleOCR F2] Stage 2: %d passes completed (%d variants × %d engines)",
            len(all_passes), len(variants), len(engines),
        )
        return all_passes

    # ------------------------------------------------------------------
    # Stage 3 — Pick the best complete pass (not line-level merging)
    # ------------------------------------------------------------------

    @staticmethod
    def _arabic_char_ratio(text: str) -> float:
        if not text:
            return 0.0
        arabic = sum(
            1 for c in text
            if "\u0600" <= c <= "\u06FF"
            or "\u0750" <= c <= "\u077F"
            or "\uFB50" <= c <= "\uFDFF"
        )
        return arabic / len(text)

    def _score_pass(
        self,
        parsed: list[tuple[str, float, list[int]]],
        language: SupportedLanguage,
    ) -> float:
        """
        Score an entire pass result.  Higher = better.

        Factors:
          • total_text_length — more text captured = more complete
          • mean_confidence   — higher confidence = more reliable
          • arabic_char_ratio — for Arabic, penalise non-Arabic junk
          • repeated-char penalty
        """
        if not parsed:
            return 0.0

        total_text = " ".join(t.strip() for t, _, _ in parsed if t.strip())
        if not total_text:
            return 0.0

        total_len = len(total_text)
        mean_conf = sum(c for _, c, _ in parsed) / len(parsed)

        score = total_len * mean_conf

        if _RE_REPEATED.search(total_text):
            score *= 0.5

        if language == SupportedLanguage.ARABIC:
            ratio = self._arabic_char_ratio(total_text)
            if ratio < 0.30:
                score *= 0.2
            elif ratio < 0.50:
                score *= 0.5

        return score

    def _select_best_pass(
        self,
        all_passes: list[tuple[str, str, list[tuple[str, float, list[int]]]]],
        language: SupportedLanguage,
    ) -> tuple[str, str, list[tuple[str, float, list[int]]]]:
        """Pick the single pass whose full-page result is best."""
        best_pass = ("", "", [])
        best_score = -1.0
        for variant_name, engine_name, parsed in all_passes:
            s = self._score_pass(parsed, language)
            logger.debug(
                "[PaddleOCR F2] Pass score: %s × %s → %.1f (%d lines)",
                variant_name, engine_name, s, len(parsed),
            )
            if s > best_score:
                best_score = s
                best_pass = (variant_name, engine_name, parsed)

        vname, ename, _ = best_pass
        logger.info(
            "[PaddleOCR F2] Stage 3: Best pass = %s × %s (score %.1f)",
            vname, ename, best_score,
        )
        return best_pass

    # ------------------------------------------------------------------
    # Stage 5 — Arabic language-model correction
    # ------------------------------------------------------------------

    @staticmethod
    def _arabic_correct(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.translate(_ARABIC_CLEANUP_MAP)
        return text.strip()

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_draw_arabic(
        img_bgr: np.ndarray, text: str, xy: tuple[int, int],
        font_size: int = 18, color: tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text(xy, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_debug_dir(self) -> Path | None:
        if not self.debug_output_dir:
            return None
        raw_dir = Path(self.debug_output_dir)
        if not raw_dir.is_absolute():
            raw_dir = Path.cwd() / raw_dir
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            return raw_dir
        except Exception:
            fallback = Path.cwd() / "debug_output"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _visualize_f2(
        self,
        img_rgb: np.ndarray,
        variants: list[tuple[str, float, np.ndarray]],
        results: list[tuple[str, str, float, list[int]]],
        run_id: str,
    ) -> dict[str, str]:
        out_dir = self._get_debug_dir()
        if not out_dir:
            return {}
        saved: dict[str, str] = {}

        # Stage 1: variant thumbnails grid
        THUMB_W, THUMB_H = 320, 200
        cells: list[np.ndarray] = []
        for vname, _scale, vimg in variants:
            thumb = cv2.resize(
                cv2.cvtColor(vimg, cv2.COLOR_RGB2BGR),
                (THUMB_W, THUMB_H), interpolation=cv2.INTER_LINEAR,
            )
            label_bar = np.zeros((25, THUMB_W, 3), dtype=np.uint8)
            cv2.putText(label_bar, vname, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cells.append(np.vstack([label_bar, thumb]))

        while len(cells) % 3:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
        grid = np.vstack(rows)
        p1 = str(out_dir / f"{run_id}_stage1_variants.jpg")
        cv2.imwrite(p1, grid)
        saved["stage1_variants"] = p1

        # Stage 4: final result overlay
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for text_raw, text_corrected, conf, (x1, y1, x2, y2) in results:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"[{conf:.2f}] {text_corrected[:40]}"
            vis = self._pil_draw_arabic(vis, label, (x1, max(y1 - 22, 2)), font_size=15)
        p4 = str(out_dir / f"{run_id}_stage4_results.jpg")
        cv2.imwrite(p4, vis)
        saved["stage4_results"] = p4

        logger.info("[PaddleOCR F2 VIS] Saved → %s", out_dir)
        return saved

    # ------------------------------------------------------------------
    # F2 pipeline orchestrator
    # ------------------------------------------------------------------

    def _run_f2_pipeline(
        self,
        img_rgb: np.ndarray,
        lang_enum: SupportedLanguage,
        debug_run_id: str | None = None,
    ) -> tuple[list[tuple[str, str, float, list[int]]], dict[str, str]]:
        """
        Full-page multi-pass F2 pipeline.

        Runs OCR on multiple preprocessed page variants × engines, then
        picks the single best *complete pass* (not a line-level merge)
        so that the output preserves full context.

        Returns:
          (results, debug_paths)
          results     — list of (text_raw, text_corrected, confidence, bbox)
          debug_paths — dict of saved visualisation file paths
        """
        # Stage 1+2: generate page variants, run full ocr on each
        all_passes = self._run_all_passes(img_rgb, lang_enum)
        if not all_passes:
            return [], {}

        # Stage 3: pick the best complete pass
        best_variant, best_engine, best_parsed = self._select_best_pass(
            all_passes, lang_enum
        )

        # Stage 4+5: apply Arabic correction to the winning pass
        results: list[tuple[str, str, float, list[int]]] = []
        for text_raw, conf, bbox in best_parsed:
            if not text_raw or not text_raw.strip():
                continue
            text_corrected = (
                self._arabic_correct(text_raw)
                if lang_enum == SupportedLanguage.ARABIC
                else text_raw
            )
            if text_corrected:
                results.append((text_raw, text_corrected, conf, bbox))

        # Sort results top-to-bottom by y1 coordinate
        results.sort(key=lambda r: r[3][1])

        logger.info(
            "[PaddleOCR F2] Pipeline done: %d lines (best pass: %s × %s, from %d total passes)",
            len(results), best_variant, best_engine, len(all_passes),
        )

        # Debug visualisation
        debug_paths: dict[str, str] = {}
        if self.debug_output_dir and debug_run_id:
            try:
                variants = self._generate_page_variants(img_rgb)
                debug_paths = self._visualize_f2(img_rgb, variants, results, debug_run_id)
            except Exception as exc:
                logger.warning("[PaddleOCR F2 VIS] Visualisation failed: %s", exc)

        return results, debug_paths

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_xyxy_bbox(poly) -> list[int]:
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

    MIN_BOX_AREA = 150      # px² — smaller boxes are almost always noise
    MIN_BOX_HEIGHT = 8      # px  — shorter than this is grid lines / artifacts

    def _parse_ocr_page(self, page_result) -> list[tuple[str, float, list[int]]]:
        raw: list[tuple[str, float, list[int]]] = []
        if isinstance(page_result, dict):
            texts  = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            polys  = page_result.get("dt_polys") or page_result.get("rec_polys") or []
            for i in range(min(len(texts), len(scores), len(polys))):
                raw.append((str(texts[i]), float(scores[i]), self._to_xyxy_bbox(polys[i])))
        elif isinstance(page_result, list):
            for line in page_result:
                if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                    continue
                bbox, text_conf = line[0], line[1]
                if not (isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2):
                    continue
                raw.append((str(text_conf[0]), float(text_conf[1]), self._to_xyxy_bbox(bbox)))

        parsed: list[tuple[str, float, list[int]]] = []
        for text, conf, (x1, y1, x2, y2) in raw:
            w, h = x2 - x1, y2 - y1
            if w * h < self.MIN_BOX_AREA or h < self.MIN_BOX_HEIGHT:
                continue
            if not text or not text.strip():
                continue
            parsed.append((text, conf, [x1, y1, x2, y2]))
        return parsed

    @staticmethod
    def _page_score(parsed_page: list[tuple[str, float, list[int]]]) -> float:
        return sum(conf for _, conf, _ in parsed_page) if parsed_page else 0.0

    def _select_all_language_page(
        self, img_array: np.ndarray
    ) -> tuple[SupportedLanguage | None, list[tuple[str, float, list[int]]], float]:
        best_lang: SupportedLanguage | None = None
        best_page: list[tuple[str, float, list[int]]] = []
        best_score = -1.0
        total_elapsed = 0.0
        for lang_enum, engine in self._engines.items():
            t0 = self._timer()
            result = engine.ocr(img_array)
            total_elapsed += self._elapsed_ms(t0)
            parsed = self._parse_ocr_page(result[0]) if result and result[0] else []
            score = self._page_score(parsed)
            if score > best_score:
                best_score, best_lang, best_page = score, lang_enum, parsed
        return best_lang, best_page, total_elapsed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language != SupportedLanguage.ALL:
            engine = self._engines.get(language)
            if not engine:
                return OCRResult.from_error(self.name, language.value, f"Language {language} not loaded")
        else:
            engine = None

        try:
            words:          list[OCRWord] = []
            page_texts:     list[str]     = []
            page_texts_raw: list[str]     = []
            total_elapsed   = 0.0
            pages           = load_document_as_rgb_images(image_bytes)
            resolved_page_languages: list[str | None] = []

            f2_active = (
                self.max_accuracy
                and language != SupportedLanguage.ALL
            )

            run_id    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            all_debug: dict[str, str] = {}

            for page_idx, image in enumerate(pages):
                img_array = np.array(image)

                if language == SupportedLanguage.ALL:
                    page_lang, parsed_std, page_elapsed = self._select_all_language_page(img_array)
                    total_elapsed += page_elapsed
                    resolved_page_languages.append(page_lang.value if page_lang else None)
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox in parsed_std:
                        words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
                        lines.append(text)
                        lines_raw.append(text)

                elif f2_active:
                    t0 = self._timer()
                    page_debug_id = f"{run_id}_p{page_idx}"
                    f2_page, debug_paths = self._run_f2_pipeline(
                        img_array, language, debug_run_id=page_debug_id
                    )
                    total_elapsed += self._elapsed_ms(t0)
                    all_debug.update({f"p{page_idx}_{k}": v for k, v in debug_paths.items()})
                    lines, lines_raw = [], []
                    for text_raw, text_corrected, conf, flat_bbox in f2_page:
                        words.append(OCRWord(text=text_corrected, confidence=conf, bbox=flat_bbox))
                        lines.append(text_corrected)
                        lines_raw.append(text_raw)

                else:
                    t0 = self._timer()
                    result = engine.ocr(img_array)  # type: ignore[union-attr]
                    total_elapsed += self._elapsed_ms(t0)
                    parsed_std = self._parse_ocr_page(result[0]) if result and result[0] else []
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox in parsed_std:
                        words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
                        lines.append(text)
                        lines_raw.append(text)

                page_texts.append("\n".join(lines))
                page_texts_raw.append("\n".join(lines_raw))

            raw_text     = "\n\n".join(t for t in page_texts     if t)
            raw_text_pre = "\n\n".join(t for t in page_texts_raw if t)
            avg_conf     = sum(w.confidence for w in words) / len(words) if words else 0.0

            metadata: dict = {
                "page_count":                len(pages),
                "max_accuracy_mode":         f2_active,
                "raw_text_before_correction": raw_text_pre,
            }
            if all_debug:
                metadata["debug_visualizations"] = all_debug
            if language == SupportedLanguage.ALL:
                metadata["best_effort_language_mode"] = "best_single_language_engine_per_page"
                metadata["resolved_page_languages"]   = resolved_page_languages

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=round(avg_conf, 4),
                metadata=metadata,
            )

        except Exception as exc:
            logger.exception("[PaddleOCR] Inference error: %s", exc)
            return OCRResult.from_error(self.name, language.value, str(exc))
