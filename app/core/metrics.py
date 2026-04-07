"""
Accuracy metrics for OCR evaluation.

Metrics:
  - CER  (Character Error Rate)  — edit distance at character level
  - WER  (Word Error Rate)       — edit distance at word level
  - NED  (Normalized Edit Distance) — Levenshtein / max(len)
  - Precision / Recall / F1     — token overlap
"""

from dataclasses import dataclass
from typing import Optional
import unicodedata
import re


@dataclass
class AccuracyMetrics:
    cer: float              # 0 = perfect, 1 = total mismatch
    wer: float
    ned: float              # normalized edit distance
    char_precision: float
    char_recall: float
    char_f1: float
    word_precision: float
    word_recall: float
    word_f1: float
    overall_accuracy: float
    exact_match: bool


def _normalize(text: str) -> str:
    """Unicode NFC normalization + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein(s1: list, s2: list) -> int:
    """Compute Levenshtein distance between two sequences."""
    m, n = len(s1), len(s2)
    # Use two-row DP for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _token_metrics(pred_tokens: list, gt_tokens: list) -> tuple:
    """Compute precision, recall, F1 via multiset intersection."""
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    intersection = sum((pred_counts & gt_counts).values())
    precision = intersection / len(pred_tokens) if pred_tokens else 0.0
    recall = intersection / len(gt_tokens) if gt_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_metrics(predicted: str, ground_truth: str) -> AccuracyMetrics:
    """
    Compute all OCR accuracy metrics between prediction and ground truth.

    Args:
        predicted:    OCR model output text
        ground_truth: Human-labeled correct text

    Returns:
        AccuracyMetrics dataclass
    """
    pred = _normalize(predicted)
    gt = _normalize(ground_truth)

    # Character-level
    pred_chars = list(pred)
    gt_chars = list(gt)
    char_dist = _levenshtein(pred_chars, gt_chars)
    cer = char_dist / max(len(gt_chars), 1)
    ned = char_dist / max(len(pred_chars), len(gt_chars), 1)
    char_p, char_r, char_f1 = _token_metrics(pred_chars, gt_chars)

    # Word-level
    pred_words = pred.split()
    gt_words = gt.split()
    word_dist = _levenshtein(pred_words, gt_words)
    wer = word_dist / max(len(gt_words), 1)
    word_p, word_r, word_f1 = _token_metrics(pred_words, gt_words)

    # Blended score to provide one easy-to-read "overall accuracy" number.
    overall_accuracy = (
        (1 - min(cer, 1.0)) +
        (1 - min(wer, 1.0)) +
        char_f1 +
        word_f1
    ) / 4

    return AccuracyMetrics(
        cer=round(cer, 4),
        wer=round(wer, 4),
        ned=round(ned, 4),
        char_precision=char_p,
        char_recall=char_r,
        char_f1=char_f1,
        word_precision=word_p,
        word_recall=word_r,
        word_f1=word_f1,
        overall_accuracy=round(overall_accuracy, 4),
        exact_match=(pred == gt),
    )
