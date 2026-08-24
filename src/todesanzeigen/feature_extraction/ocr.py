from __future__ import annotations

from typing import Any

from ..ocr_filtering import parse_tesseract_word_lines


def extract_ocr_text_features(
    text: str,
    *,
    name_hint: str = "",
    name_confidence: Any = None,
) -> dict[str, Any]:
    suspicious_chars = sum(1 for char in text if not (char.isalnum() or char.isspace()))
    return {
        "ocr_char_count": len(text),
        "ocr_word_count": len(text.split()),
        "ocr_line_count": len([line for line in text.splitlines() if line.strip()]),
        "ocr_suspicious_char_ratio": _safe_div(suspicious_chars, len(text)),
        "name_hint_length": len(name_hint),
        "name_confidence": name_confidence,
    }


def extract_tsv_features(tsv_text: str) -> dict[str, Any]:
    lines = parse_tesseract_word_lines(tsv_text)
    words = [word for line in lines for word in line.words]
    confidences = [
        word.confidence for word in words if word.confidence is not None and word.confidence >= 0
    ]
    areas = [word.width * word.height for word in words]
    page_width = max((word.left + word.width for word in words), default=0)
    page_height = max((word.top + word.height for word in words), default=0)
    page_area = page_width * page_height
    largest_area = max(areas, default=0)
    mean_confidence = _mean(confidences)
    return {
        "tsv_line_count": len(lines),
        "tsv_word_count": len(words),
        "tsv_mean_confidence": mean_confidence,
        "tsv_confidence_variance": _variance(confidences, mean_confidence),
        "tsv_low_confidence_ratio": _safe_div(
            sum(1 for confidence in confidences if confidence < 70),
            len(confidences),
        ),
        "layout_bbox_density": _safe_div(sum(areas), page_area),
        "layout_largest_text_ratio": _safe_div(largest_area, sum(areas)),
    }


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _variance(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    return sum((value - mean) ** 2 for value in values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
