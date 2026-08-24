from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS
from .ocr_filtering import parse_tesseract_word_lines
from .storage import (
    DEFAULT_DB_PATH,
    DEFAULT_LABEL_SET,
    all_documents,
    apply_migrations,
    connect,
    latest_extraction_by_method,
    latest_feature_snapshot,
    latest_ocr_output,
    upsert_feature_snapshot,
)


DEFAULT_FEATURE_SET = "router-v1"


@dataclass(frozen=True)
class FeatureBuildSummary:
    feature_set: str
    documents: int
    snapshots: int


@dataclass(frozen=True)
class RouterDatasetSummary:
    rows: int
    missing_features: int
    missing_predictions: int
    output_file: Path


def build_feature_snapshots(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> FeatureBuildSummary:
    apply_migrations(db_path)
    with connect(db_path) as connection:
        documents = all_documents(connection)
        snapshot_count = 0
        for document in documents:
            ocr = latest_ocr_output(connection, document_id=int(document["id"]))
            features = _document_features(dict(document))
            if ocr is not None:
                features.update(_ocr_features(dict(ocr), _artifact_text(connection, ocr["tsv_artifact_id"])))
            upsert_feature_snapshot(
                connection,
                document_id=int(document["id"]),
                ocr_output_id=int(ocr["id"]) if ocr is not None else None,
                feature_set=feature_set,
                features=features,
                config={"version": feature_set},
            )
            snapshot_count += 1
    return FeatureBuildSummary(
        feature_set=feature_set,
        documents=len(documents),
        snapshots=snapshot_count,
    )


def export_router_dataset(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    output_file: Path,
    label_set: str = DEFAULT_LABEL_SET,
    method: str,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> RouterDatasetSummary:
    apply_migrations(db_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing_features = 0
    missing_predictions = 0
    with connect(db_path) as connection:
        for label in connection.execute(
            """
            SELECT
                ground_truth_labels.document_id,
                ground_truth_labels.fields_json,
                documents.filename_stem,
                documents.year,
                sources.name AS source
            FROM ground_truth_labels
            JOIN documents ON documents.id = ground_truth_labels.document_id
            JOIN sources ON sources.id = documents.source_id
            WHERE ground_truth_labels.label_set = ?
            ORDER BY sources.name, documents.filename_stem, documents.id
            """,
            (label_set,),
        ):
            document_id = int(label["document_id"])
            feature_snapshot = latest_feature_snapshot(
                connection,
                document_id=document_id,
                feature_set=feature_set,
            )
            if feature_snapshot is None:
                missing_features += 1
                continue

            prediction_row = latest_extraction_by_method(
                connection,
                document_id=document_id,
                method=method,
            )
            truth = _loads(label["fields_json"])
            missing_prediction = prediction_row is None
            prediction = _loads(prediction_row["fields_json"]) if prediction_row else {}
            exact_match = False if missing_prediction else _exact_match(truth, prediction)
            if missing_prediction:
                missing_predictions += 1
            rows.append(
                {
                    "document_id": document_id,
                    "source": label["source"],
                    "filename_stem": label["filename_stem"],
                    "year": label["year"],
                    "label_set": label_set,
                    "feature_set": feature_set,
                    "method": method,
                    "features": _loads(feature_snapshot["features_json"]),
                    "target": {
                        "cheap_pipeline_failed": not exact_match,
                        "exact_match": exact_match,
                        "missing_prediction": missing_prediction,
                    },
                    "extraction_output_id": (
                        int(prediction_row["id"]) if prediction_row is not None else None
                    ),
                }
            )

    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return RouterDatasetSummary(
        rows=len(rows),
        missing_features=missing_features,
        missing_predictions=missing_predictions,
        output_file=output_file,
    )


def _document_features(document: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(document.get("image_path", "") or ""))
    features: dict[str, Any] = {
        "source": str(document.get("source", "") or ""),
        "year": document.get("year"),
        "layout_family": str(document.get("layout_family", "") or ""),
        "filename_length": len(str(document.get("filename_stem", "") or "")),
        "image_mime_type": str(document.get("mime_type", "") or ""),
    }
    if str(image_path):
        features.update(_image_features(image_path))
    return features


def _ocr_features(ocr: dict[str, Any], tsv_text: str) -> dict[str, Any]:
    text = str(ocr.get("text", "") or "")
    suspicious_chars = sum(1 for char in text if not (char.isalnum() or char.isspace()))
    features: dict[str, Any] = {
        "ocr_char_count": len(text),
        "ocr_word_count": len(text.split()),
        "ocr_line_count": len([line for line in text.splitlines() if line.strip()]),
        "ocr_suspicious_char_ratio": _safe_div(suspicious_chars, len(text)),
        "name_hint_length": len(str(ocr.get("name_hint", "") or "")),
        "name_confidence": ocr.get("name_confidence"),
    }
    features.update(_loads(str(ocr.get("features_json", "") or "{}")))
    if tsv_text:
        features.update(_tsv_features(tsv_text))
    return features


def _image_features(image_path: Path) -> dict[str, Any]:
    features: dict[str, Any] = {
        "image_path_present": image_path.exists(),
        "image_file_size": image_path.stat().st_size if image_path.exists() else 0,
        "image_suffix": image_path.suffix.lower(),
    }
    if not image_path.exists():
        return features
    try:
        from PIL import Image, ImageOps, ImageStat
    except ImportError:
        features["image_feature_status"] = "pillow_missing"
        return features

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            gray = ImageOps.grayscale(image)
            sample = gray.resize((min(width, 64), min(height, 64)))
            pixels = list(sample.getdata())
            stat = ImageStat.Stat(sample)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            features.update(
                {
                    "image_width": width,
                    "image_height": height,
                    "image_aspect_ratio": _safe_div(width, height),
                    "image_megapixels": (width * height) / 1_000_000,
                    "image_brightness": mean / 255,
                    "image_contrast": stddev / 255,
                    "image_sharpness_proxy": _sharpness_proxy(pixels, sample.size[0]),
                    "image_feature_status": "ok",
                }
            )
    except Exception as exc:
        features["image_feature_status"] = "error"
        features["image_feature_error"] = str(exc)
    return features


def _tsv_features(tsv_text: str) -> dict[str, Any]:
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


def _artifact_text(connection: Any, artifact_id: Any) -> str:
    if artifact_id in (None, ""):
        return ""
    row = connection.execute("SELECT path FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    if row is None:
        return ""
    path = Path(row["path"])
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _exact_match(truth: dict[str, Any], prediction: dict[str, Any]) -> bool:
    return all(_clean(truth.get(column, "")) == _clean(prediction.get(column, "")) for column in CSV_COLUMNS)


def _sharpness_proxy(pixels: list[int], width: int) -> float:
    if len(pixels) < 2 or width < 2:
        return 0.0
    diffs: list[int] = []
    for index, value in enumerate(pixels):
        if index % width != width - 1:
            diffs.append(abs(value - pixels[index + 1]))
        if index + width < len(pixels):
            diffs.append(abs(value - pixels[index + width]))
    return _safe_div(sum(diffs), len(diffs)) / 255


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _variance(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    return sum((value - mean) ** 2 for value in values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _loads(value: str) -> dict[str, Any]:
    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}
