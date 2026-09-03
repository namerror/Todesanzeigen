from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .feature_extraction import (
    extract_image_features,
    extract_ocr_text_features,
    extract_tsv_features,
)
from .llm import STORED_COLUMNS
from .storage import (
    DEFAULT_DB_PATH,
    DEFAULT_LABEL_SET,
    active_extraction_for_variant,
    all_documents,
    apply_migrations,
    connect,
    latest_feature_snapshot,
    latest_ocr_output,
    upsert_feature_snapshot,
)
from .variants import DEFAULT_VARIANTS_CONFIG_PATH, load_variant_config


DEFAULT_FEATURE_SET = "router-v2"


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
                features.update(_ocr_features(dict(ocr), _ocr_tsv_text(ocr["tsv_path"])))
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
    variant_alias: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    variants_config: Path = DEFAULT_VARIANTS_CONFIG_PATH,
) -> RouterDatasetSummary:
    variant = load_variant_config(variants_config).variant(variant_alias)
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

            prediction_row = active_extraction_for_variant(
                connection,
                document_id=document_id,
                method=variant.method,
                provider=variant.provider,
                model=variant.model,
                prompt_version=variant.prompt_version,
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
                    "method": variant.method,
                    "variant_alias": variant.alias,
                    "variant": variant.as_dict(),
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
    features: dict[str, Any] = {"source": str(document.get("source", "") or "")}
    image_path = str(document.get("image_path", "") or "")
    if image_path:
        features.update(extract_image_features(Path(image_path)))
    return features


def _ocr_features(ocr: dict[str, Any], tsv_text: str) -> dict[str, Any]:
    text = str(ocr.get("text", "") or "")
    features = extract_ocr_text_features(
        text,
        name_hint=str(ocr.get("name_hint", "") or ""),
        name_confidence=ocr.get("name_confidence"),
    )
    if tsv_text:
        features.update(extract_tsv_features(tsv_text))
    return features


def _ocr_tsv_text(path_value: Any) -> str:
    if path_value in (None, ""):
        return ""
    path = Path(str(path_value))
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _exact_match(truth: dict[str, Any], prediction: dict[str, Any]) -> bool:
    return all(
        _clean(truth.get(column, "")) == _clean(prediction.get(column, ""))
        for column in STORED_COLUMNS
    )


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _loads(value: str) -> dict[str, Any]:
    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}
