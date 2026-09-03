from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage import (
    DEFAULT_DB_PATH,
    DEFAULT_LABEL_SET,
    active_extraction_for_variant,
    all_documents,
    apply_migrations,
    connect,
    latest_feature_snapshot,
)
from ..variants import DEFAULT_VARIANTS_CONFIG_PATH, load_variant_config
from .labels import TargetFieldMetrics, score_target_fields


@dataclass(frozen=True)
class RouterRecord:
    document_id: int
    source: str
    filename_stem: str
    image_path: str
    year: int | None
    features: dict[str, Any]
    truth: dict[str, Any] | None
    cheap_fields: dict[str, Any] | None
    vlm_fields: dict[str, Any] | None
    cheap_output_id: int | None
    vlm_output_id: int | None
    cheap_metrics: TargetFieldMetrics | None
    vlm_metrics: TargetFieldMetrics | None
    cheap_pipeline_failed: bool | None
    subset: str = ""


@dataclass(frozen=True)
class RouterDataset:
    records: tuple[RouterRecord, ...]
    label_set: str
    feature_set: str
    split_name: str
    missing_features: int
    no_evaluated_fields: int
    missing_cheap_predictions: int
    missing_vlm_predictions: int
    text_variant_alias: str
    vlm_variant_alias: str
    text_variant: dict[str, str]
    vlm_variant: dict[str, str]


def load_router_dataset(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    label_set: str = DEFAULT_LABEL_SET,
    feature_set: str,
    target_f1_threshold: float,
    split_name: str = "",
    require_labels: bool = True,
    variants_config: Path = DEFAULT_VARIANTS_CONFIG_PATH,
) -> RouterDataset:
    variant_config = load_variant_config(variants_config)
    text_variant = variant_config.variant(variant_config.text_default)
    vlm_variant = variant_config.variant(variant_config.vlm_default)
    apply_migrations(db_path)
    records: list[RouterRecord] = []
    missing_features = 0
    no_evaluated_fields = 0
    missing_cheap_predictions = 0
    missing_vlm_predictions = 0

    with connect(db_path) as connection:
        truth_by_document = _truth_by_document(connection, label_set)
        subset_by_document = _subset_by_document(connection, split_name) if split_name else {}
        documents = all_documents(connection)
        for document in documents:
            document_id = int(document["id"])
            truth = truth_by_document.get(document_id)
            if require_labels and truth is None:
                continue

            feature_snapshot = latest_feature_snapshot(
                connection,
                document_id=document_id,
                feature_set=feature_set,
            )
            if feature_snapshot is None:
                missing_features += 1
                continue

            cheap_output = active_extraction_for_variant(
                connection,
                document_id=document_id,
                method=text_variant.method,
                provider=text_variant.provider,
                model=text_variant.model,
                prompt_version=text_variant.prompt_version,
            )
            vlm_output = active_extraction_for_variant(
                connection,
                document_id=document_id,
                method=vlm_variant.method,
                provider=vlm_variant.provider,
                model=vlm_variant.model,
                prompt_version=vlm_variant.prompt_version,
            )
            cheap_fields = _loads(cheap_output["fields_json"]) if cheap_output else None
            vlm_fields = _loads(vlm_output["fields_json"]) if vlm_output else None
            cheap_metrics = score_target_fields(truth, cheap_fields) if truth is not None else None
            vlm_metrics = score_target_fields(truth, vlm_fields) if truth is not None else None
            failed: bool | None = None
            if cheap_metrics is not None:
                if cheap_metrics.evaluated_fields == 0:
                    no_evaluated_fields += 1
                    if require_labels:
                        continue
                else:
                    failed = True if cheap_output is None else cheap_metrics.field_f1 < target_f1_threshold
            if cheap_output is None:
                missing_cheap_predictions += 1
            if vlm_output is None:
                missing_vlm_predictions += 1

            records.append(
                RouterRecord(
                    document_id=document_id,
                    source=str(document["source"]),
                    filename_stem=str(document["filename_stem"]),
                    image_path=str(document["image_path"]),
                    year=document["year"],
                    features=_loads(feature_snapshot["features_json"]),
                    truth=truth,
                    cheap_fields=cheap_fields,
                    vlm_fields=vlm_fields,
                    cheap_output_id=int(cheap_output["id"]) if cheap_output else None,
                    vlm_output_id=int(vlm_output["id"]) if vlm_output else None,
                    cheap_metrics=cheap_metrics,
                    vlm_metrics=vlm_metrics,
                    cheap_pipeline_failed=failed,
                    subset=subset_by_document.get(document_id, ""),
                )
            )

    return RouterDataset(
        records=tuple(records),
        label_set=label_set,
        feature_set=feature_set,
        split_name=split_name,
        missing_features=missing_features,
        no_evaluated_fields=no_evaluated_fields,
        missing_cheap_predictions=missing_cheap_predictions,
        missing_vlm_predictions=missing_vlm_predictions,
        text_variant_alias=text_variant.alias,
        vlm_variant_alias=vlm_variant.alias,
        text_variant=text_variant.as_dict(),
        vlm_variant=vlm_variant.as_dict(),
    )


def _truth_by_document(connection: Any, label_set: str) -> dict[int, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT document_id, fields_json
        FROM ground_truth_labels
        WHERE label_set = ?
        """,
        (label_set,),
    ).fetchall()
    return {int(row["document_id"]): _loads(row["fields_json"]) for row in rows}


def _subset_by_document(connection: Any, split_name: str) -> dict[int, str]:
    rows = connection.execute(
        """
        SELECT dataset_memberships.document_id, dataset_memberships.subset
        FROM dataset_memberships
        JOIN dataset_splits ON dataset_splits.id = dataset_memberships.split_id
        WHERE dataset_splits.name = ?
        """,
        (split_name,),
    ).fetchall()
    return {int(row["document_id"]): str(row["subset"]) for row in rows}


def _loads(value: str | bytes | None) -> dict[str, Any]:
    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}
