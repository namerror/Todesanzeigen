from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage import DEFAULT_DB_PATH, DEFAULT_LABEL_SET
from ..variants import DEFAULT_VARIANTS_CONFIG_PATH
from .dataset import load_router_dataset
from .model import load_router_model, predict_failure_probabilities


@dataclass(frozen=True)
class RouterManifestSummary:
    output_file: Path
    rows: int
    escalations: int
    missing_features: int


def write_router_manifest(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    model_dir: Path,
    output_file: Path,
    label_set: str = DEFAULT_LABEL_SET,
    feature_set: str,
    threshold: float | None = None,
    variants_config: Path = DEFAULT_VARIANTS_CONFIG_PATH,
) -> RouterManifestSummary:
    bundle = load_router_model(model_dir)
    decision_threshold = bundle.threshold if threshold is None else threshold
    dataset = load_router_dataset(
        db_path=db_path,
        label_set=label_set,
        feature_set=feature_set,
        target_f1_threshold=bundle.target_f1_threshold,
        require_labels=False,
        variants_config=variants_config,
    )
    records = list(dataset.records)
    probabilities = predict_failure_probabilities(bundle.pipeline, records)
    rows = [
        _manifest_row(
            record,
            probability=probability,
            threshold=decision_threshold,
            cheap_cost=bundle.cheap_cost,
            vlm_cost=bundle.vlm_cost,
            text_variant=dataset.text_variant,
            vlm_variant=dataset.vlm_variant,
        )
        for record, probability in zip(records, probabilities)
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    escalations = sum(1 for row in rows if row["route"] == "vlm")
    return RouterManifestSummary(
        output_file=output_file,
        rows=len(rows),
        escalations=escalations,
        missing_features=dataset.missing_features,
    )


def _manifest_row(
    record: Any,
    *,
    probability: float,
    threshold: float,
    cheap_cost: float,
    vlm_cost: float,
    text_variant: dict[str, str],
    vlm_variant: dict[str, str],
) -> dict[str, Any]:
    route = "vlm" if probability >= threshold else "ocr_llm"
    row = {
        "document_id": record.document_id,
        "source": record.source,
        "filename_stem": record.filename_stem,
        "image_path": record.image_path,
        "predicted_failure_probability": probability,
        "route": route,
        "threshold": threshold,
        "expected_cost": vlm_cost if route == "vlm" else cheap_cost,
        "ocr_llm_output_id": record.cheap_output_id,
        "vlm_output_id": record.vlm_output_id,
        "ocr_llm_variant": text_variant,
        "vlm_variant": vlm_variant,
    }
    if record.cheap_metrics is not None:
        row["ocr_llm_target_metrics"] = record.cheap_metrics.to_dict()
        row["cheap_pipeline_failed"] = record.cheap_pipeline_failed
    if record.vlm_metrics is not None:
        row["vlm_target_metrics"] = record.vlm_metrics.to_dict()
    return row
