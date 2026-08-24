from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage import DEFAULT_DB_PATH, DEFAULT_LABEL_SET
from .dataset import RouterDataset, RouterRecord, load_router_dataset
from .metrics import binary_metrics, routing_metrics


DEFAULT_MODEL_FILENAME = "model.joblib"
DEFAULT_TARGET_F1_THRESHOLD = 0.95
DEFAULT_CHEAP_COST = 1.0
DEFAULT_VLM_COST = 10.0
DEFAULT_LAMBDA_COST = 0.01
BLOCKED_FEATURE_KEYS = {
    "text",
    "ocr_text",
    "raw_text",
    "truth",
    "prediction",
    "fields_json",
    "ground_truth",
    "dateiname",
    "filename_stem",
    "image_path",
    "name_hint",
    "bemerkungen",
    "zusaetzliche_hinweise",
    "confidence_score",
}


@dataclass(frozen=True)
class RouterModelBundle:
    pipeline: Any
    threshold: float
    target_f1_threshold: float
    cheap_cost: float
    vlm_cost: float
    lambda_cost: float
    feature_schema: dict[str, Any]
    training_report: dict[str, Any]


@dataclass(frozen=True)
class RouterTrainingSummary:
    model_dir: Path
    training_rows: int
    validation_rows: int
    threshold: float
    validation_escalation_rate: float
    validation_failure_rate: float


def train_router_from_db(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    label_set: str = DEFAULT_LABEL_SET,
    feature_set: str,
    model_dir: Path,
    split_name: str = "",
    target_f1_threshold: float = DEFAULT_TARGET_F1_THRESHOLD,
    validation_ratio: float = 0.2,
    seed: int = 0,
    cheap_cost: float = DEFAULT_CHEAP_COST,
    vlm_cost: float = DEFAULT_VLM_COST,
    lambda_cost: float = DEFAULT_LAMBDA_COST,
    min_train_rows: int = 4,
) -> RouterTrainingSummary:
    dataset = load_router_dataset(
        db_path=db_path,
        label_set=label_set,
        feature_set=feature_set,
        target_f1_threshold=target_f1_threshold,
        split_name=split_name,
        require_labels=True,
    )
    train_records, validation_records, split_strategy = _split_records(
        list(dataset.records),
        validation_ratio=validation_ratio,
        seed=seed,
    )
    _validate_training_records(train_records, min_train_rows=min_train_rows)

    pipeline = _build_pipeline(seed)
    pipeline.fit(
        [_model_features(record) for record in train_records],
        [_target(record) for record in train_records],
    )

    train_probabilities = predict_failure_probabilities(pipeline, train_records)
    validation_probabilities = predict_failure_probabilities(pipeline, validation_records)
    fallback_vlm_failure_rate = _fallback_vlm_failure_rate(
        train_records + validation_records,
        target_f1_threshold=target_f1_threshold,
    )
    threshold, threshold_metrics = _select_threshold(
        validation_records,
        validation_probabilities,
        target_f1_threshold=target_f1_threshold,
        cheap_cost=cheap_cost,
        vlm_cost=vlm_cost,
        lambda_cost=lambda_cost,
        fallback_vlm_failure_rate=fallback_vlm_failure_rate,
    )
    train_binary = binary_metrics(
        [_target(record) for record in train_records],
        train_probabilities,
        threshold=threshold,
    )
    validation_binary = binary_metrics(
        [_target(record) for record in validation_records],
        validation_probabilities,
        threshold=threshold,
    )
    validation_routing = routing_metrics(
        validation_records,
        validation_probabilities,
        threshold=threshold,
        target_f1_threshold=target_f1_threshold,
        cheap_cost=cheap_cost,
        vlm_cost=vlm_cost,
        lambda_cost=lambda_cost,
        fallback_vlm_failure_rate=fallback_vlm_failure_rate,
    )
    feature_schema = _feature_schema(pipeline, train_records)
    report = {
        "label_set": dataset.label_set,
        "feature_set": dataset.feature_set,
        "split_name": dataset.split_name,
        "split_strategy": split_strategy,
        "target_f1_threshold": target_f1_threshold,
        "rows": {
            "loaded": len(dataset.records),
            "train": len(train_records),
            "validation": len(validation_records),
            "missing_features": dataset.missing_features,
            "no_evaluated_fields": dataset.no_evaluated_fields,
            "missing_cheap_predictions": dataset.missing_cheap_predictions,
            "missing_vlm_predictions": dataset.missing_vlm_predictions,
        },
        "costs": {
            "cheap_cost": cheap_cost,
            "vlm_cost": vlm_cost,
            "lambda_cost": lambda_cost,
            "fallback_vlm_failure_rate": fallback_vlm_failure_rate,
        },
        "threshold": threshold,
        "train_binary_metrics": train_binary.to_dict(),
        "validation_binary_metrics": validation_binary.to_dict(),
        "validation_routing_metrics": validation_routing.to_dict(),
        "threshold_search": [metrics.to_dict() for metrics in threshold_metrics],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _dump_model(pipeline, model_dir / DEFAULT_MODEL_FILENAME)
    _write_json(model_dir / "training-report.json", report)
    _write_json(model_dir / "feature-schema.json", feature_schema)
    _write_json(
        model_dir / "thresholds.json",
        {
            "threshold": threshold,
            "target_f1_threshold": target_f1_threshold,
            "cheap_cost": cheap_cost,
            "vlm_cost": vlm_cost,
            "lambda_cost": lambda_cost,
            "fallback_vlm_failure_rate": fallback_vlm_failure_rate,
        },
    )
    return RouterTrainingSummary(
        model_dir=model_dir,
        training_rows=len(train_records),
        validation_rows=len(validation_records),
        threshold=threshold,
        validation_escalation_rate=validation_routing.escalation_rate,
        validation_failure_rate=validation_routing.realized_failure_rate,
    )


def load_router_model(model_dir: Path) -> RouterModelBundle:
    pipeline = _load_model(model_dir / DEFAULT_MODEL_FILENAME)
    thresholds = _read_json(model_dir / "thresholds.json")
    feature_schema = _read_json(model_dir / "feature-schema.json")
    training_report = _read_json(model_dir / "training-report.json")
    return RouterModelBundle(
        pipeline=pipeline,
        threshold=float(thresholds["threshold"]),
        target_f1_threshold=float(thresholds["target_f1_threshold"]),
        cheap_cost=float(thresholds["cheap_cost"]),
        vlm_cost=float(thresholds["vlm_cost"]),
        lambda_cost=float(thresholds["lambda_cost"]),
        feature_schema=feature_schema,
        training_report=training_report,
    )


def predict_failure_probabilities(model: Any, records: list[RouterRecord]) -> list[float]:
    if not records:
        return []
    probabilities = model.predict_proba([_model_features(record) for record in records])
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("router model does not expose the failure class")
    failure_index = classes.index(1)
    return [float(row[failure_index]) for row in probabilities]


def model_record_features(record: RouterRecord) -> dict[str, Any]:
    return _model_features(record)


def _build_pipeline(seed: int) -> Any:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for router training. Install project dependencies first."
        ) from exc

    return Pipeline(
        [
            ("features", DictVectorizer(sparse=False)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _split_records(
    records: list[RouterRecord],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[RouterRecord], list[RouterRecord], str]:
    if not records:
        return [], [], "empty"
    if validation_ratio < 0 or validation_ratio >= 1:
        raise ValueError("validation_ratio must be >= 0 and < 1")

    subsets = {record.subset for record in records if record.subset}
    if subsets:
        train = [record for record in records if record.subset == "train"]
        validation = [record for record in records if record.subset == "validation"]
        if not validation:
            validation = [record for record in records if record.subset == "test"]
        return train, validation or train, "dataset_split"

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_ratio))
    if len(shuffled) > 1 and validation_count == 0:
        validation_count = 1
    if validation_count >= len(shuffled):
        validation_count = len(shuffled) - 1
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return train, validation or train, "deterministic_document"


def _validate_training_records(records: list[RouterRecord], *, min_train_rows: int) -> None:
    if len(records) < min_train_rows:
        raise ValueError(
            f"Router training requires at least {min_train_rows} training rows; "
            f"found {len(records)}. Add more ground-truth labels, feature snapshots, and paired outputs."
        )
    classes = {_target(record) for record in records}
    if classes != {0, 1}:
        raise ValueError(
            "Router training requires both cheap-pipeline success and failure examples; "
            f"found classes {sorted(classes)}."
        )


def _target(record: RouterRecord) -> int:
    if record.cheap_pipeline_failed is None:
        raise ValueError(f"record {record.document_id} has no router target label")
    return 1 if record.cheap_pipeline_failed else 0


def _select_threshold(
    records: list[RouterRecord],
    probabilities: list[float],
    *,
    target_f1_threshold: float,
    cheap_cost: float,
    vlm_cost: float,
    lambda_cost: float,
    fallback_vlm_failure_rate: float,
) -> tuple[float, list[Any]]:
    candidates = sorted({0.0, 0.5, 1.0, *probabilities})
    metrics = [
        routing_metrics(
            records,
            probabilities,
            threshold=threshold,
            target_f1_threshold=target_f1_threshold,
            cheap_cost=cheap_cost,
            vlm_cost=vlm_cost,
            lambda_cost=lambda_cost,
            fallback_vlm_failure_rate=fallback_vlm_failure_rate,
        )
        for threshold in candidates
    ]
    best = min(metrics, key=lambda item: (item.objective, item.escalation_rate, -item.threshold))
    return best.threshold, metrics


def _fallback_vlm_failure_rate(
    records: list[RouterRecord],
    *,
    target_f1_threshold: float,
) -> float:
    vlm_failures = [
        1.0 if record.vlm_metrics.field_f1 < target_f1_threshold else 0.0
        for record in records
        if record.vlm_metrics is not None and record.vlm_metrics.evaluated_fields > 0
    ]
    if not vlm_failures:
        return 0.0
    return sum(vlm_failures) / len(vlm_failures)


def _model_features(record: RouterRecord) -> dict[str, Any]:
    features = dict(record.features)
    sanitized: dict[str, Any] = {}
    for key, value in sorted(features.items()):
        clean_key = str(key)
        if clean_key.lower() in BLOCKED_FEATURE_KEYS:
            continue
        if value is None or value == "":
            sanitized[f"{clean_key}__missing"] = 1.0
            continue
        if isinstance(value, bool):
            sanitized[clean_key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                sanitized[clean_key] = numeric_value
            else:
                sanitized[f"{clean_key}__missing"] = 1.0
        elif isinstance(value, str):
            sanitized[clean_key] = value
        else:
            sanitized[clean_key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return sanitized


def _feature_schema(model: Any, records: list[RouterRecord]) -> dict[str, Any]:
    input_keys = sorted({key for record in records for key in _model_features(record)})
    vectorizer = model.named_steps["features"]
    output_features = list(vectorizer.get_feature_names_out())
    return {
        "input_feature_keys": input_keys,
        "vectorized_feature_count": len(output_features),
        "vectorized_feature_hash": hashlib.sha256(
            "\n".join(output_features).encode("utf-8")
        ).hexdigest(),
        "vectorized_features": output_features,
    }


def _dump_model(model: Any, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to persist the router model.") from exc
    joblib.dump(model, path)


def _load_model(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the router model.") from exc
    return joblib.load(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
