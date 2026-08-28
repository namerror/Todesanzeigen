from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import STORED_COLUMNS
from .storage import (
    DEFAULT_DB_PATH,
    apply_migrations,
    all_documents,
    connect,
    create_dataset_split,
    ground_truth_rows,
    insert_evaluation_result,
    insert_evaluation_run,
    latest_extraction_by_method,
)


@dataclass(frozen=True)
class SplitSummary:
    name: str
    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class EvaluationSummary:
    evaluation_run_id: int
    documents: int
    exact_record_accuracy: float
    field_precision: float
    field_recall: float
    field_f1: float
    missing_predictions: int


def create_source_year_split(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    name: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> SplitSummary:
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    apply_migrations(db_path)
    with connect(db_path) as connection:
        documents = all_documents(connection)
        groups: dict[str, list[int]] = {}
        for document in documents:
            group_key = f"{document['source']}:{document['year'] or 'unknown'}"
            groups.setdefault(group_key, []).append(int(document["id"]))

        assignments: dict[int, str] = {}
        for group_key, document_ids in groups.items():
            bucket = _bucket(group_key, seed, train_ratio, validation_ratio)
            for document_id in document_ids:
                assignments[document_id] = bucket

        create_dataset_split(
            connection,
            name=name,
            strategy="source-year",
            assignments=assignments,
            config={
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "seed": seed,
            },
        )
    return SplitSummary(
        name=name,
        train=sum(1 for subset in assignments.values() if subset == "train"),
        validation=sum(1 for subset in assignments.values() if subset == "validation"),
        test=sum(1 for subset in assignments.values() if subset == "test"),
    )


def evaluate_method(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    label_set: str,
    method: str,
    split_name: str = "",
    name: str | None = None,
) -> EvaluationSummary:
    apply_migrations(db_path)
    with connect(db_path) as connection:
        labels = ground_truth_rows(connection, label_set=label_set, split_name=split_name)
        totals = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact": 0,
            "documents": len(labels),
            "missing_predictions": 0,
        }
        per_document: list[dict[str, Any]] = []
        for label in labels:
            truth = _loads(label["fields_json"])
            prediction_row = latest_extraction_by_method(
                connection,
                document_id=int(label["document_id"]),
                method=method,
            )
            if prediction_row is None:
                totals["missing_predictions"] += 1
                field_results = {
                    column: {
                        "truth": _clean(truth.get(column, "")),
                        "prediction": "",
                        "correct": False,
                    }
                    for column in STORED_COLUMNS
                    if _clean(truth.get(column, ""))
                }
                per_document.append(
                    {
                        "document_id": int(label["document_id"]),
                        "extraction_output_id": None,
                        "exact_match": False,
                        "field_results": field_results,
                    }
                )
                totals["fn"] += len(field_results)
                continue

            prediction = _loads(prediction_row["fields_json"])
            field_results, exact = _compare_fields(truth, prediction)
            counts = _field_counts(field_results)
            totals["tp"] += counts["tp"]
            totals["fp"] += counts["fp"]
            totals["fn"] += counts["fn"]
            totals["exact"] += 1 if exact else 0
            per_document.append(
                {
                    "document_id": int(label["document_id"]),
                    "extraction_output_id": int(prediction_row["id"]),
                    "exact_match": exact,
                    "field_results": field_results,
                }
            )

        precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
        recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        exact_accuracy = _safe_div(totals["exact"], totals["documents"])
        metrics = {
            "documents": totals["documents"],
            "exact_record_accuracy": exact_accuracy,
            "field_precision": precision,
            "field_recall": recall,
            "field_f1": f1,
            "missing_predictions": totals["missing_predictions"],
        }
        evaluation_run_id = insert_evaluation_run(
            connection,
            name=name or f"{method}:{label_set}",
            label_set=label_set,
            method=method,
            split_name=split_name,
            config={},
            metrics=metrics,
        )
        for result in per_document:
            insert_evaluation_result(
                connection,
                evaluation_run_id=evaluation_run_id,
                document_id=result["document_id"],
                extraction_output_id=result["extraction_output_id"],
                exact_match=result["exact_match"],
                field_results=result["field_results"],
            )
    return EvaluationSummary(
        evaluation_run_id=evaluation_run_id,
        documents=totals["documents"],
        exact_record_accuracy=exact_accuracy,
        field_precision=precision,
        field_recall=recall,
        field_f1=f1,
        missing_predictions=totals["missing_predictions"],
    )


def _compare_fields(
    truth: dict[str, Any],
    prediction: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    field_results: dict[str, dict[str, Any]] = {}
    exact = True
    for column in STORED_COLUMNS:
        truth_value = _clean(truth.get(column, ""))
        prediction_value = _clean(prediction.get(column, ""))
        correct = truth_value == prediction_value
        if not correct:
            exact = False
        field_results[column] = {
            "truth": truth_value,
            "prediction": prediction_value,
            "correct": correct,
        }
    return field_results, exact


def _field_counts(field_results: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for result in field_results.values():
        truth = result["truth"]
        prediction = result["prediction"]
        if truth and prediction and result["correct"]:
            counts["tp"] += 1
        elif truth and prediction:
            counts["fp"] += 1
            counts["fn"] += 1
        elif prediction:
            counts["fp"] += 1
        elif truth:
            counts["fn"] += 1
    return counts


def _bucket(
    group_key: str,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 0.000001:
        raise ValueError("split ratios must sum to 1.0")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _loads(value: str) -> dict[str, Any]:
    import json

    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}
