from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ROUTER_TARGET_FIELDS = (
    "geschlecht",
    "name",
    "vorname",
    "geburtsdatum",
    "sterbedatum",
    "geburtsname",
    "titel",
    "genannt",
    "geburtsort",
    "sterbeort",
    "ort",
    "weitere_orte",
    "beruf",
)


@dataclass(frozen=True)
class FieldScore:
    field: str
    truth: str
    prediction: str
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetFieldMetrics:
    evaluated_fields: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    field_f1: float
    exact_target_match: bool
    field_scores: tuple[FieldScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_fields": self.evaluated_fields,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "field_f1": self.field_f1,
            "exact_target_match": self.exact_target_match,
            "field_scores": [score.to_dict() for score in self.field_scores],
        }


def score_target_fields(
    truth: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    target_fields: Iterable[str] = ROUTER_TARGET_FIELDS,
) -> TargetFieldMetrics:
    """Score only populated target fields.

    Blank ground-truth target fields are treated as unavailable labels, not as
    required empty outputs. Non-target columns are never evaluated here.
    """

    prediction = prediction or {}
    field_scores: list[FieldScore] = []
    tp = fp = fn = 0
    exact = True
    for field in target_fields:
        truth_value = clean_value(truth.get(field, ""))
        if not truth_value:
            continue
        prediction_value = clean_value(prediction.get(field, ""))
        correct = truth_value == prediction_value
        if correct:
            tp += 1
        elif prediction_value:
            fp += 1
            fn += 1
            exact = False
        else:
            fn += 1
            exact = False
        field_scores.append(
            FieldScore(
                field=field,
                truth=truth_value,
                prediction=prediction_value,
                correct=correct,
            )
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    field_f1 = safe_div(2 * precision * recall, precision + recall)
    evaluated_fields = len(field_scores)
    return TargetFieldMetrics(
        evaluated_fields=evaluated_fields,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        field_f1=field_f1,
        exact_target_match=exact and evaluated_fields > 0,
        field_scores=tuple(field_scores),
    )


def cheap_pipeline_failed(
    truth: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    target_f1_threshold: float,
    missing_prediction: bool = False,
) -> bool | None:
    metrics = score_target_fields(truth, prediction)
    if metrics.evaluated_fields == 0:
        return None
    if missing_prediction:
        return True
    return metrics.field_f1 < target_f1_threshold


def clean_value(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
