from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .labels import safe_div

if TYPE_CHECKING:
    from .dataset import RouterRecord


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    auroc: float | None
    auprc: float | None
    expected_calibration_error: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "expected_calibration_error": self.expected_calibration_error,
        }


@dataclass(frozen=True)
class RoutingMetrics:
    threshold: float
    documents: int
    escalations: int
    escalation_rate: float
    average_cost: float
    cost_per_1000: float
    realized_failure_rate: float
    routed_field_f1: float | None
    objective: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "documents": self.documents,
            "escalations": self.escalations,
            "escalation_rate": self.escalation_rate,
            "average_cost": self.average_cost,
            "cost_per_1000": self.cost_per_1000,
            "realized_failure_rate": self.realized_failure_rate,
            "routed_field_f1": self.routed_field_f1,
            "objective": self.objective,
        }


def binary_metrics(
    y_true: list[int],
    probabilities: list[float],
    *,
    threshold: float,
) -> BinaryMetrics:
    predicted = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for y, pred in zip(y_true, predicted) if y == 1 and pred == 1)
    fp = sum(1 for y, pred in zip(y_true, predicted) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(y_true, predicted) if y == 1 and pred == 0)
    correct = sum(1 for y, pred in zip(y_true, predicted) if y == pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    auroc, auprc = _ranking_metrics(y_true, probabilities)
    return BinaryMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=safe_div(correct, len(y_true)),
        auroc=auroc,
        auprc=auprc,
        expected_calibration_error=expected_calibration_error(y_true, probabilities),
    )


def routing_metrics(
    records: list["RouterRecord"],
    probabilities: list[float],
    *,
    threshold: float,
    target_f1_threshold: float,
    cheap_cost: float,
    vlm_cost: float,
    lambda_cost: float,
    fallback_vlm_failure_rate: float,
) -> RoutingMetrics:
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must have the same length")
    if not records:
        return RoutingMetrics(
            threshold=threshold,
            documents=0,
            escalations=0,
            escalation_rate=0.0,
            average_cost=0.0,
            cost_per_1000=0.0,
            realized_failure_rate=0.0,
            routed_field_f1=None,
            objective=0.0,
        )

    failures: list[float] = []
    costs: list[float] = []
    selected_f1_values: list[float] = []
    escalations = 0
    for record, probability in zip(records, probabilities):
        use_vlm = probability >= threshold
        if use_vlm:
            escalations += 1
            costs.append(vlm_cost)
            if record.vlm_metrics is None or record.vlm_metrics.evaluated_fields == 0:
                failures.append(fallback_vlm_failure_rate)
            else:
                failures.append(1.0 if record.vlm_metrics.field_f1 < target_f1_threshold else 0.0)
                selected_f1_values.append(record.vlm_metrics.field_f1)
        else:
            costs.append(cheap_cost)
            failures.append(1.0 if record.cheap_pipeline_failed else 0.0)
            if record.cheap_metrics is not None and record.cheap_metrics.evaluated_fields > 0:
                selected_f1_values.append(record.cheap_metrics.field_f1)

    average_cost = sum(costs) / len(costs)
    realized_failure_rate = sum(failures) / len(failures)
    return RoutingMetrics(
        threshold=threshold,
        documents=len(records),
        escalations=escalations,
        escalation_rate=safe_div(escalations, len(records)),
        average_cost=average_cost,
        cost_per_1000=average_cost * 1000,
        realized_failure_rate=realized_failure_rate,
        routed_field_f1=(
            sum(selected_f1_values) / len(selected_f1_values) if selected_f1_values else None
        ),
        objective=realized_failure_rate + lambda_cost * average_cost,
    )


def expected_calibration_error(
    y_true: list[int],
    probabilities: list[float],
    *,
    bins: int = 10,
) -> float | None:
    if not y_true:
        return None
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            (truth, probability)
            for truth, probability in zip(y_true, probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not bucket:
            continue
        accuracy = sum(truth for truth, _ in bucket) / len(bucket)
        confidence = sum(probability for _, probability in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - confidence)
    return ece


def _ranking_metrics(y_true: list[int], probabilities: list[float]) -> tuple[float | None, float | None]:
    if len(set(y_true)) < 2:
        return None, None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return None, None
    return float(roc_auc_score(y_true, probabilities)), float(
        average_precision_score(y_true, probabilities)
    )
