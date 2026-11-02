"""Backend-independent extraction scoring. Blank reference values mean confirmed absence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inference_config import fingerprint
from .normalization import normalize_stored_fields
from .storage import active_extraction_for_variant, apply_migrations, connect
from .variants import DEFAULT_VARIANTS_CONFIG_PATH, load_variant_config

TARGET_FIELDS = (
    "geschlecht", "name", "vorname", "geburtsdatum", "sterbedatum", "geburtsname",
    "titel", "genannt", "geburtsort", "sterbeort", "ort", "weitere_orte", "beruf",
)


def _ratio(a, b):
    return a / b if b else None


def _metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {**counts, "accuracy": _ratio(counts["correct"], counts["evaluated"]),
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def score_records(references: dict[int, dict], predictions: dict[int, dict]) -> dict[str, Any]:
    per_field = {field: {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "evaluated": 0, "support": 0}
                 for field in TARGET_FIELDS}
    exact = missing = 0
    mismatches = []
    for document_id, truth in references.items():
        absent = document_id not in predictions
        missing += absent
        truth = normalize_stored_fields(truth)
        prediction = normalize_stored_fields(predictions.get(document_id, {}))
        record_exact = not absent
        for field, counts in per_field.items():
            clean = lambda value: " ".join(str(value or "").split())
            expected, actual = clean(truth.get(field)), clean(prediction.get(field))
            equal = not absent and expected == actual
            counts["evaluated"] += 1
            counts["support"] += bool(expected)
            counts["correct"] += equal
            if expected and actual and equal:
                counts["tp"] += 1
            else:
                counts["fp"] += bool(actual)
                counts["fn"] += bool(expected)
            if not equal:
                record_exact = False
                mismatches.append({"document_id": document_id, "field": field, "reference": expected,
                                   "prediction": actual, "missing_prediction": absent})
        exact += record_exact
    aggregate = {key: sum(c[key] for c in per_field.values()) for key in next(iter(per_field.values()))}
    return {"documents": len(references), "missing_predictions": missing,
            "exact_records": exact, "exact_record_accuracy": _ratio(exact, len(references)),
            "micro": _metrics(aggregate), "fields": {field: _metrics(c) for field, c in per_field.items()},
            "mismatches": mismatches, "status": "ok" if references else "no evaluable documents"}


def evaluate_comparison(
    *, db_path: Path, variant_aliases=(), run_id: str | None = None,
    label_set: str | None = None, reference_variant: str | None = None,
    variants_config: Path = DEFAULT_VARIANTS_CONFIG_PATH, split_name="", subset=None, document_ids=None,
) -> dict:
    from .local_vlm import select_documents, variant_lookup
    if label_set and reference_variant:
        raise ValueError("Choose GT labels or a reference variant, not both")
    if not variant_aliases and not run_id:
        raise ValueError("Select at least one --variant or --run-id")
    apply_migrations(db_path)
    label_set = label_set or (None if reference_variant else "gt-v1")
    config = load_variant_config(variants_config) if variant_aliases or reference_variant else None
    variants = [config.variant(alias) for alias in variant_aliases]
    reference = config.variant(reference_variant) if reference_variant else None
    with connect(db_path) as connection:
        documents = select_documents(connection, label_set=label_set if reference is None else None,
                                     split_name=split_name, subset=subset)
        if document_ids is not None:
            cohort = set(document_ids)
            documents = [d for d in documents if d["id"] in cohort]
        benchmark = None
        if run_id:
            row = connection.execute("SELECT * FROM inference_benchmark_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown benchmark run: {run_id}")
            benchmark = dict(row)
            manifest_ids = {item["document_id"] for item in json.loads(row["manifest_json"])}
            documents = [d for d in documents if d["id"] in manifest_ids]
        references, reference_metadata = {}, []
        missing_references = 0
        for document in documents:
            document_id = document["id"]
            if reference:
                row = active_extraction_for_variant(connection, document_id=document_id, **variant_lookup(reference))
            else:
                row = connection.execute("SELECT * FROM ground_truth_labels WHERE label_set=? AND document_id=?",
                                         (label_set or "gt-v1", document_id)).fetchone()
            if row is None:
                missing_references += 1
                continue
            fields = json.loads(row["fields_json"])
            references[document_id] = fields
            reference_metadata.append({"document_id": document_id, "row_id": row["id"], "content_hash": fingerprint(fields)})
        sources = [(v.alias, v) for v in variants] + ([("run:" + run_id, None)] if run_id else [])
        results = {}
        for name, variant in sources:
            predictions, output_ids = {}, {}
            for document_id in references:
                if variant is not None:
                    row = active_extraction_for_variant(connection, document_id=document_id, **variant_lookup(variant))
                else:
                    # Canonical outputs survive supersession and are selected by immutable sample links.
                    row = connection.execute(
                        "SELECT e.* FROM extraction_outputs e JOIN inference_benchmark_samples s ON s.extraction_output_id=e.id "
                        "WHERE s.run_id=? AND s.document_id=? AND s.phase='measured' AND s.status='success' ORDER BY s.id LIMIT 1",
                        (run_id, document_id)).fetchone()
                if row is not None:
                    predictions[document_id] = json.loads(row["fields_json"])
                    output_ids[str(document_id)] = row["id"]
            score = score_records(references, predictions)
            score["prediction_output_ids"] = output_ids
            score["identity"] = variant.as_dict() if variant else json.loads(benchmark["variant_json"])
            results[name] = score
    mode = "variant_agreement" if reference else "gt_accuracy"
    return {"mode": mode, "metric_note": "Reference-relative agreement is not GT accuracy" if reference else "Reviewed GT accuracy",
            "blank_reference_policy": "confirmed_absent", "target_fields": list(TARGET_FIELDS),
            "reference": reference.as_dict() if reference else {"label_set": label_set or "gt-v1"},
            "reference_rows": reference_metadata, "split": split_name, "subset": (subset or "test") if split_name else None,
            "cohort_documents": len(documents), "missing_references": missing_references,
            "results": results}
