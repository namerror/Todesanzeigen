"""Local extraction orchestration and immutable inference benchmark history."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from .extract import build_image_only_vision_extraction_prompt, normalize_record, parse_json_object
from .inference_config import canonical_json
from .llm import MODEL_RESPONSE_COLUMNS
from .methods import CURRENT_PROMPT_VERSION, LOCAL_VISION_METHOD
from .storage import (
    active_extraction_for_variant, all_documents, apply_migrations, connect, create_run,
    finish_run, insert_extraction_output, insert_label_candidate, sha256_file,
    supersede_active_extraction_for_variant, upsert_document,
)
from .variants import ExtractionVariant


def select_documents(connection, *, label_set=None, split_name="", subset=None, only=None, limit=None, image_backed=False, source=""):
    if subset and not split_name:
        raise ValueError("--subset requires --split")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    documents = [dict(row) for row in all_documents(connection)]
    if source:
        documents = [d for d in documents if d["source"] == source]
    if image_backed:
        documents = [d for d in documents if d["image_path"]]
    if label_set:
        ids = {row[0] for row in connection.execute(
            "SELECT document_id FROM ground_truth_labels WHERE label_set = ?", (label_set,))}
        documents = [d for d in documents if d["id"] in ids]
    if split_name:
        if connection.execute("SELECT 1 FROM dataset_splits WHERE name=?", (split_name,)).fetchone() is None:
            raise ValueError(f"Unknown split: {split_name}")
        ids = {row[0] for row in connection.execute(
            "SELECT m.document_id FROM dataset_memberships m JOIN dataset_splits s ON s.id=m.split_id "
            "WHERE s.name=? AND m.subset=?", (split_name, subset or "test"))}
        documents = [d for d in documents if d["id"] in ids]
    if only:
        documents = [d for d in documents if d["filename_stem"] in only or Path(d["image_path"]).name in only]
    return documents[:limit] if limit else documents


def select_images(connection, *, input_dir=None, limit=None, reference=None, **filters):
    """Validate registered paths before limiting; directory selection never relocates records."""
    from .ocr import discover_images
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    documents = select_documents(connection, **filters)
    directory_images = None
    if input_dir is not None:
        directory_images = {}
        for path in discover_images(input_dir):
            directory_images.setdefault(path.resolve(), []).append(path)
    available, invalid = [], []
    for document in documents:
        image = Path(document["image_path"]).resolve() if document["image_path"] else None
        matches = directory_images.get(image, []) if directory_images is not None else None
        if matches is not None and len(matches) > 1:
            raise ValueError(f"Ambiguous image matches for document {document['id']}: {matches}")
        try:
            if image is None:
                raise ValueError("No recorded image path")
            if matches is not None and not matches:
                raise FileNotFoundError("Recorded image path is not in --input-dir")
            if not image.is_file():
                raise FileNotFoundError(str(image))
            digest = sha256_file(image)
            if document["image_sha256"] and digest != document["image_sha256"]:
                raise ValueError("Stored image SHA-256 mismatch")
        except (OSError, ValueError) as exc:
            invalid.append({"document_id": document["id"], "image_path": document["image_path"], "reason": str(exc)})
            continue
        available.append((document, {"document_id": document["id"], "image_path": str(image), "image_sha256": digest}))
    selected = available[:limit] if limit else available
    summary = {"matching_documents": len(documents), "available_images": len(available),
               "missing_invalid_images": len(invalid), "selected_images": len(selected), "invalid_images": invalid}
    if reference is not None:
        covered = sum(active_extraction_for_variant(connection, document_id=d["id"], **variant_lookup(reference)) is not None
                      for d, _ in selected)
        summary["reference_coverage"] = {"available": covered, "missing": len(selected) - covered, "total": len(selected)}
    elif filters.get("label_set"):
        summary["reference_coverage"] = {"available": len(selected), "missing": 0, "total": len(selected)}
    return selected, summary


def ingest_images(db_path: Path, input_dir: Path, source: str):
    from .ocr import discover_images, image_mime_type
    with connect(db_path) as connection:
        for image in discover_images(input_dir):
            upsert_document(connection, source_name=source, filename_stem=image.stem,
                            image_path=image, image_sha256=sha256_file(image), mime_type=image_mime_type(image))


def variant_lookup(variant: ExtractionVariant) -> dict[str, str]:
    return {"method": variant.method, "provider": variant.provider, "model": variant.model,
            "prompt_version": variant.prompt_version, "inference_config_hash": variant.inference_config_hash}


def parse_prediction(raw: str, document: dict) -> tuple[dict | None, dict]:
    flags = {"json_valid": False, "schema_valid": False}
    try:
        # Strict JSON validity is tracked independently of the tolerant pipeline parser.
        strict = json.loads(raw)
        flags["json_valid"] = isinstance(strict, dict)
    except (TypeError, ValueError):
        pass
    try:
        data = parse_json_object(raw)
    except (TypeError, ValueError):
        return None, flags
    flags["schema_valid"] = (
        set(data) == set(MODEL_RESPONSE_COLUMNS)
        and all(isinstance(value, str) for value in data.values())
        and data.get("geschlecht") in {"", "männlich", "weiblich"}
    )
    if not flags["schema_valid"]:
        return None, flags
    return normalize_record(data, filename=document["filename_stem"], source=document["source"]), flags


def _persist_prediction(connection, *, document, variant, run_id, fields, raw, metrics, image_hash):
    supersede_active_extraction_for_variant(connection, document_id=document["id"], **variant_lookup(variant))
    output_id = insert_extraction_output(
        connection, document_id=document["id"], run_id=run_id, fields=fields,
        status="vision_processed", raw_response=raw, attempts=1,
        latency_ms=round(metrics["total_ms"]), config=variant.inference.as_dict(),
        input_fingerprint=image_hash, **variant_lookup(variant),
    )
    insert_label_candidate(connection, document_id=document["id"], fields=fields,
                           source_kind="pipeline", source_name=variant.alias, extraction_output_id=output_id)
    return output_id


def _code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_local(
    *, db_path: Path, variant: ExtractionVariant, benchmark=False, label_set=None,
    split_name="", subset=None, only=None, limit=None, input_dir=None, source="",
    force=False, warmups=1, repetitions=3, backend_factory=None, reference=None,
) -> dict[str, Any]:
    if variant.method != LOCAL_VISION_METHOD or variant.inference is None:
        raise ValueError("Select a local VLM variant with inference configuration")
    if variant.prompt_version != CURRENT_PROMPT_VERSION:
        raise ValueError("Local inference supports only the current extraction prompt version")
    if warmups < 0 or repetitions < 1:
        raise ValueError("Warm-ups must be nonnegative and repetitions positive")
    apply_migrations(db_path)
    if input_dir is not None and not benchmark:
        ingest_images(db_path, input_dir, source or input_dir.name)
    with connect(db_path) as connection:
        selected, summary = select_images(
            connection, input_dir=input_dir, limit=limit, reference=reference, source=source,
            label_set=label_set, split_name=split_name, subset=subset, only=only)
    print("Selection summary: " + canonical_json(summary), flush=True)
    if not selected:
        raise ValueError("No eligible images remain after selection and image validation")
    documents = [document for document, _ in selected]
    manifest = [entry for _, entry in selected]
    selection = {"selected_document_ids": [d["id"] for d in documents], "selection": summary}
    cached = 0
    pending = []
    with connect(db_path) as connection:
        for document, entry in zip(documents, manifest):
            old = active_extraction_for_variant(connection, document_id=document["id"], **variant_lookup(variant))
            if not benchmark and not force and old is not None and entry["image_sha256"] and old["input_fingerprint"] == entry["image_sha256"]:
                cached += 1
            else:
                pending.append((document, entry))
        if not pending:
            return {"cached": cached, "successful": 0, "failed": 0, "run_id": None, **selection}
        run_id = create_run(connection, command="local-vlm benchmark" if benchmark else "local-vlm extract",
                            method=variant.method, provider=variant.provider, model=variant.model,
                            config=variant.as_dict(), code_version=_code_revision())
        if benchmark:
            connection.execute(
                "INSERT INTO inference_benchmark_runs(id,variant_json,manifest_json,warmup_count,repetitions) VALUES (?,?,?,?,?)",
                (run_id, canonical_json(variant.as_dict()), canonical_json(manifest), warmups, repetitions))
    if backend_factory is None:
        from .local_backend import TransformersBackend
        backend_factory = TransformersBackend
    canonical_outputs = {}
    failures = successes = 0
    fatal = ""
    warmup_total = 0.0
    try:
        backend = backend_factory(variant.model, variant.inference)
        load_ms, environment = backend.load()
        if benchmark:
            with connect(db_path) as connection:
                connection.execute("UPDATE inference_benchmark_runs SET load_ms=?,environment_json=? WHERE id=?",
                                   (load_ms, canonical_json(environment), run_id))
        phases = [("warmup", warmups), ("measured", repetitions)] if benchmark else [("measured", 1)]
        for phase, count in phases:
            for repetition in range(count):
                selected = pending[:1] if phase == "warmup" else pending
                for document, entry in selected:
                    started = time.perf_counter()
                    raw, fields, metrics, error, status = "", None, {}, "", "failed"
                    oom = False
                    try:
                        image = Path(entry["image_path"])
                        if not image.is_file():
                            raise FileNotFoundError(str(image))
                        if sha256_file(image) != entry["image_sha256"]:
                            raise ValueError("Image changed since the benchmark manifest was captured")
                        prompt = build_image_only_vision_extraction_prompt(filename=document["filename_stem"], source=document["source"])
                        raw, metrics = backend.infer(prompt, image)
                        fields, flags = parse_prediction(raw, document)
                        metrics.update(flags)
                        if fields is None:
                            raise ValueError("Response could not be parsed as a JSON object")
                        status = "success"
                    except Exception as exc:
                        error = str(exc)
                        oom = "outofmemory" in type(exc).__name__.lower() or "out of memory" in error.lower()
                        if oom:
                            fatal = error
                    metrics["total_ms"] = (time.perf_counter() - started) * 1000
                    if phase == "warmup":
                        warmup_total += metrics["total_ms"]
                    else:
                        successes += status == "success"
                        failures += status != "success"
                    with connect(db_path) as connection:
                        if phase == "measured" and fields is not None and document["id"] not in canonical_outputs:
                            canonical_outputs[document["id"]] = _persist_prediction(
                                connection, document=document, variant=variant, run_id=run_id,
                                fields=fields, raw=raw, metrics=metrics, image_hash=entry["image_sha256"])
                        if benchmark:
                            connection.execute(
                                "INSERT INTO inference_benchmark_samples(run_id,document_id,repetition,phase,status,raw_response,fields_json,metrics_json,error,extraction_output_id) "
                                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (run_id, document["id"], repetition, phase, status, raw,
                                 canonical_json(fields or {}), canonical_json(metrics), error,
                                 canonical_outputs.get(document["id"]) if phase == "measured" and status == "success" else None))
                        elif status != "success":
                            insert_extraction_output(connection, document_id=document["id"], run_id=run_id,
                                                     status="vision_failed", fields=None, raw_response=raw, error=error,
                                                     input_fingerprint=entry["image_sha256"], config=variant.inference.as_dict(),
                                                     **variant_lookup(variant))
                    if oom:
                        raise RuntimeError(error)
    except KeyboardInterrupt:
        fatal = "Interrupted by user"
    except Exception as exc:
        fatal = str(exc)
    finally:
        status = "failed" if fatal else ("completed_with_errors" if failures else "completed")
        with connect(db_path) as connection:
            finish_run(connection, run_id, status=status)
            if benchmark:
                connection.execute("UPDATE inference_benchmark_runs SET status=?,error=?,warmup_ms=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
                                   (status, fatal, warmup_total, run_id))
    result = {"run_id": run_id, "status": status, "successful": successes, "failed": failures, "cached": cached, "error": fatal, **selection}
    if benchmark:
        result["benchmark"] = benchmark_report(db_path, run_id)
    return result


def _stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    values = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)]}


def benchmark_report(db_path: Path, run_id: str) -> dict:
    with connect(db_path) as connection:
        row = connection.execute("SELECT * FROM inference_benchmark_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown benchmark run: {run_id}")
        run = dict(row)
        samples = [dict(s) for s in connection.execute(
            "SELECT * FROM inference_benchmark_samples WHERE run_id=? ORDER BY id", (run_id,))]
        code = connection.execute("SELECT code_version FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    for sample in samples:
        for key in ("fields", "metrics"):
            sample[key] = json.loads(sample.pop(key + "_json"))
    for key in ("variant", "manifest", "environment"):
        run[key] = json.loads(run.pop(key + "_json"))
    measured = [s for s in samples if s["phase"] == "measured"]
    # Parsing failures still have valid inference timing measurements.
    keys = ("preprocess_ms", "generation_ms", "first_token_ms", "decode_tokens_per_second", "inference_ms", "total_ms",
            "input_tokens", "output_tokens", "peak_allocated_bytes", "peak_reserved_bytes", "resident_allocated_bytes")
    complete_inferences = [s for s in measured if s["metrics"].get("generation_ms") is not None]
    aggregates = {key: _stats([s["metrics"][key] for s in complete_inferences if s["metrics"].get(key) is not None]) for key in keys}
    planned = len(run["manifest"]) * run["repetitions"]
    success = sum(s["status"] == "success" for s in measured)
    validity = {key: {"count": sum(s["metrics"].get(key) is True for s in measured), "denominator": len(measured)}
                for key in ("json_valid", "schema_valid", "output_limit_reached")}
    return {"run": run, "code_revision": code, "samples": samples, "aggregates": aggregates,
            "planned_samples": planned, "attempted_samples": len(measured), "successful_samples": success,
            "completed_inferences": len(complete_inferences),
            "completion_rate": success / planned if planned else None, "validity": validity,
            "timing_notes": "TTFT excludes preprocessing; decode rate excludes first token; output count includes EOS; p95 uses nearest rank; latency summaries exclude failures before generation completes."}


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
