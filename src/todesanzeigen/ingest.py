from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extract import PROCESSED_STATUSES
from .methods import requires_model
from .normalization import normalize_stored_fields
from .ocr import image_mime_type
from .storage import (
    DEFAULT_DB_PATH,
    apply_migrations,
    connect,
    create_run,
    finish_run,
    insert_extraction_output,
    insert_label_candidate,
    read_csv_rows,
    sha256_file,
    upsert_document,
    upsert_ocr_output,
)


@dataclass(frozen=True)
class SourceIngestSummary:
    documents: int
    text_artifacts: int
    tsv_artifacts: int
    ocr_outputs: int
    run_id: str


@dataclass(frozen=True)
class ResultsIngestSummary:
    extraction_outputs: int
    label_candidates: int
    run_id: str


def ingest_source(
    *,
    source: str,
    input_dir: Path,
    artifacts_dir: Path,
    db_path: Path = DEFAULT_DB_PATH,
    layout_family: str = "",
    limit: int | None = None,
) -> SourceIngestSummary:
    apply_migrations(db_path)
    images = _discover_images_recursive(input_dir)
    if limit is not None:
        images = images[:limit]
    text_artifacts = _artifact_index(artifacts_dir, ".txt")
    tsv_artifacts = _artifact_index(artifacts_dir, ".tsv")
    name_map = _load_name_map(artifacts_dir)

    with connect(db_path) as connection:
        run_id = create_run(
            connection,
            command="ingest source",
            method="source_inventory",
            config={
                "source": source,
                "input_dir": str(input_dir),
                "artifacts_dir": str(artifacts_dir),
                "layout_family": layout_family,
            },
        )
        document_count = 0
        text_count = 0
        tsv_count = 0
        ocr_count = 0

        for image_path in images:
            document_id = upsert_document(
                connection,
                source_name=source,
                filename_stem=image_path.stem,
                image_path=image_path,
                image_sha256=sha256_file(image_path),
                mime_type=image_mime_type(image_path) or "",
                layout_family=layout_family,
            )
            document_count += 1
            text_path = text_artifacts.get(image_path.stem)
            tsv_path = tsv_artifacts.get(image_path.stem)
            text = ""
            if text_path is not None:
                text = text_path.read_text(encoding="utf-8")
                text_count += 1
            if tsv_path is not None:
                tsv_count += 1
            if text_path is not None or tsv_path is not None:
                hint = name_map.get(f"{image_path.stem}.txt", {})
                upsert_ocr_output(
                    connection,
                    document_id=document_id,
                    run_id=run_id,
                    text=text,
                    text_path=text_path or "",
                    text_sha256=sha256_file(text_path) if text_path is not None else "",
                    tsv_path=tsv_path or "",
                    tsv_sha256=sha256_file(tsv_path) if tsv_path is not None else "",
                    features=_ocr_features(text, tsv_path),
                    name_hint=str(hint.get("name", "") or ""),
                    name_confidence=_optional_float(hint.get("confidence")),
                )
                ocr_count += 1

        finish_run(connection, run_id)

    return SourceIngestSummary(document_count, text_count, tsv_count, ocr_count, run_id)


def ingest_results(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    source: str = "",
    output_csv: Path | None = None,
    results_file: Path | None = None,
    method: str = "csv_import",
    provider: str = "",
    model: str = "",
    candidate_kind: str = "teacher",
    input_dir: Path | None = None,
) -> ResultsIngestSummary:
    if output_csv is None and results_file is None:
        raise ValueError("at least one of output_csv or results_file is required")
    resolved_provider = provider.strip()
    resolved_model = model.strip()
    if requires_model(method) and not resolved_model:
        if output_csv is not None or results_file is None:
            raise ValueError(
                f"--model is required when importing model-backed extraction method: {method}"
            )
        inferred_provider, inferred_model = _checkpoint_lineage(results_file, method)
        resolved_provider = resolved_provider or inferred_provider
        resolved_model = inferred_model
    image_index = _source_image_index(input_dir) if input_dir is not None else None
    apply_migrations(db_path)
    with connect(db_path) as connection:
        run_id = create_run(
            connection,
            command="ingest results",
            method=method,
            provider=resolved_provider,
            model=resolved_model,
            config={
                "source": source,
                "output_csv": str(output_csv or ""),
                "output_csv_sha256": (
                    sha256_file(output_csv)
                    if output_csv is not None and output_csv.exists()
                    else ""
                ),
                "results_file": str(results_file or ""),
                "results_file_sha256": (
                    sha256_file(results_file)
                    if results_file is not None and results_file.exists()
                    else ""
                ),
                "candidate_kind": candidate_kind,
                "input_dir": str(input_dir or ""),
            },
        )
        output_count = 0
        candidate_count = 0
        if output_csv is not None:
            for row in read_csv_rows(output_csv):
                document_id, _ = _document_for_row(
                    connection,
                    row,
                    source,
                    image_index,
                )
                output_id, created = _get_or_create_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=run_id,
                    method=method,
                    provider=resolved_provider,
                    model=resolved_model,
                    fields=row,
                    status="processed",
                )
                output_count += 1 if created else 0
                candidate_id = _get_or_create_label_candidate(
                    connection,
                    document_id=document_id,
                    extraction_output_id=output_id,
                    source_kind=candidate_kind,
                    source_name=method,
                    fields=row,
                )
                if candidate_id is not None:
                    candidate_count += 1
        if results_file is not None:
            for record in _read_jsonl(results_file):
                row = record.get("row")
                if record.get("status") not in PROCESSED_STATUSES or not isinstance(row, dict):
                    continue
                document_id, _ = _document_for_row(
                    connection,
                    row,
                    source,
                    image_index,
                )
                record_method = str(record.get("method") or method)
                output_id, created = _get_or_create_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=run_id,
                    method=record_method,
                    provider=str(record.get("provider") or resolved_provider),
                    model=str(record.get("model") or resolved_model),
                    fields=row,
                    status=str(record.get("status") or "processed"),
                    error=str(record.get("error") or ""),
                    attempts=int(record.get("attempts") or 0),
                )
                output_count += 1 if created else 0
                candidate_id = _get_or_create_label_candidate(
                    connection,
                    document_id=document_id,
                    extraction_output_id=output_id,
                    source_kind=candidate_kind,
                    source_name=record_method,
                    fields=row,
                )
                if candidate_id is not None:
                    candidate_count += 1
        finish_run(connection, run_id)
    return ResultsIngestSummary(output_count, candidate_count, run_id)


def _checkpoint_lineage(results_file: Path, default_method: str) -> tuple[str, str]:
    providers: set[str] = set()
    models: set[str] = set()
    for record in _read_jsonl(results_file):
        row = record.get("row")
        if record.get("status") not in PROCESSED_STATUSES or not isinstance(row, dict):
            continue
        record_method = str(record.get("method") or default_method)
        if not requires_model(record_method):
            continue
        record_model = str(record.get("model") or "").strip()
        if not record_model:
            raise ValueError(
                f"--model is required because {results_file} contains a model-backed "
                "record without model metadata"
            )
        models.add(record_model)
        record_provider = str(record.get("provider") or "").strip()
        if record_provider:
            providers.add(record_provider)

    if len(models) != 1:
        raise ValueError(
            f"--model is required because {results_file} does not contain exactly one model"
        )
    provider = next(iter(providers)) if len(providers) == 1 else ""
    return provider, next(iter(models))


def _document_for_row(
    connection: Any,
    row: dict[str, Any],
    source: str,
    image_index: dict[str, list[Path]] | None = None,
) -> tuple[int, Path | None]:
    source_name = source or str(row.get("quelle", "") or "unknown")
    filename_stem = str(row.get("dateiname", "") or "").strip()
    if not filename_stem:
        raise ValueError("result row is missing dateiname")
    image_path = _resolve_source_image(filename_stem, image_index)
    image_digest = (
        sha256_file(image_path)
        if image_path is not None and image_path.exists() and image_path.is_file()
        else ""
    )
    document_id = upsert_document(
        connection,
        source_name=source_name,
        filename_stem=filename_stem,
        image_path=image_path or "",
        image_sha256=image_digest,
        mime_type=image_mime_type(image_path) if image_path is not None else "",
    )
    return document_id, image_path


def _source_image_index(input_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for image_path in _discover_images_recursive(input_dir):
        index.setdefault(_normalized_filename_stem(image_path.stem), []).append(image_path)
    return index


def _resolve_source_image(
    filename_stem: str,
    image_index: dict[str, list[Path]] | None,
) -> Path | None:
    if image_index is None:
        return None
    matches = image_index.get(_normalized_filename_stem(filename_stem), [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No source image found for result row dateiname: {filename_stem}")
    paths = ", ".join(str(path) for path in matches)
    raise ValueError(f"Ambiguous source images for result row dateiname {filename_stem}: {paths}")


def _normalized_filename_stem(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip())
    normalized = re.sub(r"\s*_\s*", "_", normalized)
    return normalized.casefold()


def _get_or_create_extraction_output(
    connection: Any,
    *,
    document_id: int,
    run_id: str,
    method: str,
    provider: str,
    model: str,
    fields: dict[str, Any],
    status: str,
    error: str = "",
    attempts: int = 0,
) -> tuple[int, bool]:
    fields_json = _fields_json(fields)
    existing = connection.execute(
        """
        SELECT id FROM extraction_outputs
        WHERE document_id = ?
          AND method = ?
          AND provider = ?
          AND model = ?
          AND fields_json = ?
          AND status = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (document_id, method, provider, model, fields_json, status),
    ).fetchone()
    if existing is not None:
        return int(existing["id"]), False
    return (
        insert_extraction_output(
            connection,
            document_id=document_id,
            run_id=run_id,
            method=method,
            provider=provider,
            model=model,
            fields=fields,
            status=status,
            error=error,
            attempts=attempts,
        ),
        True,
    )


def _get_or_create_label_candidate(
    connection: Any,
    *,
    document_id: int,
    extraction_output_id: int,
    source_kind: str,
    source_name: str,
    fields: dict[str, Any],
) -> int | None:
    fields_json = _fields_json(fields)
    existing = connection.execute(
        """
        SELECT id FROM label_candidates
        WHERE document_id = ?
          AND source_kind = ?
          AND source_name = ?
          AND fields_json = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (document_id, source_kind, source_name, fields_json),
    ).fetchone()
    if existing is not None:
        return None
    return insert_label_candidate(
        connection,
        document_id=document_id,
        extraction_output_id=extraction_output_id,
        source_kind=source_kind,
        source_name=source_name,
        fields=fields,
    )


def _fields_json(fields: dict[str, Any]) -> str:
    normalized = normalize_stored_fields(fields)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _discover_images_recursive(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and image_mime_type(path))


def _artifact_index(artifacts_dir: Path, suffix: str) -> dict[str, Path]:
    if not artifacts_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(artifacts_dir.rglob(f"*{suffix}"))
        if path.is_file() and path.suffix == suffix
    }


def _load_name_map(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(artifacts_dir.rglob("name_map.json")) if artifacts_dir.exists() else []
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, dict):
                    merged[key] = value
    return merged


def _ocr_features(text: str, tsv_path: Path | None) -> dict[str, Any]:
    features: dict[str, Any] = {
        "ocr_char_count": len(text),
        "ocr_word_count": len(text.split()),
    }
    if tsv_path is not None and tsv_path.exists():
        tsv_lines = tsv_path.read_text(encoding="utf-8").splitlines()
        features["tsv_line_count"] = len(tsv_lines)
    return features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL file contains invalid JSON on line {line_number}: {path}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
