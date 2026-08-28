from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .llm import CSV_COLUMNS
from .methods import (
    DEFAULT_EXPORT_METHOD_PRIORITY,
    GT_EXPORT_SOURCE,
    default_route_reason,
    method_family,
    requires_model,
    result_slot,
)
from .normalization import fields_for_csv as project_fields_for_csv
from .normalization import normalize_stored_fields
from .ocr import image_mime_type


DEFAULT_DB_PATH = Path("state/todesanzeigen.sqlite3")
DEFAULT_LABEL_SET = "gt-v1"
SCHEMA_VERSION = "005_remove_artifacts"
SUCCESSFUL_EXTRACTION_STATUSES = ("processed", "rerouted_processed", "vision_processed")


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    source: str
    filename_stem: str
    image_path: str
    mime_type: str
    year: int | None


@dataclass(frozen=True)
class CsvExportSummary:
    rows: int
    ground_truth_rows: int
    method_rows: dict[str, int]
    missing_documents: int


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def apply_migrations(db_path: Path = DEFAULT_DB_PATH, migrations_dir: Path | None = None) -> list[str]:
    migrations_root = migrations_dir or Path(__file__).resolve().parents[2] / "migrations"
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        applied = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
        }
        applied_now: list[str] = []
        for path in sorted(migrations_root.glob("*.sql")):
            version = path.stem
            if version in applied:
                continue
            connection.executescript(path.read_text(encoding="utf-8"))
            connection.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
            applied_now.append(version)
        return applied_now


def create_run(
    connection: sqlite3.Connection,
    *,
    command: str,
    method: str = "",
    provider: str = "",
    model: str = "",
    config: dict[str, Any] | None = None,
    code_version: str = "",
) -> str:
    _validate_required_model(method, model)
    run_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO runs(id, command, method, provider, model, config_json, code_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            command,
            method,
            provider,
            model,
            _json(config or {}),
            code_version,
        ),
    )
    return run_id


def finish_run(connection: sqlite3.Connection, run_id: str, *, status: str = "completed") -> None:
    connection.execute(
        "UPDATE runs SET status = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, run_id),
    )


def get_or_create_source(
    connection: sqlite3.Connection,
    name: str,
    *,
    source_type: str = "",
    notes: str = "",
) -> int:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("source name is required")
    connection.execute(
        """
        INSERT INTO sources(name, source_type, notes)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source_type = COALESCE(NULLIF(excluded.source_type, ''), sources.source_type),
            notes = COALESCE(NULLIF(excluded.notes, ''), sources.notes)
        """,
        (clean_name, source_type, notes),
    )
    return int(
        connection.execute("SELECT id FROM sources WHERE name = ?", (clean_name,)).fetchone()["id"]
    )


def upsert_document(
    connection: sqlite3.Connection,
    *,
    source_name: str,
    filename_stem: str,
    image_path: Path | str = "",
    image_sha256: str = "",
    mime_type: str = "",
    year: int | None = None,
    layout_family: str = "",
) -> int:
    source_id = get_or_create_source(connection, source_name)
    image_path_text = _path_text(image_path)
    document_key = f"{source_name}:{image_path_text or filename_stem}"
    inferred_year = year if year is not None else infer_year(filename_stem)
    if image_path_text:
        existing = connection.execute(
            "SELECT id FROM documents WHERE document_key = ?",
            (document_key,),
        ).fetchone()
        if existing is None:
            legacy_rows = connection.execute(
                """
                SELECT id FROM documents
                WHERE source_id = ?
                  AND filename_stem = ?
                  AND image_path = ''
                ORDER BY id
                """,
                (source_id, filename_stem),
            ).fetchall()
            if len(legacy_rows) == 1:
                legacy_id = int(legacy_rows[0]["id"])
                connection.execute(
                    """
                    UPDATE documents
                    SET document_key = ?,
                        filename_stem = ?,
                        image_path = ?,
                        image_sha256 = COALESCE(NULLIF(?, ''), image_sha256),
                        mime_type = COALESCE(NULLIF(?, ''), mime_type),
                        year = COALESCE(?, year),
                        layout_family = COALESCE(NULLIF(?, ''), layout_family),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        document_key,
                        filename_stem,
                        image_path_text,
                        image_sha256,
                        mime_type,
                        inferred_year,
                        layout_family,
                        legacy_id,
                    ),
                )
                return legacy_id
    connection.execute(
        """
        INSERT INTO documents(
            document_key, source_id, filename_stem, image_path, image_sha256,
            mime_type, year, layout_family
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_key) DO UPDATE SET
            filename_stem = excluded.filename_stem,
            image_path = COALESCE(NULLIF(excluded.image_path, ''), documents.image_path),
            image_sha256 = COALESCE(NULLIF(excluded.image_sha256, ''), documents.image_sha256),
            mime_type = COALESCE(NULLIF(excluded.mime_type, ''), documents.mime_type),
            year = COALESCE(excluded.year, documents.year),
            layout_family = COALESCE(NULLIF(excluded.layout_family, ''), documents.layout_family),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            document_key,
            source_id,
            filename_stem,
            image_path_text,
            image_sha256,
            mime_type,
            inferred_year,
            layout_family,
        ),
    )
    return int(
        connection.execute(
            "SELECT id FROM documents WHERE document_key = ?",
            (document_key,),
        ).fetchone()["id"]
    )


def find_document_id(
    connection: sqlite3.Connection,
    *,
    source_name: str,
    filename_stem: str,
) -> int | None:
    row = connection.execute(
        """
        SELECT documents.id
        FROM documents
        JOIN sources ON sources.id = documents.source_id
        WHERE sources.name = ? AND documents.filename_stem = ?
        ORDER BY documents.id DESC
        LIMIT 1
        """,
        (source_name, filename_stem),
    ).fetchone()
    return int(row["id"]) if row else None


def upsert_ocr_output(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    run_id: str,
    text: str,
    text_path: Path | str = "",
    text_sha256: str = "",
    tsv_path: Path | str = "",
    tsv_sha256: str = "",
    settings: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    name_hint: str = "",
    name_confidence: float | None = None,
) -> int:
    connection.execute(
        """
        INSERT INTO ocr_outputs(
            document_id, run_id, text, text_path, text_sha256, tsv_path, tsv_sha256,
            settings_json, features_json, name_hint, name_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, run_id) DO UPDATE SET
            text = excluded.text,
            text_path = excluded.text_path,
            text_sha256 = excluded.text_sha256,
            tsv_path = excluded.tsv_path,
            tsv_sha256 = excluded.tsv_sha256,
            settings_json = excluded.settings_json,
            features_json = excluded.features_json,
            name_hint = excluded.name_hint,
            name_confidence = excluded.name_confidence
        """,
        (
            document_id,
            run_id,
            text,
            _path_text(text_path),
            text_sha256,
            _path_text(tsv_path),
            tsv_sha256,
            _json(settings or {}),
            _json(features or {}),
            name_hint,
            name_confidence,
        ),
    )
    return int(
        connection.execute(
            "SELECT id FROM ocr_outputs WHERE document_id = ? AND run_id = ?",
            (document_id, run_id),
        ).fetchone()["id"]
    )


def insert_extraction_output(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    run_id: str | None,
    method: str,
    fields: dict[str, Any] | None,
    status: str,
    provider: str = "",
    model: str = "",
    prompt_version: str = "death_notice_v2",
    raw_response: str = "",
    error: str = "",
    attempts: int = 0,
    estimated_tokens: int | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    method_family_value: str = "",
    route_reason: str = "",
    route_decision: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    ocr_output_id: int | None = None,
    result_slot_value: str = "",
    input_fingerprint: str = "",
    config_hash: str = "",
    superseded_at: str | None = None,
) -> int:
    family = method_family_value or method_family(method)
    _validate_required_model(method, model, family)
    reason = route_reason or default_route_reason(method)
    slot = result_slot_value or result_slot(method)
    cursor = connection.execute(
        """
        INSERT INTO extraction_outputs(
            document_id, run_id, method, provider, model, prompt_version,
            fields_json, raw_response, status, error, attempts, latency_ms,
            estimated_tokens, cost_usd, method_family,
            route_reason, route_decision_json, config_json, ocr_output_id,
            result_slot, input_fingerprint, config_hash, superseded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            run_id,
            method,
            provider,
            model,
            prompt_version,
            _json(_normalize_fields(fields or {})),
            raw_response,
            status,
            error,
            attempts,
            latency_ms,
            estimated_tokens,
            cost_usd,
            family,
            reason,
            _json(route_decision or {}),
            _json(config or {}),
            ocr_output_id,
            slot,
            input_fingerprint,
            config_hash,
            superseded_at,
        ),
    )
    return int(cursor.lastrowid)


def _validate_required_model(
    method: str,
    model: str,
    method_family_value: str = "",
) -> None:
    if requires_model(method, method_family_value) and not model.strip():
        raise ValueError(f"model is required for model-backed extraction method: {method}")


def insert_label_candidate(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    fields: dict[str, Any],
    source_kind: str,
    source_name: str = "",
    extraction_output_id: int | None = None,
    status: str = "pending",
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO label_candidates(
            document_id, extraction_output_id, source_kind, source_name,
            fields_json, confidence_score, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            extraction_output_id,
            source_kind,
            source_name,
            _json(_normalize_fields(fields)),
            _optional_float(fields.get("confidence_score")),
            status,
        ),
    )
    return int(cursor.lastrowid)


def save_ground_truth_label(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    label_set: str,
    fields: dict[str, Any],
    source_candidate_id: int | None = None,
    reviewer: str = "",
    review_notes: str = "",
) -> int:
    connection.execute(
        """
        INSERT INTO ground_truth_labels(
            document_id, label_set, fields_json, source_candidate_id, reviewer, review_notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, label_set) DO UPDATE SET
            fields_json = excluded.fields_json,
            source_candidate_id = excluded.source_candidate_id,
            reviewer = excluded.reviewer,
            review_notes = excluded.review_notes,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            document_id,
            label_set,
            _json(_normalize_fields(fields)),
            source_candidate_id,
            reviewer,
            review_notes,
        ),
    )
    if source_candidate_id is not None:
        connection.execute(
            "UPDATE label_candidates SET status = 'approved' WHERE id = ?",
            (source_candidate_id,),
        )
    return int(
        connection.execute(
            """
            SELECT id FROM ground_truth_labels
            WHERE document_id = ? AND label_set = ?
            """,
            (document_id, label_set),
        ).fetchone()["id"]
    )


def mark_candidate_status(
    connection: sqlite3.Connection,
    *,
    candidate_id: int,
    status: str,
) -> None:
    connection.execute(
        "UPDATE label_candidates SET status = ? WHERE id = ?",
        (status, candidate_id),
    )


def load_candidate(connection: sqlite3.Connection, candidate_id: int) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM label_candidates WHERE id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Label candidate does not exist: {candidate_id}")
    return row


def pending_review_items(
    connection: sqlite3.Connection,
    *,
    label_set: str = DEFAULT_LABEL_SET,
    source_name: str = "",
    limit: int = 50,
) -> list[sqlite3.Row]:
    method_filter = source_name.strip()
    params: list[Any] = [label_set]
    method_clause = ""
    if method_filter:
        method_clause = "AND label_candidates.source_name = ?"
        params.append(method_filter)
    params.append(limit)
    return list(
        connection.execute(
            f"""
            SELECT
                label_candidates.id AS candidate_id,
                documents.id AS document_id,
                sources.name AS source,
                documents.filename_stem,
                documents.image_path,
                label_candidates.source_kind,
                label_candidates.source_name,
                extraction_outputs.method AS extraction_method,
                extraction_outputs.provider AS extraction_provider,
                extraction_outputs.model AS extraction_model,
                extraction_outputs.method_family,
                extraction_outputs.route_reason,
                label_candidates.created_at
            FROM label_candidates
            JOIN documents ON documents.id = label_candidates.document_id
            JOIN sources ON sources.id = documents.source_id
            LEFT JOIN extraction_outputs
                ON extraction_outputs.id = label_candidates.extraction_output_id
            LEFT JOIN ground_truth_labels
                ON ground_truth_labels.document_id = documents.id
                AND ground_truth_labels.label_set = ?
            WHERE label_candidates.status = 'pending'
              AND ground_truth_labels.id IS NULL
              {method_clause}
            ORDER BY label_candidates.created_at, label_candidates.id
            LIMIT ?
            """,
            params,
        )
    )


def review_method_options(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT method, method_family, description
            FROM extraction_methods
            ORDER BY method
            """
        )
    )


def reviewed_ground_truth_items(
    connection: sqlite3.Connection,
    *,
    label_set: str = DEFAULT_LABEL_SET,
    limit: int = 50,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT
                ground_truth_labels.id AS label_id,
                ground_truth_labels.document_id,
                ground_truth_labels.source_candidate_id,
                ground_truth_labels.reviewer,
                ground_truth_labels.review_notes,
                ground_truth_labels.created_at,
                ground_truth_labels.updated_at,
                sources.name AS source,
                documents.filename_stem,
                documents.image_path,
                label_candidates.source_kind,
                label_candidates.source_name
            FROM ground_truth_labels
            JOIN documents ON documents.id = ground_truth_labels.document_id
            JOIN sources ON sources.id = documents.source_id
            LEFT JOIN label_candidates
                ON label_candidates.id = ground_truth_labels.source_candidate_id
            WHERE ground_truth_labels.label_set = ?
            ORDER BY ground_truth_labels.updated_at DESC, ground_truth_labels.id DESC
            LIMIT ?
            """,
            (label_set, limit),
        )
    )


def document_review_detail(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    label_set: str = DEFAULT_LABEL_SET,
) -> dict[str, Any]:
    document = connection.execute(
        """
        SELECT documents.*, sources.name AS source
        FROM documents
        JOIN sources ON sources.id = documents.source_id
        WHERE documents.id = ?
        """,
        (document_id,),
    ).fetchone()
    if document is None:
        raise ValueError(f"Document does not exist: {document_id}")
    candidates = [
        {
            **dict(row),
            "fields": _loads(row["fields_json"]),
        }
        for row in connection.execute(
            """
            SELECT
                label_candidates.*,
                extraction_outputs.method AS extraction_method,
                extraction_outputs.provider AS extraction_provider,
                extraction_outputs.model AS extraction_model,
                extraction_outputs.method_family,
                extraction_outputs.route_reason
            FROM label_candidates
            LEFT JOIN extraction_outputs
                ON extraction_outputs.id = label_candidates.extraction_output_id
            WHERE label_candidates.document_id = ?
            ORDER BY label_candidates.created_at DESC, label_candidates.id DESC
            """,
            (document_id,),
        )
    ]
    ground_truth = connection.execute(
        """
        SELECT * FROM ground_truth_labels
        WHERE document_id = ? AND label_set = ?
        """,
        (document_id, label_set),
    ).fetchone()
    ocr = connection.execute(
        """
        SELECT * FROM ocr_outputs
        WHERE document_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()
    return {
        "document": dict(document),
        "candidates": candidates,
        "ground_truth": (
            {**dict(ground_truth), "fields": _loads(ground_truth["fields_json"])}
            if ground_truth
            else None
        ),
        "ocr": dict(ocr) if ocr else None,
    }


def latest_extraction_by_method(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    method: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM extraction_outputs
        WHERE document_id = ? AND method = ? AND status IN (
            'processed', 'rerouted_processed', 'vision_processed'
        )
          AND superseded_at IS NULL
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (document_id, method),
    ).fetchone()


def latest_active_extraction_by_slot(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    result_slot_value: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM extraction_outputs
        WHERE document_id = ?
          AND result_slot = ?
          AND superseded_at IS NULL
          AND status IN ('processed', 'rerouted_processed', 'vision_processed')
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (document_id, result_slot_value),
    ).fetchone()


def supersede_active_extractions_by_slot(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    result_slot_value: str,
) -> int:
    cursor = connection.execute(
        """
        UPDATE extraction_outputs
        SET superseded_at = CURRENT_TIMESTAMP
        WHERE document_id = ?
          AND result_slot = ?
          AND superseded_at IS NULL
          AND status IN ('processed', 'rerouted_processed', 'vision_processed')
        """,
        (document_id, result_slot_value),
    )
    return int(cursor.rowcount)


def latest_ocr_output(
    connection: sqlite3.Connection,
    *,
    document_id: int,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM ocr_outputs
        WHERE document_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()


def upsert_feature_snapshot(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    feature_set: str,
    features: dict[str, Any],
    ocr_output_id: int | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    connection.execute(
        """
        INSERT INTO feature_snapshots(
            document_id, ocr_output_id, feature_set, features_json, config_json
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(document_id, feature_set) DO UPDATE SET
            ocr_output_id = excluded.ocr_output_id,
            features_json = excluded.features_json,
            config_json = excluded.config_json,
            created_at = CURRENT_TIMESTAMP
        """,
        (document_id, ocr_output_id, feature_set, _json(features), _json(config or {})),
    )
    return int(
        connection.execute(
            """
            SELECT id FROM feature_snapshots
            WHERE document_id = ? AND feature_set = ?
            """,
            (document_id, feature_set),
        ).fetchone()["id"]
    )


def latest_feature_snapshot(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    feature_set: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM feature_snapshots
        WHERE document_id = ? AND feature_set = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (document_id, feature_set),
    ).fetchone()


def ground_truth_rows(
    connection: sqlite3.Connection,
    *,
    label_set: str,
    split_name: str = "",
) -> list[sqlite3.Row]:
    if not split_name:
        return list(
            connection.execute(
                """
                SELECT ground_truth_labels.*, documents.filename_stem
                FROM ground_truth_labels
                JOIN documents ON documents.id = ground_truth_labels.document_id
                WHERE label_set = ?
                ORDER BY documents.id
                """,
                (label_set,),
            )
        )
    return list(
        connection.execute(
            """
            SELECT ground_truth_labels.*, documents.filename_stem
            FROM ground_truth_labels
            JOIN documents ON documents.id = ground_truth_labels.document_id
            JOIN dataset_memberships ON dataset_memberships.document_id = documents.id
            JOIN dataset_splits ON dataset_splits.id = dataset_memberships.split_id
            WHERE ground_truth_labels.label_set = ?
              AND dataset_splits.name = ?
              AND dataset_memberships.subset != 'train'
            ORDER BY documents.id
            """,
            (label_set, split_name),
        )
    )


def create_dataset_split(
    connection: sqlite3.Connection,
    *,
    name: str,
    strategy: str,
    assignments: dict[int, str],
    config: dict[str, Any] | None = None,
) -> int:
    connection.execute(
        """
        INSERT INTO dataset_splits(name, strategy, config_json)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            strategy = excluded.strategy,
            config_json = excluded.config_json
        """,
        (name, strategy, _json(config or {})),
    )
    split_id = int(
        connection.execute("SELECT id FROM dataset_splits WHERE name = ?", (name,)).fetchone()["id"]
    )
    connection.execute("DELETE FROM dataset_memberships WHERE split_id = ?", (split_id,))
    connection.executemany(
        """
        INSERT INTO dataset_memberships(split_id, document_id, subset)
        VALUES (?, ?, ?)
        """,
        [(split_id, document_id, subset) for document_id, subset in assignments.items()],
    )
    return split_id


def all_documents(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT documents.*, sources.name AS source
            FROM documents
            JOIN sources ON sources.id = documents.source_id
            ORDER BY sources.name, documents.filename_stem, documents.id
            """
        )
    )


def insert_evaluation_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    label_set: str,
    method: str,
    split_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO evaluation_runs(
            name, label_set, method, split_name, config_json, metrics_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, label_set, method, split_name, _json(config), _json(metrics)),
    )
    return int(cursor.lastrowid)


def insert_evaluation_result(
    connection: sqlite3.Connection,
    *,
    evaluation_run_id: int,
    document_id: int,
    extraction_output_id: int | None,
    exact_match: bool,
    field_results: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO evaluation_results(
            evaluation_run_id, document_id, extraction_output_id, exact_match, field_results_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            evaluation_run_id,
            document_id,
            extraction_output_id,
            1 if exact_match else 0,
            _json(field_results),
        ),
    )


def export_priority_csv(
    connection: sqlite3.Connection,
    *,
    output_csv: Path,
    label_set: str = DEFAULT_LABEL_SET,
    method_priority: tuple[str, ...] = DEFAULT_EXPORT_METHOD_PRIORITY,
) -> CsvExportSummary:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    method_rows = {method: 0 for method in method_priority}
    ground_truth_count = 0
    missing_count = 0
    rows: list[dict[str, str]] = []
    for document in all_documents(connection):
        source_kind, fields = _selected_export_fields(
            connection,
            document_id=int(document["id"]),
            label_set=label_set,
            method_priority=method_priority,
        )
        if fields is None:
            missing_count += 1
            continue
        row = _fields_for_csv(
            fields,
            source=str(document["source"]),
            filename_stem=str(document["filename_stem"]),
        )
        rows.append(row)
        if source_kind == GT_EXPORT_SOURCE:
            ground_truth_count += 1
        else:
            method_rows[source_kind] = method_rows.get(source_kind, 0) + 1

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return CsvExportSummary(
        rows=len(rows),
        ground_truth_rows=ground_truth_count,
        method_rows=method_rows,
        missing_documents=missing_count,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [normalize_stored_fields(row) for row in csv.DictReader(handle)]


def infer_year(value: str) -> int | None:
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", value)
    return int(match.group(1)) if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields_from_form(items: Iterable[tuple[str, Any]]) -> dict[str, str]:
    values = {key: str(value) for key, value in items}
    return normalize_stored_fields(values)


def _selected_export_fields(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    label_set: str,
    method_priority: tuple[str, ...],
) -> tuple[str, dict[str, Any] | None]:
    ground_truth = connection.execute(
        """
        SELECT fields_json FROM ground_truth_labels
        WHERE document_id = ? AND label_set = ?
        """,
        (document_id, label_set),
    ).fetchone()
    if ground_truth is not None:
        return GT_EXPORT_SOURCE, _loads(ground_truth["fields_json"])

    for method in method_priority:
        prediction = latest_extraction_by_method(
            connection,
            document_id=document_id,
            method=method,
        )
        if prediction is not None:
            return method, _loads(prediction["fields_json"])

    return "", None


def _fields_for_csv(
    fields: dict[str, Any],
    *,
    source: str,
    filename_stem: str,
) -> dict[str, str]:
    return project_fields_for_csv(
        fields,
        source=source,
        filename_stem=filename_stem,
    )


def _normalize_fields(fields: dict[str, Any]) -> dict[str, str]:
    return normalize_stored_fields(fields)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_text(path: Path | str) -> str:
    if not path:
        return ""
    path_obj = path if isinstance(path, Path) else Path(path)
    try:
        return str(path_obj.relative_to(Path.cwd()))
    except ValueError:
        return str(path_obj)


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str) -> dict[str, Any]:
    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}
