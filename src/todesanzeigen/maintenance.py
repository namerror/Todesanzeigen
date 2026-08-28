from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .normalization import normalize_stored_fields_with_report
from .storage import connect


FIELD_JSON_TABLES = (
    "extraction_outputs",
    "label_candidates",
    "ground_truth_labels",
)


@dataclass(frozen=True)
class FieldNormalizationSummary:
    scanned_rows: int
    changed_rows: int
    table_changes: dict[str, int]
    locations_from_wohnort: int
    location_conflicts: int
    dates_normalized: int
    dates_unresolved: int
    applied: bool
    backup_path: Path | None = None


def normalize_database_fields(
    db_path: Path,
    *,
    apply: bool = False,
    backup_file: Path | None = None,
) -> FieldNormalizationSummary:
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    if backup_file is not None and not apply:
        raise ValueError("--backup-file requires --apply")

    updates: list[tuple[str, int, str]] = []
    table_changes = {table: 0 for table in FIELD_JSON_TABLES}
    scanned_rows = 0
    locations_from_wohnort = 0
    location_conflicts = 0
    dates_normalized = 0
    dates_unresolved = 0

    with connect(db_path) as connection:
        for table in FIELD_JSON_TABLES:
            for row in connection.execute(f"SELECT id, fields_json FROM {table} ORDER BY id"):
                scanned_rows += 1
                fields = _load_fields(row["fields_json"], table=table, row_id=int(row["id"]))
                report = normalize_stored_fields_with_report(fields)
                locations_from_wohnort += int(report.location_from_wohnort)
                location_conflicts += int(report.location_conflict)
                dates_normalized += len(report.normalized_dates)
                dates_unresolved += len(report.unresolved_dates)
                if fields == report.fields:
                    continue
                table_changes[table] += 1
                updates.append(
                    (
                        table,
                        int(row["id"]),
                        json.dumps(report.fields, ensure_ascii=False, sort_keys=True),
                    )
                )

    created_backup: Path | None = None
    if apply and updates:
        created_backup = backup_file or _default_backup_path(db_path)
        _backup_database(db_path, created_backup)
        with connect(db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table, row_id, fields_json in updates:
                connection.execute(
                    f"UPDATE {table} SET fields_json = ? WHERE id = ?",
                    (fields_json, row_id),
                )

    return FieldNormalizationSummary(
        scanned_rows=scanned_rows,
        changed_rows=len(updates),
        table_changes=table_changes,
        locations_from_wohnort=locations_from_wohnort,
        location_conflicts=location_conflicts,
        dates_normalized=dates_normalized,
        dates_unresolved=dates_unresolved,
        applied=apply,
        backup_path=created_backup,
    )


def _load_fields(value: str, *, table: str, row_id: int) -> dict[str, object]:
    try:
        fields = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid fields_json in {table} row {row_id}") from exc
    if not isinstance(fields, dict):
        raise ValueError(f"fields_json in {table} row {row_id} is not an object")
    return fields


def _default_backup_path(db_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return db_path.with_name(
        f"{db_path.stem}-before-field-normalization-{timestamp}{db_path.suffix}"
    )


def _backup_database(db_path: Path, backup_path: Path) -> None:
    if backup_path.resolve() == db_path.resolve():
        raise ValueError("Backup path must differ from the source database")
    if backup_path.exists():
        raise ValueError(f"Backup path already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source:
        with sqlite3.connect(backup_path) as destination:
            source.backup(destination)
            result = destination.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise RuntimeError(f"Backup integrity check failed: {backup_path}")
