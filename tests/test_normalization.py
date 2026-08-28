import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.todesanzeigen.maintenance import normalize_database_fields
from src.todesanzeigen.normalization import (
    fields_for_csv,
    normalize_date,
    normalize_stored_fields,
)
from src.todesanzeigen.storage import (
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    insert_extraction_output,
    insert_label_candidate,
    save_ground_truth_label,
    upsert_document,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("24.7.1941", "24.07.1941"),
        ("17. 1. 2026", "17.01.2026"),
        ("2,6.1931", "02.06.1931"),
        ("2024-02-29", "29.02.2024"),
        ("28. Januar 2023", "28.01.2023"),
        ("+1 November 1957", "01.11.1957"),
        ("17, Oktober 1932", "17.10.1932"),
        ("21. Mürz 1941", "21.03.1941"),
        ("4. Sept. 1937", "04.09.1937"),
        ("31.02.2024", None),
        ("1948", None),
        ("", ""),
    ],
)
def test_normalize_date_supports_german_and_ocr_variants(raw: str, expected: str | None) -> None:
    assert normalize_date(raw) == expected


def test_stored_fields_merge_location_and_annotate_unresolved_date_idempotently() -> None:
    fields = {
        "ort": "Aichach",
        "wohnort": "Affing",
        "geburtsdatum": "1948",
        "sterbedatum": "12. August 2025",
        "bemerkungen": "Bestehender Hinweis.",
    }

    normalized = normalize_stored_fields(fields)
    normalized_again = normalize_stored_fields(normalized)

    assert normalized["ort"] == "Aichach"
    assert "wohnort" not in normalized
    assert normalized["geburtsdatum"] == "1948"
    assert normalized["sterbedatum"] == "12.08.2025"
    assert normalized["bemerkungen"] == (
        "Bestehender Hinweis. Nicht kanonisiertes geburtsdatum beibehalten: 1948."
    )
    assert normalized_again == normalized


def test_stored_fields_fall_back_to_wohnort_and_csv_duplicates_ort() -> None:
    normalized = normalize_stored_fields({"ort": "", "wohnort": "Pöttmes"})
    csv_row = fields_for_csv(normalized)

    assert normalized["ort"] == "Pöttmes"
    assert "wohnort" not in normalized
    assert csv_row["ort"] == "Pöttmes"
    assert csv_row["wohnort"] == "Pöttmes"


def test_database_field_normalization_is_dry_run_then_backed_up_and_transactional() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "state" / "test.sqlite3"
        backup_path = root / "state" / "backup.sqlite3"
        apply_migrations(db_path)
        legacy_fields = {
            "ort": "",
            "wohnort": "Pöttmes",
            "geburtsdatum": "24. Juni 1952",
            "sterbedatum": "2025",
            "bemerkungen": "",
            "name": "Mustermann",
        }
        legacy_json = json.dumps(legacy_fields, ensure_ascii=False, sort_keys=True)

        with connect(db_path) as connection:
            document_id = upsert_document(
                connection,
                source_name="Aichacher Nachrichten",
                filename_stem="legacy",
            )
            output_id = insert_extraction_output(
                connection,
                document_id=document_id,
                run_id=None,
                method="text_extraction",
                fields={},
                status="processed",
            )
            candidate_id = insert_label_candidate(
                connection,
                document_id=document_id,
                extraction_output_id=output_id,
                source_kind="pipeline",
                fields={},
            )
            label_id = save_ground_truth_label(
                connection,
                document_id=document_id,
                label_set=DEFAULT_LABEL_SET,
                fields={},
                source_candidate_id=candidate_id,
            )
            connection.execute(
                "UPDATE extraction_outputs SET fields_json = ? WHERE id = ?",
                (legacy_json, output_id),
            )
            connection.execute(
                "UPDATE label_candidates SET fields_json = ? WHERE id = ?",
                (legacy_json, candidate_id),
            )
            connection.execute(
                "UPDATE ground_truth_labels SET fields_json = ? WHERE id = ?",
                (legacy_json, label_id),
            )

        dry_run = normalize_database_fields(db_path)
        with connect(db_path) as connection:
            unchanged = json.loads(
                connection.execute(
                    "SELECT fields_json FROM extraction_outputs WHERE id = ?",
                    (output_id,),
                ).fetchone()["fields_json"]
            )

        assert dry_run.scanned_rows == 3
        assert dry_run.changed_rows == 3
        assert dry_run.table_changes == {
            "extraction_outputs": 1,
            "label_candidates": 1,
            "ground_truth_labels": 1,
        }
        assert dry_run.locations_from_wohnort == 3
        assert dry_run.dates_normalized == 3
        assert dry_run.dates_unresolved == 3
        assert not dry_run.applied
        assert dry_run.backup_path is None
        assert unchanged["wohnort"] == "Pöttmes"

        applied = normalize_database_fields(
            db_path,
            apply=True,
            backup_file=backup_path,
        )

        assert applied.applied
        assert applied.changed_rows == 3
        assert applied.backup_path == backup_path
        assert backup_path.exists()
        with sqlite3.connect(backup_path) as backup:
            backed_up = json.loads(
                backup.execute(
                    "SELECT fields_json FROM extraction_outputs WHERE id = ?",
                    (output_id,),
                ).fetchone()[0]
            )
        assert backed_up["wohnort"] == "Pöttmes"

        with connect(db_path) as connection:
            for table in ("extraction_outputs", "label_candidates", "ground_truth_labels"):
                stored = json.loads(
                    connection.execute(f"SELECT fields_json FROM {table}").fetchone()["fields_json"]
                )
                assert "wohnort" not in stored
                assert stored["ort"] == "Pöttmes"
                assert stored["geburtsdatum"] == "24.06.1952"
                assert stored["sterbedatum"] == "2025"
                assert stored["bemerkungen"] == (
                    "Nicht kanonisiertes sterbedatum beibehalten: 2025."
                )

        second_dry_run = normalize_database_fields(db_path)
        assert second_dry_run.changed_rows == 0
        assert second_dry_run.dates_unresolved == 3
