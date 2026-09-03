import json
import shutil
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.todesanzeigen.ingest import ingest_results
from src.todesanzeigen.review import _select_review_candidates
from src.todesanzeigen.storage import (
    apply_migrations,
    connect,
    insert_extraction_output,
    upsert_document,
)
from src.todesanzeigen.variants import ExtractionVariant, load_variant_config


def test_default_variant_config_has_operational_and_review_defaults() -> None:
    config = load_variant_config()

    assert config.text_default == "text_current"
    assert config.vlm_default == "vlm_current"
    assert [variant.alias for variant in config.review_variants] == [
        "text_current",
        "vlm_current",
    ]


def test_variant_migration_reactivates_latest_success_per_exact_variant() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "test.sqlite3"
        migrations = Path(__file__).resolve().parents[1] / "migrations"
        staged = root / "migrations"
        staged.mkdir()
        for path in sorted(migrations.glob("00[1-5]_*.sql")):
            shutil.copy2(path, staged / path.name)
        apply_migrations(db_path, staged)
        with connect(db_path) as connection:
            document_id = upsert_document(
                connection, source_name="source", filename_stem="example"
            )
            first_v1 = insert_extraction_output(
                connection,
                document_id=document_id,
                run_id=None,
                method="text_extraction",
                provider="qwen",
                model="qwen3.6-flash",
                prompt_version="death_notice_v1",
                fields={"name": "old-v1"},
                status="processed",
            )
            connection.execute(
                "UPDATE extraction_outputs SET superseded_at = CURRENT_TIMESTAMP WHERE id = ?",
                (first_v1,),
            )
            v3 = insert_extraction_output(
                connection,
                document_id=document_id,
                run_id=None,
                method="text_extraction",
                provider="qwen",
                model="qwen3.6-flash",
                prompt_version="death_notice_v3",
                fields={"name": "v3"},
                status="processed",
            )
            connection.execute(
                "UPDATE extraction_outputs SET superseded_at = CURRENT_TIMESTAMP WHERE id = ?",
                (v3,),
            )
            latest_v1 = insert_extraction_output(
                connection,
                document_id=document_id,
                run_id=None,
                method="text_extraction",
                provider="qwen",
                model="qwen3.6-flash",
                prompt_version="death_notice_v1",
                fields={"name": "latest-v1"},
                status="processed",
            )

        shutil.copy2(migrations / "006_extraction_variants.sql", staged / "006_extraction_variants.sql")
        assert apply_migrations(db_path, staged) == ["006_extraction_variants"]
        with connect(db_path) as connection:
            active = connection.execute(
                "SELECT id, prompt_version FROM extraction_outputs "
                "WHERE superseded_at IS NULL ORDER BY id"
            ).fetchall()
            assert [(row["id"], row["prompt_version"]) for row in active] == [
                (v3, "death_notice_v3"),
                (latest_v1, "death_notice_v1"),
            ]
            with pytest.raises(sqlite3.IntegrityError):
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="qwen",
                    model="qwen3.6-flash",
                    prompt_version="death_notice_v1",
                    fields={"name": "duplicate"},
                    status="processed",
                )
            insert_extraction_output(
                connection,
                document_id=document_id,
                run_id=None,
                method="text_extraction",
                provider="qwen",
                model="another-model",
                prompt_version="death_notice_v1",
                fields={"name": "other model"},
                status="processed",
            )


def test_review_selection_uses_enabled_exact_variants_and_skips_superseded() -> None:
    enabled = ExtractionVariant(
        alias="enabled",
        label="Enabled variant",
        method="text_extraction",
        provider="qwen",
        model="model-a",
        prompt_version="prompt-v2",
        review_enabled=True,
    )
    selected = _select_review_candidates(
        [
            {
                "id": 3,
                "extraction_output_id": 30,
                "extraction_method": "text_extraction",
                "extraction_provider": "qwen",
                "extraction_model": "model-a",
                "extraction_prompt_version": "prompt-v2",
                "superseded_at": "2026-01-01",
                "fields": {"name": "superseded"},
            },
            {
                "id": 2,
                "extraction_output_id": 20,
                "extraction_method": "text_extraction",
                "extraction_provider": "qwen",
                "extraction_model": "model-a",
                "extraction_prompt_version": "prompt-v2",
                "superseded_at": None,
                "fields": {"name": "selected"},
            },
            {
                "id": 1,
                "extraction_output_id": 10,
                "extraction_method": "text_extraction",
                "extraction_provider": "qwen",
                "extraction_model": "model-b",
                "extraction_prompt_version": "prompt-v2",
                "superseded_at": None,
                "fields": {"name": "disabled"},
            },
        ],
        (enabled,),
    )

    assert [candidate["id"] for candidate in selected] == [2]
    assert selected[0]["label"] == "Enabled variant"


def test_ingest_keeps_identical_fields_as_candidates_for_distinct_prompt_variants() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "test.sqlite3"
        results_file = root / "results.jsonl"
        records = [
            {
                "filename": "example.txt",
                "status": "processed",
                "method": "text_extraction",
                "provider": "qwen",
                "model": "qwen3.6-flash",
                "prompt_version": prompt,
                "row": {"dateiname": "example", "name": "Mustermann"},
            }
            for prompt in ("death_notice_v1", "death_notice_v3")
        ]
        results_file.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        first = ingest_results(
            db_path=db_path,
            source="source",
            results_file=results_file,
            method="text_extraction",
            provider="qwen",
            model="qwen3.6-flash",
        )
        second = ingest_results(
            db_path=db_path,
            source="source",
            results_file=results_file,
            method="text_extraction",
            provider="qwen",
            model="qwen3.6-flash",
        )

        assert (first.extraction_outputs, first.label_candidates) == (2, 2)
        assert (second.extraction_outputs, second.label_candidates) == (0, 0)
