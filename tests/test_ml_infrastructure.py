import csv
import json
import shutil
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from src.todesanzeigen.evaluation import evaluate_variant
from src.todesanzeigen.extract import (
    CACHED_EXISTING_STATUS,
    AsyncExtractionSettings,
    NameHint,
    RerouteCandidate,
    extract_artifacts_to_db_async,
    extract_images_to_db_async,
    reroute_candidates_to_db_async,
)
from src.todesanzeigen.features import build_feature_snapshots, export_router_dataset
from src.todesanzeigen.ingest import ingest_results, ingest_source
from src.todesanzeigen.llm import CSV_COLUMNS
from src.todesanzeigen.storage import (
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    create_run,
    export_priority_csv,
    insert_extraction_output,
    insert_label_candidate,
    load_candidate,
    pending_review_items,
    review_method_options,
    reviewed_ground_truth_items,
    save_ground_truth_label,
    sha256_file,
    upsert_document,
    upsert_ocr_output,
)


def _write_test_variants(root: Path) -> Path:
    path = root / "variants.toml"
    path.write_text(
        """[defaults]
text = "test_text"
vlm = "test_vlm"
[[variants]]
alias = "test_text"
label = "Test text"
method = "text_extraction"
provider = "test"
model = "test-text-model"
prompt_version = "death_notice_v3"
review_enabled = true
[[variants]]
alias = "test_vlm"
label = "Test VLM"
method = "vision_model_image_only"
provider = "test"
model = "test-vision-model"
prompt_version = "death_notice_v3"
review_enabled = true
""",
        encoding="utf-8",
    )
    return path


class MlInfrastructureTests(TestCase):
    def test_migrations_are_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"

            first = apply_migrations(db_path)
            second = apply_migrations(db_path)

            self.assertEqual(
                first,
                [
                    "001_initial",
                    "002_method_lineage_features",
                    "003_result_slot_cache",
                    "004_require_generation_model",
                    "005_remove_artifacts",
                    "006_extraction_variants",
                    "007_remove_evaluation_history",
                ],
            )
            self.assertEqual(second, [])
            with connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            self.assertIn("documents", tables)
            self.assertIn("ground_truth_labels", tables)
            self.assertNotIn("evaluation_runs", tables)
            self.assertNotIn("evaluation_results", tables)
            self.assertIn("extraction_methods", tables)
            self.assertIn("feature_snapshots", tables)
            self.assertNotIn("artifacts", tables)

    def test_remove_evaluation_history_migration_drops_existing_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            old_migrations = root / "old-migrations"
            old_migrations.mkdir()
            migrations = Path(__file__).resolve().parents[1] / "migrations"
            for path in sorted(migrations.glob("*.sql")):
                if path.name != "007_remove_evaluation_history.sql":
                    shutil.copy2(path, old_migrations / path.name)

            apply_migrations(db_path, old_migrations)
            with connect(db_path) as connection:
                connection.execute("INSERT INTO sources(id, name) VALUES (1, 'Test Source')")
                connection.execute(
                    """
                    INSERT INTO documents(id, document_key, source_id, filename_stem)
                    VALUES (10, 'test:example', 1, 'example')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO evaluation_runs(id, name, label_set, method)
                    VALUES (20, 'old-evaluation', 'gt-v1', 'text_extraction')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO evaluation_results(id, evaluation_run_id, document_id)
                    VALUES (30, 20, 10)
                    """
                )

            self.assertEqual(apply_migrations(db_path), ["007_remove_evaluation_history"])
            with connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                document = connection.execute("SELECT id FROM documents WHERE id = 10").fetchone()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]

            self.assertNotIn("evaluation_runs", tables)
            self.assertNotIn("evaluation_results", tables)
            self.assertEqual(document["id"], 10)
            self.assertEqual(integrity, "ok")

    def test_remove_artifacts_migration_preserves_lineage_and_foreign_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            pre_migrations = root / "pre-migrations"
            pre_migrations.mkdir()
            migrations = Path(__file__).resolve().parents[1] / "migrations"
            for name in (
                "001_initial.sql",
                "002_method_lineage_features.sql",
                "003_result_slot_cache.sql",
                "004_require_generation_model.sql",
            ):
                shutil.copy2(migrations / name, pre_migrations / name)

            apply_migrations(db_path, pre_migrations)
            with connect(db_path) as connection:
                connection.execute("INSERT INTO sources(id, name) VALUES (1, 'Test Source')")
                connection.execute(
                    """
                    INSERT INTO runs(id, command, method, provider, model, config_json, status)
                    VALUES ('ocr-run', 'ingest source', 'source_inventory', '', '', '{}', 'completed')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO runs(id, command, method, provider, model, config_json, status)
                    VALUES (
                        'import-run', 'ingest results', 'text_extraction', 'qwen',
                        'qwen3.6-flash', '{"output_csv":"output/result.csv"}', 'completed'
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO documents(
                        id, document_key, source_id, filename_stem,
                        image_path, image_sha256, mime_type
                    ) VALUES (
                        10, 'Test Source:input/example.jpg', 1, 'example',
                        'input/example.jpg', 'image-hash', 'image/jpeg'
                    )
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO artifacts(
                        id, document_id, run_id, artifact_type, path, sha256, producer
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (11, 10, "ocr-run", "source_image", "input/example.jpg", "image-hash", "ingest source"),
                        (12, 10, "ocr-run", "ocr_text", "artifacts/example.txt", "text-hash", "tesseract"),
                        (13, 10, "ocr-run", "ocr_tsv", "artifacts/example.tsv", "tsv-hash", "tesseract"),
                        (14, None, "import-run", "csv_output", "output/result.csv", "csv-hash", "ingest results"),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO ocr_outputs(
                        id, document_id, run_id, text, text_artifact_id, tsv_artifact_id
                    ) VALUES (20, 10, 'ocr-run', 'Max Mustermann', 12, 13)
                    """
                )
                connection.execute(
                    """
                    INSERT INTO extraction_outputs(
                        id, document_id, run_id, method, provider, model, fields_json,
                        status, source_artifact_id, method_family, result_slot, ocr_output_id
                    ) VALUES (
                        30, 10, 'import-run', 'text_extraction', 'qwen', 'qwen3.6-flash',
                        '{"name":"Mustermann"}', 'processed', 14, 'ocr_llm', 'ocr_llm', 20
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO label_candidates(
                        id, document_id, extraction_output_id, source_kind, fields_json
                    ) VALUES (40, 10, 30, 'pipeline', '{"name":"Mustermann"}')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO feature_snapshots(
                        id, document_id, ocr_output_id, feature_set, features_json
                    ) VALUES (50, 10, 20, 'router-v2', '{}')
                    """
                )
            self.assertEqual(
                apply_migrations(db_path),
                [
                    "005_remove_artifacts",
                    "006_extraction_variants",
                    "007_remove_evaluation_history",
                ],
            )

            with connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                ocr = connection.execute("SELECT * FROM ocr_outputs WHERE id = 20").fetchone()
                output = connection.execute(
                    "SELECT * FROM extraction_outputs WHERE id = 30"
                ).fetchone()
                run = connection.execute("SELECT * FROM runs WHERE id = 'import-run'").fetchone()
                candidate = connection.execute(
                    "SELECT * FROM label_candidates WHERE id = 40"
                ).fetchone()
                feature = connection.execute(
                    "SELECT * FROM feature_snapshots WHERE id = 50"
                ).fetchone()
                foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                ocr_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(ocr_outputs)")
                }
                output_columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(extraction_outputs)")
                }

            self.assertNotIn("artifacts", tables)
            self.assertEqual(ocr["text_path"], "artifacts/example.txt")
            self.assertEqual(ocr["text_sha256"], "text-hash")
            self.assertEqual(ocr["tsv_path"], "artifacts/example.tsv")
            self.assertEqual(ocr["tsv_sha256"], "tsv-hash")
            self.assertNotIn("text_artifact_id", ocr_columns)
            self.assertNotIn("tsv_artifact_id", ocr_columns)
            self.assertNotIn("source_artifact_id", output_columns)
            self.assertEqual(output["ocr_output_id"], 20)
            self.assertEqual(json.loads(run["config_json"])["output_csv_sha256"], "csv-hash")
            self.assertEqual(candidate["extraction_output_id"], 30)
            self.assertEqual(feature["ocr_output_id"], 20)
            self.assertEqual(foreign_key_errors, [])
            self.assertEqual(integrity, "ok")

    def test_model_backed_storage_requires_model_in_application_and_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)

            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Example",
                )
                with self.assertRaisesRegex(ValueError, "model is required"):
                    insert_extraction_output(
                        connection,
                        document_id=document_id,
                        run_id=None,
                        method="text_extraction",
                        fields={},
                        status="processed",
                    )
                with self.assertRaisesRegex(ValueError, "model is required"):
                    insert_extraction_output(
                        connection,
                        document_id=document_id,
                        run_id=None,
                        method="text_extraction",
                        method_family_value="manual",
                        fields={},
                        status="processed",
                    )
                with self.assertRaisesRegex(ValueError, "model is required"):
                    create_run(connection, command="extract", method="text_extraction")
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO extraction_outputs(document_id, method, fields_json, status)
                        VALUES (?, 'vision_model_image_only', '{}', 'vision_processed')
                        """,
                        (document_id,),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """
                        INSERT INTO runs(id, command, method)
                        VALUES ('direct-run', 'extract', 'text_extraction')
                        """
                    )

                output_id = insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    model="test-model",
                    fields={},
                    status="processed",
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE extraction_outputs SET model = '' WHERE id = ?",
                        (output_id,),
                    )

                imported_id = insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="csv_import",
                    fields={},
                    status="processed",
                )

            self.assertGreater(imported_id, output_id)

    def test_ingest_model_backed_results_requires_explicit_model(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "--model is required"):
                ingest_results(
                    db_path=root / "state" / "test.sqlite3",
                    output_csv=root / "result.csv",
                    method="text_extraction",
                )

    def test_ingest_jsonl_infers_model_lineage_from_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            results_file = root / "results.jsonl"
            results_file.write_text(
                json.dumps(
                    {
                        "filename": "Example.txt",
                        "status": "processed",
                        "method": "text_extraction",
                        "provider": "qwen",
                        "model": "qwen3.6-flash",
                        "row": {
                            "name": "Mustermann",
                            "quelle": "Aichacher Nachrichten",
                            "dateiname": "Example",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected_results_hash = sha256_file(results_file)

            summary = ingest_results(
                db_path=db_path,
                results_file=results_file,
                method="text_extraction",
            )

            with connect(db_path) as connection:
                output = connection.execute("SELECT * FROM extraction_outputs").fetchone()
                run = connection.execute("SELECT * FROM runs").fetchone()

        self.assertEqual(summary.extraction_outputs, 1)
        self.assertEqual(output["provider"], "qwen")
        self.assertEqual(output["model"], "qwen3.6-flash")
        self.assertEqual(run["provider"], "qwen")
        self.assertEqual(run["model"], "qwen3.6-flash")
        run_config = json.loads(run["config_json"])
        self.assertEqual(run_config["results_file"], str(results_file))
        self.assertEqual(run_config["results_file_sha256"], expected_results_hash)

    def test_ingest_source_records_images_artifacts_and_ocr_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            input_dir = root / "input" / "Aichacher Nachrichten"
            artifacts_dir = root / "artifacts" / "Aichacher Nachrichten"
            input_dir.mkdir(parents=True)
            artifacts_dir.mkdir(parents=True)
            (input_dir / "Example 2024_001.jpg").write_bytes(b"image")
            (artifacts_dir / "Example 2024_001.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts_dir / "Example 2024_001.tsv").write_text("level\ttext\n5\tMax", encoding="utf-8")
            (artifacts_dir / "name_map.json").write_text(
                json.dumps({"Example 2024_001.txt": {"name": "Max Mustermann", "confidence": 91}}),
                encoding="utf-8",
            )

            summary = ingest_source(
                db_path=db_path,
                source="Aichacher Nachrichten",
                input_dir=input_dir,
                artifacts_dir=artifacts_dir,
                layout_family="clean",
            )

            self.assertEqual(summary.documents, 1)
            self.assertEqual(summary.text_artifacts, 1)
            self.assertEqual(summary.tsv_artifacts, 1)
            self.assertEqual(summary.ocr_outputs, 1)
            with connect(db_path) as connection:
                document = connection.execute("SELECT * FROM documents").fetchone()
                ocr = connection.execute("SELECT * FROM ocr_outputs").fetchone()
            self.assertEqual(document["filename_stem"], "Example 2024_001")
            self.assertEqual(document["year"], 2024)
            self.assertEqual(ocr["text_path"], str(artifacts_dir / "Example 2024_001.txt"))
            self.assertTrue(ocr["text_sha256"])
            self.assertEqual(ocr["tsv_path"], str(artifacts_dir / "Example 2024_001.tsv"))
            self.assertTrue(ocr["tsv_sha256"])
            self.assertEqual(ocr["name_hint"], "Max Mustermann")
            self.assertEqual(ocr["name_confidence"], 91)

    def test_ingest_results_creates_candidates_for_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            output_csv = root / "result.csv"
            with output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        **{column: "" for column in CSV_COLUMNS},
                        "name": "Mustermann",
                        "vorname": "Max",
                        "quelle": "Aichacher Nachrichten",
                        "dateiname": "Example 2024_001",
                        "confidence_score": "0.9",
                    }
                )

            summary = ingest_results(
                db_path=db_path,
                output_csv=output_csv,
                method="vision_model_image_only",
                provider="qwen",
                model="qwen-vl-ocr",
            )

            self.assertEqual(summary.extraction_outputs, 1)
            self.assertEqual(summary.label_candidates, 1)
            with connect(db_path) as connection:
                candidate = connection.execute("SELECT * FROM label_candidates").fetchone()
                output = connection.execute("SELECT * FROM extraction_outputs").fetchone()
                run = connection.execute("SELECT * FROM runs").fetchone()
            self.assertEqual(candidate["source_kind"], "teacher")
            self.assertEqual(candidate["confidence_score"], 0.9)
            self.assertEqual(output["method"], "vision_model_image_only")
            run_config = json.loads(run["config_json"])
            self.assertEqual(run_config["output_csv"], str(output_csv))
            self.assertEqual(run_config["output_csv_sha256"], sha256_file(output_csv))

            second = ingest_results(
                db_path=db_path,
                output_csv=output_csv,
                method="vision_model_image_only",
                provider="qwen",
                model="qwen-vl-ocr",
            )

            self.assertEqual(second.extraction_outputs, 0)
            self.assertEqual(second.label_candidates, 0)
            with connect(db_path) as connection:
                output_total = connection.execute("SELECT COUNT(*) AS count FROM extraction_outputs").fetchone()
                candidate_total = connection.execute("SELECT COUNT(*) AS count FROM label_candidates").fetchone()
            self.assertEqual(output_total["count"], 1)
            self.assertEqual(candidate_total["count"], 1)

    def test_ingest_results_can_repair_legacy_document_image_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            input_dir = root / "input" / "Aichacher Nachrichten"
            input_dir.mkdir(parents=True)
            image_path = input_dir / "Example 2024_001.jpg"
            image_path.write_bytes(b"image")
            output_csv = root / "result.csv"
            with output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        **{column: "" for column in CSV_COLUMNS},
                        "name": "Mustermann",
                        "quelle": "Aichacher Nachrichten",
                        "dateiname": "Example 2024_ 001",
                    }
                )

            apply_migrations(db_path)
            with connect(db_path) as connection:
                legacy_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Example 2024_ 001",
                )

            summary = ingest_results(
                db_path=db_path,
                source="Aichacher Nachrichten",
                output_csv=output_csv,
                method="text_extraction",
                provider="qwen",
                model="qwen3.6-flash",
                candidate_kind="pipeline",
                input_dir=input_dir,
            )
            second = ingest_results(
                db_path=db_path,
                source="Aichacher Nachrichten",
                output_csv=output_csv,
                method="text_extraction",
                provider="qwen",
                model="qwen3.6-flash",
                candidate_kind="pipeline",
                input_dir=input_dir,
            )

            self.assertEqual(summary.extraction_outputs, 1)
            self.assertEqual(summary.label_candidates, 1)
            self.assertEqual(second.extraction_outputs, 0)
            self.assertEqual(second.label_candidates, 0)
            with connect(db_path) as connection:
                documents = connection.execute("SELECT * FROM documents").fetchall()
                document = documents[0]
                outputs = connection.execute(
                    "SELECT COUNT(*) AS count FROM extraction_outputs"
                ).fetchone()
                candidates = connection.execute(
                    "SELECT COUNT(*) AS count FROM label_candidates"
                ).fetchone()

        self.assertEqual(len(documents), 1)
        self.assertEqual(document["id"], legacy_document_id)
        self.assertEqual(document["filename_stem"], "Example 2024_ 001")
        self.assertEqual(document["image_path"], str(image_path))
        self.assertTrue(document["image_sha256"])
        self.assertEqual(document["mime_type"], "image/jpeg")
        self.assertEqual(outputs["count"], 1)
        self.assertEqual(candidates["count"], 1)

    def test_candidate_approval_writes_ground_truth_label(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                from src.todesanzeigen.storage import upsert_document

                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Example",
                )
                candidate_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="teacher",
                    source_name="vision",
                    fields={"name": "Mustermann", "vorname": "Max"},
                )
                candidate = load_candidate(connection, candidate_id)
                save_ground_truth_label(
                    connection,
                    document_id=document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields=json.loads(candidate["fields_json"]),
                    source_candidate_id=candidate_id,
                    reviewer="tester",
                )
                label = connection.execute("SELECT * FROM ground_truth_labels").fetchone()
                candidate = connection.execute("SELECT * FROM label_candidates").fetchone()

            self.assertEqual(label["reviewer"], "tester")
            self.assertEqual(label["source_candidate_id"], candidate_id)
            self.assertEqual(candidate["status"], "approved")

    def test_pending_review_items_can_filter_by_source_name(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                text_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Text Example",
                )
                vision_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Vision Example",
                )
                insert_label_candidate(
                    connection,
                    document_id=text_document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Text"},
                )
                insert_label_candidate(
                    connection,
                    document_id=vision_document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "Vision"},
                )

                items = pending_review_items(
                    connection,
                    source_name="vision_model_image_only",
                )

            self.assertEqual([item["filename_stem"] for item in items], ["Vision Example"])

    def test_reviewed_ground_truth_items_list_label_set_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                reviewed_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Reviewed Example",
                )
                other_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Other Label Set",
                )
                candidate_id = insert_label_candidate(
                    connection,
                    document_id=reviewed_document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "Reviewed"},
                )
                save_ground_truth_label(
                    connection,
                    document_id=reviewed_document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Reviewed"},
                    source_candidate_id=candidate_id,
                    reviewer="tester",
                )
                save_ground_truth_label(
                    connection,
                    document_id=other_document_id,
                    label_set="gt-v2",
                    fields={"name": "Other"},
                    reviewer="tester",
                )

                items = reviewed_ground_truth_items(connection, label_set=DEFAULT_LABEL_SET)

            self.assertEqual([item["filename_stem"] for item in items], ["Reviewed Example"])
            self.assertEqual(items[0]["source"], "Aichacher Nachrichten")
            self.assertEqual(items[0]["source_candidate_id"], candidate_id)
            self.assertEqual(items[0]["source_kind"], "teacher")
            self.assertEqual(items[0]["source_name"], "vision_model_image_only")

    def test_review_method_options_come_from_extraction_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                methods = [row["method"] for row in review_method_options(connection)]

            self.assertEqual(
                methods,
                [
                    "text_extraction",
                    "vision_model_image_only",
                    "vision_model_reroute",
                ],
            )

    def test_review_home_page_filters_by_method_query_param(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                text_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Text Example",
                )
                vision_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Vision Example",
                )
                insert_label_candidate(
                    connection,
                    document_id=text_document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Text"},
                )
                insert_label_candidate(
                    connection,
                    document_id=vision_document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "Vision"},
                )

            client = TestClient(create_review_app(db_path=db_path))
            response = client.get("/?method=vision_model_image_only")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Vision Example", response.text)
        self.assertNotIn("Text Example", response.text)
        self.assertIn('href="/?view=pending&amp;method=text_extraction"', response.text)
        self.assertNotIn('name="method"', response.text)

    def test_review_home_page_links_reviewed_ground_truth_items(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                reviewed_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Reviewed Example",
                )
                pending_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Pending Example",
                )
                insert_label_candidate(
                    connection,
                    document_id=pending_document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Pending"},
                )
                save_ground_truth_label(
                    connection,
                    document_id=reviewed_document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Reviewed"},
                    reviewer="tester",
                )
                newer_candidate_id = insert_label_candidate(
                    connection,
                    document_id=reviewed_document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "New Alternative"},
                )
                connection.execute(
                    "UPDATE label_candidates SET created_at = datetime('now', '+1 minute') WHERE id = ?",
                    (newer_candidate_id,),
                )

            client = TestClient(create_review_app(db_path=db_path))
            pending_response = client.get("/")
            response = client.get("/?view=reviewed")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pending Example", pending_response.text)
        self.assertNotIn("Reviewed Example", pending_response.text)
        self.assertIn("Ground truth", response.text)
        self.assertIn("Reviewed Example", response.text)
        self.assertIn("1 newer candidate", response.text)
        self.assertIn(
            f'href="/documents/{reviewed_document_id}?view=reviewed&amp;page=1"',
            response.text,
        )

    def test_review_document_starts_empty_until_a_source_is_chosen(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Unreviewed Example",
                )
                candidate_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Candidate Name", "vorname": "Candidate First"},
                )

            response = TestClient(create_review_app(db_path=db_path)).get(
                f"/documents/{document_id}"
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="name" name="name" value=""', response.text)
        self.assertIn('id="source-candidate-id" type="hidden" name="source_candidate_id" value=""', response.text)
        self.assertIn(f'data-candidate-id="{candidate_id}"', response.text)
        self.assertIn("Candidate Name", response.text)
        self.assertIn("Fill form", response.text)
        self.assertIn("Approve as GT", response.text)

    def test_review_document_only_shows_latest_candidate_per_method(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Candidate Versions",
                )
                insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Stale Candidate"},
                )
                latest_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Latest Candidate"},
                )

            response = TestClient(create_review_app(db_path=db_path)).get(
                f"/documents/{document_id}"
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Latest Candidate", response.text)
        self.assertNotIn("Stale Candidate", response.text)
        self.assertIn(f'data-candidate-id="{latest_id}"', response.text)

    def test_direct_approval_replaces_gt_and_preserves_notes(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Replacement Example",
                )
                original_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Original"},
                )
                replacement_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "Replacement"},
                )
                original_label_id = save_ground_truth_label(
                    connection,
                    document_id=document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Original"},
                    source_candidate_id=original_id,
                    reviewer="first reviewer",
                    review_notes="keep this context",
                )

            response = TestClient(
                create_review_app(db_path=db_path, reviewer="second reviewer")
            ).post(
                f"/candidates/{replacement_id}/approve",
                data={"continue_mode": "stay"},
                follow_redirects=False,
            )
            with connect(db_path) as connection:
                labels = connection.execute("SELECT * FROM ground_truth_labels").fetchall()
                candidates = {
                    row["id"]: row["status"]
                    for row in connection.execute("SELECT id, status FROM label_candidates")
                }

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], f"/documents/{document_id}")
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["id"], original_label_id)
        self.assertEqual(labels[0]["source_candidate_id"], replacement_id)
        self.assertEqual(labels[0]["review_notes"], "keep this context")
        self.assertEqual(json.loads(labels[0]["fields_json"])["name"], "Replacement")
        self.assertEqual(candidates[original_id], "superseded")
        self.assertEqual(candidates[replacement_id], "approved")

    def test_needs_review_leaves_record_pending_and_advances(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                first_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="First Pending",
                )
                second_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Second Pending",
                )
                candidate_id = insert_label_candidate(
                    connection,
                    document_id=first_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "First"},
                )
                insert_label_candidate(
                    connection,
                    document_id=second_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Second"},
                )

            response = TestClient(create_review_app(db_path=db_path)).post(
                f"/documents/{first_id}/needs-review",
                follow_redirects=False,
            )
            with connect(db_path) as connection:
                gt_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM ground_truth_labels"
                ).fetchone()["count"]
                candidate_status = load_candidate(connection, candidate_id)["status"]

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], f"/documents/{second_id}")
        self.assertEqual(gt_count, 0)
        self.assertEqual(candidate_status, "pending")

    def test_label_form_rejects_candidate_from_another_document(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Target",
                )
                other_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Other",
                )
                other_candidate_id = insert_label_candidate(
                    connection,
                    document_id=other_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Wrong Document"},
                )

            response = TestClient(create_review_app(db_path=db_path)).post(
                f"/documents/{document_id}/labels",
                data={"name": "Invalid", "source_candidate_id": str(other_candidate_id)},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not belong", response.text)

    def test_review_document_page_edits_existing_ground_truth_label(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed")

        from src.todesanzeigen.review import create_review_app

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Reviewed Example",
                )
                original_candidate_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="teacher",
                    source_name="vision_model_image_only",
                    fields={"name": "Original"},
                )
                save_ground_truth_label(
                    connection,
                    document_id=document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Original", "vorname": "Max"},
                    source_candidate_id=original_candidate_id,
                    reviewer="tester",
                    review_notes="original notes",
                )
                newer_candidate_id = insert_label_candidate(
                    connection,
                    document_id=document_id,
                    source_kind="pipeline",
                    source_name="text_extraction",
                    fields={"name": "Newer Candidate"},
                )

            client = TestClient(create_review_app(db_path=db_path, reviewer="editor"))
            response = client.get(f"/documents/{document_id}")
            self.assertEqual(response.status_code, 200)
            self.assertIn(
                f'name="source_candidate_id" value="{original_candidate_id}"',
                response.text,
            )
            self.assertNotIn(
                f'name="source_candidate_id" value="{newer_candidate_id}"',
                response.text,
            )

            form_data = {column: "" for column in CSV_COLUMNS}
            form_data.update(
                {
                    "name": "Edited",
                    "vorname": "Erika",
                    "source_candidate_id": str(original_candidate_id),
                    "review_notes": "corrected spelling",
                }
            )
            response = client.post(
                f"/documents/{document_id}/labels",
                data=form_data,
                follow_redirects=False,
            )

            with connect(db_path) as connection:
                label = connection.execute("SELECT * FROM ground_truth_labels").fetchone()

        self.assertEqual(response.status_code, 303)
        self.assertEqual(label["source_candidate_id"], original_candidate_id)
        self.assertEqual(label["reviewer"], "editor")
        self.assertEqual(label["review_notes"], "corrected spelling")
        fields = json.loads(label["fields_json"])
        self.assertEqual(fields["name"], "Edited")
        self.assertEqual(fields["vorname"], "Erika")

    def test_evaluate_variant_records_field_metrics_and_telemetry(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            apply_migrations(db_path)
            with connect(db_path) as connection:
                from src.todesanzeigen.storage import upsert_document

                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Example",
                )
                save_ground_truth_label(
                    connection,
                    document_id=document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Mustermann", "vorname": "Max"},
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="test",
                    model="test-text-model",
                    fields={"name": "Mustermann", "vorname": "Erika"},
                    status="processed",
                    estimated_tokens=125,
                    latency_ms=80,
                    cost_usd=0.0125,
                )

            variants_config = _write_test_variants(root)
            summary = evaluate_variant(
                db_path=db_path,
                label_set=DEFAULT_LABEL_SET,
                variant_alias="test_text",
                variants_config=variants_config,
            )

            self.assertEqual(summary.documents, 1)
            self.assertEqual(summary.exact_record_accuracy, 0)
            self.assertGreater(summary.field_precision, 0)
            self.assertGreater(summary.field_recall, 0)
            self.assertEqual(summary.estimated_tokens_total, 125)
            self.assertEqual(summary.latency_ms_mean, 80)
            self.assertEqual(summary.cost_usd_total, 0.0125)
            with connect(db_path) as connection:
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            self.assertNotIn("evaluation_runs", tables)
            self.assertNotIn("evaluation_results", tables)

    def test_export_priority_csv_prefers_ground_truth_then_vlm_then_text(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            output_csv = root / "output" / "result.csv"
            variants_config = _write_test_variants(root)
            apply_migrations(db_path)
            with connect(db_path) as connection:
                gt_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="gt-doc",
                )
                vlm_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="vlm-doc",
                )
                text_document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="text-doc",
                )
                upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="missing-doc",
                )
                save_ground_truth_label(
                    connection,
                    document_id=gt_document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Ground Truth", "dateiname": "gt-doc"},
                )
                insert_extraction_output(
                    connection,
                    document_id=gt_document_id,
                    run_id=None,
                    method="vision_model_image_only",
                    provider="test",
                    model="test-vision-model",
                    fields={"name": "Ignored Vision"},
                    status="vision_processed",
                )
                insert_extraction_output(
                    connection,
                    document_id=vlm_document_id,
                    run_id=None,
                    method="vision_model_image_only",
                    provider="test",
                    model="test-vision-model",
                    fields={"name": "Vision"},
                    status="vision_processed",
                )
                insert_extraction_output(
                    connection,
                    document_id=text_document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="test",
                    model="test-text-model",
                    fields={"name": "Text"},
                    status="processed",
                )

                summary = export_priority_csv(
                    connection,
                    output_csv=output_csv,
                    variants_config=variants_config,
                )

            with output_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(summary.rows, 3)
        self.assertEqual(summary.ground_truth_rows, 1)
        self.assertEqual(summary.method_rows["test_vlm"], 1)
        self.assertEqual(summary.method_rows["test_text"], 1)
        self.assertEqual(summary.missing_documents, 1)
        self.assertEqual([row["name"] for row in rows], ["Ground Truth", "Text", "Vision"])

    def test_feature_snapshots_and_router_export(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            output_file = root / "output" / "router.jsonl"
            tsv_path = root / "artifacts" / "example.tsv"
            variants_config = _write_test_variants(root)
            tsv_path.parent.mkdir()
            tsv_path.write_text(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                "left\ttop\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t90\tMax\n",
                encoding="utf-8",
            )
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="Example 2024",
                    mime_type="image/jpeg",
                    year=2024,
                    layout_family="clean",
                )
                run_id = create_run(connection, command="test", method="source_inventory")
                upsert_ocr_output(
                    connection,
                    document_id=document_id,
                    run_id=run_id,
                    text="Max Mustermann",
                    tsv_path=tsv_path,
                    tsv_sha256=sha256_file(tsv_path),
                    name_hint="Max Mustermann",
                    name_confidence=91,
                    features={
                        "ocr_word_count": 999,
                        "year": 2024,
                        "unreviewed_database_field": "leak",
                    },
                )
                save_ground_truth_label(
                    connection,
                    document_id=document_id,
                    label_set=DEFAULT_LABEL_SET,
                    fields={"name": "Mustermann", "vorname": "Max"},
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="test",
                    model="test-text-model",
                    fields={"name": "Mustermann", "vorname": "Max"},
                    status="processed",
                )

            feature_summary = build_feature_snapshots(db_path=db_path, feature_set="router-v2")
            router_summary = export_router_dataset(
                db_path=db_path,
                output_file=output_file,
                label_set=DEFAULT_LABEL_SET,
                variant_alias="test_text",
                feature_set="router-v2",
                variants_config=variants_config,
            )
            row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(feature_summary.snapshots, 1)
        self.assertEqual(router_summary.rows, 1)
        self.assertFalse(row["target"]["cheap_pipeline_failed"])
        self.assertEqual(row["features"]["name_confidence"], 91)
        self.assertEqual(row["features"]["ocr_word_count"], 2)
        self.assertEqual(row["features"]["tsv_word_count"], 1)
        self.assertEqual(row["features"]["tsv_mean_confidence"], 90)
        self.assertEqual(row["features"]["source"], "Aichacher Nachrichten")
        self.assertTrue(
            {
                "filename_length",
                "layout_family",
                "year",
                "image_mime_type",
                "image_path_present",
                "image_suffix",
                "unreviewed_database_field",
            }.isdisjoint(row["features"])
        )


class DbFirstExtractionTests(IsolatedAsyncioTestCase):
    async def test_extract_artifacts_to_db_records_output_and_candidate(self) -> None:
        class Provider:
            provider_name = "fake"
            model_name = "fake-model"

            async def async_complete(self, prompt: str) -> str:
                return json.dumps({"name": "Mustermann", "vorname": "Max"})

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            input_dir = root / "input"
            artifacts_dir.mkdir()
            input_dir.mkdir()
            (artifacts_dir / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts_dir / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 93}}),
                encoding="utf-8",
            )
            (input_dir / "example.jpg").write_bytes(b"image")
            checkpoint_file = root / "logs" / "results.jsonl"

            results = await extract_artifacts_to_db_async(
                artifacts_dir,
                db_path,
                Provider(),
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                checkpoint_file=checkpoint_file,
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output = connection.execute("SELECT * FROM extraction_outputs").fetchone()
                candidate = connection.execute("SELECT * FROM label_candidates").fetchone()
                document = connection.execute("SELECT * FROM documents").fetchone()
                run = connection.execute("SELECT * FROM runs WHERE command = 'extract'").fetchone()
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(results[0].status, "processed")
        self.assertEqual(output["method"], "text_extraction")
        self.assertEqual(output["method_family"], "ocr_llm")
        self.assertEqual(output["result_slot"], "ocr_llm")
        self.assertTrue(output["input_fingerprint"])
        self.assertTrue(output["config_hash"])
        self.assertEqual(output["provider"], "fake")
        self.assertEqual(output["model"], "fake-model")
        self.assertEqual(run["provider"], "fake")
        self.assertEqual(run["model"], "fake-model")
        self.assertEqual(checkpoint["provider"], "fake")
        self.assertEqual(checkpoint["model"], "fake-model")
        self.assertEqual(json.loads(output["fields_json"])["name"], "Mustermann")
        self.assertEqual(candidate["source_kind"], "pipeline")
        self.assertEqual(document["image_path"], str(input_dir / "example.jpg"))

    async def test_db_generation_rejects_provider_without_model_before_writing(self) -> None:
        class ProviderWithoutModel:
            provider_name = "fake"

            async def async_complete(self, prompt: str) -> str:
                raise AssertionError("provider should not be called")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "non-empty model_name"):
                await extract_artifacts_to_db_async(
                    artifacts_dir,
                    db_path,
                    ProviderWithoutModel(),
                )

            self.assertFalse(db_path.exists())

    async def test_image_only_generation_records_configured_model(self) -> None:
        class VisionProvider:
            provider_name = "fake-vision"
            model_name = "fake-vision-model"

            async def async_vision_complete(
                self,
                prompt: str,
                image_path: Path,
                mime_type: str,
            ) -> str:
                return json.dumps({"name": "Mustermann"})

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            input_dir = root / "input"
            checkpoint_file = root / "logs" / "vision-results.jsonl"
            input_dir.mkdir()
            (input_dir / "example.jpg").write_bytes(b"image")

            results = await extract_images_to_db_async(
                input_dir,
                db_path,
                VisionProvider(),
                source="Aichacher Nachrichten",
                results_file=checkpoint_file,
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output = connection.execute("SELECT * FROM extraction_outputs").fetchone()
                run = connection.execute(
                    "SELECT * FROM runs WHERE command = 'vision-extract'"
                ).fetchone()
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(results[0].status, "vision_processed")
        self.assertEqual(output["provider"], "fake-vision")
        self.assertEqual(output["model"], "fake-vision-model")
        self.assertEqual(run["model"], "fake-vision-model")
        self.assertEqual(checkpoint["model"], "fake-vision-model")

    async def test_text_extraction_uses_db_cache_before_provider_call(self) -> None:
        class FailingProvider:
            provider_name = "fake"
            model_name = "fake-model"

            async def async_complete(self, prompt: str) -> str:
                raise AssertionError("provider should not be called for cached text output")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            input_dir = root / "input"
            artifacts_dir.mkdir()
            input_dir.mkdir()
            artifact_path = artifacts_dir / "example.txt"
            image_path = input_dir / "example.jpg"
            artifact_path.write_text("Max Mustermann", encoding="utf-8")
            image_path.write_bytes(b"image")
            (artifacts_dir / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 93}}),
                encoding="utf-8",
            )
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="example",
                    image_path=image_path,
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="fake",
                    model="fake-model",
                    fields={"name": "Cached Text"},
                    status="processed",
                )

            results = await extract_artifacts_to_db_async(
                artifacts_dir,
                db_path,
                FailingProvider(),
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output_count = connection.execute("SELECT COUNT(*) AS count FROM extraction_outputs").fetchone()["count"]

        self.assertEqual(results[0].status, CACHED_EXISTING_STATUS)
        self.assertEqual(results[0].row["name"], "Cached Text")
        self.assertEqual(output_count, 1)

    async def test_image_only_vlm_does_not_use_reroute_variant_cache(self) -> None:
        class FailingVisionProvider:
            provider_name = "fake-vision"
            model_name = "fake-vision-model"

            async def async_vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
                raise AssertionError("provider should not be called for cached VLM output")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            input_dir = root / "input"
            input_dir.mkdir()
            image_path = input_dir / "example.jpg"
            image_path.write_bytes(b"image")
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="example",
                    image_path=image_path,
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="vision_model_reroute",
                    provider="fake-vision",
                    model="fake-vision-model",
                    fields={"name": "Cached VLM"},
                    status="rerouted_processed",
                )

            results = await extract_images_to_db_async(
                input_dir,
                db_path,
                FailingVisionProvider(),
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output_count = connection.execute("SELECT COUNT(*) AS count FROM extraction_outputs").fetchone()["count"]

        self.assertEqual(results[0].status, "vision_failed")
        self.assertEqual(output_count, 2)

    async def test_reroute_does_not_use_image_only_variant_cache(self) -> None:
        class FailingVisionProvider:
            provider_name = "fake-vision"
            model_name = "fake-vision-model"

            async def async_vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
                raise AssertionError("provider should not be called for cached VLM output")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            input_dir = root / "input"
            artifacts_dir.mkdir()
            input_dir.mkdir()
            artifact_path = artifacts_dir / "example.txt"
            image_path = input_dir / "example.jpg"
            artifact_path.write_text("Max Mustermann", encoding="utf-8")
            image_path.write_bytes(b"image")
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="example",
                    image_path=image_path,
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="vision_model_image_only",
                    provider="fake-vision",
                    model="fake-vision-model",
                    fields={"name": "Cached Direct VLM"},
                    status="vision_processed",
                )

            results = await reroute_candidates_to_db_async(
                [RerouteCandidate(artifact_path, NameHint("Max Mustermann", 60), 85)],
                db_path,
                FailingVisionProvider(),
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output_count = connection.execute("SELECT COUNT(*) AS count FROM extraction_outputs").fetchone()["count"]

        self.assertEqual(results[0].status, "rerouted_failed")
        self.assertEqual(output_count, 2)

    async def test_force_text_extraction_supersedes_existing_variant(self) -> None:
        class Provider:
            provider_name = "fake"
            model_name = "fake-model"

            def __init__(self) -> None:
                self.calls = 0

            async def async_complete(self, prompt: str) -> str:
                self.calls += 1
                return json.dumps({"name": "Fresh Text"})

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            input_dir = root / "input"
            artifacts_dir.mkdir()
            input_dir.mkdir()
            artifact_path = artifacts_dir / "example.txt"
            image_path = input_dir / "example.jpg"
            artifact_path.write_text("Max Mustermann", encoding="utf-8")
            image_path.write_bytes(b"image")
            (artifacts_dir / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 93}}),
                encoding="utf-8",
            )
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="example",
                    image_path=image_path,
                )
                insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="fake",
                    model="fake-model",
                    fields={"name": "Old Text"},
                    status="processed",
                )

            provider = Provider()
            results = await extract_artifacts_to_db_async(
                artifacts_dir,
                db_path,
                provider,
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
                force=True,
            )

            with connect(db_path) as connection:
                rows = connection.execute(
                    "SELECT * FROM extraction_outputs ORDER BY id"
                ).fetchall()

        self.assertEqual(provider.calls, 1)
        self.assertEqual(results[0].status, "processed")
        self.assertEqual(len(rows), 2)
        self.assertIsNotNone(rows[0]["superseded_at"])
        self.assertIsNone(rows[1]["superseded_at"])
        self.assertEqual(json.loads(rows[1]["fields_json"])["name"], "Fresh Text")

    async def test_failed_forced_extraction_keeps_previous_variant_active(self) -> None:
        class FailingProvider:
            provider_name = "fake"
            model_name = "fake-model"

            async def async_complete(self, prompt: str) -> str:
                raise RuntimeError("provider unavailable")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            artifacts_dir = root / "artifacts"
            input_dir = root / "input"
            artifacts_dir.mkdir()
            input_dir.mkdir()
            artifact_path = artifacts_dir / "example.txt"
            image_path = input_dir / "example.jpg"
            artifact_path.write_text("Max Mustermann", encoding="utf-8")
            image_path.write_bytes(b"image")
            (artifacts_dir / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 93}}),
                encoding="utf-8",
            )
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = upsert_document(
                    connection,
                    source_name="Aichacher Nachrichten",
                    filename_stem="example",
                    image_path=image_path,
                )
                old_id = insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=None,
                    method="text_extraction",
                    provider="fake",
                    model="fake-model",
                    fields={"name": "Old Text"},
                    status="processed",
                )

            results = await extract_artifacts_to_db_async(
                artifacts_dir,
                db_path,
                FailingProvider(),
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
                force=True,
            )

            with connect(db_path) as connection:
                old = connection.execute(
                    "SELECT superseded_at FROM extraction_outputs WHERE id = ?", (old_id,)
                ).fetchone()
                statuses = [
                    row["status"]
                    for row in connection.execute("SELECT status FROM extraction_outputs ORDER BY id")
                ]

        self.assertEqual(results[0].status, "failed")
        self.assertIsNone(old["superseded_at"])
        self.assertEqual(statuses, ["processed", "failed"])
