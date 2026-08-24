import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from src.todesanzeigen.evaluation import evaluate_method
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
    upsert_document,
    upsert_ocr_output,
)


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
            self.assertIn("evaluation_results", tables)
            self.assertIn("extraction_methods", tables)
            self.assertIn("feature_snapshots", tables)

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
            self.assertEqual(candidate["source_kind"], "teacher")
            self.assertEqual(candidate["confidence_score"], 0.9)
            self.assertEqual(output["method"], "vision_model_image_only")

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
                candidate_kind="pipeline",
                input_dir=input_dir,
            )
            second = ingest_results(
                db_path=db_path,
                source="Aichacher Nachrichten",
                output_csv=output_csv,
                method="text_extraction",
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
                source_image_artifacts = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM artifacts
                    WHERE artifact_type = 'source_image'
                    """
                ).fetchone()
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
        self.assertEqual(source_image_artifacts["count"], 1)
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
        self.assertIn('href="/?method=text_extraction"', response.text)
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

            client = TestClient(create_review_app(db_path=db_path))
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pending Example", response.text)
        self.assertIn("Reviewed Ground Truth", response.text)
        self.assertIn("Reviewed Example", response.text)
        self.assertIn(f'href="/documents/{reviewed_document_id}"', response.text)

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

    def test_evaluate_method_records_field_metrics(self) -> None:
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
                    fields={"name": "Mustermann", "vorname": "Erika"},
                    status="processed",
                )

            summary = evaluate_method(
                db_path=db_path,
                label_set=DEFAULT_LABEL_SET,
                method="text_extraction",
            )

            self.assertEqual(summary.documents, 1)
            self.assertEqual(summary.exact_record_accuracy, 0)
            self.assertGreater(summary.field_precision, 0)
            self.assertGreater(summary.field_recall, 0)
            with connect(db_path) as connection:
                eval_run = connection.execute("SELECT * FROM evaluation_runs").fetchone()
                eval_result = connection.execute("SELECT * FROM evaluation_results").fetchone()
            self.assertEqual(eval_run["method"], "text_extraction")
            self.assertEqual(eval_result["exact_match"], 0)

    def test_export_priority_csv_prefers_ground_truth_then_vlm_then_text(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            output_csv = root / "output" / "result.csv"
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
                    fields={"name": "Ignored Vision"},
                    status="vision_processed",
                )
                insert_extraction_output(
                    connection,
                    document_id=vlm_document_id,
                    run_id=None,
                    method="vision_model_image_only",
                    fields={"name": "Vision"},
                    status="vision_processed",
                )
                insert_extraction_output(
                    connection,
                    document_id=text_document_id,
                    run_id=None,
                    method="text_extraction",
                    fields={"name": "Text"},
                    status="processed",
                )

                summary = export_priority_csv(connection, output_csv=output_csv)

            with output_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(summary.rows, 3)
        self.assertEqual(summary.ground_truth_rows, 1)
        self.assertEqual(summary.method_rows["vision_model_image_only"], 1)
        self.assertEqual(summary.method_rows["text_extraction"], 1)
        self.assertEqual(summary.missing_documents, 1)
        self.assertEqual([row["name"] for row in rows], ["Ground Truth", "Text", "Vision"])

    def test_feature_snapshots_and_router_export(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            output_file = root / "output" / "router.jsonl"
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
                    fields={"name": "Mustermann", "vorname": "Max"},
                    status="processed",
                )

            feature_summary = build_feature_snapshots(db_path=db_path, feature_set="router-v2")
            router_summary = export_router_dataset(
                db_path=db_path,
                output_file=output_file,
                label_set=DEFAULT_LABEL_SET,
                method="text_extraction",
                feature_set="router-v2",
            )
            row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(feature_summary.snapshots, 1)
        self.assertEqual(router_summary.rows, 1)
        self.assertFalse(row["target"]["cheap_pipeline_failed"])
        self.assertEqual(row["features"]["name_confidence"], 91)
        self.assertEqual(row["features"]["ocr_word_count"], 2)
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

            results = await extract_artifacts_to_db_async(
                artifacts_dir,
                db_path,
                Provider(),
                input_dir=input_dir,
                source="Aichacher Nachrichten",
                settings=AsyncExtractionSettings(max_retries=0),
            )

            with connect(db_path) as connection:
                output = connection.execute("SELECT * FROM extraction_outputs").fetchone()
                candidate = connection.execute("SELECT * FROM label_candidates").fetchone()
                document = connection.execute("SELECT * FROM documents").fetchone()

        self.assertEqual(results[0].status, "processed")
        self.assertEqual(output["method"], "text_extraction")
        self.assertEqual(output["method_family"], "ocr_llm")
        self.assertEqual(output["result_slot"], "ocr_llm")
        self.assertTrue(output["input_fingerprint"])
        self.assertTrue(output["config_hash"])
        self.assertEqual(output["provider"], "fake")
        self.assertEqual(json.loads(output["fields_json"])["name"], "Mustermann")
        self.assertEqual(candidate["source_kind"], "pipeline")
        self.assertEqual(document["image_path"], str(input_dir / "example.jpg"))

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

    async def test_image_only_vlm_uses_reroute_cache_slot(self) -> None:
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

        self.assertEqual(results[0].status, CACHED_EXISTING_STATUS)
        self.assertEqual(results[0].row["name"], "Cached VLM")
        self.assertEqual(output_count, 1)

    async def test_reroute_uses_image_only_vlm_cache_slot(self) -> None:
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

        self.assertEqual(results[0].status, CACHED_EXISTING_STATUS)
        self.assertEqual(results[0].row["name"], "Cached Direct VLM")
        self.assertEqual(output_count, 1)

    async def test_force_text_extraction_supersedes_existing_slot(self) -> None:
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
