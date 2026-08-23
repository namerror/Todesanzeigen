import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.evaluation import evaluate_method
from src.todesanzeigen.ingest import ingest_results, ingest_source
from src.todesanzeigen.llm import CSV_COLUMNS
from src.todesanzeigen.storage import (
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    insert_extraction_output,
    insert_label_candidate,
    load_candidate,
    save_ground_truth_label,
)


class MlInfrastructureTests(TestCase):
    def test_migrations_are_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "test.sqlite3"

            first = apply_migrations(db_path)
            second = apply_migrations(db_path)

            self.assertEqual(first, ["001_initial"])
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
