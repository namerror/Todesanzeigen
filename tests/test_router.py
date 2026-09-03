import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.router.dataset import RouterRecord, load_router_dataset
from src.todesanzeigen.router.labels import ROUTER_TARGET_FIELDS, score_target_fields
from src.todesanzeigen.router.manifest import write_router_manifest
from src.todesanzeigen.router.model import model_record_features, train_router_from_db
from src.todesanzeigen.storage import (
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    create_dataset_split,
    insert_extraction_output,
    save_ground_truth_label,
    upsert_document,
    upsert_feature_snapshot,
)


class RouterLabelTests(TestCase):
    def test_wohnort_is_not_a_separate_router_target(self) -> None:
        self.assertNotIn("wohnort", ROUTER_TARGET_FIELDS)

    def test_target_scoring_ignores_blank_truth_and_non_target_fields(self) -> None:
        metrics = score_target_fields(
            {
                "name": "Mustermann",
                "vorname": "Max",
                "geburtsort": "",
                "bemerkungen": "not a target",
            },
            {
                "name": "Mustermann",
                "vorname": "Max",
                "geburtsort": "Augsburg",
                "bemerkungen": "different",
            },
        )

        self.assertEqual(metrics.evaluated_fields, 2)
        self.assertEqual(metrics.field_f1, 1.0)
        self.assertTrue(metrics.exact_target_match)

    def test_target_scoring_counts_filled_target_mismatch(self) -> None:
        metrics = score_target_fields(
            {"name": "Mustermann", "vorname": "Max"},
            {"name": "Musterfrau", "vorname": "Max"},
        )

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.field_f1, 0.5)
        self.assertFalse(metrics.exact_target_match)


class RouterDatasetTests(TestCase):
    def test_model_inputs_come_only_from_the_feature_snapshot(self) -> None:
        record = RouterRecord(
            document_id=1,
            source="Aichacher Nachrichten",
            filename_stem="Example 2024",
            image_path="input/example.jpg",
            year=2024,
            features={"source": "Aichacher Nachrichten", "ocr_word_count": 12},
            truth=None,
            cheap_fields=None,
            vlm_fields=None,
            cheap_output_id=None,
            vlm_output_id=None,
            cheap_metrics=None,
            vlm_metrics=None,
            cheap_pipeline_failed=None,
        )

        features = model_record_features(record)

        self.assertEqual(features["source"], "Aichacher Nachrichten")
        self.assertEqual(features["ocr_word_count"], 12.0)
        self.assertNotIn("year", features)
        self.assertNotIn("filename_stem", features)
        self.assertNotIn("image_path", features)

    def test_load_router_dataset_pairs_features_gt_text_and_vlm_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            variants_config = _write_test_variants(root)
            apply_migrations(db_path)
            with connect(db_path) as connection:
                document_id = _insert_router_document(
                    connection,
                    filename_stem="example",
                    failed=True,
                    confidence=42,
                )

            dataset = load_router_dataset(
                db_path=db_path,
                label_set=DEFAULT_LABEL_SET,
                feature_set="router-v1",
                target_f1_threshold=0.95,
                variants_config=variants_config,
            )

        self.assertEqual(len(dataset.records), 1)
        record = dataset.records[0]
        self.assertEqual(record.document_id, document_id)
        self.assertTrue(record.cheap_pipeline_failed)
        self.assertEqual(record.cheap_output_id, 1)
        self.assertEqual(record.vlm_output_id, 2)
        self.assertEqual(record.vlm_metrics.field_f1, 1.0)


class RouterTrainingTests(TestCase):
    def test_training_fails_when_target_has_one_class(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            model_dir = root / "models" / "router"
            variants_config = _write_test_variants(root)
            apply_migrations(db_path)
            with connect(db_path) as connection:
                for index in range(4):
                    _insert_router_document(
                        connection,
                        filename_stem=f"success-{index}",
                        failed=False,
                        confidence=90 + index,
                    )

            with self.assertRaisesRegex(ValueError, "both cheap-pipeline success and failure"):
                train_router_from_db(
                    db_path=db_path,
                    feature_set="router-v1",
                    model_dir=model_dir,
                    validation_ratio=0.25,
                    min_train_rows=3,
                    variants_config=variants_config,
                )

    def test_training_writes_model_artifacts_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state" / "test.sqlite3"
            model_dir = root / "models" / "router"
            manifest_file = root / "output" / "router-manifest.jsonl"
            variants_config = _write_test_variants(root)
            apply_migrations(db_path)
            assignments: dict[int, str] = {}
            with connect(db_path) as connection:
                for index in range(8):
                    failed = index in {1, 3, 5, 7}
                    document_id = _insert_router_document(
                        connection,
                        filename_stem=f"doc-{index}",
                        failed=failed,
                        confidence=35 + index if failed else 90 + index,
                    )
                    assignments[document_id] = "validation" if index in {6, 7} else "train"
                create_dataset_split(
                    connection,
                    name="router-split",
                    strategy="manual-test",
                    assignments=assignments,
                )

            summary = train_router_from_db(
                db_path=db_path,
                feature_set="router-v1",
                model_dir=model_dir,
                split_name="router-split",
                validation_ratio=0.25,
                min_train_rows=4,
                variants_config=variants_config,
            )
            manifest_summary = write_router_manifest(
                db_path=db_path,
                model_dir=model_dir,
                feature_set="router-v1",
                output_file=manifest_file,
                threshold=0.0,
                variants_config=variants_config,
            )
            rows = [
                json.loads(line)
                for line in manifest_file.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary.training_rows, 6)
            self.assertEqual(summary.validation_rows, 2)
            self.assertTrue((model_dir / "model.joblib").exists())
            self.assertTrue((model_dir / "training-report.json").exists())
            self.assertTrue((model_dir / "feature-schema.json").exists())
            self.assertTrue((model_dir / "thresholds.json").exists())
            self.assertEqual(manifest_summary.rows, 8)
            self.assertEqual(manifest_summary.escalations, 8)
            self.assertEqual({row["route"] for row in rows}, {"vlm"})
            self.assertIn("predicted_failure_probability", rows[0])


def _insert_router_document(
    connection,
    *,
    filename_stem: str,
    failed: bool,
    confidence: float,
) -> int:
    document_id = upsert_document(
        connection,
        source_name="Aichacher Nachrichten",
        filename_stem=filename_stem,
        image_path=Path("input") / f"{filename_stem}.jpg",
        year=2024,
        layout_family="clean",
    )
    upsert_feature_snapshot(
        connection,
        document_id=document_id,
        feature_set="router-v1",
        features={
            "source": "Aichacher Nachrichten",
            "year": 2024,
            "layout_family": "clean",
            "name_confidence": confidence,
            "tsv_mean_confidence": confidence,
            "ocr_word_count": 12,
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
        fields={"name": "Musterfrau" if failed else "Mustermann", "vorname": "Max"},
        status="processed",
    )
    insert_extraction_output(
        connection,
        document_id=document_id,
        run_id=None,
        method="vision_model_image_only",
        provider="test",
        model="test-vision-model",
        fields={"name": "Mustermann", "vorname": "Max"},
        status="vision_processed",
    )
    return document_id


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
