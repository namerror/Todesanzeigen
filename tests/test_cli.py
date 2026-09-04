from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from src.todesanzeigen import cli
from src.todesanzeigen.ocr import ConfigError
from src.todesanzeigen.ocr_filtering import NameFilterResult


class CliOcrTests(TestCase):
    def test_ocr_defaults_to_tesseract(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            artifacts_dir = root / "artifacts"
            input_dir.mkdir()
            args = cli.build_parser().parse_args(
                [
                    "ocr",
                    "--input-dir",
                    str(input_dir),
                    "--artifacts-dir",
                    str(artifacts_dir),
                ]
            )

            with (
                patch("src.todesanzeigen.cli.TesseractOcrClient") as tesseract_client,
                patch("src.todesanzeigen.cli.DocumentAiSettings.from_env") as document_settings,
                patch("src.todesanzeigen.cli.run_ocr_folder", return_value=[]) as run_folder,
            ):
                tesseract_client.return_value = "local-client"

                cli.run_ocr_command(args)

        tesseract_client.assert_called_once()
        settings = tesseract_client.call_args.args[0]
        self.assertEqual(settings.language, "deu")
        self.assertEqual(settings.page_segmentation_mode, 3)
        self.assertEqual(settings.ocr_engine_mode, 1)
        self.assertEqual(settings.tessdata_dir, Path("data"))
        self.assertEqual(settings.binary, "tesseract")
        document_settings.assert_not_called()
        run_folder.assert_called_once_with(
            input_dir,
            artifacts_dir,
            "local-client",
            overwrite=False,
            limit=None,
        )

    def test_router_workflows_default_to_feature_set_v2(self) -> None:
        parser = cli.build_parser()

        feature_args = parser.parse_args(["features", "build"])
        export_args = parser.parse_args(
            ["dataset", "export-router", "--variant", "text", "--output-file", "rows.jsonl"]
        )
        train_args = parser.parse_args(["router", "train", "--model-dir", "model"])
        manifest_args = parser.parse_args(
            ["router", "manifest", "--model-dir", "model", "--output-file", "routes.jsonl"]
        )
        eval_args = parser.parse_args(["eval", "run", "--variant", "text_current"])

        self.assertEqual(feature_args.feature_set, "router-v2")
        self.assertEqual(export_args.feature_set, "router-v2")
        self.assertEqual(train_args.feature_set, "router-v2")
        self.assertEqual(manifest_args.feature_set, "router-v2")
        self.assertEqual(eval_args.variant, "text_current")
        self.assertEqual(eval_args.variants_config, Path("config/extraction_variants.toml"))
        self.assertFalse(hasattr(eval_args, "name"))

    def test_eval_prints_terminal_summary_without_run_id(self) -> None:
        args = cli.build_parser().parse_args(
            ["eval", "run", "--variant", "text_current"]
        )
        summary = SimpleNamespace(
            documents=2,
            exact_record_accuracy=0.5,
            field_f1=0.75,
            field_precision=0.8,
            field_recall=0.7,
            missing_predictions=1,
            estimated_tokens_total=100,
            estimated_tokens_count=1,
            latency_ms_mean=25.0,
            latency_ms_count=1,
            cost_usd_total=0.01,
            cost_usd_count=1,
        )

        output = StringIO()
        with (
            patch("src.todesanzeigen.evaluation.evaluate_variant", return_value=summary),
            redirect_stdout(output),
        ):
            result = cli.run_eval_command(args)

        self.assertEqual(result, 0)
        self.assertTrue(output.getvalue().startswith("Evaluation: 2 documents;"))

    def test_documentai_requires_explicit_gcp_unlock(self) -> None:
        args = cli.build_parser().parse_args(["ocr", "--engine", "documentai"])

        with (
            patch.dict("src.todesanzeigen.cli.os.environ", {}, clear=True),
            patch("src.todesanzeigen.cli.DocumentAiSettings.from_env") as document_settings,
            patch("src.todesanzeigen.cli.DocumentAiOcrClient") as document_client,
        ):
            with self.assertRaises(ConfigError) as error:
                cli.run_ocr_command(args)

        self.assertIn("disabled to avoid accidental billing", str(error.exception))
        document_settings.assert_not_called()
        document_client.assert_not_called()

    def test_documentai_runs_when_explicitly_unlocked(self) -> None:
        args = cli.build_parser().parse_args(["ocr", "--engine", "documentai"])

        with (
            patch.dict("src.todesanzeigen.cli.os.environ", {"TODESANZEIGEN_ALLOW_GCP": "1"}),
            patch("src.todesanzeigen.cli.DocumentAiSettings.from_env") as document_settings,
            patch("src.todesanzeigen.cli.DocumentAiOcrClient") as document_client,
            patch("src.todesanzeigen.cli.run_ocr_folder", return_value=[]) as run_folder,
        ):
            document_settings.return_value = "document-settings"
            document_client.return_value = "document-client"

            cli.run_ocr_command(args)

        document_settings.assert_called_once_with()
        document_client.assert_called_once_with("document-settings")
        run_folder.assert_called_once_with(
            Path("input"),
            Path("artifacts"),
            "document-client",
            overwrite=False,
            limit=None,
        )

    def test_filter_command_prints_detected_names(self) -> None:
        args = cli.build_parser().parse_args(
            ["filter", "--artifacts-dir", "custom-artifacts", "--limit", "2"]
        )

        with patch("src.todesanzeigen.cli.filter_artifact_names") as filter_names:
            filter_names.return_value = [
                NameFilterResult(Path("custom-artifacts/a.tsv"), "Max Mustermann", 93.25),
                NameFilterResult(Path("custom-artifacts/b.tsv"), "", None),
            ]
            output = StringIO()

            with redirect_stdout(output):
                status = cli.run_filter_command(args)

        self.assertEqual(status, 0)
        filter_names.assert_called_once_with(
            Path("custom-artifacts"),
            limit=2,
            low_confidence_log_path=None,
            confidence_threshold=None,
        )
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "a.tsv: Max Mustermann (conf=93.2)",
                "b.tsv: <not found>",
                "Name map written to custom-artifacts/name_map.json.",
            ],
        )

    def test_filter_command_writes_low_confidence_log(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "filter",
                "--artifacts-dir",
                "custom-artifacts",
                "--name-confidence-threshold",
                "90",
                "--low-confidence-log-file",
                "logs/filter-low-confidence.jsonl",
            ]
        )

        with patch("src.todesanzeigen.cli.filter_artifact_names") as filter_names:
            filter_names.return_value = [
                NameFilterResult(Path("custom-artifacts/a.tsv"), "Max Mustermann", 89.9),
                NameFilterResult(Path("custom-artifacts/b.tsv"), "Erika Musterfrau", 90),
                NameFilterResult(Path("custom-artifacts/c.tsv"), "", None),
            ]
            output = StringIO()

            with redirect_stdout(output):
                status = cli.run_filter_command(args)

        self.assertEqual(status, 0)
        filter_names.assert_called_once_with(
            Path("custom-artifacts"),
            limit=None,
            low_confidence_log_path=Path("logs/filter-low-confidence.jsonl"),
            confidence_threshold=90,
        )
        self.assertIn(
            "Low-confidence skip log written to logs/filter-low-confidence.jsonl "
            "(2 entries below 90.0).",
            output.getvalue(),
        )

    def test_ingest_results_command_passes_input_dir(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "ingest",
                "results",
                "--db",
                "state/test.sqlite3",
                "--source",
                "Aichacher Nachrichten",
                "--output-csv",
                "output/result.csv",
                "--method",
                "text_extraction",
                "--provider",
                "qwen",
                "--model",
                "qwen3.6-flash",
                "--candidate-kind",
                "pipeline",
                "--input-dir",
                "input/Aichacher Nachrichten",
            ]
        )

        with patch("src.todesanzeigen.ingest.ingest_results") as ingest_results:
            ingest_results.return_value = SimpleNamespace(
                extraction_outputs=0,
                label_candidates=0,
                run_id="run-1",
            )
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_ingest_command(args)

        self.assertEqual(status, 0)
        ingest_results.assert_called_once_with(
            db_path=Path("state/test.sqlite3"),
            source="Aichacher Nachrichten",
            output_csv=Path("output/result.csv"),
            results_file=None,
            method="text_extraction",
            provider="qwen",
            model="qwen3.6-flash",
            prompt_version="death_notice_v3",
            candidate_kind="pipeline",
            input_dir=Path("input/Aichacher Nachrichten"),
        )
        self.assertIn("Run run-1", output.getvalue())

    def test_extract_command_passes_threshold_and_log_file(self) -> None:
        with patch.dict("src.todesanzeigen.cli.os.environ", {}, clear=True):
            args = cli.build_parser().parse_args(
                [
                    "extract",
                    "--db",
                    "state/test.sqlite3",
                    "--artifacts-dir",
                    "custom-artifacts",
                    "--source",
                    "Testquelle",
                    "--limit",
                    "2",
                    "--name-confidence-threshold",
                    "90",
                    "--log-dir",
                    "custom-logs",
                ]
            )

        with (
            patch("src.todesanzeigen.cli.build_llm_provider", return_value="provider") as build_provider,
            patch(
                "src.todesanzeigen.cli.extract_artifacts_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as extract,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_extract_command(args)

        self.assertEqual(status, 0)
        build_provider.assert_called_once_with("gemini")
        kwargs = extract.call_args.kwargs
        self.assertEqual(
            extract.call_args.args,
            (Path("custom-artifacts"), Path("state/test.sqlite3"), "provider"),
        )
        self.assertEqual(kwargs["input_dir"], Path("input"))
        self.assertEqual(kwargs["source"], "Testquelle")
        self.assertEqual(kwargs["limit"], 2)
        self.assertEqual(kwargs["name_confidence_threshold"], 90)
        self.assertEqual(kwargs["log_file"].parent, Path("custom-logs"))
        self.assertRegex(kwargs["log_file"].name, r"^extract-\d{8}-\d{6}\.txt$")
        self.assertEqual(kwargs["checkpoint_file"], Path("custom-logs/results.jsonl"))
        self.assertEqual(kwargs["resume_from"], Path("custom-logs/results.jsonl"))
        self.assertEqual(kwargs["settings"].concurrency, 1)
        self.assertIn("0 successful outputs recorded", output.getvalue())
        self.assertIn("0 skipped", output.getvalue())
        self.assertIn("Checkpoint written to custom-logs/results.jsonl", output.getvalue())

    def test_extract_command_uses_provider_from_env_default(self) -> None:
        with patch.dict(
            "src.todesanzeigen.cli.os.environ",
            {"TODESANZEIGEN_LLM_PROVIDER": "qwen"},
            clear=True,
        ):
            args = cli.build_parser().parse_args(["extract"])

        with (
            patch("src.todesanzeigen.cli.build_llm_provider", return_value="provider") as build_provider,
            patch(
                "src.todesanzeigen.cli.extract_artifacts_to_db_async",
                new=AsyncMock(return_value=[]),
            ),
        ):
            status = cli.run_extract_command(args)

        self.assertEqual(status, 0)
        build_provider.assert_called_once_with("qwen")

    def test_extract_command_uses_async_path_for_concurrency(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "extract",
                "--provider",
                "qwen",
                "--concurrency",
                "10",
                "--rpm-limit",
                "100",
                "--tpm-limit",
                "200000",
                "--max-retries",
                "3",
            ]
        )

        with (
            patch("src.todesanzeigen.cli.build_llm_provider", return_value="provider") as build_provider,
            patch(
                "src.todesanzeigen.cli.extract_artifacts_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as async_extract,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_extract_command(args)

        self.assertEqual(status, 0)
        build_provider.assert_called_once_with("qwen")
        kwargs = async_extract.call_args.kwargs
        self.assertEqual(async_extract.call_args.args, (Path("artifacts"), Path("state/todesanzeigen.sqlite3"), "provider"))
        self.assertEqual(kwargs["settings"].concurrency, 10)
        self.assertEqual(kwargs["settings"].rpm_limit, 100)
        self.assertEqual(kwargs["settings"].tpm_limit, 200000)
        self.assertEqual(kwargs["settings"].max_retries, 3)
        self.assertIn("0 failed", output.getvalue())

    def test_extract_command_uses_qwen_async_default_limits(self) -> None:
        args = cli.build_parser().parse_args(
            ["extract", "--provider", "qwen", "--concurrency", "2"]
        )

        with (
            patch("src.todesanzeigen.cli.build_llm_provider", return_value="provider"),
            patch(
                "src.todesanzeigen.cli.extract_artifacts_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as async_extract,
        ):
            cli.run_extract_command(args)

        settings = async_extract.call_args.kwargs["settings"]
        self.assertEqual(settings.rpm_limit, 600)
        self.assertEqual(settings.tpm_limit, 500000)

    def test_extract_command_passes_reroute_settings(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "extract",
                "--reroute",
                "--input-dir",
                "custom-input",
                "--reroute-provider",
                "gemini",
                "--reroute-model",
                "gemini-2.5-pro",
                "--reroute-results-file",
                "custom-logs/reroute-results.jsonl",
                "--reroute-concurrency",
                "2",
            ]
        )

        with (
            patch("src.todesanzeigen.cli.build_llm_provider", return_value="text-provider"),
            patch(
                "src.todesanzeigen.cli.build_vision_llm_provider",
                return_value="vision-provider",
            ) as build_vision_provider,
            patch(
                "src.todesanzeigen.cli.extract_artifacts_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as async_extract,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_extract_command(args)

        self.assertEqual(status, 0)
        build_vision_provider.assert_called_once_with("gemini", "gemini-2.5-pro")
        reroute_settings = async_extract.call_args.kwargs["reroute_settings"]
        self.assertEqual(reroute_settings.input_dir, Path("custom-input"))
        self.assertEqual(reroute_settings.provider, "vision-provider")
        self.assertEqual(reroute_settings.results_file, Path("custom-logs/reroute-results.jsonl"))
        self.assertEqual(reroute_settings.concurrency, 2)
        self.assertIn("0 rerouted", output.getvalue())

    def test_extract_command_rejects_invalid_concurrency(self) -> None:
        args = cli.build_parser().parse_args(["extract", "--concurrency", "0"])

        with self.assertRaises(ValueError) as error:
            cli.run_extract_command(args)

        self.assertIn("concurrency must be at least 1", str(error.exception))

    def test_reroute_command_processes_selected_candidates(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "reroute",
                "--db",
                "state/test.sqlite3",
                "--artifacts-dir",
                "custom-artifacts",
                "--input-dir",
                "custom-input",
                "--low-confidence-file",
                "logs/filter-low-confidence.jsonl",
                "--only",
                "a.txt",
                "--limit",
                "1",
                "--provider",
                "qwen",
                "--model",
                "qwen-vl-ocr-latest",
            ]
        )

        with (
            patch(
                "src.todesanzeigen.cli.build_vision_llm_provider",
                return_value="vision-provider",
            ) as build_vision_provider,
            patch("src.todesanzeigen.cli.load_reroute_candidates", return_value=["a", "b"]) as load_candidates,
            patch("src.todesanzeigen.cli.select_reroute_candidates", return_value=["a"]) as select_candidates,
            patch(
                "src.todesanzeigen.cli.reroute_candidates_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as reroute_async,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_reroute_command(args)

        self.assertEqual(status, 0)
        build_vision_provider.assert_called_once_with("qwen", "qwen-vl-ocr-latest")
        load_candidates.assert_called_once_with(
            Path("custom-artifacts"),
            low_confidence_file=Path("logs/filter-low-confidence.jsonl"),
            results_file=None,
            threshold=85.0,
        )
        select_candidates.assert_called_once_with(
            ["a", "b"],
            only=["a.txt"],
            sample_ratio=None,
            sample_seed=0,
            limit=1,
        )
        self.assertEqual(reroute_async.call_args.args, (["a"], Path("state/test.sqlite3"), "vision-provider"))
        kwargs = reroute_async.call_args.kwargs
        self.assertEqual(kwargs["input_dir"], Path("custom-input"))
        self.assertEqual(kwargs["candidate_kind"], "pipeline")
        self.assertIn("Vision reroute complete", output.getvalue())

    def test_vision_extract_command_processes_selected_images(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "vision-extract",
                "--db",
                "state/test.sqlite3",
                "--input-dir",
                "custom-input",
                "--source",
                "Ground Truth",
                "--provider",
                "gemini",
                "--model",
                "gemini-2.5-pro",
                "--limit",
                "5",
                "--sample-ratio",
                "0.2",
                "--sample-seed",
                "42",
                "--only",
                "a.jpg",
                "--log-dir",
                "custom-logs",
                "--results-file",
                "custom-logs/vision-results.jsonl",
                "--concurrency",
                "3",
                "--rpm-limit",
                "100",
                "--tpm-limit",
                "200000",
                "--max-retries",
                "2",
            ]
        )

        with (
            patch(
                "src.todesanzeigen.cli.build_vision_llm_provider",
                return_value="vision-provider",
            ) as build_vision_provider,
            patch(
                "src.todesanzeigen.cli.extract_images_to_db_async",
                new=AsyncMock(return_value=[]),
            ) as vision_extract_async,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_vision_extract_command(args)

        self.assertEqual(status, 0)
        build_vision_provider.assert_called_once_with("gemini", "gemini-2.5-pro")
        self.assertEqual(
            vision_extract_async.call_args.args,
            (Path("custom-input"), Path("state/test.sqlite3"), "vision-provider"),
        )
        kwargs = vision_extract_async.call_args.kwargs
        self.assertEqual(kwargs["source"], "Ground Truth")
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["only"], ["a.jpg"])
        self.assertEqual(kwargs["sample_ratio"], 0.2)
        self.assertEqual(kwargs["sample_seed"], 42)
        self.assertEqual(kwargs["log_file"].parent, Path("custom-logs"))
        self.assertRegex(kwargs["log_file"].name, r"^vision-extract-\d{8}-\d{6}\.txt$")
        self.assertEqual(kwargs["results_file"], Path("custom-logs/vision-results.jsonl"))
        self.assertEqual(kwargs["settings"].concurrency, 3)
        self.assertEqual(kwargs["settings"].rpm_limit, 100)
        self.assertEqual(kwargs["settings"].tpm_limit, 200000)
        self.assertEqual(kwargs["settings"].max_retries, 2)
        self.assertEqual(kwargs["candidate_kind"], "teacher")
        self.assertIn("Vision extraction complete", output.getvalue())
