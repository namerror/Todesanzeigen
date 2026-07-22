from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

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
        filter_names.assert_called_once_with(Path("custom-artifacts"), limit=2)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "a.tsv: Max Mustermann (conf=93.2)",
                "b.tsv: <not found>",
                "Name map written to custom-artifacts/name_map.json.",
            ],
        )

    def test_extract_command_passes_threshold_and_log_file(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "extract",
                "--artifacts-dir",
                "custom-artifacts",
                "--output-file",
                "custom-output/result.csv",
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
            patch("src.todesanzeigen.cli.GeminiSettings.from_env", return_value="settings"),
            patch("src.todesanzeigen.cli.GeminiProvider", return_value="provider"),
            patch("src.todesanzeigen.cli.extract_artifacts_to_csv", return_value=[]) as extract,
        ):
            output = StringIO()
            with redirect_stdout(output):
                status = cli.run_extract_command(args)

        self.assertEqual(status, 0)
        kwargs = extract.call_args.kwargs
        self.assertEqual(
            extract.call_args.args,
            (Path("custom-artifacts"), Path("custom-output/result.csv"), "provider"),
        )
        self.assertEqual(kwargs["source"], "Testquelle")
        self.assertEqual(kwargs["limit"], 2)
        self.assertEqual(kwargs["name_confidence_threshold"], 90)
        self.assertEqual(kwargs["log_file"].parent, Path("custom-logs"))
        self.assertRegex(kwargs["log_file"].name, r"^extract-\d{8}-\d{6}\.txt$")
        self.assertIn("0 rows written", output.getvalue())
        self.assertIn("0 skipped", output.getvalue())
