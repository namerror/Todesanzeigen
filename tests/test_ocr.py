import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

from src.todesanzeigen.ocr import (
    ConfigError,
    OcrOutput,
    TesseractOcrClient,
    TesseractSettings,
    artifact_path_for_image,
    discover_images,
    image_mime_type,
    run_ocr_folder,
    tsv_artifact_path_for_image,
)


class FakeOcrClient:
    writes_tsv = False

    def __init__(self, text: str = "recognized text") -> None:
        self.text = text
        self.calls: list[tuple[Path, str]] = []

    def process_image(self, image_path: Path, mime_type: str) -> OcrOutput:
        self.calls.append((image_path, mime_type))
        return OcrOutput(text=self.text)


class FakeLayoutOcrClient(FakeOcrClient):
    writes_tsv = True

    def __init__(self, text: str = "recognized text", tsv: str = "layout") -> None:
        super().__init__(text)
        self.tsv = tsv

    def process_image(self, image_path: Path, mime_type: str) -> OcrOutput:
        self.calls.append((image_path, mime_type))
        return OcrOutput(text=self.text, tsv=self.tsv)


class OcrTests(TestCase):
    def test_discover_images_only_direct_supported_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            nested = input_dir / "nested"
            nested.mkdir(parents=True)
            (input_dir / "a.png").write_bytes(b"image")
            (input_dir / "b.JPG").write_bytes(b"image")
            (input_dir / "notes.txt").write_text("nope", encoding="utf-8")
            (nested / "c.png").write_bytes(b"image")

            self.assertEqual(
                [path.name for path in discover_images(input_dir)],
                ["a.png", "b.JPG"],
            )

    def test_supported_mime_types(self) -> None:
        self.assertEqual(image_mime_type(Path("scan.jpg")), "image/jpeg")
        self.assertEqual(image_mime_type(Path("scan.JPEG")), "image/jpeg")
        self.assertEqual(image_mime_type(Path("scan.png")), "image/png")
        self.assertEqual(image_mime_type(Path("scan.tif")), "image/tiff")
        self.assertEqual(image_mime_type(Path("scan.webp")), "image/webp")
        self.assertIsNone(image_mime_type(Path("scan.txt")))

    def test_ocr_writes_artifact_even_for_empty_text(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            artifacts_dir = root / "artifacts"
            input_dir.mkdir()
            image = input_dir / "example.png"
            image.write_bytes(b"image")

            results = run_ocr_folder(input_dir, artifacts_dir, FakeOcrClient(""))

            artifact = artifact_path_for_image(image, artifacts_dir)
            self.assertEqual(results[0].status, "processed")
            self.assertTrue(artifact.exists())
            self.assertEqual(artifact.read_text(encoding="utf-8"), "")

    def test_ocr_writes_text_and_tsv_artifacts_for_layout_client(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            artifacts_dir = root / "artifacts"
            input_dir.mkdir()
            image = input_dir / "example.png"
            image.write_bytes(b"image")

            results = run_ocr_folder(
                input_dir,
                artifacts_dir,
                FakeLayoutOcrClient("recognized text", "level\ttext\n1\t"),
            )

            text_artifact = artifact_path_for_image(image, artifacts_dir)
            tsv_artifact = tsv_artifact_path_for_image(image, artifacts_dir)
            self.assertEqual(results[0].status, "processed")
            self.assertEqual(results[0].tsv_artifact_path, tsv_artifact)
            self.assertEqual(text_artifact.read_text(encoding="utf-8"), "recognized text")
            self.assertEqual(tsv_artifact.read_text(encoding="utf-8"), "level\ttext\n1\t")

    def test_existing_artifact_skipped_unless_overwrite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            artifacts_dir = root / "artifacts"
            input_dir.mkdir()
            artifacts_dir.mkdir()
            image = input_dir / "example.png"
            image.write_bytes(b"image")
            artifact_path_for_image(image, artifacts_dir).write_text("old", encoding="utf-8")

            client = FakeOcrClient("new")
            skipped = run_ocr_folder(input_dir, artifacts_dir, client)
            self.assertEqual(skipped[0].status, "skipped")
            self.assertEqual(client.calls, [])

            processed = run_ocr_folder(input_dir, artifacts_dir, client, overwrite=True)
            self.assertEqual(processed[0].status, "processed")
            self.assertEqual(artifact_path_for_image(image, artifacts_dir).read_text(encoding="utf-8"), "new")

    def test_existing_text_artifact_without_tsv_is_reprocessed_for_layout_client(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            artifacts_dir = root / "artifacts"
            input_dir.mkdir()
            artifacts_dir.mkdir()
            image = input_dir / "example.png"
            image.write_bytes(b"image")
            artifact_path_for_image(image, artifacts_dir).write_text("old", encoding="utf-8")

            client = FakeLayoutOcrClient("new text", "new tsv")
            processed = run_ocr_folder(input_dir, artifacts_dir, client)

            self.assertEqual(processed[0].status, "processed")
            self.assertEqual(client.calls, [(image, "image/png")])
            self.assertEqual(artifact_path_for_image(image, artifacts_dir).read_text(encoding="utf-8"), "new text")
            self.assertEqual(tsv_artifact_path_for_image(image, artifacts_dir).read_text(encoding="utf-8"), "new tsv")

    def test_tesseract_client_runs_binary_with_configured_settings_for_text_and_tsv(self) -> None:
        client = TesseractOcrClient(
            TesseractSettings(
                language="deu",
                page_segmentation_mode=3,
                ocr_engine_mode=1,
                tessdata_dir=Path("data"),
                binary="custom-tesseract",
            )
        )

        with patch("src.todesanzeigen.ocr.subprocess.run") as run:
            run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="recognized text", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="level\ttext\n1\t", stderr=""),
            ]

            output = client.process_image(Path("scan.png"), "image/png")

        self.assertEqual(output.text, "recognized text")
        self.assertEqual(output.tsv, "level\ttext\n1\t")
        base_command = [
            "custom-tesseract",
            "scan.png",
            "stdout",
            "--tessdata-dir",
            "data",
            "-l",
            "deu",
            "--oem",
            "1",
            "--psm",
            "3",
        ]
        run.assert_has_calls(
            [
                call(base_command, check=True, capture_output=True, text=True),
                call(
                    base_command + ["-c", "tessedit_create_tsv=1"],
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            ]
        )

    def test_tesseract_client_reports_command_failures(self) -> None:
        client = TesseractOcrClient()

        with patch("src.todesanzeigen.ocr.subprocess.run") as run:
            run.side_effect = subprocess.CalledProcessError(
                1,
                ["tesseract"],
                stderr="Error opening data file deu.traineddata",
            )

            with self.assertRaises(ConfigError) as error:
                client.process_image(Path("scan.png"), "image/png")

        self.assertIn("Tesseract text OCR failed for scan.png", str(error.exception))
        self.assertIn("deu.traineddata", str(error.exception))
