from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.ocr import (
    artifact_path_for_image,
    discover_images,
    image_mime_type,
    run_ocr_folder,
)


class FakeOcrClient:
    def __init__(self, text: str = "recognized text") -> None:
        self.text = text
        self.calls: list[tuple[Path, str]] = []

    def process_image(self, image_path: Path, mime_type: str) -> str:
        self.calls.append((image_path, mime_type))
        return self.text


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
