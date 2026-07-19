from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SUPPORTED_IMAGE_MIME_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing."""


@dataclass(frozen=True)
class DocumentAiSettings:
    project_id: str
    location: str
    processor_id: str
    credentials_path: str

    @classmethod
    def from_env(cls) -> "DocumentAiSettings":
        required = {
            "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            "GOOGLE_CLOUD_PROJECT_ID": os.getenv("GOOGLE_CLOUD_PROJECT_ID", ""),
            "DOCUMENT_AI_LOCATION": os.getenv("DOCUMENT_AI_LOCATION", ""),
            "DOCUMENT_AI_PROCESSOR_ID": os.getenv("DOCUMENT_AI_PROCESSOR_ID", ""),
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            names = ", ".join(missing)
            raise ConfigError(f"Missing required Google Document AI environment variables: {names}")

        return cls(
            project_id=required["GOOGLE_CLOUD_PROJECT_ID"],
            location=required["DOCUMENT_AI_LOCATION"],
            processor_id=required["DOCUMENT_AI_PROCESSOR_ID"],
            credentials_path=required["GOOGLE_APPLICATION_CREDENTIALS"],
        )


class OcrClient(Protocol):
    writes_tsv: bool

    def process_image(self, image_path: Path, mime_type: str) -> "OcrOutput":
        """Return OCR artifacts for one image."""


@dataclass(frozen=True)
class OcrOutput:
    text: str
    tsv: str | None = None


class DocumentAiOcrClient:
    writes_tsv = False

    def __init__(self, settings: DocumentAiSettings) -> None:
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai_v1

        self._documentai = documentai_v1
        endpoint = f"{settings.location}-documentai.googleapis.com"
        self._client = documentai_v1.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )
        self._processor_name = self._client.processor_path(
            settings.project_id,
            settings.location,
            settings.processor_id,
        )

    def process_image(self, image_path: Path, mime_type: str) -> OcrOutput:
        image_content = image_path.read_bytes()
        raw_document = self._documentai.RawDocument(
            content=image_content,
            mime_type=mime_type,
        )
        request = self._documentai.ProcessRequest(
            name=self._processor_name,
            raw_document=raw_document,
        )
        result = self._client.process_document(request=request)
        return OcrOutput(text=result.document.text or "")


@dataclass(frozen=True)
class TesseractSettings:
    language: str = "deu"
    page_segmentation_mode: int = 3
    ocr_engine_mode: int = 1
    tessdata_dir: Path | None = Path("data")
    binary: str = "tesseract"


class TesseractOcrClient:
    writes_tsv = True

    def __init__(self, settings: TesseractSettings | None = None) -> None:
        self._settings = settings or TesseractSettings()

    def process_image(self, image_path: Path, mime_type: str) -> OcrOutput:
        del mime_type
        text = self._run_tesseract(image_path, "text")
        tsv = self._run_tesseract(image_path, "tsv")
        return OcrOutput(text=text, tsv=tsv)

    def _base_command(self, image_path: Path) -> list[str]:
        command = [
            self._settings.binary,
            str(image_path),
            "stdout",
        ]
        if self._settings.tessdata_dir is not None:
            command.extend(["--tessdata-dir", str(self._settings.tessdata_dir)])
        command.extend(
            [
                "-l",
                self._settings.language,
                "--oem",
                str(self._settings.ocr_engine_mode),
                "--psm",
                str(self._settings.page_segmentation_mode),
            ]
        )
        return command

    def _run_tesseract(self, image_path: Path, output_format: str) -> str:
        command = self._base_command(image_path)
        if output_format == "tsv":
            command.extend(["-c", "tessedit_create_tsv=1"])
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ConfigError(
                f"Tesseract binary not found: {self._settings.binary}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = f"Tesseract {output_format} OCR failed for {image_path}"
            if details:
                message = f"{message}: {details}"
            raise ConfigError(message) from exc

        return result.stdout or ""


@dataclass(frozen=True)
class OcrRunResult:
    image_path: Path
    artifact_path: Path
    status: str
    text_length: int = 0
    tsv_artifact_path: Path | None = None
    tsv_length: int = 0


def image_mime_type(path: Path) -> str | None:
    return SUPPORTED_IMAGE_MIME_TYPES.get(path.suffix.lower())


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and image_mime_type(path)
    )


def artifact_path_for_image(image_path: Path, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"{image_path.stem}.txt"


def tsv_artifact_path_for_image(image_path: Path, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"{image_path.stem}.tsv"


def run_ocr_folder(
    input_dir: Path,
    artifacts_dir: Path,
    client: OcrClient,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[OcrRunResult]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(input_dir)
    if limit is not None:
        images = images[:limit]

    results: list[OcrRunResult] = []
    for image_path in images:
        artifact_path = artifact_path_for_image(image_path, artifacts_dir)
        tsv_artifact_path = tsv_artifact_path_for_image(image_path, artifacts_dir)
        expects_tsv = bool(getattr(client, "writes_tsv", False))
        artifact_set_exists = artifact_path.exists() and (
            not expects_tsv or tsv_artifact_path.exists()
        )
        if artifact_set_exists and not overwrite:
            results.append(
                OcrRunResult(
                    image_path,
                    artifact_path,
                    "skipped",
                    tsv_artifact_path=tsv_artifact_path if expects_tsv else None,
                )
            )
            continue

        mime_type = image_mime_type(image_path)
        if mime_type is None:
            results.append(OcrRunResult(image_path, artifact_path, "unsupported"))
            continue

        output = client.process_image(image_path, mime_type)
        artifact_path.write_text(output.text, encoding="utf-8")
        if output.tsv is not None:
            tsv_artifact_path.write_text(output.tsv, encoding="utf-8")
        results.append(
            OcrRunResult(
                image_path,
                artifact_path,
                "processed",
                len(output.text),
                tsv_artifact_path if output.tsv is not None else None,
                len(output.tsv or ""),
            )
        )

    return results
