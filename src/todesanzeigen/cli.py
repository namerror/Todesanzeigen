from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .extract import extract_artifacts_to_csv
from .llm import GeminiProvider, GeminiSettings
from .ocr import (
    ConfigError,
    DocumentAiOcrClient,
    DocumentAiSettings,
    TesseractOcrClient,
    TesseractSettings,
    run_ocr_folder,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="todesanzeigen",
        description="OCR images and extract structured Todesanzeigen CSV rows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr = subparsers.add_parser("ocr", help="Run local Tesseract OCR for images in input/.")
    ocr.add_argument("--input-dir", type=Path, default=Path("input"))
    ocr.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    ocr.add_argument("--engine", choices=["tesseract", "documentai"], default="tesseract")
    ocr.add_argument("--tesseract-lang", default="deu+eng")
    ocr.add_argument("--tesseract-psm", type=int, default=6)
    ocr.add_argument("--tesseract-bin", default="tesseract")
    ocr.add_argument("--overwrite", action="store_true")
    ocr.add_argument("--limit", type=int)

    extract = subparsers.add_parser("extract", help="Parse OCR artifacts and write output/result.csv.")
    extract.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    extract.add_argument("--output-file", type=Path, default=Path("output/result.csv"))
    extract.add_argument("--provider", choices=["gemini"], default="gemini")
    extract.add_argument("--source", default=os.getenv("TODESANZEIGEN_SOURCE", ""))
    extract.add_argument("--limit", type=int)

    return parser


def run_ocr_command(args: argparse.Namespace) -> int:
    if args.engine == "tesseract":
        client = TesseractOcrClient(
            TesseractSettings(
                language=args.tesseract_lang,
                page_segmentation_mode=args.tesseract_psm,
                binary=args.tesseract_bin,
            )
        )
    elif args.engine == "documentai":
        if os.getenv("TODESANZEIGEN_ALLOW_GCP") != "1":
            raise ConfigError(
                "Google Document AI OCR is disabled to avoid accidental billing. "
                "Set TODESANZEIGEN_ALLOW_GCP=1 and pass --engine documentai to use it."
            )
        settings = DocumentAiSettings.from_env()
        client = DocumentAiOcrClient(settings)
    else:
        raise ConfigError(f"Unsupported OCR engine: {args.engine}")

    results = run_ocr_folder(
        args.input_dir,
        args.artifacts_dir,
        client,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    processed = sum(1 for result in results if result.status == "processed")
    skipped = sum(1 for result in results if result.status == "skipped")
    print(f"OCR complete: {processed} processed, {skipped} skipped.")
    for result in results:
        if result.status == "processed":
            print(f"processed {result.image_path} -> {result.artifact_path} ({result.text_length} chars)")
        elif result.status == "skipped":
            print(f"skipped {result.image_path} -> {result.artifact_path}")
    return 0


def run_extract_command(args: argparse.Namespace) -> int:
    if args.provider != "gemini":
        raise ConfigError(f"Unsupported LLM provider: {args.provider}")

    provider = GeminiProvider(GeminiSettings.from_env())
    results = extract_artifacts_to_csv(
        args.artifacts_dir,
        args.output_file,
        provider,
        source=args.source,
        limit=args.limit,
    )
    print(f"Extraction complete: {len(results)} rows written to {args.output_file}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ocr":
            return run_ocr_command(args)
        if args.command == "extract":
            return run_extract_command(args)
    except (ConfigError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
