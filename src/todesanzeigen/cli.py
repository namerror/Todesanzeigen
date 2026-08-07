from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from .extract import (
    DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    AsyncExtractionSettings,
    extract_artifacts_to_csv_async,
)
from .llm import LLM_PROVIDERS, build_llm_provider
from .ocr import (
    ConfigError,
    DocumentAiOcrClient,
    DocumentAiSettings,
    TesseractOcrClient,
    TesseractSettings,
    run_ocr_folder,
)
from .ocr_filtering import filter_artifact_names, name_map_artifact_path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


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
    ocr.add_argument("--tesseract-lang", default="deu")
    ocr.add_argument("--tesseract-psm", type=int, default=3)
    ocr.add_argument("--tesseract-oem", type=int, default=1)
    ocr.add_argument("--tesseract-tessdata-dir", type=Path, default=Path("data"))
    ocr.add_argument("--tesseract-bin", default="tesseract")
    ocr.add_argument("--overwrite", action="store_true")
    ocr.add_argument("--limit", type=int)

    extract = subparsers.add_parser("extract", help="Parse OCR artifacts and write output/result.csv.")
    extract.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    extract.add_argument("--output-file", type=Path, default=Path("output/result.csv"))
    extract.add_argument(
        "--provider",
        choices=LLM_PROVIDERS,
        default=os.getenv("TODESANZEIGEN_LLM_PROVIDER", "gemini"),
    )
    extract.add_argument("--source", default=os.getenv("TODESANZEIGEN_SOURCE", ""))
    extract.add_argument("--limit", type=int)
    extract.add_argument(
        "--name-confidence-threshold",
        type=float,
        default=DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    )
    extract.add_argument("--log-dir", type=Path, default=Path("logs"))
    extract.add_argument(
        "--concurrency",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_CONCURRENCY", 1),
    )
    extract.add_argument(
        "--rpm-limit",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_RPM_LIMIT"),
    )
    extract.add_argument(
        "--tpm-limit",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_TPM_LIMIT"),
    )
    extract.add_argument(
        "--max-retries",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_MAX_RETRIES", 5),
    )
    extract.add_argument("--resume-from", type=Path)

    filter_parser = subparsers.add_parser(
        "filter",
        help="Detect likely deceased names from OCR TSV layout artifacts.",
    )
    filter_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    filter_parser.add_argument("--limit", type=int)

    return parser


def run_ocr_command(args: argparse.Namespace) -> int:
    if args.engine == "tesseract":
        client = TesseractOcrClient(
            TesseractSettings(
                language=args.tesseract_lang,
                page_segmentation_mode=args.tesseract_psm,
                ocr_engine_mode=args.tesseract_oem,
                tessdata_dir=args.tesseract_tessdata_dir,
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
            layout = (
                f", {result.tsv_artifact_path} ({result.tsv_length} tsv chars)"
                if result.tsv_artifact_path
                else ""
            )
            print(
                f"processed {result.image_path} -> "
                f"{result.artifact_path} ({result.text_length} chars){layout}"
            )
        elif result.status == "skipped":
            layout = f", {result.tsv_artifact_path}" if result.tsv_artifact_path else ""
            print(f"skipped {result.image_path} -> {result.artifact_path}{layout}")
    return 0


def run_extract_command(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("LLM extraction concurrency must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    if args.rpm_limit is not None and args.rpm_limit < 0:
        raise ValueError("LLM RPM limit must be at least 0.")
    if args.tpm_limit is not None and args.tpm_limit < 0:
        raise ValueError("LLM TPM limit must be at least 0.")

    provider = build_llm_provider(args.provider)
    log_file = args.log_dir / f"extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    checkpoint_file = args.resume_from or args.log_dir / "results.jsonl"
    rpm_limit = args.rpm_limit
    tpm_limit = args.tpm_limit
    if args.provider == "qwen":
        rpm_limit = 600 if rpm_limit is None else rpm_limit
        tpm_limit = 500000 if tpm_limit is None else tpm_limit
    results = asyncio.run(
        extract_artifacts_to_csv_async(
            args.artifacts_dir,
            args.output_file,
            provider,
            source=args.source,
            limit=args.limit,
            name_confidence_threshold=args.name_confidence_threshold,
            log_file=log_file,
            checkpoint_file=checkpoint_file,
            resume_from=checkpoint_file,
            settings=AsyncExtractionSettings(
                concurrency=args.concurrency,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                max_retries=args.max_retries,
            ),
        )
    )
    processed = sum(1 for result in results if result.status == "processed")
    skipped = sum(1 for result in results if result.status == "skipped_low_confidence")
    failed = sum(1 for result in results if result.status == "failed")
    print(
        f"Extraction complete: {processed} rows written to {args.output_file}; "
        f"{skipped} skipped; {failed} failed. "
        f"Log written to {log_file}. Checkpoint written to {checkpoint_file}."
    )
    return 0


def run_filter_command(args: argparse.Namespace) -> int:
    results = filter_artifact_names(args.artifacts_dir, limit=args.limit)
    for result in results:
        if not result.name:
            print(f"{result.artifact_path.name}: <not found>")
            continue
        confidence = (
            f" (conf={result.confidence:.1f})" if result.confidence is not None else ""
        )
        print(f"{result.artifact_path.name}: {result.name}{confidence}")
    print(f"Name map written to {name_map_artifact_path(args.artifacts_dir)}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.command == "ocr":
            return run_ocr_command(args)
        if args.command == "extract":
            return run_extract_command(args)
        if args.command == "filter":
            return run_filter_command(args)
    except (ConfigError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
