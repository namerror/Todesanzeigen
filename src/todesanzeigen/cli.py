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
    VISION_FAILED_STATUS,
    VISION_PROCESSED_STATUS,
    VisionRerouteSettings,
    extract_artifacts_to_csv_async,
    extract_images_to_csv_async,
    load_reroute_candidates,
    reroute_candidates_to_csv_async,
    select_reroute_candidates,
)
from .llm import LLM_PROVIDERS, VISION_LLM_PROVIDERS, build_llm_provider, build_vision_llm_provider
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
    extract.add_argument("--input-dir", type=Path, default=Path("input"))
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
    extract.add_argument("--reroute", action="store_true")
    extract.add_argument(
        "--reroute-provider",
        choices=VISION_LLM_PROVIDERS,
        default=os.getenv("TODESANZEIGEN_REROUTE_PROVIDER", "qwen"),
    )
    extract.add_argument("--reroute-model", default=os.getenv("TODESANZEIGEN_REROUTE_MODEL"))
    extract.add_argument("--reroute-results-file", type=Path)
    extract.add_argument(
        "--reroute-concurrency",
        type=int,
        default=_env_int("TODESANZEIGEN_REROUTE_CONCURRENCY", 1),
    )

    filter_parser = subparsers.add_parser(
        "filter",
        help="Detect likely deceased names from OCR TSV layout artifacts.",
    )
    filter_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    filter_parser.add_argument("--limit", type=int)
    filter_parser.add_argument(
        "--name-confidence-threshold",
        type=float,
        default=DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    )
    filter_parser.add_argument("--low-confidence-log-file", type=Path)

    reroute = subparsers.add_parser(
        "reroute",
        help="Run direct vision extraction for low-confidence or selected artifacts.",
    )
    reroute.add_argument("--input-dir", type=Path, default=Path("input"))
    reroute.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    reroute.add_argument("--output-file", type=Path, default=Path("output/rerouted.csv"))
    reroute.add_argument("--merge-output-file", type=Path)
    reroute.add_argument("--source", default=os.getenv("TODESANZEIGEN_SOURCE", ""))
    reroute.add_argument(
        "--provider",
        choices=VISION_LLM_PROVIDERS,
        default=os.getenv("TODESANZEIGEN_REROUTE_PROVIDER", "qwen"),
    )
    reroute.add_argument("--model", default=os.getenv("TODESANZEIGEN_REROUTE_MODEL"))
    reroute.add_argument("--low-confidence-file", type=Path)
    reroute.add_argument("--from-results", type=Path)
    reroute.add_argument(
        "--name-confidence-threshold",
        type=float,
        default=DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    )
    reroute.add_argument("--limit", type=int)
    reroute.add_argument("--sample-ratio", type=float)
    reroute.add_argument("--sample-seed", type=int, default=0)
    reroute.add_argument("--only", action="append")
    reroute.add_argument("--log-dir", type=Path, default=Path("logs"))
    reroute.add_argument("--results-file", type=Path, default=Path("logs/reroute-results.jsonl"))
    reroute.add_argument(
        "--concurrency",
        type=int,
        default=_env_int("TODESANZEIGEN_REROUTE_CONCURRENCY", 1),
    )
    reroute.add_argument("--rpm-limit", type=int, default=_env_int("TODESANZEIGEN_LLM_RPM_LIMIT"))
    reroute.add_argument("--tpm-limit", type=int, default=_env_int("TODESANZEIGEN_LLM_TPM_LIMIT"))
    reroute.add_argument(
        "--max-retries",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_MAX_RETRIES", 5),
    )

    vision_extract = subparsers.add_parser(
        "vision-extract",
        help="Run image-only vision extraction for source images.",
    )
    vision_extract.add_argument("--input-dir", type=Path, default=Path("input"))
    vision_extract.add_argument("--output-file", type=Path, default=Path("output/vision-result.csv"))
    vision_extract.add_argument("--source", default=os.getenv("TODESANZEIGEN_SOURCE", ""))
    vision_extract.add_argument(
        "--provider",
        choices=VISION_LLM_PROVIDERS,
        default=os.getenv(
            "TODESANZEIGEN_VISION_PROVIDER",
            os.getenv("TODESANZEIGEN_REROUTE_PROVIDER", "qwen"),
        ),
    )
    vision_extract.add_argument(
        "--model",
        default=os.getenv(
            "TODESANZEIGEN_VISION_MODEL",
            os.getenv("TODESANZEIGEN_REROUTE_MODEL"),
        ),
    )
    vision_extract.add_argument("--limit", type=int)
    vision_extract.add_argument("--sample-ratio", type=float)
    vision_extract.add_argument("--sample-seed", type=int, default=0)
    vision_extract.add_argument("--only", action="append")
    vision_extract.add_argument("--log-dir", type=Path, default=Path("logs"))
    vision_extract.add_argument("--results-file", type=Path)
    vision_extract.add_argument(
        "--concurrency",
        type=int,
        default=_env_int(
            "TODESANZEIGEN_VISION_CONCURRENCY",
            _env_int("TODESANZEIGEN_REROUTE_CONCURRENCY", 1),
        ),
    )
    vision_extract.add_argument("--rpm-limit", type=int, default=_env_int("TODESANZEIGEN_LLM_RPM_LIMIT"))
    vision_extract.add_argument("--tpm-limit", type=int, default=_env_int("TODESANZEIGEN_LLM_TPM_LIMIT"))
    vision_extract.add_argument(
        "--max-retries",
        type=int,
        default=_env_int("TODESANZEIGEN_LLM_MAX_RETRIES", 5),
    )

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
    if args.reroute_concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")

    provider = build_llm_provider(args.provider)
    reroute_settings = None
    reroute_results_file = args.reroute_results_file or args.log_dir / "reroute-results.jsonl"
    if args.reroute:
        reroute_provider = build_vision_llm_provider(args.reroute_provider, args.reroute_model)
        reroute_settings = VisionRerouteSettings(
            input_dir=args.input_dir,
            provider=reroute_provider,
            results_file=reroute_results_file,
            concurrency=args.reroute_concurrency,
        )

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
            reroute_settings=reroute_settings,
        )
    )
    processed = sum(1 for result in results if result.status == "processed")
    rerouted = sum(1 for result in results if result.status == "rerouted_processed")
    skipped = sum(1 for result in results if result.status == "skipped_low_confidence")
    failed = sum(1 for result in results if result.status in {"failed", "rerouted_failed"})
    rows_written = processed + rerouted
    reroute_note = f" Reroute checkpoint written to {reroute_results_file}." if args.reroute else ""
    print(
        f"Extraction complete: {rows_written} rows written to {args.output_file}; "
        f"{skipped} skipped; {rerouted} rerouted; {failed} failed. "
        f"Log written to {log_file}. Checkpoint written to {checkpoint_file}."
        f"{reroute_note}"
    )
    return 0


def run_reroute_command(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    if args.rpm_limit is not None and args.rpm_limit < 0:
        raise ValueError("LLM RPM limit must be at least 0.")
    if args.tpm_limit is not None and args.tpm_limit < 0:
        raise ValueError("LLM TPM limit must be at least 0.")
    if args.low_confidence_file is not None and args.from_results is not None:
        raise ValueError("Use either --low-confidence-file or --from-results, not both.")

    provider = build_vision_llm_provider(args.provider, args.model)
    low_confidence_file = args.low_confidence_file
    if low_confidence_file is None and args.from_results is None:
        default_low_confidence_file = args.log_dir / "filter-low-confidence.jsonl"
        low_confidence_file = (
            default_low_confidence_file if default_low_confidence_file.exists() else None
        )
    candidates = load_reroute_candidates(
        args.artifacts_dir,
        low_confidence_file=low_confidence_file,
        results_file=args.from_results,
        threshold=args.name_confidence_threshold,
    )
    selected_candidates = select_reroute_candidates(
        candidates,
        only=args.only,
        sample_ratio=args.sample_ratio,
        sample_seed=args.sample_seed,
        limit=args.limit,
    )
    log_file = args.log_dir / f"reroute-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    results = asyncio.run(
        reroute_candidates_to_csv_async(
            selected_candidates,
            args.output_file,
            provider,
            input_dir=args.input_dir,
            source=args.source,
            log_file=log_file,
            results_file=args.results_file,
            merge_output_file=args.merge_output_file,
            settings=AsyncExtractionSettings(
                concurrency=args.concurrency,
                rpm_limit=args.rpm_limit,
                tpm_limit=args.tpm_limit,
                max_retries=args.max_retries,
            ),
            concurrency=args.concurrency,
        )
    )
    rerouted = sum(1 for result in results if result.status == "rerouted_processed")
    failed = sum(1 for result in results if result.status == "rerouted_failed")
    merge_note = f" Merged into {args.merge_output_file}." if args.merge_output_file else ""
    print(
        f"Vision reroute complete: {rerouted} rows written to {args.output_file}; "
        f"{failed} failed. Log written to {log_file}. "
        f"Results written to {args.results_file}.{merge_note}"
    )
    return 0


def run_vision_extract_command(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("Vision extraction concurrency must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    if args.rpm_limit is not None and args.rpm_limit < 0:
        raise ValueError("LLM RPM limit must be at least 0.")
    if args.tpm_limit is not None and args.tpm_limit < 0:
        raise ValueError("LLM TPM limit must be at least 0.")

    provider = build_vision_llm_provider(args.provider, args.model)
    log_file = args.log_dir / f"vision-extract-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    results_file = args.results_file or args.log_dir / "vision-results.jsonl"
    rpm_limit = args.rpm_limit
    tpm_limit = args.tpm_limit
    if args.provider == "qwen":
        rpm_limit = 600 if rpm_limit is None else rpm_limit
        tpm_limit = 500000 if tpm_limit is None else tpm_limit

    results = asyncio.run(
        extract_images_to_csv_async(
            args.input_dir,
            args.output_file,
            provider,
            source=args.source,
            limit=args.limit,
            only=args.only,
            sample_ratio=args.sample_ratio,
            sample_seed=args.sample_seed,
            log_file=log_file,
            results_file=results_file,
            settings=AsyncExtractionSettings(
                concurrency=args.concurrency,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                max_retries=args.max_retries,
            ),
        )
    )
    processed = sum(1 for result in results if result.status == VISION_PROCESSED_STATUS)
    failed = sum(1 for result in results if result.status == VISION_FAILED_STATUS)
    print(
        f"Vision extraction complete: {processed} rows written to {args.output_file}; "
        f"{failed} failed. Log written to {log_file}. Results written to {results_file}."
    )
    return 0


def run_filter_command(args: argparse.Namespace) -> int:
    results = filter_artifact_names(
        args.artifacts_dir,
        limit=args.limit,
        low_confidence_log_path=args.low_confidence_log_file,
        confidence_threshold=(
            args.name_confidence_threshold if args.low_confidence_log_file is not None else None
        ),
    )
    for result in results:
        if not result.name:
            print(f"{result.artifact_path.name}: <not found>")
            continue
        confidence = (
            f" (conf={result.confidence:.1f})" if result.confidence is not None else ""
        )
        print(f"{result.artifact_path.name}: {result.name}{confidence}")
    print(f"Name map written to {name_map_artifact_path(args.artifacts_dir)}.")
    if args.low_confidence_log_file is not None:
        skipped = sum(
            1
            for result in results
            if result.confidence is None or result.confidence < args.name_confidence_threshold
        )
        print(
            f"Low-confidence skip log written to {args.low_confidence_log_file} "
            f"({skipped} entries below {args.name_confidence_threshold:.1f})."
        )
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
        if args.command == "reroute":
            return run_reroute_command(args)
        if args.command == "vision-extract":
            return run_vision_extract_command(args)
    except (ConfigError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
