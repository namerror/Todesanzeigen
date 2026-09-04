from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .extract import (
    DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    AsyncExtractionSettings,
    CACHED_EXISTING_STATUS,
    VISION_FAILED_STATUS,
    VISION_PROCESSED_STATUS,
    VisionRerouteSettings,
    extract_artifacts_to_db_async,
    extract_images_to_db_async,
    load_reroute_candidates,
    reroute_candidates_to_db_async,
    select_reroute_candidates,
)
from .features import DEFAULT_FEATURE_SET
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
from .storage import DEFAULT_DB_PATH, DEFAULT_LABEL_SET
from .methods import CURRENT_PROMPT_VERSION
from .variants import DEFAULT_VARIANTS_CONFIG_PATH


CANDIDATE_KINDS = ("teacher", "pipeline", "manual_seed")


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

    extract = subparsers.add_parser("extract", help="Parse OCR artifacts and record DB outputs.")
    extract.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    extract.add_argument("--input-dir", type=Path, default=Path("input"))
    extract.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
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
    extract.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when a successful DB result already exists for the method slot.",
    )
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
    extract.add_argument(
        "--candidate-kind",
        choices=CANDIDATE_KINDS,
        default="pipeline",
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
    reroute.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    reroute.add_argument("--input-dir", type=Path, default=Path("input"))
    reroute.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
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
        "--force",
        action="store_true",
        help="Re-run even when a successful DB VLM result already exists.",
    )
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
    reroute.add_argument(
        "--candidate-kind",
        choices=CANDIDATE_KINDS,
        default="pipeline",
    )

    vision_extract = subparsers.add_parser(
        "vision-extract",
        help="Run image-only vision extraction for source images.",
    )
    vision_extract.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    vision_extract.add_argument("--input-dir", type=Path, default=Path("input"))
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
        "--force",
        action="store_true",
        help="Re-run even when a successful DB VLM result already exists.",
    )
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
    vision_extract.add_argument(
        "--candidate-kind",
        choices=CANDIDATE_KINDS,
        default="teacher",
    )

    db = subparsers.add_parser("db", help="Manage the local SQLite project database.")
    db_subparsers = db.add_subparsers(dest="db_command", required=True)
    db_init = db_subparsers.add_parser("init", help="Create or migrate the SQLite database.")
    db_init.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    db_normalize = db_subparsers.add_parser(
        "normalize-fields",
        help="Normalize stored locations and dates, with dry-run as the default.",
    )
    db_normalize.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    db_normalize.add_argument(
        "--apply",
        action="store_true",
        help="Back up the database and apply the reported field changes.",
    )
    db_normalize.add_argument("--backup-file", type=Path)

    ingest = subparsers.add_parser("ingest", help="Import local data and generated outputs into SQLite.")
    ingest_subparsers = ingest.add_subparsers(dest="ingest_command", required=True)
    ingest_source = ingest_subparsers.add_parser("source", help="Import source images and OCR artifacts.")
    ingest_source.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ingest_source.add_argument("--source", required=True)
    ingest_source.add_argument("--input-dir", type=Path, default=Path("input"))
    ingest_source.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    ingest_source.add_argument("--layout-family", default="")
    ingest_source.add_argument("--limit", type=int)
    ingest_results = ingest_subparsers.add_parser("results", help="Import CSV/JSONL extraction outputs.")
    ingest_results.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ingest_results.add_argument("--source", default="")
    ingest_results.add_argument("--output-csv", type=Path)
    ingest_results.add_argument("--results-file", type=Path)
    ingest_results.add_argument("--input-dir", type=Path)
    ingest_results.add_argument("--method", default="csv_import")
    ingest_results.add_argument("--provider", default="")
    ingest_results.add_argument("--model", default="")
    ingest_results.add_argument("--prompt-version", default=CURRENT_PROMPT_VERSION)
    ingest_results.add_argument(
        "--candidate-kind",
        choices=["teacher", "pipeline", "manual_seed"],
        default="teacher",
    )

    dataset = subparsers.add_parser("dataset", help="Create and export benchmark dataset splits.")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_split = dataset_subparsers.add_parser("split", help="Create a deterministic source-year split.")
    dataset_split.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    dataset_split.add_argument("--name", required=True)
    dataset_split.add_argument("--strategy", choices=["source-year"], default="source-year")
    dataset_split.add_argument("--train-ratio", type=float, default=0.7)
    dataset_split.add_argument("--validation-ratio", type=float, default=0.15)
    dataset_split.add_argument("--test-ratio", type=float, default=0.15)
    dataset_split.add_argument("--seed", type=int, default=0)
    dataset_export = dataset_subparsers.add_parser("export", help="Export a split as JSONL.")
    dataset_export.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    dataset_export.add_argument("--split", required=True)
    dataset_export.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    dataset_export.add_argument("--output-file", type=Path, required=True)
    dataset_export.add_argument("--format", choices=["jsonl"], default="jsonl")
    dataset_export_router = dataset_subparsers.add_parser(
        "export-router",
        help="Export feature/label rows for failure prediction.",
    )
    dataset_export_router.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    dataset_export_router.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    dataset_export_router.add_argument("--variant", required=True)
    dataset_export_router.add_argument(
        "--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH
    )
    dataset_export_router.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    dataset_export_router.add_argument("--output-file", type=Path, required=True)

    features = subparsers.add_parser("features", help="Build and manage ML feature snapshots.")
    features_subparsers = features.add_subparsers(dest="features_command", required=True)
    features_build = features_subparsers.add_parser("build", help="Build router feature snapshots.")
    features_build.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    features_build.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)

    export = subparsers.add_parser("export", help="Export operational outputs from SQLite.")
    export_subparsers = export.add_subparsers(dest="export_command", required=True)
    export_csv = export_subparsers.add_parser("csv", help="Export final prioritized CSV rows.")
    export_csv.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    export_csv.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    export_csv.add_argument("--output-file", type=Path, default=Path("output/result.csv"))
    export_csv.add_argument("--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH)

    eval_parser = subparsers.add_parser("eval", help="Evaluate extraction outputs against ground truth labels.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subparsers.add_parser("run", help="Run field-level extraction evaluation.")
    eval_run.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    eval_run.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    eval_run.add_argument("--variant", required=True)
    eval_run.add_argument("--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH)
    eval_run.add_argument("--split", default="")

    router = subparsers.add_parser("router", help="Train and apply learned model-routing policies.")
    router_subparsers = router.add_subparsers(dest="router_command", required=True)
    router_train = router_subparsers.add_parser(
        "train",
        help="Train the OCR+LLM failure predictor from SQLite feature snapshots and labels.",
    )
    router_train.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    router_train.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    router_train.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    router_train.add_argument("--split", default="")
    router_train.add_argument("--model-dir", type=Path, required=True)
    router_train.add_argument("--target-f1-threshold", type=float, default=0.95)
    router_train.add_argument("--validation-ratio", type=float, default=0.2)
    router_train.add_argument("--seed", type=int, default=0)
    router_train.add_argument("--cheap-cost", type=float, default=1.0)
    router_train.add_argument("--vlm-cost", type=float, default=10.0)
    router_train.add_argument("--lambda-cost", type=float, default=0.01)
    router_train.add_argument("--min-train-rows", type=int, default=4)
    router_train.add_argument("--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH)
    router_manifest = router_subparsers.add_parser(
        "manifest",
        help="Write a document-to-route manifest from a trained router model.",
    )
    router_manifest.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    router_manifest.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    router_manifest.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    router_manifest.add_argument("--model-dir", type=Path, required=True)
    router_manifest.add_argument("--output-file", type=Path, required=True)
    router_manifest.add_argument("--threshold", type=float)
    router_manifest.add_argument("--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH)

    review = subparsers.add_parser("review", help="Review and approve label candidates.")
    review_subparsers = review.add_subparsers(dest="review_command", required=True)
    review_serve = review_subparsers.add_parser("serve", help="Start the local review web UI.")
    review_serve.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    review_serve.add_argument("--label-set", default=DEFAULT_LABEL_SET)
    review_serve.add_argument("--reviewer", default="")
    review_serve.add_argument("--host", default="127.0.0.1")
    review_serve.add_argument("--port", type=int, default=8000)
    review_serve.add_argument("--variants-config", type=Path, default=DEFAULT_VARIANTS_CONFIG_PATH)

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
        extract_artifacts_to_db_async(
            args.artifacts_dir,
            args.db,
            provider,
            input_dir=args.input_dir,
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
            candidate_kind=args.candidate_kind,
            force=args.force,
        )
    )
    processed = sum(1 for result in results if result.status == "processed")
    rerouted = sum(1 for result in results if result.status == "rerouted_processed")
    cached = sum(1 for result in results if result.status == CACHED_EXISTING_STATUS)
    skipped = sum(1 for result in results if result.status == "skipped_low_confidence")
    failed = sum(1 for result in results if result.status in {"failed", "rerouted_failed"})
    rows_recorded = processed + rerouted
    reroute_note = f" Reroute checkpoint written to {reroute_results_file}." if args.reroute else ""
    print(
        f"Extraction complete: {rows_recorded} successful outputs recorded in {args.db}; "
        f"{cached} cached; {skipped} skipped; {rerouted} rerouted; {failed} failed. "
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
        reroute_candidates_to_db_async(
            selected_candidates,
            args.db,
            provider,
            input_dir=args.input_dir,
            source=args.source,
            log_file=log_file,
            results_file=args.results_file,
            settings=AsyncExtractionSettings(
                concurrency=args.concurrency,
                rpm_limit=args.rpm_limit,
                tpm_limit=args.tpm_limit,
                max_retries=args.max_retries,
            ),
            concurrency=args.concurrency,
            candidate_kind=args.candidate_kind,
            force=args.force,
        )
    )
    rerouted = sum(1 for result in results if result.status == "rerouted_processed")
    cached = sum(1 for result in results if result.status == CACHED_EXISTING_STATUS)
    failed = sum(1 for result in results if result.status == "rerouted_failed")
    print(
        f"Vision reroute complete: {rerouted} successful outputs recorded in {args.db}; "
        f"{cached} cached; {failed} failed. Log written to {log_file}. "
        f"Results written to {args.results_file}."
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
        extract_images_to_db_async(
            args.input_dir,
            args.db,
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
            candidate_kind=args.candidate_kind,
            force=args.force,
        )
    )
    processed = sum(1 for result in results if result.status == VISION_PROCESSED_STATUS)
    cached = sum(1 for result in results if result.status == CACHED_EXISTING_STATUS)
    failed = sum(1 for result in results if result.status == VISION_FAILED_STATUS)
    print(
        f"Vision extraction complete: {processed} successful outputs recorded in {args.db}; "
        f"{cached} cached; {failed} failed. Log written to {log_file}. "
        f"Results written to {results_file}."
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


def run_db_command(args: argparse.Namespace) -> int:
    if args.db_command == "init":
        from .storage import apply_migrations

        applied = apply_migrations(args.db)
        applied_note = ", ".join(applied) if applied else "no new migrations"
        print(f"Database ready at {args.db} ({applied_note}).")
        return 0
    if args.db_command == "normalize-fields":
        from .maintenance import normalize_database_fields

        summary = normalize_database_fields(
            args.db,
            apply=args.apply,
            backup_file=args.backup_file,
        )
        mode = "Applied" if summary.applied else "Dry run"
        table_parts = ", ".join(
            f"{table}={count}" for table, count in summary.table_changes.items()
        )
        print(
            f"{mode}: scanned={summary.scanned_rows}; changed={summary.changed_rows} "
            f"({table_parts}); locations_from_wohnort={summary.locations_from_wohnort}; "
            f"location_conflicts={summary.location_conflicts}; "
            f"dates_normalized={summary.dates_normalized}; "
            f"dates_unresolved={summary.dates_unresolved}."
        )
        if summary.backup_path is not None:
            print(f"Backup written to {summary.backup_path}.")
        return 0
    raise ValueError(f"Unsupported db command: {args.db_command}")


def run_ingest_command(args: argparse.Namespace) -> int:
    if args.ingest_command == "source":
        from .ingest import ingest_source

        summary = ingest_source(
            db_path=args.db,
            source=args.source,
            input_dir=args.input_dir,
            artifacts_dir=args.artifacts_dir,
            layout_family=args.layout_family,
            limit=args.limit,
        )
        print(
            f"Ingested {summary.documents} documents into {args.db}; "
            f"{summary.text_artifacts} OCR text artifacts, {summary.tsv_artifacts} TSV artifacts, "
            f"{summary.ocr_outputs} OCR outputs. Run {summary.run_id}."
        )
        return 0
    if args.ingest_command == "results":
        from .ingest import ingest_results

        summary = ingest_results(
            db_path=args.db,
            source=args.source,
            output_csv=args.output_csv,
            results_file=args.results_file,
            method=args.method,
            provider=args.provider,
            model=args.model,
            prompt_version=args.prompt_version,
            candidate_kind=args.candidate_kind,
            input_dir=args.input_dir,
        )
        print(
            f"Ingested {summary.extraction_outputs} extraction outputs and "
            f"{summary.label_candidates} label candidates into {args.db}. Run {summary.run_id}."
        )
        return 0
    raise ValueError(f"Unsupported ingest command: {args.ingest_command}")


def run_dataset_command(args: argparse.Namespace) -> int:
    if args.dataset_command == "split":
        from .evaluation import create_source_year_split

        summary = create_source_year_split(
            db_path=args.db,
            name=args.name,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(
            f"Dataset split {summary.name} written: "
            f"{summary.train} train, {summary.validation} validation, {summary.test} test."
        )
        return 0
    if args.dataset_command == "export":
        rows = _export_dataset_jsonl(args.db, args.split, args.label_set)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )
        print(f"Exported {len(rows)} dataset rows to {args.output_file}.")
        return 0
    if args.dataset_command == "export-router":
        from .features import export_router_dataset

        summary = export_router_dataset(
            db_path=args.db,
            output_file=args.output_file,
            label_set=args.label_set,
            variant_alias=args.variant,
            feature_set=args.feature_set,
            variants_config=args.variants_config,
        )
        print(
            f"Exported {summary.rows} router rows to {summary.output_file}; "
            f"missing_features={summary.missing_features}; "
            f"missing_predictions={summary.missing_predictions}."
        )
        return 0
    raise ValueError(f"Unsupported dataset command: {args.dataset_command}")


def run_features_command(args: argparse.Namespace) -> int:
    if args.features_command == "build":
        from .features import build_feature_snapshots

        summary = build_feature_snapshots(
            db_path=args.db,
            feature_set=args.feature_set,
        )
        print(
            f"Feature set {summary.feature_set} built: "
            f"{summary.snapshots} snapshots for {summary.documents} documents."
        )
        return 0
    raise ValueError(f"Unsupported features command: {args.features_command}")


def run_export_command(args: argparse.Namespace) -> int:
    if args.export_command == "csv":
        from .storage import apply_migrations, connect, export_priority_csv

        apply_migrations(args.db)
        with connect(args.db) as connection:
            summary = export_priority_csv(
                connection,
                output_csv=args.output_file,
                label_set=args.label_set,
                variants_config=args.variants_config,
            )
        method_parts = ", ".join(
            f"{method}={count}" for method, count in summary.method_rows.items()
        )
        print(
            f"Exported {summary.rows} rows to {args.output_file}; "
            f"ground_truth={summary.ground_truth_rows}; {method_parts}; "
            f"missing_documents={summary.missing_documents}."
        )
        return 0
    raise ValueError(f"Unsupported export command: {args.export_command}")


def run_eval_command(args: argparse.Namespace) -> int:
    if args.eval_command == "run":
        from .evaluation import evaluate_variant

        summary = evaluate_variant(
            db_path=args.db,
            label_set=args.label_set,
            variant_alias=args.variant,
            variants_config=args.variants_config,
            split_name=args.split,
        )
        print(
            f"Evaluation: {summary.documents} documents; "
            f"exact={summary.exact_record_accuracy:.3f}; "
            f"field_f1={summary.field_f1:.3f}; "
            f"precision={summary.field_precision:.3f}; recall={summary.field_recall:.3f}; "
            f"missing_predictions={summary.missing_predictions}; "
            f"estimated_tokens={summary.estimated_tokens_total} "
            f"({summary.estimated_tokens_count} rows); "
            f"mean_latency_ms={summary.latency_ms_mean} "
            f"({summary.latency_ms_count} rows); "
            f"cost_usd={summary.cost_usd_total} ({summary.cost_usd_count} rows)."
        )
        return 0
    raise ValueError(f"Unsupported eval command: {args.eval_command}")


def run_router_command(args: argparse.Namespace) -> int:
    if args.router_command == "train":
        from .router.model import train_router_from_db

        summary = train_router_from_db(
            db_path=args.db,
            label_set=args.label_set,
            feature_set=args.feature_set,
            model_dir=args.model_dir,
            split_name=args.split,
            target_f1_threshold=args.target_f1_threshold,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            cheap_cost=args.cheap_cost,
            vlm_cost=args.vlm_cost,
            lambda_cost=args.lambda_cost,
            min_train_rows=args.min_train_rows,
            variants_config=args.variants_config,
        )
        print(
            f"Router trained in {summary.model_dir}: "
            f"{summary.training_rows} train, {summary.validation_rows} validation; "
            f"threshold={summary.threshold:.3f}; "
            f"validation_escalation_rate={summary.validation_escalation_rate:.3f}; "
            f"validation_failure_rate={summary.validation_failure_rate:.3f}."
        )
        return 0
    if args.router_command == "manifest":
        from .router.manifest import write_router_manifest

        summary = write_router_manifest(
            db_path=args.db,
            model_dir=args.model_dir,
            output_file=args.output_file,
            label_set=args.label_set,
            feature_set=args.feature_set,
            threshold=args.threshold,
            variants_config=args.variants_config,
        )
        print(
            f"Router manifest written to {summary.output_file}: "
            f"{summary.rows} rows; {summary.escalations} VLM routes; "
            f"missing_features={summary.missing_features}."
        )
        return 0
    raise ValueError(f"Unsupported router command: {args.router_command}")


def run_review_command(args: argparse.Namespace) -> int:
    if args.review_command == "serve":
        from .review import serve_review_app

        serve_review_app(
            db_path=args.db,
            label_set=args.label_set,
            reviewer=args.reviewer,
            host=args.host,
            port=args.port,
            variants_config=args.variants_config,
        )
        return 0
    raise ValueError(f"Unsupported review command: {args.review_command}")


def _export_dataset_jsonl(db_path: Path, split_name: str, label_set: str) -> list[dict[str, object]]:
    from .storage import apply_migrations, connect

    apply_migrations(db_path)
    with connect(db_path) as connection:
        records = []
        for row in connection.execute(
            """
            SELECT
                dataset_memberships.subset,
                documents.id AS document_id,
                sources.name AS source,
                documents.filename_stem,
                documents.image_path,
                documents.year,
                ground_truth_labels.fields_json
            FROM dataset_memberships
            JOIN dataset_splits ON dataset_splits.id = dataset_memberships.split_id
            JOIN documents ON documents.id = dataset_memberships.document_id
            JOIN sources ON sources.id = documents.source_id
            LEFT JOIN ground_truth_labels
                ON ground_truth_labels.document_id = documents.id
                AND ground_truth_labels.label_set = ?
            WHERE dataset_splits.name = ?
            ORDER BY dataset_memberships.subset, sources.name, documents.filename_stem
            """,
            (label_set, split_name),
        ):
            fields = json.loads(row["fields_json"] or "{}") if row["fields_json"] else None
            records.append(
                {
                    "subset": row["subset"],
                    "document_id": row["document_id"],
                    "source": row["source"],
                    "filename_stem": row["filename_stem"],
                    "image_path": row["image_path"],
                    "year": row["year"],
                    "ground_truth": fields,
                }
            )
    return records


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
        if args.command == "db":
            return run_db_command(args)
        if args.command == "ingest":
            return run_ingest_command(args)
        if args.command == "dataset":
            return run_dataset_command(args)
        if args.command == "features":
            return run_features_command(args)
        if args.command == "export":
            return run_export_command(args)
        if args.command == "eval":
            return run_eval_command(args)
        if args.command == "router":
            return run_router_command(args)
        if args.command == "review":
            return run_review_command(args)
    except (ConfigError, FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
