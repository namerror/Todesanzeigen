from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS, STORED_COLUMNS, LlmProvider, VisionLlmProvider
from .methods import (
    TEXT_EXTRACTION_METHOD,
    VISION_IMAGE_ONLY_METHOD,
    VISION_REROUTE_METHOD,
    result_slot,
)
from .ocr import discover_images, image_mime_type
from .ocr_filtering import is_below_confidence_threshold, name_map_artifact_path
from .normalization import fields_for_csv, normalize_stored_fields


@dataclass(frozen=True)
class ExtractionResult:
    artifact_path: Path
    status: str
    row: dict[str, str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class AsyncExtractionSettings:
    concurrency: int = 1
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    max_retries: int = 5


@dataclass(frozen=True)
class NameHint:
    name: str
    confidence: float | None


@dataclass(frozen=True)
class VisionRerouteSettings:
    input_dir: Path
    provider: VisionLlmProvider
    results_file: Path | None = None
    concurrency: int = 1


@dataclass(frozen=True)
class RerouteCandidate:
    artifact_path: Path
    name_hint: NameHint
    threshold: float
    reason: str | None = None


@dataclass(frozen=True)
class DbExtractionRecord:
    result: ExtractionResult
    method: str
    name_hint: NameHint | None = None
    threshold: float | None = None
    reason: str = ""
    raw_response: str = ""
    attempts: int = 0
    estimated_tokens: int | None = None
    latency_ms: int | None = None
    image_path: Path | None = None
    prompt_version: str = "death_notice_v2"
    provider_name: str = ""
    model_name: str = ""


DEFAULT_NAME_CONFIDENCE_THRESHOLD = 85.0
VISION_PROCESSED_STATUS = "vision_processed"
VISION_FAILED_STATUS = "vision_failed"
CACHED_EXISTING_STATUS = "cached_existing"
PROCESSED_STATUSES = {"processed", "rerouted_processed", VISION_PROCESSED_STATUS}
CHECKPOINT_REUSE_STATUSES = {"processed", "rerouted_processed", "skipped_low_confidence"}
LLM_EXCLUDED_COLUMNS = {"foto", "bemerkungen", "quelle", "dateiname"}
LLM_PROMPT_COLUMNS = [column for column in STORED_COLUMNS if column not in LLM_EXCLUDED_COLUMNS]


def discover_artifacts(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Artifacts path is not a directory: {artifacts_dir}")

    return sorted(path for path in artifacts_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def build_extraction_prompt(
    ocr_text: str,
    *,
    filename: str,
    name_hint: NameHint,
    source: str = "",
) -> str:
    columns = ", ".join(LLM_PROMPT_COLUMNS)
    name_signal = _format_name_hint_for_prompt(name_hint)
    return f"""Du extrahierst strukturierte Daten aus OCR-Texten deutscher Todesanzeigen.

Gib ausschliesslich ein einzelnes valides JSON-Objekt zurueck. Verwende exakt diese Keys:
{columns}

Regeln:
- confidence_score ist eine Zahl von 0 bis 1 als String, z.B. "0.82".
- Fehlende oder unsichere Felder bleiben "".
- geburtsdatum und sterbedatum muessen exakt DD.MM.YYYY verwenden, z.B. "04.09.1937".
- Ort und Wohnort sind dasselbe Konzept. Gib nur ort zurueck.
- Bei mehreren sonstigen Orten kommt der zuerst genannte in ort; weitere kommen in weitere_orte.
- Orte, die als geburtsort oder sterbeort zugeordnet sind, kommen nicht zusaetzlich in ort oder weitere_orte.
- Stehen Geburts- und Sterbedatum nebeneinander und direkt darunter zwei Orte, ordne den linken Ort dem Geburtsdatum und den rechten Ort dem Sterbedatum zu.
- Unsicherheiten, OCR-Probleme und Interpretationshinweise kommen in zusaetzliche_hinweise.
- Erfinde keine Daten.
- Nutze das lokale OCR-Name-Signal als Zusatzhinweis fuer den Namen der verstorbenen Person.
- Verwende das lokale OCR-Name-Signal nur als Hinweis; bei Widerspruch zaehlt der erkennbare OCR-Text.

Lokales OCR-Name-Signal:
{name_signal}

OCR-Text:
{ocr_text}
"""


def build_vision_extraction_prompt(
    ocr_text: str,
    *,
    filename: str,
    name_hint: NameHint,
    source: str = "",
) -> str:
    del source
    columns = ", ".join(LLM_PROMPT_COLUMNS)
    name_signal = _format_name_hint_for_prompt(name_hint)
    ocr_hint = ocr_text.strip() or "<kein OCR-Text vorhanden>"
    return f"""Du extrahierst strukturierte Daten aus dem Bild einer deutschen Todesanzeige.

Gib ausschliesslich ein einzelnes valides JSON-Objekt zurueck. Verwende exakt diese Keys:
{columns}

Regeln:
- Lies den Text direkt aus dem Bild. Das Bild ist die massgebliche Quelle.
- confidence_score ist eine Zahl von 0 bis 1 als String, z.B. "0.82".
- Fehlende oder unsichere Felder bleiben "".
- geburtsdatum und sterbedatum muessen exakt DD.MM.YYYY verwenden, z.B. "04.09.1937".
- Ort und Wohnort sind dasselbe Konzept. Gib nur ort zurueck.
- Bei mehreren sonstigen Orten kommt der zuerst genannte in ort; weitere kommen in weitere_orte.
- Orte, die als geburtsort oder sterbeort zugeordnet sind, kommen nicht zusaetzlich in ort oder weitere_orte.
- Stehen Geburts- und Sterbedatum nebeneinander und direkt darunter zwei Orte, ordne den linken Ort dem Geburtsdatum und den rechten Ort dem Sterbedatum zu.
- Unsicherheiten, schwer lesbare Stellen und Interpretationshinweise kommen in zusaetzliche_hinweise.
- Erfinde keine Daten.
- Nutze den OCR-Text und das lokale OCR-Name-Signal nur als Zusatzhinweise.
- Bei Widerspruch zaehlt der im Bild erkennbare Text.

Lokales OCR-Name-Signal:
{name_signal}

OCR-Text aus dem schwachen lokalen Lauf:
{ocr_hint}
"""


def build_image_only_vision_extraction_prompt(
    *,
    filename: str,
    source: str = "",
) -> str:
    del filename, source
    columns = ", ".join(LLM_PROMPT_COLUMNS)
    return f"""Du extrahierst strukturierte Daten direkt aus dem Bild einer deutschen Todesanzeige.

Gib ausschliesslich ein einzelnes valides JSON-Objekt zurueck. Verwende exakt diese Keys:
{columns}

Regeln:
- Das Bild ist die einzige Quelle.
- confidence_score ist eine Zahl von 0 bis 1 als String, z.B. "0.82".
- Fehlende oder unsichere Felder bleiben "".
- geburtsdatum und sterbedatum muessen exakt DD.MM.YYYY verwenden, z.B. "04.09.1937".
- Ort und Wohnort sind dasselbe Konzept. Gib nur ort zurueck.
- Bei mehreren sonstigen Orten kommt der zuerst genannte in ort; weitere kommen in weitere_orte.
- Orte, die als geburtsort oder sterbeort zugeordnet sind, kommen nicht zusaetzlich in ort oder weitere_orte.
- Orte direkt unter Geburts- und Sterbedatum werden anhand ihrer visuellen Position zugeordnet.
- Unsicherheiten, schwer lesbare Stellen und Interpretationshinweise kommen in zusaetzliche_hinweise.
- Erfinde keine Daten.
"""


def _format_name_hint_for_prompt(name_hint: NameHint) -> str:
    if not name_hint.name:
        return "Unser OCR-Filtering-Algorithmus konnte keinen wahrscheinlichen Namen bestimmen."
    if name_hint.confidence is None:
        return (
            f'Unser OCR-Filtering-Algorithmus schaetzt, dass "{name_hint.name}" '
            "der Name der verstorbenen Person ist; es liegt keine verwertbare "
            "OCR-Konfidenz vor."
        )
    return (
        f'Unser OCR-Filtering-Algorithmus hat {name_hint.confidence:.1f}% Konfidenz, '
        f'dass "{name_hint.name}" der Name der verstorbenen Person ist.'
    )


def load_name_map(artifacts_dir: Path) -> dict[str, NameHint]:
    map_path = name_map_artifact_path(artifacts_dir)
    if not map_path.exists():
        raise FileNotFoundError(
            f"Missing OCR name map artifact: expected {map_path}. "
            "Run `todesanzeigen filter` before `todesanzeigen extract`."
        )

    raw_data = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"OCR name map must be a JSON object: {map_path}")

    name_map: dict[str, NameHint] = {}
    for filename, value in raw_data.items():
        if not isinstance(filename, str):
            raise ValueError(f"OCR name map keys must be filenames: {map_path}")
        if not isinstance(value, dict):
            raise ValueError(f"OCR name map entry must be an object for {filename}: {map_path}")

        name_value = value.get("name", "")
        confidence_value = value.get("confidence")
        if name_value is None:
            name = ""
        elif isinstance(name_value, str):
            name = name_value
        else:
            name = str(name_value)

        if confidence_value in (None, ""):
            confidence = None
        else:
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"OCR name map confidence must be numeric for {filename}: {map_path}"
                ) from exc

        name_map[filename] = NameHint(name=name, confidence=confidence)
    return name_map


def load_name_map_if_exists(artifacts_dir: Path) -> dict[str, NameHint]:
    map_path = name_map_artifact_path(artifacts_dir)
    if not map_path.exists():
        return {}
    return load_name_map(artifacts_dir)


def find_image_for_artifact(artifact_path: Path, input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif"):
        candidate = input_dir / f"{artifact_path.stem}{suffix}"
        if candidate.exists() and image_mime_type(candidate):
            return candidate

    matches = sorted(
        path
        for path in input_dir.rglob(f"{artifact_path.stem}.*")
        if path.is_file() and image_mime_type(path)
    )
    if not matches:
        raise FileNotFoundError(
            f"Source image for {artifact_path.name} was not found under {input_dir}"
        )
    return matches[0]


def parse_json_object(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")
    return data


def normalize_record(data: dict[str, Any], *, filename: str, source: str = "") -> dict[str, str]:
    record = {
        **data,
        "foto": "",
        "bemerkungen": "",
        "quelle": source,
        "dateiname": filename,
    }
    return normalize_stored_fields(record)


def extract_artifact(
    artifact_path: Path,
    provider: LlmProvider,
    *,
    name_hint: NameHint,
    source: str = "",
) -> dict[str, str]:
    ocr_text = artifact_path.read_text(encoding="utf-8")
    filename = artifact_path.stem
    prompt = build_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    response_text = provider.complete(prompt)
    data = parse_json_object(response_text)
    return normalize_record(data, filename=filename, source=source)


def extract_artifact_with_vision(
    artifact_path: Path,
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    name_hint: NameHint,
    source: str = "",
) -> dict[str, str]:
    ocr_text = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for reroute: {image_path}")
    filename = artifact_path.stem
    prompt = build_vision_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    response_text = provider.vision_complete(prompt, image_path, mime_type)
    data = parse_json_object(response_text)
    return normalize_record(data, filename=filename, source=source)


def extract_image_with_vision(
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    source: str = "",
) -> dict[str, str]:
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for vision extraction: {image_path}")
    filename = image_path.stem
    prompt = build_image_only_vision_extraction_prompt(
        filename=filename,
        source=source,
    )
    response_text = provider.vision_complete(prompt, image_path, mime_type)
    data = parse_json_object(response_text)
    return normalize_record(data, filename=filename, source=source)


def extract_artifacts_to_csv(
    artifacts_dir: Path,
    output_csv: Path,
    provider: LlmProvider,
    *,
    source: str = "",
    limit: int | None = None,
    name_confidence_threshold: float = DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    log_file: Path | None = None,
) -> list[ExtractionResult]:
    artifacts = discover_artifacts(artifacts_dir)
    if limit is not None:
        artifacts = artifacts[:limit]

    name_map = load_name_map(artifacts_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"Extraction log started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    results: list[ExtractionResult] = []
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for artifact_path in artifacts:
            try:
                name_hint = name_map[artifact_path.name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing OCR name map entry for {artifact_path.name}. "
                    "Run `todesanzeigen filter` for the same artifacts before extracting."
                ) from exc
            if _is_below_confidence_threshold(name_hint, name_confidence_threshold):
                _write_low_confidence_warning(
                    log_file,
                    artifact_path.name,
                    name_hint.confidence,
                    name_confidence_threshold,
                )
                results.append(ExtractionResult(artifact_path, "skipped_low_confidence"))
                continue
            row = extract_artifact(
                artifact_path,
                provider,
                source=source,
                name_hint=name_hint,
            )
            writer.writerow(fields_for_csv(row))
            results.append(ExtractionResult(artifact_path, "processed", row))

    return results


async def extract_artifacts_to_csv_async(
    artifacts_dir: Path,
    output_csv: Path,
    provider: LlmProvider,
    *,
    source: str = "",
    limit: int | None = None,
    name_confidence_threshold: float = DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    log_file: Path | None = None,
    checkpoint_file: Path | None = None,
    resume_from: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
    reroute_settings: VisionRerouteSettings | None = None,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if settings.concurrency < 1:
        raise ValueError("LLM extraction concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    if reroute_settings is not None and reroute_settings.concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")

    artifacts = discover_artifacts(artifacts_dir)
    if limit is not None:
        artifacts = artifacts[:limit]

    name_map = load_name_map(artifacts_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"Extraction log started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )

    resumed_results = _load_completed_checkpoint_results(
        artifacts_dir,
        resume_from,
        reuse_skipped_low_confidence=reroute_settings is None,
    )
    checkpoint_path = checkpoint_file or resume_from
    if checkpoint_path is None and log_file is not None:
        checkpoint_path = log_file.with_suffix(".results.jsonl")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
    reroute_checkpoint_path = reroute_settings.results_file if reroute_settings else None
    if reroute_checkpoint_path is not None:
        reroute_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        reroute_checkpoint_path.touch()

    results_by_name: dict[str, ExtractionResult] = {}
    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    reroute_checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.concurrency)
    reroute_semaphore = asyncio.Semaphore(
        reroute_settings.concurrency if reroute_settings else 1
    )

    async def append_checkpoint_records(
        filename: str,
        result: ExtractionResult,
        attempts: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await _append_checkpoint_record_async(
            checkpoint_path,
            checkpoint_lock,
            filename,
            result,
            attempts,
            metadata=metadata,
        )
        if (
            reroute_checkpoint_path is not None
            and reroute_checkpoint_path != checkpoint_path
            and metadata is not None
            and metadata.get("method") == VISION_REROUTE_METHOD
        ):
            await _append_checkpoint_record_async(
                reroute_checkpoint_path,
                reroute_checkpoint_lock,
                filename,
                result,
                attempts,
                metadata=metadata,
            )

    async def process_artifact(artifact_path: Path, name_hint: NameHint) -> ExtractionResult:
        async with semaphore:
            try:
                row, attempts = await _extract_artifact_async_with_retry(
                    artifact_path,
                    provider,
                    name_hint=name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(artifact_path, "processed", row)
                await append_checkpoint_records(
                    artifact_path.name,
                    result,
                    attempts,
                    _checkpoint_metadata(
                        method=TEXT_EXTRACTION_METHOD,
                        provider=provider,
                        artifact_path=artifact_path,
                        name_hint=name_hint,
                        threshold=name_confidence_threshold,
                    ),
                )
                return result
            except Exception as exc:
                result = ExtractionResult(artifact_path, "failed", error=str(exc))
                _write_extraction_error(log_file, artifact_path.name, exc)
                await append_checkpoint_records(
                    artifact_path.name,
                    result,
                    settings.max_retries + 1,
                    _checkpoint_metadata(
                        method=TEXT_EXTRACTION_METHOD,
                        provider=provider,
                        artifact_path=artifact_path,
                        name_hint=name_hint,
                        threshold=name_confidence_threshold,
                    ),
                )
                return result

    async def process_reroute_artifact(
        artifact_path: Path,
        name_hint: NameHint,
    ) -> ExtractionResult:
        assert reroute_settings is not None
        async with reroute_semaphore:
            image_path: Path | None = None
            try:
                image_path = find_image_for_artifact(artifact_path, reroute_settings.input_dir)
                row, attempts = await _extract_artifact_with_vision_async_with_retry(
                    artifact_path,
                    image_path,
                    reroute_settings.provider,
                    name_hint=name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(artifact_path, "rerouted_processed", row)
                metadata = _checkpoint_metadata(
                    method=VISION_REROUTE_METHOD,
                    provider=reroute_settings.provider,
                    artifact_path=artifact_path,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    source_image=image_path,
                )
                await append_checkpoint_records(
                    artifact_path.name,
                    result,
                    attempts,
                    metadata,
                )
                return result
            except Exception as exc:
                result = ExtractionResult(artifact_path, "rerouted_failed", error=str(exc))
                _write_extraction_error(log_file, artifact_path.name, exc, method=VISION_REROUTE_METHOD)
                metadata = _checkpoint_metadata(
                    method=VISION_REROUTE_METHOD,
                    provider=reroute_settings.provider,
                    artifact_path=artifact_path,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    source_image=image_path,
                )
                attempts = 0 if image_path is None else settings.max_retries + 1
                await append_checkpoint_records(
                    artifact_path.name,
                    result,
                    attempts,
                    metadata,
                )
                return result

    tasks: list[asyncio.Task[ExtractionResult]] = []
    for artifact_path in artifacts:
        resumed_result = resumed_results.get(artifact_path.name)
        if resumed_result is not None:
            results_by_name[artifact_path.name] = resumed_result
            continue

        try:
            name_hint = name_map[artifact_path.name]
        except KeyError as exc:
            raise ValueError(
                f"Missing OCR name map entry for {artifact_path.name}. "
                "Run `todesanzeigen filter` for the same artifacts before extracting."
            ) from exc
        if _is_below_confidence_threshold(name_hint, name_confidence_threshold):
            _write_low_confidence_warning(
                log_file,
                artifact_path.name,
                name_hint.confidence,
                name_confidence_threshold,
            )
            if reroute_settings is not None:
                tasks.append(asyncio.create_task(process_reroute_artifact(artifact_path, name_hint)))
                continue

            result = ExtractionResult(artifact_path, "skipped_low_confidence")
            results_by_name[artifact_path.name] = result
            await append_checkpoint_records(
                artifact_path.name,
                result,
                0,
                _checkpoint_metadata(
                    method=TEXT_EXTRACTION_METHOD,
                    provider=provider,
                    artifact_path=artifact_path,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                ),
            )
            continue

        tasks.append(asyncio.create_task(process_artifact(artifact_path, name_hint)))

    for result in await asyncio.gather(*tasks):
        results_by_name[result.artifact_path.name] = result

    ordered_results = [results_by_name[path.name] for path in artifacts]
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in ordered_results:
            if result.status in PROCESSED_STATUSES and result.row is not None:
                writer.writerow(fields_for_csv(result.row))

    return ordered_results


def _is_below_confidence_threshold(name_hint: NameHint, threshold: float) -> bool:
    return is_below_confidence_threshold(name_hint.confidence, threshold)


def _write_low_confidence_warning(
    log_file: Path | None,
    filename: str,
    confidence: float | None,
    threshold: float,
) -> None:
    if confidence is None:
        message = (
            f"WARNING: file {filename} has missing confidence, "
            "not passed to extraction"
        )
    else:
        message = (
            f"WARNING: file {filename} has low confidence "
            f"({confidence:.1f} < {threshold:.1f}), not passed to extraction"
        )
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def load_reroute_candidates(
    artifacts_dir: Path,
    *,
    low_confidence_file: Path | None = None,
    results_file: Path | None = None,
    threshold: float = DEFAULT_NAME_CONFIDENCE_THRESHOLD,
) -> list[RerouteCandidate]:
    if low_confidence_file is not None:
        return _load_low_confidence_file_candidates(artifacts_dir, low_confidence_file, threshold)
    if results_file is not None:
        return _load_results_file_candidates(artifacts_dir, results_file, threshold)

    name_map = load_name_map(artifacts_dir)
    candidates: list[RerouteCandidate] = []
    for filename, name_hint in sorted(name_map.items()):
        if _is_below_confidence_threshold(name_hint, threshold):
            candidates.append(
                RerouteCandidate(
                    artifacts_dir / filename,
                    name_hint,
                    threshold,
                    _low_confidence_reason(name_hint.confidence),
                )
            )
    return candidates


def select_reroute_candidates(
    candidates: list[RerouteCandidate],
    *,
    only: list[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 0,
    limit: int | None = None,
) -> list[RerouteCandidate]:
    selected = sorted(candidates, key=lambda candidate: candidate.artifact_path.name)
    if only:
        wanted = set(only)
        selected = [
            candidate
            for candidate in selected
            if candidate.artifact_path.name in wanted or candidate.artifact_path.stem in wanted
        ]
    if sample_ratio is not None:
        if sample_ratio <= 0 or sample_ratio > 1:
            raise ValueError("sample_ratio must be greater than 0 and at most 1.")
        if sample_ratio < 1 and selected:
            sample_size = max(1, math.ceil(len(selected) * sample_ratio))
            sampled_names = {
                candidate.artifact_path.name
                for candidate in random.Random(sample_seed).sample(selected, sample_size)
            }
            selected = [
                candidate for candidate in selected if candidate.artifact_path.name in sampled_names
            ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def select_image_paths(
    images: list[Path],
    *,
    only: list[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 0,
    limit: int | None = None,
) -> list[Path]:
    selected = sorted(images, key=lambda path: path.name)
    if only:
        wanted = set(only)
        selected = [
            path
            for path in selected
            if path.name in wanted or path.stem in wanted or str(path) in wanted
        ]
    if sample_ratio is not None:
        if sample_ratio <= 0 or sample_ratio > 1:
            raise ValueError("sample_ratio must be greater than 0 and at most 1.")
        if sample_ratio < 1 and selected:
            sample_size = max(1, math.ceil(len(selected) * sample_ratio))
            sampled_names = {
                path.name for path in random.Random(sample_seed).sample(selected, sample_size)
            }
            selected = [path for path in selected if path.name in sampled_names]
    if limit is not None:
        selected = selected[:limit]
    return selected


async def reroute_candidates_to_csv_async(
    candidates: list[RerouteCandidate],
    output_csv: Path,
    provider: VisionLlmProvider,
    *,
    input_dir: Path,
    source: str = "",
    log_file: Path | None = None,
    results_file: Path | None = None,
    merge_output_file: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
    concurrency: int = 1,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"Vision reroute log started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    if results_file is not None:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.touch()

    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def process_candidate(candidate: RerouteCandidate) -> ExtractionResult:
        async with semaphore:
            image_path: Path | None = None
            try:
                image_path = find_image_for_artifact(candidate.artifact_path, input_dir)
                row, attempts = await _extract_artifact_with_vision_async_with_retry(
                    candidate.artifact_path,
                    image_path,
                    provider,
                    name_hint=candidate.name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(candidate.artifact_path, "rerouted_processed", row)
                await _append_checkpoint_record_async(
                    results_file,
                    checkpoint_lock,
                    candidate.artifact_path.name,
                    result,
                    attempts,
                    metadata=_checkpoint_metadata(
                        method=VISION_REROUTE_METHOD,
                        provider=provider,
                        artifact_path=candidate.artifact_path,
                        name_hint=candidate.name_hint,
                        threshold=candidate.threshold,
                        source_image=image_path,
                        reason=candidate.reason,
                    ),
                )
                return result
            except Exception as exc:
                result = ExtractionResult(
                    candidate.artifact_path,
                    "rerouted_failed",
                    error=str(exc),
                )
                _write_extraction_error(
                    log_file,
                    candidate.artifact_path.name,
                    exc,
                    method=VISION_REROUTE_METHOD,
                )
                attempts = 0 if image_path is None else settings.max_retries + 1
                await _append_checkpoint_record_async(
                    results_file,
                    checkpoint_lock,
                    candidate.artifact_path.name,
                    result,
                    attempts,
                    metadata=_checkpoint_metadata(
                        method=VISION_REROUTE_METHOD,
                        provider=provider,
                        artifact_path=candidate.artifact_path,
                        name_hint=candidate.name_hint,
                        threshold=candidate.threshold,
                        source_image=image_path,
                        reason=candidate.reason,
                    ),
                )
                return result

    results = await asyncio.gather(
        *(asyncio.create_task(process_candidate(candidate)) for candidate in candidates)
    )
    _write_results_csv(output_csv, list(results))
    if merge_output_file is not None:
        merge_rows_into_csv(
            merge_output_file,
            [result.row for result in results if result.status in PROCESSED_STATUSES and result.row],
        )
    return list(results)


async def extract_images_to_csv_async(
    input_dir: Path,
    output_csv: Path,
    provider: VisionLlmProvider,
    *,
    source: str = "",
    limit: int | None = None,
    only: list[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 0,
    log_file: Path | None = None,
    results_file: Path | None = None,
    resume_from: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if settings.concurrency < 1:
        raise ValueError("Vision extraction concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")

    images = select_image_paths(
        discover_images(input_dir),
        only=only,
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
        limit=limit,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"Image-only vision extraction log started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )

    checkpoint_path = results_file or resume_from
    resume_path = resume_from or results_file
    resumed_results = _load_completed_image_checkpoint_results(input_dir, resume_path)
    if checkpoint_path is None and log_file is not None:
        checkpoint_path = log_file.with_suffix(".results.jsonl")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()

    results_by_name: dict[str, ExtractionResult] = {}
    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def process_image(image_path: Path) -> ExtractionResult:
        async with semaphore:
            try:
                row, attempts = await _extract_image_with_vision_async_with_retry(
                    image_path,
                    provider,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(image_path, VISION_PROCESSED_STATUS, row)
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    image_path.name,
                    result,
                    attempts,
                    metadata=_image_checkpoint_metadata(
                        method=VISION_IMAGE_ONLY_METHOD,
                        provider=provider,
                        image_path=image_path,
                    ),
                )
                return result
            except Exception as exc:
                result = ExtractionResult(image_path, VISION_FAILED_STATUS, error=str(exc))
                _write_extraction_error(log_file, image_path.name, exc, method=VISION_IMAGE_ONLY_METHOD)
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    image_path.name,
                    result,
                    settings.max_retries + 1,
                    metadata=_image_checkpoint_metadata(
                        method=VISION_IMAGE_ONLY_METHOD,
                        provider=provider,
                        image_path=image_path,
                    ),
                )
                return result

    tasks: list[asyncio.Task[ExtractionResult]] = []
    for image_path in images:
        resumed_result = resumed_results.get(image_path.name)
        if resumed_result is not None:
            results_by_name[image_path.name] = resumed_result
            continue
        tasks.append(asyncio.create_task(process_image(image_path)))

    for result in await asyncio.gather(*tasks):
        results_by_name[result.artifact_path.name] = result

    ordered_results = [results_by_name[path.name] for path in images]
    _write_results_csv(output_csv, ordered_results)
    return ordered_results


async def extract_artifacts_to_db_async(
    artifacts_dir: Path,
    db_path: Path,
    provider: LlmProvider,
    *,
    input_dir: Path = Path("input"),
    source: str = "",
    limit: int | None = None,
    name_confidence_threshold: float = DEFAULT_NAME_CONFIDENCE_THRESHOLD,
    log_file: Path | None = None,
    checkpoint_file: Path | None = None,
    resume_from: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
    reroute_settings: VisionRerouteSettings | None = None,
    candidate_kind: str = "pipeline",
    force: bool = False,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    _validate_async_settings(settings, reroute_settings)
    _require_provider_model(provider)
    if reroute_settings is not None:
        _require_provider_model(reroute_settings.provider)
    _ensure_db_ready(db_path)

    artifacts = discover_artifacts(artifacts_dir)
    if limit is not None:
        artifacts = artifacts[:limit]

    name_map = load_name_map(artifacts_dir)
    _initialize_log(log_file, "DB extraction")
    checkpoint_path = _checkpoint_path(checkpoint_file, resume_from, log_file)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
    reroute_checkpoint_path = reroute_settings.results_file if reroute_settings else None
    if reroute_checkpoint_path is not None:
        reroute_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        reroute_checkpoint_path.touch()

    resumed_records = _load_completed_checkpoint_db_records(
        artifacts_dir,
        resume_from or checkpoint_file,
        reuse_skipped_low_confidence=reroute_settings is None,
    )
    records_by_name: dict[str, DbExtractionRecord] = {}
    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    reroute_checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.concurrency)
    reroute_semaphore = asyncio.Semaphore(
        reroute_settings.concurrency if reroute_settings else 1
    )

    async def append_checkpoint_records(
        filename: str,
        record: DbExtractionRecord,
    ) -> None:
        metadata = _db_record_checkpoint_metadata(record)
        await _append_checkpoint_record_async(
            checkpoint_path,
            checkpoint_lock,
            filename,
            record.result,
            record.attempts,
            metadata=metadata,
        )
        if (
            reroute_checkpoint_path is not None
            and reroute_checkpoint_path != checkpoint_path
            and record.method == VISION_REROUTE_METHOD
        ):
            await _append_checkpoint_record_async(
                reroute_checkpoint_path,
                reroute_checkpoint_lock,
                filename,
                record.result,
                record.attempts,
                metadata=metadata,
            )

    async def process_artifact(artifact_path: Path, name_hint: NameHint) -> DbExtractionRecord:
        async with semaphore:
            started = time.perf_counter()
            try:
                row, raw_response, attempts, estimated_tokens = await _extract_artifact_db_payload(
                    artifact_path,
                    provider,
                    name_hint=name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(artifact_path, "processed", row)
                record = DbExtractionRecord(
                    result=result,
                    method=TEXT_EXTRACTION_METHOD,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    raw_response=raw_response,
                    attempts=attempts,
                    estimated_tokens=estimated_tokens,
                    latency_ms=_elapsed_ms(started),
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await append_checkpoint_records(artifact_path.name, record)
                return record
            except Exception as exc:
                result = ExtractionResult(artifact_path, "failed", error=str(exc))
                _write_extraction_error(log_file, artifact_path.name, exc)
                record = DbExtractionRecord(
                    result=result,
                    method=TEXT_EXTRACTION_METHOD,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    reason="extraction_failed",
                    attempts=settings.max_retries + 1,
                    latency_ms=_elapsed_ms(started),
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await append_checkpoint_records(artifact_path.name, record)
                return record

    async def process_reroute_artifact(
        artifact_path: Path,
        name_hint: NameHint,
    ) -> DbExtractionRecord:
        assert reroute_settings is not None
        async with reroute_semaphore:
            started = time.perf_counter()
            image_path: Path | None = None
            try:
                image_path = find_image_for_artifact(artifact_path, reroute_settings.input_dir)
                (
                    row,
                    raw_response,
                    attempts,
                    estimated_tokens,
                ) = await _extract_artifact_with_vision_db_payload(
                    artifact_path,
                    image_path,
                    reroute_settings.provider,
                    name_hint=name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(artifact_path, "rerouted_processed", row)
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_REROUTE_METHOD,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    reason=_low_confidence_reason(name_hint.confidence),
                    raw_response=raw_response,
                    attempts=attempts,
                    estimated_tokens=estimated_tokens,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(reroute_settings.provider),
                    model_name=_model_name(reroute_settings.provider),
                )
                await append_checkpoint_records(artifact_path.name, record)
                return record
            except Exception as exc:
                result = ExtractionResult(artifact_path, "rerouted_failed", error=str(exc))
                _write_extraction_error(
                    log_file,
                    artifact_path.name,
                    exc,
                    method=VISION_REROUTE_METHOD,
                )
                attempts = 0 if image_path is None else settings.max_retries + 1
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_REROUTE_METHOD,
                    name_hint=name_hint,
                    threshold=name_confidence_threshold,
                    reason=_low_confidence_reason(name_hint.confidence),
                    attempts=attempts,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(reroute_settings.provider),
                    model_name=_model_name(reroute_settings.provider),
                )
                await append_checkpoint_records(artifact_path.name, record)
                return record

    tasks: list[asyncio.Task[DbExtractionRecord]] = []
    for artifact_path in artifacts:
        try:
            name_hint = name_map[artifact_path.name]
        except KeyError as exc:
            raise ValueError(
                f"Missing OCR name map entry for {artifact_path.name}. "
                "Run `todesanzeigen filter` for the same artifacts before extracting."
            ) from exc
        below_threshold = _is_below_confidence_threshold(name_hint, name_confidence_threshold)
        request_method = (
            VISION_REROUTE_METHOD
            if below_threshold and reroute_settings is not None
            else TEXT_EXTRACTION_METHOD
        )
        cached_record = _cached_db_record_for_request(
            db_path,
            artifact_path=artifact_path,
            method=request_method,
            source=source,
            input_dir=(
                reroute_settings.input_dir
                if request_method == VISION_REROUTE_METHOD and reroute_settings is not None
                else input_dir
            ),
            name_hint=name_hint,
            threshold=name_confidence_threshold,
            reason=(
                _low_confidence_reason(name_hint.confidence)
                if request_method == VISION_REROUTE_METHOD or below_threshold
                else ""
            ),
            force=force,
        )
        if cached_record is not None:
            records_by_name[artifact_path.name] = cached_record
            await append_checkpoint_records(artifact_path.name, cached_record)
            continue

        resumed_record = None if force else resumed_records.get(artifact_path.name)
        if resumed_record is not None:
            records_by_name[artifact_path.name] = resumed_record
            continue

        if below_threshold:
            _write_low_confidence_warning(
                log_file,
                artifact_path.name,
                name_hint.confidence,
                name_confidence_threshold,
            )
            if reroute_settings is not None:
                tasks.append(asyncio.create_task(process_reroute_artifact(artifact_path, name_hint)))
                continue

            result = ExtractionResult(artifact_path, "skipped_low_confidence")
            record = DbExtractionRecord(
                result=result,
                method=TEXT_EXTRACTION_METHOD,
                name_hint=name_hint,
                threshold=name_confidence_threshold,
                reason=_low_confidence_reason(name_hint.confidence),
                provider_name=_provider_name(provider),
                model_name=_model_name(provider),
            )
            records_by_name[artifact_path.name] = record
            await append_checkpoint_records(artifact_path.name, record)
            continue

        tasks.append(asyncio.create_task(process_artifact(artifact_path, name_hint)))

    for record in await asyncio.gather(*tasks):
        records_by_name[record.result.artifact_path.name] = record

    ordered_records = [records_by_name[path.name] for path in artifacts]
    _persist_db_extraction_records(
        db_path,
        ordered_records,
        command="extract",
        source=source,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        candidate_kind=candidate_kind,
        run_method=TEXT_EXTRACTION_METHOD,
        run_provider=_provider_name(provider),
        run_model=_model_name(provider),
        run_config={
            "name_confidence_threshold": name_confidence_threshold,
            "reroute": reroute_settings is not None,
            "checkpoint_file": str(checkpoint_path or ""),
        },
    )
    return [record.result for record in ordered_records]


async def reroute_candidates_to_db_async(
    candidates: list[RerouteCandidate],
    db_path: Path,
    provider: VisionLlmProvider,
    *,
    input_dir: Path,
    source: str = "",
    log_file: Path | None = None,
    results_file: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
    concurrency: int = 1,
    candidate_kind: str = "pipeline",
    force: bool = False,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    _require_provider_model(provider)
    _ensure_db_ready(db_path)

    _initialize_log(log_file, "DB vision reroute")
    if results_file is not None:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.touch()

    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def process_candidate(candidate: RerouteCandidate) -> DbExtractionRecord:
        async with semaphore:
            started = time.perf_counter()
            image_path: Path | None = None
            try:
                image_path = find_image_for_artifact(candidate.artifact_path, input_dir)
                (
                    row,
                    raw_response,
                    attempts,
                    estimated_tokens,
                ) = await _extract_artifact_with_vision_db_payload(
                    candidate.artifact_path,
                    image_path,
                    provider,
                    name_hint=candidate.name_hint,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(candidate.artifact_path, "rerouted_processed", row)
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_REROUTE_METHOD,
                    name_hint=candidate.name_hint,
                    threshold=candidate.threshold,
                    reason=candidate.reason or _low_confidence_reason(candidate.name_hint.confidence),
                    raw_response=raw_response,
                    attempts=attempts,
                    estimated_tokens=estimated_tokens,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await _append_checkpoint_record_async(
                    results_file,
                    checkpoint_lock,
                    candidate.artifact_path.name,
                    result,
                    attempts,
                    metadata=_db_record_checkpoint_metadata(record),
                )
                return record
            except Exception as exc:
                result = ExtractionResult(candidate.artifact_path, "rerouted_failed", error=str(exc))
                _write_extraction_error(
                    log_file,
                    candidate.artifact_path.name,
                    exc,
                    method=VISION_REROUTE_METHOD,
                )
                attempts = 0 if image_path is None else settings.max_retries + 1
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_REROUTE_METHOD,
                    name_hint=candidate.name_hint,
                    threshold=candidate.threshold,
                    reason=candidate.reason or _low_confidence_reason(candidate.name_hint.confidence),
                    attempts=attempts,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await _append_checkpoint_record_async(
                    results_file,
                    checkpoint_lock,
                    candidate.artifact_path.name,
                    result,
                    attempts,
                    metadata=_db_record_checkpoint_metadata(record),
                )
                return record

    records_by_name: dict[str, DbExtractionRecord] = {}
    tasks: list[asyncio.Task[DbExtractionRecord]] = []
    for candidate in candidates:
        cached_record = _cached_db_record_for_request(
            db_path,
            artifact_path=candidate.artifact_path,
            method=VISION_REROUTE_METHOD,
            source=source,
            input_dir=input_dir,
            name_hint=candidate.name_hint,
            threshold=candidate.threshold,
            reason=candidate.reason or _low_confidence_reason(candidate.name_hint.confidence),
            force=force,
        )
        if cached_record is not None:
            records_by_name[candidate.artifact_path.name] = cached_record
            await _append_checkpoint_record_async(
                results_file,
                checkpoint_lock,
                candidate.artifact_path.name,
                cached_record.result,
                cached_record.attempts,
                metadata=_db_record_checkpoint_metadata(cached_record),
            )
            continue
        tasks.append(asyncio.create_task(process_candidate(candidate)))

    for record in await asyncio.gather(*tasks):
        records_by_name[record.result.artifact_path.name] = record

    records = [records_by_name[candidate.artifact_path.name] for candidate in candidates]
    _persist_db_extraction_records(
        db_path,
        records,
        command="reroute",
        source=source,
        input_dir=input_dir,
        artifacts_dir=Path(""),
        candidate_kind=candidate_kind,
        run_method=VISION_REROUTE_METHOD,
        run_provider=_provider_name(provider),
        run_model=_model_name(provider),
        run_config={"results_file": str(results_file or "")},
    )
    return [record.result for record in records]


async def extract_images_to_db_async(
    input_dir: Path,
    db_path: Path,
    provider: VisionLlmProvider,
    *,
    source: str = "",
    limit: int | None = None,
    only: list[str] | None = None,
    sample_ratio: float | None = None,
    sample_seed: int = 0,
    log_file: Path | None = None,
    results_file: Path | None = None,
    resume_from: Path | None = None,
    settings: AsyncExtractionSettings | None = None,
    candidate_kind: str = "teacher",
    force: bool = False,
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if settings.concurrency < 1:
        raise ValueError("Vision extraction concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    _require_provider_model(provider)
    _ensure_db_ready(db_path)

    images = select_image_paths(
        discover_images(input_dir),
        only=only,
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
        limit=limit,
    )
    _initialize_log(log_file, "DB image-only vision extraction")
    checkpoint_path = results_file or resume_from
    resume_path = resume_from or results_file
    if checkpoint_path is None and log_file is not None:
        checkpoint_path = log_file.with_suffix(".results.jsonl")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()

    resumed_records = _load_completed_image_checkpoint_db_records(input_dir, resume_path)
    records_by_name: dict[str, DbExtractionRecord] = {}
    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def process_image(image_path: Path) -> DbExtractionRecord:
        async with semaphore:
            started = time.perf_counter()
            try:
                row, raw_response, attempts, estimated_tokens = await _extract_image_db_payload(
                    image_path,
                    provider,
                    source=source,
                    limiter=limiter,
                    max_retries=settings.max_retries,
                )
                result = ExtractionResult(image_path, VISION_PROCESSED_STATUS, row)
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_IMAGE_ONLY_METHOD,
                    reason="image_only",
                    raw_response=raw_response,
                    attempts=attempts,
                    estimated_tokens=estimated_tokens,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    image_path.name,
                    result,
                    attempts,
                    metadata=_db_record_checkpoint_metadata(record),
                )
                return record
            except Exception as exc:
                result = ExtractionResult(image_path, VISION_FAILED_STATUS, error=str(exc))
                _write_extraction_error(log_file, image_path.name, exc, method=VISION_IMAGE_ONLY_METHOD)
                record = DbExtractionRecord(
                    result=result,
                    method=VISION_IMAGE_ONLY_METHOD,
                    reason="image_only",
                    attempts=settings.max_retries + 1,
                    latency_ms=_elapsed_ms(started),
                    image_path=image_path,
                    provider_name=_provider_name(provider),
                    model_name=_model_name(provider),
                )
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    image_path.name,
                    result,
                    record.attempts,
                    metadata=_db_record_checkpoint_metadata(record),
                )
                return record

    tasks: list[asyncio.Task[DbExtractionRecord]] = []
    for image_path in images:
        cached_record = _cached_db_record_for_request(
            db_path,
            artifact_path=image_path,
            method=VISION_IMAGE_ONLY_METHOD,
            source=source,
            input_dir=input_dir,
            image_path=image_path,
            reason="image_only",
            force=force,
        )
        if cached_record is not None:
            records_by_name[image_path.name] = cached_record
            await _append_checkpoint_record_async(
                checkpoint_path,
                checkpoint_lock,
                image_path.name,
                cached_record.result,
                cached_record.attempts,
                metadata=_db_record_checkpoint_metadata(cached_record),
            )
            continue

        resumed_record = None if force else resumed_records.get(image_path.name)
        if resumed_record is not None:
            records_by_name[image_path.name] = resumed_record
            continue
        tasks.append(asyncio.create_task(process_image(image_path)))

    for record in await asyncio.gather(*tasks):
        records_by_name[record.result.artifact_path.name] = record

    ordered_records = [records_by_name[path.name] for path in images]
    _persist_db_extraction_records(
        db_path,
        ordered_records,
        command="vision-extract",
        source=source,
        input_dir=input_dir,
        artifacts_dir=Path(""),
        candidate_kind=candidate_kind,
        run_method=VISION_IMAGE_ONLY_METHOD,
        run_provider=_provider_name(provider),
        run_model=_model_name(provider),
        run_config={
            "results_file": str(checkpoint_path or ""),
            "sample_ratio": sample_ratio,
            "sample_seed": sample_seed,
        },
    )
    return [record.result for record in ordered_records]


def merge_rows_into_csv(output_csv: Path, rows: list[dict[str, str] | None]) -> None:
    clean_rows = [row for row in rows if row is not None]
    if not clean_rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_by_filename: dict[str, dict[str, str]] = {}
    if output_csv.exists():
        with output_csv.open(encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                filename = row.get("dateiname", "")
                if filename:
                    merged_by_filename[filename] = normalize_stored_fields(row)
    for row in clean_rows:
        merged_by_filename[row.get("dateiname", "")] = row
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in merged_by_filename.values():
            writer.writerow(fields_for_csv(row))


def _write_results_csv(output_csv: Path, results: list[ExtractionResult]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            if result.status in PROCESSED_STATUSES and result.row is not None:
                writer.writerow(fields_for_csv(result.row))


def _validate_async_settings(
    settings: AsyncExtractionSettings,
    reroute_settings: VisionRerouteSettings | None = None,
) -> None:
    if settings.concurrency < 1:
        raise ValueError("LLM extraction concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")
    if reroute_settings is not None and reroute_settings.concurrency < 1:
        raise ValueError("Vision reroute concurrency must be at least 1.")


def _initialize_log(log_file: Path | None, title: str) -> None:
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        f"{title} log started {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


def _checkpoint_path(
    checkpoint_file: Path | None,
    resume_from: Path | None,
    log_file: Path | None,
) -> Path | None:
    path = checkpoint_file or resume_from
    if path is None and log_file is not None:
        path = log_file.with_suffix(".results.jsonl")
    return path


def _ensure_db_ready(db_path: Path) -> None:
    from .storage import apply_migrations

    apply_migrations(db_path)


def _cached_db_record_for_request(
    db_path: Path,
    *,
    artifact_path: Path,
    method: str,
    source: str,
    input_dir: Path,
    name_hint: NameHint | None = None,
    threshold: float | None = None,
    reason: str = "",
    image_path: Path | None = None,
    force: bool = False,
) -> DbExtractionRecord | None:
    from .storage import (
        connect,
        latest_active_extraction_by_slot,
        sha256_file,
        supersede_active_extractions_by_slot,
        upsert_document,
    )

    resolved_image = image_path or _safe_record_image_path(artifact_path, input_dir)
    image_digest = (
        sha256_file(resolved_image)
        if resolved_image is not None and resolved_image.exists() and resolved_image.is_file()
        else ""
    )
    slot = result_slot(method)

    with connect(db_path) as connection:
        document_id: int | None = None
        if source:
            document_id = upsert_document(
                connection,
                source_name=source,
                filename_stem=artifact_path.stem,
                image_path=resolved_image or "",
                image_sha256=image_digest,
                mime_type=image_mime_type(resolved_image) if resolved_image is not None else "",
            )
        elif resolved_image is not None:
            existing_document = connection.execute(
                """
                SELECT id FROM documents
                WHERE image_path = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (str(resolved_image),),
            ).fetchone()
            document_id = int(existing_document["id"]) if existing_document is not None else None

        if document_id is None:
            return None

        if force:
            supersede_active_extractions_by_slot(
                connection,
                document_id=document_id,
                result_slot_value=slot,
            )
            return None

        existing = latest_active_extraction_by_slot(
            connection,
            document_id=document_id,
            result_slot_value=slot,
        )
        if existing is None:
            return None

        return DbExtractionRecord(
            result=ExtractionResult(
                artifact_path,
                CACHED_EXISTING_STATUS,
                _fields_from_output_row(existing),
            ),
            method=str(existing["method"]),
            name_hint=name_hint,
            threshold=threshold,
            reason=reason or str(existing["route_reason"] or "cached_existing"),
            attempts=0,
            estimated_tokens=0,
            latency_ms=0,
            image_path=resolved_image,
            provider_name=str(existing["provider"] or ""),
            model_name=str(existing["model"] or ""),
        )


def _safe_record_image_path(artifact_path: Path, input_dir: Path) -> Path | None:
    if image_mime_type(artifact_path):
        return artifact_path
    if artifact_path.suffix != ".txt":
        return None
    try:
        return find_image_for_artifact(artifact_path, input_dir)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _fields_from_output_row(row: Any) -> dict[str, str]:
    try:
        data = json.loads(row["fields_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        return {}
    return {str(key): "" if value is None else str(value) for key, value in data.items()}


def _persist_db_extraction_records(
    db_path: Path,
    records: list[DbExtractionRecord],
    *,
    command: str,
    source: str,
    input_dir: Path,
    artifacts_dir: Path,
    candidate_kind: str,
    run_method: str,
    run_provider: str,
    run_model: str,
    run_config: dict[str, Any],
) -> None:
    del artifacts_dir
    from .storage import (
        apply_migrations,
        connect,
        create_run,
        finish_run,
        insert_extraction_output,
        insert_label_candidate,
        record_artifact,
        sha256_file,
        upsert_document,
        upsert_ocr_output,
    )

    apply_migrations(db_path)
    with connect(db_path) as connection:
        run_id = create_run(
            connection,
            command=command,
            method=run_method,
            provider=run_provider,
            model=run_model,
            config=run_config,
        )
        try:
            for record in records:
                result = record.result
                if result.status == CACHED_EXISTING_STATUS:
                    continue
                row = result.row or {}
                source_name = source or row.get("quelle", "") or "unknown"
                filename_stem = row.get("dateiname", "") or result.artifact_path.stem
                image_path = _record_image_path(record, input_dir)
                image_digest = (
                    sha256_file(image_path)
                    if image_path is not None and image_path.exists() and image_path.is_file()
                    else ""
                )
                document_id = upsert_document(
                    connection,
                    source_name=source_name,
                    filename_stem=filename_stem,
                    image_path=image_path or "",
                    image_sha256=image_digest,
                    mime_type=image_mime_type(image_path) if image_path is not None else "",
                )
                image_artifact_id = None
                if image_path is not None:
                    image_artifact_id = record_artifact(
                        connection,
                        document_id=document_id,
                        run_id=run_id,
                        artifact_type="source_image",
                        path=image_path,
                        producer=record.method,
                    )

                text_artifact_id = None
                tsv_artifact_id = None
                ocr_output_id = None
                if result.artifact_path.suffix == ".txt":
                    text_artifact_id = record_artifact(
                        connection,
                        document_id=document_id,
                        run_id=run_id,
                        artifact_type="ocr_text",
                        path=result.artifact_path,
                        producer="tesseract",
                    )
                    tsv_path = result.artifact_path.with_suffix(".tsv")
                    if tsv_path.exists():
                        tsv_artifact_id = record_artifact(
                            connection,
                            document_id=document_id,
                            run_id=run_id,
                            artifact_type="ocr_tsv",
                            path=tsv_path,
                            producer="tesseract",
                        )
                    text = (
                        result.artifact_path.read_text(encoding="utf-8")
                        if result.artifact_path.exists()
                        else ""
                    )
                    ocr_output_id = upsert_ocr_output(
                        connection,
                        document_id=document_id,
                        run_id=run_id,
                        text=text,
                        text_artifact_id=text_artifact_id,
                        tsv_artifact_id=tsv_artifact_id,
                        features=_basic_ocr_features(text, tsv_path if tsv_path.exists() else None),
                        name_hint=record.name_hint.name if record.name_hint is not None else "",
                        name_confidence=(
                            record.name_hint.confidence if record.name_hint is not None else None
                        ),
                    )

                source_artifact_id = image_artifact_id if record.method.startswith("vision") else text_artifact_id
                config_hash = _stable_config_hash(
                    {
                        **run_config,
                        "method": record.method,
                        "provider": record.provider_name,
                        "model": record.model_name,
                        "prompt_version": record.prompt_version,
                    }
                )
                output_id = insert_extraction_output(
                    connection,
                    document_id=document_id,
                    run_id=run_id,
                    method=record.method,
                    provider=record.provider_name,
                    model=record.model_name,
                    prompt_version=record.prompt_version,
                    fields=row,
                    status=result.status,
                    raw_response=record.raw_response,
                    error=result.error or "",
                    attempts=record.attempts,
                    estimated_tokens=record.estimated_tokens,
                    source_artifact_id=source_artifact_id,
                    latency_ms=record.latency_ms,
                    route_reason=record.reason,
                    route_decision=_route_decision(record),
                    config=run_config,
                    ocr_output_id=ocr_output_id,
                    result_slot_value=result_slot(record.method),
                    input_fingerprint=_record_input_fingerprint(record, input_dir, sha256_file),
                    config_hash=config_hash,
                )
                if result.status in PROCESSED_STATUSES and result.row is not None:
                    insert_label_candidate(
                        connection,
                        document_id=document_id,
                        extraction_output_id=output_id,
                        source_kind=candidate_kind,
                        source_name=record.method,
                        fields=result.row,
                    )
            finish_run(connection, run_id)
        except Exception:
            finish_run(connection, run_id, status="failed")
            raise


def _record_image_path(record: DbExtractionRecord, input_dir: Path) -> Path | None:
    if record.image_path is not None:
        return record.image_path
    artifact_path = record.result.artifact_path
    if image_mime_type(artifact_path):
        return artifact_path
    if artifact_path.suffix == ".txt":
        try:
            return find_image_for_artifact(artifact_path, input_dir)
        except FileNotFoundError:
            return None
    return None


def _record_input_fingerprint(
    record: DbExtractionRecord,
    input_dir: Path,
    sha256_file_func: Any,
) -> str:
    input_path = (
        _record_image_path(record, input_dir)
        if result_slot(record.method) == result_slot(VISION_IMAGE_ONLY_METHOD)
        else record.result.artifact_path
    )
    if input_path is None or not input_path.exists() or not input_path.is_file():
        return ""
    return str(sha256_file_func(input_path))


def _stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _route_decision(record: DbExtractionRecord) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "method": record.method,
        "status": record.result.status,
    }
    if record.threshold is not None:
        decision["threshold"] = record.threshold
    if record.reason:
        decision["reason"] = record.reason
    if record.name_hint is not None:
        decision["original_name_hint"] = record.name_hint.name
        decision["original_name_confidence"] = record.name_hint.confidence
    if record.image_path is not None:
        decision["source_image"] = str(record.image_path)
    return decision


def _db_record_checkpoint_metadata(record: DbExtractionRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": record.method,
        "provider": record.provider_name or None,
        "model": record.model_name or None,
        "route_reason": record.reason,
    }
    if record.result.artifact_path.suffix == ".txt":
        metadata["ocr_artifact"] = str(record.result.artifact_path)
        metadata["tsv_artifact"] = str(record.result.artifact_path.with_suffix(".tsv"))
    if record.image_path is not None:
        metadata["source_image"] = str(record.image_path)
        metadata["mime_type"] = image_mime_type(record.image_path)
    if record.name_hint is not None:
        metadata["original_name_hint"] = record.name_hint.name
        metadata["original_name_confidence"] = record.name_hint.confidence
    if record.threshold is not None:
        metadata["threshold"] = record.threshold
    if record.reason:
        metadata["reason"] = record.reason
    if record.estimated_tokens is not None:
        metadata["estimated_tokens"] = record.estimated_tokens
    if record.latency_ms is not None:
        metadata["latency_ms"] = record.latency_ms
    return metadata


def _load_completed_checkpoint_db_records(
    artifacts_dir: Path,
    checkpoint_path: Path | None,
    *,
    reuse_skipped_low_confidence: bool = True,
) -> dict[str, DbExtractionRecord]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    records: dict[str, DbExtractionRecord] = {}
    for record in _read_jsonl_records(checkpoint_path):
        filename = record.get("filename")
        status = record.get("status")
        if not isinstance(filename, str) or status not in CHECKPOINT_REUSE_STATUSES:
            continue
        if status == "skipped_low_confidence" and not reuse_skipped_low_confidence:
            continue
        row = record.get("row") if status in PROCESSED_STATUSES else None
        if row is not None and not isinstance(row, dict):
            continue
        method = str(record.get("method") or "")
        if not method:
            method = VISION_REROUTE_METHOD if status == "rerouted_processed" else TEXT_EXTRACTION_METHOD
        source_image = record.get("source_image")
        records[filename] = DbExtractionRecord(
            result=ExtractionResult(
                artifacts_dir / filename,
                str(status),
                row={key: str(value) for key, value in row.items()} if row is not None else None,
            ),
            method=method,
            name_hint=NameHint(
                str(record.get("original_name_hint") or ""),
                _optional_float(record.get("original_name_confidence")),
            ),
            threshold=_optional_float(record.get("threshold")),
            reason=str(record.get("reason") or record.get("route_reason") or ""),
            attempts=int(record.get("attempts") or 0),
            estimated_tokens=(
                int(record["estimated_tokens"]) if record.get("estimated_tokens") is not None else None
            ),
            latency_ms=int(record["latency_ms"]) if record.get("latency_ms") is not None else None,
            image_path=Path(source_image) if isinstance(source_image, str) and source_image else None,
            provider_name=str(record.get("provider") or ""),
            model_name=str(record.get("model") or ""),
        )
    return records


def _load_completed_image_checkpoint_db_records(
    input_dir: Path,
    checkpoint_path: Path | None,
) -> dict[str, DbExtractionRecord]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    records: dict[str, DbExtractionRecord] = {}
    for record in _read_jsonl_records(checkpoint_path):
        filename = record.get("filename")
        status = record.get("status")
        method = record.get("method")
        if (
            not isinstance(filename, str)
            or status != VISION_PROCESSED_STATUS
            or method != VISION_IMAGE_ONLY_METHOD
        ):
            continue
        row = record.get("row")
        if not isinstance(row, dict):
            continue
        image_path = input_dir / filename
        records[filename] = DbExtractionRecord(
            result=ExtractionResult(
                image_path,
                VISION_PROCESSED_STATUS,
                row={key: str(value) for key, value in row.items()},
            ),
            method=VISION_IMAGE_ONLY_METHOD,
            reason=str(record.get("reason") or record.get("route_reason") or "image_only"),
            attempts=int(record.get("attempts") or 0),
            estimated_tokens=(
                int(record["estimated_tokens"]) if record.get("estimated_tokens") is not None else None
            ),
            latency_ms=int(record["latency_ms"]) if record.get("latency_ms") is not None else None,
            image_path=image_path,
            provider_name=str(record.get("provider") or ""),
            model_name=str(record.get("model") or ""),
        )
    return records


async def _extract_artifact_db_payload(
    artifact_path: Path,
    provider: LlmProvider,
    *,
    name_hint: NameHint,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], str, int, int]:
    ocr_text = artifact_path.read_text(encoding="utf-8")
    filename = artifact_path.stem
    prompt = build_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    estimated_tokens = estimate_llm_tokens(prompt)
    attempts = 0
    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_complete(prompt)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), response_text, attempts, estimated_tokens
        except Exception as exc:
            if attempts > max_retries or not _is_transient_llm_error(exc):
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


async def _extract_artifact_with_vision_db_payload(
    artifact_path: Path,
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    name_hint: NameHint,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], str, int, int]:
    ocr_text = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for reroute: {image_path}")
    filename = artifact_path.stem
    prompt = build_vision_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    estimated_tokens = estimate_vision_llm_tokens(prompt, image_path)
    attempts = 0
    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_vision_complete(prompt, image_path, mime_type)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), response_text, attempts, estimated_tokens
        except Exception:
            if attempts > max_retries:
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


async def _extract_image_db_payload(
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], str, int, int]:
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for vision extraction: {image_path}")
    filename = image_path.stem
    prompt = build_image_only_vision_extraction_prompt(
        filename=filename,
        source=source,
    )
    estimated_tokens = estimate_vision_llm_tokens(prompt, image_path)
    attempts = 0
    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_vision_complete(prompt, image_path, mime_type)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), response_text, attempts, estimated_tokens
        except Exception:
            if attempts > max_retries:
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


def _basic_ocr_features(text: str, tsv_path: Path | None) -> dict[str, Any]:
    features: dict[str, Any] = {
        "ocr_char_count": len(text),
        "ocr_word_count": len(text.split()),
    }
    if tsv_path is not None and tsv_path.exists():
        features["tsv_line_count"] = len(tsv_path.read_text(encoding="utf-8").splitlines())
    return features


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _provider_name(provider: object) -> str:
    return str(getattr(provider, "provider_name", "") or "")


def _model_name(provider: object) -> str:
    return str(getattr(provider, "model_name", "") or "")


def _require_provider_model(provider: object) -> str:
    model = _model_name(provider).strip()
    if not model:
        raise ValueError("Model-backed extraction provider must expose a non-empty model_name.")
    return model


def _load_low_confidence_file_candidates(
    artifacts_dir: Path,
    low_confidence_file: Path,
    default_threshold: float,
) -> list[RerouteCandidate]:
    records = _read_jsonl_records(low_confidence_file)
    candidates: list[RerouteCandidate] = []
    for record in records:
        if record.get("status") != "skipped_low_confidence":
            continue
        filename = record.get("filename")
        if not isinstance(filename, str):
            continue
        confidence = _optional_float(record.get("confidence"))
        name_value = record.get("name", "")
        name = name_value if isinstance(name_value, str) else str(name_value or "")
        threshold = _optional_float(record.get("threshold")) or default_threshold
        reason = record.get("reason")
        candidates.append(
            RerouteCandidate(
                artifacts_dir / filename,
                NameHint(name, confidence),
                threshold,
                reason if isinstance(reason, str) else _low_confidence_reason(confidence),
            )
        )
    return candidates


def _load_results_file_candidates(
    artifacts_dir: Path,
    results_file: Path,
    threshold: float,
) -> list[RerouteCandidate]:
    name_map = load_name_map_if_exists(artifacts_dir)
    latest_by_filename: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl_records(results_file):
        filename = record.get("filename")
        if isinstance(filename, str):
            latest_by_filename[filename] = record

    candidates: list[RerouteCandidate] = []
    for filename, record in sorted(latest_by_filename.items()):
        if record.get("status") != "skipped_low_confidence":
            continue
        name_hint = name_map.get(filename, NameHint("", None))
        candidates.append(
            RerouteCandidate(
                artifacts_dir / filename,
                name_hint,
                threshold,
                _low_confidence_reason(name_hint.confidence),
            )
        )
    return candidates


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL file contains invalid JSON on line {line_number}: {path}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _low_confidence_reason(confidence: float | None) -> str:
    return "missing_confidence" if confidence is None else "low_confidence"


class AsyncLlmRateLimiter:
    def __init__(self, rpm_limit: int | None, tpm_limit: int | None) -> None:
        self._rpm_limit = rpm_limit if rpm_limit and rpm_limit > 0 else None
        self._tpm_limit = tpm_limit if tpm_limit and tpm_limit > 0 else None
        self._request_times: deque[float] = deque()
        self._token_records: deque[tuple[float, int]] = deque()
        self._token_total = 0
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        if self._rpm_limit is None and self._tpm_limit is None:
            return

        token_cost = max(1, estimated_tokens)
        if self._tpm_limit is not None:
            token_cost = min(token_cost, self._tpm_limit)

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                request_allowed = (
                    self._rpm_limit is None or len(self._request_times) < self._rpm_limit
                )
                token_allowed = (
                    self._tpm_limit is None
                    or self._token_total + token_cost <= self._tpm_limit
                )
                if request_allowed and token_allowed:
                    if self._rpm_limit is not None:
                        self._request_times.append(now)
                    if self._tpm_limit is not None:
                        self._token_records.append((now, token_cost))
                        self._token_total += token_cost
                    return

                wait_for = self._wait_seconds(now, token_cost)

            await asyncio.sleep(max(0.05, wait_for))

    def _prune(self, now: float) -> None:
        expires_before = now - 60.0
        while self._request_times and self._request_times[0] <= expires_before:
            self._request_times.popleft()
        while self._token_records and self._token_records[0][0] <= expires_before:
            _, tokens = self._token_records.popleft()
            self._token_total -= tokens

    def _wait_seconds(self, now: float, token_cost: int) -> float:
        waits: list[float] = []
        if self._rpm_limit is not None and len(self._request_times) >= self._rpm_limit:
            waits.append(60.0 - (now - self._request_times[0]))
        if (
            self._tpm_limit is not None
            and self._token_total + token_cost > self._tpm_limit
            and self._token_records
        ):
            waits.append(60.0 - (now - self._token_records[0][0]))
        return max(0.05, min(waits) if waits else 0.05)


async def _extract_artifact_async_with_retry(
    artifact_path: Path,
    provider: LlmProvider,
    *,
    name_hint: NameHint,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], int]:
    ocr_text = artifact_path.read_text(encoding="utf-8")
    filename = artifact_path.stem
    prompt = build_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    estimated_tokens = estimate_llm_tokens(prompt)
    attempts = 0

    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_complete(prompt)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), attempts
        except Exception as exc:
            if attempts > max_retries or not _is_transient_llm_error(exc):
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


async def _extract_artifact_with_vision_async_with_retry(
    artifact_path: Path,
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    name_hint: NameHint,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], int]:
    ocr_text = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for reroute: {image_path}")
    filename = artifact_path.stem
    prompt = build_vision_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        name_hint=name_hint,
    )
    estimated_tokens = estimate_vision_llm_tokens(prompt, image_path)
    attempts = 0

    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_vision_complete(prompt, image_path, mime_type)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), attempts
        except Exception:
            if attempts > max_retries:
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


async def _extract_image_with_vision_async_with_retry(
    image_path: Path,
    provider: VisionLlmProvider,
    *,
    source: str,
    limiter: AsyncLlmRateLimiter,
    max_retries: int,
) -> tuple[dict[str, str], int]:
    mime_type = image_mime_type(image_path)
    if mime_type is None:
        raise ValueError(f"Unsupported image type for vision extraction: {image_path}")
    filename = image_path.stem
    prompt = build_image_only_vision_extraction_prompt(
        filename=filename,
        source=source,
    )
    estimated_tokens = estimate_vision_llm_tokens(prompt, image_path)
    attempts = 0

    while True:
        attempts += 1
        await limiter.acquire(estimated_tokens)
        try:
            response_text = await provider.async_vision_complete(prompt, image_path, mime_type)
            data = parse_json_object(response_text)
            return normalize_record(data, filename=filename, source=source), attempts
        except Exception:
            if attempts > max_retries:
                raise
            await asyncio.sleep(_retry_delay_seconds(attempts))


def estimate_llm_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / 3) + 300


def estimate_vision_llm_tokens(prompt: str, image_path: Path) -> int:
    image_bytes = image_path.stat().st_size if image_path.exists() else 0
    return estimate_llm_tokens(prompt) + max(500, math.ceil(image_bytes / 256))


def _retry_delay_seconds(attempt: int) -> float:
    return min(60.0, 2 ** (attempt - 1)) + random.uniform(0, 0.5)


def _is_transient_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True

    error_name = type(exc).__name__.lower()
    transient_names = (
        "apiconnectionerror",
        "apitimesouterror",
        "ratelimiterror",
        "serviceunavailableerror",
        "timeout",
        "timeouterror",
    )
    if any(name in error_name for name in transient_names):
        return True

    return isinstance(exc, RuntimeError) and "response did not contain" in str(exc)


def _load_completed_checkpoint_results(
    artifacts_dir: Path,
    checkpoint_path: Path | None,
    *,
    reuse_skipped_low_confidence: bool = True,
) -> dict[str, ExtractionResult]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    results: dict[str, ExtractionResult] = {}
    for line_number, line in enumerate(checkpoint_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Checkpoint contains invalid JSON on line {line_number}: {checkpoint_path}"
            ) from exc

        filename = record.get("filename")
        status = record.get("status")
        if not isinstance(filename, str) or status not in CHECKPOINT_REUSE_STATUSES:
            continue
        if status == "skipped_low_confidence" and not reuse_skipped_low_confidence:
            continue
        row = record.get("row") if status in PROCESSED_STATUSES else None
        if row is not None and not isinstance(row, dict):
            continue
        results[filename] = ExtractionResult(
            artifacts_dir / filename,
            status,
            row={key: str(value) for key, value in row.items()} if row is not None else None,
        )
    return results


def _load_completed_image_checkpoint_results(
    input_dir: Path,
    checkpoint_path: Path | None,
) -> dict[str, ExtractionResult]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    results: dict[str, ExtractionResult] = {}
    for line_number, line in enumerate(checkpoint_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Checkpoint contains invalid JSON on line {line_number}: {checkpoint_path}"
            ) from exc

        filename = record.get("filename")
        status = record.get("status")
        method = record.get("method")
        if (
            not isinstance(filename, str)
            or status != VISION_PROCESSED_STATUS
            or method != VISION_IMAGE_ONLY_METHOD
        ):
            continue
        row = record.get("row")
        if not isinstance(row, dict):
            continue
        results[filename] = ExtractionResult(
            input_dir / filename,
            status,
            row={key: str(value) for key, value in row.items()},
        )
    return results


async def _append_checkpoint_record_async(
    checkpoint_path: Path | None,
    checkpoint_lock: asyncio.Lock,
    filename: str,
    result: ExtractionResult,
    attempts: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    if checkpoint_path is None:
        return

    record = {
        "filename": filename,
        "status": result.status,
        "attempts": attempts,
        "row": result.row,
        "error": result.error,
    }
    if metadata:
        record.update(metadata)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    async with checkpoint_lock:
        _append_line(checkpoint_path, line)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _checkpoint_metadata(
    *,
    method: str,
    provider: object,
    artifact_path: Path,
    name_hint: NameHint,
    threshold: float,
    source_image: Path | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": method,
        "provider": getattr(provider, "provider_name", None),
        "model": getattr(provider, "model_name", None),
        "ocr_artifact": str(artifact_path),
        "tsv_artifact": str(artifact_path.with_suffix(".tsv")),
        "original_name_hint": name_hint.name,
        "original_name_confidence": name_hint.confidence,
        "threshold": threshold,
    }
    if source_image is not None:
        metadata["source_image"] = str(source_image)
    if reason is not None:
        metadata["reason"] = reason
    elif _is_below_confidence_threshold(name_hint, threshold):
        metadata["reason"] = _low_confidence_reason(name_hint.confidence)
    return metadata


def _image_checkpoint_metadata(
    *,
    method: str,
    provider: object,
    image_path: Path,
) -> dict[str, Any]:
    return {
        "method": method,
        "provider": getattr(provider, "provider_name", None),
        "model": getattr(provider, "model_name", None),
        "source_image": str(image_path),
        "mime_type": image_mime_type(image_path),
    }


def _write_extraction_error(
    log_file: Path | None,
    filename: str,
    exc: Exception,
    *,
    method: str = TEXT_EXTRACTION_METHOD,
) -> None:
    if log_file is not None:
        action = "extraction" if method == TEXT_EXTRACTION_METHOD else method
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"ERROR: file {filename} failed {action}: {exc}\n")
