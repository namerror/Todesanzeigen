from __future__ import annotations

import asyncio
import csv
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS, LlmProvider
from .ocr_filtering import is_below_confidence_threshold, name_map_artifact_path


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


DEFAULT_NAME_CONFIDENCE_THRESHOLD = 85.0
LLM_EXCLUDED_COLUMNS = {"foto", "bemerkungen", "quelle", "dateiname"}
LLM_PROMPT_COLUMNS = [column for column in CSV_COLUMNS if column not in LLM_EXCLUDED_COLUMNS]


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
- Unsicherheiten, OCR-Probleme und Interpretationshinweise kommen in zusaetzliche_hinweise.
- Erfinde keine Daten.
- Nutze das lokale OCR-Name-Signal als Zusatzhinweis fuer den Namen der verstorbenen Person.
- Verwende das lokale OCR-Name-Signal nur als Hinweis; bei Widerspruch zaehlt der erkennbare OCR-Text.

Lokales OCR-Name-Signal:
{name_signal}

OCR-Text:
{ocr_text}
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
    row: dict[str, str] = {}
    for column in CSV_COLUMNS:
        value = data.get(column, "")
        if value is None:
            value = ""
        elif isinstance(value, list):
            value = "; ".join(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            value = str(value)
        row[column] = value

    row["foto"] = ""
    row["bemerkungen"] = ""
    row["quelle"] = source
    row["dateiname"] = filename
    return row


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
            writer.writerow(row)
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
) -> list[ExtractionResult]:
    settings = settings or AsyncExtractionSettings()
    if settings.concurrency < 1:
        raise ValueError("LLM extraction concurrency must be at least 1.")
    if settings.max_retries < 0:
        raise ValueError("LLM max retries must be at least 0.")

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

    resumed_results = _load_completed_checkpoint_results(artifacts_dir, resume_from)
    checkpoint_path = checkpoint_file or resume_from
    if checkpoint_path is None and log_file is not None:
        checkpoint_path = log_file.with_suffix(".results.jsonl")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()

    results_by_name: dict[str, ExtractionResult] = {}
    limiter = AsyncLlmRateLimiter(settings.rpm_limit, settings.tpm_limit)
    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.concurrency)

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
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    artifact_path.name,
                    result,
                    attempts,
                )
                return result
            except Exception as exc:
                result = ExtractionResult(artifact_path, "failed", error=str(exc))
                _write_extraction_error(log_file, artifact_path.name, exc)
                await _append_checkpoint_record_async(
                    checkpoint_path,
                    checkpoint_lock,
                    artifact_path.name,
                    result,
                    settings.max_retries + 1,
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
            result = ExtractionResult(artifact_path, "skipped_low_confidence")
            results_by_name[artifact_path.name] = result
            await _append_checkpoint_record_async(
                checkpoint_path,
                checkpoint_lock,
                artifact_path.name,
                result,
                0,
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
            if result.status == "processed" and result.row is not None:
                writer.writerow(result.row)

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


def estimate_llm_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / 3) + 300


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
        if not isinstance(filename, str) or status not in {"processed", "skipped_low_confidence"}:
            continue
        row = record.get("row") if status == "processed" else None
        if row is not None and not isinstance(row, dict):
            continue
        results[filename] = ExtractionResult(
            artifacts_dir / filename,
            status,
            row={key: str(value) for key, value in row.items()} if row is not None else None,
        )
    return results


async def _append_checkpoint_record_async(
    checkpoint_path: Path | None,
    checkpoint_lock: asyncio.Lock,
    filename: str,
    result: ExtractionResult,
    attempts: int,
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
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    async with checkpoint_lock:
        _append_line(checkpoint_path, line)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _write_extraction_error(log_file: Path | None, filename: str, exc: Exception) -> None:
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"ERROR: file {filename} failed extraction: {exc}\n")
