from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS, LlmProvider
from .ocr_filtering import name_map_artifact_path


@dataclass(frozen=True)
class ExtractionResult:
    artifact_path: Path
    status: str
    row: dict[str, str] | None = None


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


def _is_below_confidence_threshold(name_hint: NameHint, threshold: float) -> bool:
    return name_hint.confidence is None or name_hint.confidence < threshold


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
