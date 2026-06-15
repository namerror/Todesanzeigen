from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS, LlmProvider


@dataclass(frozen=True)
class ExtractionResult:
    artifact_path: Path
    status: str
    row: dict[str, str] | None = None


def discover_artifacts(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Artifacts path is not a directory: {artifacts_dir}")

    return sorted(path for path in artifacts_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def build_extraction_prompt(ocr_text: str, *, filename: str, source: str = "") -> str:
    columns = ", ".join(CSV_COLUMNS)
    return f"""Du extrahierst strukturierte Daten aus OCR-Texten deutscher Todesanzeigen.

Gib ausschliesslich ein einzelnes valides JSON-Objekt zurueck. Verwende exakt diese Keys:
{columns}

Regeln:
- dateiname muss "{filename}" sein.
- quelle muss "{source}" sein, falls keine Quelle im Text erkennbar ist.
- foto ist "ja", "nein" oder "".
- confidence_score ist eine Zahl von 0 bis 1 als String, z.B. "0.82".
- Fehlende oder unsichere Felder bleiben "".
- Unsicherheiten, OCR-Probleme und Interpretationshinweise kommen in zusaetzliche_hinweise.
- Erfinde keine Daten.

OCR-Text:
{ocr_text}
"""


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

    row["dateiname"] = filename
    if not row["quelle"] and source:
        row["quelle"] = source
    return row


def extract_artifact(
    artifact_path: Path,
    provider: LlmProvider,
    *,
    source: str = "",
) -> dict[str, str]:
    ocr_text = artifact_path.read_text(encoding="utf-8")
    filename = artifact_path.stem
    prompt = build_extraction_prompt(ocr_text, filename=filename, source=source)
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
) -> list[ExtractionResult]:
    artifacts = discover_artifacts(artifacts_dir)
    if limit is not None:
        artifacts = artifacts[:limit]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[ExtractionResult] = []
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for artifact_path in artifacts:
            row = extract_artifact(artifact_path, provider, source=source)
            writer.writerow(row)
            results.append(ExtractionResult(artifact_path, "processed", row))

    return results
