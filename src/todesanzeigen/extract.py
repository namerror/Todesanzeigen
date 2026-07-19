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


@dataclass(frozen=True)
class LayoutLine:
    block_num: int
    par_num: int
    line_num: int
    left: int
    top: int
    width: int
    height: int
    avg_confidence: float | None
    text: str


@dataclass(frozen=True)
class LayoutBlock:
    block_num: int
    left: int
    top: int
    width: int
    height: int
    lines: list[LayoutLine]


def discover_artifacts(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Artifacts path is not a directory: {artifacts_dir}")

    return sorted(path for path in artifacts_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def layout_artifact_path_for_text_artifact(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".tsv")


def _int_field(row: dict[str, str], field: str) -> int | None:
    try:
        return int(row.get(field, ""))
    except ValueError:
        return None


def _float_field(row: dict[str, str], field: str) -> float | None:
    try:
        return float(row.get(field, ""))
    except ValueError:
        return None


def _bbox_from_row(row: dict[str, str]) -> tuple[int, int, int, int] | None:
    left = _int_field(row, "left")
    top = _int_field(row, "top")
    width = _int_field(row, "width")
    height = _int_field(row, "height")
    if left is None or top is None or width is None or height is None:
        return None
    return left, top, width, height


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    min_left = min(left for left, _, _, _ in boxes)
    min_top = min(top for _, top, _, _ in boxes)
    max_right = max(left + width for left, _, width, _ in boxes)
    max_bottom = max(top + height for _, top, _, height in boxes)
    return min_left, min_top, max_right - min_left, max_bottom - min_top


def parse_tesseract_tsv(tsv_text: str) -> list[LayoutBlock]:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    block_boxes: dict[int, tuple[int, int, int, int]] = {}
    line_boxes: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    line_words: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    for row in reader:
        level = _int_field(row, "level")
        block_num = _int_field(row, "block_num")
        par_num = _int_field(row, "par_num")
        line_num = _int_field(row, "line_num")
        if level is None or block_num is None:
            continue

        bbox = _bbox_from_row(row)
        if level == 2 and bbox is not None:
            block_boxes[block_num] = bbox
        elif level == 4 and par_num is not None and line_num is not None and bbox is not None:
            line_boxes[(block_num, par_num, line_num)] = bbox
        elif level == 5 and par_num is not None and line_num is not None:
            text = row.get("text", "").strip()
            if not text:
                continue
            word_bbox = _bbox_from_row(row)
            if word_bbox is None:
                continue
            line_words.setdefault((block_num, par_num, line_num), []).append(
                {
                    "text": text,
                    "bbox": word_bbox,
                    "confidence": _float_field(row, "conf"),
                }
            )

    lines_by_block: dict[int, list[LayoutLine]] = {}
    for (block_num, par_num, line_num), words in sorted(line_words.items()):
        word_boxes = [word["bbox"] for word in words]
        left, top, width, height = line_boxes.get(
            (block_num, par_num, line_num),
            _union_bbox(word_boxes),
        )
        confidences = [
            word["confidence"]
            for word in words
            if word["confidence"] is not None and word["confidence"] >= 0
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        lines_by_block.setdefault(block_num, []).append(
            LayoutLine(
                block_num=block_num,
                par_num=par_num,
                line_num=line_num,
                left=left,
                top=top,
                width=width,
                height=height,
                avg_confidence=avg_confidence,
                text=" ".join(str(word["text"]) for word in words),
            )
        )

    blocks: list[LayoutBlock] = []
    for block_num, lines in sorted(lines_by_block.items()):
        line_boxes_for_block = [
            (line.left, line.top, line.width, line.height) for line in lines
        ]
        left, top, width, height = block_boxes.get(
            block_num,
            _union_bbox(line_boxes_for_block),
        )
        blocks.append(
            LayoutBlock(
                block_num=block_num,
                left=left,
                top=top,
                width=width,
                height=height,
                lines=lines,
            )
        )
    return blocks


def format_layout_for_prompt(blocks: list[LayoutBlock]) -> str:
    if not blocks:
        return "Keine verwertbaren TSV-Layoutzeilen erkannt."

    output: list[str] = []
    for block in blocks:
        output.append(
            f"Block {block.block_num} bbox=({block.left},{block.top},{block.width},{block.height})"
        )
        for line in block.lines:
            confidence = (
                f"{line.avg_confidence:.1f}" if line.avg_confidence is not None else ""
            )
            output.append(
                "  "
                f"Line b{line.block_num}.p{line.par_num}.l{line.line_num} "
                f"bbox=({line.left},{line.top},{line.width},{line.height}) "
                f"height={line.height} avg_conf={confidence}: {line.text}"
            )
    return "\n".join(output)


def build_extraction_prompt(
    ocr_text: str,
    *,
    filename: str,
    source: str = "",
    layout_text: str = "",
) -> str:
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
- Nutze das OCR-Layout als Zusatzsignal: groessere oder zentrierte Zeilen sind oft Name oder Ueberschrift.
- Verwende Layoutsignale nur als Hinweis; bei Widerspruch zaehlt der erkennbare Text.

OCR-Text:
{ocr_text}

OCR-Layout aus TSV:
{layout_text}
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
    layout_path = layout_artifact_path_for_text_artifact(artifact_path)
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Missing TSV layout artifact for {artifact_path}: expected {layout_path}"
        )
    layout_blocks = parse_tesseract_tsv(layout_path.read_text(encoding="utf-8"))
    layout_text = format_layout_for_prompt(layout_blocks)
    filename = artifact_path.stem
    prompt = build_extraction_prompt(
        ocr_text,
        filename=filename,
        source=source,
        layout_text=layout_text,
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
