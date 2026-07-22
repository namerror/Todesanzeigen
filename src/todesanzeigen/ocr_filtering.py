from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OcrWord:
    page_num: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    left: int
    top: int
    width: int
    height: int
    confidence: float | None
    text: str


@dataclass(frozen=True)
class OcrLine:
    page_num: int
    block_num: int
    par_num: int
    line_num: int
    left: int
    top: int
    width: int
    height: int
    words: list[OcrWord]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


@dataclass(frozen=True)
class NameFilterResult:
    artifact_path: Path
    name: str
    confidence: float | None


NAME_MAP_FILENAME = "name_map.json"
_WORD_EDGE_CHARS = "\"'`´‘’‚“”„()[]{}<>|/\\:;,_~=+*#%!?«»"
_DATE_OR_NOTICE_PATTERN = re.compile(
    r"(\d|veröffentlicht|nachrichten|januar|februar|märz|maerz|april|mai|juni|"
    r"juli|august|september|oktober|november|dezember|rosenkranz|"
    r"trauergottesdienst|beerdigung|urnenbeisetzung|iban|spende|uhr)",
    re.IGNORECASE,
)
_CONTEXT_WORDS = {
    "abschied",
    "aichach",
    "angehörigen",
    "dankbarkeit",
    "erinnerung",
    "geb",
    "geb.",
    "herzen",
    "liebe",
    "lieber",
    "liebevoller",
    "mutter",
    "nachrichten",
    "oma",
    "papa",
    "trauer",
    "uroma",
    "vater",
    "veröffentlicht",
    "wir",
}
_TITLE_NORMALIZATIONS = {
    "dipl": "Dipl.",
    "ing": "Ing.",
    "dr": "Dr.",
    "prof": "Prof.",
}
_NAME_PARTICLES = {"von", "van", "der", "den", "de", "del", "da", "zu", "zur"}


def discover_tsv_artifacts(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Artifacts path is not a directory: {artifacts_dir}")

    return sorted(
        path for path in artifacts_dir.iterdir() if path.is_file() and path.suffix == ".tsv"
    )


def name_map_artifact_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / NAME_MAP_FILENAME


def build_name_map(results: list[NameFilterResult]) -> dict[str, dict[str, Any]]:
    return {
        result.artifact_path.with_suffix(".txt").name: {
            "name": result.name,
            "confidence": result.confidence,
        }
        for result in results
    }


def write_name_map(
    results: list[NameFilterResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_name_map(results), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def filter_artifact_names(
    artifacts_dir: Path,
    *,
    limit: int | None = None,
    write_map: bool = True,
    map_path: Path | None = None,
) -> list[NameFilterResult]:
    artifact_paths = discover_tsv_artifacts(artifacts_dir)
    if limit is not None:
        artifact_paths = artifact_paths[:limit]

    results: list[NameFilterResult] = []
    for artifact_path in artifact_paths:
        detected_name = detect_name_from_tsv(artifact_path.read_text(encoding="utf-8"))
        results.append(
            NameFilterResult(
                artifact_path=artifact_path,
                name=detected_name.name,
                confidence=detected_name.confidence,
            )
        )
    if write_map:
        write_name_map(results, map_path or name_map_artifact_path(artifacts_dir))
    return results


@dataclass(frozen=True)
class DetectedName:
    name: str
    confidence: float | None


def detect_name_from_tsv(tsv_text: str) -> DetectedName:
    lines = parse_tesseract_word_lines(tsv_text)
    candidates = [
        (_score_name_line(line), line)
        for line in lines
        if _clean_name_text(line.text) and _is_candidate_line(line)
    ]
    if not candidates:
        return DetectedName(name="", confidence=None)

    _, best_line = max(candidates, key=lambda candidate: candidate[0])
    return DetectedName(
        name=_clean_name_text(best_line.text),
        confidence=_cleaned_name_confidence(best_line),
    )


def parse_tesseract_word_lines(tsv_text: str) -> list[OcrLine]:
    words_by_line: dict[tuple[int, int, int, int], list[OcrWord]] = {}
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)

    for row in reader:
        if _int_field(row, "level") != 5:
            continue

        text = row.get("text", "").strip()
        if not text or not _has_alpha(text):
            continue

        page_num = _int_field(row, "page_num")
        block_num = _int_field(row, "block_num")
        par_num = _int_field(row, "par_num")
        line_num = _int_field(row, "line_num")
        word_num = _int_field(row, "word_num")
        left = _int_field(row, "left")
        top = _int_field(row, "top")
        width = _int_field(row, "width")
        height = _int_field(row, "height")
        if None in (
            page_num,
            block_num,
            par_num,
            line_num,
            word_num,
            left,
            top,
            width,
            height,
        ):
            continue
        if width <= 0 or height <= 0:
            continue

        word = OcrWord(
            page_num=page_num,
            block_num=block_num,
            par_num=par_num,
            line_num=line_num,
            word_num=word_num,
            left=left,
            top=top,
            width=width,
            height=height,
            confidence=_float_field(row, "conf"),
            text=text,
        )
        words_by_line.setdefault((page_num, block_num, par_num, line_num), []).append(word)

    lines: list[OcrLine] = []
    for (page_num, block_num, par_num, line_num), words in sorted(words_by_line.items()):
        sorted_words = sorted(words, key=lambda word: word.word_num)
        min_left = min(word.left for word in sorted_words)
        min_top = min(word.top for word in sorted_words)
        max_right = max(word.left + word.width for word in sorted_words)
        max_bottom = max(word.top + word.height for word in sorted_words)
        lines.append(
            OcrLine(
                page_num=page_num,
                block_num=block_num,
                par_num=par_num,
                line_num=line_num,
                left=min_left,
                top=min_top,
                width=max_right - min_left,
                height=max_bottom - min_top,
                words=sorted_words,
            )
        )
    return lines


def _is_candidate_line(line: OcrLine) -> bool:
    cleaned = _clean_name_text(line.text)
    if not cleaned or _DATE_OR_NOTICE_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split()
    if not 1 <= len(tokens) <= 5:
        return False
    if _name_token_count(tokens) == 0:
        return False
    if len(tokens) == 1 and tokens[0].lower().rstrip(".") in _CONTEXT_WORDS:
        return False

    return True


def _score_name_line(line: OcrLine) -> float:
    cleaned = _clean_name_text(line.text)
    tokens = cleaned.split()
    alpha_chars = sum(1 for char in cleaned if char.isalpha()) or 1
    total_word_area = sum(word.width * word.height for word in line.words)
    word_heights = sorted(word.height for word in line.words)
    median_height = word_heights[len(word_heights) // 2]
    average_char_area = total_word_area / alpha_chars
    token_bonus = 1.25 if _name_token_count(tokens) >= 2 else 0.65
    confidence_bonus = _average_confidence(line.words) / 100

    return median_height * average_char_area * token_bonus * (1 + confidence_bonus * 0.1)


def _clean_name_text(text: str) -> str:
    tokens: list[str] = []
    for raw_token in text.split():
        token = raw_token.strip(_WORD_EDGE_CHARS)
        token = token.replace("‚", "").replace("„", "")
        if not _has_alpha(token):
            continue

        normalized = token.rstrip(".").lower()
        if normalized in _TITLE_NORMALIZATIONS:
            tokens.append(_TITLE_NORMALIZATIONS[normalized])
        else:
            tokens.append(token)

    while tokens and tokens[0].lower().rstrip(".") in _CONTEXT_WORDS:
        tokens.pop(0)
    while tokens and _is_trailing_noise_token(tokens[-1], tokens):
        tokens.pop()

    return " ".join(tokens)


def _is_trailing_noise_token(token: str, tokens: list[str]) -> bool:
    if len(tokens) < 3:
        return False
    normalized = token.rstrip(".")
    if normalized.lower() in _NAME_PARTICLES:
        return False
    if normalized.lower() in _TITLE_NORMALIZATIONS:
        return False
    if len(normalized) <= 2 and normalized[:1].isupper():
        return True
    return False


def _name_token_count(tokens: list[str]) -> int:
    count = 0
    for token in tokens:
        normalized = token.rstrip(".")
        lower = normalized.lower()
        if lower in _TITLE_NORMALIZATIONS or lower in _NAME_PARTICLES:
            continue
        if lower in _CONTEXT_WORDS:
            continue
        if normalized[:1].isupper() and _has_alpha(normalized):
            count += 1
    return count


def _average_confidence(words: list[OcrWord]) -> float:
    confidences = [
        word.confidence for word in words if word.confidence is not None and word.confidence >= 0
    ]
    if not confidences:
        return 0
    return sum(confidences) / len(confidences)


def _cleaned_name_confidence(line: OcrLine) -> float | None:
    kept_words = _cleaned_name_words(line.words)
    confidences = [
        word.confidence
        for word in kept_words
        if word.confidence is not None and word.confidence >= 0
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _cleaned_name_words(words: list[OcrWord]) -> list[OcrWord]:
    cleaned_words: list[OcrWord] = []
    for word in words:
        token = word.text.strip(_WORD_EDGE_CHARS).replace("‚", "").replace("„", "")
        if _has_alpha(token):
            cleaned_words.append(word)

    while cleaned_words:
        token = cleaned_words[0].text.strip(_WORD_EDGE_CHARS).lower().rstrip(".")
        if token not in _CONTEXT_WORDS:
            break
        cleaned_words.pop(0)
    while cleaned_words:
        tokens = [
            word.text.strip(_WORD_EDGE_CHARS).replace("‚", "").replace("„", "")
            for word in cleaned_words
        ]
        if not _is_trailing_noise_token(tokens[-1], tokens):
            break
        cleaned_words.pop()

    return cleaned_words


def _has_alpha(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _int_field(row: dict[str, str], field: str) -> int | None:
    try:
        return int(row.get(field, ""))
    except (TypeError, ValueError):
        return None


def _float_field(row: dict[str, str], field: str) -> float | None:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
