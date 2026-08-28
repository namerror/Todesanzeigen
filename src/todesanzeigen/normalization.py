from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from .llm import CSV_COLUMNS, STORED_COLUMNS


DATE_FIELDS = ("geburtsdatum", "sterbedatum")

_GERMAN_MONTHS = {
    "jan": 1,
    "januar": 1,
    "jaenner": 1,
    "janner": 1,
    "jänner": 1,
    "feb": 2,
    "februar": 2,
    "maerz": 3,
    "marz": 3,
    "mär": 3,
    "märz": 3,
    "mrz": 3,
    "muerz": 3,
    "murz": 3,
    "mürz": 3,
    "apr": 4,
    "april": 4,
    "mai": 5,
    "jun": 6,
    "juni": 6,
    "jul": 7,
    "juli": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "okt": 10,
    "oktober": 10,
    "nov": 11,
    "november": 11,
    "dez": 12,
    "dezember": 12,
}

_GERMAN_NUMERIC_DATE = re.compile(
    r"^(?P<day>\d{1,2})\s*[.,/-]\s*(?P<month>\d{1,2})\s*[.,/-]\s*(?P<year>\d{4})$"
)
_ISO_DATE = re.compile(
    r"^(?P<year>\d{4})\s*[-/]\s*(?P<month>\d{1,2})\s*[-/]\s*(?P<day>\d{1,2})$"
)
_GERMAN_TEXT_DATE = re.compile(
    r"^(?P<day>\d{1,2})[.,]?\s+(?P<month>[^\W\d_]+)\.?\s+(?P<year>\d{4})$",
    re.UNICODE,
)


@dataclass(frozen=True)
class FieldNormalizationReport:
    fields: dict[str, str]
    location_from_wohnort: bool = False
    location_conflict: bool = False
    normalized_dates: tuple[str, ...] = ()
    unresolved_dates: tuple[str, ...] = ()


def normalize_date(value: Any) -> str | None:
    raw = _stringify(value).strip()
    if not raw:
        return ""
    raw = raw.lstrip("+*†").strip()

    numeric_match = _GERMAN_NUMERIC_DATE.fullmatch(raw)
    if numeric_match is not None:
        return _validated_date(
            int(numeric_match["year"]),
            int(numeric_match["month"]),
            int(numeric_match["day"]),
        )

    iso_match = _ISO_DATE.fullmatch(raw)
    if iso_match is not None:
        return _validated_date(
            int(iso_match["year"]),
            int(iso_match["month"]),
            int(iso_match["day"]),
        )

    text_match = _GERMAN_TEXT_DATE.fullmatch(raw)
    if text_match is None:
        return None
    month_name = text_match["month"].casefold()
    month = _GERMAN_MONTHS.get(month_name)
    if month is None:
        return None
    return _validated_date(
        int(text_match["year"]),
        month,
        int(text_match["day"]),
    )


def normalize_stored_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    return normalize_stored_fields_with_report(fields).fields


def normalize_stored_fields_with_report(
    fields: Mapping[str, Any],
) -> FieldNormalizationReport:
    normalized = {column: _stringify(fields.get(column, "")) for column in STORED_COLUMNS}

    ort = normalized["ort"].strip()
    wohnort = _stringify(fields.get("wohnort", "")).strip()
    location_from_wohnort = not ort and bool(wohnort)
    location_conflict = bool(ort and wohnort and ort != wohnort)
    normalized["ort"] = ort or wohnort

    normalized_dates: list[str] = []
    unresolved_dates: list[str] = []
    for field in DATE_FIELDS:
        original = normalized[field].strip()
        canonical = normalize_date(original)
        if canonical is None:
            if original:
                unresolved_dates.append(field)
                normalized["bemerkungen"] = _append_date_note(
                    normalized["bemerkungen"],
                    field,
                    original,
                )
            normalized[field] = original
            continue
        normalized[field] = canonical
        if original and canonical != original:
            normalized_dates.append(field)

    return FieldNormalizationReport(
        fields=normalized,
        location_from_wohnort=location_from_wohnort,
        location_conflict=location_conflict,
        normalized_dates=tuple(normalized_dates),
        unresolved_dates=tuple(unresolved_dates),
    )


def fields_for_csv(
    fields: Mapping[str, Any],
    *,
    source: str = "",
    filename_stem: str = "",
) -> dict[str, str]:
    stored = normalize_stored_fields(fields)
    row = {column: stored.get(column, "") for column in CSV_COLUMNS}
    row["wohnort"] = stored["ort"]
    if not row["quelle"]:
        row["quelle"] = source
    if not row["dateiname"]:
        row["dateiname"] = filename_stem
    return row


def _validated_date(year: int, month: int, day: int) -> str | None:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return parsed.strftime("%d.%m.%Y")


def _append_date_note(existing: str, field: str, value: str) -> str:
    note = f"Nicht kanonisiertes {field} beibehalten: {value}."
    cleaned = existing.strip()
    if note in cleaned:
        return cleaned
    return f"{cleaned} {note}".strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
