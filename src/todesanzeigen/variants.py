from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .methods import METHOD_DEFINITIONS


DEFAULT_VARIANTS_CONFIG_PATH = Path("config/extraction_variants.toml")


@dataclass(frozen=True)
class ExtractionVariant:
    alias: str
    label: str
    method: str
    provider: str
    model: str
    prompt_version: str
    review_enabled: bool = False

    def as_dict(self) -> dict[str, str]:
        return {
            "alias": self.alias,
            "label": self.label,
            "method": self.method,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class ExtractionVariantConfig:
    variants: tuple[ExtractionVariant, ...]
    text_default: str
    vlm_default: str

    def variant(self, alias: str) -> ExtractionVariant:
        for variant in self.variants:
            if variant.alias == alias:
                return variant
        available = ", ".join(item.alias for item in self.variants)
        raise ValueError(f"Unknown extraction variant alias {alias!r}. Available: {available}")

    @property
    def review_variants(self) -> tuple[ExtractionVariant, ...]:
        return tuple(variant for variant in self.variants if variant.review_enabled)

    @property
    def operational_variants(self) -> tuple[ExtractionVariant, ExtractionVariant]:
        return self.variant(self.vlm_default), self.variant(self.text_default)


def load_variant_config(
    path: Path = DEFAULT_VARIANTS_CONFIG_PATH,
) -> ExtractionVariantConfig:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Extraction variants config does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid extraction variants config {path}: {exc}") from exc

    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError(f"Extraction variants config must define at least one [[variants]] entry: {path}")
    variants = tuple(_parse_variant(item, path) for item in raw_variants)
    aliases = [variant.alias for variant in variants]
    if len(aliases) != len(set(aliases)):
        raise ValueError(f"Extraction variant aliases must be unique: {path}")
    identities = [
        (variant.method, variant.provider, variant.model, variant.prompt_version)
        for variant in variants
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Extraction variant tuples must be unique: {path}")

    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError(f"Extraction variants config must define [defaults]: {path}")
    text_default = _required_text(defaults, "text", path)
    vlm_default = _required_text(defaults, "vlm", path)
    config = ExtractionVariantConfig(
        variants=variants,
        text_default=text_default,
        vlm_default=vlm_default,
    )
    text_variant = config.variant(text_default)
    vlm_variant = config.variant(vlm_default)
    if METHOD_DEFINITIONS[text_variant.method].method_family != "ocr_llm":
        raise ValueError(f"Default text variant {text_default!r} must use an OCR+LLM method")
    if METHOD_DEFINITIONS[vlm_variant.method].method_family != "vlm":
        raise ValueError(f"Default VLM variant {vlm_default!r} must use a VLM method")
    return config


def _parse_variant(data: Any, path: Path) -> ExtractionVariant:
    if not isinstance(data, dict):
        raise ValueError(f"Each [[variants]] entry must be a table: {path}")
    method = _required_text(data, "method", path)
    if method not in METHOD_DEFINITIONS:
        raise ValueError(f"Unknown extraction method {method!r} in {path}")
    review_enabled = data.get("review_enabled", False)
    if not isinstance(review_enabled, bool):
        raise ValueError(f"Extraction variant review_enabled must be true or false: {path}")
    return ExtractionVariant(
        alias=_required_text(data, "alias", path),
        label=_required_text(data, "label", path),
        method=method,
        provider=_required_text(data, "provider", path),
        model=_required_text(data, "model", path),
        prompt_version=_required_text(data, "prompt_version", path),
        review_enabled=review_enabled,
    )


def _required_text(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Extraction variants config requires non-empty {key!r}: {path}")
    return value.strip()
