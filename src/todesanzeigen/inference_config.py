"""Serializable local inference settings; importing this module never imports torch."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class InferenceConfig:
    revision: str
    weight_format: str = "fp16"
    compute_dtype: str = "float16"
    use_cache: bool = True
    attention_backend: str = "sdpa"
    min_pixels: int = 256 * 32 * 32
    max_pixels: int = 1280 * 32 * 32
    max_new_tokens: int = 512
    seed: int = 42
    do_sample: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Local inference revision must be an immutable 40-character commit SHA")
        if self.weight_format not in {"fp16", "bf16", "int8", "nf4"}:
            raise ValueError("weight_format must be fp16, bf16, int8, or nf4")
        if self.compute_dtype not in {"float16", "bfloat16"}:
            raise ValueError("compute_dtype must be float16 or bfloat16")
        if self.weight_format in {"fp16", "bf16"} and self.compute_dtype != {"fp16": "float16", "bf16": "bfloat16"}[self.weight_format]:
            raise ValueError("Unquantized weight format and compute_dtype must match")
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be sdpa or eager")
        if type(self.use_cache) is not bool or self.do_sample is not False:
            raise ValueError("use_cache must be boolean; v1 requires greedy generation (do_sample=false)")
        for name in ("min_pixels", "max_pixels", "max_new_tokens", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 < self.min_pixels <= self.max_pixels or self.max_new_tokens < 1 or self.seed < 0:
            raise ValueError("Invalid pixel limits, output limit, or seed")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_hash(self) -> str:
        return fingerprint(self.as_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceConfig:
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Invalid local inference configuration: {exc}") from exc
