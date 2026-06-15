from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .ocr import ConfigError


CSV_COLUMNS = [
    "geschlecht",
    "name",
    "vorname",
    "foto",
    "geburtsdatum",
    "sterbedatum",
    "geburtsname",
    "titel",
    "genannt",
    "geburtsort",
    "sterbeort",
    "wohnort",
    "ort",
    "weitere_orte",
    "beruf",
    "bemerkungen",
    "quelle",
    "dateiname",
    "zusaetzliche_hinweise",
    "confidence_score",
]


class LlmProvider(Protocol):
    def complete(self, prompt: str) -> str:
        """Return the model response body for one prompt."""


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str = "gemini-2.0-flash-lite"

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required Gemini environment variable: GEMINI_API_KEY")

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
        return cls(api_key=api_key, model=model)


class GeminiProvider:
    def __init__(self, settings: GeminiSettings) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=settings.api_key)
        self._model = settings.model
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config,
        )
        text = getattr(response, "text", None)
        if text:
            return text
        raise RuntimeError("Gemini response did not contain text.")
