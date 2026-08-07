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


LLM_PROVIDERS = ("gemini", "qwen")


def build_llm_provider(provider_name: str | None = None) -> LlmProvider:
    raw_provider = (
        provider_name
        if provider_name is not None
        else os.getenv("TODESANZEIGEN_LLM_PROVIDER", "gemini")
    )
    provider = raw_provider.strip().lower() or "gemini"
    if provider == "gemini":
        return GeminiProvider(GeminiSettings.from_env())
    if provider == "qwen":
        return QwenProvider(QwenSettings.from_env())
    raise ConfigError(f"Unsupported LLM provider: {provider}")


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


@dataclass(frozen=True)
class QwenSettings:
    api_key: str
    model: str = "qwen3.7-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def from_env(cls) -> "QwenSettings":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required Qwen environment variable: DASHSCOPE_API_KEY")

        model = os.getenv("QWEN_MODEL", "qwen3.7-plus").strip()
        base_url = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip()
        return cls(api_key=api_key, model=model, base_url=base_url)


class QwenProvider:
    def __init__(self, settings: QwenSettings) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._model = settings.model

    def complete(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = completion.choices[0].message.content
        if content:
            return content
        raise RuntimeError("Qwen response did not contain message content.")
