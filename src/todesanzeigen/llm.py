from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
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

STORED_COLUMNS = [column for column in CSV_COLUMNS if column != "wohnort"]

# Model-facing extraction contract. ``nachname`` is intentionally translated to
# the legacy ``name`` field at the persistence boundary in extract.py.
MODEL_RESPONSE_COLUMNS = [
    "geschlecht",
    "nachname",
    "vorname",
    "geburtsdatum",
    "sterbedatum",
    "geburtsname",
    "titel",
    "genannt",
    "geburtsort",
    "sterbeort",
    "ort",
    "weitere_orte",
    "beruf",
    "zusaetzliche_hinweise",
    "confidence_score",
]

_MODEL_FIELD_DESCRIPTIONS = {
    "geschlecht": 'Exakt "männlich", "weiblich" oder leer.',
    "nachname": "Aktueller Nachname der verstorbenen Person, ohne Vorname, Titel oder Geburtsname.",
    "vorname": "Vorname beziehungsweise Vornamen der verstorbenen Person.",
    "geburtsdatum": 'Explizites Geburtsdatum im Format DD.MM.YYYY oder leer.',
    "sterbedatum": 'Explizites Sterbedatum im Format DD.MM.YYYY oder leer.',
    "geburtsname": "Nur ein ausdrücklich genannter Geburts- oder Mädchenname der verstorbenen Person.",
    "titel": "Akademischer, adliger oder sonstiger Titel der verstorbenen Person.",
    "genannt": "Nur ein ausdrücklich genannter Rufname, Spitzname oder Alias der verstorbenen Person.",
    "geburtsort": "Nur ein ausdrücklich als Geburtsort genannter Ort.",
    "sterbeort": "Nur ein ausdrücklich als Sterbeort genannter Ort.",
    "ort": "Erster allgemeiner Ort aus einer Ortszeile zur verstorbenen Person; kein Veranstaltungs-, Kirchen-, Friedhofs-, Bestattungs-, Geburts- oder Sterbeort.",
    "weitere_orte": "Weitere allgemeine Orte derselben Art, kommasepariert; keine Veranstaltungs-, Kirchen-, Friedhofs-, Bestattungs-, Geburts- oder Sterbeorte.",
    "beruf": "Nur der Beruf der verstorbenen Person, keine Verwandtschaftsrolle.",
    "zusaetzliche_hinweise": "Kurzer Hinweis auf relevante Unklarheiten oder schwer lesbare Stellen.",
    "confidence_score": 'Gesamtkonfidenz als Dezimalzahl von 0 bis 1 im Stringformat, zum Beispiel "0.82".',
}

QWEN_VISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "death_notice_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                field: {
                    "type": "string",
                    "description": _MODEL_FIELD_DESCRIPTIONS[field],
                    **(
                        {"enum": ["", "männlich", "weiblich"]}
                        if field == "geschlecht"
                        else {}
                    ),
                }
                for field in MODEL_RESPONSE_COLUMNS
            },
            "required": MODEL_RESPONSE_COLUMNS,
            "additionalProperties": False,
        },
    },
}


class LlmProvider(Protocol):
    provider_name: str
    model_name: str

    def complete(self, prompt: str) -> str:
        """Return the model response body for one prompt."""

    async def async_complete(self, prompt: str) -> str:
        """Return the model response body for one prompt from async extraction."""


class VisionLlmProvider(Protocol):
    provider_name: str
    model_name: str

    def vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        """Return the model response body for one image-backed prompt."""

    async def async_vision_complete(
        self,
        prompt: str,
        image_path: Path,
        mime_type: str,
    ) -> str:
        """Return the model response body for one async image-backed prompt."""


LLM_PROVIDERS = ("gemini", "qwen")
VISION_LLM_PROVIDERS = ("qwen", "gemini", "openai")


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


def build_vision_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> VisionLlmProvider:
    raw_provider = (
        provider_name
        if provider_name is not None
        else os.getenv("TODESANZEIGEN_REROUTE_PROVIDER", "qwen")
    )
    provider = raw_provider.strip().lower() or "qwen"
    if provider == "qwen":
        settings = QwenVisionSettings.from_env()
        if model_name:
            settings = QwenVisionSettings(
                api_key=settings.api_key,
                model=model_name.strip(),
                base_url=settings.base_url,
            )
        return QwenVisionProvider(settings)
    if provider == "gemini":
        settings = GeminiVisionSettings.from_env()
        if model_name:
            settings = GeminiVisionSettings(api_key=settings.api_key, model=model_name.strip())
        return GeminiVisionProvider(settings)
    if provider == "openai":
        settings = OpenAiVisionSettings.from_env()
        if model_name:
            settings = OpenAiVisionSettings(api_key=settings.api_key, model=model_name.strip())
        return OpenAiVisionProvider(settings)
    raise ConfigError(f"Unsupported vision LLM provider: {provider}")


def _image_data_url(image_path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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

        self.provider_name = "gemini"
        self.model_name = settings.model
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

    async def async_complete(self, prompt: str) -> str:
        return self.complete(prompt)


@dataclass(frozen=True)
class GeminiVisionSettings:
    api_key: str
    model: str = "gemini-2.5-pro"

    @classmethod
    def from_env(cls) -> "GeminiVisionSettings":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required Gemini environment variable: GEMINI_API_KEY")

        model = os.getenv(
            "GEMINI_VISION_MODEL",
            os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        ).strip()
        return cls(api_key=api_key, model=model)


class GeminiVisionProvider:
    def __init__(self, settings: GeminiVisionSettings) -> None:
        from google import genai
        from google.genai import types

        self.provider_name = "gemini"
        self.model_name = settings.model
        self._client = genai.Client(api_key=settings.api_key)
        self._model = settings.model
        self._types = types
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        )

    def vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        image_part = self._types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=mime_type,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[image_part, prompt],
            config=self._config,
        )
        text = getattr(response, "text", None)
        if text:
            return text
        raise RuntimeError("Gemini vision response did not contain text.")

    async def async_vision_complete(
        self,
        prompt: str,
        image_path: Path,
        mime_type: str,
    ) -> str:
        return self.vision_complete(prompt, image_path, mime_type)


@dataclass(frozen=True)
class QwenSettings:
    api_key: str
    model: str = "qwen3.6-flash"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def from_env(cls) -> "QwenSettings":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required Qwen environment variable: DASHSCOPE_API_KEY")

        model = os.getenv("QWEN_MODEL", "qwen3.6-flash").strip()
        base_url = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip()
        return cls(api_key=api_key, model=model, base_url=base_url)


class QwenProvider:
    def __init__(self, settings: QwenSettings) -> None:
        from openai import AsyncOpenAI, OpenAI

        self.provider_name = "qwen"
        self.model_name = settings.model
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._async_client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
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

    async def async_complete(self, prompt: str) -> str:
        completion = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = completion.choices[0].message.content
        if content:
            return content
        raise RuntimeError("Qwen response did not contain message content.")


@dataclass(frozen=True)
class QwenVisionSettings:
    api_key: str
    model: str = "qwen3.7-plus-2026-05-26"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def from_env(cls) -> "QwenVisionSettings":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required Qwen environment variable: DASHSCOPE_API_KEY")

        model = os.getenv("QWEN_VISION_MODEL", "qwen3.7-plus-2026-05-26").strip()
        base_url = os.getenv(
            "QWEN_VISION_BASE_URL",
            os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ).strip()
        return cls(api_key=api_key, model=model, base_url=base_url)


class QwenVisionProvider:
    def __init__(self, settings: QwenVisionSettings) -> None:
        from openai import AsyncOpenAI, OpenAI

        self.provider_name = "qwen"
        self.model_name = settings.model
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._async_client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._model = settings.model

    def vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[_vision_chat_message(prompt, image_path, mime_type)],
            response_format=QWEN_VISION_RESPONSE_FORMAT,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        content = completion.choices[0].message.content
        if content:
            return content
        raise RuntimeError("Qwen vision response did not contain message content.")

    async def async_vision_complete(
        self,
        prompt: str,
        image_path: Path,
        mime_type: str,
    ) -> str:
        completion = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[_vision_chat_message(prompt, image_path, mime_type)],
            response_format=QWEN_VISION_RESPONSE_FORMAT,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        content = completion.choices[0].message.content
        if content:
            return content
        raise RuntimeError("Qwen vision response did not contain message content.")


def _vision_chat_message(prompt: str, image_path: Path, mime_type: str) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path, mime_type)},
            },
        ],
    }


@dataclass(frozen=True)
class OpenAiVisionSettings:
    api_key: str
    model: str = "gpt-5.6-luna"

    @classmethod
    def from_env(cls) -> "OpenAiVisionSettings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing required OpenAI environment variable: OPENAI_API_KEY")

        model = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna").strip()
        return cls(api_key=api_key, model=model)


class OpenAiVisionProvider:
    def __init__(self, settings: OpenAiVisionSettings) -> None:
        from openai import AsyncOpenAI, OpenAI

        self.provider_name = "openai"
        self.model_name = settings.model
        self._client = OpenAI(api_key=settings.api_key)
        self._async_client = AsyncOpenAI(api_key=settings.api_key)
        self._model = settings.model

    def vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=[_vision_responses_message(prompt, image_path, mime_type)],
            text={"format": {"type": "json_object"}},
            temperature=0,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text
        raise RuntimeError("OpenAI vision response did not contain output text.")

    async def async_vision_complete(
        self,
        prompt: str,
        image_path: Path,
        mime_type: str,
    ) -> str:
        response = await self._async_client.responses.create(
            model=self._model,
            input=[_vision_responses_message(prompt, image_path, mime_type)],
            text={"format": {"type": "json_object"}},
            temperature=0,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text
        raise RuntimeError("OpenAI vision response did not contain output text.")


def _vision_responses_message(prompt: str, image_path: Path, mime_type: str) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": _image_data_url(image_path, mime_type),
            },
        ],
    }
