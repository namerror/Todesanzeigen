import types
from unittest import TestCase
from unittest.mock import Mock, patch

from src.todesanzeigen.llm import (
    QwenProvider,
    QwenSettings,
    build_llm_provider,
)
from src.todesanzeigen.ocr import ConfigError


class LlmProviderTests(TestCase):
    def test_build_llm_provider_defaults_to_gemini(self) -> None:
        with (
            patch.dict("src.todesanzeigen.llm.os.environ", {}, clear=True),
            patch("src.todesanzeigen.llm.GeminiSettings.from_env", return_value="settings"),
            patch("src.todesanzeigen.llm.GeminiProvider", return_value="provider") as provider,
        ):
            result = build_llm_provider()

        self.assertEqual(result, "provider")
        provider.assert_called_once_with("settings")

    def test_build_llm_provider_uses_env_provider(self) -> None:
        with (
            patch.dict(
                "src.todesanzeigen.llm.os.environ",
                {"TODESANZEIGEN_LLM_PROVIDER": "qwen"},
                clear=True,
            ),
            patch("src.todesanzeigen.llm.QwenSettings.from_env", return_value="settings"),
            patch("src.todesanzeigen.llm.QwenProvider", return_value="provider") as provider,
        ):
            result = build_llm_provider()

        self.assertEqual(result, "provider")
        provider.assert_called_once_with("settings")

    def test_build_llm_provider_uses_explicit_provider(self) -> None:
        with (
            patch("src.todesanzeigen.llm.QwenSettings.from_env", return_value="settings"),
            patch("src.todesanzeigen.llm.QwenProvider", return_value="provider") as provider,
        ):
            result = build_llm_provider("qwen")

        self.assertEqual(result, "provider")
        provider.assert_called_once_with("settings")

    def test_build_llm_provider_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ConfigError) as error:
            build_llm_provider("unknown")

        self.assertIn("Unsupported LLM provider: unknown", str(error.exception))


class QwenSettingsTests(TestCase):
    def test_qwen_settings_require_api_key(self) -> None:
        with patch.dict("src.todesanzeigen.llm.os.environ", {}, clear=True):
            with self.assertRaises(ConfigError) as error:
                QwenSettings.from_env()

        self.assertIn("DASHSCOPE_API_KEY", str(error.exception))

    def test_qwen_settings_use_defaults(self) -> None:
        with patch.dict(
            "src.todesanzeigen.llm.os.environ",
            {"DASHSCOPE_API_KEY": " key "},
            clear=True,
        ):
            settings = QwenSettings.from_env()

        self.assertEqual(settings.api_key, "key")
        self.assertEqual(settings.model, "qwen3.7-plus")
        self.assertEqual(
            settings.base_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_qwen_settings_use_env_overrides(self) -> None:
        with patch.dict(
            "src.todesanzeigen.llm.os.environ",
            {
                "DASHSCOPE_API_KEY": " key ",
                "QWEN_MODEL": " qwen-plus ",
                "QWEN_BASE_URL": " https://dashscope-us.aliyuncs.com/compatible-mode/v1 ",
            },
            clear=True,
        ):
            settings = QwenSettings.from_env()

        self.assertEqual(settings.api_key, "key")
        self.assertEqual(settings.model, "qwen-plus")
        self.assertEqual(
            settings.base_url,
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        )


class QwenProviderTests(TestCase):
    def test_qwen_provider_requests_json_chat_completion(self) -> None:
        completion = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"name":"Mustermann"}')
                )
            ]
        )
        completions = Mock()
        completions.create.return_value = completion
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        openai_constructor = Mock(return_value=client)
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = openai_constructor

        with patch.dict("sys.modules", {"openai": openai_module}):
            provider = QwenProvider(
                QwenSettings(
                    api_key="key",
                    model="qwen-plus",
                    base_url="https://example.test/compatible-mode/v1",
                )
            )
            result = provider.complete("Bitte JSON extrahieren.")

        self.assertEqual(result, '{"name":"Mustermann"}')
        openai_constructor.assert_called_once_with(
            api_key="key",
            base_url="https://example.test/compatible-mode/v1",
        )
        completions.create.assert_called_once_with(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Bitte JSON extrahieren."}],
            response_format={"type": "json_object"},
            temperature=0,
        )

    def test_qwen_provider_rejects_empty_response_content(self) -> None:
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
        )
        completions = Mock()
        completions.create.return_value = completion
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = Mock(return_value=client)

        with patch.dict("sys.modules", {"openai": openai_module}):
            provider = QwenProvider(QwenSettings(api_key="key"))
            with self.assertRaises(RuntimeError) as error:
                provider.complete("prompt")

        self.assertIn("Qwen response did not contain message content", str(error.exception))
