import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from src.todesanzeigen.llm import (
    QwenProvider,
    QwenSettings,
    QwenVisionProvider,
    QwenVisionSettings,
    build_llm_provider,
    build_vision_llm_provider,
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

    def test_build_vision_llm_provider_defaults_to_qwen(self) -> None:
        with (
            patch.dict("src.todesanzeigen.llm.os.environ", {}, clear=True),
            patch("src.todesanzeigen.llm.QwenVisionSettings.from_env", return_value="settings"),
            patch("src.todesanzeigen.llm.QwenVisionProvider", return_value="provider") as provider,
        ):
            result = build_vision_llm_provider()

        self.assertEqual(result, "provider")
        provider.assert_called_once_with("settings")

    def test_build_vision_llm_provider_uses_explicit_model(self) -> None:
        settings = QwenVisionSettings(
            api_key="key",
            model="qwen-vl-ocr",
            base_url="https://example.test/compatible-mode/v1",
        )
        with (
            patch("src.todesanzeigen.llm.QwenVisionSettings.from_env", return_value=settings),
            patch("src.todesanzeigen.llm.QwenVisionProvider", return_value="provider") as provider,
        ):
            result = build_vision_llm_provider("qwen", "qwen-vl-ocr-latest")

        self.assertEqual(result, "provider")
        provider.assert_called_once_with(
            QwenVisionSettings(
                api_key="key",
                model="qwen-vl-ocr-latest",
                base_url="https://example.test/compatible-mode/v1",
            )
        )


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
        self.assertEqual(settings.model, "qwen3.6-flash")
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


class QwenVisionSettingsTests(TestCase):
    def test_qwen_vision_settings_use_defaults(self) -> None:
        with patch.dict(
            "src.todesanzeigen.llm.os.environ",
            {"DASHSCOPE_API_KEY": " key "},
            clear=True,
        ):
            settings = QwenVisionSettings.from_env()

        self.assertEqual(settings.api_key, "key")
        self.assertEqual(settings.model, "qwen-vl-ocr")
        self.assertEqual(
            settings.base_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_qwen_vision_settings_use_env_overrides(self) -> None:
        with patch.dict(
            "src.todesanzeigen.llm.os.environ",
            {
                "DASHSCOPE_API_KEY": " key ",
                "QWEN_VISION_MODEL": " qwen-vl-ocr-latest ",
                "QWEN_VISION_BASE_URL": " https://example.test/compatible-mode/v1 ",
            },
            clear=True,
        ):
            settings = QwenVisionSettings.from_env()

        self.assertEqual(settings.api_key, "key")
        self.assertEqual(settings.model, "qwen-vl-ocr-latest")
        self.assertEqual(settings.base_url, "https://example.test/compatible-mode/v1")


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
        async_openai_constructor = Mock()
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = openai_constructor
        openai_module.AsyncOpenAI = async_openai_constructor

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
        async_openai_constructor.assert_called_once_with(
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
        openai_module.AsyncOpenAI = Mock()

        with patch.dict("sys.modules", {"openai": openai_module}):
            provider = QwenProvider(QwenSettings(api_key="key"))
            with self.assertRaises(RuntimeError) as error:
                provider.complete("prompt")

        self.assertIn("Qwen response did not contain message content", str(error.exception))


class QwenVisionProviderTests(TestCase):
    def test_qwen_vision_provider_requests_json_chat_completion_with_image(self) -> None:
        completion = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"name":"Vision"}')
                )
            ]
        )
        completions = Mock()
        completions.create.return_value = completion
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        openai_constructor = Mock(return_value=client)
        async_openai_constructor = Mock()
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = openai_constructor
        openai_module.AsyncOpenAI = async_openai_constructor

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "example.jpg"
            image_path.write_bytes(b"image-bytes")
            with patch.dict("sys.modules", {"openai": openai_module}):
                provider = QwenVisionProvider(
                    QwenVisionSettings(
                        api_key="key",
                        model="qwen-vl-ocr",
                        base_url="https://example.test/compatible-mode/v1",
                    )
                )
                result = provider.vision_complete("Bitte JSON extrahieren.", image_path, "image/jpeg")

        self.assertEqual(result, '{"name":"Vision"}')
        completions.create.assert_called_once()
        kwargs = completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen-vl-ocr")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        content = kwargs["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Bitte JSON extrahieren."})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))


class QwenAsyncProviderTests(IsolatedAsyncioTestCase):
    async def test_qwen_provider_requests_async_json_chat_completion(self) -> None:
        completion = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"name":"Mustermann"}')
                )
            ]
        )
        completions = Mock()
        completions.create = Mock()

        async def create_completion(**kwargs: object) -> object:
            completions.create(**kwargs)
            return completion

        async_completions = types.SimpleNamespace(create=create_completion)
        async_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=async_completions)
        )
        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = Mock()
        openai_module.AsyncOpenAI = Mock(return_value=async_client)

        with patch.dict("sys.modules", {"openai": openai_module}):
            provider = QwenProvider(
                QwenSettings(
                    api_key="key",
                    model="qwen-plus",
                    base_url="https://example.test/compatible-mode/v1",
                )
            )
            result = await provider.async_complete("Bitte JSON extrahieren.")

        self.assertEqual(result, '{"name":"Mustermann"}')
        completions.create.assert_called_once_with(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Bitte JSON extrahieren."}],
            response_format={"type": "json_object"},
            temperature=0,
        )
