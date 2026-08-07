import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from src.todesanzeigen.extract import (
    AsyncExtractionSettings,
    discover_artifacts,
    estimate_llm_tokens,
    extract_artifacts_to_csv,
    extract_artifacts_to_csv_async,
    load_name_map,
    parse_json_object,
)
from src.todesanzeigen.llm import CSV_COLUMNS


class FakeLlmProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class AsyncFakeLlmProvider:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def async_complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for marker, response in self.responses.items():
            if marker in prompt:
                return response
        return "{}"


class ExtractTests(TestCase):
    def test_discover_artifacts_only_direct_txt_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            nested = artifacts / "nested"
            nested.mkdir(parents=True)
            (artifacts / "a.txt").write_text("a", encoding="utf-8")
            (artifacts / "b.md").write_text("b", encoding="utf-8")
            (nested / "c.txt").write_text("c", encoding="utf-8")

            self.assertEqual([path.name for path in discover_artifacts(artifacts)], ["a.txt"])

    def test_extract_artifact_to_csv_with_missing_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann 1900-1980", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 92.0}}),
                encoding="utf-8",
            )
            provider = FakeLlmProvider(
                json.dumps(
                    {
                        "name": "Mustermann",
                        "vorname": "Max",
                        "foto": "ja",
                        "bemerkungen": "Vom LLM geliefert",
                        "quelle": "LLM-Quelle",
                        "dateiname": "llm-dateiname",
                        "confidence_score": "0.8",
                    }
                )
            )

            results = extract_artifacts_to_csv(artifacts, output, provider, source="Testquelle")

            self.assertEqual(len(results), 1)
            csv_text = output.read_text(encoding="utf-8")
            header = csv_text.splitlines()[0].split(",")
            self.assertEqual(header, CSV_COLUMNS)
            row = results[0].row
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["name"], "Mustermann")
            self.assertEqual(row["dateiname"], "example")
            self.assertEqual(row["quelle"], "Testquelle")
            self.assertEqual(row["foto"], "")
            self.assertEqual(row["bemerkungen"], "")
            self.assertIn("Lokales OCR-Name-Signal", provider.prompts[0])
            self.assertIn("92.0% Konfidenz", provider.prompts[0])
            self.assertIn('"Max Mustermann"', provider.prompts[0])
            self.assertNotIn("OCR-Layout aus TSV", provider.prompts[0])
            prompt_fields = provider.prompts[0].split("Regeln:")[0]
            self.assertNotIn("foto", prompt_fields)
            self.assertNotIn("bemerkungen", prompt_fields)
            self.assertNotIn("quelle", prompt_fields)
            self.assertNotIn("dateiname", prompt_fields)
            self.assertNotIn("foto ist", provider.prompts[0])
            self.assertNotIn("quelle muss", provider.prompts[0])
            self.assertNotIn("dateiname muss", provider.prompts[0])

    def test_extract_skips_low_confidence_artifact_and_logs_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            log_file = root / "logs" / "extract.txt"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 82.4}}),
                encoding="utf-8",
            )
            provider = FakeLlmProvider("{}")

            results = extract_artifacts_to_csv(artifacts, output, provider, log_file=log_file)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "skipped_low_confidence")
            self.assertEqual(provider.prompts, [])
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [",".join(CSV_COLUMNS)],
            )
            self.assertIn(
                "WARNING: file example.txt has low confidence (82.4 < 85.0), "
                "not passed to extraction",
                log_file.read_text(encoding="utf-8"),
            )

    def test_extract_skips_missing_confidence_artifact_and_logs_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            log_file = root / "logs" / "extract.txt"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": None}}),
                encoding="utf-8",
            )
            provider = FakeLlmProvider("{}")

            results = extract_artifacts_to_csv(artifacts, output, provider, log_file=log_file)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "skipped_low_confidence")
            self.assertEqual(provider.prompts, [])
            self.assertIn(
                "WARNING: file example.txt has missing confidence, not passed to extraction",
                log_file.read_text(encoding="utf-8"),
            )

    def test_extract_processes_confidence_equal_to_threshold(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Max Mustermann", "confidence": 85.0}}),
                encoding="utf-8",
            )
            provider = FakeLlmProvider(json.dumps({"name": "Mustermann"}))

            results = extract_artifacts_to_csv(artifacts, output, provider)

            self.assertEqual(results[0].status, "processed")
            self.assertEqual(len(provider.prompts), 1)

    def test_extract_requires_name_map_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            provider = FakeLlmProvider("{}")

            with self.assertRaises(FileNotFoundError) as error:
                extract_artifacts_to_csv(artifacts, output, provider)

            self.assertIn("Missing OCR name map artifact", str(error.exception))

    def test_extract_requires_name_map_entry_for_each_text_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            (artifacts / "name_map.json").write_text("{}", encoding="utf-8")
            provider = FakeLlmProvider("{}")

            with self.assertRaises(ValueError) as error:
                extract_artifacts_to_csv(artifacts, output, provider)

            self.assertIn("Missing OCR name map entry for example.txt", str(error.exception))

    def test_load_name_map_normalizes_values(self) -> None:
        with TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": None, "confidence": ""}}),
                encoding="utf-8",
            )

            name_map = load_name_map(artifacts)

        self.assertEqual(name_map["example.txt"].name, "")
        self.assertIsNone(name_map["example.txt"].confidence)

    def test_invalid_json_response_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_json_object("not json")

    def test_estimate_llm_tokens_includes_output_budget(self) -> None:
        self.assertEqual(estimate_llm_tokens("abc"), 301)


class AsyncExtractTests(IsolatedAsyncioTestCase):
    async def test_async_extract_writes_rows_in_artifact_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "a.txt").write_text("alpha notice", encoding="utf-8")
            (artifacts / "b.txt").write_text("beta notice", encoding="utf-8")
            (artifacts / "c.txt").write_text("gamma notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps(
                    {
                        "a.txt": {"name": "Alpha Person", "confidence": 90},
                        "b.txt": {"name": "Beta Person", "confidence": 90},
                        "c.txt": {"name": "Gamma Person", "confidence": 90},
                    }
                ),
                encoding="utf-8",
            )

            class DelayedProvider:
                async def async_complete(self, prompt: str) -> str:
                    if "alpha notice" in prompt:
                        return json.dumps({"name": "Alpha"})
                    if "beta notice" in prompt:
                        return json.dumps({"name": "Beta"})
                    return json.dumps({"name": "Gamma"})

            results = await extract_artifacts_to_csv_async(
                artifacts,
                output,
                DelayedProvider(),
                settings=AsyncExtractionSettings(concurrency=3),
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([result.status for result in results], ["processed"] * 3)
        self.assertEqual([row["name"] for row in rows], ["Alpha", "Beta", "Gamma"])

    async def test_async_extract_skips_low_confidence_without_llm_call(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("low confidence notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Example Person", "confidence": 80}}),
                encoding="utf-8",
            )
            provider = AsyncFakeLlmProvider({"low confidence notice": "{}"})

            results = await extract_artifacts_to_csv_async(artifacts, output, provider)

        self.assertEqual(results[0].status, "skipped_low_confidence")
        self.assertEqual(provider.prompts, [])

    async def test_async_extract_retries_transient_errors(self) -> None:
        class RateLimitError(Exception):
            status_code = 429

        class RetryProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def async_complete(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError("rate limited")
                return json.dumps({"name": "Recovered"})

        async def no_sleep(seconds: float) -> None:
            return None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("retry notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Recovered Person", "confidence": 90}}),
                encoding="utf-8",
            )
            provider = RetryProvider()

            with (
                patch("src.todesanzeigen.extract.asyncio.sleep", new=no_sleep),
                patch("src.todesanzeigen.extract.random.uniform", return_value=0),
            ):
                results = await extract_artifacts_to_csv_async(
                    artifacts,
                    output,
                    provider,
                    settings=AsyncExtractionSettings(max_retries=1),
                )

        self.assertEqual(provider.calls, 2)
        self.assertEqual(results[0].status, "processed")
        assert results[0].row is not None
        self.assertEqual(results[0].row["name"], "Recovered")

    async def test_async_extract_records_permanent_failure(self) -> None:
        class FailingProvider:
            async def async_complete(self, prompt: str) -> str:
                raise ValueError("bad response")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            log_file = root / "logs" / "extract.txt"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("failure notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Failure Person", "confidence": 90}}),
                encoding="utf-8",
            )

            results = await extract_artifacts_to_csv_async(
                artifacts,
                output,
                FailingProvider(),
                log_file=log_file,
            )

            csv_lines = output.read_text(encoding="utf-8").splitlines()
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(results[0].status, "failed")
        self.assertIn("bad response", results[0].error or "")
        self.assertEqual(csv_lines, [",".join(CSV_COLUMNS)])
        self.assertIn("ERROR: file example.txt failed extraction: bad response", log_text)

    async def test_async_extract_resumes_completed_checkpoint_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            checkpoint = root / "logs" / "extract.results.jsonl"
            checkpoint.parent.mkdir()
            artifacts.mkdir()
            (artifacts / "a.txt").write_text("already done", encoding="utf-8")
            (artifacts / "b.txt").write_text("new notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps(
                    {
                        "a.txt": {"name": "Already Done", "confidence": 90},
                        "b.txt": {"name": "New Person", "confidence": 90},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint.write_text(
                json.dumps(
                    {
                        "filename": "a.txt",
                        "status": "processed",
                        "attempts": 1,
                        "row": {"name": "Already", "dateiname": "a"},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider = AsyncFakeLlmProvider({"new notice": json.dumps({"name": "New"})})

            results = await extract_artifacts_to_csv_async(
                artifacts,
                output,
                provider,
                resume_from=checkpoint,
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([result.status for result in results], ["processed", "processed"])
        self.assertEqual(len(provider.prompts), 1)
        self.assertEqual([row["name"] for row in rows], ["Already", "New"])

    async def test_async_extract_creates_missing_resume_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            checkpoint = root / "logs" / "missing.results.jsonl"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("new notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "New Person", "confidence": 90}}),
                encoding="utf-8",
            )
            provider = AsyncFakeLlmProvider({"new notice": json.dumps({"name": "New"})})

            results = await extract_artifacts_to_csv_async(
                artifacts,
                output,
                provider,
                resume_from=checkpoint,
            )

            checkpoint_lines = checkpoint.read_text(encoding="utf-8").splitlines()

        self.assertEqual(results[0].status, "processed")
        self.assertEqual(len(checkpoint_lines), 1)
        self.assertIn('"filename": "example.txt"', checkpoint_lines[0])

    async def test_async_extract_appends_to_resume_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            checkpoint = root / "logs" / "resume.results.jsonl"
            checkpoint.parent.mkdir()
            artifacts.mkdir()
            (artifacts / "a.txt").write_text("already done", encoding="utf-8")
            (artifacts / "b.txt").write_text("new notice", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps(
                    {
                        "a.txt": {"name": "Already Done", "confidence": 90},
                        "b.txt": {"name": "New Person", "confidence": 90},
                    }
                ),
                encoding="utf-8",
            )
            existing_record = {
                "filename": "a.txt",
                "status": "processed",
                "attempts": 1,
                "row": {"name": "Already", "dateiname": "a"},
                "error": None,
            }
            checkpoint.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")
            provider = AsyncFakeLlmProvider({"new notice": json.dumps({"name": "New"})})

            await extract_artifacts_to_csv_async(
                artifacts,
                output,
                provider,
                checkpoint_file=checkpoint,
                resume_from=checkpoint,
            )

            checkpoint_lines = checkpoint.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(checkpoint_lines), 2)
        self.assertIn('"filename": "a.txt"', checkpoint_lines[0])
        self.assertIn('"filename": "b.txt"', checkpoint_lines[1])
        self.assertEqual(len(provider.prompts), 1)
