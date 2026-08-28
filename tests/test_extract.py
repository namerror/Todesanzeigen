import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from src.todesanzeigen.extract import (
    AsyncExtractionSettings,
    VisionRerouteSettings,
    discover_artifacts,
    estimate_llm_tokens,
    extract_artifacts_to_csv,
    extract_artifacts_to_csv_async,
    extract_image_with_vision,
    extract_images_to_csv_async,
    load_reroute_candidates,
    load_name_map,
    parse_json_object,
    reroute_candidates_to_csv_async,
    select_image_paths,
    select_reroute_candidates,
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


class AsyncFakeVisionProvider:
    provider_name = "fake-vision"
    model_name = "fake-vision-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, Path, str]] = []

    def vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        self.calls.append((prompt, image_path, mime_type))
        return self.response

    async def async_vision_complete(self, prompt: str, image_path: Path, mime_type: str) -> str:
        self.calls.append((prompt, image_path, mime_type))
        return self.response


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
            self.assertNotIn("wohnort", prompt_fields)
            self.assertIn("DD.MM.YYYY", provider.prompts[0])
            self.assertIn("direkt darunter zwei Orte", provider.prompts[0])
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

    def test_load_reroute_candidates_from_results_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            checkpoint = root / "logs" / "results.jsonl"
            artifacts.mkdir()
            checkpoint.parent.mkdir()
            (artifacts / "name_map.json").write_text(
                json.dumps(
                    {
                        "low.txt": {"name": "Low Person", "confidence": 80},
                        "done.txt": {"name": "Done Person", "confidence": 95},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint.write_text(
                "\n".join(
                    [
                        json.dumps({"filename": "low.txt", "status": "skipped_low_confidence"}),
                        json.dumps({"filename": "done.txt", "status": "processed", "row": {}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = load_reroute_candidates(artifacts, results_file=checkpoint)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].artifact_path.name, "low.txt")
        self.assertEqual(candidates[0].name_hint.name, "Low Person")
        self.assertEqual(candidates[0].name_hint.confidence, 80)

    def test_select_image_paths_applies_only_sample_and_limit(self) -> None:
        images = [Path("b.jpg"), Path("a.png"), Path("c.webp")]

        selected = select_image_paths(images, only=["c.webp", "a"], limit=1)
        sampled = select_image_paths(images, sample_ratio=0.5, sample_seed=42)

        self.assertEqual([path.name for path in selected], ["a.png"])
        self.assertEqual(len(sampled), 2)
        self.assertEqual(sampled, select_image_paths(images, sample_ratio=0.5, sample_seed=42))

    def test_extract_image_with_vision_rejects_unsupported_file(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "example.txt"
            image_path.write_text("not an image", encoding="utf-8")
            provider = AsyncFakeVisionProvider("{}")

            with self.assertRaises(ValueError) as error:
                extract_image_with_vision(image_path, provider)

        self.assertIn("Unsupported image type for vision extraction", str(error.exception))


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

    async def test_async_extract_reroutes_low_confidence_to_vision_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            input_dir = root / "input"
            output = root / "output" / "result.csv"
            reroute_results = root / "logs" / "reroute-results.jsonl"
            artifacts.mkdir()
            input_dir.mkdir()
            (artifacts / "example.txt").write_text("weak local OCR", encoding="utf-8")
            (artifacts / "name_map.json").write_text(
                json.dumps({"example.txt": {"name": "Wrong Person", "confidence": 40}}),
                encoding="utf-8",
            )
            (input_dir / "example.jpg").write_bytes(b"image")
            text_provider = AsyncFakeLlmProvider({"weak local OCR": "{}"})
            vision_provider = AsyncFakeVisionProvider(json.dumps({"name": "Vision Name"}))

            results = await extract_artifacts_to_csv_async(
                artifacts,
                output,
                text_provider,
                reroute_settings=VisionRerouteSettings(
                    input_dir=input_dir,
                    provider=vision_provider,
                    results_file=reroute_results,
                ),
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            reroute_records = [
                json.loads(line)
                for line in reroute_results.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(results[0].status, "rerouted_processed")
        self.assertEqual(text_provider.prompts, [])
        self.assertEqual(rows[0]["name"], "Vision Name")
        self.assertEqual(vision_provider.calls[0][1].name, "example.jpg")
        self.assertEqual(vision_provider.calls[0][2], "image/jpeg")
        self.assertIn("weak local OCR", vision_provider.calls[0][0])
        self.assertEqual(reroute_records[0]["method"], "vision_model_reroute")
        self.assertEqual(reroute_records[0]["original_name_confidence"], 40)

    async def test_standalone_reroute_processes_selected_low_confidence_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            input_dir = root / "input"
            output = root / "output" / "rerouted.csv"
            results_file = root / "logs" / "reroute-results.jsonl"
            artifacts.mkdir()
            input_dir.mkdir()
            (artifacts / "a.txt").write_text("a ocr", encoding="utf-8")
            (artifacts / "b.txt").write_text("b ocr", encoding="utf-8")
            (input_dir / "a.jpg").write_bytes(b"a")
            (input_dir / "b.jpg").write_bytes(b"b")
            low_confidence_file = root / "logs" / "filter-low-confidence.jsonl"
            low_confidence_file.parent.mkdir()
            low_confidence_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "filename": "a.txt",
                                "status": "skipped_low_confidence",
                                "name": "A Person",
                                "confidence": 80,
                                "threshold": 85,
                            }
                        ),
                        json.dumps(
                            {
                                "filename": "b.txt",
                                "status": "skipped_low_confidence",
                                "name": "B Person",
                                "confidence": 70,
                                "threshold": 85,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = select_reroute_candidates(
                load_reroute_candidates(artifacts, low_confidence_file=low_confidence_file),
                only=["b.txt"],
            )
            vision_provider = AsyncFakeVisionProvider(json.dumps({"name": "Selected Vision"}))

            results = await reroute_candidates_to_csv_async(
                candidates,
                output,
                vision_provider,
                input_dir=input_dir,
                results_file=results_file,
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([candidate.artifact_path.name for candidate in candidates], ["b.txt"])
        self.assertEqual(results[0].status, "rerouted_processed")
        self.assertEqual(rows[0]["name"], "Selected Vision")
        self.assertEqual(vision_provider.calls[0][1].name, "b.jpg")

    async def test_image_only_vision_extract_processes_images_without_ocr_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "output" / "vision.csv"
            results_file = root / "logs" / "vision-results.jsonl"
            input_dir.mkdir()
            (input_dir / "a.jpg").write_bytes(b"a")
            (input_dir / "b.png").write_bytes(b"b")
            (input_dir / "ignored.txt").write_text("ignore", encoding="utf-8")
            vision_provider = AsyncFakeVisionProvider(json.dumps({"name": "Vision Name"}))

            results = await extract_images_to_csv_async(
                input_dir,
                output,
                vision_provider,
                source="Ground Truth",
                results_file=results_file,
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            records = [
                json.loads(line)
                for line in results_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([result.status for result in results], ["vision_processed", "vision_processed"])
        self.assertEqual([row["dateiname"] for row in rows], ["a", "b"])
        self.assertEqual([row["quelle"] for row in rows], ["Ground Truth", "Ground Truth"])
        self.assertEqual([call[1].name for call in vision_provider.calls], ["a.jpg", "b.png"])
        self.assertEqual([call[2] for call in vision_provider.calls], ["image/jpeg", "image/png"])
        self.assertNotIn("Lokales OCR-Name-Signal", vision_provider.calls[0][0])
        self.assertNotIn("OCR-Text", vision_provider.calls[0][0])
        self.assertEqual(records[0]["method"], "vision_model_image_only")
        self.assertEqual(records[0]["source_image"], str(input_dir / "a.jpg"))
        self.assertEqual(records[0]["mime_type"], "image/jpeg")

    async def test_image_only_vision_extract_resumes_completed_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "output" / "vision.csv"
            results_file = root / "logs" / "vision-results.jsonl"
            input_dir.mkdir()
            results_file.parent.mkdir()
            (input_dir / "a.jpg").write_bytes(b"a")
            (input_dir / "b.jpg").write_bytes(b"b")
            results_file.write_text(
                json.dumps(
                    {
                        "filename": "a.jpg",
                        "status": "vision_processed",
                        "method": "vision_model_image_only",
                        "attempts": 1,
                        "row": {"name": "Already", "dateiname": "a"},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            vision_provider = AsyncFakeVisionProvider(json.dumps({"name": "New"}))

            await extract_images_to_csv_async(
                input_dir,
                output,
                vision_provider,
                results_file=results_file,
            )
            with output.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([row["name"] for row in rows], ["Already", "New"])
        self.assertEqual([call[1].name for call in vision_provider.calls], ["b.jpg"])

    async def test_image_only_vision_extract_records_failure(self) -> None:
        class FailingVisionProvider:
            provider_name = "fake-vision"
            model_name = "fake-vision-model"

            async def async_vision_complete(
                self,
                prompt: str,
                image_path: Path,
                mime_type: str,
            ) -> str:
                raise ValueError("bad image")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "output" / "vision.csv"
            log_file = root / "logs" / "vision.txt"
            input_dir.mkdir()
            (input_dir / "example.jpg").write_bytes(b"image")

            results = await extract_images_to_csv_async(
                input_dir,
                output,
                FailingVisionProvider(),
                log_file=log_file,
                settings=AsyncExtractionSettings(max_retries=0),
            )
            log_text = log_file.read_text(encoding="utf-8")

        self.assertEqual(results[0].status, "vision_failed")
        self.assertIn("bad image", results[0].error or "")
        self.assertIn(
            "ERROR: file example.jpg failed vision_model_image_only: bad image",
            log_text,
        )

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
