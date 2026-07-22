import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.extract import (
    discover_artifacts,
    extract_artifacts_to_csv,
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
