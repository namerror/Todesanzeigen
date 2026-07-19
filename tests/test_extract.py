import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.extract import (
    discover_artifacts,
    extract_artifacts_to_csv,
    parse_tesseract_tsv,
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
            (artifacts / "example.tsv").write_text(
                "\n".join(
                    [
                        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                        "2\t1\t1\t0\t0\t0\t100\t100\t260\t40\t-1\t",
                        "4\t1\t1\t1\t1\t0\t100\t100\t260\t40\t-1\t",
                        "5\t1\t1\t1\t1\t1\t100\t100\t80\t40\t93\tMax",
                        "5\t1\t1\t1\t1\t2\t190\t100\t170\t40\t91\tMustermann",
                    ]
                ),
                encoding="utf-8",
            )
            provider = FakeLlmProvider(
                json.dumps(
                    {
                        "name": "Mustermann",
                        "vorname": "Max",
                        "confidence_score": "0.8",
                    }
                )
            )

            results = extract_artifacts_to_csv(artifacts, output, provider, source="Testquelle")

            self.assertEqual(len(results), 1)
            csv_text = output.read_text(encoding="utf-8")
            header = csv_text.splitlines()[0].split(",")
            self.assertEqual(header, CSV_COLUMNS)
            self.assertIn("Mustermann", csv_text)
            self.assertIn("example", csv_text)
            self.assertIn("Testquelle", csv_text)
            self.assertIn("OCR-Layout aus TSV", provider.prompts[0])
            self.assertIn("height=40", provider.prompts[0])
            self.assertIn("avg_conf=92.0", provider.prompts[0])

    def test_extract_requires_matching_tsv_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            output = root / "output" / "result.csv"
            artifacts.mkdir()
            (artifacts / "example.txt").write_text("Max Mustermann", encoding="utf-8")
            provider = FakeLlmProvider("{}")

            with self.assertRaises(FileNotFoundError) as error:
                extract_artifacts_to_csv(artifacts, output, provider)

            self.assertIn("Missing TSV layout artifact", str(error.exception))

    def test_parse_tesseract_tsv_groups_words_into_layout_lines(self) -> None:
        tsv_text = "\n".join(
            [
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                "2\t1\t6\t0\t0\t0\t253\t124\t259\t31\t-1\t",
                "4\t1\t6\t1\t1\t0\t253\t124\t259\t31\t-1\t",
                "5\t1\t6\t1\t1\t1\t253\t124\t110\t23\t92.9\tTheresia",
                "5\t1\t6\t1\t1\t2\t372\t125\t140\t30\t91.3\tMenzinger",
            ]
        )

        blocks = parse_tesseract_tsv(tsv_text)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_num, 6)
        self.assertEqual(blocks[0].lines[0].text, "Theresia Menzinger")
        self.assertEqual(blocks[0].lines[0].height, 31)
        self.assertAlmostEqual(blocks[0].lines[0].avg_confidence or 0, 92.1)

    def test_invalid_json_response_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_json_object("not json")
