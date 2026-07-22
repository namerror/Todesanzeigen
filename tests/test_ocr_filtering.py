import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.todesanzeigen.ocr_filtering import (
    detect_name_from_tsv,
    discover_tsv_artifacts,
    filter_artifact_names,
    name_map_artifact_path,
    parse_tesseract_word_lines,
)


HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def tsv(*rows: str) -> str:
    return "\n".join([HEADER, *rows])


class OcrFilteringTests(TestCase):
    def test_detects_largest_name_line(self) -> None:
        tsv_text = tsv(
            "5\t1\t1\t1\t1\t1\t100\t50\t45\t9\t96\tIn",
            "5\t1\t1\t1\t1\t2\t150\t50\t90\t9\t96\tLiebe",
            "5\t1\t2\t1\t1\t1\t250\t150\t68\t32\t93\tJosef",
            "5\t1\t2\t1\t1\t2\t327\t150\t149\t25\t91\tStrohhofer",
            "5\t1\t3\t1\t1\t1\t125\t429\t143\t19\t96\tveröffentlicht",
            "5\t1\t3\t1\t1\t2\t276\t435\t32\t13\t96\tam",
            "5\t1\t3\t1\t1\t3\t317\t431\t125\t17\t93\t30.01.2023",
        )

        detected_name = detect_name_from_tsv(tsv_text)

        self.assertEqual(detected_name.name, "Josef Strohhofer")
        self.assertAlmostEqual(detected_name.confidence or 0, 92)

    def test_cleans_leading_and_trailing_ocr_noise(self) -> None:
        tsv_text = tsv(
            "5\t1\t1\t1\t1\t1\t253\t124\t20\t31\t20\t4%",
            "5\t1\t1\t1\t1\t2\t280\t124\t110\t31\t92\tTheresia",
            "5\t1\t1\t1\t1\t3\t399\t125\t140\t30\t91\tMenzinger",
            "5\t1\t2\t1\t1\t1\t300\t180\t25\t9\t95\tLiebe",
        )

        detected_name = detect_name_from_tsv(tsv_text)

        self.assertEqual(detected_name.name, "Theresia Menzinger")
        self.assertAlmostEqual(detected_name.confidence or 0, 91.5)

    def test_keeps_titles_and_drops_short_trailing_noise(self) -> None:
        tsv_text = tsv(
            "5\t1\t1\t1\t1\t1\t100\t68\t45\t30\t92\tDipl.",
            "5\t1\t1\t1\t1\t2\t150\t68\t40\t30\t92\tIng.",
            "5\t1\t1\t1\t1\t3\t195\t68\t95\t41\t92\tArmin",
            "5\t1\t1\t1\t1\t4\t300\t68\t95\t41\t92\tRieber",
            "5\t1\t1\t1\t1\t5\t405\t68\t20\t20\t40\tSn",
        )

        detected_name = detect_name_from_tsv(tsv_text)

        self.assertEqual(detected_name.name, "Dipl. Ing. Armin Rieber")
        self.assertAlmostEqual(detected_name.confidence or 0, 92)

    def test_parser_disables_csv_quoting_for_stray_ocr_quote(self) -> None:
        tsv_text = tsv(
            "5\t1\t4\t1\t2\t1\t260\t117\t22\t8\t27\t\"Oma",
            "5\t1\t4\t1\t2\t2\t287\t117\t18\t8\t93\tund",
            "5\t1\t5\t1\t1\t1\t275\t151\t122\t33\t96\tJohanna",
            "5\t1\t5\t1\t1\t2\t408\t151\t115\t26\t96\tLechner",
        )

        lines = parse_tesseract_word_lines(tsv_text)

        self.assertEqual([line.text for line in lines], ['"Oma und', "Johanna Lechner"])
        detected_name = detect_name_from_tsv(tsv_text)
        self.assertEqual(detected_name.name, "Johanna Lechner")
        self.assertAlmostEqual(detected_name.confidence or 0, 96)

    def test_filter_artifact_names_reads_tsv_files_with_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "b.tsv").write_text(
                tsv("5\t1\t1\t1\t1\t1\t100\t100\t80\t30\t95\tAnna"),
                encoding="utf-8",
            )
            (artifacts / "a.tsv").write_text(
                tsv("5\t1\t1\t1\t1\t1\t100\t100\t80\t30\t95\tMax"),
                encoding="utf-8",
            )
            (artifacts / "ignore.txt").write_text("ignored", encoding="utf-8")

            self.assertEqual([path.name for path in discover_tsv_artifacts(artifacts)], ["a.tsv", "b.tsv"])
            results = filter_artifact_names(artifacts, limit=1)
            name_map = json.loads(name_map_artifact_path(artifacts).read_text(encoding="utf-8"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].artifact_path.name, "a.tsv")
        self.assertEqual(results[0].name, "Max")
        self.assertEqual(results[0].confidence, 95)
        self.assertEqual(name_map, {"a.txt": {"name": "Max", "confidence": 95}})
