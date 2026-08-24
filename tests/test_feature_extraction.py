from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.todesanzeigen.feature_extraction import (
    extract_image_features,
    extract_ocr_text_features,
    extract_tsv_features,
)


TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext"
)


class ImageFeatureExtractionTests(TestCase):
    def test_extracts_content_features_without_path_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "notice.png"
            Image.new("L", (100, 50), color=128).save(image_path)

            features = extract_image_features(image_path)

        self.assertEqual(features["image_width"], 100)
        self.assertEqual(features["image_height"], 50)
        self.assertEqual(features["image_aspect_ratio"], 2.0)
        self.assertEqual(features["image_feature_status"], "ok")
        self.assertGreater(features["image_file_size"], 0)
        self.assertNotIn("image_path_present", features)
        self.assertNotIn("image_suffix", features)

    def test_image_errors_are_reduced_to_a_stable_status(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "invalid.jpg"
            image_path.write_bytes(b"not an image")

            features = extract_image_features(image_path)

        self.assertEqual(features["image_feature_status"], "error")
        self.assertNotIn("image_feature_error", features)


class OcrFeatureExtractionTests(TestCase):
    def test_extracts_text_quality_features(self) -> None:
        features = extract_ocr_text_features(
            "Max Mustermann!\nAichach",
            name_hint="Max Mustermann",
            name_confidence=91,
        )

        self.assertEqual(features["ocr_word_count"], 3)
        self.assertEqual(features["ocr_line_count"], 2)
        self.assertEqual(features["name_hint_length"], 14)
        self.assertEqual(features["name_confidence"], 91)
        self.assertGreater(features["ocr_suspicious_char_ratio"], 0)

    def test_extracts_tsv_confidence_and_layout_features(self) -> None:
        tsv_text = "\n".join(
            [
                TSV_HEADER,
                "5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t90\tMax",
                "5\t1\t1\t1\t1\t2\t60\t20\t80\t10\t50\tMustermann",
            ]
        )

        features = extract_tsv_features(tsv_text)

        self.assertEqual(features["tsv_line_count"], 1)
        self.assertEqual(features["tsv_word_count"], 2)
        self.assertEqual(features["tsv_mean_confidence"], 70)
        self.assertEqual(features["tsv_low_confidence_ratio"], 0.5)
        self.assertGreater(features["layout_bbox_density"], 0)
