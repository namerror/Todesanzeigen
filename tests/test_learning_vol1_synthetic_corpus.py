import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


LEARNING_VOL1 = Path(__file__).resolve().parents[1] / "learning" / "vol1"
sys.path.insert(0, str(LEARNING_VOL1))

from synthetic_notice_generator import (  # noqa: E402
    BOS_TOKEN,
    CORRUPTION_KINDS,
    EOS_TOKEN,
    GENERATOR_VERSION,
    GeneratorConfig,
    apply_ocr_corruption,
    build_manifest,
    generate_records,
    iter_bigram_events,
    iter_corpus_bigram_events,
    main,
    serialize_records,
    write_corpus,
)


DEFAULT_V2_FINGERPRINT = "4367f6610b5df498b5cbf3ffdcd79d91cb4fae058b483e1ee19df057e36029d2"


class SyntheticNoticeGeneratorTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GeneratorConfig()
        cls.records = generate_records(cls.config)
        cls.manifest = build_manifest(cls.records, cls.config)

    def test_default_corpus_is_deterministic_and_frozen(self) -> None:
        repeated_records = generate_records(self.config)
        repeated_manifest = build_manifest(repeated_records, self.config)

        self.assertEqual(self.records, repeated_records)
        self.assertEqual(serialize_records(self.records), serialize_records(repeated_records))
        self.assertEqual(self.manifest, repeated_manifest)
        self.assertEqual(len(self.records), 1_200)
        self.assertEqual(
            self.manifest["corpus_fingerprint"]["value"],
            DEFAULT_V2_FINGERPRINT,
        )

    def test_different_seed_changes_content_and_fingerprint(self) -> None:
        changed = GeneratorConfig(seed=self.config.seed + 1)
        changed_records = generate_records(changed)
        changed_manifest = build_manifest(changed_records, changed)

        self.assertNotEqual(serialize_records(self.records), serialize_records(changed_records))
        self.assertNotEqual(
            self.manifest["corpus_fingerprint"]["value"],
            changed_manifest["corpus_fingerprint"]["value"],
        )

    def test_document_ids_are_unique_and_groups_do_not_cross_splits(self) -> None:
        self.assertEqual(len({record.document_id for record in self.records}), len(self.records))
        group_splits: dict[str, set[str]] = {}
        for record in self.records:
            group_splits.setdefault(record.group_id, set()).add(record.split)

        self.assertEqual(len(group_splits), 400)
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        self.assertEqual({record.split for record in self.records}, {"train", "validation", "test"})

    def test_each_group_has_clean_light_and_heavy_variants(self) -> None:
        group_variants: dict[str, set[str]] = {}
        for record in self.records:
            variants = {name for name in record.slices if name.startswith("variant-")}
            self.assertEqual(len(variants), 1)
            group_variants.setdefault(record.group_id, set()).update(variants)

        self.assertTrue(
            all(
                variants == {"variant-clean", "variant-light", "variant-heavy"}
                for variants in group_variants.values()
            )
        )

    def test_required_semantic_and_corruption_slices_are_present(self) -> None:
        slices = set(self.manifest["slice_counts"])
        self.assertEqual(self.manifest["configuration"]["corruptions"], list(CORRUPTION_KINDS))
        self.assertEqual(self.manifest["split_id"], "synthetic-v2-grouped")
        self.assertTrue({"date", "relationship", "ceremony", "umlaut"} <= slices)
        self.assertTrue({"title", "maiden-name", "occupation"} <= slices)
        self.assertTrue(
            {
                "ocr-rn-to-m",
                "ocr-m-to-rn",
                "ocr-1-to-l",
                "ocr-l-to-1",
                "ocr-dropped-punctuation",
                "ocr-merged-spaces",
                "ocr-broken-umlaut",
                "ocr-missing-date-digit",
                "ocr-wrong-date-digit",
            }
            <= slices
        )

    def test_each_ocr_corruption_changes_text_and_reports_a_slice(self) -> None:
        samples = {
            "rn-m": "Urne und warmer Morgen",
            "one-ell": "1 leiser Gruß",
            "dropped-punctuation": "In Liebe, für immer.",
            "merged-spaces": "In stiller Trauer",
            "broken-umlaut": "Für schöne Jahre",
            "missing-date-digit": "Geboren am 7. März 1942",
            "wrong-date-digit": "Gestorben am 08.11.2025",
        }
        for kind in CORRUPTION_KINDS:
            with self.subTest(kind=kind):
                changed, label = apply_ocr_corruption(samples[kind], kind)
                self.assertNotEqual(changed, samples[kind])
                self.assertIsNotNone(label)
                self.assertTrue(label.startswith("ocr-"))

        reverse_rn, rn_label = apply_ocr_corruption("immer", "rn-m", reverse=True)
        reverse_ell, ell_label = apply_ocr_corruption("leise", "one-ell", reverse=True)
        self.assertNotEqual(reverse_rn, "immer")
        self.assertEqual(rn_label, "ocr-m-to-rn")
        self.assertNotEqual(reverse_ell, "leise")
        self.assertEqual(ell_label, "ocr-l-to-1")

    def test_date_corruptions_target_only_recognized_date_digits(self) -> None:
        format_cases = {
            "7. März 1942": ". März 1942",
            "07. März 1942": "7. März 1942",
            "7.3.1942": ".3.1942",
            "07.03.1942": "7.03.1942",
        }
        for original, expected in format_cases.items():
            with self.subTest(date_format=original):
                changed, label = apply_ocr_corruption(original, "missing-date-digit")
                self.assertEqual(changed, expected)
                self.assertEqual(label, "ocr-missing-date-digit")

        text = "Geboren 7. März 1942; gestorben 08.11.2025; um 10.00 Uhr."

        missing, missing_label = apply_ocr_corruption(text, "missing-date-digit")
        self.assertEqual(missing, "Geboren . März 1942; gestorben 08.11.2025; um 10.00 Uhr.")
        self.assertEqual(missing_label, "ocr-missing-date-digit")

        wrong, wrong_label = apply_ocr_corruption(text, "wrong-date-digit", reverse=True)
        self.assertEqual(wrong, "Geboren 7. März 1942; gestorben 08.11.2026; um 10.00 Uhr.")
        self.assertEqual(wrong_label, "ocr-wrong-date-digit")

        unchanged, label = apply_ocr_corruption("Treffen um 10.00 Uhr", "missing-date-digit")
        self.assertEqual(unchanged, "Treffen um 10.00 Uhr")
        self.assertIsNone(label)

    def test_vocabulary_is_training_only_sorted_and_covers_evaluation(self) -> None:
        symbols = self.manifest["vocabulary"]["symbols"]
        training_characters = sorted(
            {character for record in self.records if record.split == "train" for character in record.text}
        )
        evaluation_characters = {
            character for record in self.records if record.split != "train" for character in record.text
        }

        self.assertEqual(symbols[:2], [BOS_TOKEN, EOS_TOKEN])
        self.assertEqual(symbols[2:], training_characters)
        self.assertTrue(evaluation_characters <= set(training_characters))

    def test_bigram_events_reset_at_every_document_boundary(self) -> None:
        self.assertEqual(
            list(iter_bigram_events("abc")),
            [(BOS_TOKEN, "a"), ("a", "b"), ("b", "c"), ("c", EOS_TOKEN)],
        )
        empty_events = list(iter_bigram_events(""))
        self.assertEqual(empty_events, [(BOS_TOKEN, EOS_TOKEN)])

        records = self.records[:2]
        events = list(iter_corpus_bigram_events(records))
        for record in records:
            document_events = [event for event in events if event[0] == record.document_id]
            expected = [
                (record.document_id, source, target)
                for source, target in iter_bigram_events(record.text)
            ]
            self.assertEqual(document_events, expected)

    def test_fingerprint_changes_with_text_or_split_assignment(self) -> None:
        text_changed = list(self.records)
        text_changed[0] = replace(text_changed[0], text=text_changed[0].text + "!")
        text_manifest = build_manifest(text_changed, self.config)
        self.assertNotEqual(
            self.manifest["corpus_fingerprint"]["value"],
            text_manifest["corpus_fingerprint"]["value"],
        )

        source_group = next(record.group_id for record in self.records if record.split == "train")
        split_changed = [
            replace(record, split="validation") if record.group_id == source_group else record
            for record in self.records
        ]
        split_manifest = build_manifest(split_changed, self.config)
        self.assertNotEqual(
            self.manifest["corpus_fingerprint"]["value"],
            split_manifest["corpus_fingerprint"]["value"],
        )

    def test_manifest_has_only_aggregate_synthetic_provenance(self) -> None:
        serialized = json.dumps(self.manifest, ensure_ascii=False).lower()
        self.assertEqual(self.manifest["privacy"], "synthetic")
        self.assertEqual(self.manifest["generator_version"], GENERATOR_VERSION)
        self.assertNotIn("artifacts/", serialized)
        self.assertNotIn("aichacher", serialized)
        self.assertNotIn(str(Path.cwd()).lower(), serialized)
        self.assertNotIn('"text"', serialized)

    def test_write_corpus_is_canonical_and_refuses_implicit_overwrite(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "synthetic-v2"
            manifest = write_corpus(output_dir, self.config)
            corpus_bytes = (output_dir / "corpus.jsonl").read_bytes()
            disk_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest, disk_manifest)
            self.assertEqual(
                hashlib.sha256(corpus_bytes).hexdigest(),
                manifest["corpus_fingerprint"]["value"],
            )
            self.assertEqual(len(corpus_bytes.splitlines()), len(self.records))
            with self.assertRaises(FileExistsError):
                write_corpus(output_dir, self.config)

            overwritten = write_corpus(output_dir, self.config, overwrite=True)
            self.assertEqual(overwritten, manifest)

    def test_cli_writes_corpus_and_requires_force_to_replace_it(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "cli-output"
            output = io.StringIO()
            with redirect_stdout(output):
                first_status = main(["--output-dir", str(output_dir)])
                second_status = main(["--output-dir", str(output_dir)])
                forced_status = main(["--output-dir", str(output_dir), "--force"])

            self.assertEqual(first_status, 0)
            self.assertEqual(second_status, 2)
            self.assertEqual(forced_status, 0)
            self.assertTrue((output_dir / "corpus.jsonl").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertIn("synthetic documents", output.getvalue())

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_records(GeneratorConfig(families_per_template=0))
        with self.assertRaises(ValueError):
            generate_records(
                GeneratorConfig(train_percent=80, validation_percent=15, test_percent=10)
            )
