"""Generate a deterministic, privacy-safe German death-notice corpus.

The generator deliberately has no input path and never reads the project's OCR
artifacts.  Its templates, lexicons, and fictional entities are maintained here
as synthetic teaching data for Volume I.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence


GENERATOR_VERSION = "v1"
SCHEMA_VERSION = 1
SPLIT_ID = "synthetic-v1-grouped"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SUBSETS = ("train", "validation", "test")
VARIANTS = ("clean", "light", "heavy")

MONTHS = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)
WEEKDAYS = (
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
)

# These values are independently authored for this synthetic corpus. They are
# not extracted from, or checked against, private OCR text.
FEMALE_GIVEN_NAMES = (
    "Alwina",
    "Belinda",
    "Cäcilia",
    "Dorina",
    "Svea",
    "Fenja",
    "Gertrud",
    "Hedda",
    "Irmelin",
    "Jorinde",
    "Klara",
    "Liesbeth",
    "Melitta",
    "Nora",
    "Ottilie",
    "Pauline",
)
MALE_GIVEN_NAMES = (
    "Alwin",
    "Benedikt",
    "Cornelius",
    "Detlev",
    "Egon",
    "Falko",
    "Gero",
    "Hannes",
    "Ingmar",
    "Jasper",
    "Konrad",
    "Leander",
    "Marten",
    "Norbert",
    "Ortwin",
    "Quirin",
)
SURNAME_PREFIXES = (
    "Abend",
    "Birken",
    "Drossel",
    "Eichen",
    "Finken",
    "Glocken",
    "Heide",
    "Kiesel",
    "Lerchen",
    "Morgen",
    "Nebel",
    "Quellen",
    "Rosen",
    "Sonnen",
    "Tannen",
    "Wolken",
    "Zirben",
    "Auen",
    "Felsen",
    "Wiesen",
)
SURNAME_SUFFIXES = (
    "bach",
    "berger",
    "born",
    "brück",
    "feld",
    "furt",
    "hain",
    "horst",
    "kamm",
    "ried",
    "steg",
    "tal",
    "wald",
    "wart",
    "winkel",
    "quell",
    "grund",
    "pfad",
    "rain",
    "höhe",
)
TOWN_PREFIXES = (
    "Birken",
    "Dämmer",
    "Falken",
    "Hirsch",
    "Linden",
    "Mühlen",
    "Raben",
    "Sonnen",
    "Tannen",
    "Wiesen",
    "Wolken",
    "Zirben",
)
TOWN_SUFFIXES = (
    "au",
    "brück",
    "dorf",
    "fels",
    "hagen",
    "kirchen",
    "ried",
    "tal",
    "werth",
    "winkel",
    "zell",
    "höhe",
)
OCCUPATIONS = (
    "Buchbinderin",
    "Drechsler",
    "Gärtnerin",
    "Instrumentenbauer",
    "Keramikerin",
    "Korbflechter",
    "Schneiderin",
    "Uhrmacher",
)
TITLES = ("Dr. ", "Prof. ", "Dipl.-Ing. ")
TEMPLATE_IDS = tuple(f"template{index:02d}" for index in range(1, 9))
CORRUPTION_KINDS = (
    "rn-m",
    "one-ell",
    "dropped-punctuation",
    "merged-spaces",
    "broken-umlaut",
)


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 17
    families_per_template: int = 50
    train_percent: int = 80
    validation_percent: int = 10
    test_percent: int = 10
    generator_version: str = GENERATOR_VERSION

    def validate(self) -> None:
        if self.families_per_template < 1:
            raise ValueError("families_per_template must be positive")
        percentages = (
            self.train_percent,
            self.validation_percent,
            self.test_percent,
        )
        if any(value < 0 for value in percentages) or sum(percentages) != 100:
            raise ValueError("split percentages must be non-negative and total 100")
        if self.train_percent == 0:
            raise ValueError("the training split must be non-empty")


@dataclass(frozen=True)
class NoticeRecord:
    document_id: str
    group_id: str
    split: str
    text: str
    slices: tuple[str, ...]
    privacy: str = "synthetic"
    generator_version: str = GENERATOR_VERSION

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["slices"] = list(self.slices)
        return result


@dataclass(frozen=True)
class _NoticeFacts:
    given_name: str
    surname: str
    maiden_surname: str | None
    title: str
    occupation: str | None
    town: str
    birth_date: date
    death_date: date
    funeral_date: date
    funeral_hour: int
    relative_names: tuple[str, str]
    female: bool
    date_style: int

    @property
    def full_name(self) -> str:
        return f"{self.title}{self.given_name} {self.surname}".strip()


def _stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _family_surname(group_index: int) -> str:
    prefix = SURNAME_PREFIXES[group_index % len(SURNAME_PREFIXES)]
    suffix = SURNAME_SUFFIXES[(group_index // len(SURNAME_PREFIXES)) % len(SURNAME_SUFFIXES)]
    return prefix + suffix


def _town(group_index: int, rng: random.Random) -> str:
    prefix = TOWN_PREFIXES[(group_index + rng.randrange(len(TOWN_PREFIXES))) % len(TOWN_PREFIXES)]
    suffix = TOWN_SUFFIXES[(group_index * 3 + rng.randrange(len(TOWN_SUFFIXES))) % len(TOWN_SUFFIXES)]
    return prefix + suffix


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def _make_facts(config: GeneratorConfig, group_id: str, group_index: int) -> _NoticeFacts:
    rng = random.Random(_stable_seed(config.seed, group_id, "facts"))
    female = bool(rng.randrange(2))
    names = FEMALE_GIVEN_NAMES if female else MALE_GIVEN_NAMES
    other_names = MALE_GIVEN_NAMES if female else FEMALE_GIVEN_NAMES
    given_name = names[rng.randrange(len(names))]
    surname = _family_surname(group_index)
    maiden_surname = None
    if female and rng.random() < 0.35:
        maiden_surname = _family_surname((group_index + 137) % 400)
    title = TITLES[rng.randrange(len(TITLES))] if rng.random() < 0.16 else ""
    occupation = OCCUPATIONS[rng.randrange(len(OCCUPATIONS))] if rng.random() < 0.45 else None
    birth_date = _random_date(rng, date(1925, 1, 1), date(1975, 12, 31))
    death_date = _random_date(rng, date(2022, 1, 1), date(2025, 12, 31))
    funeral_date = death_date + timedelta(days=rng.randrange(4, 15))
    relative_names = (
        other_names[rng.randrange(len(other_names))],
        names[rng.randrange(len(names))],
    )
    return _NoticeFacts(
        given_name=given_name,
        surname=surname,
        maiden_surname=maiden_surname,
        title=title,
        occupation=occupation,
        town=_town(group_index, rng),
        birth_date=birth_date,
        death_date=death_date,
        funeral_date=funeral_date,
        funeral_hour=rng.randrange(9, 18),
        relative_names=relative_names,
        female=female,
        date_style=rng.randrange(4),
    )


def _format_date(value: date, style: int) -> str:
    if style == 0:
        return f"{value.day}. {MONTHS[value.month - 1]} {value.year}"
    if style == 1:
        return f"{value.day:02d}.{value.month:02d}.{value.year}"
    if style == 2:
        return f"{value.day}.{value.month}.{value.year}"
    return f"{value.day:02d}. {MONTHS[value.month - 1]} {value.year}"


def _name_block(facts: _NoticeFacts) -> list[str]:
    lines = [facts.full_name]
    if facts.maiden_surname:
        lines.append(f"geb. {facts.maiden_surname}")
    if facts.occupation:
        lines.append(facts.occupation)
    lines.append(
        f"* {_format_date(facts.birth_date, facts.date_style)}   "
        f"† {_format_date(facts.death_date, facts.date_style)}"
    )
    return lines


def _ceremony(facts: _NoticeFacts, kind: str = "Trauerfeier") -> str:
    funeral_date = _format_date(facts.funeral_date, (facts.date_style + 1) % 4)
    return (
        f"Die {kind} findet am {WEEKDAYS[facts.funeral_date.weekday()]}, "
        f"den {funeral_date}, um {facts.funeral_hour}.00 Uhr in {facts.town} statt."
    )


def _render_notice(facts: _NoticeFacts, template_index: int) -> str:
    first, second = facts.relative_names
    parent = "Mutter" if facts.female else "Vater"
    spouse = "Ehefrau" if facts.female else "Ehemann"
    sibling = "Schwester" if facts.female else "Bruder"
    name = _name_block(facts)

    if template_index == 0:
        sections = [
            "In Liebe und Dankbarkeit nehmen wir Abschied von",
            "\n".join(name),
            f"{facts.town}\nIn stiller Trauer: {first} und {second}\nim Namen aller Angehörigen",
            _ceremony(facts),
        ]
    elif template_index == 1:
        sections = [
            f"Ein erfülltes Leben ist zu Ende gegangen. Wir trauern um unsere {parent}.",
            "\n".join(name),
            f"Du bleibst in unseren Herzen.\n{first}, {second} und alle Verwandten",
            _ceremony(facts, "Urnenbeisetzung"),
        ]
    elif template_index == 2:
        sections = [
            "Nach langer Lebensreise durfte ein lieber Mensch friedlich heimgehen.",
            "\n".join(name),
            f"Für die gemeinsame Zeit danken: {first} und {second} mit Familien.",
            _ceremony(facts, "Beerdigung"),
        ]
    elif template_index == 3:
        sections = [
            f"Wir müssen Abschied nehmen von unserem geliebten {sibling}.",
            "\n".join(name),
            f"{facts.town}, im Namen der Familie\n{first} und {second}",
            "Von Beileidsbekundungen am Grab bitten wir abzusehen.",
            _ceremony(facts, "Trauergottesdienst"),
        ]
    elif template_index == 4:
        sections = [
            "Was bleibt, sind Erinnerung, Dankbarkeit und Liebe.",
            "\n".join(name),
            f"Wir vermissen dich.\nDeine Familie {facts.surname}",
            _ceremony(facts),
            "Für erwiesene Anteilnahme danken wir herzlich.",
        ]
    elif template_index == 5:
        sections = [
            f"Plötzlich und unerwartet verstarb unsere liebe {spouse}.",
            "\n".join(name),
            f"In tiefer Trauer: {first}, {second} und die Angehörigen",
            _ceremony(facts, "Abschiedsfeier"),
        ]
    elif template_index == 6:
        sections = [
            "Ein gütiges Herz hat aufgehört zu schlagen.",
            "\n".join(name),
            f"In liebevoller Erinnerung\n{first} und {second}\nmit allen Verwandten",
            _ceremony(facts, "Seelengottesdienst"),
            "Die Beisetzung erfolgt anschließend im engsten Familienkreis.",
        ]
    else:
        sections = [
            "Dankbar für viele schöne Jahre sagen wir leise Lebewohl.",
            "\n".join(name),
            f"{facts.town}\nFür immer verbunden: {first}, {second} und Familien",
            _ceremony(facts, "Urnenfeier"),
            "Anstelle von Blumen bitten wir um eine gute Tat für einen Menschen in Not.",
        ]
    return "\n\n".join(sections)


def _semantic_slices(text: str, facts: _NoticeFacts) -> set[str]:
    slices = {"date", "relationship", "ceremony"}
    slices.add("date-written" if MONTHS[facts.birth_date.month - 1] in text else "date-numeric")
    if facts.title:
        slices.add("title")
    if facts.maiden_surname:
        slices.add("maiden-name")
    if facts.occupation:
        slices.add("occupation")
    if any(character in text for character in "ÄÖÜäöüß"):
        slices.add("umlaut")
    return slices


def _replace_rn_m(text: str, reverse: bool) -> tuple[str, str | None]:
    old, new, label = ("m", "rn", "ocr-m-to-rn") if reverse else ("rn", "m", "ocr-rn-to-m")
    if old not in text:
        old, new, label = ("rn", "m", "ocr-rn-to-m") if reverse else ("m", "rn", "ocr-m-to-rn")
    changed = text.replace(old, new, 1)
    return changed, label if changed != text else None


def _replace_one_ell(text: str, reverse: bool) -> tuple[str, str | None]:
    old, new, label = ("l", "1", "ocr-l-to-1") if reverse else ("1", "l", "ocr-1-to-l")
    if old not in text:
        old, new, label = ("1", "l", "ocr-1-to-l") if reverse else ("l", "1", "ocr-l-to-1")
    changed = text.replace(old, new, 1)
    return changed, label if changed != text else None


def _drop_punctuation(text: str) -> tuple[str, str | None]:
    punctuation = set(".,:;!?")
    position = next((index for index, character in enumerate(text) if character in punctuation), None)
    if position is None:
        return text, None
    return text[:position] + text[position + 1 :], "ocr-dropped-punctuation"


def _merge_spaces(text: str) -> tuple[str, str | None]:
    for position in range(1, len(text) - 1):
        if text[position] == " " and text[position - 1].isalpha() and text[position + 1].isalpha():
            return text[:position] + text[position + 1 :], "ocr-merged-spaces"
    return text, None


def _break_umlaut(text: str) -> tuple[str, str | None]:
    replacements = str.maketrans({"Ä": "A", "Ö": "O", "Ü": "U", "ä": "a", "ö": "o", "ü": "u"})
    for position, character in enumerate(text):
        replacement = character.translate(replacements)
        if replacement != character:
            return text[:position] + replacement + text[position + 1 :], "ocr-broken-umlaut"
    return text, None


def apply_ocr_corruption(text: str, kind: str, *, reverse: bool = False) -> tuple[str, str | None]:
    """Apply one deterministic corruption and return its slice label."""

    if kind == "rn-m":
        return _replace_rn_m(text, reverse)
    if kind == "one-ell":
        return _replace_one_ell(text, reverse)
    if kind == "dropped-punctuation":
        return _drop_punctuation(text)
    if kind == "merged-spaces":
        return _merge_spaces(text)
    if kind == "broken-umlaut":
        return _break_umlaut(text)
    raise ValueError(f"unknown OCR corruption: {kind}")


def _corrupt(text: str, group_index: int, variant: str) -> tuple[str, set[str]]:
    if variant == "clean":
        return text, set()
    start = group_index % len(CORRUPTION_KINDS)
    count = 1 if variant == "light" else 3
    labels: set[str] = set()
    corrupted = text
    for offset in range(count):
        kind = CORRUPTION_KINDS[(start + offset) % len(CORRUPTION_KINDS)]
        corrupted, label = apply_ocr_corruption(
            corrupted,
            kind,
            reverse=bool((group_index + offset) % 2),
        )
        if label:
            labels.add(label)
    return corrupted, labels


def assign_split(group_id: str, config: GeneratorConfig) -> str:
    """Assign a complete group using a stable SHA-256 bucket."""

    bucket = _stable_seed(config.seed, group_id, "split") % 10_000
    train_cutoff = config.train_percent * 100
    validation_cutoff = train_cutoff + config.validation_percent * 100
    if bucket < train_cutoff:
        return "train"
    if bucket < validation_cutoff:
        return "validation"
    return "test"


def generate_records(config: GeneratorConfig = GeneratorConfig()) -> tuple[NoticeRecord, ...]:
    """Return the complete deterministic corpus in document-id order."""

    config.validate()
    records: list[NoticeRecord] = []
    for template_index, template_id in enumerate(TEMPLATE_IDS):
        for family_index in range(config.families_per_template):
            group_id = f"{template_id}-family{family_index:03d}"
            group_index = template_index * config.families_per_template + family_index
            facts = _make_facts(config, group_id, group_index)
            clean_text = _render_notice(facts, template_index)
            base_slices = _semantic_slices(clean_text, facts)
            split = assign_split(group_id, config)
            for variant_index, variant in enumerate(VARIANTS):
                text, corruption_slices = _corrupt(clean_text, group_index, variant)
                records.append(
                    NoticeRecord(
                        document_id=f"syn-{group_id}-{variant_index:03d}",
                        group_id=group_id,
                        split=split,
                        text=text,
                        slices=tuple(sorted(base_slices | corruption_slices | {f"variant-{variant}"})),
                        generator_version=config.generator_version,
                    )
                )
    return tuple(sorted(records, key=lambda record: record.document_id))


def iter_bigram_events(text: str) -> Iterator[tuple[str, str]]:
    """Yield one document's character events with explicit boundaries."""

    previous = BOS_TOKEN
    for character in text:
        yield previous, character
        previous = character
    yield previous, EOS_TOKEN


def iter_corpus_bigram_events(
    records: Iterable[NoticeRecord],
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(document_id, input, target)`` while resetting each document."""

    for record in records:
        for input_symbol, target_symbol in iter_bigram_events(record.text):
            yield record.document_id, input_symbol, target_symbol


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_records(records: Sequence[NoticeRecord]) -> bytes:
    ordered = sorted(records, key=lambda record: record.document_id)
    return "".join(f"{_canonical_json(record.to_dict())}\n" for record in ordered).encode("utf-8")


def _counts_by_subset(records: Sequence[NoticeRecord], measure) -> dict[str, int]:
    return {subset: sum(measure(record) for record in records if record.split == subset) for subset in SUBSETS}


def build_manifest(
    records: Sequence[NoticeRecord],
    config: GeneratorConfig = GeneratorConfig(),
) -> dict[str, object]:
    """Build the public manifest and validate corpus invariants."""

    config.validate()
    if not records:
        raise ValueError("cannot build a manifest for an empty corpus")
    if len({record.document_id for record in records}) != len(records):
        raise ValueError("document IDs must be unique")
    if any(record.split not in SUBSETS for record in records):
        raise ValueError("records contain an unknown split")
    if any(not record.text for record in records):
        raise ValueError("documents must not be empty")
    if any(record.privacy != "synthetic" for record in records):
        raise ValueError("all records must have synthetic privacy provenance")
    if any(record.generator_version != config.generator_version for record in records):
        raise ValueError("record and manifest generator versions must match")

    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaked_groups = sorted(group_id for group_id, splits in group_splits.items() if len(splits) != 1)
    if leaked_groups:
        raise ValueError(f"groups span multiple splits: {leaked_groups[:3]}")

    training_characters = sorted({character for record in records if record.split == "train" for character in record.text})
    evaluation_characters = {character for record in records if record.split != "train" for character in record.text}
    missing = sorted(evaluation_characters - set(training_characters))
    if missing:
        raise ValueError(f"evaluation characters absent from training vocabulary: {missing!r}")

    document_counts = _counts_by_subset(records, lambda _: 1)
    character_counts = _counts_by_subset(records, lambda record: len(record.text))
    slice_counts: dict[str, dict[str, int]] = {}
    for slice_name in sorted({name for record in records for name in record.slices}):
        per_subset = {
            subset: sum(slice_name in record.slices for record in records if record.split == subset)
            for subset in SUBSETS
        }
        slice_counts[slice_name] = {"total": sum(per_subset.values()), **per_subset}

    corpus_bytes = serialize_records(records)
    vocabulary = [BOS_TOKEN, EOS_TOKEN, *training_characters]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": config.generator_version,
        "privacy": "synthetic",
        "split_id": SPLIT_ID,
        "seed": config.seed,
        "configuration": {
            "templates": list(TEMPLATE_IDS),
            "families_per_template": config.families_per_template,
            "variants": list(VARIANTS),
            "corruptions": list(CORRUPTION_KINDS),
            "birth_date_range": ["1925-01-01", "1975-12-31"],
            "death_date_range": ["2022-01-01", "2025-12-31"],
            "lexicon_provenance": "independently-authored-synthetic",
        },
        "split_rule": {
            "algorithm": "sha256",
            "key": "seed:group_id:split",
            "bucket_count": 10_000,
            "percentages": {
                "train": config.train_percent,
                "validation": config.validation_percent,
                "test": config.test_percent,
            },
        },
        "documents": {"total": len(records), "by_subset": document_counts},
        "characters": {"total": sum(character_counts.values()), "by_subset": character_counts},
        "slice_counts": slice_counts,
        "boundary_tokens": {"bos": BOS_TOKEN, "eos": EOS_TOKEN},
        "vocabulary": {
            "size": len(vocabulary),
            "ordering": "boundary-tokens-then-unicode-code-point",
            "symbols": vocabulary,
        },
        "corpus_fingerprint": {
            "algorithm": "sha256",
            "value": hashlib.sha256(corpus_bytes).hexdigest(),
        },
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_corpus(
    output_dir: Path | str,
    config: GeneratorConfig = GeneratorConfig(),
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate and atomically write ``corpus.jsonl`` and ``manifest.json``."""

    output_dir = Path(output_dir)
    corpus_path = output_dir / "corpus.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (corpus_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {joined}")

    records = generate_records(config)
    manifest = build_manifest(records, config)
    corpus_bytes = serialize_records(records)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(corpus_path, corpus_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument(
        "--families-per-template",
        type=int,
        default=GeneratorConfig.families_per_template,
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing corpus files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = GeneratorConfig(seed=args.seed, families_per_template=args.families_per_template)
    try:
        manifest = write_corpus(args.output_dir, config, overwrite=args.force)
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}")
        return 2
    fingerprint = manifest["corpus_fingerprint"]["value"]
    print(
        f"wrote {manifest['documents']['total']} synthetic documents to {args.output_dir} "
        f"(sha256:{fingerprint})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
