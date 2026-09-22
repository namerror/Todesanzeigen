# Volume I, Chapter 5 Notes: Building a Synthetic Corpus

Chapter 5 is about constructing the **data contract** for a tiny language model. Before training a model, we need to decide what examples exist, which examples the model may learn from, how characters are represented, and how an experiment can be reproduced.

The implementation is in [`synthetic_notice_generator.py`](./synthetic_notice_generator.py), and its structural overview is in [`SYNTHETIC_GENERATOR_STRUCTURE.md`](./SYNTHETIC_GENERATOR_STRUCTURE.md).

## The big picture

The generator creates fictional German death notices. It deliberately does not read real OCR files or private identities.

```text
authored word lists + template + deterministic random facts
                         ↓
                 one fictional notice
                         ↓
          clean, light-OCR, and heavy-OCR variants
                         ↓
             train / validation / test split
                         ↓
              corpus.jsonl + manifest.json
                         ↓
             character-level model examples
```

With the default settings, the corpus contains:

```text
8 templates × 50 families per template × 3 variants = 1,200 documents
```

Here, “family” is a generator concept. It does not refer to a real family.

## What is a corpus?

A **corpus** is the collection of text used to train and evaluate a language model. In this project, `corpus.jsonl` is a sequence of synthetic notice records.

Each line is one JSON object resembling:

```json
{
  "document_id": "syn-template03-family017-002",
  "group_id": "template03-family017",
  "split": "train",
  "text": "...the generated notice...",
  "slices": ["ceremony", "date", "ocr-rn-to-m", "variant-heavy"],
  "privacy": "synthetic",
  "generator_version": "v2"
}
```

The model learns from `text`. The other fields describe and organize the example; they are not part of the notice presented to the model.

The corpus serves several purposes:

- It supplies training examples.
- It supplies separate validation and test examples.
- It creates controlled cases such as dates, umlauts, and OCR errors.
- It is safe to commit because its people, towns, and notices are invented.
- It is reproducible: the same configuration produces exactly the same bytes.

This is **synthetic data**, meaning data created by a program rather than collected from real notices. Synthetic data is useful for learning mechanics and writing tests, although it will not perfectly reproduce the distribution of real OCR data.

## What is a document or record?

A **document** is one complete notice variant. A `NoticeRecord` stores its text and metadata. Its `document_id` uniquely identifies that exact variant.

The three variants of one underlying notice are three documents:

```text
syn-template03-family017-000  clean
syn-template03-family017-001  light corruption
syn-template03-family017-002  heavy corruption
```

These documents are related, so the generator must prevent them from being scattered across different dataset splits.

## What is a template?

A **template** is a reusable notice layout. It defines the broad wording and ordering of sections while leaving slots for generated facts.

A simplified template might be:

```text
In Liebe nehmen wir Abschied von
{full_name}
* {birth_date}  † {death_date}
In stiller Trauer: {relative_names}
Die Trauerfeier findet am {funeral_date} in {town} statt.
```

The generator has eight templates. They provide structural variety: for example, one may mention a funeral, another an urn burial, and another a memorial service. `_render_notice()` chooses the wording, while helpers such as `_name_block()`, `_format_date()`, and `_ceremony()` fill in the facts.

A template is not itself a document. It becomes a document only after its slots are filled with one set of facts.

## What is a family?

In this generator, a **family** means one fictional set of underlying notice facts created within one template:

- name and optional title;
- optional maiden name and occupation;
- birth, death, and funeral dates;
- town and relatives;
- formatting choices.

The identifier `template03-family017` means “generated fact set 17 for template 3.” The same facts and clean text are then used to create the clean, light, and heavy OCR variants.

This naming can be slightly misleading: a family is best understood as an **example group** or **notice group**, not a real household.

## What are variants and OCR corruptions?

Each family produces three **variants**:

| Variant | Corruptions applied | Purpose |
|---|---:|---|
| `clean` | 0 | Original synthetic target text |
| `light` | 1 | Mild OCR-like noise |
| `heavy` | Up to 3 | More difficult OCR-like noise |

The supported corruptions imitate common OCR mistakes:

- `rn ↔ m`, because those shapes can look similar;
- `1 ↔ l`, confusing the digit one and lowercase L;
- dropped punctuation;
- merged spaces;
- umlauts simplified, such as `ü → u`;
- a digit missing from a recognized birth, death, or funeral date;
- a digit replaced with a different digit inside a recognized date.

Date corruption is deliberately limited to the four date formats emitted by the
generator. It does not mistake ceremony times or unrelated numbers for dates.

The corruption schedule is deterministic. It depends on the group index rather than uncontrolled randomness, so rerunning the generator creates the same variants.

## What is a group ID, and why have one?

`group_id` identifies documents that share the same underlying example. All three variants from one family have the same group ID:

```text
group_id: template03-family017
    ├── clean document
    ├── light-corruption document
    └── heavy-corruption document
```

Without grouping, the clean version could enter training while its almost-identical noisy version entered validation. The model might then appear to generalize when it has effectively already seen the answer.

This problem is called **data leakage**. Leakage makes evaluation results unrealistically good and therefore untrustworthy.

`document_id` identifies one exact document; `group_id` connects related documents.

## What is a dataset split?

A **split** partitions the corpus by purpose:

| Split | Default share | Role |
|---|---:|---|
| Training | 80% | Fit counts or model parameters |
| Validation | 10% | Compare designs and tune choices |
| Test | 10% | Perform a final, relatively unbiased evaluation |

“Split by group” means assigning the entire group—not individual documents or character windows—to one split. If `template03-family017` is assigned to validation, all three of its variants are validation documents.

The test split should remain **sealed** while making modeling decisions. Repeatedly checking test performance and changing the model in response gradually turns the test set into another validation set.

## How is a group assigned to a split?

`assign_split()` hashes the seed, `group_id`, and the word `split` with SHA-256. The resulting number is mapped to one of 10,000 buckets:

```text
0–7999     → train
8000–8999  → validation
9000–9999  → test
```

This is a **deterministic hash split**:

- the same seed and group ID always produce the same assignment;
- every document in a group gets the same assignment;
- assignment does not depend on the order in which records are generated.

The percentages are targets, not promises of exact counts. Hashing approximates the requested ratio, especially as the number of groups grows.

## What is a random seed, and what does deterministic mean?

A pseudo-random number generator produces a sequence that looks random but is completely determined by its **seed**. The generator derives local seeds from the configured seed and stable identifiers using SHA-256.

Consequently:

- the same code, configuration, and seed produce the same corpus;
- changing the seed changes generated facts and split assignments;
- each group has its own stable random stream.

This is **reproducibility**: another run can recreate the same experimental input. It is important because a metric cannot be compared fairly if the data silently changes between runs.

## What is the manifest?

The corpus contains the examples. The **manifest** describes the corpus as a whole. `manifest.json` records:

- generator and schema versions;
- configuration and seed;
- template, variant, and corruption names;
- split algorithm and percentages;
- document and character counts by split;
- counts for each slice;
- vocabulary and its ordering;
- boundary-token definitions;
- privacy classification;
- a SHA-256 fingerprint of `corpus.jsonl`.

The manifest answers questions such as:

```text
Which data produced this model?
How was it split?
Which symbols did the model know?
Was the corpus changed?
Can another run reconstruct the setup?
```

It deliberately contains aggregate metadata, not notice text or private file paths.

### Corpus versus manifest

| File | Contains | Used for |
|---|---|---|
| `corpus.jsonl` | Individual records and text | Training and evaluation |
| `manifest.json` | Aggregate description and identity | Reproducibility, validation, and provenance |

An analogy: the corpus is the contents of a shipment; the manifest is the inventory and tracking information attached to it.

## What is a corpus fingerprint?

The **fingerprint** is a SHA-256 hash of the exact serialized corpus bytes. A tiny content change produces a different fingerprint.

It is useful for identifying data, not for understanding it. Matching fingerprints provide strong evidence that two runs used byte-for-byte identical corpus files. A fingerprint changes if text, metadata such as a split assignment, ordering, or serialization changes.

The fingerprint is not encryption and does not make private data safe. That is why the generator avoids private data in the first place.

## What is a slice?

A **slice** labels a meaningful subset of records so performance can be inspected separately. Examples include:

- semantic slices: `date`, `ceremony`, `relationship`, `occupation`;
- text-property slices: `umlaut`, `date-written`, `date-numeric`;
- corruption slices: `ocr-rn-to-m`, `ocr-merged-spaces`;
- variant slices: `variant-clean`, `variant-light`, `variant-heavy`.

An overall average can hide a serious weakness. For example, model loss might be acceptable overall but much worse on broken umlauts. Slice metrics expose that difference.

A record may belong to several slices at once. Slices are labels for analysis, not mutually exclusive dataset partitions.

## What is a vocabulary?

The **vocabulary** is the complete set of symbols the model can represent. This chapter uses characters, so entries include letters, digits, whitespace, punctuation, Unicode characters such as `ä`, and the boundary tokens.

The vocabulary is built from training text only:

```text
<BOS>, <EOS>, then training characters sorted by Unicode code point
```

Validation and test characters must all occur in this training vocabulary. Otherwise the model would encounter a symbol for which no input or output index exists. Later systems often introduce an unknown token, but this chapter explicitly checks coverage instead.

### Why does vocabulary order matter?

Models operate on numeric IDs, not strings. For example:

```text
0 → <BOS>
1 → <EOS>
2 → newline
3 → space
...
```

Changing the order changes what every index means. A saved model weight associated with index 17 remains at index 17, even if a reordered vocabulary now assigns that index to a different character. Therefore, vocabulary order is part of model identity and must be recorded.

## Why start with characters?

A character model treats a notice as a sequence of individual characters. This is inefficient but transparent:

- there is no tokenizer to learn or debug yet;
- any invented surname can be represented from known characters;
- OCR errors can be observed directly;
- the next-character training events are easy to inspect by hand.

The tradeoff is sequence length. Character sequences are usually much longer than sequences produced by a subword tokenizer such as BPE. Longer sequences require more model steps, and in later Transformer models they make attention more expensive.

## What is a sentinel or boundary token?

A **sentinel** is a special symbol that represents structure rather than ordinary text. This implementation uses two boundary tokens:

- `<BOS>`: beginning of sequence/document;
- `<EOS>`: end of sequence/document.

They are conceptual vocabulary symbols, not the literal characters `<`, `B`, `O`, `S`, and `>` in the notice.

For the document `abc`, `iter_bigram_events()` emits:

```text
<BOS> → a
a     → b
b     → c
c     → <EOS>
```

Boundary tokens let the model learn which characters commonly begin a notice and when a notice should end. Separate BOS and EOS tokens give the two roles different meanings, although a simple bigram model could use one shared sentinel.

## Why must events reset at document boundaries?

Suppose the corpus contains two documents, `abc` and `xyz`. Correct events include:

```text
<BOS> → a ... c → <EOS>
<BOS> → x ... z → <EOS>
```

There must not be an event `c → x`. Those characters are in unrelated notices; their adjacency is only an artifact of file storage. `iter_corpus_bigram_events()` calls `iter_bigram_events()` separately for every record to enforce this rule.

## What is a bigram?

A **bigram** is a pair of adjacent symbols. In a character model, examples include `a → b`, `e → r`, or newline `→ I`.

The first symbol is the context and the second is the target to predict:

```text
input/context: a
target:        b
```

A document of length `n` creates `n + 1` bigram events because BOS and EOS add the boundary transitions.

## What is a bigram language model?

A **bigram language model** predicts the next symbol using only the current symbol:

$$
P(x_{t+1} \mid x_t)
$$

For example, after seeing `q`, a German character model should assign high probability to `u`. A bigram model does not remember earlier characters, so it cannot distinguish contexts that end in the same character.

A count bigram model learns by counting each transition. If $N_{ij}$ is the number of times symbol $j$ follows symbol $i$, then without smoothing:

$$
P(j \mid i) = \frac{N_{ij}}{\sum_k N_{ik}}
$$

This model has a table with shape `[V, V]`, where `V` is vocabulary size:

- each row represents the current character;
- each column represents a possible next character;
- each normalized row is a probability distribution.

Chapter 6 builds this count model. Chapter 7 builds a neural version. Both use the events and vocabulary contract established here.

## What does “the language model sees only text” mean?

Fields such as `split`, `slices`, and `group_id` control the experiment but should not become inputs to the basic language model. Otherwise the model would solve a different task or exploit information unavailable in normal use.

The training pipeline uses metadata to select records, then converts only each selected record's `text` into character IDs and input-target events.

```text
metadata → decides whether/how to evaluate the record
text     → becomes the model's sequence
```

## Why write files atomically?

`_atomic_write()` first writes a complete temporary file, then replaces the destination with `os.replace()`. This reduces the risk of leaving a half-written corpus or manifest if writing fails.

The program also refuses to overwrite existing output unless `--force` is supplied. That guards against accidentally replacing the data used by an earlier experiment.

## How the important concepts connect

1. A **template** supplies a notice structure.
2. A **family** supplies one fictional set of facts for that template.
3. Three OCR **variants** are made from that underlying notice.
4. Their shared **group ID** keeps them together.
5. The group is assigned to one **split** to prevent leakage.
6. All records form the **corpus**.
7. **Slices** describe cases worth measuring separately.
8. The **manifest** records the corpus configuration and identity.
9. The training vocabulary maps characters and **sentinels** to indices.
10. Each document becomes separate **bigram events** for the next-character model.

## Answers to the chapter's prediction questions

### Will a random-window split look better than a grouped split?

Usually yes, but for the wrong reason. Near-identical material can appear in both training and validation, so validation loss benefits from leakage. The grouped split is harder and more honest.

### Which OCR corruption increases vocabulary size most?

For this particular implementation, the listed corruptions mostly replace or remove characters that already occur elsewhere. The exact effect must be measured rather than assumed. In general, a corruption that introduces new symbols increases vocabulary size; dropped punctuation and merged spaces cannot introduce new symbols.

### Are rare surnames easier to represent with characters or words?

Characters make them easier to **represent** because a new surname can be composed from known letters instead of requiring a dedicated word-vocabulary entry. They do not automatically make the name easy to predict: a bigram has too little context to model a whole surname coherently.

### Which sequences are longer: characters or BPE tokens?

Character sequences are normally longer. A BPE token may represent several characters, so it shortens the sequence. Longer sequences require more sequential prediction steps and, for a Transformer, more attention computation and memory.

## Common misconceptions

- **“Family means a real family.”** No. It is one synthetic fact set and its related variants.
- **“A template is copied unchanged.”** No. It is a structure whose slots are filled with generated facts.
- **“The manifest is training data.”** No. It documents and verifies the training data.
- **“The split applies to characters.”** No. It applies to complete correlated groups.
- **“SHA-256 makes private data anonymous.”** No. It supplies stable assignment and identification here; it is not the privacy mechanism.
- **“BOS and EOS are visible notice text.”** No. They are model-level boundary symbols.
- **“A bigram sees two previous characters.”** No. The pair consists of one current symbol and the one next symbol being predicted.
- **“Deterministic means every record is identical.”** No. Records vary, but rerunning with the same inputs reproduces the same variation.

## Quick self-check

You understand this step if you can explain:

1. Why three variants of one notice must share a split.
2. Why the corpus and manifest are separate artifacts.
3. Why vocabulary order must stay fixed when loading a model.
4. Why `c → x` is invalid when `c` ends one document and `x` begins another.
5. What information a bigram model uses—and what information it cannot use.
6. Why a synthetic corpus helps privacy without fully replacing evaluation on real data.
