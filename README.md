# Todesanzeigen

OCR and structured extraction pipeline for German death notice images.

The current workflow has separate steps:

1. Run OCR on images from `input/` and write plain text artifacts to `artifacts/`.
2. Optionally run the local TSV filter to print likely deceased names.
3. Parse OCR/image inputs with OCR+LLM or VLM methods and record structured outputs in SQLite.
4. Review candidates into ground truth, build ML features/evaluation datasets, and export CSV only when needed.

Local Tesseract OCR is the default. Google Document AI OCR is kept as a guarded legacy fallback and is blocked unless explicitly unlocked.

## Setup

Install the Python package in editable mode:

```sh
python -m pip install -e .
```

Install Tesseract. On Ubuntu/Debian:

```sh
sudo apt install tesseract-ocr
```

Add the high-quality German traineddata file at:

```sh
data/deu.traineddata
```

The default OCR command uses that local tessdata directory with German-only OCR:

```text
--tesseract-tessdata-dir data
--tesseract-lang deu
--tesseract-oem 1
--tesseract-psm 3
```

`--tesseract-oem 1` selects Tesseract's LSTM OCR engine. This is the right mode
for `tessdata_best` models such as `data/deu.traineddata`.

For structured extraction, create a `.env` file from `.env.example` and configure an LLM provider.
Gemini remains the default:

```sh
TODESANZEIGEN_LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-lite
```

To use Qwen through Alibaba Cloud Model Studio's OpenAI-compatible API:

```sh
TODESANZEIGEN_LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=your-dashscope-api-key
QWEN_MODEL=qwen3.6-flash
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Qwen base URLs are region-specific. Override `QWEN_BASE_URL` if your API key is
for another region, such as `https://dashscope-us.aliyuncs.com/compatible-mode/v1`
for US Virginia.

Low-confidence OCR/name cases can be sent directly to a vision model. Qwen OCR
is the default reroute provider:

```sh
TODESANZEIGEN_REROUTE_PROVIDER=qwen
TODESANZEIGEN_REROUTE_MODEL=qwen-vl-ocr
QWEN_VISION_MODEL=qwen-vl-ocr
QWEN_VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Gemini and OpenAI vision reroute providers are also supported:

```sh
TODESANZEIGEN_REROUTE_PROVIDER=gemini
GEMINI_VISION_MODEL=gemini-2.5-pro

TODESANZEIGEN_REROUTE_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_VISION_MODEL=gpt-5.6-luna
```

Image-only vision extraction uses the same vision provider implementations. It
defaults to the reroute provider settings, but can be configured separately:

```sh
TODESANZEIGEN_VISION_PROVIDER=qwen
TODESANZEIGEN_VISION_MODEL=qwen-vl-ocr
TODESANZEIGEN_VISION_CONCURRENCY=1
```

## OCR Usage

Run local Tesseract OCR for all supported images in `input/`:

```sh
todesanzeigen ocr
```

By default this writes one `.txt` file per image into `artifacts/`.
It also writes one `.tsv` layout artifact per image. For `example.jpg`, the OCR
step creates:

```text
artifacts/example.txt
artifacts/example.tsv
```

Common options:

```sh
todesanzeigen ocr --limit 10
todesanzeigen ocr --overwrite
todesanzeigen ocr --input-dir input --artifacts-dir artifacts
todesanzeigen ocr --tesseract-lang deu --tesseract-psm 3
todesanzeigen ocr --tesseract-tessdata-dir data --tesseract-oem 1
```

Useful Tesseract page segmentation modes:

- `3`: default; fully automatic page segmentation, useful for notices with multiple text areas.
- `4`: assumes a single column of variable-size text, useful for some newspaper crops.
- `6`: assumes one uniform text block, useful only for tightly cropped clean notices.
- `11`: sparse text mode, useful for messy or loosely cropped images.

Supported image extensions are:

```text
.bmp .gif .jpeg .jpg .png .tif .tiff .webp
```

## TSV Name Filtering Usage

After OCR artifacts exist, run the preliminary local filter:

```sh
todesanzeigen filter
```

This reads `artifacts/*.tsv`, uses the largest visible text lines as the main
signal, and prints one likely name per TSV artifact with the average OCR
confidence of the retained name words. It also writes `artifacts/name_map.json`,
mapping each `.txt` OCR artifact filename to the detected name and confidence.
It does not call an LLM and does not discard low-confidence entries by default.

Common options:

```sh
todesanzeigen filter --limit 10
todesanzeigen filter --artifacts-dir artifacts
todesanzeigen filter --low-confidence-log-file logs/filter-low-confidence.jsonl
todesanzeigen filter --name-confidence-threshold 90 --low-confidence-log-file logs/filter-low-confidence.jsonl
```

The optional low-confidence log is a JSONL report of entries that would be
skipped by extraction under the configured threshold. Missing confidence and
confidence below the threshold are logged; confidence equal to the threshold is
kept. The generated `name_map.json` still contains every TSV result.

## Structured Extraction Usage

After OCR artifacts exist and `todesanzeigen filter` has written
`artifacts/name_map.json`, record OCR+LLM structured outputs in SQLite:

```sh
todesanzeigen extract --source "Aichacher Nachrichten"
```

This reads `artifacts/*.txt`, requires `artifacts/name_map.json`, and writes
records to `state/todesanzeigen.sqlite3`. The LLM receives the OCR text plus the
locally detected name and confidence as a hint; it does not receive the TSV
layout artifact. Artifacts with missing name confidence or name confidence below
85.0 are skipped unless vision reroute is enabled. Each extraction run writes a
timestamped text log under `logs/` and JSONL checkpoints for resumability.

Common options:

```sh
todesanzeigen extract --limit 10
todesanzeigen extract --artifacts-dir artifacts --db state/todesanzeigen.sqlite3
todesanzeigen extract --source "Augsburger Allgemeine"
todesanzeigen extract --provider qwen
todesanzeigen extract --provider qwen --concurrency 10
todesanzeigen extract --name-confidence-threshold 90 --log-dir logs
todesanzeigen extract --reroute --input-dir input --reroute-provider qwen
todesanzeigen extract --force
```

The extraction step makes remote API calls to the configured LLM provider. Local
OCR does not remove LLM usage. Rerun OCR with `--overwrite` if you have old
text-only artifacts without matching TSV files, then run `todesanzeigen filter`
before extraction.

SQLite is the authoritative cache for extraction outputs. Before a model
request is scheduled, the command checks whether the same document already has
an active successful row for the requested result slot. `text_extraction` uses
the `ocr_llm` slot. Direct VLM extraction and low-confidence VLM reroute both
use the `vlm` slot, so those two VLM paths do not coexist for the same
document. A cache hit returns `cached_existing`, avoids the provider call, and
does not insert a duplicate extraction row. Pass `--force` to supersede the
active row for that slot and run the provider again.

Extraction is still checkpointed by default for run-level audit and local
resume. Every `extract` run writes completed, skipped, cached, rerouted, and
failed file records to `logs/results.jsonl` unless `--resume-from` points at
another JSONL checkpoint. Existing `processed` and `rerouted_processed`
checkpoint records are reused only after the DB cache check. Existing
`skipped_low_confidence` checkpoint records are reused when reroute is off, but
retried through the vision provider when `--reroute` is enabled.

With `--reroute`, low-confidence cases are sent to the configured vision model
using the original source image. Successful reroutes are stored as
`method=vision_model_reroute`, `method_family=vlm`, and
`route_reason=low_confidence`. Reroute audit records are also written to
`logs/reroute-results.jsonl` unless overridden:

```sh
todesanzeigen extract \
  --reroute \
  --input-dir "input/Aichacher Nachrichten" \
  --artifacts-dir "artifacts/Aichacher Nachrichten" \
  --reroute-results-file logs/aichacher_nachrichten/reroute-results.jsonl
```

The JSONL records include `method`, `provider`, `model`, `source_image`,
`ocr_artifact`, `tsv_artifact`, `original_name_hint`,
`original_name_confidence`, and `threshold` so rerouted rows can be audited
alongside the database records.

Extraction is sequential by default. Passing `--concurrency` greater than `1`
enables concurrent extraction with retry and rate limiting. Qwen runs default to
conservative limits of 600 requests/minute and 500000 estimated tokens/minute
unless overridden:

```sh
TODESANZEIGEN_LLM_CONCURRENCY=10
TODESANZEIGEN_LLM_RPM_LIMIT=600
TODESANZEIGEN_LLM_TPM_LIMIT=500000
TODESANZEIGEN_LLM_MAX_RETRIES=5
```

Resume or continue a run with a specific checkpoint file:

```sh
todesanzeigen extract --provider qwen --concurrency 10 --resume-from logs/results.jsonl
```

## Vision Reroute Usage

If high-confidence rows were already extracted without `--reroute`, process only
the rejected cases later with the standalone reroute command:

```sh
todesanzeigen reroute \
  --input-dir "input/Aichacher Nachrichten" \
  --artifacts-dir "artifacts/Aichacher Nachrichten" \
  --low-confidence-file logs/aichacher_nachrichten/filter-low-confidence.jsonl \
  --source "Aichacher Nachrichten"
```

You can also source candidates from an existing extraction checkpoint:

```sh
todesanzeigen reroute \
  --input-dir "input/Aichacher Nachrichten" \
  --artifacts-dir "artifacts/Aichacher Nachrichten" \
  --from-results logs/aichacher_nachrichten/results.jsonl
```

Useful selection and merge options:

```sh
todesanzeigen reroute --only "Aichacher Nachrichten 2023_063.txt"
todesanzeigen reroute --sample-ratio 0.1 --sample-seed 42
todesanzeigen reroute --limit 20
todesanzeigen reroute --force
```

Standalone reroute records VLM outputs in SQLite and writes audit details in
`logs/reroute-results.jsonl`. It uses the same `vlm` result slot as
`vision-extract`, so an existing direct VLM result prevents a reroute call for
the same document unless `--force` is passed.

## Image-Only Vision Extraction

Use `vision-extract` to process source images directly through a VLM without
running OCR, TSV filtering, or low-confidence reroute selection:

```sh
todesanzeigen vision-extract \
  --input-dir "input/Aichacher Nachrichten" \
  --source "Aichacher Nachrichten" \
  --provider qwen \
  --model qwen-vl-ocr \
  --limit 25 \
  --sample-seed 42
```

This records candidate rows in SQLite with `method=vision_model_image_only`,
`method_family=vlm`, and `route_reason=image_only`. Results are checkpointed by
default in `logs/vision-results.jsonl`. The default candidate kind is `teacher`
because this command is intended to bootstrap review candidates for ground truth.

Useful dataset selection options:

```sh
todesanzeigen vision-extract --only "Aichacher Nachrichten 2023_063.jpg"
todesanzeigen vision-extract --sample-ratio 0.05 --sample-seed 42
todesanzeigen vision-extract --limit 20
todesanzeigen vision-extract --concurrency 3 --rpm-limit 100 --tpm-limit 200000
todesanzeigen vision-extract --force
```

## ML Infrastructure Usage

OCR artifacts remain files, but extraction outputs are now DB-first. SQLite is
the durable project state for document inventory, OCR lineage, method outputs,
teacher/pipeline candidates, reviewed ground truth, feature snapshots, dataset
splits, evaluation results, and final CSV export.

The default database path is:

```text
state/todesanzeigen.sqlite3
```

Local SQLite files are ignored by Git. Images and generated OCR artifacts also
stay as files. Source-image paths and hashes live on `documents`; OCR text/TSV
paths and hashes live on `ocr_outputs`.

Initialize or migrate the database:

```sh
todesanzeigen db init
```

Import a source folder and its OCR artifacts:

```sh
todesanzeigen ingest source \
  --source "Aichacher Nachrichten" \
  --input-dir "input/Aichacher Nachrichten" \
  --artifacts-dir "artifacts/Aichacher Nachrichten" \
  --layout-family clean
```

This records source images on `documents`, OCR text and TSV metadata on
`ocr_outputs`, name-map hints, basic OCR features, and a run record. It does not
move or rewrite the original files.

Import existing extraction outputs:

```sh
todesanzeigen ingest results \
  --output-csv output/result.csv \
  --method text_extraction \
  --provider qwen \
  --model qwen3.6-flash \
  --candidate-kind teacher
```

You can also import JSONL checkpoints:

```sh
todesanzeigen ingest results \
  --results-file logs/aichacher_nachrichten/results.jsonl \
  --method text_extraction \
  --provider qwen \
  --model qwen3.6-flash \
  --candidate-kind teacher
```

Imported rows are stored as extraction outputs and as pending label candidates.
Model-backed methods require an exact model name. Normal `extract`, `reroute`,
and `vision-extract` runs record the provider and model automatically; imports
must pass `--model` because legacy CSV and JSONL files may not contain it.
They are not treated as ground truth automatically. This is intentional:
VLM-generated or LLM-generated "ground truth" should be reviewed before it is
used as a benchmark label.

For `--candidate-kind`, the following are supported:
- `pipeline`: a candidate from the main extraction pipeline, usually OCR+LLM.
- `teacher`: a candidate from a teacher model, usually a VLM. Stronger results that we may review as a possible ground truth label. Default.
- `manual_seed`: a candidate from a manual seed, usually a known good record. 

Make sure to use the flags explicitly to avoid accidental mislabeling of candidates.

### Method Records And CSV Export

Extraction methods are stored separately for the same document:

```text
text_extraction           method_family=ocr_llm, result_slot=ocr_llm
vision_model_image_only   method_family=vlm, result_slot=vlm, route_reason=image_only
vision_model_reroute      method_family=vlm, result_slot=vlm, route_reason=low_confidence
```

Reviewed ground truth is stored only in `ground_truth_labels`. It is not mirrored
as another extraction method. Extraction commands do not write the operational
CSV directly; generate it from the DB when needed:

```sh
todesanzeigen export csv \
  --db state/todesanzeigen.sqlite3 \
  --label-set gt-v1 \
  --output-file output/result.csv
```

CSV priority is: reviewed GT, image-only VLM, low-confidence VLM reroute,
OCR+LLM text extraction, then no row. Superseded extraction rows are ignored by
this export. Internally, `ort` is the single stored location field. CSV export
copies it to both `ort` and the legacy `wohnort` column.

Normalize legacy location and date fields with a dry run first, then apply the
transactional update with an automatic timestamped backup:

```sh
todesanzeigen db normalize-fields --db state/todesanzeigen.sqlite3
todesanzeigen db normalize-fields --db state/todesanzeigen.sqlite3 --apply
```

### Label Review

Start the local review UI:

```sh
todesanzeigen review serve --reviewer "your-name"
```

Then open:

```text
http://127.0.0.1:8000
```

The review UI has separate **Needs review** and **Ground truth** views. On a
record page it shows the source image, OCR text, current GT when available, and
a field-by-field comparison of the latest result from each extraction method.
The GT editor starts empty for an unreviewed record: use **Fill form** on the
text or VLM result to choose an explicit starting source, or enter a label
manually. **Approve as GT** accepts a method result directly.

There is one GT row per document and label set. Saving or approving a different
source replaces that row rather than creating another one; the UI asks for
confirmation when a direct approval would replace existing GT. **Needs review
— next** leaves the record pending and moves on without changing candidate
statuses. Saved labels use the default label set `gt-v1`.

Use a custom label set when needed:

```sh
todesanzeigen review serve --label-set gt-v2 --port 8001
```

The review app is local-only and has no authentication. Do not bind it to a
public interface unless you add access control.

### Dataset Splits

Create a deterministic source/year split:

```sh
todesanzeigen dataset split --name benchmark-v1 --strategy source-year
```

The split keeps documents from the same source/year group together, which helps
avoid overly optimistic results from near-duplicate newspaper templates.

Export a split as JSONL:

```sh
todesanzeigen dataset export \
  --split benchmark-v1 \
  --label-set gt-v1 \
  --output-file output/datasets/benchmark-v1.jsonl
```

Build router feature snapshots:

```sh
todesanzeigen features build --feature-set router-v2
```

Export supervised failure-prediction rows for the first router milestone:

```sh
todesanzeigen dataset export-router \
  --label-set gt-v1 \
  --method text_extraction \
  --feature-set router-v2 \
  --output-file output/datasets/router-v2-text-extraction.jsonl
```

The router export uses reviewed GT to mark whether the selected method produced
an exact-record match. Features include source metadata, image quality proxies,
OCR text statistics, TSV confidence/layout statistics, and the local name-hint
confidence. Filename, year, layout-family, image-path, suffix, and MIME metadata
are intentionally excluded from model inputs.

### Learned Router Training

The learned router predicts whether the cheap OCR+LLM path should be trusted or
whether a document should be escalated to the VLM path. The first-pass decision
problem is binary:

```text
ocr_llm  = OCR text -> text LLM extraction
vlm      = raw image -> vision-language extraction
```

The router trains from SQLite, not from the exported CSV. Each labeled training
record is expected to point to one document/image and have:

- a feature snapshot, usually `router-v2`
- a reviewed GT label in `ground_truth_labels`
- an active OCR+LLM output in the `ocr_llm` result slot
- optionally an active VLM output in the `vlm` result slot for routed quality/cost metrics

Build feature snapshots before training:

```sh
todesanzeigen features build \
  --db state/todesanzeigen.sqlite3 \
  --feature-set router-v2
```

Train the router:

```sh
todesanzeigen router train \
  --db state/todesanzeigen.sqlite3 \
  --label-set gt-v1 \
  --feature-set router-v2 \
  --split benchmark-v1 \
  --model-dir models/router/router-v2
```

Training writes:

```text
models/router/router-v2/model.joblib
models/router/router-v2/training-report.json
models/router/router-v2/feature-schema.json
models/router/router-v2/thresholds.json
```

The default target is `cheap_pipeline_failed = target_field_f1 < 0.95`.
Only these structured target fields are evaluated:

```text
geschlecht, name, vorname, geburtsdatum, sterbedatum, geburtsname, titel,
genannt, geburtsort, sterbeort, ort, weitere_orte, beruf
```

Blank GT fields are treated as unavailable labels, not required empty outputs.
Non-target fields such as `bemerkungen`, `quelle`, `dateiname`,
`zusaetzliche_hinweise`, and `confidence_score` are excluded from router target
scoring.

Write a routing manifest:

```sh
todesanzeigen router manifest \
  --db state/todesanzeigen.sqlite3 \
  --label-set gt-v1 \
  --feature-set router-v2 \
  --model-dir models/router/router-v2 \
  --output-file output/router-manifest.jsonl
```

Each manifest row includes the document id, source, filename stem, image path,
predicted failure probability, selected route, threshold, expected cost, and the
available OCR+LLM/VLM output ids. This manifest is the artifact later routing
code can use to decide whether a request should run through `ocr_llm` or `vlm`.

Useful training knobs:

```sh
todesanzeigen router train \
  --feature-set router-v2 \
  --model-dir models/router/router-v2 \
  --target-f1-threshold 0.95 \
  --cheap-cost 1.0 \
  --vlm-cost 10.0 \
  --lambda-cost 0.01 \
  --min-train-rows 20
```

If there are too few labeled rows, no feature snapshots, or only one target
class, training fails with a clear message instead of writing a misleading
model.

### Evaluation

Evaluate the latest outputs for a method against reviewed labels:

```sh
todesanzeigen eval run \
  --label-set gt-v1 \
  --method text_extraction
```

Evaluate on a named split:

```sh
todesanzeigen eval run \
  --label-set gt-v1 \
  --method vision_model_image_only \
  --split benchmark-v1
```

Evaluation records exact-record accuracy, field-level precision, recall, F1, and
missing predictions. Aggregate metrics are stored in `evaluation_runs`; per
document field comparisons are stored in `evaluation_results`.

### Recommended ML Workflow

For a first benchmark pass:

```sh
todesanzeigen db init
todesanzeigen ocr --input-dir "input/Aichacher Nachrichten" --artifacts-dir "artifacts/Aichacher Nachrichten"
todesanzeigen filter --artifacts-dir "artifacts/Aichacher Nachrichten"
todesanzeigen extract --input-dir "input/Aichacher Nachrichten" --artifacts-dir "artifacts/Aichacher Nachrichten" --source "Aichacher Nachrichten"
todesanzeigen vision-extract --input-dir "input/Aichacher Nachrichten" --source "Aichacher Nachrichten" --limit 100
todesanzeigen review serve --reviewer "your-name"
todesanzeigen dataset split --name benchmark-v1 --strategy source-year
todesanzeigen features build --feature-set router-v2
todesanzeigen eval run --label-set gt-v1 --method text_extraction --split benchmark-v1
todesanzeigen dataset export-router --label-set gt-v1 --method text_extraction --feature-set router-v2 --output-file output/datasets/router-v2.jsonl
todesanzeigen router train --label-set gt-v1 --feature-set router-v2 --split benchmark-v1 --model-dir models/router/router-v2
todesanzeigen router manifest --label-set gt-v1 --feature-set router-v2 --model-dir models/router/router-v2 --output-file output/router-manifest.jsonl
todesanzeigen export csv --label-set gt-v1 --output-file output/result.csv
```

## Google Document AI Fallback

Document AI is disabled by default to avoid accidental GCP billing. Passing `--engine documentai` is not enough; the environment must also explicitly allow GCP:

```sh
TODESANZEIGEN_ALLOW_GCP=1 todesanzeigen ocr --engine documentai
```

Required Document AI environment variables:

```sh
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GOOGLE_CLOUD_PROJECT_ID=your-google-cloud-project-id
DOCUMENT_AI_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-document-ai-processor-id
```

Without `TODESANZEIGEN_ALLOW_GCP=1`, the program fails before creating a Document AI client.

## Testing

Run tests from the repository root:

```sh
PYTHONPATH=. uv run pytest -q
```

The project declares `pytest` in the `dev` dependency group, so `uv run` uses
the same dependency-managed environment as the package.
