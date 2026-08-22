# Todesanzeigen

OCR and structured extraction pipeline for German death notice images.

The current workflow has separate steps:

1. Run OCR on images from `input/` and write plain text artifacts to `artifacts/`.
2. Optionally run the local TSV filter to print likely deceased names.
3. Parse those OCR text artifacts with an LLM and write structured rows to `output/result.csv`.

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
`artifacts/name_map.json`, extract structured CSV rows:

```sh
todesanzeigen extract
```

This reads `artifacts/*.txt`, requires `artifacts/name_map.json`, and writes
`output/result.csv`. The LLM receives the OCR text plus the locally detected name
and confidence as a hint; it does not receive the TSV layout artifact. Artifacts
with missing name confidence or name confidence below 85.0 are skipped unless
vision reroute is enabled. Each extraction run writes a timestamped text log
under `logs/`.

Common options:

```sh
todesanzeigen extract --limit 10
todesanzeigen extract --artifacts-dir artifacts --output-file output/result.csv
todesanzeigen extract --source "Augsburger Allgemeine"
todesanzeigen extract --provider qwen
todesanzeigen extract --provider qwen --concurrency 10
todesanzeigen extract --name-confidence-threshold 90 --log-dir logs
todesanzeigen extract --reroute --input-dir input --reroute-provider qwen
```

The extraction step makes remote API calls to the configured LLM provider. Local
OCR does not remove LLM usage. Rerun OCR with `--overwrite` if you have old
text-only artifacts without matching TSV files, then run `todesanzeigen filter`
before extraction.

Extraction is checkpointed by default. Every `extract` run writes completed,
skipped, rerouted, and failed file records to `logs/results.jsonl` unless
`--resume-from` points at another JSONL checkpoint. Existing `processed` and
`rerouted_processed` records are reused on the next run. Existing
`skipped_low_confidence` records are reused when reroute is off, but retried
through the vision provider when `--reroute` is enabled.

With `--reroute`, low-confidence cases are sent to the configured vision model
using the original source image. Successful reroutes are merged into the normal
CSV with status `rerouted_processed` in JSONL logs. Reroute audit records are
also written to `logs/reroute-results.jsonl` unless overridden:

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
without changing the CSV schema.

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
  --output-file output/aichacher-rerouted.csv
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
todesanzeigen reroute --merge-output-file output/result.csv
```

Standalone reroute writes `output/rerouted.csv` by default and records audit
details in `logs/reroute-results.jsonl`. When `--merge-output-file` is passed,
rerouted rows are merged into that CSV by `dateiname`, replacing any existing
row for the same notice.

## Image-Only Vision Extraction

Use `vision-extract` to process source images directly through a VLM without
running OCR, TSV filtering, or low-confidence reroute selection:

```sh
todesanzeigen vision-extract \
  --input-dir "input/Aichacher Nachrichten" \
  --output-file output/ground-truth/aichacher-vlm.csv \
  --provider qwen \
  --model qwen-vl-ocr \
  --limit 25 \
  --sample-seed 42
```

This writes the same CSV schema as normal extraction, with `dateiname` set from
the image stem. Results are checkpointed by default in
`logs/vision-results.jsonl`, and audit records use
`method=vision_model_image_only`.

Useful dataset selection options:

```sh
todesanzeigen vision-extract --only "Aichacher Nachrichten 2023_063.jpg"
todesanzeigen vision-extract --sample-ratio 0.05 --sample-seed 42
todesanzeigen vision-extract --limit 20
todesanzeigen vision-extract --concurrency 3 --rpm-limit 100 --tpm-limit 200000
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
PYTHONPATH=. pytest -q
```

If `pytest` is not installed in the active environment, the current test suite
also runs with:

```sh
python -m unittest discover -s tests
```
