# Todesanzeigen

OCR and structured extraction pipeline for German death notice images.

The current workflow has separate steps:

1. Run OCR on images from `input/` and write plain text artifacts to `artifacts/`.
2. Optionally run the local TSV filter to print likely deceased names.
3. Parse those OCR text artifacts with Gemini and write structured rows to `output/result.csv`.

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

For structured extraction, create a `.env` file from `.env.example` and set:

```sh
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-lite
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
It does not call an LLM.

Common options:

```sh
todesanzeigen filter --limit 10
todesanzeigen filter --artifacts-dir artifacts
```

## Structured Extraction Usage

After OCR artifacts exist and `todesanzeigen filter` has written
`artifacts/name_map.json`, extract structured CSV rows:

```sh
todesanzeigen extract
```

This reads `artifacts/*.txt`, requires `artifacts/name_map.json`, and writes
`output/result.csv`. The LLM receives the OCR text plus the locally detected name
and confidence as a hint; it does not receive the TSV layout artifact. Artifacts
with missing name confidence or name confidence below 85.0 are not sent to the
LLM. Each extraction run writes a timestamped text log under `logs/`.

Common options:

```sh
todesanzeigen extract --limit 10
todesanzeigen extract --artifacts-dir artifacts --output-file output/result.csv
todesanzeigen extract --source "Augsburger Allgemeine"
todesanzeigen extract --name-confidence-threshold 90 --log-dir logs
```

The extraction step currently uses Gemini and therefore makes remote API calls.
Local OCR does not remove Gemini usage. Rerun OCR with `--overwrite` if you have
old text-only artifacts without matching TSV files, then run `todesanzeigen filter`
before extraction.

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
