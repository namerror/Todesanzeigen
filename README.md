# Todesanzeigen

OCR and structured extraction pipeline for German death notice images.

The current workflow has two separate steps:

1. Run OCR on images from `input/` and write plain text artifacts to `artifacts/`.
2. Parse those OCR text artifacts with Gemini and write structured rows to `output/result.csv`.

Local Tesseract OCR is the default. Google Document AI OCR is kept as a guarded legacy fallback and is blocked unless explicitly unlocked.

## Setup

Install the Python package in editable mode:

```sh
python -m pip install -e .
```

Install Tesseract and German language data. On Ubuntu/Debian:

```sh
sudo apt install tesseract-ocr tesseract-ocr-deu
```

Confirm that German OCR data is available:

```sh
tesseract --list-langs
```

The default OCR language is `deu+eng`, so the language list must include `deu` and `eng`.

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

Common options:

```sh
todesanzeigen ocr --limit 10
todesanzeigen ocr --overwrite
todesanzeigen ocr --input-dir input --artifacts-dir artifacts
todesanzeigen ocr --tesseract-lang deu+eng --tesseract-psm 6
```

Useful Tesseract page segmentation modes:

- `6`: default; assumes one uniform text block, good for cropped clean notices.
- `4`: assumes a single column of variable-size text, useful for some newspaper crops.
- `11`: sparse text mode, useful for messy or loosely cropped images.

Supported image extensions are:

```text
.bmp .gif .jpeg .jpg .png .tif .tiff .webp
```

## Structured Extraction Usage

After OCR artifacts exist, extract structured CSV rows:

```sh
todesanzeigen extract
```

This reads `artifacts/*.txt` and writes `output/result.csv`.

Common options:

```sh
todesanzeigen extract --limit 10
todesanzeigen extract --artifacts-dir artifacts --output-file output/result.csv
todesanzeigen extract --source "Augsburger Allgemeine"
```

The extraction step currently uses Gemini and therefore makes remote API calls. Local OCR does not remove Gemini usage.

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
