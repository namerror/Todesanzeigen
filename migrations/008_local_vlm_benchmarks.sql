ALTER TABLE extraction_outputs ADD COLUMN inference_config_hash TEXT NOT NULL DEFAULT '';
DROP INDEX idx_active_successful_output_variant;
CREATE UNIQUE INDEX idx_active_successful_output_variant
ON extraction_outputs(document_id, method, provider, model, prompt_version, inference_config_hash)
WHERE superseded_at IS NULL AND status IN ('processed', 'rerouted_processed', 'vision_processed');
CREATE INDEX idx_extraction_inference_variant
ON extraction_outputs(document_id, method, provider, model, prompt_version, inference_config_hash);
INSERT INTO extraction_methods(method, method_family, default_route_reason, description)
VALUES ('local_vision_model_image_only', 'local_vlm', 'image_only', 'Local image-only vision-language extraction.');

CREATE TABLE inference_benchmark_runs (
    id TEXT PRIMARY KEY REFERENCES runs(id),
    variant_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    environment_json TEXT NOT NULL DEFAULT '{}',
    load_ms REAL,
    warmup_ms REAL,
    warmup_count INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);
CREATE TABLE inference_benchmark_samples (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES inference_benchmark_runs(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    repetition INTEGER NOT NULL,
    phase TEXT NOT NULL CHECK(phase IN ('warmup', 'measured')),
    status TEXT NOT NULL,
    raw_response TEXT NOT NULL DEFAULT '',
    fields_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    extraction_output_id INTEGER REFERENCES extraction_outputs(id),
    UNIQUE(run_id, document_id, repetition, phase)
);
CREATE INDEX idx_benchmark_samples_run ON inference_benchmark_samples(run_id, phase);

DROP TRIGGER trg_extraction_outputs_require_model_insert;
DROP TRIGGER trg_extraction_outputs_require_model_update;
DROP TRIGGER trg_runs_require_model_insert;
DROP TRIGGER trg_runs_require_model_update;
CREATE TRIGGER IF NOT EXISTS trg_extraction_outputs_require_model_insert
BEFORE INSERT ON extraction_outputs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND (
        NEW.method_family IN ('ocr_llm', 'vlm', 'local_vlm')
        OR COALESCE(
            (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
            ''
        ) IN ('ocr_llm', 'vlm', 'local_vlm')
    )
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction output');
END;

CREATE TRIGGER IF NOT EXISTS trg_extraction_outputs_require_model_update
BEFORE UPDATE OF method, method_family, model ON extraction_outputs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND (
        NEW.method_family IN ('ocr_llm', 'vlm', 'local_vlm')
        OR COALESCE(
            (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
            ''
        ) IN ('ocr_llm', 'vlm', 'local_vlm')
    )
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction output');
END;

CREATE TRIGGER IF NOT EXISTS trg_runs_require_model_insert
BEFORE INSERT ON runs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND COALESCE(
        (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
        ''
    ) IN ('ocr_llm', 'vlm', 'local_vlm')
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction run');
END;

CREATE TRIGGER IF NOT EXISTS trg_runs_require_model_update
BEFORE UPDATE OF method, model ON runs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND COALESCE(
        (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
        ''
    ) IN ('ocr_llm', 'vlm', 'local_vlm')
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction run');
END;
