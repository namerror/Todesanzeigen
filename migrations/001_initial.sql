CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    document_key TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    filename_stem TEXT NOT NULL,
    image_path TEXT NOT NULL DEFAULT '',
    image_sha256 TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL DEFAULT '',
    year INTEGER,
    layout_family TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_source_stem
ON documents(source_id, filename_stem);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    code_version TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    run_id TEXT REFERENCES runs(id),
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    producer TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, artifact_type, path)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_document_type
ON artifacts(document_id, artifact_type);

CREATE TABLE IF NOT EXISTS ocr_outputs (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    run_id TEXT REFERENCES runs(id),
    text TEXT NOT NULL DEFAULT '',
    text_artifact_id INTEGER REFERENCES artifacts(id),
    tsv_artifact_id INTEGER REFERENCES artifacts(id),
    settings_json TEXT NOT NULL DEFAULT '{}',
    features_json TEXT NOT NULL DEFAULT '{}',
    name_hint TEXT NOT NULL DEFAULT '',
    name_confidence REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, run_id)
);

CREATE TABLE IF NOT EXISTS extraction_outputs (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    run_id TEXT REFERENCES runs(id),
    method TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT 'death_notice_v1',
    fields_json TEXT NOT NULL DEFAULT '{}',
    raw_response TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    estimated_tokens INTEGER,
    cost_usd REAL,
    source_artifact_id INTEGER REFERENCES artifacts(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_extraction_outputs_document_method
ON extraction_outputs(document_id, method, created_at);

CREATE TABLE IF NOT EXISTS label_candidates (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    extraction_output_id INTEGER REFERENCES extraction_outputs(id),
    source_kind TEXT NOT NULL,
    source_name TEXT NOT NULL DEFAULT '',
    fields_json TEXT NOT NULL DEFAULT '{}',
    confidence_score REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_label_candidates_status
ON label_candidates(status, created_at);

CREATE TABLE IF NOT EXISTS ground_truth_labels (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    label_set TEXT NOT NULL,
    fields_json TEXT NOT NULL DEFAULT '{}',
    source_candidate_id INTEGER REFERENCES label_candidates(id),
    reviewer TEXT NOT NULL DEFAULT '',
    review_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, label_set)
);

CREATE INDEX IF NOT EXISTS idx_ground_truth_label_set
ON ground_truth_labels(label_set);

CREATE TABLE IF NOT EXISTS dataset_splits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    strategy TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_memberships (
    split_id INTEGER NOT NULL REFERENCES dataset_splits(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    subset TEXT NOT NULL,
    PRIMARY KEY(split_id, document_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    label_set TEXT NOT NULL,
    method TEXT NOT NULL,
    split_name TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY,
    evaluation_run_id INTEGER NOT NULL REFERENCES evaluation_runs(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    extraction_output_id INTEGER REFERENCES extraction_outputs(id),
    exact_match INTEGER NOT NULL DEFAULT 0,
    field_results_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(evaluation_run_id, document_id)
);
