CREATE TABLE IF NOT EXISTS extraction_methods (
    method TEXT PRIMARY KEY,
    method_family TEXT NOT NULL,
    default_route_reason TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO extraction_methods(method, method_family, default_route_reason, description)
VALUES
    ('text_extraction', 'ocr_llm', '', 'OCR text plus local name hint passed to a text LLM.'),
    ('vision_model_image_only', 'vlm', 'image_only', 'Direct image-only vision-language extraction.'),
    ('vision_model_reroute', 'vlm', 'low_confidence', 'Vision-language extraction used after low OCR/name confidence.')
ON CONFLICT(method) DO UPDATE SET
    method_family = excluded.method_family,
    default_route_reason = excluded.default_route_reason,
    description = excluded.description;

ALTER TABLE extraction_outputs
ADD COLUMN method_family TEXT NOT NULL DEFAULT '';

ALTER TABLE extraction_outputs
ADD COLUMN route_reason TEXT NOT NULL DEFAULT '';

ALTER TABLE extraction_outputs
ADD COLUMN route_decision_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE extraction_outputs
ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE extraction_outputs
ADD COLUMN ocr_output_id INTEGER REFERENCES ocr_outputs(id);

UPDATE extraction_outputs
SET method_family = COALESCE(
        NULLIF(method_family, ''),
        (SELECT method_family FROM extraction_methods WHERE extraction_methods.method = extraction_outputs.method),
        ''
    ),
    route_reason = COALESCE(
        NULLIF(route_reason, ''),
        (SELECT default_route_reason FROM extraction_methods WHERE extraction_methods.method = extraction_outputs.method),
        ''
    )
WHERE method_family = '' OR route_reason = '';

CREATE INDEX IF NOT EXISTS idx_extraction_outputs_method_family
ON extraction_outputs(method_family, created_at);

CREATE INDEX IF NOT EXISTS idx_extraction_outputs_ocr_output
ON extraction_outputs(ocr_output_id);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    ocr_output_id INTEGER REFERENCES ocr_outputs(id),
    feature_set TEXT NOT NULL,
    features_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, feature_set)
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_feature_set
ON feature_snapshots(feature_set, created_at);
