ALTER TABLE extraction_methods
ADD COLUMN result_slot TEXT NOT NULL DEFAULT '';

UPDATE extraction_methods
SET result_slot = CASE
    WHEN method = 'text_extraction' THEN 'ocr_llm'
    WHEN method IN ('vision_model_image_only', 'vision_model_reroute') THEN 'vlm'
    ELSE method
END
WHERE result_slot = '';

INSERT INTO extraction_methods(method, method_family, result_slot, default_route_reason, description)
VALUES
    ('text_extraction', 'ocr_llm', 'ocr_llm', '', 'OCR text plus local name hint passed to a text LLM.'),
    ('vision_model_image_only', 'vlm', 'vlm', 'image_only', 'Direct image-only vision-language extraction.'),
    ('vision_model_reroute', 'vlm', 'vlm', 'low_confidence', 'Vision-language extraction used after low OCR/name confidence.')
ON CONFLICT(method) DO UPDATE SET
    method_family = excluded.method_family,
    result_slot = excluded.result_slot,
    default_route_reason = excluded.default_route_reason,
    description = excluded.description;

ALTER TABLE extraction_outputs
ADD COLUMN result_slot TEXT NOT NULL DEFAULT '';

ALTER TABLE extraction_outputs
ADD COLUMN input_fingerprint TEXT NOT NULL DEFAULT '';

ALTER TABLE extraction_outputs
ADD COLUMN config_hash TEXT NOT NULL DEFAULT '';

ALTER TABLE extraction_outputs
ADD COLUMN superseded_at TEXT;

UPDATE extraction_outputs
SET result_slot = COALESCE(
        NULLIF(result_slot, ''),
        (SELECT result_slot FROM extraction_methods WHERE extraction_methods.method = extraction_outputs.method),
        method
    )
WHERE result_slot = '';

UPDATE extraction_outputs
SET superseded_at = CURRENT_TIMESTAMP
WHERE result_slot != ''
  AND superseded_at IS NULL
  AND status IN ('processed', 'rerouted_processed', 'vision_processed')
  AND id NOT IN (
      SELECT MAX(id)
      FROM extraction_outputs
      WHERE result_slot != ''
        AND superseded_at IS NULL
        AND status IN ('processed', 'rerouted_processed', 'vision_processed')
      GROUP BY document_id, result_slot
  );

CREATE INDEX IF NOT EXISTS idx_extraction_methods_result_slot
ON extraction_methods(result_slot);

CREATE INDEX IF NOT EXISTS idx_extraction_outputs_result_slot
ON extraction_outputs(document_id, result_slot, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_active_successful_output_slot
ON extraction_outputs(document_id, result_slot)
WHERE result_slot != ''
  AND superseded_at IS NULL
  AND status IN ('processed', 'rerouted_processed', 'vision_processed');
