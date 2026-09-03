DROP INDEX IF EXISTS idx_active_successful_output_slot;

UPDATE extraction_outputs
SET superseded_at = CURRENT_TIMESTAMP
WHERE status IN ('processed', 'rerouted_processed', 'vision_processed');

UPDATE extraction_outputs
SET superseded_at = NULL
WHERE id IN (
    SELECT MAX(id)
    FROM extraction_outputs
    WHERE status IN ('processed', 'rerouted_processed', 'vision_processed')
    GROUP BY document_id, method, provider, model, prompt_version
);

CREATE INDEX IF NOT EXISTS idx_extraction_outputs_variant
ON extraction_outputs(document_id, method, provider, model, prompt_version, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_active_successful_output_variant
ON extraction_outputs(document_id, method, provider, model, prompt_version)
WHERE superseded_at IS NULL
  AND status IN ('processed', 'rerouted_processed', 'vision_processed');
