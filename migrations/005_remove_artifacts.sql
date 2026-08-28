ALTER TABLE ocr_outputs ADD COLUMN text_path TEXT NOT NULL DEFAULT '';
ALTER TABLE ocr_outputs ADD COLUMN text_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE ocr_outputs ADD COLUMN tsv_path TEXT NOT NULL DEFAULT '';
ALTER TABLE ocr_outputs ADD COLUMN tsv_sha256 TEXT NOT NULL DEFAULT '';

UPDATE ocr_outputs
SET text_path = COALESCE(
        (SELECT path FROM artifacts WHERE artifacts.id = ocr_outputs.text_artifact_id),
        ''
    ),
    text_sha256 = COALESCE(
        (SELECT sha256 FROM artifacts WHERE artifacts.id = ocr_outputs.text_artifact_id),
        ''
    ),
    tsv_path = COALESCE(
        (SELECT path FROM artifacts WHERE artifacts.id = ocr_outputs.tsv_artifact_id),
        ''
    ),
    tsv_sha256 = COALESCE(
        (SELECT sha256 FROM artifacts WHERE artifacts.id = ocr_outputs.tsv_artifact_id),
        ''
    );

UPDATE runs
SET config_json = json_set(
        CASE WHEN json_valid(config_json) THEN config_json ELSE '{}' END,
        '$.output_csv',
        COALESCE(
            NULLIF(
                json_extract(
                    CASE WHEN json_valid(config_json) THEN config_json ELSE '{}' END,
                    '$.output_csv'
                ),
                ''
            ),
            (SELECT path FROM artifacts
             WHERE artifacts.run_id = runs.id AND artifact_type = 'csv_output'
             ORDER BY id DESC LIMIT 1),
            ''
        ),
        '$.output_csv_sha256',
        COALESCE(
            (SELECT sha256 FROM artifacts
             WHERE artifacts.run_id = runs.id AND artifact_type = 'csv_output'
             ORDER BY id DESC LIMIT 1),
            ''
        )
    )
WHERE EXISTS (
    SELECT 1 FROM artifacts
    WHERE artifacts.run_id = runs.id AND artifact_type = 'csv_output'
);

UPDATE runs
SET config_json = json_set(
        CASE WHEN json_valid(config_json) THEN config_json ELSE '{}' END,
        '$.results_file',
        COALESCE(
            NULLIF(
                json_extract(
                    CASE WHEN json_valid(config_json) THEN config_json ELSE '{}' END,
                    '$.results_file'
                ),
                ''
            ),
            (SELECT path FROM artifacts
             WHERE artifacts.run_id = runs.id AND artifact_type = 'jsonl_checkpoint'
             ORDER BY id DESC LIMIT 1),
            ''
        ),
        '$.results_file_sha256',
        COALESCE(
            (SELECT sha256 FROM artifacts
             WHERE artifacts.run_id = runs.id AND artifact_type = 'jsonl_checkpoint'
             ORDER BY id DESC LIMIT 1),
            ''
        )
    )
WHERE EXISTS (
    SELECT 1 FROM artifacts
    WHERE artifacts.run_id = runs.id AND artifact_type = 'jsonl_checkpoint'
);

ALTER TABLE ocr_outputs DROP COLUMN text_artifact_id;
ALTER TABLE ocr_outputs DROP COLUMN tsv_artifact_id;
ALTER TABLE extraction_outputs DROP COLUMN source_artifact_id;

DROP TABLE artifacts;
