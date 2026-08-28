CREATE TRIGGER IF NOT EXISTS trg_extraction_outputs_require_model_insert
BEFORE INSERT ON extraction_outputs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND (
        NEW.method_family IN ('ocr_llm', 'vlm')
        OR COALESCE(
            (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
            ''
        ) IN ('ocr_llm', 'vlm')
    )
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction output');
END;

CREATE TRIGGER IF NOT EXISTS trg_extraction_outputs_require_model_update
BEFORE UPDATE OF method, method_family, model ON extraction_outputs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND (
        NEW.method_family IN ('ocr_llm', 'vlm')
        OR COALESCE(
            (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
            ''
        ) IN ('ocr_llm', 'vlm')
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
    ) IN ('ocr_llm', 'vlm')
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction run');
END;

CREATE TRIGGER IF NOT EXISTS trg_runs_require_model_update
BEFORE UPDATE OF method, model ON runs
WHEN trim(COALESCE(NEW.model, '')) = ''
 AND COALESCE(
        (SELECT method_family FROM extraction_methods WHERE method = NEW.method),
        ''
    ) IN ('ocr_llm', 'vlm')
BEGIN
    SELECT RAISE(ABORT, 'model is required for model-backed extraction run');
END;
