-- -- 1. Ensure the column exists (safe no-op if it already does)
-- ALTER TABLE document_vector_store
--   ADD COLUMN IF NOT EXISTS bm25_tsvector tsvector;

-- -- 2. Trigger function: auto-populate on every INSERT or UPDATE
-- CREATE OR REPLACE FUNCTION update_document_bm25_tsvector()
-- RETURNS TRIGGER AS $$
-- BEGIN
--   NEW.bm25_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
--   RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- -- 3. Attach trigger (drop first to allow idempotent re-runs)
-- DROP TRIGGER IF EXISTS trg_document_vector_store_bm25 ON document_vector_store;

-- CREATE TRIGGER trg_document_vector_store_bm25
-- BEFORE INSERT OR UPDATE OF content ON document_vector_store
-- FOR EACH ROW
-- EXECUTE FUNCTION update_document_bm25_tsvector();

-- -- 4. GIN index for fast @@ operator
-- CREATE INDEX IF NOT EXISTS idx_dvs_bm25_tsvector_gin
--   ON document_vector_store
--   USING gin (bm25_tsvector);

-- 5. Backfill existing NULL rows in batches to avoid statement timeout
SET statement_timeout = 0;
DO $$
DECLARE
  batch_size INT := 5000;
  rows_updated INT;
BEGIN
  LOOP
    UPDATE document_vector_store
    SET bm25_tsvector = to_tsvector('english', COALESCE(content, ''))
    WHERE id IN (
      SELECT id FROM document_vector_store
      WHERE bm25_tsvector IS NULL
      LIMIT batch_size
    );
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    EXIT WHEN rows_updated = 0;
    RAISE NOTICE 'Backfilled % rows', rows_updated;
  END LOOP;
END $$;