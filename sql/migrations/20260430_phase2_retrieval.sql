-- =============================================================================
-- Phase 2 Migration: hierarchical schema + updated RPCs
-- Apply this entire script in the Supabase SQL editor.
-- Safe to run multiple times (IF NOT EXISTS / CREATE OR REPLACE throughout).
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Rename existing ada-002 column (if it hasn't been renamed already)
--    The original column created by the legacy ingest path was called `embedding`.
--    We rename it so the purpose is unambiguous going forward.
-- -----------------------------------------------------------------------------

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'document_vector_store' AND column_name = 'embedding'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'document_vector_store' AND column_name = 'embedding_ada_legacy'
  ) THEN
    ALTER TABLE document_vector_store
      RENAME COLUMN embedding TO embedding_ada_legacy;
  END IF;
END $$;


-- -----------------------------------------------------------------------------
-- 2. Add voyage-law-2 column (1024-dim)
-- -----------------------------------------------------------------------------

ALTER TABLE document_vector_store
  ADD COLUMN IF NOT EXISTS embedding_voyage_2 vector(1024);


-- -----------------------------------------------------------------------------
-- 3. Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS document_vector_store_embedding_ada_legacy_hnsw
  ON document_vector_store
  USING hnsw (embedding_ada_legacy vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS document_vector_store_embedding_voyage_2_hnsw
  ON document_vector_store
  USING hnsw (embedding_voyage_2 vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS document_vector_store_embedding_voyage_2_source
  ON document_vector_store (source_id)
  WHERE embedding_voyage_2 IS NOT NULL;


-- -----------------------------------------------------------------------------
-- 4. Updated match_document_chunks_hnsw RPC
--
--    p_use_voyage boolean:
--      true  -> searches embedding_voyage_2   (voyage-law-2, 1024-dim)
--      false -> searches embedding_ada_legacy (ada-002,      1536-dim)
--
--    All new params default to NULL/false so existing 3-param callers
--    continue to work unchanged.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION match_document_chunks_hnsw(
    p_project_id     uuid,
    p_query          vector,
    p_k              int     DEFAULT 10,
    p_chunk_types    text[]  DEFAULT NULL,
    p_source_ids     uuid[]  DEFAULT NULL,
    p_section_prefix text    DEFAULT NULL,
    p_use_voyage     boolean DEFAULT false
)
RETURNS TABLE (
    id               uuid,
    source_id        uuid,
    project_id       uuid,
    content          text,
    chunk_summary    text,
    section_path     text,
    chunk_type       text,
    parent_chunk_id  uuid,
    metadata         jsonb,
    page_number      int,
    chunk_index      int,
    similarity       double precision
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    IF p_use_voyage THEN
        RETURN QUERY
            SELECT
                dvs.id,
                dvs.source_id,
                dvs.project_id,
                dvs.content,
                dvs.chunk_summary,
                dvs.section_path,
                dvs.chunk_type,
                dvs.parent_chunk_id,
                dvs.metadata,
                dvs.page_number,
                dvs.chunk_index,
                (1.0 - (dvs.embedding_voyage_2 <=> p_query))::double precision AS similarity
            FROM document_vector_store dvs
            WHERE dvs.project_id = p_project_id
              AND dvs.embedding_voyage_2 IS NOT NULL
              AND (p_chunk_types    IS NULL OR dvs.chunk_type  = ANY(p_chunk_types))
              AND (p_source_ids     IS NULL OR dvs.source_id   = ANY(p_source_ids))
              AND (p_section_prefix IS NULL OR dvs.section_path LIKE p_section_prefix || '%')
            ORDER BY dvs.embedding_voyage_2 <=> p_query
            LIMIT p_k;
    ELSE
        RETURN QUERY
            SELECT
                dvs.id,
                dvs.source_id,
                dvs.project_id,
                dvs.content,
                dvs.chunk_summary,
                dvs.section_path,
                dvs.chunk_type,
                dvs.parent_chunk_id,
                dvs.metadata,
                dvs.page_number,
                dvs.chunk_index,
                (1.0 - (dvs.embedding_ada_legacy <=> p_query))::double precision AS similarity
            FROM document_vector_store dvs
            WHERE dvs.project_id = p_project_id
              AND dvs.embedding_ada_legacy IS NOT NULL
              AND (p_chunk_types    IS NULL OR dvs.chunk_type  = ANY(p_chunk_types))
              AND (p_source_ids     IS NULL OR dvs.source_id   = ANY(p_source_ids))
              AND (p_section_prefix IS NULL OR dvs.section_path LIKE p_section_prefix || '%')
            ORDER BY dvs.embedding_ada_legacy <=> p_query
            LIMIT p_k;
    END IF;
END;
$$;


-- -----------------------------------------------------------------------------
-- 5. Section-level vector search RPC
--    document_sections.embedding is a single column (active model at ingest time).
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION match_document_sections(
    p_project_id uuid,
    p_query      vector,
    p_k          int    DEFAULT 5,
    p_source_ids uuid[] DEFAULT NULL
)
RETURNS TABLE (
    id               uuid,
    source_id        uuid,
    project_id       uuid,
    section_path     text,
    section_summary  text,
    start_chunk_idx  int,
    end_chunk_idx    int,
    similarity       double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        ds.id,
        ds.source_id,
        ds.project_id,
        ds.section_path,
        ds.section_summary,
        ds.start_chunk_idx,
        ds.end_chunk_idx,
        (1.0 - (ds.embedding <=> p_query))::double precision AS similarity
    FROM document_sections ds
    WHERE ds.project_id = p_project_id
      AND ds.embedding IS NOT NULL
      AND (p_source_ids IS NULL OR ds.source_id = ANY(p_source_ids))
    ORDER BY ds.embedding <=> p_query
    LIMIT p_k;
$$;


-- -----------------------------------------------------------------------------
-- 6. Grants
-- -----------------------------------------------------------------------------

GRANT EXECUTE ON FUNCTION match_document_chunks_hnsw(uuid, vector, int, text[], uuid[], text, boolean)
    TO authenticated, service_role;

GRANT EXECUTE ON FUNCTION match_document_sections(uuid, vector, int, uuid[])
    TO authenticated, service_role;
