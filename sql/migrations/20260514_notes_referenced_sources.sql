-- =============================================================================
-- Migration: add referenced_sources column to notes
-- Stores the document_sources UUIDs whose chunks were used during RAG.
-- Safe to run multiple times (IF NOT EXISTS throughout).
-- Apply in the Supabase SQL editor.
-- =============================================================================

ALTER TABLE notes
  ADD COLUMN IF NOT EXISTS referenced_sources uuid[] DEFAULT '{}';
