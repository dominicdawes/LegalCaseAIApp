# tasks/hierarchical_ingest_tasks.py

"""
Phase 1 post-embed Celery sub-tasks for hierarchical ingest.

These tasks are fired as async (fire-and-forget) by _process_document_async_workflow
AFTER the main chunk embedding is complete, so they never block note generation.

Tasks
-----
extract_doc_summary     — cheap LLM call → document_sources.doc_summary (~140 chars)
extract_doc_concepts    — cheap LLM call → document_sources.doc_concepts (text[])
build_section_summaries — groups chunks by section_path, summarises each with cheap
                          LLM, embeds with Voyage, inserts into document_sections table
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

from tasks.celery_app import celery_app, run_async_in_worker
from tasks.database import get_db_connection, init_async_pools, get_global_async_db_pool

load_dotenv()
logger = get_task_logger(__name__)
logger.propagate = False

# ——— Config ———————————————————————————————————————————————————————————————————

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()

# Cheap model for doc-level metadata extraction
CHEAP_MODEL = "claude-haiku-4-5-20251001"

# Section summary model (same cheap tier)
SECTION_SUMMARY_MODEL = "claude-haiku-4-5-20251001"

# Max characters of document sample passed to extract_doc_* tasks
DOC_SAMPLE_MAX_CHARS = 8_000

# Max characters of section text passed to build_section_summaries
SECTION_TEXT_MAX_CHARS = 4_000


# ——— Shared HTTP helper ——————————————————————————————————————————————————————


async def _call_anthropic_simple(prompt: str, max_tokens: int = 256) -> str:
    """Minimal Anthropic call (no caching) for doc-level metadata extraction."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CHEAP_MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


# ——— Task: extract_doc_summary ————————————————————————————————————————————————


@celery_app.task(bind=True, queue="ingest", acks_late=True, max_retries=3, default_retry_delay=10)
def extract_doc_summary(self, source_id: str, doc_sample_text: str) -> str:
    """
    Generate a ~140-char document summary and persist it to document_sources.doc_summary.

    Args:
        source_id      — UUID of the document_sources row
        doc_sample_text — first N chars of the parsed document text
    """
    try:
        return run_async_in_worker(_extract_doc_summary_async(source_id, doc_sample_text))
    except Exception as exc:
        logger.error(f"❌ extract_doc_summary failed for {source_id[:8]}: {exc}")
        raise self.retry(exc=exc)


async def _extract_doc_summary_async(source_id: str, doc_sample_text: str) -> str:
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    prompt = (
        "You are summarising a legal document for a search index.\n\n"
        "Write a single sentence of NO MORE than 140 characters that captures "
        "the document's primary legal topic, jurisdiction (if stated), and type "
        "(e.g. case brief, statute, law review article). "
        "Output only the sentence, no preamble.\n\n"
        f"DOCUMENT EXCERPT:\n{doc_sample_text[:DOC_SAMPLE_MAX_CHARS]}"
    )

    summary = await _call_anthropic_simple(prompt, max_tokens=80)
    summary = summary[:140]  # hard cap

    async with get_db_connection() as conn:
        await conn.execute(
            "UPDATE document_sources SET doc_summary = $1 WHERE id = $2",
            summary, source_id,
        )

    logger.info(f"📝 doc_summary saved for {source_id[:8]}: {summary[:60]}…")
    return summary


# ——— Task: extract_doc_concepts ———————————————————————————————————————————————


@celery_app.task(bind=True, queue="ingest", acks_late=True, max_retries=3, default_retry_delay=10)
def extract_doc_concepts(self, source_id: str, doc_sample_text: str) -> List[str]:
    """
    Extract key legal concepts and persist to document_sources.doc_concepts (text[]).

    Args:
        source_id       — UUID of the document_sources row
        doc_sample_text — first N chars of the parsed document text
    """
    try:
        return run_async_in_worker(_extract_doc_concepts_async(source_id, doc_sample_text))
    except Exception as exc:
        logger.error(f"❌ extract_doc_concepts failed for {source_id[:8]}: {exc}")
        raise self.retry(exc=exc)


async def _extract_doc_concepts_async(source_id: str, doc_sample_text: str) -> List[str]:
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    prompt = (
        "You are a legal indexer.  Extract 5-10 key legal concepts, doctrines, "
        "or topics from the document excerpt below.\n"
        "Return ONLY a JSON array of short concept strings (2-5 words each), "
        'e.g. ["promissory estoppel","statute of frauds","UCC Article 2"].\n'
        "No extra text.\n\n"
        f"DOCUMENT EXCERPT:\n{doc_sample_text[:DOC_SAMPLE_MAX_CHARS]}"
    )

    raw = await _call_anthropic_simple(prompt, max_tokens=200)

    # Robust JSON parse
    try:
        concepts: List[str] = json.loads(raw)
        if not isinstance(concepts, list):
            raise ValueError("not a list")
        concepts = [str(c) for c in concepts if c][:20]  # max 20
    except Exception:
        # Fallback: split by comma if plain text
        concepts = [c.strip().strip('"') for c in raw.split(",") if c.strip()][:20]

    async with get_db_connection() as conn:
        await conn.execute(
            "UPDATE document_sources SET doc_concepts = $1 WHERE id = $2",
            concepts, source_id,
        )

    logger.info(f"🏷️  doc_concepts saved for {source_id[:8]}: {concepts[:5]}…")
    return concepts


# ——— Task: build_section_summaries ————————————————————————————————————————————


@celery_app.task(bind=True, queue="ingest", acks_late=True, max_retries=3, default_retry_delay=30)
def build_section_summaries(self, source_id: str, project_id: str) -> Dict:
    """
    Group vector-store chunks by section_path, summarise each section with a
    cheap LLM call, embed summaries with Voyage, and insert into document_sections.

    Reads chunks from document_vector_store after the main embedding is complete.
    Skips sections that already have a row in document_sections (idempotent).

    Args:
        source_id  — UUID of the document_sources row
        project_id — UUID of the project
    """
    try:
        return run_async_in_worker(_build_section_summaries_async(source_id, project_id))
    except Exception as exc:
        logger.error(f"❌ build_section_summaries failed for {source_id[:8]}: {exc}")
        raise self.retry(exc=exc)


async def _build_section_summaries_async(source_id: str, project_id: str) -> Dict:
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()

    # ── 1. Load all chunks for this document ─────────────────────────────
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, chunk_summary, section_path, chunk_type, chunk_index
            FROM document_vector_store
            WHERE source_id = $1
              AND section_path IS NOT NULL
              AND section_path <> ''
            ORDER BY chunk_index
            """,
            source_id,
        )

    if not rows:
        logger.info(f"ℹ️  No sectioned chunks found for {source_id[:8]}, skipping.")
        return {"sections_built": 0}

    # ── 2. Group by section_path ──────────────────────────────────────────
    sections: Dict[str, List[Dict]] = {}
    for row in rows:
        sp = row["section_path"] or ""
        sections.setdefault(sp, []).append(dict(row))

    # ── 3. Check which sections already exist ────────────────────────────
    async with get_db_connection() as conn:
        existing = await conn.fetch(
            "SELECT section_path FROM document_sections WHERE source_id = $1",
            source_id,
        )
    existing_paths = {r["section_path"] for r in existing}

    # ── 4. Generate summaries and embed ──────────────────────────────────
    from utils.llm_clients.voyage_client import VoyageEmbeddingsClient
    voyage = VoyageEmbeddingsClient()

    new_sections: List[Dict] = []

    for section_path, chunks in sections.items():
        if section_path in existing_paths:
            continue

        # Concatenate chunk texts for this section (capped)
        section_text = "\n\n".join(
            c.get("chunk_summary") or c.get("content", "")
            for c in chunks
            if c.get("chunk_type") != "table"  # tables have their own summaries
        )[:SECTION_TEXT_MAX_CHARS]

        if len(section_text.strip()) < 50:
            continue

        # Determine start/end chunk indices
        indices = [c["chunk_index"] for c in chunks if c.get("chunk_index") is not None]
        start_idx = min(indices) if indices else 0
        end_idx = max(indices) if indices else 0

        # Generate section summary
        summary_prompt = (
            f"Summarise this section of a legal document in 2-4 sentences for a "
            f"retrieval index.  Section heading path: {section_path}\n\n"
            f"SECTION TEXT:\n{section_text}"
        )
        try:
            section_summary = await _call_anthropic_simple(summary_prompt, max_tokens=200)
        except Exception as e:
            logger.warning(f"⚠️ Section summary failed for '{section_path[:40]}': {e}")
            section_summary = section_text[:400]

        new_sections.append({
            "section_path": section_path,
            "section_summary": section_summary,
            "start_chunk_idx": start_idx,
            "end_chunk_idx": end_idx,
        })

    if not new_sections:
        return {"sections_built": 0}

    # ── 5. Batch-embed all section summaries ─────────────────────────────
    loop = asyncio.get_event_loop()
    summary_texts = [s["section_summary"] for s in new_sections]
    embeddings: List[List[float]] = await loop.run_in_executor(
        None, voyage.embed_documents, summary_texts
    )

    # ── 6. Insert into document_sections ─────────────────────────────────
    now = datetime.now(timezone.utc)
    records = []
    for section, vec in zip(new_sections, embeddings):
        records.append((
            str(uuid.uuid4()),
            source_id,
            project_id,
            section["section_path"],
            section["section_summary"],
            vec,
            section["start_chunk_idx"],
            section["end_chunk_idx"],
            now,
        ))

    async with get_db_connection() as conn:
        await conn.executemany(
            """
            INSERT INTO document_sections
              (id, source_id, project_id, section_path, section_summary,
               embedding, start_chunk_idx, end_chunk_idx, created_at)
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9)
            ON CONFLICT DO NOTHING
            """,
            records,
        )

    logger.info(
        f"✅ build_section_summaries: {len(new_sections)} sections built "
        f"for {source_id[:8]}"
    )
    return {"sections_built": len(new_sections)}
