# utils/progress_events.py

"""
Real-time ingest / note progress events.

These are published to the SAME Redis pub/sub channel the chat WebSocket already
listens on (`chat:{chat_session_id}` — see app/ws_handlers/endpoints.py), so the
WeWeb `/chat` page needs no second socket: it just handles two extra message
types on the connection it already holds open.

Message shapes
--------------
    {"type": "ingest_progress", "doc_id": "...", "stage": "BLURBS",
     "label": "Understanding the document", "pct": 70,
     "filename": "smith-v-jones.pdf", "timestamp": "..."}

    {"type": "note_progress", "note_id": "...", "stage": "NOTE_STARTED",
     "label": "Writing your Cold Call", "note_type": "cold_call",
     "timestamp": "..."}

    {"type": "waiting_for_documents", "remaining": 1, "elapsed_s": 8.4,
     "label": "Still reading your document...", "timestamp": "..."}

Design notes
------------
* Publishing is ALWAYS best-effort. A dead Redis connection or a closed socket
  must never fail an ingest — every helper swallows its own exceptions.
* Terminal state still lands in Postgres (`document_sources.vector_embed_status`,
  `notes.note_progress_status`) and is picked up by the existing Supabase
  realtime subscriptions. These events are a latency-hiding overlay, never the
  source of truth.
* `chat_session_id` is optional at every call site: the non-chat flows
  (/new-rag-project/, /embed-new-docs/) simply do not have one, and the helpers
  no-op rather than forcing callers to branch.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from tasks.database import get_redis_connection

logger = logging.getLogger(__name__)


# ——— Stage Definitions ————————————————————————————————————————————————————————


class IngestStage(str, Enum):
    """Stages surfaced to the user during document ingest."""

    RECEIVED = "RECEIVED"
    DOWNLOADED = "DOWNLOADED"
    PARSING = "PARSING"
    PARSED = "PARSED"
    BLURBS = "BLURBS"
    EMBEDDING = "EMBEDDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# stage → (percent complete, user-facing copy)
# Percentages are deliberately non-linear: they track *perceived* progress, not
# elapsed time. Parsing is the longest phase, so it spans the widest band.
_STAGE_META = {
    IngestStage.RECEIVED:   (5,   "Got your file"),
    IngestStage.DOWNLOADED: (15,  "Uploading your document"),
    IngestStage.PARSING:    (25,  "Reading the document"),
    IngestStage.PARSED:     (50,  "Reading the document"),
    IngestStage.BLURBS:     (70,  "Understanding the document"),
    IngestStage.EMBEDDING:  (85,  "Indexing for search"),
    IngestStage.COMPLETE:   (100, "Ready"),
    IngestStage.FAILED:     (100, "We couldn't read this document"),
}


class NoteStage(str, Enum):
    """Stages surfaced to the user during note generation."""

    NOTE_QUEUED = "NOTE_QUEUED"
    NOTE_STARTED = "NOTE_STARTED"
    NOTE_COMPLETE = "NOTE_COMPLETE"
    NOTE_FAILED = "NOTE_FAILED"


_NOTE_STAGE_LABELS = {
    NoteStage.NOTE_QUEUED:   "Queued",
    NoteStage.NOTE_STARTED:  "Writing your {note_type}",
    NoteStage.NOTE_COMPLETE: "Done",
    NoteStage.NOTE_FAILED:   "We couldn't finish this note",
}


# ——— Internal ————————————————————————————————————————————————————————————————


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _publish(chat_session_id: Optional[str], payload: dict) -> None:
    """
    Best-effort publish to the session's chat channel.

    No-ops when there is no chat_session_id (non-chat ingest flows) and never
    raises — progress reporting must not be able to fail an ingest.
    """
    if not chat_session_id:
        return

    try:
        async with get_redis_connection() as r:
            await r.publish(f"chat:{chat_session_id}", json.dumps(payload))
        logger.debug(
            f"📡 {payload.get('type')} → chat:{chat_session_id[:8]} "
            f"({payload.get('stage', '')})"
        )
    except Exception as e:
        logger.warning(f"⚠️ Progress publish failed ({payload.get('type')}): {e}")


# ——— Public API (async) ——————————————————————————————————————————————————————


async def publish_ingest_progress(
    chat_session_id: Optional[str],
    doc_id: str,
    stage: IngestStage,
    filename: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Emit one ingest stage transition for a single document."""
    pct, label = _STAGE_META.get(stage, (0, ""))
    await _publish(
        chat_session_id,
        {
            "type": "ingest_progress",
            "doc_id": doc_id,
            "stage": stage.value,
            "label": label,
            "pct": pct,
            "filename": filename,
            "detail": detail,
            "timestamp": _now(),
        },
    )


async def publish_note_progress(
    chat_session_id: Optional[str],
    note_id: str,
    stage: NoteStage,
    note_type: Optional[str] = None,
) -> None:
    """Emit one note-generation stage transition."""
    label = _NOTE_STAGE_LABELS.get(stage, "")
    if "{note_type}" in label:
        label = label.format(note_type=_humanize_note_type(note_type))

    await _publish(
        chat_session_id,
        {
            "type": "note_progress",
            "note_id": note_id,
            "stage": stage.value,
            "label": label,
            "note_type": note_type,
            "timestamp": _now(),
        },
    )


async def publish_waiting_for_documents(
    chat_session_id: Optional[str],
    remaining: int,
    elapsed_s: float,
) -> None:
    """
    Emitted by the RAG barrier while it holds a query waiting on in-flight
    embeddings, so the composer can explain the wait instead of looking hung.
    """
    await _publish(
        chat_session_id,
        {
            "type": "waiting_for_documents",
            "remaining": remaining,
            "elapsed_s": round(elapsed_s, 1),
            "label": "Still reading your document...",
            "timestamp": _now(),
        },
    )


# ——— Public API (sync) ————————————————————————————————————————————————————————


def publish_ingest_progress_sync(
    chat_session_id: Optional[str],
    doc_id: str,
    stage: IngestStage,
    filename: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """
    Sync wrapper for use inside Celery task bodies (which are plain functions).

    Routes onto the worker's persistent event loop when one exists, otherwise
    falls back to a throwaway loop. Fire-and-forget; never raises.
    """
    if not chat_session_id:
        return

    coro = publish_ingest_progress(chat_session_id, doc_id, stage, filename, detail)
    try:
        # Imported lazily: tasks.celery_app imports the task modules, which
        # import this module — a module-level import would be circular.
        from tasks.celery_app import run_async_in_worker

        run_async_in_worker(coro)
    except Exception:
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.warning(f"⚠️ Sync progress publish failed: {e}")


# ——— Helpers —————————————————————————————————————————————————————————————————


_NOTE_TYPE_LABELS = {
    "cold_call": "Cold Call",
    "case_brief": "Case Brief",
    "exam_questions": "Exam Questions",
    "attack_outline": "Attack Outline",
    "outline": "Outline",
    "quiz": "Quiz",
    "flashcards": "Flashcards",
}


def _humanize_note_type(note_type: Optional[str]) -> str:
    """Mirror of docs/note_to_readable.json for user-facing progress copy."""
    if not note_type:
        return "note"
    return _NOTE_TYPE_LABELS.get(note_type, note_type.replace("_", " ").title())
