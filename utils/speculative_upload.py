# utils/speculative_upload.py

"""
Speculative Pre-Upload helpers.

Redis gate pattern:
  Key:    pending_uploads:{chat_session_id}   (Redis Hash)
  Fields: {doc_id} → {celery_task_id}
  TTL:    600s (safety net against orphaned uploads)

Workflow:
  1. On drag-drop  → register_speculative_upload()
  2. After embed   → clear_speculative_upload()   (called from upload_tasks.py)
  3. User clicks x → cancel_speculative_upload()
  4. User hits Send → persist_inline_upload()

Deferred note gate:
  Key:    pending_note:{project_id}   (Redis String, JSON payload)
  TTL:    3600s

  Ingest now starts on drag-drop, before the user has chosen a note type, so the
  note request and the ingest completion arrive in either order.  Both sides call
  try_fire_pending_note(); whichever finds all documents terminal claims the key
  with a DEL that returns 1 and dispatches rag_note_task.  See that function.

**IN THE UI**

Drag-drop
    └─ POST /speculative-ingest/         ← Goal 2 (backend)
        returns { doc_id, document: {...} }

Send button
    ├─ Append file cards to chatHistory   ← Goal 1 (WeWeb variable)
    ├─ Supabase INSERT into messages      ← Goal 1 (WeWeb direct)
    │    with inline_documents = jsonb
    ├─ Append query to chatHistory        ← existing
    └─ POST /rag-chat/                    ← existing + Goal 2 barrier

"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tasks.database import get_redis_connection, get_db_connection
from utils.supabase_utils import supabase_client

logger = logging.getLogger(__name__)

_GATE_TTL = 600   # seconds — speculative upload gate
_NOTE_TTL = 3600  # seconds — deferred note request

# A document is "terminal" once ingest can do no more with it. FAILED_* counts:
# a note should still be attempted from whatever else embedded successfully,
# and finalize_batch_and_create_note handles the all-failed case separately.
_TERMINAL_STATUSES = {
    "COMPLETE",
    "PARTIAL",
    "FAILED_DOWNLOAD",
    "FAILED_PARSING",
    "FAILED_EMBEDDING",
    "FAILED_FINALIZATION",
}


# ——— Redis Gate ——————————————————————————————————————————————————————————————


async def register_speculative_upload(
    chat_session_id: str,
    doc_id: str,
    celery_task_id: str,
) -> None:
    """Register a doc as speculatively uploading. Sets the Redis gate."""
    gate_key = f"pending_uploads:{chat_session_id}"
    async with get_redis_connection() as r:
        await r.hset(gate_key, doc_id, celery_task_id)
        await r.expire(gate_key, _GATE_TTL)
    logger.info(f"🚦 Registered speculative upload {doc_id[:8]} for session {chat_session_id[:8]}")


async def clear_speculative_upload(chat_session_id: str, doc_id: str) -> None:
    """Remove a doc from the Redis gate once its embeddings are committed."""
    async with get_redis_connection() as r:
        await r.hdel(f"pending_uploads:{chat_session_id}", doc_id)
    logger.info(f"✅ Cleared speculative upload {doc_id[:8]} for session {chat_session_id[:8]}")


# ——— Deferred Note Gate ——————————————————————————————————————————————————————


def _note_key(project_id: str) -> str:
    return f"pending_note:{project_id}"


async def register_pending_note(project_id: str, payload: Dict[str, Any]) -> None:
    """
    Record a note request that should fire once its documents finish ingesting.

    `payload` must carry `document_ids` plus the full rag_note_task kwargs.
    """
    async with get_redis_connection() as r:
        await r.set(_note_key(project_id), json.dumps(payload), ex=_NOTE_TTL)
    logger.info(
        f"📌 Registered pending note {payload.get('note_id', '?')[:8]} "
        f"for project {project_id[:8]} ({len(payload.get('document_ids', []))} doc(s))"
    )


async def _all_documents_terminal(document_ids: List[str]) -> bool:
    """True when every requested document has finished ingesting (or failed)."""
    if not document_ids:
        return True

    async with get_db_connection() as conn:
        rows = await conn.fetch(
            "SELECT id, vector_embed_status FROM document_sources WHERE id = ANY($1::uuid[])",
            document_ids,
        )

    statuses = {str(row["id"]): row["vector_embed_status"] for row in rows}

    # A missing row means the doc was cancelled or never created — don't block on it.
    pending = [
        doc_id
        for doc_id in document_ids
        if doc_id in statuses and statuses[doc_id] not in _TERMINAL_STATUSES
    ]

    if pending:
        logger.info(
            f"⏳ Pending note still waiting on {len(pending)} document(s): "
            f"{[p[:8] for p in pending]}"
        )
    return not pending


async def try_fire_pending_note(project_id: str) -> Optional[str]:
    """
    Dispatch the deferred note for `project_id` if its documents are all done.

    Called from BOTH ends of the race:
      * POST /new-rag-project/attach-note/ — covers "ingest finished while the
        user was still picking a note type"
      * finalize_batch_and_create_note     — covers the normal case

    The claim is the DEL: Redis returns the number of keys removed, so exactly
    one caller can ever see 1 and dispatch, no matter how the two interleave or
    how many per-file batches finalize concurrently.

    Returns the note_id if this call dispatched, else None.
    """
    key = _note_key(project_id)

    async with get_redis_connection() as r:
        raw = await r.get(key)

    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(f"⚠️ Corrupt pending_note payload for project {project_id[:8]} — dropping")
        async with get_redis_connection() as r:
            await r.delete(key)
        return None

    if not await _all_documents_terminal(payload.get("document_ids", [])):
        return None

    # ── Claim ─────────────────────────────────────────────────────────────
    async with get_redis_connection() as r:
        claimed = await r.delete(key)

    if not claimed:
        logger.info(f"🤝 Pending note for project {project_id[:8]} already claimed elsewhere")
        return None

    note_id = payload.get("note_id")
    task_kwargs = payload.get("task_kwargs", {})

    # Imported here: tasks.note_tasks pulls in the whole Celery app, and this
    # module is imported from inside it.
    from tasks.note_tasks import rag_note_task

    rag_note_task.apply_async(kwargs=task_kwargs)
    logger.info(
        f"🎯 Fired deferred note {str(note_id)[:8]} "
        f"({payload.get('note_type')}) for project {project_id[:8]}"
    )

    # Tell the chat page the note phase has started.
    try:
        from utils.progress_events import publish_note_progress, NoteStage

        await publish_note_progress(
            payload.get("chat_session_id"),
            str(note_id),
            NoteStage.NOTE_STARTED,
            payload.get("note_type"),
        )
    except Exception as e:
        logger.debug(f"Note progress publish failed: {e}")

    return note_id


async def cancel_pending_note(project_id: str) -> None:
    """Drop a deferred note request (e.g. the user abandoned the project)."""
    async with get_redis_connection() as r:
        await r.delete(_note_key(project_id))


# ——— Cancellation ————————————————————————————————————————————————————————————


async def cancel_speculative_upload(
    chat_session_id: str,
    doc_id: str,
    project_id: str,
) -> None:
    """
    User cancelled the file. Undo everything:
    1. Revoke the Celery task
    2. Clear from Redis gate
    3. Delete embeddings if any landed
    4. Delete the document_source row
    5. Delete from S3
    """
    from tasks.celery_app import celery_app

    # 1 & 2 — Revoke and clear gate
    async with get_redis_connection() as r:
        celery_task_id = await r.hget(f"pending_uploads:{chat_session_id}", doc_id)
        if celery_task_id:
            task_id_str = celery_task_id.decode() if isinstance(celery_task_id, bytes) else celery_task_id
            celery_app.control.revoke(task_id_str, terminate=True, signal="SIGTERM")
            logger.info(f"🛑 Revoked Celery task {task_id_str} for doc {doc_id[:8]}")
        await r.hdel(f"pending_uploads:{chat_session_id}", doc_id)

    # 3 & 4 — Delete DB rows
    async with get_db_connection() as conn:
        await conn.execute(
            "DELETE FROM document_chunks WHERE document_id = $1", doc_id
        )
        await conn.execute(
            "DELETE FROM document_sources WHERE id = $1", doc_id
        )

    # 5 — Best-effort S3 cleanup
    try:
        from utils.s3_utils import delete_from_s3
        await asyncio.to_thread(delete_from_s3, doc_id)
    except Exception as e:
        logger.warning(f"⚠️ S3 cleanup failed for {doc_id[:8]}: {e}")

    logger.info(f"🚫 Speculative upload {doc_id[:8]} fully cancelled")


# ——— Inline Timeline Message ————————————————————————————————————————————————


def persist_inline_upload(
    user_id: str,
    chat_session_id: str,
    document_ids: List[str],
) -> str:
    """
    Insert a synthetic file_upload anchor message into the conversation timeline.
    Returns the new message id.
    """
    response = (
        supabase_client.table("messages")
        .insert(
            {
                "user_id": user_id,
                "chat_session_id": chat_session_id,
                "role": "user",
                "message_type": "file_upload",
                "content": "",
                "inline_document_ids": document_ids,
                "status": "complete",
                "format": "markdown",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .execute()
    )
    message_id = response.data[0]["id"]
    logger.info(
        f"📎 Persisted file_upload message {message_id[:8]} "
        f"for {len(document_ids)} doc(s) in session {chat_session_id[:8]}"
    )
    return message_id
