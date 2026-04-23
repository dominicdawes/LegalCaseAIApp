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
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from tasks.database import get_redis_connection, get_db_connection
from utils.supabase_utils import supabase_client

logger = logging.getLogger(__name__)

_GATE_TTL = 600  # seconds


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
