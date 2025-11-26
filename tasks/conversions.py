# tasks/conversions.py
# Converting chat Responses into notes

import uuid
import logging
from datetime import datetime, timezone
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from celery.utils.log import get_task_logger

# Project modules
from tasks.celery_app import celery_app, run_async_in_worker
from tasks.database import get_db_connection

logger = get_task_logger(__name__)

# ——— Configuration ————————————————————————————————————————————————————
CONVERSION_QUEUE = 'notes' # Reusing the notes queue for simplicity
MAX_RETRIES = 3

class BaseTaskWithRetry(Task):
    """Base task with automatic retries"""
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": MAX_RETRIES}
    retry_jitter = True

# ——— Async Implementation —————————————————————————————————————————————

async def _save_converted_note_async(
    user_id: str,
    project_id: str,
    content: str,
    note_type: str
) -> str:
    """
    Async implementation to insert the chat content as a note.
    Generates a default title based on timestamp or content content.
    """
    try:
        note_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Generate a simple title since one isn't provided
        # e.g., "Chat Save Dec 12, 10:00 AM"
        readable_date = created_at.strftime("%b %d, %H:%M")
        title = f"Chat Note - {readable_date}"
        
        # Database insertion
        async with get_db_connection() as conn:
            await conn.execute(
                """
                INSERT INTO notes (
                    id, user_id, project_id, title, content_markdown, 
                    note_type, is_generated, is_shareable, created_at, is_essential
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                note_id, 
                uuid.UUID(user_id), 
                uuid.UUID(project_id), 
                title, 
                content,
                note_type, 
                True,  # is_generated (Assuming yes since it comes from chat)
                False, # is_shareable
                created_at, 
                False  # is_essential
            )
            
        logger.info(f"✅ Successfully converted chat to note {note_id}")
        return note_id

    except Exception as e:
        logger.error(f"❌ Failed to save converted note: {e}")
        raise

# ——— Celery Task Wrapper ——————————————————————————————————————————————

@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    queue=CONVERSION_QUEUE,
    acks_late=True
)
def convert_chat_to_note_task(
    self,
    user_id: str,
    project_id: str,
    content: str,
    note_type: str = "chat"
):
    """
    Celery task to persist chat content as a note.
    """
    try:
        task_id = self.request.id
        logger.info(f"🚀 Starting conversion task {task_id}")

        # Execute async logic in the worker's persistent event loop
        # consistent with note_tasks.py pattern
        note_id = run_async_in_worker(
            _save_converted_note_async(
                user_id=user_id,
                project_id=project_id,
                content=content,
                note_type=note_type
            )
        )
        
        return {"note_id": note_id, "status": "SUCCESS"}

    except Exception as e:
        logger.error(f"❌ Conversion task failed: {e}")
        raise self.retry(exc=e)