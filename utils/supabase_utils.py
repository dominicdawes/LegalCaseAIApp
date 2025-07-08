import os
import json
import logging
import uuid
import boto3
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_client: Client = create_client(
    os.getenv("SUPABASE_PROJECT_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),  # Ensure this key is the service role
)

# Init logger
logger = logging.getLogger(__name__)


def create_new_chat_session(
    client,
    table_name,
    user_id,
    project_id,
):
    """INSERT/CREATE a new chat_session object table public.chat_sessions"""
    try:
        response = (
            client.table(table_name)
            .insert(
                {
                    "user_id": user_id,
                    "project_id": project_id,
                }
            )
            .execute()
        )
    except Exception as e:
        raise Exception(f"Error saving to public.chat_sessions: {e}")


def insert_mp3_supabase_record(
    client,
    table_name,
    podcast_title,
    cdn_url,
    transcript,
    content_tags,
    uploaded_by,
    is_public,
    is_playlist,
):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "podcast_title": podcast_title,
            "cdn_url": cdn_url,
            "uploaded_by": uploaded_by,
            "is_public": is_public,
            "transcript": transcript,
            "description": "this description could be workshopped just a bit...",
            "is_playlist": False,
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()

        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase 'podcast' row insert")
            return response.data[0][
                "id"
            ]  # Return podcast ID... prior version is just response.data
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")


def insert_document_supabase_record(
    client, table_name, cdn_url, project_id, content_tags, uploaded_by
):
    """Inserts a record into the Supabase Library table."""
    try:
        data = {
            "cdn_url": cdn_url,
            "project_id": project_id,
            "content_tags": content_tags,
            "uploaded_by": uploaded_by,
        }
        # Execute the insert query
        response = client.table(table_name).insert(data).execute()

        # Check if the response contains data
        if response.data:
            # Successful insertion
            print("Successful Supabase row insert")
            return response.data[0]["id"]  # Return document_sources.id
        else:
            # If there's no data, check for errors
            raise Exception(f"Supabase error: {response.error}")
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase: {e}")


def insert_vector_supabase_record(
    client, table_name, source_id, project_id, content, metadata, embedding
):
    """INSERT into table public.document_vector_store"""
    try:
        response = (
            client.table(table_name)
            .insert(
                {
                    "source_id": source_id,
                    "content": content,
                    "metadata": json.dumps(metadata),
                    "embedding": embedding,
                    "project_id": project_id,
                }
            )
            .execute()
        )
        logger.info(
            f"Response status={response.status} and statusText={response.statusText}"
        )
    except Exception as e:
        raise Exception(f"Failed to insert into Supabase vector-store: {e}")


def insert_note_supabase_record(
    client,
    table_name,
    user_id,
    project_id,
    note_title,
    content_markdown,
    note_type,
    is_generated,
    is_shareable,
    created_at,
):
    """INSERT note into table public.note"""
    try:
        response = (
            client.table(table_name)
            .insert(
                {
                    "user_id": user_id,
                    "project_id": project_id,
                    "note_type": note_type,
                    "title": note_title,
                    "content_markdown": content_markdown,
                    "created_at": created_at,
                    "is_generated": is_generated,
                    "is_shareable": is_shareable,
                }
            )
            .execute()
        )
    except Exception as e:
        raise Exception(f"Error saving to public.note: {e}")


def insert_chat_message_supabase_record(
    client,
    table_name,
    user_id,
    chat_session_id,
    dialogue_role,
    message_content,
    query_response_status,
    created_at,
):
    """INSERT message into table public.messages"""
    try:

        raw = {"raw": "boiler plate"}

        response = (
            client.table(table_name)
            .insert(
                {
                    "user_id": user_id,
                    "chat_session_id": chat_session_id,
                    "role": dialogue_role,  # enum(user, assistant)
                    "content": message_content,
                    "query_response_status": query_response_status,  # status for SUCCESS or FAILED llm query response
                    "raw_response": raw,
                    "created_at": created_at,
                }
            )
            .execute()
        )
        return response
    except Exception as e:
        raise Exception(f"Error saving to public.messages: {e}")


def update_table_realtime_status_log(
    client,
    table_name,
    project_uuid,
    log,
):
    """UPDATE error/status log in public.table_name, mainly (public.projects)"""
    try:
        response = (
            client.table(table_name)
            .update({"status_logger": log})
            .eq("id", project_uuid)
            .execute()
        )  # Use .eq() to target the specific row by its 'id'

        # Check for errors in the response
        if response.data:
            print(f"Successfully updated row {project_uuid} in public.{table_name}.")
            return response.data
        else:
            raise Exception(
                f"No data returned or error updating row {project_uuid}: {response.error}"
            )

    except Exception as e:
        raise Exception(f"Error saving to public.{table_name}: {e}")


def update_document_sources_realtime_status_log(
    status: str, source_id: str, error_message: str = None
):
    """Helper function to update the status in document_sources."""
    try:
        payload = {"vector_embed_status": status}
        if error_message:
            payload["error_message"] = error_message[:255]  # if you have that column
            logger.error(
                f"[DB] Setting status={status} for {source_id} w/ error: {error_message}"
            )

        logger.debug(
            f"[DB] update document_sources set status={status} where id={source_id}"
        )
        update_resp = (
            supabase_client.table("document_sources")
            .update(payload)
            .eq("id", source_id)
            .execute()
        )

        # **New**: inspect the Supabase response
        if hasattr(update_resp, "data"):
            logger.debug(f"[DB] Update returned data: {update_resp.data}")
        if hasattr(update_resp, "status_code"):
            logger.debug(f"[DB] HTTP status code: {update_resp.status_code}")

    except Exception as db_e:
        logger.critical(
            f"[DB] Failed to update status for {source_id}: {db_e}", exc_info=True
        )


def log_llm_error(
    client, table_name, task_name, error_message, project_id=None, user_id=None
):
    """Insert a log row for LLM related errors."""
    payload = {
        "task_name": task_name,
        "error_message": error_message,
        "project_id": project_id,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.table(table_name).insert(payload).execute()
    except Exception as e:
        logger.error(f"Failed to log LLM error: {e}", exc_info=True)
