"""
This file runs Celery tasks for handling RAG chat tasks (non-streaming LLM responses)
Handles RAG chat tasks without token streaming. Returns full answer in one go.

Models
o4-mini
gpt-4.1-nano

"""

from celery import Task
from tasks.celery_app import (
    celery_app,
)  # Import the Celery app instance (see celery_app.py for LocalHost config)
import logging
import os
import gc
import redis
from utils.supabase_utils import (
    supabase_client,
    insert_chat_message_supabase_record,
    create_new_chat_session,
    log_llm_error,
)
from datetime import datetime, timezone
import tempfile
import uuid
import json
import tiktoken
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from utils.prompt_utils import load_yaml_prompt, build_chat_messages_from_yaml


#### GLOBAL ####
# API Keys
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")  # ← only for embeddings

# Define which models support temp (and their allowed ranges, if you like)
_MODEL_TEMPERATURE_CONFIG = {
    # no-temperature models
    "o4-mini": {"supports_temperature": False},
    "gpt-4o-mini": {"supports_temperature": False},
    # temperature-capable models
    "gpt-4.1-nano": {"supports_temperature": True, "min": 0.0, "max": 2.0},
}

logger = logging.getLogger(__name__)

# Initialize Redis sync client for pub/sub
REDIS_LABS_URL = (
    "redis://default:"
    + os.getenv("REDIS_PASSWORD")
    + "@"
    + os.getenv("REDIS_PUBLIC_ENDPOINT")
)
redis_sync = redis.Redis.from_url(REDIS_LABS_URL, decode_responses=True)


class BaseTaskWithRetry(Task):
    """
    Base Celery Task class enabling automatic retries on exceptions.
    """

    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": 5}
    retry_jitter = True


def publish_token(chat_session_id: str, token: str):
    """
    Stub function to publish a single token to clients subscribed to a chat session.
    Replace with actual pub/sub (e.g. Redis, Supabase Realtime).
    """
    # Example: redis_sync.publish(f"chat:{chat_session_id}", token)
    pass


class StreamToClientHandler(BaseCallbackHandler):
    """
    LangChain callback handler that emits each new LLM token via publish_token().
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        publish_token(self.session_id, token)


def get_chat_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    callback_manager: any = None,
) -> "ChatOpenAI":
    """
    DEPRECATED!!!!
    Factory to create a ChatOpenAI instance.
    If callback_manager is provided, enable streaming and attach handlers.

    (default) OpenAI o4-mini model
    """
    from langchain_openai import ChatOpenAI  # Lazy-import to reduce startup footprint

    cfg = _MODEL_TEMPERATURE_CONFIG.get(model_name, {"supports_temperature": True})
    llm_kwargs = {
        "api_key": OPENAI_API_KEY,
        "model": model_name,
        "streaming": False,  # Disable streaming by default
    }
    # Only include temperature if this model supports it
    if cfg.get("supports_temperature", False):
        lo, hi = cfg.get("min", 0.0), cfg.get("max", 2.0)
        safe_temp = max(lo, min(temperature, hi))
        llm_kwargs["temperature"] = safe_temp

    # Attach streaming callback if provided
    if callback_manager:
        llm_kwargs["streaming"] = True
        llm_kwargs["callback_manager"] = callback_manager

    return ChatOpenAI(**llm_kwargs)


@celery_app.task(bind=True, base=BaseTaskWithRetry)
def new_chat_session(self, user_id, project_id):
    """
    Simple task to creae a new chat_session and associate it query to public.messages in Supabase before performing RAG Q&A

    1) INSERT a new blank conversation row into Supabase public.chat_sessions
    2) UPDATE a given public.project's newly created "chat_session_id" column
    """
    try:
        # Step 1) Create blank chat_session object if needed
        response = create_new_chat_session(
            supabase_client,
            table_name="chat_sessions",
            user_id=user_id,
            project_id=project_id,
        )
        new_chat_session_id = response.data[0]["id"]

        # Step 2) UPDATE/REPLACE public.projects, chat_session_id column
        try:
            _ = (
                supabase_client.table("projects")
                .update({"chat_session_id": new_chat_session_id})
                .eq("id", project_id)
                .execute()
            )
        except Exception as e:
            raise Exception(f"Error saving to public.projects: {e}")
        return None

    except Exception as e:
        logger.error(f"Create new_chat_session task failed: {e}", exc_info=True)
        # Retry on failure according to BaseTaskWithRetry policies
        raise self.retry(exc=e)


# def create_new_conversation(user_id, project_id):
#     """
#     🆕 Inserts a new row into chat_session and returns its UUID... for now not in use (using WeWeb to insert a new convo)
#     """
#     new_chat_session = {
#         "user_id": user_id,
#         "created_at": datetime.now(timezone.utc).isoformat(),
#         "project_id": project_id
#     }
#     response = supabase_client.table("chat_session").insert(new_chat_session).execute()
#     return response.data[0]["id"]


def restart_chat_session(user_id: str, project_id: str) -> str:
    """
    🔃 Creates a new chat session in public.chat_sessions, updates the
    chat_session_id in public.projects, then deletes the old session.

    Returns the new chat_session_id.
    """
    # 1) Fetch the old session id for this project
    proj_res = (
        supabase_client.table("projects")
        .select("chat_session_id")
        .eq("id", project_id)
        .single()
        .execute()
    )
    if proj_res.error:
        raise RuntimeError(f"Error fetching project: {proj_res.error.message}")

    old_session_id = proj_res.data["chat_session_id"]

    # 2) Insert a new chat_sessions row
    new_chat_session = {
        "user_id": user_id,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    insert_res = (
        supabase_client.table("chat_sessions").insert(new_chat_session).execute()
    )
    if insert_res.error:
        raise RuntimeError(f"Error creating chat session: {insert_res.error.message}")

    # Supabase returns a list of inserted rows
    new_session_id = insert_res.data[0]["id"]

    # 3) Update the project to point to the new session
    update_res = (
        supabase_client.table("projects")
        .update({"chat_session_id": new_session_id})
        .eq("id", project_id)
        .execute()
    )
    if update_res.error:
        raise RuntimeError(f"Error updating project: {update_res.error.message}")

    # 4) Delete the old chat session
    #    (only if it existed in the first place)
    if old_session_id:
        delete_res = (
            supabase_client.table("chat_sessions")
            .delete()
            .eq("id", old_session_id)
            .execute()
        )
        if delete_res.error:
            raise RuntimeError(
                f"Error deleting old session: {delete_res.error.message}"
            )

    return new_session_id


@celery_app.task(bind=True, base=BaseTaskWithRetry)
def persist_user_query(self, user_id, chat_session_id, query, project_id, model_name):
    """
    Simple task to persist user query to public.messages in Supabase before performing RAG Q&A
    """
    try:
        # Set explicit start time metadata
        self.update_state(
            state="STARTED", meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # Step 1) Persist user query to public.messages
        response = insert_chat_message_supabase_record(
            supabase_client,
            table_name="messages",
            user_id=user_id,
            chat_session_id=chat_session_id,
            dialogue_role="user",
            message_content=query,
            query_response_status="PENDING",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 2) Return inserted record ID for downstream tasks
        inserted_id = response.data[0]["id"]
        return inserted_id

    except Exception as e:
        logger.error(f"RAG Chat Task failed: {e}", exc_info=True)
        log_llm_error(
            supabase_client,
            "llm_error_logs",
            "rag_chat_task",
            str(e),
            project_id=project_id,
            user_id=user_id,
        )
        raise self.retry(exc=e)


@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_chat_task(
    self,
    message_id,  # ← note: passed in first by chained task
    user_id,
    chat_session_id,
    query,
    project_id,
    provider: str,  # ← “openai”, “anthropic”, etc.
    model_name: str,
    temperature: float = 0.7,
):
    """
    Main RAG workflow, implementing the standard “optimistic UI” pattern
    1. Embed the user query
    2. Fetch top-K relevant chunks
    3. Generate llm_assistant answer
    4. Persist llm assistant answer in Supabase public.messages table
    """
    try:
        # Set explicit start time metadata
        self.update_state(
            state="STARTED", meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        if not model_name:
            model_name = "o4-mini"

        # Step 1) Embed the query
        # from langchain_openai import OpenAIEmbeddings       # Lazy-import heavy modules
        # from langchain.embeddings.openai import OpenAIEmbeddings    # Lazy-import heavy modules
        from langchain_community.embeddings import OpenAIEmbeddings

        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
        )
        query_embedding = embedding_model.embed_query(query)

        # Step 2) Fetch top-K relevant chunks via Supabase RPC
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)

        # Step 3) Generate llm client from factory
        from utils.llm_factory import LLMFactory  # Lazy-import heavy modules

        llm_client = LLMFactory.get_client(
            provider=provider, model_name=model_name, temperature=temperature
        )

        # Step 3.1) Generate answer for client
        assistant_response = generate_rag_answer(
            llm_client=llm_client,
            query=query,
            chat_session_id=chat_session_id,
            relevant_chunks=relevant_chunks,
        )

        # Step 4) Insert assistant response
        _ = insert_chat_message_supabase_record(
            supabase_client,
            table_name="messages",
            user_id=user_id,
            chat_session_id=chat_session_id,
            dialogue_role="assistant",
            message_content=assistant_response,
            query_response_status="COMPLETE",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 5) UPDATE public.messages status
        try:
            _ = (
                supabase_client.table("messages")
                .update({"query_response_status": "COMPLETE"})
                .eq("id", message_id)
                .execute()
            )
        except Exception as e:
            raise Exception(f"Error updating public.messages status: {e}")

    except Exception as e:
        logger.error(f"RAG Chat Task failed: {e}", exc_info=True)
        raise self.retry(exc=e)
    else:
        # Cleanup large in-memory objects
        try:
            del query_embedding, relevant_chunks, assistant_response
        except NameError:
            pass
        gc.collect()
        return None


def fetch_relevant_chunks(query_embedding, project_id, match_count=10):
    """
    Calls Supabase RPC to retrieve the nearest neighbor chunks using HNSW index.
    """
    try:
        response = supabase_client.rpc(
            "match_document_chunks_hnsw",
            {
                "p_project_id": project_id,
                "p_query": query_embedding,
                "p_k": match_count,
            },
        ).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching relevant chunks: {e}", exc_info=True)
        raise


def generate_rag_answer(
    llm_client,
    query: str,
    chat_session_id: str,
    relevant_chunks: list,
    max_chat_history: int = 10,
) -> str:
    """
    Build prompt, invoke LLM, return the full generated answer at completion.
    """

    # # Build conversational context
    # chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
    # formatted_history = format_chat_history(chat_history) if chat_history else ""
    # chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)

    # full_context = (
    #     f"{formatted_history}\n\nRelevant Context:\n{chunk_context}\n\n"
    #     f"User Query: {query}\nAssistant:"
    # )

    # Step 2) Load “system” messages from YAML and concatenate them
    try:
        # Adjust this filename if yours is named differently
        yaml_dict = load_yaml_prompt("chat-persona-prompt.yaml")
        sys_msgs: list = build_chat_messages_from_yaml(yaml_dict)
        # sys_msgs is a list of {"role": "system", "content": "..."}
        # We only care about content (each is already a block of text)
        system_instructions = "\n\n".join(
            msg["content"] for msg in sys_msgs if msg["role"] == "system"
        )
    except Exception as e:
        # If anything goes wrong loading YAML, fallback to empty system instructions
        logger.warning(f"Unable to load system instructions YAML: {e}")
        system_instructions = ""

    # Step 2) Build conversational context from chat history + chunks
    chat_history = fetch_chat_history(chat_session_id)[-max_chat_history:]
    formatted_history = format_chat_history(chat_history) if chat_history else ""
    chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)

    # Step 3) Stitch everything: (1) system instructions, (2) chat history, (3) relevant context, (4) user query
    # ────────────────────────────────────────────────────────────
    user_context = (
        f"{formatted_history}\n\n"
        f"Relevant Context:\n{chunk_context}\n\n"
        f"User Query: {query}\nAssistant:"
    )

    # trim in case caht context is too long...
    trimmed_user_context = trim_context_length(
        full_context=user_context,
        query=query,
        relevant_chunks=relevant_chunks,
        model_name=llm_client.model_name,
        max_tokens=127999,
    )

    # Then, prepend the un‐trimmed system instructions in front
    prompt_body = f"{system_instructions}\n\n{trimmed_user_context}"

    # 5) Finally, call the LLM client once
    try:
        # All of your LLMClient implementations expose `.chat(prompt: str) -> str`
        answer = llm_client.chat(prompt_body)
        return answer
    except Exception as e:
        logger.error(
            f"Error in LLM call (model={llm_client.model_name}): {e}", exc_info=True
        )
        log_llm_error(
            supabase_client,
            "llm_error_logs",
            "generate_rag_answer",
            str(e),
        )
        raise


def fetch_chat_history(chat_session_id):
    response = (
        supabase_client.table("messages")
        .select("*")
        .eq("chat_session_id", chat_session_id)
        .order("created_at")
        .execute()
    )
    return response.data


def format_chat_history(chat_history):
    return "".join(
        f"{m['role'].capitalize()}: {m['content']}\n" for m in chat_history
    ).strip()


def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    model_to_encoding = {
        "o4-mini": "o200k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }
    encoding_name = model_to_encoding.get(model_name, "cl100k_base")
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    history = full_context
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop()
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = (
            f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
        )
    return history
