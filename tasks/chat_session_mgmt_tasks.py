"""
Chat session management utils
Reset chat, delete chat, etc...

"""
import asyncio
import logging
import os
import gc
import redis
import hashlib
import pickle
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import json

from dotenv import load_dotenv

# ===== ASYNC DATABASE =====
import asyncpg
from contextlib import asynccontextmanager

# ===== LLM & LANGCHAIN =====
import tiktoken
from langchain.callbacks.base import BaseCallbackHandler

# ===== CELERY & TASK QUEUE =====
from celery import Task
from celery import chord, group
from celery.signals import worker_init, worker_shutdown
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
from celery.exceptions import Retry as CeleryRetry

# ===== MONITORING & METRICS =====
import psutil

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app
from utils.prompt_utils import load_yaml_prompt, build_chat_messages_from_yaml
from utils.supabase_utils import (
    supabase_client,
    insert_chat_message_supabase_record,
    create_new_chat_session,
    log_llm_error,
)
from utils.llm_factory_enhanced import CitationAwareLLMFactory  # 🆕 Enhanced LLM factory
from utils.citation_processor import CitationProcessor  # 🆕 Citation processing utility
from utils.performance_monitor import PerformanceMonitor  # 🆕 Performance tracking

# ——— Logging & Env Load ———————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
logger.propagate = False
load_dotenv()

# ——— Configuration & Constants ————————————————————————————————————————————————————

# Queue configuration
INGEST_QUEUE = 'query_ingest'
PARSE_QUEUE = 'message_streaming'

# Performance, Retries & Batching
MAX_RETRIES = 5
RETRY_BACKOFF_MULTIPLIER = 2
DEFAULT_RETRY_DELAY = 5
RATE_LIMIT = '150/m' # Tuned for a 2-CPU / 4GB RAM instance instead of 1000/m

# Database (asyncpg)
DB_DSN = os.getenv("POSTGRES_DSN_POOL") # e.g., Supabase -> Connection -> Get Direct URL
# DB_POOL_MIN_SIZE = 5  # <-- if i had more compute
# DB_POOL_MAX_SIZE = 20
DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 5

# Browser headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# 🆕 Streaming-specific configuration
STREAMING_CONFIG = {
    "chunk_broadcast_interval": 0.05,  # 50ms between WebSocket broadcasts
    "db_batch_update_interval": 0.1,   # 100ms between batched DB writes
    "citation_confidence_threshold": 0.7,  # Minimum confidence for citations
    "highlight_extraction_enabled": True,  # Extract metrics and key facts
    "max_concurrent_streams": 10,      # Limit concurrent streaming sessions
}

# 🆕 Cache configuration
CACHE_CONFIG = {
    "embedding_cache_ttl": 3600,       # 1 hour for query embeddings
    "chat_history_cache_ttl": 1800,    # 30 minutes for chat history
    "citation_preview_cache_ttl": 86400,  # 24 hours for link previews
    "max_cache_size": 10000,           # Maximum cached embeddings
}

# API Keys (inherited from original)
OPENAI_API_KEY = os.environ.get("OPENAI_API_PROD_KEY")

# Model temperature configuration (inherited from original)
_MODEL_TEMPERATURE_CONFIG = {
    "o4-mini": {"supports_temperature": False},
    "gpt-4o-mini": {"supports_temperature": False},
    "gpt-4.1-nano": {"supports_temperature": True, "min": 0.0, "max": 2.0},
}

# Initialize Redis sync client for pub/sub
REDIS_LABS_URL = (
    "redis://default:"
    + os.getenv("REDIS_PASSWORD")
    + "@"
    + os.getenv("REDIS_PUBLIC_ENDPOINT")
)
redis_sync = redis.Redis.from_url(REDIS_LABS_URL, decode_responses=True)

# ——— Global Variables ———————————————————————————————————————————————————————————

db_pool = None  # Async database connection pool
redis_pool = None  # Redis connection pool
performance_monitor = None  # Performance tracking instance

# ——— Data Structures ——————————————————————————————————————————

class MessageStatus(Enum):
    """Enhanced message status for streaming support"""
    pending = "pending"
    streaming = "streaming"  # 🆕 New status for active streaming
    complete = "complete"
    error = "error"
    cancelled = "cancelled"

@dataclass
class Citation:
    """Rich citation data structure"""
    id: str
    text: str
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    preview_image: Optional[str] = None
    source_type: str = "document"
    confidence: float = 1.0
    relevant_excerpt: Optional[str] = None

@dataclass
class Highlight:
    """Content highlight data structure"""
    text: str
    highlight_type: str  # 'metric', 'fact', 'quote', 'key_point'
    confidence: float = 1.0
    source_citation_id: Optional[str] = None

@dataclass
class StreamingResponse:
    """Complete streaming response data"""
    content: str
    citations: List[Citation] = field(default_factory=list)
    highlights: List[Highlight] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False

@dataclass
class PerformanceMetrics:
    """Performance tracking data"""
    embedding_time: float = 0
    retrieval_time: float = 0
    llm_time: float = 0
    db_write_time: float = 0
    total_time: float = 0
    tokens_per_second: float = 0
    cache_hit_rate: float = 0

# ——— Database Connection Management ———————————————————————————————————————

# ==== Get Asynchronous DB pool =======
async def initialize_async_database_pool():
    """🆕 Initialize async database connection pool with enhanced error handling"""
    global db_pool
    
    if db_pool is not None:
        logger.info("🔄 Closing existing database pool...")
        await db_pool.close()
        db_pool = None
    
    if not DB_DSN:
        raise ValueError("❌ POSTGRES_DSN environment variable not set")
    
    logger.info("🏊 Creating optimized asyncpg connection pool...")
    
    try:
        db_pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            command_timeout=30,
            server_settings={'application_name': 'streaming_chat_worker'},
            timeout=60,
            statement_cache_size=0  # pgBouncer compatibility
        )
        
        # Test connection
        async with db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
            
        logger.info("✅ Database pool initialized successfully")
        return db_pool
        
    except Exception as e:
        logger.error(f"❌ Database pool initialization failed: {e}")
        raise

async def initialize_redis_pool():
    """🆕 Initialize Redis connection pool for caching and pub/sub"""
    global redis_pool
    
    try:
        import redis.asyncio as aioredis
        redis_pool = aioredis.ConnectionPool.from_url(
            REDIS_LABS_URL,
            max_connections=20,
            decode_responses=True
        )
        
        # Test connection
        async with aioredis.Redis(connection_pool=redis_pool) as r:
            await r.ping()
            
        logger.info("✅ Redis pool initialized successfully")
        return redis_pool
        
    except Exception as e:
        logger.error(f"❌ Redis pool initialization failed: {e}")
        raise

@asynccontextmanager
async def get_db_connection():
    """🆕 Context manager for database connections"""
    if not db_pool:
        await initialize_database_pool()
    
    async with db_pool.acquire() as conn:
        yield conn

# Create a sync database connection pool for gevent
sync_db_pool = None

# ==== Get Synchronous DB pool =========
def initialize_sync_database_pool():
    """Get or create a synchronous database connection pool for gevent"""
    global sync_db_pool
    if sync_db_pool is None:
        import urllib.parse
        parsed = urllib.parse.urlparse(DB_DSN)
        sync_db_pool = ThreadedConnectionPool(
            minconn=DB_POOL_MIN_SIZE,
            maxconn=DB_POOL_MAX_SIZE,
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:],  # Remove leading slash
            user=parsed.username,
            password=parsed.password,
            application_name='celery_worker_gevent'
        )
    return sync_db_pool

# ——— Retry & Circuit Break Management ———————————————————————————————————————

class BaseTaskWithRetry(Task):
    """
    Base Celery Task class enabling automatic retries on exceptions.
    """

    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": 5}
    retry_jitter = True


# ——— Chat Management Tasks ——————————————————————————————————————————————————————

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
            table_name="messages", # ← public.messages
            user_id=user_id,
            chat_session_id=chat_session_id,
            dialogue_role="user",
            message_content=query,
            query_response_status="PENDING",
            format="markdown",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 2) Return inserted record ID for downstream tasks
        inserted_id = response.data[0]["id"]
        return inserted_id

    except Exception as e:
        logger.error(f"RAG Chat Task failed: {e}", exc_info=True)
        log_llm_error(
            client=supabase_client,
            table_name="messages", # ← public.messages
            task_name="rag_chat_task",
            error_message=str(e),
            project_id=project_id,
            chat_session_id=chat_session_id,
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
            user_id=user_id,
            chat_session_id=chat_session_id,
            relevant_chunks=relevant_chunks,
        )

        # Step 4) Insert assistant response
        _ = insert_chat_message_supabase_record(
            supabase_client,
            table_name="messages", # ← public.messages
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
    user_id,
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
        # log_llm_error(
        #     supabase_client,
        #     "llm_error_logs",
        #     "generate_rag_answer",
        #     str(e),
        # )
        log_llm_error(
            client=supabase_client,
            table_name="messages", # ← public.messages
            task_name="generate_rag_answer",
            error_message=str(e),
            chat_session_id=chat_session_id,
            user_id=user_id,
        )
        raise


def fetch_chat_history(chat_session_id):
    response = (
        supabase_client.table("messages") # ← public.messages
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
