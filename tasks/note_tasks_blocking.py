# tasks/note_tasks.py

"""
[THE OG METHOD] This file runs Celery tasks for handling RAG AI note creation tasks (outlines, summaries, compare-contrast)
Note genereation (with RAG) is done without token streaming. Returns full answer in one go.
"""

# ===== STANDARD LIBRARY IMPORTS =====
import multiprocessing as mp
import gc
import traceback
import logging
import os
from datetime import datetime, timezone
import requests
import tempfile
import uuid
import json
import re
import atexit
import signal
from dotenv import load_dotenv

# ===== DATABASE =====
import redis
from supabase import create_client, Client
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch

# ===== ASYNC & CONCURRENCY & SOCKET =====
import asyncio
import redis.asyncio as aioredis
import gevent
import gevent.socket
import socket

# ===== NETWORKING & HTTP =====
import requests
from requests.adapters import HTTPAdapter
import httpx

# ===== CELERY & TASK QUEUE =====
from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError, TimeoutError
from celery import chord, group
from celery.signals import worker_init, worker_shutdown
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
from celery.exceptions import Retry as CeleryRetry

# ===== MACHINE LEARNING & TEXT PROCESSING =====
import tiktoken
from langchain_core.load import dumpd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app
from utils.prompt_utils import load_yaml_prompt, build_prompt_template_from_yaml
from utils.supabase_utils import (
    insert_note_supabase_record,
    supabase_client,
    log_llm_error,
)
from utils.llm_clients.llm_factory import LLMFactory                    # Simple LLM Client factory
from utils.llm_clients.citation_processor import CitationProcessor      # detects citations in streaming chunks
from utils.llm_clients.performance_monitor import PerformanceMonitor    # 🆕 Performance tracking
from utils.llm_clients.stream_normalizer import StreamNormalizer        # Format streamed results from several providers
from tasks.celery_app import run_async_in_worker
from tasks.database import get_db_connection, get_redis_connection, get_global_async_db_pool, get_global_redis_pool, init_async_pools, check_db_pool_health
# from tasks.celery_app import (
#     run_async_in_worker,
#     get_global_async_db_pool,
#     get_global_redis_pool,
#     init_async_pools,
#     get_db_connection,      # ← Context manager
#     get_redis_connection    # ← Context manager
# )
# # Import health checks from the shared module:
# from tasks.pool_utils import (
#     check_async_db_pool_health,
#     check_redis_pool_health
# )

# ——— Logging & Env Load ———————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
logger.propagate = False
load_dotenv()

# ——— Configuration & Constants ————————————————————————————————————————————————————

# Queue configuration
INGEST_QUEUE = 'ingest'
PARSE_QUEUE = 'parsing'
EMBED_QUEUE = 'embedding'
FINAL_QUEUE = 'finalize'

# Performance, Retries & Batching
MAX_RETRIES = 5
RETRY_BACKOFF_MULTIPLIER = 2
DEFAULT_RETRY_DELAY = 5
RATE_LIMIT = '150/m' # Tuned for a 2-CPU / 4GB RAM instance instead of 1000/m

# OpenAI API & Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()  # ← only for embeddings
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_MAX_TOKENS_PER_BATCH = 8190 # Safety margin below the 8192 limit
EXPECTED_EMBEDDING_LEN = 1536
MAX_CONCURRENT_DOWNLOADS = 3 # A modest limit instead of 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# # ——— Global Production Instances (Initialized once per worker) —————————————————————

# OpenAI Embeddings Client
embedding_model = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,
    request_timeout=60
)

# Tokenizer
try:
    tokenizer = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ——— HELPER FUNCTIONS ————————————————————————————————————————————————————

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
    # Example: redis.publish(f"chat:{chat_session_id}", token)
    pass


class StreamToClientHandler(BaseCallbackHandler):
    """
    LangChain callback handler that emits each new LLM token via publish_token().
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Called by LangChain for every new token in streaming mode
        publish_token(self.session_id, token)

# ——— Task: Naieve RAG ———————————————————————————————————————————

@celery_app.task(bind=True, base=BaseTaskWithRetry)
def rag_note_task(
    self,
    user_id,
    note_type,
    project_id,
    note_title,
    provider: str,  # ← “openai”, “anthropic”, etc.
    model_name: str,
    temperature: float = 0.7,
    addtl_params: dict = None,  # <-- a dict containing optional overrides, like {"num_questions": 10, "document_ids":[...]}
):
    """
    RAG workflow for various note types (exam questions, case briefs, outlines, etc.).

    Steps:
    0) Mark start time in Celery state
    1) Embed a short “retrieval” query (as dim-1536 embedding)
    2) Fetch top-K relevant chunks via Supabase RPC
    3) Load the appropriate YAML prompt based on note_type
    4) Format that YAML template with {context} + any override from meta (e.g. num_questions)
    5) Call the LLM once with the fully formed prompt (Stream LLM response tokens or return all at once)
    6) Persist note output to public.notes
    """

    if addtl_params is None:
        addtl_params = {}

    try:
        # Step 0) Mark explicit start time in self.metadata
        # This manually sets result = AsyncResult(task_id) when checking on this celery task via job.id
        # result.state → "PENDING"
        # result.info → {"start_time": "..."}
        task_id = self.request.id
        logger.info(f"🎯 Starting task {task_id}... Triggered note generation for project: {project_id}")

        self.update_state(
            state="STARTED", meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # Step 1) Choose the prompt based on user selection of "note_type"
        if note_type == "outline":
            yaml_file = "case-outline-prompt.yaml"
        elif note_type == "exam_questions":
            yaml_file = "exam-questions-prompt.yaml"
        elif note_type == "case_brief":
            yaml_file = "case-brief-prompt.yaml"
        elif note_type == "compare_contrast":
            yaml_file = "compare-contrast-prompt.yaml"
        elif note_type == "flashcards":
            yaml_file = "flashcards-prompt.yaml"
        else:
            raise ValueError(f"Unknown note_type: {note_type}")
        logger.info(f"🙋‍♂️ NOTE TYPE: {note_type} generation chosen...")

        # Load YAML and extract "base_prompt" and "template"
        yaml_dict = load_yaml_prompt(yaml_file)
        base_query = yaml_dict.get("base_prompt")
        prompt_template = build_prompt_template_from_yaml(yaml_dict)
        if not base_query:
            raise KeyError(f"`base_prompt` not found in {yaml_file}")

        # Step 2) Embed the short base query using OpenAI Ada embeddings (1536 dims)
        # embedding_model = OpenAIEmbeddings(
        #     model="text-embedding-ada-002",
        #     api_key=OPENAI_API_KEY,
        # )
        logger.info(f"🤖 Generating embeddings...")
        query_embedding = embedding_model.embed_query(base_query)

        # Step 3) Fetch top-K relevant chunks via Supabase RPC && Format for llm context window
        relevant_chunks = fetch_relevant_chunks(query_embedding, project_id)
        chunk_context = "\n\n".join(chunk["content"] for chunk in relevant_chunks)

        # Step 4) Question-type specific adjustments (question number, legal course name, what ever idiosyncracies you can think of)
        if note_type == "exam_questions":
            # Decide how many questions to generate (override via addtl_params)
            num_questions = addtl_params.get("num_questions")
            if num_questions and isinstance(num_questions, int) and num_questions > 0:
                llm_input = prompt_template.format(
                    context=chunk_context, n_questions=num_questions
                )
            else:
                # fallback to YAML default of 15 if no override
                llm_input = prompt_template.format(
                    context=chunk_context,
                    n_questions=addtl_params.get("num_questions", 15),
                )

        elif note_type == "case_brief":
            # YAML template only needs {context}
            llm_input = prompt_template.format(context=chunk_context)

        elif note_type == "outline":
            # Assuming your "case-outline-prompt.yaml" has only {context}
            llm_input = prompt_template.format(context=chunk_context)

        elif note_type == "compare_contrast":
            llm_input = prompt_template.format(context=chunk_context)
        else:
            # (We already checked above, but safe‐guard here)
            raise ValueError(f"Unsupported note_type: {note_type}")

        # Step 5) Generate llm client from factory
        llm_client = LLMFactory.get_client_for(
            provider=provider, 
            model_name=model_name, 
            temperature=temperature,
            streaming=True
        )

        # Step 6) Generate answer for client
        full_answer = llm_client.chat(llm_input)

        # Step 7) Save note to the public.notes table in Supabase (realtime Supabase table)
        save_note(
            project_id=project_id,
            user_id=user_id,
            note_type=note_type,
            note_title=note_title,
            content=full_answer,  # full rag reponse
        )

        # Garbage collect cleanup and return success (proactively release large in-memory buffers)
        try:
            del relevant_chunks, chunk_context, llm_input, full_answer
        except NameError:
            pass
        gc.collect()

        # Return nothing
        return "RAG Note Task suceess"

    except Exception as e:
        logger.error(f"RAG Note Task failed: {e}", exc_info=True)
        log_llm_error(
            client=supabase_client,
            table_name="notes", # ← public.notes
            task_name="rag_note_task",
            error_message=str(e),
            project_id=project_id,
            user_id=user_id,
        )
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            raise RuntimeError(
                f"[CELERY] Step 2) RAG Note creattion failed permanently after {self.max_retries} retries: {e}"
            ) from e


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


def save_note(project_id, user_id, note_type, note_title, content):
    """
    Persists generated summary note into Supabase public.notes table.
    """
    logger.info(f"💾 Inserting {note_type} NOTE into public.notes")
    insert_note_supabase_record(
        client=supabase_client,
        table_name="notes", # ← public.notes
        user_id=user_id,
        project_id=project_id,
        note_title=note_title,
        content_markdown=content,
        note_type=note_type,
        is_generated=True,
        is_shareable=False,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


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
