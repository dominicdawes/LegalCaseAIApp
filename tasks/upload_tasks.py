# tasks/upload_tasks.py

"""
Edge cases for the Custom + LightRag implementation

1) Reused Docs (~ln 1338): vector embeddings are copied from an existing processed document, but what about copying over the knolwdge graph from DGraph?
- maybe its an assocuated kg_id, im not sure...
- maybe still genrate a KG because the old kg is probably merged with other docs so its impossible/expansive to parse it out

"""

# ===== STANDARD LIBRARY IMPORTS =====
import os
import sys
import gc
import uuid
import tempfile
import urllib.parse
import re
import atexit
import signal
import logging
import threading
import io
import json
import hashlib
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta

# ===== ENVIRONMENT & CONFIGURATION =====
from dotenv import load_dotenv

# ===== ASYNC & CONCURRENCY & SOCKET =====
import asyncio
# import gevent
# import gevent.socket
import socket
from concurrent.futures import ThreadPoolExecutor

# ===== NETWORKING & HTTP =====
import requests
from requests.adapters import HTTPAdapter
import httpx

# ===== RETRY & RESILIENCE (with aliases to avoid conflicts) =====
from urllib3.util.retry import Retry as UrllibRetry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type 
from pybreaker import CircuitBreaker
# Note: Celery's Retry is imported separately below to avoid naming conflicts

# ===== DATABASE =====
import asyncpg
import psycopg2
from psycopg2.extras import Json
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch, execute_values

# ===== CELERY & TASK QUEUE =====
from celery import chord, group, chain
from celery.signals import worker_init, worker_shutdown
from celery.result import AsyncResult, EagerResult
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
from celery.exceptions import Retry as CeleryRetry

# ===== MACHINE LEARNING & TEXT PROCESSING =====
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ===== MONITORING & METRICS =====
import psutil

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app
from tasks.note_tasks import rag_note_task
# from utils.lightrag.lightrag_utils import lightrag_client, lightrag_integration  # LightRAG disabled
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url, cloudfront_domain
from utils.progress_events import (
    publish_ingest_progress,
    publish_ingest_progress_sync,
    IngestStage,
)
from utils.supabase_utils import supabase_client
from utils.document_loaders.base import BaseDocumentLoader
from utils.document_loaders.loader_factory import get_loader_for, analyze_document_before_processing, get_high_performance_loader
from utils.document_loaders.performance import create_optimized_processor, BatchTokenCounter
from utils.metrics import MetricsCollector, Timer
from utils.connection_pool import ConnectionPoolManager
from utils.memory_manager import MemoryManager # Kept for health checks

from tasks.celery_app import run_async_in_worker
from tasks.database import get_db_connection, get_redis_connection, get_global_sync_db_pool, get_global_async_db_pool, get_global_redis_pool, init_async_pools, check_db_pool_health, check_redis_pool_health

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
#     check_redis_pool_health
# )

# ——— Logging & Env Load ———————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
logger.propagate = False
load_dotenv()

# ——— Configuration & Constants ————————————————————————————————————————————————————
USE_LIGHTRAG_INTEGRATION = False

# ── Phase 1 hierarchical ingest feature flags ────────────────────────────────
# Set USE_HIERARCHICAL_INGEST=true to activate Docling + Voyage + contextual-blurb pipeline.
# Set USE_VOYAGE_EMBEDDINGS=true to switch the embedding model to voyage-law-2 (1024-dim).
# Both can be toggled independently; turning off either falls back to the legacy path.
USE_HIERARCHICAL_INGEST = os.getenv("USE_HIERARCHICAL_INGEST", "false").lower() == "true"
USE_VOYAGE_EMBEDDINGS   = os.getenv("USE_VOYAGE_EMBEDDINGS",   "false").lower() == "true"

# ── Ingest LLM — hot-swappable via env vars ──────────────────────────────────
# Controls the provider/model used for chunk blurb generation.
# Must match the same vars read by hierarchical_ingest_tasks.py.
INGEST_LLM_PROVIDER = os.getenv("INGEST_LLM_PROVIDER", "gemini").strip()
INGEST_LLM_MODEL    = os.getenv("INGEST_LLM_MODEL",    "gemini-3.1-flash-lite").strip()

# ── Blurb generation batching ─────────────────────────────────────────────────
# Number of prose chunks to pack into a single LLM call for blurb generation.
# Table chunks are always processed individually (they need longer summaries).
# Increasing this value reduces API roundtrips on large documents (100+ pages).
# Override via BLURB_BATCH_SIZE env var.
BLURB_BATCH_SIZE = int(os.getenv("BLURB_BATCH_SIZE", "20"))

# Max characters of document synopsis (title + TOC + opening) shared as the
# system prompt across every blurb call for one document.  Kept small and
# byte-identical per document so provider-side prompt caching can hit — the
# per-chunk context the model actually needs comes from `section_path`.
BLURB_CONTEXT_CHARS = int(os.getenv("BLURB_CONTEXT_CHARS", "2000"))

# ── Ingest byte cache ─────────────────────────────────────────────────────────
# The batch coordinator downloads every file to hash it.  Rather than making the
# parser download the same bytes a second time from CloudFront, the coordinator
# parks them here keyed by content_hash and the parser picks them up.  Falls back
# to a CDN download on a miss (different process, restart, eviction).
INGEST_CACHE_DIR = os.getenv("INGEST_CACHE_DIR", os.path.join(tempfile.gettempdir(), "ingest_cache"))
# Entries older than this are swept at worker start — they can only exist if a
# worker was hard-killed between download and parse.
INGEST_CACHE_MAX_AGE_S = int(os.getenv("INGEST_CACHE_MAX_AGE_S", "3600"))

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

# ── Poison-message guard ──────────────────────────────────────────────────────
# Celery runs with task_acks_late=True, so a message is only acked once its task
# finishes. If a document is heavy enough to OOM-kill the worker, the broker
# redelivers it on restart, it OOMs again, and the worker enters a permanent
# crash loop — replaying the same documents for days. (The `celery purge` in the
# Dockerfile does not help: unacked messages are requeued by the broker *after*
# the purge, when it notices the dead connection.)
#
# Each attempt bumps a Redis counter; past the limit the document is failed
# permanently so the message finally gets acked and the loop breaks.
MAX_INGEST_ATTEMPTS = int(os.getenv("MAX_INGEST_ATTEMPTS", "3"))
_INGEST_ATTEMPT_TTL = 24 * 3600  # seconds

# OpenAI configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # XTREMELY OUTDATAE
OPENAI_MAX_TOKENS_PER_BATCH = 8190 # Safety margin below the 8192 limit
# Dimension changes with embedding model: voyage-law-2 → 1024, ada-002 → 1536
EXPECTED_EMBEDDING_LEN = 1024 if USE_VOYAGE_EMBEDDINGS else 1536
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

# ——— Data Structures (Unchanged from v5) ——————————————————————————————————————————

class ProcessingStatus(Enum):
    INITIALIZING = "INITIALIZING"
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED_DOWNLOAD = "FAILED_DOWNLOAD"
    FAILED_PARSING = "FAILED_PARSING"
    FAILED_EMBEDDING = "FAILED_EMBEDDING"
    FAILED_FINALIZATION = "FAILED_FINALIZATION"

class BatchProgressStatus(Enum):
    """Batch-level progress tracking for UI observability"""
    BATCH_INITIALIZING = "BATCH_INITIALIZING"
    BATCH_ANALYZING = "BATCH_ANALYZING"        # Document classification phase
    BATCH_DOWNLOADING = "BATCH_DOWNLOADING"    # Concurrent downloads
    BATCH_PROCESSING = "BATCH_PROCESSING"
    BATCH_PARSING = "BATCH_PARSING"           # Document parsing phase  [REUSE & COPYING are not logged]
    BATCH_EMBEDDING = "BATCH_EMBEDDING"       # Embedding generation phase
    BATCH_FINALIZING = "BATCH_FINALIZING"     # Coordination & cleanup
    BATCH_COMPLETE = "BATCH_COMPLETE"         # All processing done
    BATCH_PARTIAL = "BATCH_PARTIAL"           # Some docs failed
    BATCH_FAILED = "BATCH_FAILED"             # Complete failure

@dataclass
class DocumentMetrics:
    """Enhanced telemetry for document processing"""
    doc_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    download_time_ms: Optional[int] = None
    parse_time_ms: Optional[int] = None
    chunk_time_ms: Optional[int] = None
    embed_time_ms: Optional[int] = None
    total_time_ms: Optional[int] = None
    
    file_size_bytes: int = 0
    total_chunks: int = 0
    total_batches: int = 0
    processed_chunks: int = 0
    processed_batches: int = 0
    failed_batches: int = 0
    
    peak_memory_mb: float = 0.0
    avg_chunk_size: float = 0.0
    tokens_processed: int = 0
    embedding_calls: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'start_time': self.start_time.isoformat(),
            'download_time_ms': self.download_time_ms,
            'parse_time_ms': self.parse_time_ms,
            'chunk_time_ms': self.chunk_time_ms,
            'embed_time_ms': self.embed_time_ms,
            'total_time_ms': self.total_time_ms,
            'file_size_bytes': self.file_size_bytes,
            'total_chunks': self.total_chunks,
            'total_batches': self.total_batches,
            'processed_chunks': self.processed_chunks,
            'processed_batches': self.processed_batches,
            'failed_batches': self.failed_batches,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_chunk_size': self.avg_chunk_size,
            'tokens_processed': self.tokens_processed,
            'embedding_calls': self.embedding_calls,
            'retry_count': self.retry_count
        }


# ——— 6. Retry Strategies ————————————————————————————————————————————————

# HTTP Retry Strategy (for requests library)
HTTP_RETRY_STRATEGY = UrllibRetry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]  # Be explicit about retry methods
)

# Tenacity Retry Decorator
embedding_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)

# ——— Global Production Instances (Initialized once per worker) —————————————————————

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

# Global Metrics Collector
metrics_collector = MetricsCollector()

# Global Metrics Collector
logger.info("📊 Initializing metrics collector...")
metrics_collector = MetricsCollector()
logger.info("✅ Metrics collector initialized")

# ——— Dedicated Executors ——————————————————————————————————————————————————————————
#
# Every `run_in_executor(None, ...)` in this module used to share the event
# loop's DEFAULT ThreadPoolExecutor, which is sized min(32, cpu_count + 4) — only
# 6 threads on a 2-vCPU instance.  That silently capped the blurb semaphore and
# let a CPU-bound Docling parse starve the LLM calls.  Two purpose-built pools:
#
#   _LLM_EXECUTOR — blurb + embedding API calls.  Pure network wait, so heavy
#                   oversubscription relative to CPU count is correct.
#   _CPU_EXECUTOR — Docling parsing.  Deliberately small so a parse can never
#                   monopolise the process or starve I/O work.
_LLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("INGEST_LLM_THREADS", "24")),
    thread_name_prefix="ingest-llm",
)
_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("INGEST_PARSE_THREADS", str(max(2, (os.cpu_count() or 2))))),
    thread_name_prefix="ingest-parse",
)

@worker_shutdown.connect
def _shutdown_ingest_executors(sender=None, **kwargs):
    """Drain the ingest executors when the worker stops."""
    _LLM_EXECUTOR.shutdown(wait=False)
    _CPU_EXECUTOR.shutdown(wait=False)
    logger.info("🛑 Ingest executors shut down")

# ——— Worker Startup Hooks ————————————————————————————————————————————————————————

@worker_init.connect
def _warm_docling_on_startup(sender=None, **kwargs):
    """
    Pre-load Docling layout + OCR models when the Celery worker starts.

    Without pre-warming, the first upload cold-loads ~40 MB of RapidOCR / Tesseract
    models plus the layout transformer, adding ~30s to the first document's parse time.

    NOTE: constructing a DocumentConverter does NOT load models — docling builds
    its pipeline lazily on the first convert(). _ensure_ready() alone therefore
    left the first real upload paying the whole cost anyway. warm_up() runs one
    conversion over a tiny in-memory PDF, which is what actually resides the
    models.

    Non-fatal: if Docling isn't installed or warm-up fails, the worker still starts.
    """
    if USE_HIERARCHICAL_INGEST:
        try:
            # Warm the SINGLETON — warming a throwaway instance loaded models
            # into an object that was immediately garbage collected, so the
            # first real upload still paid the full cold-load.
            from utils.document_loaders.docling_loader import get_docling_loader
            logger.info("🔥 Pre-warming Docling models at worker startup...")
            started = time.perf_counter()
            if get_docling_loader().warm_up():
                logger.info(
                    f"✅ Docling models warm in {time.perf_counter() - started:.1f}s "
                    f"— first upload will not cold-load"
                )
        except Exception as exc:
            logger.warning(f"⚠️ Docling pre-warm failed (non-fatal): {exc}")


@worker_init.connect
def _log_memory_ceiling_on_startup(sender=None, **kwargs):
    """
    Record the container's memory limit at boot so every later percentage in the
    log is interpretable. Reads the cgroup limit, not host RAM — on Render the
    host figure is far larger than the instance cap and makes usage look safe
    right up to the OOM kill.
    """
    try:
        from utils.ingest_telemetry import log_startup_memory
        log_startup_memory()
    except Exception as exc:
        logger.debug(f"Memory ceiling log failed: {exc}")


@worker_init.connect
def _sweep_ingest_cache_on_startup(sender=None, **kwargs):
    """
    Drop stale ingest-cache files left behind by a previous worker.

    _process_document_async_workflow purges its own entry in a finally block, so
    leftovers only appear if the worker was hard-killed mid-document. Without
    this sweep those PDFs would accumulate on a small instance disk.
    """
    try:
        if not os.path.isdir(INGEST_CACHE_DIR):
            return
        cutoff = time.time() - INGEST_CACHE_MAX_AGE_S
        removed = 0
        for entry in os.listdir(INGEST_CACHE_DIR):
            path = os.path.join(INGEST_CACHE_DIR, entry)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info(f"🧹 Swept {removed} stale ingest-cache file(s)")
    except Exception as exc:
        logger.warning(f"⚠️ Ingest cache sweep failed (non-fatal): {exc}")


# ——— Helpers & Utilities ——————————————————————————————————————————————————————————

class PhaseTimer:
    """
    Accumulates per-phase wall-clock timings for one document so the ingest
    pipeline can be optimised against measurements instead of guesses.

    Usage:
        timings = PhaseTimer(doc_id)
        with timings.phase("parse"):
            ...
        timings.log()   # → ⏱️ [DOC-abc12345] parse=8420ms embed=1130ms total=9550ms
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.phases: Dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.phases[name] = self.phases.get(name, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def to_dict(self) -> Dict[str, int]:
        out = {k: int(v) for k, v in self.phases.items()}
        out["total"] = int(self.total_ms)
        return out

    def log(self) -> None:
        """
        The single most useful line per document: where the time went, and what
        the worker looked like while it went there.
        """
        parts = " ".join(f"{k}={int(v)}ms" for k, v in self.phases.items())

        state = ""
        try:
            from utils.ingest_telemetry import telemetry
            state = f" | {telemetry.format_state()}"
        except Exception:
            pass

        slowest = ""
        if self.phases:
            name, ms = max(self.phases.items(), key=lambda kv: kv[1])
            share = 100.0 * ms / self.total_ms if self.total_ms else 0
            slowest = f" | slowest={name} ({share:.0f}%)"

        logger.info(
            f"⏱️ [DOC-{self.doc_id[:8]}] {parts} total={int(self.total_ms)}ms"
            f"{slowest}{state}"
        )


# ——— Poison-message Guard ————————————————————————————————————————————————————————

async def _record_ingest_attempt(doc_id: str) -> int:
    """
    Count this delivery of `doc_id` and return the attempt number (1 = first).

    Fails open at 1: if Redis is unreachable we would rather process a document
    twice than refuse to process it at all.
    """
    try:
        key = f"ingest_attempts:{doc_id}"
        async with get_redis_connection() as r:
            count = await r.incr(key)
            if count == 1:
                await r.expire(key, _INGEST_ATTEMPT_TTL)
            return int(count)
    except Exception as e:
        logger.warning(f"⚠️ Attempt counter unavailable for {doc_id[:8]}: {e}")
        return 1


async def _clear_ingest_attempts(doc_id: str) -> None:
    """Reset the counter once the document has been processed successfully."""
    try:
        async with get_redis_connection() as r:
            await r.delete(f"ingest_attempts:{doc_id}")
    except Exception as e:
        logger.debug(f"Attempt counter clear failed for {doc_id[:8]}: {e}")


# ——— Ingest Byte Cache ————————————————————————————————————————————————————————————

def _is_own_cdn_url(url: str) -> bool:
    """
    True when the URL already points at our own CloudFront distribution.

    WeWeb uploads to S3/CDN before it calls us, so for those URLs the file is
    already exactly where the pipeline wants it — downloading it only to push
    the identical bytes back to a fresh S3 key is a wasted round trip.
    """
    if not url or not cloudfront_domain:
        return False
    try:
        return urllib.parse.urlparse(url).netloc.lower() == cloudfront_domain.lower()
    except Exception:
        return False


def _ingest_cache_path(content_hash: str) -> str:
    return os.path.join(INGEST_CACHE_DIR, f"{content_hash}.bin")


def _write_ingest_cache(content_hash: str, stream: io.BytesIO) -> None:
    """Park already-downloaded bytes for the parser. Best-effort."""
    try:
        os.makedirs(INGEST_CACHE_DIR, exist_ok=True)
        path = _ingest_cache_path(content_hash)
        stream.seek(0)
        # Write to a temp name then rename, so a reader can never observe a
        # half-written file.
        tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
        with open(tmp_path, "wb") as fh:
            fh.write(stream.read())
        os.replace(tmp_path, path)
        stream.seek(0)
        logger.info(f"💾 Cached {content_hash[:8]} for parser handoff")
    except Exception as e:
        logger.warning(f"⚠️ Ingest cache write failed for {content_hash[:8]}: {e}")


def _read_ingest_cache(content_hash: Optional[str]) -> Optional[io.BytesIO]:
    """Retrieve parked bytes. Returns None on any miss — caller re-downloads."""
    if not content_hash:
        return None
    try:
        path = _ingest_cache_path(content_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            buf = io.BytesIO(fh.read())
        buf.seek(0)
        return buf
    except Exception as e:
        logger.warning(f"⚠️ Ingest cache read failed for {content_hash[:8]}: {e}")
        return None


def _purge_ingest_cache(content_hash: Optional[str]) -> None:
    """Drop parked bytes once the document has been parsed. Best-effort."""
    if not content_hash:
        return
    try:
        path = _ingest_cache_path(content_hash)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.debug(f"Ingest cache purge failed for {content_hash[:8]}: {e}")


def _calculate_stream_hash(stream: io.BytesIO) -> str:
    """Calculate SHA-256 hash from an in-memory stream without consuming it."""
    sha256_hash = hashlib.sha256()
    stream.seek(0)
    # Read in chunks to handle large streams efficiently
    while chunk := stream.read(4096):
        sha256_hash.update(chunk)
    stream.seek(0) # Reset stream position after reading
    return sha256_hash.hexdigest()

def _update_document_status_sync(doc_id: str, status: ProcessingStatus, error_message: str = None, stats: Optional[Dict[str, Any]] = None):
    """
    [PER DOCUMENT] Synchronous version of document status update helper.
    Performs Supabase public.document_sources updates for processed documents

    Args:
    - doc_id: document uuid
    - status: Enum defined above PENDING, COMPLETE, etc...
    - stats (Dict): a dict that contains 'total chunks', 'batch_size', etc
    """
    logger.info(f"📋 Doc {doc_id[:8]}... → {status.value}")
    pool = get_global_sync_db_pool()
    retries = 3

    for attempt in range(retries):
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                if status == ProcessingStatus.COMPLETE and stats:
                    cur.execute(
                        """
                        UPDATE document_sources
                        SET 
                            vector_embed_status = %s, 
                            error_message = %s, 
                            updated_at = NOW(),
                            total_chunks = %s,
                            total_tokens = %s
                        WHERE id = %s
                        """,
                        (
                            status.value, 
                            error_message, 
                            stats.get('chunks_created', 0), 
                            stats.get('total_tokens', 0), 
                            doc_id
                        )
                    )
                else:
                    cur.execute(
                        """
                        UPDATE document_sources
                        SET vector_embed_status = %s, error_message = %s, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (status.value, error_message, doc_id)
                    )
            conn.commit()
            return # Success

        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.warning(f"⚠️ DB Connection failed in doc update (attempt {attempt+1}/{retries}): {e}")
            if conn:
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = None
            
            if attempt == retries - 1:
                raise e
            time.sleep(0.5)
            
        finally:
            if conn:
                pool.putconn(conn)

def _update_batch_progress_sync(
    batch_id: str,
    project_id: str,
    status: BatchProgressStatus,
    doc_ids: Optional[List[str]] = None,
):
    """
    [PER BATCH] Update batch_progress for all documents in a batch
    Includes retry logic for stale connections.

    Rows are normally matched on processing_metadata->>'batch_id', which is only
    stamped once the per-document INSERT in _process_document_async_workflow has
    run.  The early batch phases (ANALYZING, PROCESSING) fire *before* that, so
    those calls used to update zero rows and the UI stayed blind through the
    whole download/dedupe phase.  Callers that already know the document ids —
    the speculative flow pre-creates its row — pass `doc_ids` to match directly.
    """
    pool = get_global_sync_db_pool()
    retries = 3

    if doc_ids:
        sql = """
            UPDATE document_sources
            SET batch_progress = %s, updated_at = NOW()
            WHERE project_id = %s
            AND (id = ANY(%s::uuid[]) OR processing_metadata->>'batch_id' = %s)
        """
        params = (status.value, project_id, list(doc_ids), batch_id)
    else:
        sql = """
            UPDATE document_sources
            SET batch_progress = %s, updated_at = NOW()
            WHERE project_id = %s
            AND processing_metadata->>'batch_id' = %s
        """
        params = (status.value, project_id, batch_id)

    for attempt in range(retries):
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows_updated = cur.rowcount
            conn.commit()
            logger.info(f"📊 [BATCH-{batch_id[:8]}] Progress → {status.value} ({rows_updated} docs)")
            return # Success, exit loop

        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.warning(f"⚠️ DB Connection failed in batch update (attempt {attempt+1}/{retries}): {e}")
            if conn:
                # CRITICAL: Return the bad connection to the pool with close=True
                # This forces the pool to discard it and create a new one next time
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = None
            
            if attempt == retries - 1:
                logger.error(f"❌ Failed to update batch progress after {retries} attempts")
                raise e
            
            # Small backoff before retry
            time.sleep(0.5)

        finally:
            if conn:
                pool.putconn(conn)

def create_http_session() -> requests.Session:
    """Helper to create a requests session with retry strategy"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=HTTP_RETRY_STRATEGY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

async def get_embedding_reuse_stats(project_id: str) -> Dict[str, Any]:
    """
    Get statistics about embedding reuse ♻️ for a project
    """
    # Use global async pool instead of local pool
    pool = get_global_async_db_pool()
    if not pool:
        await init_async_pools()
        pool = get_global_async_db_pool()

    async with pool.acquire() as conn:
        stats = await conn.fetchrow(
            '''
            SELECT 
            COUNT(*) as total_documents,
            COUNT(CASE WHEN vector_embed_status = 'COMPLETE' THEN 1 END) as processed_docs,
            SUM(total_chunks) as total_chunks,
            SUM(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN total_chunks ELSE 0 END) as recent_chunks
            FROM document_sources 
            WHERE project_id = $1''',
            uuid.UUID(project_id)
        )
        
        return {
            'total_documents': stats['total_documents'] or 0,
            'processed_documents': stats['processed_docs'] or 0,
            'total_chunks': stats['total_chunks'] or 0,
            'recent_chunks': stats['recent_chunks'] or 0
        }

def get_file_extension_from_url(url: str) -> str:
    """
    Comprehensive file extension detection for URLs
    Handles academic repositories, government sites, publishers, and more
    """
    url_lower = url.lower()
    
    # ===== ACADEMIC & RESEARCH REPOSITORIES =====
    
    # ArXiv (all variants)
    if 'arxiv.org' in url_lower and '/pdf/' in url_lower:
        return '.pdf'
    
    # ResearchGate
    if 'researchgate.net' in url_lower:
        if 'publication' in url_lower or 'profile' in url_lower:
            return '.pdf'
    
    # Academia.edu
    if 'academia.edu' in url_lower:
        return '.pdf'
    
    # Semantic Scholar
    if 'semanticscholar.org' in url_lower or 'pdfs.semanticscholar.org' in url_lower:
        return '.pdf'
    
    # SSRN
    if 'ssrn.com' in url_lower or 'papers.ssrn.com' in url_lower:
        if 'abstract' in url_lower or 'papers.cfm' in url_lower:
            return '.pdf'
    
    # ===== GOVERNMENT & INSTITUTIONAL =====
    
    # US Congress & Government
    if any(domain in url_lower for domain in [
        'congress.gov', 'govinfo.gov', 'cbo.gov', 'gao.gov', 
        'federalregister.gov', 'supremecourt.gov', 'uscourts.gov'
    ]):
        return '.pdf'
    
    # ===== UNIVERSITY REPOSITORIES =====
    
    # Common university repository patterns
    university_patterns = [
        'dspace', 'repository', 'dash', 'ecommons', 'scholarworks', 
        'deepblue', 'handle', 'bitstream', 'viewcontent.cgi'
    ]
    if any(pattern in url_lower for pattern in university_patterns):
        return '.pdf'
    
    # ===== PUBLISHERS & JOURNALS =====
    
    # Major academic publishers
    publisher_domains = [
        'springer.com', 'wiley.com', 'nature.com', 'sciencemag.org',
        'plos.org', 'mdpi.com', 'frontiersin.org', 'elsevier.com',
        'tandfonline.com', 'sagepub.com', 'ieee.org'
    ]
    if any(domain in url_lower for domain in publisher_domains):
        return '.pdf'
    
    # ===== MEDICAL & BIOMEDICAL =====
    
    # NCBI, PubMed, PMC
    if any(domain in url_lower for domain in ['ncbi.nlm.nih.gov', 'pubmed.ncbi.nlm.nih.gov']):
        return '.pdf'
    
    # ===== INTERNATIONAL REPOSITORIES =====
    
    # European and international
    international_patterns = [
        'hal.archives-ouvertes.fr', 'orbit.dtu.dk', 'pure.', 'research-repository',
        'eprints.', 'ir.library.', 'digitalcommons.'
    ]
    if any(pattern in url_lower for pattern in international_patterns):
        return '.pdf'
    
    # ===== CLOUD STORAGE & CDNs =====
    
    # Google Drive
    if 'drive.google.com' in url_lower:
        # Try to detect from URL parameters or context
        if 'export=download' in url_lower:
            return '.pdf'  # Default assumption
        return '.pdf'  # Most shared academic docs are PDFs
    
    # Dropbox
    if 'dropbox.com' in url_lower:
        # Extract filename from URL
        match = re.search(r'/([^/]+\.[a-zA-Z]{2,5})', url)
        if match:
            filename = match.group(1)
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.pdf', '.docx', '.doc', '.epub', '.txt']:
                return ext
        return '.pdf'  # Default
    
    # OneDrive
    if 'onedrive.live.com' in url_lower or '1drv.ms' in url_lower:
        return '.pdf'  # Default assumption
    
    # ===== DIRECT FILE EXTENSIONS =====
    
    # Check for direct file extensions
    direct_extensions = ['.pdf', '.docx', '.doc', '.epub', '.txt', '.rtf', '.odt']
    for ext in direct_extensions:
        if url_lower.endswith(ext):
            return ext
        # Also check with query parameters
        if f'{ext}?' in url_lower or f'{ext}#' in url_lower:
            return ext
    
    # ===== CONTENT-TYPE GUESSING FROM URL PATTERNS =====
    
    # PDF indicators in URL
    pdf_indicators = [
        '/pdf/', '.pdf', 'format=pdf', 'type=pdf', 'download=pdf',
        'export=pdf', 'view=pdf', 'filetype/pdf', 'document.pdf'
    ]
    if any(indicator in url_lower for indicator in pdf_indicators):
        return '.pdf'
    
    # Word document indicators
    doc_indicators = ['/doc/', '.docx', '.doc', 'format=docx', 'type=docx']
    if any(indicator in url_lower for indicator in doc_indicators):
        return '.docx'
    
    # ===== FALLBACK: PARSE URL PATH =====
    
    try:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        
        if path:
            # Get the last segment that looks like a filename
            segments = [seg for seg in path.split('/') if seg]
            for segment in reversed(segments):
                if '.' in segment:
                    # Extract potential extension
                    potential_ext = os.path.splitext(segment)[1].lower()
                    if potential_ext in ['.pdf', '.docx', '.doc', '.epub', '.txt', '.rtf']:
                        return potential_ext
                    
        # Check query parameters for filename
        query_params = urllib.parse.parse_qs(parsed_url.query)
        for param_name, param_values in query_params.items():
            for value in param_values:
                if '.' in value:
                    potential_ext = os.path.splitext(value)[1].lower()
                    if potential_ext in ['.pdf', '.docx', '.doc', '.epub', '.txt', '.rtf']:
                        return potential_ext
                        
    except Exception as e:
        logger.debug(f"URL parsing failed for {url}: {e}")
    
    # ===== ULTIMATE FALLBACK =====
    
    # If it's an academic/research domain, assume PDF
    academic_tlds = ['.edu', '.gov', '.org']
    research_keywords = [
        'research', 'academic', 'scholar', 'journal', 'paper', 'publication',
        'article', 'conference', 'proceedings', 'thesis', 'dissertation'
    ]
    
    if (any(tld in url_lower for tld in academic_tlds) or 
        any(keyword in url_lower for keyword in research_keywords)):
        return '.pdf'
    
    # Final fallback - assume PDF for unknown academic content
    return '.pdf'

def parse_clean_filename_from_url(url: str, extension: str) -> str:
    """
    Generate a clean filename from URL
    """
    try:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        
        # Try to extract meaningful filename
        if path:
            segments = [seg for seg in path.split('/') if seg and not seg.isdigit()]
            
            # Look for segments that look like filenames
            for segment in reversed(segments):
                if len(segment) > 3 and ('.' in segment or '_' in segment or '-' in segment):
                    # Clean the segment
                    clean_name = re.sub(r'[^\w\-_.]', '_', segment)
                    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
                    if len(clean_name) > 3:
                        # Remove existing extension and add detected one
                        base_name = os.path.splitext(clean_name)[0]
                        return f"{base_name}{extension}"
            
            # Use the last meaningful segment
            if segments:
                last_segment = segments[-1]
                clean_name = re.sub(r'[^\w\-_]', '_', last_segment)
                clean_name = re.sub(r'_+', '_', clean_name).strip('_')
                if len(clean_name) > 3:
                    return f"{clean_name}{extension}"
        
        # Extract from domain
        domain = parsed_url.netloc.split('.')[0]
        if domain and len(domain) > 2:
            return f"{domain}_document_{uuid.uuid4().hex[:8]}{extension}"
            
    except Exception:
        pass
    
    # Final fallback
    return f"document_{uuid.uuid4().hex[:8]}{extension}"

# ——— Utils: Document Analysis/Classification and Embedding Utils ——————————————————————————————————————————

async def _analyze_download_and_store_document_for_workflow(
    client: httpx.AsyncClient, 
    url: str, 
    project_id: str, 
    user_id: str
) -> Dict[str, Any]:
    """
    [ASYNC]
    Description:
    - Downloads (in-memory stream), hashes, Persist to AWS and Amazon CloudFront CDN and analyzes a document to determine processing type 
    using a single, robust SQL query.
    - DUPLICATE: Same hash, same project.
    - REUSED: Same hash, different project, already processed.
    - NEW: No match found.

    Definitions:
    1. Download (in-memory stream)
    2. Persist to AWS and Amazon CloudFront CDN
    3. Hash to document and analyze document to determine processing type:
    - NEW: Requires full processing pipeline (parse → embed → store)
    - REUSED: Existing processed content can be copied (smart reuse)
    
    Returns document metadata with processing classification
    """
    try:
        # ——— 1. Download & Prepare Document ——————————————————
        doc_data = await _download_and_prep_doc(client, url, project_id, user_id)
        if not doc_data:
            raise Exception(f"Failed to download document from {url}")
        
        # ——— 2. Check Processing Type with a Single, Combined Query ——————————
        pool = get_global_sync_db_pool()
        conn = pool.getconn()
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                content_hash = doc_data['content_hash']
                
                # This single query finds any match and determines its type
                cur.execute(
                    """
                    SELECT
                        id,
                        -- Use a CASE statement to determine the processing type in SQL
                        CASE
                            WHEN project_id = %(current_project_id)s THEN 'DUPLICATE'
                            ELSE 'REUSED'
                        END as processing_type,
                        total_chunks
                    FROM document_sources
                    WHERE content_hash = %(content_hash)s
                      AND (
                           project_id = %(current_project_id)s OR 
                           (vector_embed_status = 'COMPLETE' AND total_chunks > 0)
                      )
                    ORDER BY
                        -- Prioritize the DUPLICATE case to ensure it's found first
                        CASE WHEN project_id = %(current_project_id)s THEN 0 ELSE 1 END
                    LIMIT 1;
                    """,
                    {
                        'content_hash': content_hash,
                        'current_project_id': project_id
                    }
                )
                existing_doc = cur.fetchone()
                
                if existing_doc:
                    processing_type = existing_doc['processing_type']
                    
                    if processing_type == 'DUPLICATE':
                        return {
                            'processing_type': 'DUPLICATE',
                            'existing_doc_id': str(existing_doc['id']),
                            'doc_data': doc_data,
                            'project_id': project_id
                        }
                    else: # REUSED socument found
                        return {
                            'processing_type': 'REUSED',
                            'existing_doc_id': str(existing_doc['id']),
                            'doc_data': doc_data,
                            'project_id': project_id,
                            'chunks_available': existing_doc['total_chunks']
                        }
                else:
                    # No existing document found, it's NEW
                    return {
                        'processing_type': 'NEW',
                        'doc_data': doc_data,
                        'project_id': project_id
                    }
                    
        finally:
            pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Document analysis failed for {url}: {e}")
        raise

def copy_embeddings_for_project_sync(existing_source_id: str, new_source_id: str, project_id: str, user_id: str) -> Dict[str, Any]:
    """
    [SYNC] Copy embeddings from an existing processed document to a new project.
    Handles both Ada-legacy and Voyage-law-2 embeddings, all hierarchical columns
    (chunk_summary, section_path, chunk_type), and remaps parent_chunk_id UUIDs so
    the parent-expansion retrieval path works correctly in the new project.
    Also copies document_sections rows for section-level hierarchical retrieval.

    Args:
        existing_source_id: Source ID of the already-processed document
        new_source_id: Source ID of the new document entry
        project_id: Target project ID
        user_id: User who uploaded the document

    Returns:
        Dict with copy statistics: copied_count, total_tokens, sections_copied
    """
    pool = get_global_sync_db_pool()
    conn = pool.getconn()

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── 1. Fetch all chunks (Ada or Voyage) ───────────────────────────
            cur.execute(
                '''
                SELECT id, content, metadata,
                       embedding_ada_legacy, embedding_voyage_2,
                       num_tokens, page_number, chunk_index,
                       chunk_summary, section_path, chunk_type, parent_chunk_id
                FROM document_vector_store
                WHERE source_id = %s
                  AND (embedding_ada_legacy IS NOT NULL OR embedding_voyage_2 IS NOT NULL)
                ORDER BY chunk_index
                ''',
                (existing_source_id,)
            )
            existing_chunks = cur.fetchall()

            if not existing_chunks:
                logger.warning(f"No embeddings found for source_id {existing_source_id}")
                return {'copied_count': 0, 'total_tokens': 0, 'sections_copied': 0}

            # ── 2. Build old-UUID → new-UUID mapping for parent_chunk_id remap ─
            id_map = {str(row['id']): str(uuid.uuid4()) for row in existing_chunks}

            # ── 3. Build chunk insert records ─────────────────────────────────
            chunk_records = []
            total_tokens = 0
            now = datetime.now(timezone.utc)

            for row in existing_chunks:
                new_id = id_map[str(row['id'])]
                old_parent = row.get('parent_chunk_id')
                new_parent = id_map.get(str(old_parent)) if old_parent else None

                chunk_records.append((
                    new_id,
                    new_source_id,
                    project_id,
                    row['content'],
                    Json(row['metadata']),
                    row['embedding_ada_legacy'],
                    row['embedding_voyage_2'],
                    row['num_tokens'],
                    row['page_number'],
                    row['chunk_index'],
                    row.get('chunk_summary'),
                    row.get('section_path'),
                    row.get('chunk_type'),
                    new_parent,
                    user_id,
                    now,
                ))
                total_tokens += row['num_tokens'] or 0

            cur.executemany(
                '''INSERT INTO document_vector_store
                   (id, source_id, project_id, content, metadata,
                    embedding_ada_legacy, embedding_voyage_2,
                    num_tokens, page_number, chunk_index,
                    chunk_summary, section_path, chunk_type, parent_chunk_id,
                    user_id, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                chunk_records,
            )

            # ── 4. Copy document_sections (section-level Voyage embeddings) ───
            cur.execute(
                '''
                SELECT section_path, section_summary, embedding,
                       start_chunk_idx, end_chunk_idx
                FROM document_sections
                WHERE source_id = %s
                ''',
                (existing_source_id,)
            )
            existing_sections = cur.fetchall()

            section_records = []
            for sec in existing_sections:
                section_records.append((
                    str(uuid.uuid4()),
                    new_source_id,
                    project_id,
                    sec['section_path'],
                    sec['section_summary'],
                    sec['embedding'],
                    sec['start_chunk_idx'],
                    sec['end_chunk_idx'],
                    now,
                ))

            if section_records:
                cur.executemany(
                    '''INSERT INTO document_sections
                       (id, source_id, project_id, section_path, section_summary,
                        embedding, start_chunk_idx, end_chunk_idx, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT DO NOTHING''',
                    section_records,
                )

            conn.commit()

        logger.info(
            f"✅ Copied {len(chunk_records)} chunks + {len(section_records)} sections "
            f"from {existing_source_id} to {new_source_id} for project {project_id}"
        )
        return {
            'copied_count': len(chunk_records),
            'total_tokens': total_tokens,
            'sections_copied': len(section_records),
        }

    finally:
        pool.putconn(conn)

async def _download_and_prep_doc(client: httpx.AsyncClient, url: str, project_id: str, user_id: str) -> Optional[Dict]:
    """
    Helper for `_analyze_download_and_store_document_for_workflow` to download to memory (does not write to Disk),
    hash, stream to S3 & Store in AWS CLoudfront, and prep data.

    Transfer minimisation (two fixes, both material on large PDFs):

    1. When `url` is already on our own CloudFront distribution — which is the
       case for every WeWeb upload, since WeWeb pushes to S3/CDN before calling
       us — the S3 re-upload is skipped entirely and the incoming URL is reused.
       Previously the identical bytes were pushed back up to a fresh S3 key.
    2. The downloaded bytes are parked in the ingest cache keyed by content_hash
       so the parser can pick them up instead of re-downloading from CloudFront.

    Together these take the common path from three full transfers of the file
    (download → upload → download) down to one.
    """
    try:
        # Define headers for anti-bot detection
        headers = DEFAULT_HEADERS

        # Define client stream
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()

            # Stream response into an in-memory buffer
            content_stream = io.BytesIO()
            async for chunk in response.aiter_bytes():
                content_stream.write(chunk)

            file_size = content_stream.tell()
            if file_size == 0:
                logger.warning(f"Skipping zero-byte file from URL: {url}")
                return None

            content_hash = _calculate_stream_hash(content_stream)

            # Enhanced file extension detection
            ext = get_file_extension_from_url(url)
            filename = parse_clean_filename_from_url(url, ext)

            # Park the bytes for the parser FIRST. boto3's upload_fileobj closes
            # the stream it is handed (multipart TransferConfig), so caching
            # after an upload always failed with "I/O operation on closed file"
            # and the parser silently fell back to re-downloading from the CDN —
            # defeating the whole point of the cache.
            await asyncio.to_thread(_write_ingest_cache, content_hash, content_stream)

            if _is_own_cdn_url(url):
                # Already on our CDN — nothing to copy.
                cdn_url = url
                logger.info(f"⚡ Skipped S3 re-upload — '{filename}' is already on our CDN")
            else:
                s3_key = f"{project_id}/{uuid.uuid4()}{ext}"
                # Stream directly to S3 && AWS CloudFront from the in-memory buffer
                # (boto3 is blocking, so keep it off the shared event loop).
                content_stream.seek(0)
                await asyncio.to_thread(
                    upload_to_s3,
                    client=s3_client,
                    file_source=content_stream,
                    s3_object_key=s3_key,
                )
                cdn_url = get_cloudfront_url(s3_key)

            # Leverages my url (AWS Cloudfront) to perform in-memory streaming for the rest of the ingest pipeline
            return {
                'cdn_url': cdn_url,
                'project_id': project_id,
                'uploaded_by': user_id,
                'filename': filename,
                'file_size_bytes': file_size,
                'content_hash': content_hash,
            }
    except Exception as e:
        logger.error(f"Failed to download and prep {url}: {e}")
        return None

def _create_smart_embedding_batches(chunks: List[str], metadatas: List[Dict]) -> List[Dict]:
    """
    Create token-aware batches - same logic as your legacy code
    """
    batches = []
    current_batch_texts = []
    current_batch_metas = []
    current_batch_tokens = 0
    
    for text, metadata in zip(chunks, metadatas):
        # Use your existing tokenizer
        token_count = len(tokenizer.encode(text))
        
        # Check if adding this text would exceed batch limit
        if current_batch_tokens + token_count > OPENAI_MAX_TOKENS_PER_BATCH and current_batch_texts:
            # Finalize current batch
            batches.append({
                'texts': current_batch_texts,
                'metadatas': current_batch_metas
            })
            current_batch_texts, current_batch_metas, current_batch_tokens = [], [], 0
        
        current_batch_texts.append(text)
        current_batch_metas.append(metadata)
        current_batch_tokens += token_count
    
    # Add final batch if not empty
    if current_batch_texts:
        batches.append({
            'texts': current_batch_texts,
            'metadatas': current_batch_metas
        })
    
    return batches

async def _call_voyage_embeddings_async(texts: List[str]) -> List[List[float]]:
    """Voyage AI embeddings via voyage-law-2 (1024-dim). Called when USE_VOYAGE_EMBEDDINGS=true."""
    from utils.llm_clients.voyage_client import get_voyage_client
    loop = asyncio.get_event_loop()
    voyage = get_voyage_client()
    try:
        return await loop.run_in_executor(_LLM_EXECUTOR, voyage.embed_documents, texts)
    except Exception as e:
        logger.error(f"Voyage API call failed: {e}")
        return []


async def _call_openai_embeddings_async(texts: List[str]) -> List[List[float]]:
    """
    Call the active embedding API asynchronously.
    Routes to Voyage (voyage-law-2) when USE_VOYAGE_EMBEDDINGS=true,
    otherwise calls OpenAI text-embedding-ada-002.
    """
    if USE_VOYAGE_EMBEDDINGS:
        return await _call_voyage_embeddings_async(texts)

    # ── Legacy: OpenAI ada-002 ────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()
            return [item['embedding'] for item in data['data']]

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return []
    
# ——— Hierarchical Ingest Helpers (Phase 1) ———————————————————————————————————


async def _parse_and_chunk_docling_async(
    source_id: str,
    source_filename: str,
    cdn_url: str,
    project_id: str,
    content_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hierarchical ingest PDF path: download → DoclingPDFLoader (HybridChunker).
    Returns the same dict contract as _parse_document_async so the caller is
    unaware of which path ran.

    Extra key 'raw_doc' carries the DoclingDocument for TOC extraction and
    blurb generation; it is popped before the result is returned upstream.

    When `content_hash` is supplied and the batch coordinator has already parked
    the bytes in the ingest cache, the CloudFront download is skipped entirely.
    """
    await asyncio.to_thread(
        _update_document_status_sync, source_id, ProcessingStatus.PARSING
    )
    short_id = source_id[:8]

    try:
        # ── Acquire bytes: cache first, CDN as fallback ──────────────────────
        file_buffer = await asyncio.to_thread(_read_ingest_cache, content_hash)

        if file_buffer is not None:
            logger.info(
                f"⚡ [DOCLING-{short_id}] Reused {file_buffer.getbuffer().nbytes:,} "
                f"cached bytes — CDN download skipped"
            )
        else:
            file_buffer = io.BytesIO()
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("GET", cdn_url, headers=DEFAULT_HEADERS) as response:
                    response.raise_for_status()
                    async for raw_bytes in response.aiter_bytes(chunk_size=8_192):
                        if raw_bytes:
                            file_buffer.write(raw_bytes)
            downloaded_bytes = file_buffer.tell()
            file_buffer.seek(0)
            logger.info(f"📥 [DOCLING-{short_id}] Downloaded {downloaded_bytes:,} bytes")

        # ── Docling parse + HybridChunker (CPU-bound → dedicated executor) ───
        file_ext = source_filename.rsplit(".", 1)[-1].upper() if "." in source_filename else "PDF"
        logger.info(f"🦆 Parsing {file_ext} with Docling 🪿")
        # Shared singleton — a per-document loader would load its own copy of the
        # layout model (~hundreds of MB), and N concurrent uploads would then
        # multiply that until the worker OOMed.
        from utils.document_loaders.docling_loader import get_docling_loader
        loader = get_docling_loader()
        loop = asyncio.get_event_loop()
        texts, metadatas, raw_doc = await loop.run_in_executor(
            _CPU_EXECUTOR,
            lambda: loader.load_document(file_buffer, source_id, source_filename, cdn_url),
        )

        for meta in metadatas:
            meta.setdefault('filename', source_filename)
            meta.setdefault('title', source_filename)

        pages = {m.get('page') for m in metadatas if m.get('page') is not None}
        total_pages = max(pages, default=0)

        logger.info(
            f"✅ [DOCLING-{short_id}] {len(texts)} chunks, ~{total_pages} pages"
        )
        return {
            'success': True,
            'chunks': texts,
            'metadatas': metadatas,
            'total_pages': total_pages,
            'raw_doc': raw_doc,
            'performance_metrics': {},
        }

    except Exception as e:
        logger.error(f"💥 [DOCLING-{short_id}] Parsing failed: {e}", exc_info=True)
        await asyncio.to_thread(
            _update_document_status_sync,
            source_id,
            ProcessingStatus.FAILED_PARSING,
            str(e),
        )
        return {'success': False, 'error': str(e)}


def _build_doc_synopsis(
    chunks: List[str],
    metadatas: List[Dict],
    filename: Optional[str] = None,
    toc: Optional[List[Dict]] = None,
) -> str:
    """
    Compact stand-in for the full document text in the blurb system prompt.

    Previously the entire document (up to 50 000 chars ≈ 12.5k tokens) was
    re-sent on EVERY blurb call — ~475k input tokens for a 300-chunk document,
    and the dominant wall-clock cost of the hierarchical path.

    Contextual retrieval only needs enough to situate a chunk, and the precise
    location already travels per-chunk in `section_path`. So the shared context
    is reduced to: filename, the Docling-extracted table of contents (the
    document's skeleton), and the opening passage (caption, court, parties).
    """
    parts: List[str] = []

    if filename:
        parts.append(f"Title: {filename}")

    if toc:
        # extract_toc_from_docling_doc emits {"text", "level", "page"}
        outline = "\n".join(
            f"{'  ' * max(0, int(entry.get('level') or 1) - 1)}- {entry.get('text')}"
            for entry in toc[:60]
            if entry.get("text")
        )
        if outline:
            parts.append(f"Table of contents:\n{outline}")
    else:
        # No TOC (scanned doc, flat structure) — fall back to the distinct
        # section paths Docling assigned, which convey the same skeleton.
        seen, sections = set(), []
        for meta in metadatas:
            sp = (meta or {}).get("section_path")
            if sp and sp not in seen:
                seen.add(sp)
                sections.append(sp)
            if len(sections) >= 40:
                break
        if sections:
            parts.append("Sections:\n" + "\n".join(f"- {s}" for s in sections))

    opening = "\n\n".join(chunks[:3])[:BLURB_CONTEXT_CHARS]
    if opening:
        parts.append(f"Opening passage:\n{opening}")

    return "\n\n".join(parts)


async def _generate_chunk_blurbs_async(
    chunks: List[str],
    metadatas: List[Dict],
    doc_id: str,
    filename: Optional[str] = None,
    toc: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    [AGENTIC INGEST]
    Contextual retrieval: generate a 1-2 line situating blurb for every chunk.
    Provider and model are controlled by INGEST_LLM_PROVIDER / INGEST_LLM_MODEL
    (defaults: gemini / gemini-3.1-flash-lite) and can be hot-swapped via .env.

    Batching strategy (reduces API roundtrips on large documents):
      - Prose/heading/list chunks: batched BLURB_BATCH_SIZE per LLM call.
        The model returns a JSON array [{"i": 0, "blurb": "..."}, ...].
        Falls back to empty blurbs for any batch that fails JSON parsing.
      - Table chunks: processed individually (need longer 2-4 sentence summaries).

    The shared system prompt is a compact synopsis (see _build_doc_synopsis)
    rather than the full document, and is byte-identical across every call for
    a given document so provider-side prompt caching can hit.

    Populates two keys in each metadata dict:
      chunk_summary — stored in document_vector_store.chunk_summary
      embed_text    — the text actually passed to the embedding model
                      (blurb+content for prose; LLM summary for tables)

    Returns the updated metadatas list.
    """
    from utils.llm_clients.llm_factory import LLMFactory

    short_id = doc_id[:8]

    # Compact, cache-friendly document context shared by every call below.
    synopsis = _build_doc_synopsis(chunks, metadatas, filename, toc)
    system_prompt = (
        "You are a legal-document analyst helping to build a retrieval index.\n\n"
        f"<document>\n{synopsis}\n</document>"
    )
    logger.info(
        f"📝 [BLURB-{short_id}] Shared context {len(synopsis):,} chars "
        f"(was up to 50,000)"
    )

    # Two clients per document:
    #   client_table — 200 tokens (individual table summaries)
    #   client_batch — BLURB_BATCH_SIZE * 80 + 150 tokens (batched prose blurbs)
    # client.chat() only accepts (prompt, system_prompt) — max_output_tokens is
    # set at construction time.
    batch_max_tokens = BLURB_BATCH_SIZE * 80 + 150
    try:
        client_table = LLMFactory.get_client_for(
            INGEST_LLM_PROVIDER, INGEST_LLM_MODEL,
            temperature=0.2, streaming=False, max_output_tokens=200,
        )
        client_batch = LLMFactory.get_client_for(
            INGEST_LLM_PROVIDER, INGEST_LLM_MODEL,
            temperature=0.2, streaming=False, max_output_tokens=batch_max_tokens,
        )
    except Exception as e:
        logger.warning(f"⚠️ [BLURB-{short_id}] Could not create ingest LLM client ({INGEST_LLM_PROVIDER}/{INGEST_LLM_MODEL}): {e} — skipping blurbs")
        return metadatas

    loop = asyncio.get_event_loop()
    # Semaphore caps concurrent API calls (batch calls count the same as individual).
    # Sized against _LLM_EXECUTOR — with the old default executor this could never
    # exceed ~6 on a 2-vCPU box no matter what number was written here.
    semaphore = asyncio.Semaphore(int(os.getenv("BLURB_CONCURRENCY", "16")))

    # ── Individual table blurb (unchanged logic, uses client_table) ──────
    async def _blurb_one_table(idx: int, text: str, meta: Dict) -> tuple:
        truncated = text[:2_000]
        user_msg = (
            "Write 2-4 sentences summarising the following table from a legal "
            "document.  Identify what it shows, any key legal concepts or "
            "statutes, and what a law student would learn from it.  "
            "Output only the sentences, no preamble.\n\n"
            f"<table>\n{truncated}\n</table>"
        )
        async with semaphore:
            try:
                blurb = await loop.run_in_executor(
                    _LLM_EXECUTOR, lambda: client_table.chat(user_msg, system_prompt)
                )
                blurb = blurb.strip()
            except Exception as e:
                logger.debug(f"Table blurb gen failed for chunk {idx}: {e}")
                blurb = ""

        updated_meta = dict(meta)
        updated_meta['chunk_summary'] = blurb
        updated_meta['embed_text'] = blurb if blurb else text
        return idx, updated_meta

    # ── Batched prose blurbs ──────────────────────────────────────────────
    async def _blurb_batch(batch_global_indices: List[int]) -> List[tuple]:
        """
        Send BLURB_BATCH_SIZE prose chunks in a single LLM call.
        Returns a list of (global_idx, updated_meta) tuples.
        Falls back to empty blurbs for the whole batch on any failure.
        """
        batch_texts = [chunks[i][:1_000] for i in batch_global_indices]
        batch_metas = [metadatas[i] for i in batch_global_indices]

        # Build numbered chunk list for the prompt. Each chunk carries its own
        # section_path, which is the per-chunk context that used to be inferred
        # from the full-document dump in the system prompt.
        chunks_block = "\n".join(
            f"[{local_i}] ({(m or {}).get('section_path') or 'body'}) <chunk>{t}</chunk>"
            for local_i, (t, m) in enumerate(zip(batch_texts, batch_metas))
        )
        user_msg = (
            "Write exactly 1-2 sentences for each chunk below that situate it "
            "in the context of the document above, mentioning the relevant legal "
            "concept, rule, or case name.  The parenthesised text after each "
            "index is the chunk's location in the document outline.\n"
            f"Return ONLY a JSON array with {len(batch_global_indices)} objects: "
            '[{"i": 0, "blurb": "..."}, {"i": 1, "blurb": "..."}, ...]\n\n'
            f"<chunks>\n{chunks_block}\n</chunks>"
        )
        async with semaphore:
            try:
                raw = await loop.run_in_executor(
                    _LLM_EXECUTOR, lambda: client_batch.chat(user_msg, system_prompt)
                )
                raw = raw.strip()
            except Exception as e:
                logger.debug(f"Batch blurb gen failed for indices {batch_global_indices}: {e}")
                raw = "[]"

        # Parse the JSON array response
        blurb_map: Dict[int, str] = {}
        try:
            # Strip markdown code fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
                clean = clean.rstrip("`").strip()
            parsed = json.loads(clean)
            for item in parsed:
                if isinstance(item, dict) and "i" in item and "blurb" in item:
                    blurb_map[int(item["i"])] = str(item["blurb"]).strip()
        except Exception as e:
            logger.debug(f"Batch blurb JSON parse failed: {e} — raw: {raw[:200]}")

        results_out = []
        for local_i, (global_idx, text, meta) in enumerate(
            zip(batch_global_indices, batch_texts, batch_metas)
        ):
            blurb = blurb_map.get(local_i, "")
            updated_meta = dict(meta)
            updated_meta['chunk_summary'] = blurb
            updated_meta['embed_text'] = f"{blurb}\n\n{chunks[global_idx]}" if blurb else chunks[global_idx]
            results_out.append((global_idx, updated_meta))

        return results_out

    # ── Separate table vs prose indices ──────────────────────────────────
    table_indices = [i for i, m in enumerate(metadatas) if m.get('chunk_type') == 'table']
    prose_indices = [i for i, m in enumerate(metadatas) if m.get('chunk_type') != 'table']

    # Build batches for prose chunks
    batches = [
        prose_indices[i:i + BLURB_BATCH_SIZE]
        for i in range(0, len(prose_indices), BLURB_BATCH_SIZE)
    ]

    n_batches = len(batches)
    n_tables  = len(table_indices)
    logger.info(
        f"📝 [BLURB-{short_id}] {len(chunks)} chunks → "
        f"{n_batches} prose batch(es) of ≤{BLURB_BATCH_SIZE} + {n_tables} table(s)"
    )

    # ── Launch all tasks concurrently ─────────────────────────────────────
    all_tasks = (
        [_blurb_batch(b) for b in batches]
        + [_blurb_one_table(i, chunks[i], metadatas[i]) for i in table_indices]
    )
    raw_results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # ── Merge results back into the metadatas list ────────────────────────
    updated = list(metadatas)
    n_ok = 0
    for r in raw_results:
        if isinstance(r, Exception):
            logger.debug(f"Blurb task raised: {r}")
            continue
        if isinstance(r, list):
            # batch result: list of (idx, meta) tuples
            for item in r:
                if isinstance(item, tuple):
                    idx, meta = item
                    updated[idx] = meta
                    if meta.get('chunk_summary'):
                        n_ok += 1
        elif isinstance(r, tuple):
            # individual table result
            idx, meta = r
            updated[idx] = meta
            if meta.get('chunk_summary'):
                n_ok += 1

    logger.info(f"📝 [BLURB-{short_id}] {n_ok}/{len(chunks)} blurbs generated via {INGEST_LLM_PROVIDER}/{INGEST_LLM_MODEL}")
    return updated


# ——— Async Helper Functions ——————————————————————————————————————————————————————

async def _parse_document_async(
    source_id: str,
    source_filename: str,
    cdn_url: str,
    project_id: str,
    content_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse single document (or any type) asynchronously.
    Uses the DocumentLoader class to parse documents

    Returns:
        Dict: parse results dict including (
            'success': True,
            'chunks': all_chunks,
            'metadatas': all_metadatas,  # Include metadata for embedding
            'total_pages': processing_summary['total_pages'],
            'performance_metrics': perf_summary)
    """
    await asyncio.to_thread(
        _update_document_status_sync, source_id, ProcessingStatus.PARSING
    )

    short_id = source_id[:8]
    file_buffer = io.BytesIO()
    doc_metrics = DocumentMetrics(doc_id=source_id)
    perf_summary = {}

    with Timer() as total_timer:
        try:
            # Phase 1: Acquire bytes — ingest cache first, streaming download as fallback
            with Timer() as download_timer:
                cached_buffer = await asyncio.to_thread(_read_ingest_cache, content_hash)

                if cached_buffer is not None:
                    file_buffer = cached_buffer
                    logger.info(
                        f"⚡ [PARSE-{short_id}] Reused {file_buffer.getbuffer().nbytes:,} "
                        f"cached bytes — CDN download skipped"
                    )
                else:
                    logger.info(f"🚀 Starting document streaming for {source_id}, with url: {cdn_url}")

                    # ✅ FIXED: Use async HTTP client instead of requests
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream("GET", cdn_url, headers=DEFAULT_HEADERS) as response:
                            response.raise_for_status()

                            # Log response headers for debugging
                            content_length = response.headers.get('content-length')
                            content_type = response.headers.get('content-type', 'unknown')

                            # Stream into memory buffer
                            downloaded_bytes = 0
                            chunk_count = 0

                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                if chunk:
                                    file_buffer.write(chunk)
                                    downloaded_bytes += len(chunk)
                                    chunk_count += 1

                                    # Log progress every 5MB or 500 chunks
                                    if chunk_count % 500 == 0 or downloaded_bytes % (1024 * 1024 * 5) == 0:
                                        logger.info(f"📥 [PARSE-{short_id}] Downloaded {downloaded_bytes:,} bytes ({chunk_count} chunks)")

                                    # ✅ FIXED: Use asyncio.sleep instead of gevent.sleep
                                    await asyncio.sleep(0)

                file_buffer.seek(0)
                doc_metrics.download_time_ms = download_timer.elapsed_ms
                doc_metrics.file_size_bytes = file_buffer.getbuffer().nbytes
                perf_summary['download_ms'] = download_timer.elapsed_ms

            # Phase 2: Document Analysis & Optimal Loader Selection (CPU-bound, keep sync)
            with Timer() as analysis_timer:
                # This is CPU-bound, keep it sync - it's fast
                doc_analysis = analyze_document_before_processing(cdn_url, file_buffer)
                
                logger.info(f"📊 [PARSE-{short_id}] Analysis: {doc_analysis['file_size_mb']:.1f}MB, "
                            f"complexity: {doc_analysis.get('complexity_score', 'N/A')}, "
                            f"estimated time: {doc_analysis['processing_time_estimate']:.1f}s")
                
                # Choose optimal loader based on analysis
                if doc_analysis.get('complexity_score', 0) >= 7:
                    loader = get_high_performance_loader(cdn_url, file_buffer)
                    processor_mode = "fast"
                    logger.info(f"⚡ [PARSE-{short_id}] Using HIGH PERFORMANCE mode")
                else:
                    loader = get_loader_for(cdn_url, file_buffer, performance_mode="auto")
                    processor_mode = "balanced"
                    logger.info(f"⚖️ [PARSE-{short_id}] Using BALANCED mode")
                
                perf_summary['analysis_ms'] = analysis_timer.elapsed_ms

            # Phase 3: Text Processing & Chunking (CPU-bound, but with async yields)
            with Timer() as process_timer:
                processor = create_optimized_processor(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    performance_mode=processor_mode
                )
                
                all_chunks, all_metadatas = [], []
                
                logger.info(f"🔄 [PARSE-{short_id}] Starting streaming processing...")
                
                # Process document stream into chunks
                logger.info(f"USING {loader.name} loader to process documents...")

                # Returns a tuple (text, meta). `meta` has keys {paragraph_index, page, estmated_page...}
                document_stream = loader.stream_documents(file_buffer)

                text_stream = processor.process_documents_streaming(
                    source_filename=source_filename,
                    documents=document_stream, 
                    source_id=source_id,
                    source_url=cdn_url
                )

                # ✅ FIXED: Add async yields for long-running CPU work
                for chunk_text, chunk_metadata in text_stream:

                    if isinstance(chunk_metadata, dict): # Check if metadata is a dict
                        chunk_metadata['filename'] = source_filename
                        chunk_metadata['title'] = source_filename # Use filename as title for now
                    
                    all_chunks.append(chunk_text)
                    all_metadatas.append(chunk_metadata)
                    
                    # Log progress every 200 chunks
                    if len(all_chunks) % 200 == 0:
                        logger.info(f"✂️ [PARSE-{short_id}] Created {len(all_chunks)} chunks")
                    
                    # ✅ FIXED: Yield control every 50 chunks to prevent blocking
                    if len(all_chunks) % 50 == 0:
                        await asyncio.sleep(0)
                
                logger.info(f"⏱️ PARSING COMPLETE...")
                
                # Get processing performance summary
                processing_summary = processor.get_performance_summary()
                perf_summary.update({
                    'text_processing_ms': process_timer.elapsed_ms,
                    'total_chunks': len(all_chunks),
                    'total_pages': processing_summary['total_pages'],
                    'chars_per_second': processing_summary['chars_per_second'],
                    'chunks_per_second': processing_summary['chunks_per_second']
                })
                
                logger.info(f"✅ [PARSE-{short_id}] Processing complete: {len(all_chunks)} chunks from {processing_summary['total_pages']} pages")
            
            return {
                'success': True,
                'chunks': all_chunks,
                'metadatas': all_metadatas,  # Include metadata for embedding
                'total_pages': processing_summary['total_pages'],
                'performance_metrics': perf_summary
            }
            
        except Exception as e:
            logger.error(f"💥 [PARSE-{short_id}] PARSING FAILED: {e}", exc_info=True)
            _update_document_status_sync(source_id, ProcessingStatus.FAILED_PARSING, str(e))
            return {
                'success': False,
                'error': str(e)
            }

async def _embed_batch_async(
        doc_id: str, 
        project_id: str, 
        batch_info: List[str],
) -> Dict[str, Any]:
    """
    Process a SINGLE embedding batch - combines your legacy robustness with async benefits
    
    This replaces _embed_batch_gevent but keeps all the good parts:
    - Same error handling and retry logic
    - Same database operations (using your sync pool)
    - Same token counting logic
    - Added: Async HTTP calls and proper error handling
    """
    texts = batch_info['texts']
    metadatas = batch_info['metadatas']
    short_id = doc_id[:8]
    
    try:
        logger.info(f"🤖 [BATCH-{short_id}] Processing {len(texts)} texts")
        
        # ——— 1. Generate Embeddings ——————————————————————————————————————————————
        # When USE_HIERARCHICAL_INGEST=true, metadatas carry an 'embed_text' key
        # set by _generate_chunk_blurbs_async.  We embed that instead of the raw
        # content (blurb+content for prose, LLM summary for tables).
        embed_inputs = [
            (m.get('embed_text') or t) if isinstance(m, dict) else t
            for t, m in zip(texts, metadatas)
        ]
        embeddings = await _call_openai_embeddings_async(embed_inputs)

        if not embeddings:
            return {
                'success': False,
                'error': 'Failed to generate embeddings'
            }

        # ——— 2. Prepare Data ————————————————————————————————————————————————————
        records_to_insert = []
        total_tokens = 0

        for text, meta, vec in zip(texts, metadatas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN:
                logger.warning(f"⚠️ [BATCH-{short_id}] Skipping malformed embedding (len={len(vec)})")
                continue

            token_count = len(tokenizer.encode(text))
            total_tokens += token_count

            page_number = meta.get('page', meta.get('estimated_page')) if isinstance(meta, dict) else None
            chunk_index = meta.get('chunk_index') if isinstance(meta, dict) else None
            source_url  = meta.get('source_id', None)

            # New hierarchical columns (NULL when USE_HIERARCHICAL_INGEST=false)
            chunk_summary    = meta.get('chunk_summary')    if isinstance(meta, dict) else None
            section_path     = meta.get('section_path')     if isinstance(meta, dict) else None
            chunk_type       = meta.get('chunk_type')       if isinstance(meta, dict) else None
            parent_chunk_id  = meta.get('parent_chunk_id')  if isinstance(meta, dict) else None

            # Strip internal-only keys before persisting to the metadata JSONB column
            meta_clean = {
                k: v for k, v in (meta or {}).items()
                if k not in ('embed_text',)
            }

            records_to_insert.append((
                str(uuid.uuid4()),
                str(uuid.UUID(doc_id)),
                str(uuid.UUID(project_id)),
                text,
                json.dumps(meta_clean),
                vec,
                token_count,
                page_number,
                chunk_index,
                source_url,
                chunk_summary,
                section_path,
                chunk_type,
                parent_chunk_id,
                datetime.now(timezone.utc),
            ))

        if not records_to_insert:
            return {
                'success': False,
                'error': 'No valid embeddings to insert'
            }

        # ——— 3. Database Insert ——————————————————————————————————————————————————
        # Route embedding vector to the correct column:
        #   USE_VOYAGE_EMBEDDINGS=true  → embedding_voyage_2   (vector(1024), voyage-law-2)
        #   USE_VOYAGE_EMBEDDINGS=false → embedding_ada_legacy (vector(1536), ada-002)
        embedding_col = "embedding_voyage_2" if USE_VOYAGE_EMBEDDINGS else "embedding_ada_legacy"

        # psycopg2 is blocking and the worker shares ONE event loop across every
        # concurrent document and chat stream, so this must not run inline:
        # a few hundred vector rows would stall everything else in the process.
        # execute_values also beats executemany substantially at this row count.
        def _insert_vectors():
            pool = get_global_sync_db_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f'''INSERT INTO document_vector_store
                        (id, source_id, project_id, content, metadata, {embedding_col},
                         num_tokens, page_number, chunk_index, cdn_url,
                         chunk_summary, section_path, chunk_type, parent_chunk_id,
                         created_at)
                        VALUES %s''',
                        records_to_insert,
                        page_size=200,
                    )
                    conn.commit()
            finally:
                pool.putconn(conn)

        await asyncio.to_thread(_insert_vectors)

        logger.info(f"✅ [BATCH-{short_id}] Stored {len(records_to_insert)} embeddings")
        
        return {
            'success': True,
            'chunks_embedded': len(records_to_insert),
            'token_count': total_tokens
        }
        
    except Exception as e:
        logger.error(f"❌ [BATCH-{short_id}] Batch processing failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    
async def _handle_batch_failure_async(
    batch_id: str, 
    metadata: Dict[str, Any], 
    errors: List[str]
) -> None:
    """
    Handle complete batch failure (all documents failed to download)
    """
    project_id = metadata['project_id']
    
    logger.error(f"💥 [BATCH-{batch_id[:8]}] Complete batch failure:")
    for i, error in enumerate(errors, 1):
        logger.error(f"   {i}. {error}")
    
    # Update batch progress
    await asyncio.to_thread(
        _update_batch_progress_sync, batch_id, project_id, BatchProgressStatus.BATCH_FAILED
    )

async def _process_embeddings_async(doc_id: str, project_id: str, chunks: List[str], metadatas: List[Dict] = None) -> Dict[str, Any]:
    """
    Process document chunks into OpenAI embeddings with smart batching.
    
    Features:
    - Token-aware batching (respects OpenAI limits)
    - Concurrent processing with rate limiting (max 5 requests) 
    - Robust error handling with partial success support
    - Direct database insertion using sync connection pool
    """
    short_id = doc_id[:8]

    try:
        # Create batches (Token-Aware batching)
        embedding_batches = _create_smart_embedding_batches(chunks, metadatas or [{}] * len(chunks))
        
        logger.info(f"🤖 [DOC-{short_id}] Created {len(embedding_batches)} embedding batches")
        
        # ——— 2. Process Batches with Concurrency Control ————————————————————————
        # Rate limiting to respect OpenAI limits
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def process_single_batch(batch_info):
            async with semaphore:
                return await _embed_batch_async(doc_id, project_id, batch_info)
        
        # Process all batches concurrently
        batch_results = await asyncio.gather(
            *[process_single_batch(batch) for batch in embedding_batches],
            return_exceptions=True
        )
        
        # ——— 3. Analyze Results  ———————————————————————————
        results_dict = {}
        successful_batches = []
        failed_batches = []
        total_chunks_embedded = 0
        total_tokens = 0
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"❌ [DOC-{short_id}] Batch failed with exception: {result}")
                failed_batches.append(str(result))
            elif result and result.get('success'):
                successful_batches.append(result)
                total_chunks_embedded += result.get('chunks_embedded', 0)
                total_tokens += result.get('token_count', 0)
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                logger.error(f"❌ [DOC-{short_id}] Batch failed: {error_msg}")
                failed_batches.append(error_msg)

        results_dict = {
            'success': True,
            'chunks_embedded': total_chunks_embedded,
            'total_tokens': total_tokens,
            'successful_batches': len(successful_batches),
            'failed_batches': len(failed_batches),
            'processing_time_ms': 0  # Could add timing if needed
        }
        
        # ——— 4. Determine Final Status ———————————————————————————————————————————
        # Status writes are sync psycopg2 — keep them off the shared event loop.
        if len(successful_batches) == 0:
            await asyncio.to_thread(
                _update_document_status_sync, doc_id, ProcessingStatus.FAILED_EMBEDDING
            )
            return {
                'success': False,
                'error': f'All {len(embedding_batches)} embedding batches failed'
            }
        elif len(failed_batches) > 0:
            await asyncio.to_thread(
                _update_document_status_sync, doc_id, ProcessingStatus.PARTIAL
            )
            logger.warning(f"⚠️ [DOC-{short_id}] Partial success: {len(successful_batches)}/{len(embedding_batches)} batches")
        else:
            await asyncio.to_thread(
                _update_document_status_sync, doc_id, ProcessingStatus.COMPLETE, None, results_dict
            )
            logger.info(f"✅ [DOC-{short_id}] All embeddings successful")

        return results_dict

    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Embedding processing failed: {e}")
        await asyncio.to_thread(
            _update_document_status_sync, doc_id, ProcessingStatus.FAILED_EMBEDDING, str(e)
        )
        return {
            'success': False,
            'error': str(e)
        }

# ——— [GLOBAL BATCH LEVEL] Kickoff & Coordinate Ingest (Fully Async) ———————————————————————————————————————————  

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_document_batch_workflow(
    self, 
    file_urls: List[str], 
    metadata: Dict[str, Any], 
    create_note: bool = False
) -> Dict[str, Any]:
    """
    [SYNC ORCHESTRATOR] 🚶‍➡️ Main entry point for batch document processing, this is the task that gets called by your API/frontend

    1. Use persistent event loop for coordination/analysis
    - Handles mixed new/reused document scenarios

    2. Spawn separate workers for heavy processing

    VISUAL FLOW:
    Celery Task (process_document_batch_workflow)
    ├── run_async_in_worker(_execute_batch_workflow)  # Same process, persistent loop
    │   ├── Async HTTP calls for document analysis    # Concurrent via gather()
    │   ├── Document classification                   # Fast in-memory work
    │   └── Build Celery workflow signatures          # Return coordination plan
    │
    └── workflow_signature.apply_async()              # NEW processes/workers
        ├── Worker 1: process_complete_document_workflow
        ├── Worker 2: process_complete_document_workflow  
        └── Worker 3: finalize_batch_and_create_note
    """
    batch_id = str(uuid.uuid4())
    metadata['create_note'] = create_note
    
    logger.info(f"🎯 [BATCH-{batch_id[:8]}] 🚀 Starting workflow for {len(file_urls)} documents")

    try:
        # Execute async workflow coordination CRITICAL: Delegate to async function (same pattern as your RAG chat)
        workflow_result = run_async_in_worker(
            _execute_batch_workflow(batch_id, file_urls, metadata)
        )
        
        # ✅ Launch Celery workflow signature execution (embedding finalization)
        if workflow_result['status'] == 'WORKFLOW_READY':
            workflow_signature = workflow_result['workflow_signature']
            _update_batch_progress_sync(batch_id, metadata['project_id'], BatchProgressStatus.BATCH_EMBEDDING)
            chord_result = workflow_signature.apply_async()  # ← Execute here, not in async function
            
            return {
                'batch_id': batch_id,
                'workflow_id': chord_result.id,
                'document_count': workflow_result['document_count'],
                'workflow_path': workflow_result['workflow_path'],
                'status': 'WORKFLOW_LAUNCHED'
            }
        else:
            # Edge case handled (duplicates, failures, etc.)
            return workflow_result
            
    except Exception as e:
        logger.error(f"❌ [BATCH-{batch_id[:8]}] Workflow creation failed: {e}")
        raise

async def _execute_batch_workflow(batch_id: str, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    [ASYNC COORDINATOR] Executes the full batch workflow:
    - Downloads and classifies documents concurrently (smart reuse feature)
    - Builds appropriate Celery workflow based on document types
    - Returns workflow execution results

    Batch Level Workflow:
    ├── Document A : (async) Download/stream → (async) classify → REUSED
    ├── Document B : (async) Download/stream → (async) classify → NEW
    ├── Document C : (async) Download/stream → (async) classify → NEW
    └── Build Celery workflow signa tures → New: 2, Reused: 1  # Return coordination plan

    """
    project_id = metadata['project_id']
    user_id = metadata['user_id']

    # Rows pre-created by /speculative-ingest/ exist before this task runs but do
    # not yet carry batch_id in processing_metadata — match them by id so the
    # early phases are actually visible to the UI.
    known_doc_ids = [d for d in [metadata.get('speculative_doc_id')] if d]

    # ——— Step 1: Concurrent Document Analysis ————————————————————————————————————

    logger.info(f"🔍 [BATCH-{batch_id[:8]}] Analyzing document types...")
    await asyncio.to_thread(
        _update_batch_progress_sync,
        batch_id,
        project_id,
        BatchProgressStatus.BATCH_ANALYZING,
        known_doc_ids,
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        analysis_tasks = [
            _analyze_download_and_store_document_for_workflow(client, url, project_id, user_id) 
            for url in file_urls
        ]
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    
    # ——— Step 2: Classify Documents by Processing Type ———————————————————————————

    new_documents = []
    reused_documents = []
    duplicate_documents = []  # Track same-project duplicates separately
    failed_downloads = []
    
    # Classify results, sort into lists
    for result in analysis_results:
        if isinstance(result, Exception):
            failed_downloads.append(str(result))
            continue
        
        if result['processing_type'] == 'NEW':
            new_documents.append(result)
        elif result['processing_type'] == 'REUSED':
            reused_documents.append(result)
        elif result['processing_type'] == 'DUPLICATE':
            duplicate_documents.append(result)
            logger.info(f"📋 [BATCH-{batch_id[:8]}] Duplicate document skipped: {result.get('content_hash', 'unknown')[:8]}")
        else:
            failed_downloads.append(f"Unknown processing type: {result}")
    
    logger.info(f"📊 [BATCH-{batch_id[:8]}] Classification complete:")
    logger.info(f"   🆕 New documents: {len(new_documents)}")
    logger.info(f"   ♻️ Reused documents: {len(reused_documents)}")
    logger.info(f"   📋 Duplicate documents: {len(duplicate_documents)}")
    logger.info(f"   ❌ Failed downloads: {len(failed_downloads)}")
    await asyncio.to_thread(
        _update_batch_progress_sync,
        batch_id,
        project_id,
        BatchProgressStatus.BATCH_PROCESSING,
        known_doc_ids,
    )

    # flattened dictionary
    logger.info(f"🪲 DEBUG: Original metadata keys: {list(metadata.keys())}")
    workflow_metadata = {
        **metadata,
        'batch_id': batch_id,
        'total_documents': len(file_urls),
        'new_count': len(new_documents),
        'reused_count': len(reused_documents),
        'duplicate_count': len(duplicate_documents),
        'failed_count': len(failed_downloads)
    }
    
    # ——— Step 3: Determine Processing Path (ALL REUSED vs NEW/MIXED) ———————————————————————————————————————————
    
    # Calculate processable documents (EXCLUDE duplicates and failures) 
    processable_docs = len(new_documents) + len(reused_documents)
    
    # ——— 1️⃣ Path A: Early exit if nothing to process ———————————————————————————————————————
    if processable_docs == 0:
        logger.warning(f"⚠️ [BATCH-{batch_id[:8]}] No processable documents - all duplicates or failed")
        
        # Sub-case A: All duplicates (documents already exist in THIS project)
        if len(duplicate_documents) > 0:
            logger.info(f"📋 [BATCH-{batch_id[:8]}] All documents are duplicates in current project")
            
            if metadata.get('create_note'):
                # Generate note using existing documents in this project
                logger.info(f"📝 [BATCH-{batch_id[:8]}] Generating note from existing duplicates")
                return await _handle_duplicate_only_batch(batch_id, project_id, metadata, duplicate_documents)
            else:
                # No note requested - just return duplicate status
                return {
                    'batch_id': batch_id,
                    'status': 'ALL_DUPLICATES',
                    'duplicate_count': len(duplicate_documents),
                    'note_generation_triggered': False,
                    'workflow_path': 'DUPLICATE_ONLY'
                }
        
        # Sub-case B: All failed downloads (network/access issues)
        elif len(failed_downloads) > 0:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] All documents failed to download")
            await asyncio.to_thread(
                _update_batch_progress_sync, batch_id, project_id, BatchProgressStatus.BATCH_FAILED
            )
            
            # FIXED: Don't use apply_async - handle failure synchronously
            await _handle_batch_failure_async(batch_id, metadata, failed_downloads)
            
            return {
                'batch_id': batch_id,
                'status': 'ALL_FAILED',
                'failed_count': len(failed_downloads),
                'errors': failed_downloads,
                'workflow_path': 'COMPLETE_FAILURE'
            }
        
        # Sub-case C: No documents at all (shouldn't happen, but defensive)
        else:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] No documents provided")
            return {
                'batch_id': batch_id,
                'status': 'NO_DOCUMENTS',
                'workflow_path': 'EMPTY_BATCH'
            }

    # ——— 2️⃣ Path B: All-Reused ⚡Fast Track ———————————————————————————————————————
    if len(reused_documents) == processable_docs and len(reused_documents) > 0:     
        logger.info(f"⚡ [BATCH-{batch_id[:8]}] All-reused batch - fast track processing")
        
        document_tasks = []  # ← Use consistent variable name
        for doc_info in reused_documents:
            task_sig = process_reused_document_task.s(
                doc_info['existing_doc_id'],
                doc_info['doc_data'],
                doc_info['project_id'],
                {**metadata, 'batch_id': batch_id}
            )
            document_tasks.append(task_sig)  # ← Same variable
        
        # Gather the workflow chord
        workflow_signature = chord(
            group(document_tasks),
            finalize_batch_and_create_note.s(batch_id, workflow_metadata)
        )
        
        return {
            'batch_id': batch_id,
            'workflow_signature': workflow_signature,  # ← Return signature (for execution in `process_document_batch_workflow`)
            'document_count': len(document_tasks),
            'workflow_path': 'ALL_REUSED_FAST_TRACK',
            'status': 'WORKFLOW_READY'  # ← Not launched yet
        }

    
    # ——— 3️⃣ Path C: Mixed/New Documents Standard Workflow ——————————————————————————
    else:      
        logger.info(f"🔄 [BATCH-{batch_id[:8]}] Mixed batch - standard workflow processing")
        
        document_tasks = []
        
        # For NEW documents: Use async delegation pattern
        for doc_info in new_documents:
            # Single task per document - handles everything internally
            task_sig = process_new_document_wrapper.s(
                doc_data=doc_info['doc_data'],
                project_id=doc_info['project_id'],
                workflow_metadata={**metadata, 'batch_id': batch_id}
            )
            document_tasks.append(task_sig)     # ← Add NEW docs to chord signature

        # --- INTEGRATION ⚙️: This could be where we can kick off ainsert() with LightRag -------------

        # For reused documents (keep existing logic)
        for doc_info in reused_documents:
            task_sig = process_reused_document_task.s(
                doc_info['existing_doc_id'],
                doc_info['doc_data'],
                doc_info['project_id'],
                workflow_metadata
            )
            document_tasks.append(task_sig)     # ← Add REUSED docs to chord signature

        # ——— Simple Chord Coordination (for reused_documents && new_documents) ——————————————————————————————————————————————
        logger.info(f"🚀 [BATCH-{batch_id[:8]}] Launching {len(document_tasks)} complete document tasks")
        
        workflow_signature = chord(
            group(document_tasks),
            finalize_batch_and_create_note.s(batch_id, workflow_metadata)
        )
        await asyncio.to_thread(
            _update_batch_progress_sync, batch_id, project_id, BatchProgressStatus.BATCH_EMBEDDING
        )  # ← Technically embedding starts with apply_async() in the parent function but this is a good place
        
        return {
            'batch_id': batch_id,
            'workflow_signature': workflow_signature,  # ← Return signature
            'document_count': len(document_tasks),
            'workflow_path': 'ASYNC_DELEGATION_PATTERN',
            'status': 'WORKFLOW_READY'  # ← Not launched yet
        }

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def finalize_batch_and_create_note(
    self, 
    workflow_results: List[AsyncResult], 
    batch_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Final batch coordinator - analyzes results and triggers note generation.
    
    Process:
    1. Extract final document statuses from workflow results
    2. Calculate batch success metrics (COMPLETE/PARTIAL/FAILED)
    3. Trigger RAG note generation for successful batches
    
    Handles: AsyncResult extraction, status aggregation, note coordination
    Input: List of completed document workflow results 
    Output: Batch summary with note generation status
    """
    project_id = workflow_metadata['project_id']
    logger.info(f"🎯 [BATCH-{batch_id[:8]}] Finalizing batch with {len(workflow_results)} completed workflows")
    logger.info(f"🪲 [BATCH-{batch_id[:8]}] DEBUG : {workflow_results}")
    
    # ——— Extract Results from AsyncResult Objects —————————————————————————————————
    processing_results = []
    
    for i, workflow_result in enumerate(workflow_results):
        try:
            if isinstance(workflow_result, AsyncResult):
                # Extract the actual result value from the AsyncResult
                result_value = workflow_result.result
                
                # Debug: Log what we're getting
                logger.info(f"🔍 [BATCH-{batch_id[:8]}] Workflow {i+1} result type: {type(result_value)}")
                logger.info(f"🔍 [BATCH-{batch_id[:8]}] Workflow {i+1} result value: {result_value}")
                
                # The result_value should be the return from finalize_document_processing
                # which should be a dict, not a list
                if isinstance(result_value, dict):
                    processing_results.append(result_value)
                elif isinstance(result_value, list):
                    # If it's a list, it might be from a chord that returned multiple results
                    # In our case, it should be a single result from finalize_document_processing
                    if len(result_value) == 1 and isinstance(result_value[0], dict):
                        processing_results.append(result_value[0])
                    else:
                        logger.error(f"❌ [BATCH-{batch_id[:8]}] Unexpected list result: {result_value}")
                        processing_results.append({
                            'status': 'FAILED',
                            'error': f'Unexpected result format: {type(result_value)}'
                        })
                else:
                    logger.error(f"❌ [BATCH-{batch_id[:8]}] Unexpected result type: {type(result_value)}")
                    processing_results.append({
                        'status': 'FAILED',
                        'error': f'Unexpected result type: {type(result_value)}'
                    })
            else:
                # Direct result (for reused documents or immediate results)
                logger.info(f"🔍 [BATCH-{batch_id[:8]}] Direct result {i+1}: {workflow_result}")
                processing_results.append(workflow_result)
                
        except Exception as e:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] Failed to extract workflow result {i+1}: {e}")
            processing_results.append({
                'status': 'FAILED',
                'error': f'Failed to extract result: {str(e)}'
            })
    
    # ——— Debug: Log processed results ————————————————————————————————————————————
    logger.info(f"🔍 [BATCH-{batch_id[:8]}] Processed results:")
    for i, result in enumerate(processing_results):
        logger.info(f"   Result {i+1}: {result}")
    
    # ——— Standard Batch Analysis ——————————————————————————————————————————————————
    successful_docs = []
    failed_docs = []
    cancelled_docs = []

    for result in processing_results:
        if result and isinstance(result, dict):
            status = result.get('status')
            if status in ['COMPLETE', 'PARTIAL']:
                successful_docs.append(result)
            elif status == 'CANCELLED':
                # The user removed this file mid-ingest. Neither a success nor a
                # failure — it's excluded from the totals entirely so a cancelled
                # file can't drag the batch to FAILED or block note generation.
                cancelled_docs.append(result)
            elif status == 'FAILED':
                failed_docs.append(result)
            else:
                logger.warning(f"⚠️ [BATCH-{batch_id[:8]}] Unknown status: {status}")
                failed_docs.append(result)
        else:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] Invalid result format: {result}")
            failed_docs.append({
                'status': 'FAILED',
                'error': 'Invalid result format'
            })

    total_docs = len(processing_results) - len(cancelled_docs)
    success_count = len(successful_docs)
    failure_count = len(failed_docs)

    if cancelled_docs:
        logger.info(f"🚫 [BATCH-{batch_id[:8]}] {len(cancelled_docs)} document(s) cancelled by user")
    
    # Calculate total chunks safely
    total_chunks = 0
    total_tokens_reused = 0
    
    for doc in successful_docs:
        if isinstance(doc, dict):
            # Check for either key to correctly sum chunks from NEW or REUSED docs
            chunks_processed = doc.get('chunks_created', 0) or doc.get('chunks_reused', 0)
            total_chunks += chunks_processed
            total_tokens_reused += doc.get('tokens_reused', 0)
    
    # ——— Determine Final Batch Status ————————————————————————————————————————————
    if total_docs == 0:
        # Every document in the batch was cancelled. Nothing succeeded and
        # nothing failed — don't mark the batch FAILED and don't persist a
        # PARSE_ERROR note stub for a file the user deliberately removed.
        logger.info(f"🚫 [BATCH-{batch_id[:8]}] All documents cancelled — nothing to finalize")
        return {
            'batch_id': batch_id,
            'batch_status': 'CANCELLED',
            'successful_documents': 0,
            'failed_documents': 0,
            'cancelled_documents': len(cancelled_docs),
            'total_chunks_processed': 0,
            'tokens_saved': 0,
            'note_generation_triggered': False,
        }

    if success_count == total_docs:
        batch_status = 'COMPLETE'
        final_progress = BatchProgressStatus.BATCH_COMPLETE
        note_context = 'All documents processed successfully'
        _update_batch_progress_sync(batch_id, project_id, final_progress)
    elif success_count > 0:
        batch_status = 'PARTIAL'
        final_progress = BatchProgressStatus.BATCH_PARTIAL
        note_context = f'{success_count}/{total_docs} documents processed successfully'
        _update_batch_progress_sync(batch_id, project_id, final_progress)
    else:
        batch_status = 'FAILED'
        final_progress = BatchProgressStatus.BATCH_FAILED
        note_context = 'All documents failed to process'
        _update_batch_progress_sync(batch_id, project_id, final_progress)
    
    logger.info(f"📊 [BATCH-{batch_id[:8]}] Final analysis:")
    logger.info(f"   ✅ Successful: {success_count}/{total_docs}")
    logger.info(f"   ❌ Failed: {failure_count}")
    logger.info(f"   📄 Total chunks: {total_chunks:,}")
    logger.info(f"   🔄 Status: {batch_status}")

    # ——— 🚦 Clear Speculative-Upload Redis Gate ———————————————————————————————————
    # Covers both the new-doc path (process_new_document_wrapper also clears, but
    # that's a no-op) and the reused-doc path which has no per-task clearing.
    chat_session_id = workflow_metadata.get('chat_session_id')
    if batch_status in ('COMPLETE', 'PARTIAL'):
        from utils.speculative_upload import clear_speculative_upload
        for doc_result in successful_docs:
            doc_id_to_clear = doc_result.get('doc_id')
            if doc_id_to_clear:
                run_async_in_worker(
                    clear_speculative_upload(doc_id_to_clear, chat_session_id)
                )

    # ——— 📝 Trigger Note Generation (Based on Resilience Rules) ——————————————————————
    if workflow_metadata.get('create_note') and batch_status in ['COMPLETE', 'PARTIAL']:
        # Enhance metadata with batch context for note generation
        note_metadata = {
            **workflow_metadata,
            'batch_status': batch_status,
            'successful_documents': success_count,
            'total_documents': total_docs, 
            'processing_context': note_context,
            'total_chunks_available': total_chunks,
            'batch_id': batch_id
        }
        
        logger.info(f"🎯 [BATCH-{batch_id[:8]}] Triggering RAG note generation...")
        try:
            # Pre-create stub row so the frontend realtime listener fires and
            # _save_note_async has a row to UPDATE (same pattern as the API path).
            note_id = str(uuid.uuid4())
            supabase_client.table("notes").insert({
                "id":                   note_id,
                "user_id":              note_metadata["user_id"],
                "project_id":           note_metadata["project_id"],
                "title":                note_metadata["note_title"],
                "note_type":            note_metadata["note_type"],
                "note_progress_status": "INITIALIZED",
                "created_at":           datetime.now(timezone.utc).isoformat(),
            }).execute()

            rag_note_task.apply_async(kwargs={
                "note_id":    note_id,
                "user_id":    note_metadata["user_id"],
                "note_type":  note_metadata["note_type"],
                "project_id": note_metadata["project_id"],
                "note_title": note_metadata["note_title"],
                "provider":   note_metadata.get("provider"),
                "model_name": note_metadata.get("model_name"),
                "temperature": note_metadata.get("temperature"),
                "addtl_params": {
                    **note_metadata.get("addtl_params", {}),
                    'batch_context': {
                        'batch_id': batch_id,
                        'batch_status': batch_status,
                        'document_count': success_count,
                        'total_chunks': total_chunks
                    }
                }
            })
            logger.info(f"✅ [BATCH-{batch_id[:8]}] RAG note generation triggered (note_id={note_id[:8]})")
            
        except Exception as e:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] Failed to trigger note generation: {e}")
            batch_status = 'NOTE_GENERATION_FAILED'
            
    elif batch_status == 'FAILED':
        logger.info(f"⚠️ [BATCH-{batch_id[:8]}] All documents failed — persisting PARSE_ERROR note stub")
        if workflow_metadata.get('create_note'):
            first_error = (failed_docs[0].get('error', 'Document parsing failed') if failed_docs else 'All documents failed to process')
            parse_error_note_id = str(uuid.uuid4())
            try:
                supabase_client.table("notes").insert({
                    "id":                   parse_error_note_id,
                    "user_id":              workflow_metadata["user_id"],
                    "project_id":           workflow_metadata["project_id"],
                    "title":                workflow_metadata.get("note_title", "Failed Note"),
                    "note_type":            workflow_metadata.get("note_type", "unknown"),
                    "note_progress_status": "PARSE_ERROR",
                    "error_message":        first_error[:500],
                    "created_at":           datetime.now(timezone.utc).isoformat(),
                }).execute()
                logger.info(f"📋 [BATCH-{batch_id[:8]}] PARSE_ERROR note persisted (note_id={parse_error_note_id[:8]})")
            except Exception as pe:
                logger.warning(f"⚠️ [BATCH-{batch_id[:8]}] Could not persist PARSE_ERROR note: {pe}")
        
    else:
        logger.info(f"ℹ️ [BATCH-{batch_id[:8]}] Note generation not requested")

    # ——— 🎯 Deferred Note (speculative-ingest flow) ————————————————————————————————
    # Ingest for this batch was started on drag-drop, before the user had chosen a
    # note type, so `create_note` is False here.  If they have since submitted,
    # POST /new-rag-project/attach-note/ parked the request in Redis; fire it now
    # that the documents are terminal.  The DEL-claim inside try_fire_pending_note
    # makes this safe against the endpoint racing us and against the multiple
    # finalizers a multi-file drop produces.
    deferred_note_id = None
    if not workflow_metadata.get('create_note'):
        try:
            from utils.speculative_upload import try_fire_pending_note
            deferred_note_id = run_async_in_worker(try_fire_pending_note(project_id))
        except Exception as e:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] Deferred note dispatch failed: {e}")

    return {
        'batch_id': batch_id,
        'batch_status': batch_status,
        'successful_documents': success_count,
        'failed_documents': failure_count,
        'total_chunks_processed': total_chunks,
        'tokens_saved': total_tokens_reused,
        'note_generation_triggered': bool(
            (workflow_metadata.get('create_note') and batch_status in ['COMPLETE', 'PARTIAL'])
            or deferred_note_id
        ),
        'deferred_note_id': deferred_note_id,
    }

# ——— [DOCUMENT LEVEL] Document Processing  ————————————————————————————————————————————————

# DOCUMENT PRCESSING: Happens after the batch of documents are hashed and sorted into the categories of
# NEW documment (create new embedings), REUSED document (reused embeddings) and DUPLICATE document (skip to note generation)

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_new_document_wrapper(
    self, 
    doc_data: Dict[str, Any], 
    project_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [DOCUMENT PROCESSOR] Thin Celery task wrapper that delegates to async processing:
    - Similar pattern to your RAG chat task
    - Single responsibility: coordinate one complete document
    - Delegates complex async work to dedicated function
    - Clean error handling and state management

    Args:
        - doc_data (Dict): filename, document_uuid, other keys... 
        - project_id (str): project uuid
        - workflow_metadata: Dict[str, Any]
    """
    # Use the pre-generated UUID from speculative ingest if provided,
    # otherwise mint a fresh one (standard non-speculative path).
    doc_id = workflow_metadata.get('speculative_doc_id') or str(uuid.uuid4())
    doc_data['id'] = doc_id
    short_id = doc_id[:8]

    # ——— Poison-message guard ————————————————————————————————————————————————
    # Under task_acks_late, an OOM kill leaves this message unacked and the
    # broker redelivers it forever. Give up after MAX_INGEST_ATTEMPTS so the
    # message is finally acked and the worker stops crash-looping on it.
    attempt = run_async_in_worker(_record_ingest_attempt(doc_id))
    if attempt > MAX_INGEST_ATTEMPTS:
        logger.error(
            f"☠️ [DOC-{short_id}] Delivered {attempt} times — giving up. "
            f"This document repeatedly killed the worker; failing it permanently "
            f"so the queue can drain."
        )
        _update_document_status_sync(
            doc_id,
            ProcessingStatus.FAILED_PARSING,
            f"Abandoned after {attempt} failed attempts (worker kept dying on this document)",
        )
        chat_session_id = workflow_metadata.get('chat_session_id')
        from utils.speculative_upload import clear_speculative_upload
        run_async_in_worker(clear_speculative_upload(doc_id, chat_session_id))
        publish_ingest_progress_sync(
            chat_session_id, doc_id, IngestStage.FAILED,
            filename=doc_data.get('filename'),
            detail="Document could not be processed",
        )
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'FAILED',
            'error': f'Abandoned after {attempt} attempts',
            'chunks_created': 0,
        }

    if attempt > 1:
        logger.warning(f"🔁 [DOC-{short_id}] Redelivery — attempt {attempt}/{MAX_INGEST_ATTEMPTS}")

    # Fan-out + memory telemetry. The counters here are what reveal "six
    # documents at once" in the log instead of leaving it to be inferred.
    from utils.ingest_telemetry import telemetry
    doc_start_rss = telemetry.document_started(doc_id, doc_data.get('filename', ''))
    result_status = 'UNKNOWN'

    try:
        # Set explicit start time metadata (like your pattern)
        self.update_state(
            state="STARTED",
            meta={
                "start_time": datetime.now(timezone.utc).isoformat(),
                "doc_id": doc_id,
                "filename": doc_data.get('filename')
            }
        )
        
        logger.info(f"🚀 [DOC-{short_id}] Starting complete document processing")
        
        # ——— 🔥 CRITICAL: Naive vs LightRag Processing ————————————
        logger.info(f"🪲 DEBUG: Workflow metadata keys: {list(workflow_metadata.keys())}")
        result = run_async_in_worker(
            _process_document_async_workflow(
                doc_id, doc_data, project_id, workflow_metadata
            )
        )
        
        # Update document status (sync call, like your pattern)
        chat_session_id = workflow_metadata.get('chat_session_id')

        # A cancelled document has already had its rows deleted by
        # cancel_speculative_upload. Writing COMPLETE here would re-create the
        # document_sources row via the status UPDATE and resurrect the file in
        # the UI, so skip straight to cleanup.
        if result.get('status') == 'CANCELLED':
            from utils.speculative_upload import (
                clear_document_cancelled,
                clear_speculative_upload,
            )
            run_async_in_worker(clear_speculative_upload(doc_id, chat_session_id))
            run_async_in_worker(clear_document_cancelled(doc_id))
            logger.info(f"🚫 [DOC-{short_id}] Processing stopped — upload was cancelled")
            result_status = 'CANCELLED'
            return result

        _update_document_status_sync(doc_id, ProcessingStatus.COMPLETE, stats=result)

        publish_ingest_progress_sync(
            chat_session_id,
            doc_id,
            IngestStage.COMPLETE,
            filename=doc_data.get('filename'),
        )

        # Clear the speculative-upload tracking for this doc. The doc-scoped
        # task key always clears; the session barrier gate is cleared too when
        # this ingest came from an in-chat drag-drop (chat_session_id present).
        from utils.speculative_upload import clear_speculative_upload
        run_async_in_worker(clear_speculative_upload(doc_id, chat_session_id))

        run_async_in_worker(_clear_ingest_attempts(doc_id))
        logger.info(f"✅ [DOC-{short_id}] Document processing completed successfully")
        result_status = result.get('status', 'COMPLETE')
        return result
        
    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Document processing failed: {e}", exc_info=True)
        _update_document_status_sync(doc_id, ProcessingStatus.FAILED_PARSING, str(e))
        publish_ingest_progress_sync(
            workflow_metadata.get('chat_session_id'),
            doc_id,
            IngestStage.FAILED,
            filename=doc_data.get('filename'),
            detail=str(e)[:200],
        )

        # Return error result instead of raising (better for batch coordination)
        result_status = 'FAILED'
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'FAILED',
            'error': str(e),
            'chunks_created': 0
        }

    finally:
        # Drop the parked bytes — the parser is done with them either way.
        _purge_ingest_cache(doc_data.get('content_hash'))
        # Cleanup (like your pattern). gc runs BEFORE the telemetry read so the
        # reported net delta reflects memory actually retained by this document
        # rather than garbage that simply hasn't been collected yet.
        gc.collect()
        telemetry.document_finished(doc_id, doc_start_rss, result_status)

async def _process_document_async_workflow(
    doc_id: str,
    doc_data: Dict[str, Any], 
    project_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Asynchronous processing for a NEW document (one file) that handles the complete workflow: 
    1. INSERT document into Supaabse
    2. PARSE document into semantic chunks
    3. EMBED chunks using 'smart batching' using OpenAI embeddings

    Techincal Features
    - No Celery task coordination needed
    - Can use async/await for I/O operations
    - Clean error handling
    - Returns final result

    doc_id (str): created earlier in task
    doc_data (Dict): document metadata
    """
    short_id = doc_id[:8]
    timings = PhaseTimer(doc_id)
    chat_session_id = workflow_metadata.get('chat_session_id')
    filename = doc_data.get('filename')
    content_hash = doc_data.get('content_hash')

    from utils.speculative_upload import (
        DocumentCancelledError,
        is_document_cancelled,
    )

    async def _checkpoint(phase: str):
        """
        Bail out if the user cancelled since the last phase.

        Celery cannot kill a running task under `-P threads`, so cancellation is
        cooperative: /speculative-upload/cancel/ raises a Redis flag and we stop
        here. Without this, a cancelled ingest runs to completion and re-creates
        the document_sources row that cancel just deleted.
        """
        if await is_document_cancelled(doc_id):
            raise DocumentCancelledError(
                f"Document {short_id} cancelled by user before {phase}"
            )

    await publish_ingest_progress(
        chat_session_id, doc_id, IngestStage.DOWNLOADED, filename=filename
    )

    try:
        await _checkpoint("insert")
        # ——— 1. INSERT Document Record (Sync DB, off the shared loop) ——————————————
        # psycopg2 blocks; the worker's single event loop serves every concurrent
        # document and chat stream, so this runs in a thread.
        def _insert_document_row():
            # Use global pool instead of local pool
            pool = get_global_sync_db_pool()
            conn = pool.getconn()
            try:
                logger.info(f"📋 [DOC-{short_id}] workflow_meta → {workflow_metadata}")
                with conn.cursor() as cur:
                    # FIXED: Use .get() with default value instead of direct key access
                    is_essential = workflow_metadata.get('is_essential', False)

                    if is_essential:
                        # Get "1L Essential" course and section with defaults
                        essential_course = workflow_metadata.get('essential_course')
                        essential_section = workflow_metadata.get('essential_section')

                        cur.execute(
                            '''INSERT INTO document_sources
                            (id, essential_course, essential_section, is_essential, cdn_url, content_hash, project_id, content_tags, uploaded_by,
                            vector_embed_status, filename, file_size_bytes, file_extension, created_at, processing_metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                content_hash = EXCLUDED.content_hash,
                                vector_embed_status = EXCLUDED.vector_embed_status,
                                processing_metadata = EXCLUDED.processing_metadata''',
                            (doc_id, essential_course, essential_section, is_essential, doc_data['cdn_url'], doc_data['content_hash'],
                            project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                            ProcessingStatus.PENDING.value, doc_data['filename'],
                            doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                            datetime.now(timezone.utc), Json(workflow_metadata))
                        )
                    else:
                        # Non-essential document - use standard insert
                        cur.execute(
                            '''INSERT INTO document_sources
                            (id, cdn_url, content_hash, project_id, content_tags, uploaded_by,
                            vector_embed_status, filename, file_size_bytes, file_extension, created_at, processing_metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                content_hash = EXCLUDED.content_hash,
                                vector_embed_status = EXCLUDED.vector_embed_status,
                                processing_metadata = EXCLUDED.processing_metadata''',
                            (doc_id, doc_data['cdn_url'], doc_data['content_hash'],
                            project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                            ProcessingStatus.PENDING.value, doc_data['filename'],
                            doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                            datetime.now(timezone.utc), Json(workflow_metadata))
                        )
                    conn.commit()
            finally:
                pool.putconn(conn)

        with timings.phase("insert"):
            await asyncio.to_thread(_insert_document_row)

        # ——— 2. PARSE Document ———————————————————————————————————————————————————
        await _checkpoint("parse")
        logger.info(f"📋 [DOC-{short_id}] → PARSING ({'Docling' if USE_HIERARCHICAL_INGEST else 'legacy'})")
        await publish_ingest_progress(
            chat_session_id, doc_id, IngestStage.PARSING, filename=filename
        )

        used_hierarchical = USE_HIERARCHICAL_INGEST
        with timings.phase("parse"):
            if USE_HIERARCHICAL_INGEST:
                parse_result = await _parse_and_chunk_docling_async(
                    doc_id,
                    doc_data['filename'],
                    doc_data['cdn_url'],
                    project_id,
                    content_hash=content_hash,
                )

                # Docling is the richest path but also the most fragile — it
                # drags in torch, a downloaded layout model and (historically) a
                # C++ toolchain, and any of those breaking used to fail the
                # document outright. The legacy loader needs none of that, so
                # fall back rather than lose the upload. The user gets their
                # document searchable; only the hierarchical extras
                # (section_path, chunk_type, table structure) are missing.
                if not parse_result.get('success'):
                    used_hierarchical = False
                    logger.warning(
                        f"⚠️ [DOC-{short_id}] Docling parse failed "
                        f"({parse_result.get('error', 'unknown')!r:.200}) — "
                        f"falling back to the legacy loader"
                    )
                    with timings.phase("parse_fallback"):
                        parse_result = await _parse_document_async(
                            doc_id,
                            doc_data['filename'],
                            doc_data['cdn_url'],
                            project_id,
                            content_hash=content_hash,
                        )
                    if parse_result.get('success'):
                        logger.info(
                            f"✅ [DOC-{short_id}] Legacy fallback recovered the document "
                            f"({len(parse_result.get('chunks', []))} chunks, no hierarchy)"
                        )
            else:
                parse_result = await _parse_document_async(
                    doc_id,
                    doc_data['filename'],
                    doc_data['cdn_url'],
                    project_id,
                    content_hash=content_hash,
                )

        if not parse_result.get('success'):
            timings.log()
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'FAILED',
                'error': parse_result.get('error', 'Parsing failed'),
                'chunks_created': 0
            }

        chunks = parse_result['chunks']
        chunks_metadata = parse_result.get('metadatas', [])
        logger.info(f"✅ [DOC-{short_id}] Parsed {len(chunks)} chunks")
        await publish_ingest_progress(
            chat_session_id, doc_id, IngestStage.PARSED, filename=filename
        )

        # LightRAG hook (disabled)
        if USE_LIGHTRAG_INTEGRATION:
            lightrag_client.insert_document_into_kg(doc_id=doc_id, chunks=chunks)

        # ——— 2a. TOC EXTRACTION (hierarchical path only) ——————————————————————
        # Hoisted ahead of blurb generation: the TOC is the document skeleton
        # that _build_doc_synopsis uses in place of the old full-text dump.
        toc = None
        raw_doc = parse_result.get('raw_doc')
        if used_hierarchical and raw_doc is not None:
            try:
                from utils.document_loaders.docling_loader import extract_toc_from_docling_doc
                with timings.phase("toc"):
                    toc = await asyncio.to_thread(extract_toc_from_docling_doc, raw_doc)
            except Exception as toc_err:
                logger.warning(f"⚠️ TOC extraction failed: {toc_err}")

        # ——— 2b. CONTEXTUAL BLURBS (hierarchical path only) ——————————————————
        # Must run before embedding: table chunks embed their LLM summary, not raw markdown.
        if used_hierarchical or USE_HIERARCHICAL_INGEST:
            await _checkpoint("blurb generation")
            logger.info(f"📝 [DOC-{short_id}] → BLURB GENERATION")
            await publish_ingest_progress(
                chat_session_id, doc_id, IngestStage.BLURBS, filename=filename
            )
            with timings.phase("blurbs"):
                chunks_metadata = await _generate_chunk_blurbs_async(
                    chunks, chunks_metadata, doc_id, filename=filename, toc=toc
                )

        # ——— 3. EMBEDDING Process, async with concurrency control ————————————————
        # Last checkpoint before we write vectors — past this point a cancel has
        # rows to clean up, which is exactly what cancel_speculative_upload does.
        await _checkpoint("embedding")
        logger.info(f"📋 [DOC-{short_id}] → EMBEDDING")
        await publish_ingest_progress(
            chat_session_id, doc_id, IngestStage.EMBEDDING, filename=filename
        )
        with timings.phase("embed"):
            embedding_result = await _process_embeddings_async(doc_id, project_id, chunks, chunks_metadata)

        if not embedding_result.get('success'):
            timings.log()
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'FAILED',
                'error': embedding_result.get('error', 'Embedding failed'),
                'chunks_created': len(chunks)
            }

        logger.info(f"✅ [DOC-{short_id}] Embedded {embedding_result['chunks_embedded']} chunks")

        # ——— 3b. POST-EMBED SUB-TASKS (hierarchical path only) ———————————————
        # Fired as fire-and-forget Celery tasks; never blocks note generation.
        # Gated on cancellation: these write document_sections / document_sources
        # rows, and dispatching them after a cancel would resurrect the document.
        if used_hierarchical and not await is_document_cancelled(doc_id):
            from tasks.hierarchical_ingest_tasks import (
                extract_doc_summary,
                extract_doc_concepts,
                build_section_summaries,
            )
            doc_sample = "\n\n".join(chunks[:15])[:8_000]
            extract_doc_summary.delay(doc_id, doc_sample)
            extract_doc_concepts.delay(doc_id, doc_sample)
            build_section_summaries.delay(doc_id, project_id)
            logger.info(f"🚀 [DOC-{short_id}] Post-embed sub-tasks dispatched")

            # Persist the TOC extracted above (best-effort)
            if toc:
                try:
                    async with get_db_connection() as conn:
                        await conn.execute(
                            "UPDATE document_sources SET toc = $1 WHERE id = $2",
                            json.dumps(toc), doc_id,
                        )
                    logger.info(f"📑 [DOC-{short_id}] TOC saved ({len(toc)} entries)")
                except Exception as toc_err:
                    logger.warning(f"⚠️ TOC save failed: {toc_err}")

        # ——— Return Success Result ———————————————————————————————————————————————
        timings.log()
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'COMPLETE',
            'chunks_created': embedding_result['chunks_embedded'],
            'total_tokens': embedding_result['total_tokens'],
            'processing_time_ms': int(timings.total_ms),
            'phase_timings_ms': timings.to_dict(),
        }

    except DocumentCancelledError as e:
        # Not a failure — the user asked for this. Return a distinct status so
        # finalize_batch_and_create_note doesn't count it as a failed document
        # (and so an all-cancelled batch doesn't persist a PARSE_ERROR note).
        logger.info(f"🚫 [DOC-{short_id}] {e}")
        timings.log()
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'CANCELLED',
            'chunks_created': 0
        }

    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Async processing failed: {e}")
        timings.log()
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'FAILED',
            'error': str(e),
            'chunks_created': 0
        }

async def _lightrag_document_processing_async(
    doc_id: str,
    doc_data: Dict[str, Any], 
    project_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    SIMPLIFIED enhanced document processing
    
    This would replace or supplement your existing function in upload_tasks.py
    """
    
    short_id = doc_id[:8]
    
    try:
        logger.info(f"🚀 [DOC-{short_id}] Starting enhanced processing with LightRAG")
        
        # ——— 1. Your Existing Document Processing ————————————————————————————————
        # Keep your existing parsing logic
        from upload_tasks import _parse_document_async
        parse_result = await _parse_document_async(doc_id, doc_data['filename'], doc_data['cdn_url'], project_id)
        
        if not parse_result.get('success'):
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'FAILED',
                'error': parse_result.get('error', 'Parsing failed'),
                'chunks_created': 0
            }
        
        chunks = parse_result['chunks']
        chunks_metadata = parse_result.get('metadatas', [])
        
        # ——— 2. Embeddings Processing ————————————————————————————————————————————
        logger.info(f"⚡ [DOC-{short_id}] Running embeddings processing")

        from upload_tasks import _process_embeddings_async

        embedding_task = asyncio.create_task(
            _process_embeddings_async(doc_id, project_id, chunks, chunks_metadata)
        )

        # LightRAG disabled — uncomment when re-enabling
        # lightrag_task = asyncio.create_task(
        #     lightrag_integration.enhance_document_processing(
        #         doc_id, chunks, chunks_metadata, project_id
        #     )
        # )

        # Wait for embedding to complete (LightRAG disabled)
        embedding_result = await embedding_task
        lightrag_result = {'success': False, 'entities_count': 0, 'relationships_count': 0}

        # ——— 3. Analyze Results ———————————————————————————————————————————————————
        final_status = 'COMPLETE'

        # Check embedding results
        if isinstance(embedding_result, Exception) or not embedding_result.get('success'):
            final_status = 'PARTIAL'
            logger.warning(f"⚠️ [DOC-{short_id}] Embedding processing failed")
        
        # ——— 4. Return Enhanced Results ————————————————————————————————————————————
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW_ENHANCED',
            'status': final_status,
            'chunks_created': embedding_result.get('chunks_embedded', 0) if not isinstance(embedding_result, Exception) else 0,
            'entities_extracted': lightrag_result.get('entities_count', 0) if not isinstance(lightrag_result, Exception) else 0,
            'relationships_extracted': lightrag_result.get('relationships_count', 0) if not isinstance(lightrag_result, Exception) else 0,
            'lightrag_enabled': not isinstance(lightrag_result, Exception) and lightrag_result.get('success', False)
        }
        
    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Enhanced processing failed: {e}")
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW_ENHANCED',
            'status': 'FAILED', 
            'error': str(e),
            'chunks_created': 0
        }
    
# ——— Edge Case Processing Tasks ———————————————————————————————————————

async def _handle_duplicate_only_batch(batch_id: str, project_id: str, workflow_metadata: Dict[str, Any], duplicate_docs: List[Dict]) -> Dict[str, Any]:
    """  
    [DUPLICATE HANDLER] Handle batches containing only duplicate documents:
    - Documents already EXIST in this project, so no processing needed (It;s a type of USER dumb error)
    - Can still generate notes using existing document content
    """
    # Remeber to update do the UI recieves a SUPABASE REALTRINE update
    _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_COMPLETE)
    logger.info(f"📋 [BATCH-{batch_id[:8]}] Handling duplicate-only batch with {len(duplicate_docs)} documents")
    
    if workflow_metadata.get('create_note'):
        # Enhance metadata for duplicate-only note generation
        note_metadata = {
            **workflow_metadata,
            'batch_status': 'DUPLICATE_ONLY',
            'duplicate_document_ids': [doc.get('existing_doc_id') for doc in duplicate_docs],
            'processing_context': f'All {len(duplicate_docs)} documents already exist in project'
        }
        
        logger.info(f"🎯 [BATCH-{batch_id[:8]}] Triggering note generation for duplicate-only batch")
        note_id = str(uuid.uuid4())
        supabase_client.table("notes").insert({
            "id":                   note_id,
            "user_id":              note_metadata["user_id"],
            "project_id":           note_metadata["project_id"],
            "title":                note_metadata["note_title"],
            "note_type":            note_metadata["note_type"],
            "note_progress_status": "INITIALIZED",
            "created_at":           datetime.now(timezone.utc).isoformat(),
        }).execute()
        rag_note_task.apply_async(kwargs={
            "note_id":    note_id,
            "user_id":    note_metadata["user_id"],
            "note_type":  note_metadata["note_type"],
            "project_id": note_metadata["project_id"],
            "note_title": note_metadata["note_title"],
            "provider":   note_metadata.get("provider"),
            "model_name": note_metadata.get("model_name"),
            "temperature": note_metadata.get("temperature"),
            "addtl_params": {
                **note_metadata.get("addtl_params", {}),
                'batch_context': {
                    'batch_id': batch_id,
                    'batch_status': 'DUPLICATE_ONLY',
                    'document_count': len(duplicate_docs),
                    'duplicate_doc_ids': note_metadata['duplicate_document_ids']
                }
            }
        })
    
    return {
        'batch_id': batch_id,
        'status': 'DUPLICATE_ONLY_COMPLETE',
        'duplicate_documents': len(duplicate_docs),
        'note_generation_triggered': workflow_metadata.get('create_note', False)
    }

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)  
def process_reused_document_task(
    self, 
    existing_doc_id: str, 
    doc_data: Dict[str, Any], 
    project_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [REUSED DOC PIPELINE] Smart reuse: copy existing embeddings to new project:
    - Create new document entry → Copy embeddings → Mark complete
        * New document entry is created from a reused document ♻️
        * Embeddings are copied from existing processed document ♻️ as a shortcut
    - Much faster than full processing pipeline
    - Returns processing results for workflow coordination
    """
    # Use the pre-generated UUID from speculative ingest if present so the
    # pre-created PENDING row is updated in place rather than creating an orphan.
    new_doc_id = workflow_metadata.get('speculative_doc_id') or str(uuid.uuid4())

    try:
        # ——— Create New Document Entry + Copy Embeddings ——————————————————————————
        # Use global pool instead of local pool
        pool = get_global_sync_db_pool()
        conn = pool.getconn()

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # FIND source document info for copying (including pre-computed metadata)
                cur.execute(
                    '''SELECT total_chunks, total_batches, doc_summary, doc_concepts
                       FROM document_sources WHERE id = %s''',
                    (existing_doc_id,)
                )
                source_info = cur.fetchone()

                if not source_info:
                    raise Exception(f"Source document {existing_doc_id} not found")

                # UPSERT: if this is a speculative doc the row already exists with
                # a placeholder content_hash and PENDING status — overwrite those fields.
                # doc_summary and doc_concepts are copied from the existing source so the
                # reused doc has them immediately without re-running LLM extraction tasks.
                cur.execute(
                    '''INSERT INTO document_sources
                    (id, cdn_url, content_hash, project_id, content_tags, uploaded_by,
                    vector_embed_status, filename, file_size_bytes, file_extension,
                    total_chunks, total_batches, created_at, processing_metadata,
                    doc_summary, doc_concepts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content_hash        = EXCLUDED.content_hash,
                        vector_embed_status = EXCLUDED.vector_embed_status,
                        total_chunks        = EXCLUDED.total_chunks,
                        total_batches       = EXCLUDED.total_batches,
                        processing_metadata = EXCLUDED.processing_metadata,
                        doc_summary         = EXCLUDED.doc_summary,
                        doc_concepts        = EXCLUDED.doc_concepts''',
                    (new_doc_id, doc_data['cdn_url'], doc_data['content_hash'],
                    project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                    ProcessingStatus.COMPLETE.value, doc_data['filename'],
                    doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                    source_info['total_chunks'], source_info['total_batches'],
                    datetime.now(timezone.utc), Json(workflow_metadata),
                    source_info.get('doc_summary'), source_info.get('doc_concepts'))
                )
                conn.commit()
        finally:
            pool.putconn(conn)
        
        # ——— Copy Embeddings (reuse existing sync function) ————————————————————————
        copy_result = copy_embeddings_for_project_sync(
            existing_doc_id, 
            new_doc_id, 
            project_id, 
            workflow_metadata['user_id']
        )
        
        logger.info(
            f"♻️ [DOC-{new_doc_id[:8]}] Smart reuse complete: "
            f"{copy_result['copied_count']} chunks, "
            f"{copy_result['sections_copied']} sections"
        )
        return {
            'doc_id': new_doc_id,
            'processing_type': 'REUSED',
            'status': 'COMPLETE',
            'chunks_reused': copy_result['copied_count'],
            'tokens_reused': copy_result['total_tokens'],
            'sections_copied': copy_result['sections_copied'],
        }
        
    except Exception as e:
        logger.error(f"❌ [DOC-{new_doc_id[:8]}] Reused document processing failed: {e}", exc_info=True)
        return {
            'doc_id': new_doc_id,
            'processing_type': 'REUSED',
            'status': 'FAILED', 
            'error': str(e)
        }


# ——— Module Exports ————————————————————————————————————————————————————————————————

__all__ = [
    # ——— Main Workflow Tasks ———————————————————————————————————————————————————————
    'process_document_batch_workflow',           # NEW: Main entry point (replaces process_document_task)
    'process_new_document_wrapper',              # NEW: Handle new documents in workflow
    'process_reused_document_task',              # NEW: Handle reused documents in workflow
    'finalize_batch_and_create_note',            # NEW: Batch coordination & note triggering
    'handle_batch_failure',                      # NEW: Batch failure handling
    
    # ——— Legacy/Individual Document Tasks ——————————————————————————————————————————
    'parse_document_task',                       # KEEP: Still used for individual parsing
    'embed_batch_task',                          # KEEP: Core embedding functionality
    # 'finalize_embeddings',                       # LEGACY: May be removed (replaced by batch finalization)
    
    # ——— Helper Functions ——————————————————————————————————————————————————————————
    'copy_embeddings_for_project_sync',         # KEEP: Used by reused document processing
    '_execute_batch_workflow',                   # NEW: Core workflow execution logic
    '_analyze_download_and_store_document_for_workflow',            # NEW: Document classification
    '_parse_document_for_workflow',       # NEW: Workflow-optimized parsing
    '_handle_duplicate_only_batch',              # NEW: Handle all-duplicate scenarios
    
    # ——— System Management Tasks ———————————————————————————————————————————————————
    # 'test_celery_log_task',                      # KEEP: Testing functionality
    'system_health_check',                       # KEEP: If you have this
    'cleanup_orphaned_resources',                # KEEP: If you have this
    'optimize_embedding_performance',            # KEEP: If you have this
    'initialize_production_pipeline',            # KEEP: If you have this
    'validate_production_readiness'              # KEEP: If you have this
]
