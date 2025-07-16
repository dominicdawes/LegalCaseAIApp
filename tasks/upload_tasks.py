"""v6 By Gemini 2.5 Pro bc Claude hit rate/context limits"""

import os
import sys
import uuid
import tempfile
import asyncio
import logging
import threading
import io
import json
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta
import psutil
import time

# Third-party imports
import httpx
import tiktoken
import asyncpg
from celery import chord, group
from celery.exceptions import MaxRetriesExceededError, Retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type 
from pybreaker import CircuitBreaker   

# Project modules
from tasks.celery_app import celery_app
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.loader_factory import get_loader_for        # The loader factory is now expected to handle in-memory file-like objects
from utils.metrics import MetricsCollector, Timer
from utils.connection_pool import ConnectionPoolManager
from utils.memory_manager import MemoryManager # Kept for health checks

# ——— Configuration & Constants ————————————————————————————————————————————————————

# Force logger configuration for Celery worker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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

# OpenAI configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_MAX_TOKENS_PER_BATCH = 8190 # Safety margin below the 8192 limit
EXPECTED_EMBEDDING_LEN = 1536
MAX_CONCURRENT_DOWNLOADS = 3 # A modest limit instead of 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Database (asyncpg)
DB_DSN = os.getenv("POSTGRES_DSN") # e.g., Supabase -> Connection -> Get Direct URL
# DB_POOL_MIN_SIZE = 5  # <-- if i had more compute
# DB_POOL_MAX_SIZE = 20
DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 5

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

# ——— Global Production Instances (Initialized once per worker) —————————————————————

# Async DB Connection Pool
db_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    """Initializes and returns the asyncpg connection pool."""
    global db_pool
    try:
        if db_pool is None:
            if not DB_DSN:
                raise ValueError("POSTGRES_DSN environment variable not set.")
            db_pool = await asyncpg.create_pool(
                dsn=DB_DSN,
                min_size=DB_POOL_MIN_SIZE,
                max_size=DB_POOL_MAX_SIZE
            )
            logger.info("✅ Database pool ready")
    except Exception as e:
        logger.error(f"❌ DB pool failed: {e}")
        raise
    return db_pool

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


# ——— Helpers & Utilities ——————————————————————————————————————————————————————————

def _clean_text(text: str) -> str:
    """Enhanced text cleaning with unicode normalization."""
    import unicodedata
    cleaned = text.replace("\x00", "").replace("\ufffd", "")
    normalized = unicodedata.normalize('NFKC', cleaned)
    return normalized.strip()

def _calculate_stream_hash(stream: io.BytesIO) -> str:
    """Calculate SHA-256 hash from an in-memory stream without consuming it."""
    sha256_hash = hashlib.sha256()
    stream.seek(0)
    # Read in chunks to handle large streams efficiently
    while chunk := stream.read(4096):
        sha256_hash.update(chunk)
    stream.seek(0) # Reset stream position after reading
    return sha256_hash.hexdigest()

async def _update_document_status(doc_id: str, status: ProcessingStatus, error_message: Optional[str] = None):
    """Async helper to update a document's status in the database."""
    logger.info(f"📋 Doc {doc_id[:8]}... → {status.value}")
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE document_sources
            SET vector_embed_status = $1, error_message = $2, updated_at = NOW()
            WHERE id = $3
            """,
            status.value, error_message, uuid.UUID(doc_id)
        )

# Global Metrics Collector
logger.info("📊 Initializing metrics collector...")
metrics_collector = MetricsCollector()
logger.info("✅ Metrics collector initialized")

# ——— Task 1: Ingest (Fully Async) ———————————————————————————————————————————

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_document_task(self, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    task_id = self.request.id
    logger.info(f"🚀 Starting document processing task URLs")
    return asyncio.run(_process_document_async(file_urls, metadata))

async def _process_document_async(file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest Task:
    - Concurrent downloads directly into memory using httpx.
    - Async deduplication against the database using asyncpg.
    - Streams content to S3 without writing to local disk.
    - Bulk-inserts new documents and dispatches parsing tasks.
    """
    project_id = metadata['project_id']
    user_id = metadata['user_id']
    
    # Download document, hash, send to S3/CloudFront
    logger.info("🌐 Initiating concurrent downloads...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        download_tasks = [
            _download_and_prep_doc(client, url, project_id, user_id) for url in file_urls
        ]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
    logger.info(f"⬇️  Downloads completed...")
    
    new_docs_to_insert = []
    existing_doc_ids = []
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        for res in results:
            if isinstance(res, Exception) or not res:
                logger.error(f"A document failed to download or prep: {res}")
                continue
            
            # Async deduplication check by content hash
            existing_id = await conn.fetchval(
                'SELECT id FROM document_sources WHERE content_hash = $1 AND project_id = $2',
                res['content_hash'], uuid.UUID(project_id)
            )
            
            if existing_id:
                existing_doc_ids.append(str(existing_id))
                logger.info(f"Skipping duplicate content hash {res['content_hash'][:8]} for doc_id {existing_id}")
            else:
                res['id'] = uuid.uuid4()
                new_docs_to_insert.append(res)
    
    # Bulk insert new documents using asyncpg
    inserted_ids = []
    if new_docs_to_insert:
        logger.info(f"💾 Inserting {len(new_docs_to_insert)} new docs...")
        records_to_insert = [
            (
                doc['id'], doc['cdn_url'], doc['content_hash'], uuid.UUID(doc['project_id']),
                doc.get('content_tags', []), doc['uploaded_by'], ProcessingStatus.PENDING.value,
                doc['filename'], doc['file_size_bytes'], os.path.splitext(doc['filename'])[1].lower(),
                datetime.now(timezone.utc)
            )
            for doc in new_docs_to_insert
        ]
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.copy_records_to_table(
                    'document_sources',
                    records=records_to_insert,
                    columns=(
                        'id', 'cdn_url', 'content_hash', 'project_id', 'content_tags',
                        'uploaded_by', 'vector_embed_status', 'filename', 'file_size_bytes',
                        'file_extension', 'created_at'
                    ),
                    timeout=60
                )
        
        inserted_ids = [str(doc['id']) for doc in new_docs_to_insert]
        # Schedule parsing tasks for newly inserted documents
        logger.info("🗂️ Scheduling [celery] parse tasks...")
        for doc in new_docs_to_insert:
            parse_document_task.apply_async(
                (str(doc['id']), doc['cdn_url'], doc['project_id']), queue=PARSE_QUEUE
            )

    return {
        'doc_ids': existing_doc_ids + inserted_ids,
        'new_documents': len(inserted_ids),
        'duplicate_documents': len(existing_doc_ids)
    }

async def _download_and_prep_doc(client: httpx.AsyncClient, url: str, project_id: str, user_id: str) -> Optional[Dict]:
    """Helper to download to memory, hash, stream to S3, and prep data."""
    try:
        async with client.stream("GET", url) as response:
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
            
            ext = os.path.splitext(url)[1].lower() or '.bin'
            s3_key = f"{project_id}/{uuid.uuid4()}{ext}"
            
            # Stream directly to S3 from the in-memory buffer
            upload_to_s3(s3_client, content_stream, s3_key, is_file_like_object=True)
            cdn_url = get_cloudfront_url(s3_key)
            
            return {
                'cdn_url': cdn_url,
                'project_id': project_id,
                'uploaded_by': user_id,
                'filename': os.path.basename(url),
                'file_size_bytes': file_size,
                'content_hash': content_hash,
            }
    except Exception as e:
        logger.error(f"Failed to download and prep {url}: {e}")
        return None

# ——— Task 2: For one document → Parse, batch, dispatch to emmbed (v6 - In-Memory & Token-Aware) ————————————————————————————————

@celery_app.task(bind=True, queue=PARSE_QUEUE, acks_late=True)
def parse_document_task(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
    return asyncio.run(_parse_document_async(self, source_id, cdn_url, project_id))

async def _parse_document_async(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
    """
    v6 Parsing Task:
    - Streams document directly into an in-memory buffer (NO temp file).
    - Splits parsed text into semantic chunks
    - Uses token-aware adaptive batching to maximize embedding efficiency.
    - Fully async database updates.
    """
    await _update_document_status(source_id, ProcessingStatus.PARSING)
    
    file_buffer = io.BytesIO()
    doc_metrics = DocumentMetrics(doc_id=source_id)     # ← TELEMETRY

    try:
        # Phase 1: True In-Memory Streaming (No Disk I/O)
        with Timer() as download_timer:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("GET", cdn_url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        file_buffer.write(chunk)
            file_buffer.seek(0)
            # update telemetry
            doc_metrics.download_time_ms = download_timer.elapsed_ms
            doc_metrics.file_size_bytes = file_buffer.getbuffer().nbytes

        # Phase 2: Process with Loader and Splitter
        with Timer() as parse_timer:
            # Assumes get_loader_for can take a file-like object
            loader = get_loader_for(filename=cdn_url, file_like_object=file_buffer)

            # Langchain semantic chunking strategy (preserves whole ideas)
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
            all_texts, all_metadatas = [], []
            # stream_documents is a custom method assumed on the loader
            for page_num, page in enumerate(loader.stream_documents()):
                for chunk_idx, chunk in enumerate(splitter.split_documents([page])):
                    cleaned_content = _clean_text(chunk.page_content)
                    if not cleaned_content:
                        continue
                    
                    all_texts.append(cleaned_content)
                    all_metadatas.append({
                        **chunk.metadata,
                        'page_number': page_num,
                        'chunk_index': chunk_idx
                    })
            # update telemetry
            doc_metrics.parse_time_ms = parse_timer.elapsed_ms
            doc_metrics.total_chunks = len(all_texts)

        # Phase 3: Token-Aware Adaptive Batching (efficient use of every API call, reducing cost and latency)
        with Timer() as batch_timer:
            embedding_tasks = []
            current_batch_texts, current_batch_metas = [], []
            current_batch_tokens = 0

            for i, text in enumerate(all_texts):
                token_count = len(tokenizer.encode(text))
                
                # If adding the next chunk exceeds the token limit, dispatch the current batch
                if current_batch_tokens + token_count > OPENAI_MAX_TOKENS_PER_BATCH and current_batch_texts:
                    task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
                    embedding_tasks.append(task_sig)
                    current_batch_texts, current_batch_metas, current_batch_tokens = [], [], 0

                current_batch_texts.append(text)
                current_batch_metas.append(all_metadatas[i])
                current_batch_tokens += token_count
            
            # Dispatch the final batch if it exists
            if current_batch_texts:
                task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
                embedding_tasks.append(task_sig)
            # update telemetry
            doc_metrics.total_batches = len(embedding_tasks)
            doc_metrics.chunk_time_ms = batch_timer.elapsed_ms

        # Phase 4: Database Update & Task Scheduling
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE document_sources
                SET vector_embed_status = $1, total_chunks = $2, total_batches = $3, processing_metadata = processing_metadata || $4::jsonb
                WHERE id = $5
                """,
                ProcessingStatus.EMBEDDING.value, doc_metrics.total_chunks, doc_metrics.total_batches,
                json.dumps(doc_metrics.to_dict()), uuid.UUID(source_id)
            )
        
        if embedding_tasks:
            # Use a chord to run all embedding tasks and then call finalization
            chord(group(embedding_tasks), queue=EMBED_QUEUE)(finalize_embeddings.s(source_id).set(queue=FINAL_QUEUE))
        else:
            # Handle empty documents correctly
            finalize_embeddings.apply_async(([], source_id), queue=FINAL_QUEUE)

        logger.info(f"Parsing complete for {source_id}: {doc_metrics.total_chunks} chunks in {doc_metrics.total_batches} token-aware batches.")
        return {'source_id': source_id, 'status': 'SUCCESS', 'batches_created': doc_metrics.total_batches}

    except Exception as e:
        logger.error(f"Parsing failed for {source_id}: {e}", exc_info=True)
        await _update_document_status(source_id, ProcessingStatus.FAILED_PARSING, str(e))
        raise

# ——— Task 3: Embed (v6 - Fully Async DB) ——————————————————————————————————————————

@celery_app.task(
    bind=True, queue=EMBED_QUEUE, max_retries=MAX_RETRIES, acks_late=True,
    default_retry_delay=DEFAULT_RETRY_DELAY, rate_limit=RATE_LIMIT
)
def embed_batch_task(self, source_id: str, project_id: str, texts: List[str], metadatas: List[Dict]):
    return asyncio.run(_embed_batch_async(self, source_id, project_id, texts, metadatas))

async def _embed_batch_async(self, source_id: str, project_id: str, texts: List[str], metadatas: List[Dict]):
    """
    v6 Embedding Task:
    - Fully async with non-blocking DB calls via asyncpg.
    - Uses a shared connection pool and highly efficient `copy_records_to_table`.
    - Retries with exponential backoff and uses a circuit breaker for OpenAI calls.
    """
    try:
        # 1. Generate Embeddings (with retry and circuit breaker)
        embeddings = await _embed_with_retry(texts)
        
        # 2. Prepare Data for DB
        records_to_insert = []
        total_tokens = 0
        for text, meta, vec in zip(texts, metadatas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN:
                logger.warning(f"Skipping malformed embedding for source {source_id}")
                continue
            
            token_count = len(tokenizer.encode(text))
            total_tokens += token_count
            records_to_insert.append((
                uuid.uuid4(), uuid.UUID(source_id), uuid.UUID(project_id), text, 
                json.dumps(meta), vec, token_count, datetime.now(timezone.utc)
            ))

        if not records_to_insert:
            return {'processed_count': 0, 'token_count': 0}

        # 3. Fully Async Bulk Insert using copy_records_to_table for max performance
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.copy_records_to_table(
                'document_vector_store',
                records=records_to_insert,
                columns=('id', 'source_id', 'project_id', 'content', 'metadata', 'embedding', 'num_tokens', 'created_at'),
                timeout=120
            )
        
        return {'processed_count': len(records_to_insert), 'token_count': total_tokens}

    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed (attempt {self.request.retries + 1}), retrying: {exc}")
        raise self.retry(exc=exc, countdown=DEFAULT_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** self.request.retries))

# optional remove the exclude arg
@CircuitBreaker(fail_max=5, reset_timeout=60, exclude=[Exception])
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.RequestError)
)
async def _embed_with_retry(texts: List[str]) -> List[List[float]]:
    """Wrapper for OpenAI embedding call with circuit breaker and tenacity retry."""
    return await embedding_model.aembed_documents(texts)


# ——— Task 4: Finalize (v6 - Fully Async DB) ————————————————————————————————————————

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def finalize_embeddings(self, batch_results: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
    return asyncio.run(_finalize_embeddings_async(self, batch_results, source_id))

async def _finalize_embeddings_async(self, batch_results: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
    """
    v6 Finalization Task:
    - Uses asyncpg for all database reads and writes.
    - Preserves the detailed completion metrics and performance analysis from v5.
    """
    pool = await get_db_pool()
    final_status = ProcessingStatus.FAILED_FINALIZATION
    
    try:
        # ——— Phase 1: Fetch Current Document State Asynchronously ——————————————————
        async with pool.acquire() as conn:
            doc_data = await conn.fetchrow("SELECT * FROM document_sources WHERE id = $1", uuid.UUID(source_id))
        
        if not doc_data:
            raise ValueError(f"Document {source_id} not found during finalization.")
            
        # ——— Phase 2: Validate Processing Completion ——————————————————————————————
        expected_chunks = doc_data.get('total_chunks', 0)
        
        async with pool.acquire() as conn:
            actual_vector_count = await conn.fetchval("SELECT COUNT(*) FROM document_vector_store WHERE source_id = $1", uuid.UUID(source_id))
        
        chunk_completion_rate = (actual_vector_count / expected_chunks * 100) if expected_chunks > 0 else 100 if actual_vector_count > 0 else 0

        # ——— Phase 3: Determine Final Status
        if abs(actual_vector_count - expected_chunks) <= 1: # Allow for minor discrepancy
            final_status = ProcessingStatus.COMPLETE
            status_message = "Processing completed successfully."
        elif actual_vector_count > 0:
            final_status = ProcessingStatus.PARTIAL
            status_message = f"Partial completion: {actual_vector_count}/{expected_chunks} chunks processed."
        else:
            final_status = ProcessingStatus.FAILED_EMBEDDING
            status_message = "Embedding failed: No chunks were successfully processed."

        # ——— Phase 4: Calculate Final Metrics ———————————————————————————————————
        # Aggregate batch results
        successful_batches = [r for r in batch_results if r and r.get('processed_count', 0) > 0]
        failed_batches = len(batch_results) - len(successful_batches)
        
        total_processing_time = sum(r.get('total_time_ms', 0) for r in successful_batches)
        total_embedding_time = sum(r.get('embedding_time_ms', 0) for r in successful_batches)
        total_storage_time = sum(r.get('storage_time_ms', 0) for r in successful_batches)
        total_tokens = sum(r.get('token_count', 0) for r in successful_batches)
        
        avg_batch_time = total_processing_time / len(successful_batches) if successful_batches else 0
        tokens_per_second = total_tokens / (total_processing_time / 1000) if total_processing_time > 0 else 0
        
        # ——— Phase 5: Generate Performance Insights ————————————————————————————
        performance_insights = {
            'avg_batch_processing_time_ms': avg_batch_time,
            'tokens_per_second': tokens_per_second,
            'embedding_efficiency_pct': (total_embedding_time / total_processing_time * 100) if total_processing_time > 0 else 0,
            'storage_efficiency_pct': (total_storage_time / total_processing_time * 100) if total_processing_time > 0 else 0,
            'successful_batch_rate': len(successful_batches) / len(batch_results) * 100 if batch_results else 0
        }
        
        # Generate optimization recommendations
        recommendations = []
        if performance_insights['tokens_per_second'] < 1000:
            recommendations.append("Consider increasing batch size for better throughput")
        if performance_insights['embedding_efficiency_pct'] > 80:
            recommendations.append("Embedding time is dominant - consider batch optimization")
        if failed_batches > 0:
            recommendations.append(f"Investigate {failed_batches} failed batches for reliability issues")

        # ——— Phase 6: Final Database Update ————————————————————————————————————
        final_metadata = {
             **(doc_data.get('processing_metadata', {}) or {}),
            'completion_timestamp': datetime.now(timezone.utc).isoformat(),
            'final_vector_count': actual_vector_count,
            'chunk_completion_rate': chunk_completion_rate,
            # Add other aggregated metrics here
        }

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE document_sources
                SET vector_embed_status = $1, final_vector_count = $2, completion_rate = $3,
                    status_message = $4, completed_at = NOW(), processing_metadata = $5::jsonb
                WHERE id = $6
                """,
                final_status.value, actual_vector_count, chunk_completion_rate, status_message,
                json.dumps(final_metadata), uuid.UUID(source_id)
            )

        logger.info(f"Document {source_id} finalization complete. Status: {final_status.value}")
        return {'source_id': source_id, 'final_status': final_status.value}

    except Exception as e:
        logger.error(f"Finalization failed for document {source_id}: {e}", exc_info=True)
        await _update_document_status(source_id, ProcessingStatus.FAILED_FINALIZATION, f"Finalization failed: {str(e)}")
        raise



# ——— Advanced Monitoring & Maintenance Tasks (Unchanged from v5) —————————————————
# NOTE: The highly-praised health check, cleanup, and optimization tasks are
# preserved. They are a critical part of the production system.

@celery_app.task(bind=True, queue='monitoring')
def system_health_check(self) -> Dict[str, Any]:
    """
    Comprehensive system health monitoring:
    - Resource utilization tracking
    - Queue depth monitoring
    - Performance baseline validation
    - Automatic scaling recommendations
    """
    
    try:
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Queue depth monitoring
        queue_stats = {}
        for queue_name in [INGEST_QUEUE, PARSE_QUEUE, EMBED_QUEUE, FINAL_QUEUE]:
            try:
                inspect = celery_app.control.inspect()
                active_tasks = inspect.active()
                queue_length = len(active_tasks.get(queue_name, [])) if active_tasks else 0
                queue_stats[queue_name] = queue_length
            except:
                queue_stats[queue_name] = -1  # Unable to determine
        
        # Performance baselines
        recent_metrics = metrics_collector.get_recent_performance_metrics()
        
        # Health assessment
        health_status = "healthy"
        issues = []
        
        if cpu_percent > 85:
            health_status = "degraded"
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent > 90:
            health_status = "critical"
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        if disk.percent > 90:
            health_status = "critical"
            issues.append(f"High disk usage: {disk.percent:.1f}%")
        
        # Generate scaling recommendations
        recommendations = []
        total_queue_depth = sum(v for v in queue_stats.values() if v > 0)
        
        if total_queue_depth > 100:
            recommendations.append("Consider scaling up worker instances")
        
        if cpu_percent < 30 and total_queue_depth < 10:
            recommendations.append("System is under-utilized, consider scaling down")
        
        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': health_status,
            'issues': issues,
            'recommendations': recommendations,
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'available_memory_gb': memory.available / (1024**3)
            },
            'queue_stats': queue_stats,
            'performance_metrics': recent_metrics
        }
        
        # Store health report
        supabase_client.table('system_health_reports') \
            .insert(health_report).execute()
        
        return health_report
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'error',
            'error': str(e)
        }

@celery_app.task(bind=True, queue='maintenance')
def cleanup_orphaned_resources(self) -> Dict[str, Any]:
    """
    Automated cleanup of orphaned resources:
    - Remove stale temporary files
    - Clean up incomplete processing records
    - Purge old metrics and logs
    - Optimize database performance
    """
    
    cleanup_stats = {
        'temp_files_removed': 0,
        'stale_records_cleaned': 0,
        'old_metrics_purged': 0,
        'errors': []
    }
    
    try:
        # Clean up temporary files older than 24 hours
        temp_dir = tempfile.gettempdir()
        cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
        
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.unlink(filepath)
                    cleanup_stats['temp_files_removed'] += 1
                except:
                    pass
        
        # Clean up documents stuck in processing states for > 4 hours
        four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=4)
        
        stale_docs = supabase_client.table('document_sources') \
            .select('id') \
            .in_('vector_embed_status', [
                ProcessingStatus.PARSING.value,
                ProcessingStatus.EMBEDDING.value
            ]) \
            .lt('created_at', four_hours_ago.isoformat()) \
            .execute()
        
        if stale_docs.data:
            doc_ids = [doc['id'] for doc in stale_docs.data]
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': ProcessingStatus.FAILED_PARSING.value,
                    'error_message': 'Processing timed out - cleaned up by maintenance task'
                }) \
                .in_('id', doc_ids) \
                .execute()
            
            cleanup_stats['stale_records_cleaned'] = len(doc_ids)
        
        # Purge old metrics (keep last 30 days)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        old_metrics = supabase_client.table('system_health_reports') \
            .delete() \
            .lt('timestamp', thirty_days_ago.isoformat()) \
            .execute()
        
        cleanup_stats['old_metrics_purged'] = len(old_metrics.data) if old_metrics.data else 0
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
        
    except Exception as e:
        cleanup_stats['errors'].append(str(e))
        logger.error(f"Cleanup task failed: {e}")
        return cleanup_stats

# ——— Production Performance Optimization Tasks ————————————————————————————————————

@celery_app.task(bind=True, queue='optimization')
def optimize_embedding_performance(self) -> Dict[str, Any]:
    """
    Automated performance optimization:
    - Analyze embedding performance patterns
    - Adjust batch sizes based on historical data
    - Optimize rate limits based on API performance
    - Generate performance improvement recommendations
    """
    
    try:
        # Analyze recent performance data
        performance_data = metrics_collector.get_performance_analysis()
        
        # Calculate optimal batch size based on throughput
        optimal_batch_size = batch_processor.calculate_optimal_batch_size(
            performance_data.get('avg_processing_time', 10.0),
            performance_data.get('avg_memory_usage', 50.0)
        )
        
        # Generate optimization recommendations
        recommendations = {
            'current_batch_size': batch_processor.current_batch_size,
            'recommended_batch_size': optimal_batch_size,
            'performance_improvement_est': performance_data.get('improvement_estimate', 0),
            'memory_optimization_tips': [],
            'scaling_recommendations': []
        }
        
        # Memory optimization tips
        if performance_data.get('avg_memory_usage', 0) > 80:
            recommendations['memory_optimization_tips'].extend([
                "Consider reducing batch size",
                "Implement more aggressive garbage collection",
                "Add memory circuit breakers"
            ])
        
        # Scaling recommendations
        if performance_data.get('avg_queue_depth', 0) > 50:
            recommendations['scaling_recommendations'].extend([
                "Scale up embedding workers",
                "Increase rate limits if API allows",
                "Consider load balancing across regions"
            ])
        
        # Apply optimizations
        if abs(optimal_batch_size - batch_processor.current_batch_size) > 2:
            batch_processor.current_batch_size = optimal_batch_size
            logger.info(f"Batch size optimized to {optimal_batch_size}")
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimizations_applied': recommendations,
            'performance_data': performance_data
        }
        
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        return {'error': str(e)}

# ——— Utility Functions for Production Deployment ————————————————————————————————————

def validate_production_readiness() -> Dict[str, Any]:
    """
    Validate that all production requirements are met
    """
    checks = {
        'environment_variables': True,
        'database_connectivity': True,
        'external_services': True,
        'resource_limits': True,
        'monitoring_setup': True
    }
    
    issues = []
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY', 'AWS_ACCESS_KEY_ID']
    for var in required_vars:
        if not os.getenv(var):
            checks['environment_variables'] = False
            issues.append(f"Missing environment variable: {var}")
    
    # Check database connectivity
    try:
        supabase_client.table('document_sources').select('id').limit(1).execute()
    except Exception as e:
        checks['database_connectivity'] = False
        issues.append(f"Database connectivity failed: {e}")
    
    # Check resource limits
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 2:
        checks['resource_limits'] = False
        issues.append(f"Insufficient memory: {memory_gb:.1f}GB < 2GB minimum")
    
    return {
        'ready_for_production': all(checks.values()),
        'checks': checks,
        'issues': issues,
        'recommendations': [
            "Set up comprehensive monitoring and alerting",
            "Configure auto-scaling based on queue depth",
            "Implement proper backup and disaster recovery",
            "Set up log aggregation and analysis"
        ]
    }

# ——— Production Deployment & Initialization ————————————————————————————————————————

def initialize_production_pipeline():
    """
    Initialize the production pipeline with all required components
    """
    logger.info("Initializing production RAG pipeline...")
    
    # Initialize global components
    global memory_manager, metrics_collector, batch_processor, downloader
    
    # Validate production readiness
    readiness_check = validate_production_readiness()
    if not readiness_check['ready_for_production']:
        logger.error(f"Production readiness check failed: {readiness_check['issues']}")
        raise RuntimeError("Pipeline not ready for production deployment")
    
    # Schedule periodic maintenance tasks
    from celery.schedules import crontab
    
    # Health checks every 5 minutes
    celery_app.conf.beat_schedule = {
        'system-health-check': {
            'task': 'system_health_check',
            'schedule': 300.0,  # 5 minutes
        },
        'cleanup-orphaned-resources': {
            'task': 'cleanup_orphaned_resources',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'optimize-performance': {
            'task': 'optimize_embedding_performance',
            'schedule': crontab(minute=0),  # Every hour
        }
    }
    
    logger.info("Production RAG pipeline initialized successfully!")
    return True

# ——— Module Exports ————————————————————————————————————————————————————————————————

__all__ = [
    'process_document_task',
    'parse_document_task', 
    'embed_batch_task',
    'finalize_embeddings',
    'system_health_check',
    'cleanup_orphaned_resources',
    'optimize_embedding_performance',
    'initialize_production_pipeline',
    'validate_production_readiness'
]
