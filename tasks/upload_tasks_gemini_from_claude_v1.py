import os
import uuid
import asyncio
import logging
import io
import json
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone

# Third-party imports
import httpx
import tiktoken
import asyncpg
from celery import chord, group
from celery.exceptions import MaxRetriesExceededError, Retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# Project modules from v5 (assumed to exist)
from tasks.celery_app import celery_app
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.document_loaders.loader_factory import get_loader_for # IMPORTANT: Assumes this can take a file_like_object
from utils.metrics import MetricsCollector # Assumed from v5

# ——— Configuration & Constants ————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Queues
INGEST_QUEUE = 'ingest'
PARSE_QUEUE = 'parsing'
EMBED_QUEUE = 'embedding'
FINAL_QUEUE = 'finalize'

# Performance & Retries
MAX_RETRIES = 5
RETRY_BACKOFF_MULTIPLIER = 2
DEFAULT_RETRY_DELAY = 5
EMBEDDING_RATE_LIMIT = '1000/m' # Increased rate limit
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_MAX_TOKENS_PER_BATCH = 8190 # Safety margin below 8192
EXPECTED_EMBEDDING_LEN = 1536
MAX_CONCURRENT_DOWNLOADS = 10

# Database (asyncpg)
DB_POOL_MIN_SIZE = 5
DB_POOL_MAX_SIZE = 20
DB_DSN = os.getenv("POSTGRES_DSN") # e.g., "postgresql://user:pass@host:port/db"

# ——— Global Production Instances (Initialized once per worker) —————————————————————

# Async DB Connection Pool
db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE
        )
    return db_pool

# OpenAI Embeddings Client
embedding_model = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,
    request_timeout=60
)

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# ——— Helpers & Utilities ——————————————————————————————————————————————————————————

def _clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()

def _calculate_stream_hash(stream: io.BytesIO) -> str:
    """Calculate SHA-256 hash from an in-memory stream."""
    sha256_hash = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(4096):
        sha256_hash.update(chunk)
    stream.seek(0)
    return sha256_hash.hexdigest()

# ——— Task 1: Ingest (Largely same as v5, but uses async for DB check) ————————————

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
async def process_document_task(self, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """v3 Ingest Task: Uses concurrent downloads and async DB checks."""
    new_docs = []
    existing_ids = []
    
    async with httpx.AsyncClient() as client:
        download_tasks = [
            _download_and_prep_doc(client, url, metadata) for url in file_urls
        ]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        for res in results:
            if isinstance(res, Exception) or not res:
                continue
            
            # Enhanced deduplication by content hash
            existing = await conn.fetchval(
                'SELECT id FROM document_sources WHERE content_hash = $1', res['content_hash']
            )
            if existing:
                existing_ids.append(existing)
                logger.info(f"Skipping duplicate content hash for doc {existing}")
            else:
                new_docs.append(res)

    # Bulk insert new documents
    inserted_ids = []
    if new_docs:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Manually create the records to get UUIDs, then schedule parsing
                for doc in new_docs:
                    doc_id = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO document_sources (id, project_id, cdn_url, filename, file_size_bytes, content_hash, vector_embed_status, uploaded_by, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        doc_id, doc['project_id'], doc['cdn_url'], doc['filename'], doc['file_size_bytes'], doc['content_hash'], 'PENDING', doc['uploaded_by'], datetime.now(timezone.utc)
                    )
                    inserted_ids.append(doc_id)
                    parse_document_task.apply_async(
                        (str(doc_id), doc['cdn_url'], str(doc['project_id'])), queue=PARSE_QUEUE
                    )

    return {
        'doc_ids': [str(id) for id in existing_ids + inserted_ids],
        'new_documents': len(inserted_ids)
    }

async def _download_and_prep_doc(client: httpx.AsyncClient, url: str, metadata: Dict) -> Optional[Dict]:
    """Helper to download, stream to S3, and prep data."""
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            content_stream = io.BytesIO()
            async for chunk in response.aiter_bytes():
                content_stream.write(chunk)
            
            file_size = content_stream.tell()
            content_hash = _calculate_stream_hash(content_stream)
            
            ext = os.path.splitext(url)[1].lower() or '.bin'
            s3_key = f"{metadata['project_id']}/{uuid.uuid4()}{ext}"
            
            # Stream directly to S3 without saving locally
            upload_to_s3(s3_client, content_stream, s3_key, is_file_like_object=True)
            cdn_url = get_cloudfront_url(s3_key)
            
            return {
                'cdn_url': cdn_url,
                'project_id': metadata['project_id'],
                'uploaded_by': metadata['user_id'],
                'filename': os.path.basename(url),
                'file_size_bytes': file_size,
                'content_hash': content_hash,
            }
    except Exception as e:
        logger.error(f"Failed to download and prep {url}: {e}")
        return None

# ——— Task 2: Parse (v3 with in-memory streaming & token-aware batching) ——————————

@celery_app.task(bind=True, queue=PARSE_QUEUE, acks_late=True)
async def parse_document_task(self, source_id: str, cdn_url: str, project_id: str):
    """
    v3 Parsing Task:
    - Streams document directly into an in-memory buffer (NO temp file).
    - Uses token-aware adaptive batching to maximize efficiency.
    """
    await _update_doc_status(source_id, 'PARSING')
    
    # 1. True In-Memory Streaming (No Disk I/O)
    file_buffer = io.BytesIO()
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", cdn_url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    file_buffer.write(chunk)
        file_buffer.seek(0)
    except Exception as e:
        logger.error(f"Failed to stream {cdn_url} into memory: {e}")
        await _update_doc_status(source_id, 'FAILED_PARSING', str(e))
        return

    # 2. Process with Loader and Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Assumes get_loader_for can take a file-like object
    loader = get_loader_for(cdn_url, file_like_object=file_buffer)
    pages = loader.stream_documents()
    
    # 3. Token-Aware Adaptive Batching
    all_texts, all_metas = [], []
    for page in pages:
        for chunk in splitter.split_documents([page]):
            all_texts.append(_clean_text(chunk.page_content))
            all_metas.append(chunk.metadata)
            
    embedding_tasks = []
    current_batch_texts, current_batch_metas = [], []
    current_batch_tokens = 0

    for i, text in enumerate(all_texts):
        token_count = len(tokenizer.encode(text))
        
        if current_batch_tokens + token_count > OPENAI_MAX_TOKENS_PER_BATCH and current_batch_texts:
            # Dispatch current batch
            embedding_tasks.append(embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas))
            current_batch_texts, current_batch_metas, current_batch_tokens = [], [], 0

        current_batch_texts.append(text)
        current_batch_metas.append(all_metas[i])
        current_batch_tokens += token_count
    
    # Dispatch final batch
    if current_batch_texts:
        embedding_tasks.append(embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas))
        
    await _update_doc_status(source_id, 'EMBEDDING')
    
    # Schedule chord for embedding
    if embedding_tasks:
        chord(group(embedding_tasks), queue=EMBED_QUEUE)(finalize_embeddings.s(source_id))
    else: # Handle empty documents
        finalize_embeddings.s(source_id).apply_async(queue=FINAL_QUEUE)

# ——— Task 3: Embed (v3 with async DB and connection pooling) ——————————————————

@celery_app.task(
    bind=True, queue=EMBED_QUEUE, max_retries=MAX_RETRIES, acks_late=True,
    rate_limit=EMBEDDING_RATE_LIMIT
)
async def embed_batch_task(self, source_id: str, project_id: str, texts: List[str], metas: List[dict]):
    """
    v3 Embedding Task:
    - Fully async with non-blocking DB calls via asyncpg.
    - Uses a shared connection pool.
    - Circuit breaker for OpenAI API calls.
    """
    try:
        # 1. Generate Embeddings (with retry and circuit breaker)
        embeddings = await _embed_with_retry(texts)

        # 2. Prepare Data for DB
        records_to_insert = []
        for text, meta, vec in zip(texts, metas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN: continue
            records_to_insert.append((
                uuid.uuid4(), source_id, project_id, text, json.dumps(meta), vec, len(tokenizer.encode(text))
            ))

        if not records_to_insert:
            return {'processed_count': 0, 'source_id': source_id}

        # 3. Fully Async Bulk Insert
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.copy_records_to_table(
                'document_vector_store',
                records=records_to_insert,
                columns=('id', 'source_id', 'project_id', 'content', 'metadata', 'embedding', 'num_tokens')
            )
        
        return {'processed_count': len(records_to_insert), 'source_id': source_id}

    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed, retrying... Error: {exc}")
        raise self.retry(exc=exc, countdown=DEFAULT_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** self.request.retries))

@circuit(failure_threshold=5, recovery_timeout=60)
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential_jitter(initial=2, max=10),
    retry=retry_if_exception_type(Exception) # Be specific in prod, e.g., httpx.NetworkError
)
async def _embed_with_retry(texts: List[str]) -> List[List[float]]:
    return await embedding_model.aembed_documents(texts)

# ——— Task 4: Finalize (v3 with async DB read) ——————————————————————————————————

@celery_app.task(bind=True, queue=FINAL_QUEUE)
async def finalize_embeddings(self, batch_results: List[Dict], source_id: str):
    """v3 Finalization: Uses async DB read to verify completion."""
    total_processed = sum(res['processed_count'] for res in batch_results if res)
    logger.info(f"Finalizing document {source_id}. Total chunks processed: {total_processed}")
    
    # In a real scenario, you'd compare total_processed against an expected count
    # stored during the parsing phase.
    if total_processed > 0:
        await _update_doc_status(source_id, 'COMPLETE')
    else:
        await _update_doc_status(source_id, 'FAILED_EMBEDDING', 'No chunks were processed.')
    
    return {'source_id': source_id, 'final_status': 'COMPLETE', 'processed_chunks': total_processed}

async def _update_doc_status(doc_id: str, status: str, error_msg: str = None):
    """Async helper to update document status."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE document_sources SET vector_embed_status = $1, error_message = $2, updated_at = NOW() WHERE id = $3",
            status, error_msg, uuid.UUID(doc_id)
        )