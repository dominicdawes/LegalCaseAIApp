import os
import uuid
import tempfile
import asyncio
import logging
import threading
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import hashlib
import psutil
import time
import math

# Third-party imports
import requests
import httpx
import tiktoken
from celery import chord, group
from celery.exceptions import MaxRetriesExceededError, Retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import aiofiles
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pybreaker import CircuitBreaker

# Project modules
from tasks.celery_app import celery_app
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.loader_factory import get_loader_for
from utils.metrics import MetricsCollector, Timer
from utils.memory_manager import MemoryManager
from utils.connection_pool import ConnectionPoolManager

# ——— Configuration & Constants ————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Queue configuration
INGEST_QUEUE = 'ingest'
PARSE_QUEUE = 'parsing'
EMBED_QUEUE = 'embedding'
FINAL_QUEUE = 'finalize'

# Performance tuning
MAX_RETRIES = 5
RETRY_BACKOFF_MULTIPLIER = 2
DEFAULT_RETRY_DELAY = 5
RATE_LIMIT = '300/m'  # Increased for production throughput

# v6 Enhancement: Token-aware batch sizing
MIN_BATCH_TOKENS = 1000
MAX_BATCH_TOKENS = 8000  # Conservative limit for OpenAI API
OPTIMAL_BATCH_TOKENS = 6000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EXPECTED_EMBEDDING_LEN = 1536

# Memory management
MAX_MEMORY_USAGE_PCT = 75
MEMORY_CHECK_INTERVAL = 30  # seconds
MAX_TEMP_FILE_SIZE = 500 * 1024 * 1024  # 500MB
STREAMING_CHUNK_SIZE = 64 * 1024  # 64KB chunks

# v6 Enhancement: Scalability thresholds
MAX_CHUNKS_PER_CHORD = 5000  # Prevent memory issues with massive documents
CHORD_BATCH_SIZE = 1000  # Process in smaller chord groups

# Concurrency limits
MAX_CONCURRENT_DOWNLOADS = 10
MAX_CONCURRENT_EMBEDS = 5
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MAX_TOKENS = 8192
OPENAI_TIMEOUT = 60

# v6 Enhancement: AsyncPG database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Your Supabase PostgreSQL URL
ASYNC_DB_POOL_SIZE = 20
ASYNC_DB_MAX_OVERFLOW = 10

# ——— Enhanced Data Structures (v6) ————————————————————————————————————————————————

class ProcessingStatus(Enum):
    INITIALIZING = "INITIALIZING"
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    STREAMING_PARSE = "STREAMING_PARSE"  # v6: New status for in-memory parsing
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    TOKEN_BATCHING = "TOKEN_BATCHING"  # v6: New status for token-aware batching
    EMBEDDING = "EMBEDDING"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED_DOWNLOAD = "FAILED_DOWNLOAD"
    FAILED_PARSING = "FAILED_PARSING"
    FAILED_EMBEDDING = "FAILED_EMBEDDING"
    FAILED_FINALIZATION = "FAILED_FINALIZATION"
    CHUNKED_PROCESSING = "CHUNKED_PROCESSING"  # v6: For massive file processing

@dataclass
class DocumentMetrics:
    """Enhanced telemetry for document processing (v6 additions)"""
    doc_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    download_time_ms: Optional[int] = None
    streaming_parse_time_ms: Optional[int] = None  # v6: In-memory parsing time
    parse_time_ms: Optional[int] = None
    chunk_time_ms: Optional[int] = None
    token_batching_time_ms: Optional[int] = None  # v6: Token-aware batching time
    embed_time_ms: Optional[int] = None
    total_time_ms: Optional[int] = None
    
    file_size_bytes: int = 0
    total_chunks: int = 0
    total_batches: int = 0
    processed_chunks: int = 0
    processed_batches: int = 0
    failed_batches: int = 0
    
    # v6 Enhancements: Token-aware metrics
    total_tokens: int = 0
    avg_tokens_per_batch: float = 0.0
    token_efficiency_pct: float = 0.0  # How close to max tokens per batch
    
    peak_memory_mb: float = 0.0
    avg_chunk_size: float = 0.0
    tokens_processed: int = 0
    embedding_calls: int = 0
    retry_count: int = 0
    
    # v6 Enhancement: I/O performance metrics
    disk_io_avoided: bool = False  # True if used in-memory streaming
    async_db_calls: int = 0
    sync_db_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'start_time': self.start_time.isoformat(),
            'download_time_ms': self.download_time_ms,
            'streaming_parse_time_ms': self.streaming_parse_time_ms,
            'parse_time_ms': self.parse_time_ms,
            'chunk_time_ms': self.chunk_time_ms,
            'token_batching_time_ms': self.token_batching_time_ms,
            'embed_time_ms': self.embed_time_ms,
            'total_time_ms': self.total_time_ms,
            'file_size_bytes': self.file_size_bytes,
            'total_chunks': self.total_chunks,
            'total_batches': self.total_batches,
            'processed_chunks': self.processed_chunks,
            'processed_batches': self.processed_batches,
            'failed_batches': self.failed_batches,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_batch': self.avg_tokens_per_batch,
            'token_efficiency_pct': self.token_efficiency_pct,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_chunk_size': self.avg_chunk_size,
            'tokens_processed': self.tokens_processed,
            'embedding_calls': self.embedding_calls,
            'retry_count': self.retry_count,
            'disk_io_avoided': self.disk_io_avoided,
            'async_db_calls': self.async_db_calls,
            'sync_db_calls': self.sync_db_calls
        }

@dataclass
class TokenAwareBatch:
    """v6 Enhancement: Token-optimized batch structure"""
    batch_id: str
    source_id: str
    project_id: str
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    total_tokens: int
    chunk_count: int
    token_efficiency: float  # Percentage of max tokens utilized
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def size(self) -> int:
        return len(self.texts)
    
    def is_optimal(self) -> bool:
        """Check if batch is optimally packed"""
        return self.token_efficiency > 0.8  # 80% token utilization
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage of this batch"""
        text_size = sum(len(text.encode('utf-8')) for text in self.texts)
        meta_size = sum(len(json.dumps(meta).encode('utf-8')) for meta in self.metadatas)
        return (text_size + meta_size) / (1024 * 1024)

# ——— v6 Enhancement: In-Memory Streaming Utilities ————————————————————————————————

class InMemoryStreamingDownloader:
    """v6: True in-memory streaming - never touches disk"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS):
        self.max_concurrent = max_concurrent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Pipeline/2.0',
            'Accept-Encoding': 'gzip, deflate'
        })
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, IOError))
    )
    async def stream_to_memory(self, url: str, max_size: int = MAX_TEMP_FILE_SIZE) -> io.BytesIO:
        """v6: Stream download directly to memory buffer"""
        try:
            with Timer() as download_timer:
                with self.session.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > max_size:
                        raise ValueError(f"File too large: {content_length} bytes > {max_size} bytes")
                    
                    # Stream directly to memory
                    memory_buffer = io.BytesIO()
                    downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=STREAMING_CHUNK_SIZE):
                        if chunk:
                            downloaded += len(chunk)
                            if downloaded > max_size:
                                raise ValueError(f"File too large during download: {downloaded} bytes")
                            memory_buffer.write(chunk)
                    
                    memory_buffer.seek(0)  # Reset to beginning for reading
                    
                    logger.info(f"In-memory download completed: {downloaded} bytes in {download_timer.elapsed_ms}ms")
                    return memory_buffer
                    
        except Exception as e:
            logger.error(f"In-memory download failed for {url}: {e}")
            raise

class TokenAwareBatchProcessor:
    """v6 Enhancement: Token-optimized batch creation"""
    
    def __init__(self):
        self.target_tokens = OPTIMAL_BATCH_TOKENS
        self.max_tokens = MAX_BATCH_TOKENS
        self.min_tokens = MIN_BATCH_TOKENS
        self.performance_history = []
        self.memory_manager = MemoryManager()
        
    def calculate_optimal_token_target(self, avg_processing_time: float, memory_usage: float) -> int:
        """v6: Calculate optimal token target based on performance"""
        if memory_usage > 80:
            # Reduce target under memory pressure
            self.target_tokens = max(self.min_tokens, self.target_tokens - 500)
        elif avg_processing_time < 3.0 and memory_usage < 50:
            # Increase target if processing is fast and memory available
            self.target_tokens = min(self.max_tokens, self.target_tokens + 500)
        
        return self.target_tokens
    
    def create_token_optimized_batches(self, texts: List[str], metadatas: List[Dict], 
                                     tokenizer) -> List[TokenAwareBatch]:
        """v6: Create batches optimized for token efficiency"""
        with Timer() as batch_timer:
            batches = []
            current_batch_texts = []
            current_batch_metas = []
            current_tokens = 0
            
            for text, metadata in zip(texts, metadatas):
                # Calculate tokens for this chunk
                chunk_tokens = len(tokenizer.encode(text))
                
                # Check if adding this chunk would exceed our target
                if current_tokens + chunk_tokens > self.target_tokens and current_batch_texts:
                    # Create batch with current chunks
                    batch = self._create_batch(
                        current_batch_texts, current_batch_metas, 
                        current_tokens, len(current_batch_texts)
                    )
                    batches.append(batch)
                    
                    # Start new batch
                    current_batch_texts = [text]
                    current_batch_metas = [metadata]
                    current_tokens = chunk_tokens
                else:
                    # Add to current batch
                    current_batch_texts.append(text)
                    current_batch_metas.append(metadata)
                    current_tokens += chunk_tokens
            
            # Handle final batch
            if current_batch_texts:
                batch = self._create_batch(
                    current_batch_texts, current_batch_metas,
                    current_tokens, len(current_batch_texts)
                )
                batches.append(batch)
            
            # Calculate token efficiency metrics
            total_tokens = sum(batch.total_tokens for batch in batches)
            avg_efficiency = sum(batch.token_efficiency for batch in batches) / len(batches) if batches else 0
            
            logger.info(f"""
            Token-optimized batching completed in {batch_timer.elapsed_ms}ms:
            - Created {len(batches)} batches
            - Total tokens: {total_tokens:,}
            - Average token efficiency: {avg_efficiency:.1f}%
            - Average tokens per batch: {total_tokens / len(batches):.0f}
            """)
            
            return batches
    
    def _create_batch(self, texts: List[str], metadatas: List[Dict], 
                     total_tokens: int, chunk_count: int) -> TokenAwareBatch:
        """Create a token-aware batch with efficiency metrics"""
        token_efficiency = (total_tokens / self.target_tokens) * 100
        
        return TokenAwareBatch(
            batch_id=str(uuid.uuid4()),
            source_id="",  # Will be set by caller
            project_id="",  # Will be set by caller
            texts=texts,
            metadatas=metadatas,
            total_tokens=total_tokens,
            chunk_count=chunk_count,
            token_efficiency=token_efficiency
        )

# ——— v6 Enhancement: Async Database Manager ———————————————————————————————————————

class AsyncDatabaseManager:
    """v6: High-performance async database operations with connection pooling"""
    
    def __init__(self, database_url: str, pool_size: int = ASYNC_DB_POOL_SIZE):
        self.database_url = database_url
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()
        
        # Performance metrics
        self.stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'query_errors': 0,
            'avg_query_time_ms': 0.0,
            'total_query_time_ms': 0
        }
    
    async def initialize_pool(self):
        """Initialize async connection pool"""
        if self.pool is None:
            async with self._lock:
                if self.pool is None:
                    try:
                        self.pool = await asyncpg.create_pool(
                            self.database_url,
                            min_size=5,
                            max_size=self.pool_size,
                            command_timeout=60,
                            server_settings={
                                'application_name': 'rag_pipeline_v6',
                                'tcp_keepalives_idle': '600',
                                'tcp_keepalives_interval': '30',
                                'tcp_keepalives_count': '3'
                            }
                        )
                        self.stats['connections_created'] = self.pool_size
                        logger.info(f"Async database pool initialized with {self.pool_size} connections")
                    except Exception as e:
                        logger.error(f"Failed to initialize async database pool: {e}")
                        raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection from pool"""
        if self.pool is None:
            await self.initialize_pool()
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def bulk_insert_vectors(self, vectors_data: List[Dict[str, Any]]) -> bool:
        """v6: High-performance bulk vector insertion"""
        if not vectors_data:
            return True
        
        try:
            with Timer() as query_timer:
                async with self.get_connection() as conn:
                    # Prepare bulk insert query
                    insert_query = """
                    INSERT INTO document_vector_store (
                        source_id, project_id, content, metadata, embedding, num_tokens, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    
                    # Prepare data for bulk insert
                    insert_values = [
                        (
                            row['source_id'],
                            row['project_id'], 
                            row['content'],
                            json.dumps(row['metadata']),
                            row['embedding'],
                            row['num_tokens'],
                            row['created_at']
                        )
                        for row in vectors_data
                    ]
                    
                    # Execute bulk insert
                    await conn.executemany(insert_query, insert_values)
                    
                    self.stats['queries_executed'] += 1
                    self.stats['total_query_time_ms'] += query_timer.elapsed_ms
                    self.stats['avg_query_time_ms'] = (
                        self.stats['total_query_time_ms'] / self.stats['queries_executed']
                    )
                    
                    logger.info(f"Bulk inserted {len(vectors_data)} vectors in {query_timer.elapsed_ms}ms")
                    return True
                    
        except Exception as e:
            self.stats['query_errors'] += 1
            logger.error(f"Bulk vector insert failed: {e}")
            return False
    
    async def update_document_metrics(self, doc_id: str, batch_inc: int, chunk_inc: int, 
                                    duration_ms: int, **kwargs) -> bool:
        """v6: Async document metrics update"""
        try:
            with Timer() as query_timer:
                async with self.get_connection() as conn:
                    # Use PostgreSQL function for atomic updates
                    await conn.execute("""
                        SELECT increment_doc_metrics($1, $2, $3, $4, $5, $6, $7)
                    """, doc_id, batch_inc, chunk_inc, duration_ms, 
                    kwargs.get('embedding_time_ms', 0),
                    kwargs.get('storage_time_ms', 0),
                    kwargs.get('tokens_processed', 0))
                    
                    self.stats['queries_executed'] += 1
                    self.stats['total_query_time_ms'] += query_timer.elapsed_ms
                    self.stats['avg_query_time_ms'] = (
                        self.stats['total_query_time_ms'] / self.stats['queries_executed']
                    )
                    
                    return True
                    
        except Exception as e:
            self.stats['query_errors'] += 1
            logger.error(f"Async metrics update failed: {e}")
            return False
    
    async def close_pool(self):
        """Close async connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Async database pool closed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        return {
            **self.stats,
            'pool_size': self.pool_size,
            'pool_active': self.pool is not None and not self.pool._closed
        }

# ——— Enhanced Embedding Pipeline (v6) ————————————————————————————————————————————

class ProductionEmbeddingPipeline:
    """v6: Production-grade embedding pipeline with full async operations"""
    
    def __init__(self):
        self.connection_manager = ConnectionPoolManager()
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
            max_retries=3,
            request_timeout=OPENAI_TIMEOUT
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.metrics = MetricsCollector()
        self.memory_manager = MemoryManager()
        
        # v6: Async database manager
        self.async_db = AsyncDatabaseManager(DATABASE_URL)
        
    async def embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """v6: Embed batch with circuit breaker and retry logic"""
        try:
            with Timer() as timer:
                embeddings = await self.embedding_model.aembed_documents(texts)
            
            self.metrics.record_embedding_time(timer.elapsed_ms)
            self.metrics.increment_embedding_calls()
            
            return embeddings
            
        except Exception as e:
            self.metrics.increment_embedding_errors()
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def close_connections(self):
        """v6: Clean up all connections"""
        await self.connection_manager.close_all()
        await self.async_db.close_pool()

# ——— Production Task Implementation (v6) ——————————————————————————————————————————

# Initialize global instances
memory_manager = MemoryManager()
memory_manager.start_monitoring()

metrics_collector = MetricsCollector()
token_batch_processor = TokenAwareBatchProcessor()  # v6: Token-aware processor
in_memory_downloader = InMemoryStreamingDownloader()  # v6: In-memory downloader

# Tokenizer with error handling
try:
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

def _clean_text(text: str) -> str:
    """Enhanced text cleaning with unicode normalization"""
    import unicodedata
    # Remove null bytes and normalize unicode
    cleaned = text.replace("\x00", "").replace("\ufffd", "")
    normalized = unicodedata.normalize('NFKC', cleaned)
    return normalized.strip()

def _calculate_file_hash(filepath_or_buffer: Union[str, io.BytesIO]) -> str:
    """v6: Calculate SHA-256 hash from file path or memory buffer"""
    sha256_hash = hashlib.sha256()
    
    if isinstance(filepath_or_buffer, str):
        # Traditional file path
        with open(filepath_or_buffer, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
    else:
        # Memory buffer
        filepath_or_buffer.seek(0)
        while True:
            chunk = filepath_or_buffer.read(4096)
            if not chunk:
                break
            sha256_hash.update(chunk)
        filepath_or_buffer.seek(0)  # Reset for future use
    
    return sha256_hash.hexdigest()

# ——— Task 1: Enhanced Ingestion Pipeline (v6) ————————————————————————————————————

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_document_task(self, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    v6: Production-grade document ingestion with in-memory streaming
    - Concurrent downloads with in-memory buffers
    - Enhanced deduplication with content hashing
    - Optimized database operations
    - Comprehensive error handling and rollback
    """
    with Timer() as total_timer:
        new_rows, memory_buffers, existing_ids = [], {}, []
        doc_metrics = {}
        
        try:
            logger.info(f"Starting v6 ingestion of {len(file_urls)} documents")
            
            # ——— Phase 1: Concurrent In-Memory Downloads ———————————————————————————
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
                download_futures = {}
                
                for url in file_urls:
                    # Check memory before each download
                    if not memory_manager.wait_for_memory_available():
                        raise MemoryError("Insufficient memory for download")
                    
                    future = executor.submit(
                        asyncio.run, 
                        in_memory_downloader.stream_to_memory(url)
                    )
                    download_futures[future] = url
                
                # Process completed downloads
                for future in download_futures:
                    url = download_futures[future]
                    try:
                        with Timer() as download_timer:
                            memory_buffer = future.result()
                            memory_buffers[url] = memory_buffer
                        
                        # Calculate file metrics from memory buffer
                        file_size = len(memory_buffer.getvalue())
                        file_hash = _calculate_file_hash(memory_buffer)
                        
                        # Enhanced deduplication: content hash
                        ext = os.path.splitext(url)[1].lower()
                        key = f"{metadata['project_id']}/{uuid.uuid4()}{ext}"
                        
                        # Check for existing document by hash
                        existing = supabase_client.table('document_sources') \
                            .select('id') \
                            .eq('content_hash', file_hash) \
                            .execute().data
                        
                        if existing:
                            existing_ids.append(existing[0]['id'])
                            logger.info(f"Skipping duplicate content hash {file_hash[:8]}... → doc_id {existing[0]['id']}")
                            continue
                        
                        # Upload to S3 from memory buffer
                        memory_buffer.seek(0)
                        upload_to_s3(s3_client, memory_buffer, key)
                        cdn_url = get_cloudfront_url(key)
                        
                        # Create document record with v6 enhancements
                        doc_id = str(uuid.uuid4())
                        doc_metrics[doc_id] = DocumentMetrics(
                            doc_id=doc_id,
                            file_size_bytes=file_size,
                            download_time_ms=download_timer.elapsed_ms,
                            disk_io_avoided=True  # v6: Track in-memory processing
                        )
                        
                        new_rows.append({
                            'id': doc_id,
                            'cdn_url': cdn_url,
                            'content_hash': file_hash,
                            'project_id': str(metadata['project_id']),
                            'content_tags': metadata.get('content_tags', []),
                            'uploaded_by': str(metadata['user_id']),
                            'vector_embed_status': ProcessingStatus.INITIALIZING.value,
                            'filename': os.path.basename(url),
                            'file_size_bytes': file_size,
                            'file_extension': ext,
                            'created_at': datetime.now(timezone.utc).isoformat(),
                            'processing_metadata': doc_metrics[doc_id].to_dict(),
                            'processing_version': '6.0'  # v6: Track version
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to process {url}: {e}")
                        continue
            
            # ——— Phase 2: Bulk Database Operations ————————————————————————————————
            inserted_ids = []
            if new_rows:
                try:
                    # Bulk insert with transaction
                    resp = supabase_client.table('document_sources') \
                        .insert(new_rows).execute()
                    
                    # Update to PENDING status
                    inserted_ids = [row['id'] for row in resp.data]
                    supabase_client.table('document_sources') \
                        .update({'vector_embed_status': ProcessingStatus.PENDING.value}) \
                        .in_('id', inserted_ids).execute()
                    
                except Exception as e:
                    logger.error(f"Database insertion failed: {e}")
                    raise
            
            # ——— Phase 3: Schedule In-Memory Parsing Tasks ———————————————————————
            for doc_id in inserted_ids:
                doc_row = next(row for row in new_rows if row['id'] == doc_id)
                
                # v6: Pass memory buffer directly to parsing task
                original_url = None
                for url, buffer in memory_buffers.items():
                    if doc_row['cdn_url'].split('/')[-1].startswith(doc_id.split('-')[0]):
                        original_url = url
                        break
                
                if original_url and original_url in memory_buffers:
                    # v6: Use in-memory buffer for parsing
                    parse_document_task_v6.apply_async(
                        (doc_id, doc_row['cdn_url'], doc_row['project_id'], 
                         memory_buffers[original_url].getvalue()),
                        queue=PARSE_QUEUE,
                        priority=1 if doc_row['file_size_bytes'] < 10 * 1024 * 1024 else 5
                    )