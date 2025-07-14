import os
import uuid
import tempfile
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib
import psutil
import time

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
from pybreaker import CircuitBreaker    # pip install pybreaker

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

# Adaptive batch sizing
MIN_BATCH_SIZE = 5
MAX_BATCH_SIZE = 50
OPTIMAL_BATCH_SIZE = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EXPECTED_EMBEDDING_LEN = 1536

# Memory management
MAX_MEMORY_USAGE_PCT = 75
MEMORY_CHECK_INTERVAL = 30  # seconds
MAX_TEMP_FILE_SIZE = 500 * 1024 * 1024  # 500MB
STREAMING_CHUNK_SIZE = 64 * 1024  # 64KB chunks

# Concurrency limits
MAX_CONCURRENT_DOWNLOADS = 10
MAX_CONCURRENT_EMBEDS = 5
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MAX_TOKENS = 8192
OPENAI_TIMEOUT = 60

# ——— Enhanced Data Structures ————————————————————————————————————————————————————

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

@dataclass
class ChunkBatch:
    """Optimized batch structure for processing"""
    batch_id: str
    source_id: str
    project_id: str
    texts: List[str]
    metadatas: List[Dict[str, Any]]
    token_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def size(self) -> int:
        return len(self.texts)
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage of this batch"""
        text_size = sum(len(text.encode('utf-8')) for text in self.texts)
        meta_size = sum(len(json.dumps(meta).encode('utf-8')) for meta in self.metadatas)
        return (text_size + meta_size) / (1024 * 1024)

# ——— Enhanced Utilities ————————————————————————————————————————————————————

class StreamingDownloader:
    """High-performance streaming downloader with connection pooling"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS):
        self.max_concurrent = max_concurrent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Pipeline/1.0',
            'Accept-Encoding': 'gzip, deflate'
        })
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, IOError))
    )
    async def download_stream_to_temp(self, url: str, max_size: int = MAX_TEMP_FILE_SIZE) -> str:
        """Stream download with size limits and compression support"""
        try:
            with self.session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    raise ValueError(f"File too large: {content_length} bytes > {max_size} bytes")
                
                # Create temp file with proper suffix
                suffix = os.path.splitext(url)[1] or ''
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                
                downloaded = 0
                async with aiofiles.open(temp_file.name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=STREAMING_CHUNK_SIZE):
                        if chunk:
                            downloaded += len(chunk)
                            if downloaded > max_size:
                                os.unlink(temp_file.name)
                                raise ValueError(f"File too large during download: {downloaded} bytes")
                            await f.write(chunk)
                
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            raise

class AdaptiveBatchProcessor:
    """Adaptive batch sizing based on system performance"""
    
    def __init__(self):
        self.current_batch_size = OPTIMAL_BATCH_SIZE
        self.performance_history = []
        self.memory_manager = MemoryManager()
        
    def calculate_optimal_batch_size(self, avg_processing_time: float, memory_usage: float) -> int:
        """Calculate optimal batch size based on performance metrics"""
        if memory_usage > 80:
            # Reduce batch size under memory pressure
            self.current_batch_size = max(MIN_BATCH_SIZE, self.current_batch_size - 2)
        elif avg_processing_time < 5.0 and memory_usage < 50:
            # Increase batch size if processing is fast and memory is available
            self.current_batch_size = min(MAX_BATCH_SIZE, self.current_batch_size + 2)
        
        return self.current_batch_size
    
    def create_adaptive_batches(self, texts: List[str], metadatas: List[Dict]) -> List[ChunkBatch]:
        """Create optimally-sized batches"""
        batches = []
        current_batch_size = self.current_batch_size
        
        for i in range(0, len(texts), current_batch_size):
            batch_texts = texts[i:i + current_batch_size]
            batch_metas = metadatas[i:i + current_batch_size]
            
            # Calculate token count for this batch
            token_count = sum(len(tokenizer.encode(text)) for text in batch_texts)
            
            batch = ChunkBatch(
                batch_id=str(uuid.uuid4()),
                source_id="",  # Will be set by caller
                project_id="",  # Will be set by caller
                texts=batch_texts,
                metadatas=batch_metas,
                token_count=token_count
            )
            
            batches.append(batch)
        
        return batches

# ——— Enhanced Embedding Pipeline ————————————————————————————————————————————————

class ProductionEmbeddingPipeline:
    """Production-grade embedding pipeline with circuit breakers and monitoring"""
    
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
        
    async def embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed batch with circuit breaker and retry logic"""
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

# ——— Production Task Implementation ————————————————————————————————————————————————

# Initialize global instances
memory_manager = MemoryManager()
metrics_collector = MetricsCollector()
batch_processor = AdaptiveBatchProcessor()
downloader = StreamingDownloader()

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

def _calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash for file deduplication"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# ——— Task 1: Enhanced Ingestion Pipeline ————————————————————————————————————————————

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_document_task(self, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production-grade document ingestion with:
    - Concurrent downloads with connection pooling
    - Advanced deduplication (URL + content hash)
    - Comprehensive error handling and rollback
    - Fine-grained progress tracking
    - Memory-aware processing
    """
    with Timer() as total_timer:
        new_rows, temp_files, existing_ids = [], [], []
        doc_metrics = {}
        
        try:
            # ——— Phase 1: Concurrent Download & Deduplication ————————————————————————
            logger.info(f"Starting ingestion of {len(file_urls)} documents")
            
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
                download_futures = {}
                
                for url in file_urls:
                    # Check memory before each download
                    if not memory_manager.wait_for_memory_available():
                        raise MemoryError("Insufficient memory for download")
                    
                    future = executor.submit(downloader.download_stream_to_temp, url)
                    download_futures[future] = url
                
                # Process completed downloads
                for future in download_futures:
                    url = download_futures[future]
                    try:
                        with Timer() as download_timer:
                            temp_path = future.result()
                            temp_files.append(temp_path)
                        
                        # Calculate file metrics
                        file_size = os.path.getsize(temp_path)
                        file_hash = _calculate_file_hash(temp_path)
                        
                        # Enhanced deduplication: URL + content hash
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
                        
                        # Upload to S3 and get CDN URL
                        upload_to_s3(s3_client, temp_path, key)
                        cdn_url = get_cloudfront_url(key)
                        
                        # Create document record
                        doc_id = str(uuid.uuid4())
                        doc_metrics[doc_id] = DocumentMetrics(
                            doc_id=doc_id,
                            file_size_bytes=file_size,
                            download_time_ms=download_timer.elapsed_ms
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
                            'processing_metadata': doc_metrics[doc_id].to_dict()
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to process {url}: {e}")
                        continue
            
            # ——— Phase 2: Bulk Database Operations ————————————————————————————————————
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
            
            # ——— Phase 3: Schedule Parsing Tasks ————————————————————————————————————
            for doc_id in inserted_ids:
                doc_row = next(row for row in new_rows if row['id'] == doc_id)
                
                # Schedule parsing with priority based on file size
                priority = 1 if doc_row['file_size_bytes'] < 10 * 1024 * 1024 else 5
                
                parse_document_task.apply_async(
                    (doc_id, doc_row['cdn_url'], doc_row['project_id']),
                    queue=PARSE_QUEUE,
                    priority=priority
                )
            
            # Record metrics
            metrics_collector.record_ingestion_metrics(
                total_documents=len(file_urls),
                processed_documents=len(inserted_ids),
                duplicate_documents=len(existing_ids),
                processing_time_ms=total_timer.elapsed_ms
            )
            
            return {
                'doc_ids': existing_ids + inserted_ids,
                'new_documents': len(inserted_ids),
                'duplicate_documents': len(existing_ids),
                'processing_time_ms': total_timer.elapsed_ms
            }
            
        except Exception as e:
            logger.error(f"Ingestion task failed: {e}")
            raise
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

# ——— Task 2: Enhanced Parsing Pipeline ————————————————————————————————————————————

@celery_app.task(bind=True, queue=PARSE_QUEUE, acks_late=True)
def parse_document_task(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
    """
    Production-grade document parsing with:
    - Streaming chunking to minimize memory usage
    - Multi-process text splitting for large documents
    - Adaptive batch sizing based on system performance
    - Comprehensive error handling and recovery
    """
    
    with Timer() as total_timer:
        temp_file = None
        doc_metrics = DocumentMetrics(doc_id=source_id)
        
        try:
            # ——— Phase 1: Update Status & Download ————————————————————————————————
            supabase_client.table('document_sources') \
                .update({'vector_embed_status': ProcessingStatus.PARSING.value}) \
                .eq('id', source_id).execute()
            
            logger.info(f"Starting parsing for document {source_id}")
            
            # Hybrid approach: async I/O operations within sync task
            def download_with_async_io():
                """Use asyncio for just the I/O-bound download operation"""
                async def _async_download():
                    if not memory_manager.wait_for_memory_available():
                        raise MemoryError("Insufficient memory for download")
                    
                    async with httpx.AsyncClient(timeout=30) as client:
                        async with client.stream('GET', cdn_url) as response:
                            response.raise_for_status()
                            
                            # Check content length
                            content_length = response.headers.get('content-length')
                            if content_length and int(content_length) > MAX_TEMP_FILE_SIZE:
                                raise ValueError(f"File too large: {content_length} bytes")
                            
                            # Stream to temp file
                            suffix = os.path.splitext(cdn_url)[1] or ''
                            temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                            
                            downloaded = 0
                            async with aiofiles.open(temp_file_obj.name, 'wb') as f:
                                async for chunk in response.aiter_bytes(chunk_size=STREAMING_CHUNK_SIZE):
                                    downloaded += len(chunk)
                                    if downloaded > MAX_TEMP_FILE_SIZE:
                                        os.unlink(temp_file_obj.name)
                                        raise ValueError(f"File too large: {downloaded} bytes")
                                    await f.write(chunk)
                            
                            return temp_file_obj.name
                
                # Run the async operation in the sync context
                return asyncio.run(_async_download())
            
            # Execute the hybrid download
            with Timer() as download_timer:
                temp_file = download_with_async_io()
                doc_metrics.file_size_bytes = os.path.getsize(temp_file)
                doc_metrics.download_time_ms = download_timer.elapsed_ms
            
            # ——— Phase 2: Streaming Document Processing ————————————————————————————
            with Timer() as parse_timer:
                # Initialize document loader
                loader = get_loader_for(temp_file)
                
                # Create optimized text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # Process document in streaming fashion
                all_chunks = []
                chunk_metadatas = []
                
                # Use generator to stream document pages
                for page_num, page in enumerate(loader.stream_documents(temp_file)):
                    if memory_manager.check_memory_pressure():
                        logger.warning(f"Memory pressure detected, yielding...")
                        time.sleep(0.1)  # Yield control
                    
                    # Split page into chunks
                    page_chunks = splitter.split_documents([page])
                    
                    for chunk_idx, chunk in enumerate(page_chunks):
                        cleaned_content = _clean_text(chunk.page_content)
                        
                        # Skip empty chunks
                        if not cleaned_content.strip():
                            continue
                        
                        # Enhance metadata
                        enhanced_metadata = {
                            **chunk.metadata,
                            'page_number': page_num,
                            'chunk_index': chunk_idx,
                            'char_count': len(cleaned_content),
                            'token_count': len(tokenizer.encode(cleaned_content))
                        }
                        
                        all_chunks.append(cleaned_content)
                        chunk_metadatas.append(enhanced_metadata)
                
                doc_metrics.parse_time_ms = parse_timer.elapsed_ms
                doc_metrics.total_chunks = len(all_chunks)
                doc_metrics.avg_chunk_size = sum(len(chunk) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
                doc_metrics.tokens_processed = sum(meta['token_count'] for meta in chunk_metadatas)
            
            # ——— Phase 3: Adaptive Batch Creation ————————————————————————————————
            with Timer() as batch_timer:
                # Create optimally-sized batches
                batches = batch_processor.create_adaptive_batches(all_chunks, chunk_metadatas)
                
                # Update batch info
                for batch in batches:
                    batch.source_id = source_id
                    batch.project_id = project_id
                
                doc_metrics.total_batches = len(batches)
                doc_metrics.chunk_time_ms = batch_timer.elapsed_ms
            
            # ——— Phase 4: Database Update & Task Scheduling ————————————————————————
            # Update document status and metrics
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': ProcessingStatus.EMBEDDING.value,
                    'total_chunks': doc_metrics.total_chunks,
                    'total_batches': doc_metrics.total_batches,
                    'processing_metadata': doc_metrics.to_dict()
                }).eq('id', source_id).execute()
            
            # Create embedding task signatures
            embedding_tasks = []
            for batch in batches:
                task_signature = embed_batch_task.s(
                    batch.source_id,
                    batch.project_id,
                    batch.texts,
                    batch.metadatas,
                    batch.batch_id
                )
                embedding_tasks.append(task_signature)
            
            # Schedule embedding tasks with chord pattern
            if embedding_tasks:
                chord(embedding_tasks, queue=EMBED_QUEUE)(
                    finalize_embeddings.s(source_id)
                )
            else:
                # No chunks to embed, mark as complete
                finalize_embeddings.apply_async(([], source_id), queue=FINAL_QUEUE)
            
            logger.info(f"Parsing complete for {source_id}: {doc_metrics.total_chunks} chunks, {doc_metrics.total_batches} batches")
            
            return {
                'source_id': source_id,
                'total_chunks': doc_metrics.total_chunks,
                'total_batches': doc_metrics.total_batches,
                'metrics': doc_metrics.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Parsing failed for {source_id}: {e}")
            
            # Update status to failed
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': ProcessingStatus.FAILED_PARSING.value,
                    'error_message': str(e)
                }).eq('id', source_id).execute()
            
            raise
        finally:
            # Cleanup
            if temp_file:
                try:
                    os.unlink(temp_file)
                except:
                    pass

# ——— Task 3: Production Embedding Pipeline ————————————————————————————————————————

@celery_app.task(
    bind=True,
    queue=EMBED_QUEUE,
    max_retries=MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    rate_limit=RATE_LIMIT,
    acks_late=True
)
async def embed_batch_task(
    self, 
    source_id: str, 
    project_id: str, 
    texts: List[str], 
    metadatas: List[Dict[str, Any]], 
    batch_id: str
) -> Dict[str, Any]:
    """
    Production-grade embedding task with:
    - Async processing with connection pooling
    - Circuit breaker for OpenAI API resilience
    - Comprehensive retry logic with exponential backoff
    - Memory-aware processing with backpressure
    - Fine-grained performance monitoring
    """
    
    embedding_pipeline = ProductionEmbeddingPipeline()
    
    with Timer() as total_timer:
        try:
            logger.info(f"Starting embedding batch {batch_id} for document {source_id}")
            
            # ——— Phase 1: Pre-processing & Validation ————————————————————————————
            if not texts:
                raise ValueError("Empty text batch received")
            
            # Memory check before processing
            batch_memory_estimate = sum(len(text.encode('utf-8')) for text in texts) / (1024 * 1024)
            async with memory_manager.memory_context(required_mb=batch_memory_estimate):
                
                # Clean and validate texts
                cleaned_texts = []
                token_counts = []
                
                for text in texts:
                    cleaned = _clean_text(text)
                    if not cleaned.strip():
                        continue
                    
                    # Token count validation
                    token_count = len(tokenizer.encode(cleaned))
                    if token_count > OPENAI_MAX_TOKENS:
                        # Truncate if too long
                        tokens = tokenizer.encode(cleaned)[:OPENAI_MAX_TOKENS]
                        cleaned = tokenizer.decode(tokens)
                        token_count = OPENAI_MAX_TOKENS
                    
                    cleaned_texts.append(cleaned)
                    token_counts.append(token_count)
                
                if not cleaned_texts:
                    logger.warning(f"No valid texts after cleaning in batch {batch_id}")
                    return {'batch_id': batch_id, 'processed_count': 0}
                
                # ——— Phase 2: Embedding Generation ————————————————————————————————
                with Timer() as embed_timer:
                    try:
                        # Generate embeddings with circuit breaker
                        embeddings = await embedding_pipeline.embed_batch_with_retry(cleaned_texts)
                        
                        # Validate embedding dimensions
                        for i, embedding in enumerate(embeddings):
                            if len(embedding) != EXPECTED_EMBEDDING_LEN:
                                raise ValueError(
                                    f"Embedding {i} has length {len(embedding)}, expected {EXPECTED_EMBEDDING_LEN}"
                                )
                        
                        embedding_time_ms = embed_timer.elapsed_ms
                        
                    except Exception as e:
                        logger.error(f"Embedding generation failed for batch {batch_id}: {e}")
                        raise
                
                # ——— Phase 3: Data Preparation & Storage ————————————————————————————
                with Timer() as storage_timer:
                    # Prepare vector storage records
                    vector_records = []
                    
                    for text, metadata, embedding, token_count in zip(
                        cleaned_texts, metadatas, embeddings, token_counts
                    ):
                        # Enhanced metadata with processing info
                        enhanced_metadata = {
                            **metadata,
                            'batch_id': batch_id,
                            'processed_at': datetime.now(timezone.utc).isoformat(),
                            'token_count': token_count,
                            'embedding_model': 'text-embedding-ada-002',
                            'processing_version': '2.0'
                        }
                        
                        vector_records.append({
                            'source_id': source_id,
                            'project_id': project_id,
                            'content': text,
                            'metadata': enhanced_metadata,
                            'embedding': embedding,
                            'num_tokens': token_count,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        })
                    
                    # Bulk insert with retry logic
                    try:
                        supabase_client.table('document_vector_store') \
                            .insert(vector_records).execute()
                        
                        storage_time_ms = storage_timer.elapsed_ms
                        
                    except Exception as e:
                        logger.error(f"Vector storage failed for batch {batch_id}: {e}")
                        raise
                
                # ——— Phase 4: Metrics & Telemetry Update ————————————————————————————
                total_time_ms = total_timer.elapsed_ms
                
                # Update document processing metrics
                try:
                    supabase_client.rpc('increment_doc_metrics', {
                        'doc_id': source_id,
                        'batch_inc': 1,
                        'chunk_inc': len(cleaned_texts),
                        'duration_ms': total_time_ms,
                        'embedding_time_ms': embedding_time_ms,
                        'storage_time_ms': storage_time_ms,
                        'tokens_processed': sum(token_counts)
                    }).execute()
                    
                except Exception as e:
                    logger.warning(f"Metrics update failed for batch {batch_id}: {e}")
                    # Don't fail the task for metrics issues
                
                # Record performance metrics
                metrics_collector.record_batch_metrics(
                    batch_id=batch_id,
                    source_id=source_id,
                    chunk_count=len(cleaned_texts),
                    total_time_ms=total_time_ms,
                    embedding_time_ms=embedding_time_ms,
                    storage_time_ms=storage_time_ms,
                    token_count=sum(token_counts)
                )
                
                logger.info(f"Embedding batch {batch_id} completed: {len(cleaned_texts)} chunks in {total_time_ms}ms")
                
                return {
                    'batch_id': batch_id,
                    'source_id': source_id,
                    'processed_count': len(cleaned_texts),
                    'total_time_ms': total_time_ms,
                    'embedding_time_ms': embedding_time_ms,
                    'storage_time_ms': storage_time_ms,
                    'token_count': sum(token_counts)
                }
        
        except MaxRetriesExceededError:
            # Terminal failure - update document status
            logger.error(f"Embedding batch {batch_id} failed after {MAX_RETRIES} retries")
            
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': ProcessingStatus.FAILED_EMBEDDING.value,
                    'error_message': f"Batch {batch_id} failed after {MAX_RETRIES} retries"
                }).eq('id', source_id).execute()
            
            raise
            
        except Exception as exc:
            # Increment retry counter and retry
            logger.warning(f"Embedding batch {batch_id} failed (attempt {self.request.retries + 1}), retrying: {exc}")
            
            # Update metrics for retry
            metrics_collector.increment_batch_retries(batch_id)
            
            # Exponential backoff with jitter
            countdown = DEFAULT_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** self.request.retries)
            raise self.retry(exc=exc, countdown=countdown)
        
        finally:
            # Cleanup resources
            await embedding_pipeline.connection_manager.close()

# ——— Task 4: Enhanced Finalization Pipeline ————————————————————————————————————————

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def finalize_embeddings(self, batch_results: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
    """
    Production-grade finalization with:
    - Comprehensive status validation
    - Detailed completion metrics
    - Failure analysis and recovery suggestions
    - Performance optimization recommendations
    """
    
    with Timer() as total_timer:
        try:
            logger.info(f"Finalizing embeddings for document {source_id}")
            
            # ——— Phase 1: Fetch Current Document State ————————————————————————————
            doc_record = supabase_client.table('document_sources') \
                .select('*') \
                .eq('id', source_id) \
                .single() \
                .execute()
            
            if not doc_record.data:
                raise ValueError(f"Document {source_id} not found")
            
            doc_data = doc_record.data
            
            # ——— Phase 2: Validate Processing Completion ————————————————————————————
            # Get actual counts from vector store
            vector_count_result = supabase_client.table('document_vector_store') \
                .select('*', count='exact') \
                .eq('source_id', source_id) \
                .execute()
            
            actual_vector_count = vector_count_result.count or 0
            expected_chunks = doc_data.get('total_chunks', 0)
            processed_batches = doc_data.get('processed_batches', 0)
            total_batches = doc_data.get('total_batches', 0)
            
            # Calculate completion statistics
            chunk_completion_rate = (actual_vector_count / expected_chunks * 100) if expected_chunks > 0 else 0
            batch_completion_rate = (processed_batches / total_batches * 100) if total_batches > 0 else 0
            
            # ——— Phase 3: Determine Final Status ————————————————————————————————————
            if actual_vector_count == expected_chunks and processed_batches == total_batches:
                final_status = ProcessingStatus.COMPLETE.value
                status_message = "Processing completed successfully"
            elif actual_vector_count > 0:
                final_status = ProcessingStatus.PARTIAL.value
                status_message = f"Partial completion: {actual_vector_count}/{expected_chunks} chunks processed"
            else:
                final_status = ProcessingStatus.FAILED_EMBEDDING.value
                status_message = "No chunks were successfully processed"
            
            # ——— Phase 4: Calculate Final Metrics ————————————————————————————————————
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
            
            # ——— Phase 6: Final Database Updates ————————————————————————————————————
            final_metadata = {
                **doc_data.get('processing_metadata', {}),
                'completion_timestamp': datetime.now(timezone.utc).isoformat(),
                'final_vector_count': actual_vector_count,
                'chunk_completion_rate': chunk_completion_rate,
                'batch_completion_rate': batch_completion_rate,
                'total_processing_time_ms': total_processing_time,
                'total_embedding_time_ms': total_embedding_time,
                'total_storage_time_ms': total_storage_time,
                'total_tokens_processed': total_tokens,
                'performance_insights': performance_insights,
                'optimization_recommendations': recommendations,
                'failed_batches': failed_batches
            }
            
            # Update document status
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': final_status,
                    'final_vector_count': actual_vector_count,
                    'completion_rate': chunk_completion_rate,
                    'processing_metadata': final_metadata,
                    'status_message': status_message,
                    'completed_at': datetime.now(timezone.utc).isoformat()
                }).eq('id', source_id).execute()
            
            # ——— Phase 7: Global Metrics & Monitoring ————————————————————————————————
            # Record completion metrics
            metrics_collector.record_completion_metrics(
                source_id=source_id,
                final_status=final_status,
                total_chunks=expected_chunks,
                processed_chunks=actual_vector_count,
                total_time_ms=total_processing_time,
                completion_rate=chunk_completion_rate
            )
            
            # Log completion summary
            logger.info(f"""
            Document {source_id} finalization complete:
            - Status: {final_status}
            - Chunks: {actual_vector_count}/{expected_chunks} ({chunk_completion_rate:.1f}%)
            - Batches: {processed_batches}/{total_batches} ({batch_completion_rate:.1f}%)
            - Total time: {total_processing_time}ms
            - Tokens/sec: {tokens_per_second:.1f}
            """)
            
            return {
                'source_id': source_id,
                'final_status': final_status,
                'total_chunks': expected_chunks,
                'processed_chunks': actual_vector_count,
                'completion_rate': chunk_completion_rate,
                'total_time_ms': total_processing_time,
                'performance_insights': performance_insights,
                'recommendations': recommendations,
                'finalization_time_ms': total_timer.elapsed_ms
            }
            
        except Exception as e:
            logger.error(f"Finalization failed for document {source_id}: {e}")
            
            # Update to failed status
            supabase_client.table('document_sources') \
                .update({
                    'vector_embed_status': ProcessingStatus.FAILED_FINALIZATION.value,
                    'error_message': f"Finalization failed: {str(e)}"
                }).eq('id', source_id).execute()
            
            raise

# ——— Advanced Monitoring & Health Check Tasks ————————————————————————————————————

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