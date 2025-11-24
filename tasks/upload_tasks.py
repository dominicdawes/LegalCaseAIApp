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
from contextlib import asynccontextmanager
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
from psycopg2.extras import execute_batch

# ===== CELERY & TASK QUEUE =====
from celery import chord, group, chain
from celery.signals import worker_init, worker_shutdown
from celery.result import AsyncResult, EagerResult
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
from celery.exceptions import Retry as CeleryRetry

# ===== MACHINE LEARNING & TEXT PROCESSING =====
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ===== MONITORING & METRICS =====
import psutil

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app
from tasks.note_tasks import rag_note_task
from utils.lightrag.lightrag_utils import lightrag_client, lightrag_integration
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
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

# ——— Helpers & Utilities ——————————————————————————————————————————————————————————

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

def _update_batch_progress_sync(batch_id: str, project_id: str, status: BatchProgressStatus):
    """
    [PER BATCH] Update batch_progress for all documents in a batch
    Includes retry logic for stale connections.
    """
    pool = get_global_sync_db_pool()
    retries = 3
    
    for attempt in range(retries):
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE document_sources 
                    SET batch_progress = %s, updated_at = NOW()
                    WHERE project_id = %s 
                    AND processing_metadata->>'batch_id' = %s
                    """,
                    (status.value, project_id, batch_id)
                )
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
    [SYNC] Copy embeddings from an existing processed document to a new project
    Converts python dict rows into the JSONB rows required by Supabase
    Also used in the ultra-low latency path for 100% reused documents
    
    Args:
        existing_source_id: Source ID of the already-processed document
        new_source_id: Source ID of the new document entry
        project_id: Target project ID
        user_id: User who uploaded the document
    
    Returns:
        Dict with copy statistics
    """
    # Use global pool instead of local pool
    pool = get_global_sync_db_pool()
    conn = pool.getconn()
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get vector embeddings from the existing document
            cur.execute(
                '''
                SELECT content, metadata, embedding, num_tokens, page_number, chunk_index
                FROM document_vector_store 
                WHERE source_id = %s AND embedding IS NOT NULL
                ORDER BY page_number, chunk_index
                ''',
                (existing_source_id,)
            )
            existing_embeddings = cur.fetchall()
            
            if not existing_embeddings:
                logger.warning(f"No embeddings found for source_id {existing_source_id}")
                return {'copied_count': 0, 'total_tokens': 0}
            
            # Prepare records for bulk insert
            records_to_insert = []
            total_tokens = 0
            
            # Parse fetched embeddings
            for embedding_row in existing_embeddings:
                records_to_insert.append((
                    str(uuid.uuid4()),
                    new_source_id,
                    project_id,
                    embedding_row['content'],
                    Json(embedding_row['metadata']), # <-- FIXED HERE: Serialize dict back to JSON
                    embedding_row['embedding'],
                    embedding_row['num_tokens'],
                    embedding_row['page_number'],
                    embedding_row['chunk_index'],
                    user_id,
                    datetime.now(timezone.utc)
                ))
                total_tokens += embedding_row['num_tokens'] or 0
            
            # Bulk insert the copied embeddings
            cur.executemany(
                '''INSERT INTO document_vector_store 
                (id, source_id, project_id, content, metadata, embedding, num_tokens, page_number, chunk_index, user_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                records_to_insert
            )
            conn.commit()
            
            logger.info(f"✅ Copied {len(records_to_insert)} embeddings from {existing_source_id} to {new_source_id} for project {project_id}")
            
            return {
                'copied_count': len(records_to_insert),
                'total_tokens': total_tokens
            }
            
    finally:
        pool.putconn(conn)

async def _download_and_prep_doc(client: httpx.AsyncClient, url: str, project_id: str, user_id: str) -> Optional[Dict]:
    """
    Helper for `_analyze_download_and_store_document_for_workflow` to download to memory (does not write to Disk),
    hash, stream to S3 & Store in AWS CLoudfront, and prep data.
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
            s3_key = f"{project_id}/{uuid.uuid4()}{ext}"
            
            # Stream directly to S3 && AWS CloudFront from the in-memory buffer
            upload_to_s3(client=s3_client, file_source=content_stream, s3_object_key=s3_key)
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

async def _call_openai_embeddings_async(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI embeddings API asynchronously
    
    Options:
    1. Use httpx for direct API calls (more control)
    2. Use async OpenAI client (cleaner)
    3. Fall back to sync client in thread pool (hybrid)
    """
    
    # Option 1: Direct httpx call (matches your pattern)
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
            
            # Extract embeddings in correct order
            embeddings = [item['embedding'] for item in data['data']]
            return embeddings
            
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return []
    
# ——— Async Helper Functions ——————————————————————————————————————————————————————

async def _parse_document_async(source_id: str, source_filename: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
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
    _update_document_status_sync(source_id, ProcessingStatus.PARSING)

    short_id = source_id[:8]
    file_buffer = io.BytesIO()
    doc_metrics = DocumentMetrics(doc_id=source_id)
    perf_summary = {}

    with Timer() as total_timer:
        try:
            # Phase 1: FIXED - Async HTTP streaming download
            with Timer() as download_timer:
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
        
        # ——— 1. Generate Embeddings (ASYNC HTTP) ————————————————————————————————
        embeddings = await _call_openai_embeddings_async(texts)
        
        if not embeddings:
            return {
                'success': False,
                'error': 'Failed to generate embeddings'
            }
        
        # ——— 2. Prepare Data (Same as legacy) ———————————————————————————————————
        records_to_insert = []
        total_tokens = 0
        
        for text, meta, vec in zip(texts, metadatas, embeddings):
            if len(vec) != EXPECTED_EMBEDDING_LEN:
                logger.warning(f"⚠️ [BATCH-{short_id}] Skipping malformed embedding")
                continue
            
            # Use your existing tokenizer (same as legacy)
            token_count = len(tokenizer.encode(text))
            total_tokens += token_count
            
            # Extract page number from metadata 'page' from PDFs first, but fall back to 'estimated_page' from DOCX.
            page_number = meta.get('page', meta.get('estimated_page')) if isinstance(meta, dict) else None
            chunk_index = meta.get('chunk_index') if isinstance(meta, dict) else None
            source_url = meta.get('source_id', None)
            
            records_to_insert.append((
                str(uuid.uuid4()), 
                str(uuid.UUID(doc_id)), 
                str(uuid.UUID(project_id)), 
                text, 
                json.dumps(meta), 
                vec, 
                token_count,
                page_number,  # Add page number
                chunk_index,  # Add chunk index for ordering
                source_url,
                datetime.now(timezone.utc)
            ))

        if not records_to_insert:
            return {
                'success': False,
                'error': 'No valid embeddings to insert'
            }

        # ——— 3. Database Insert (Keep your sync pool - it works!) ———————————————
        # Use global pool instead of local pool
        pool = get_global_sync_db_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    '''INSERT INTO document_vector_store 
                    (id, source_id, project_id, content, metadata, embedding, num_tokens, page_number, chunk_index, cdn_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    records_to_insert
                )
                conn.commit()
        finally:
            pool.putconn(conn)
        
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
    _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_FAILED)

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
        if len(successful_batches) == 0:
            _update_document_status_sync(doc_id, ProcessingStatus.FAILED_EMBEDDING)
            return {
                'success': False,
                'error': f'All {len(embedding_batches)} embedding batches failed'
            }
        elif len(failed_batches) > 0:
            _update_document_status_sync(doc_id, ProcessingStatus.PARTIAL)
            logger.warning(f"⚠️ [DOC-{short_id}] Partial success: {len(successful_batches)}/{len(embedding_batches)} batches")
        else:
            _update_document_status_sync(doc_id, ProcessingStatus.COMPLETE, stats=results_dict)
            logger.info(f"✅ [DOC-{short_id}] All embeddings successful")
        
        return results_dict
        
    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Embedding processing failed: {e}")
        _update_document_status_sync(doc_id, ProcessingStatus.FAILED_EMBEDDING, str(e))
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
    
    # ——— Step 1: Concurrent Document Analysis ————————————————————————————————————

    logger.info(f"🔍 [BATCH-{batch_id[:8]}] Analyzing document types...")
    _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_ANALYZING)

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
    _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_PROCESSING)

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
            _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_FAILED)
            
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
        _update_batch_progress_sync(batch_id, project_id, BatchProgressStatus.BATCH_EMBEDDING) # ← Technically embedding start with apply_async() in the parent function but this is a good place
        
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
    
    for result in processing_results:
        if result and isinstance(result, dict):
            status = result.get('status')
            if status in ['COMPLETE', 'PARTIAL']:
                successful_docs.append(result)
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
    
    total_docs = len(processing_results)
    success_count = len(successful_docs)
    failure_count = len(failed_docs)
    
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
            rag_note_task.apply_async(kwargs={
                "user_id": note_metadata["user_id"],
                "note_type": note_metadata["note_type"], 
                "project_id": note_metadata["project_id"],
                "note_title": note_metadata["note_title"],
                "provider": note_metadata.get("provider"),
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
            logger.info(f"✅ [BATCH-{batch_id[:8]}] RAG note generation triggered")
            
        except Exception as e:
            logger.error(f"❌ [BATCH-{batch_id[:8]}] Failed to trigger note generation: {e}")
            batch_status = 'NOTE_GENERATION_FAILED'
            
    elif batch_status == 'FAILED':
        logger.info(f"⚠️ [BATCH-{batch_id[:8]}] Skipping note generation - all documents failed")
        
    else:
        logger.info(f"ℹ️ [BATCH-{batch_id[:8]}] Note generation not requested")
    
    return {
        'batch_id': batch_id,
        'batch_status': batch_status,
        'successful_documents': success_count,
        'failed_documents': failure_count,
        'total_chunks_processed': total_chunks,
        'tokens_saved': total_tokens_reused,
        'note_generation_triggered': workflow_metadata.get('create_note') and batch_status in ['COMPLETE', 'PARTIAL']
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
    doc_id = str(uuid.uuid4())  # create a new uuid
    doc_data['id'] = doc_id
    short_id = doc_id[:8]
    
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
        _update_document_status_sync(doc_id, ProcessingStatus.COMPLETE, stats=result)
        
        logger.info(f"✅ [DOC-{short_id}] Document processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Document processing failed: {e}", exc_info=True)
        _update_document_status_sync(doc_id, ProcessingStatus.FAILED_PARSING, str(e))
        
        # Return error result instead of raising (better for batch coordination)
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'FAILED',
            'error': str(e),
            'chunks_created': 0
        }
    
    finally:
        # Cleanup (like your pattern)
        gc.collect()

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
    
    try:
        # ——— 1. INSERT Document Record (Sync DB) ———————————————————————————————————
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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                        (doc_id, doc_data['cdn_url'], doc_data['content_hash'], 
                        project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                        ProcessingStatus.PENDING.value, doc_data['filename'], 
                        doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                        datetime.now(timezone.utc), Json(workflow_metadata))
                    )
                conn.commit()
        finally:
            pool.putconn(conn)
        
        # ——— 2. PARSE Document ———————————————————————————————————————————————————
        logger.info(f"📋 [DOC-{short_id}] → PARSING")

        # Parsing contains these subtasks: select optimal loader → in-memory streaming → clean/chunk text
        parse_result = await _parse_document_async(
            doc_id, 
            doc_data['filename'],
            doc_data['cdn_url'], 
            project_id,
            )
        
        if not parse_result.get('success'):
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'FAILED',
                'error': parse_result.get('error', 'Parsing failed'),
                'chunks_created': 0
            }
        
        chunks = parse_result['chunks']
        chunks_metadata = parse_result.get('metadatas', [])  # ← Extract metadatas
        logger.info(f"✅ [DOC-{short_id}] Parsed {len(chunks)} chunks")

        # --- INTEGRATION ⚙️: This could also be where we can kick off ainsert() with LightRag -------------
        # You can also add document UUIDs here so that LightRAG can use them in its pipeline for consistency
        # LightRAG only accepts full parsed documents so I'll need to concatenate the chunks back into a single document
        
        # @TODO: ADD IN A SWITCH FOR LIGHTRAG INTEGRATION
        if USE_LIGHTRAG_INTEGRATION:
            lightrag_client.insert_document_into_kg(
                doc_id=doc_id,
                chunks=chunks,
            )

        # ——— 3. EMBEDDING Process, async with concurrency control ————————————————
        logger.info(f"📋 [DOC-{short_id}] → EMBEDDING")
        embedding_result = await _process_embeddings_async(doc_id, project_id, chunks, chunks_metadata)
        
        if not embedding_result.get('success'):
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW', 
                'status': 'FAILED',
                'error': embedding_result.get('error', 'Embedding failed'),
                'chunks_created': len(chunks)
            }
        
        logger.info(f"✅ [DOC-{short_id}] Embedded {embedding_result['chunks_embedded']} chunks")
        
        # ——— Return Success Result ———————————————————————————————————————————————
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW',
            'status': 'COMPLETE',
            'chunks_created': embedding_result['chunks_embedded'],
            'total_tokens': embedding_result['total_tokens'],
            'processing_time_ms': embedding_result.get('processing_time_ms', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ [DOC-{short_id}] Async processing failed: {e}")
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
        
        # ——— 2. Parallel Processing: Traditional Embeddings + LightRAG ——————————————
        logger.info(f"⚡ [DOC-{short_id}] Running parallel: embeddings + LightRAG")
        
        # Start both processes concurrently
        from upload_tasks import _process_embeddings_async
        
        embedding_task = asyncio.create_task(
            _process_embeddings_async(doc_id, project_id, chunks, chunks_metadata)
        )
        
        lightrag_task = asyncio.create_task(
            lightrag_integration.enhance_document_processing(
                doc_id, chunks, chunks_metadata, project_id
            )
        )
        
        # Wait for both to complete
        embedding_result, lightrag_result = await asyncio.gather(
            embedding_task, lightrag_task, return_exceptions=True
        )
        
        # ——— 3. Analyze Results ———————————————————————————————————————————————————
        final_status = 'COMPLETE'
        
        # Check embedding results
        if isinstance(embedding_result, Exception) or not embedding_result.get('success'):
            final_status = 'PARTIAL'
            logger.warning(f"⚠️ [DOC-{short_id}] Embedding processing failed")
        
        # Check LightRAG results (non-critical)
        if isinstance(lightrag_result, Exception) or not lightrag_result.get('success'):
            logger.warning(f"⚠️ [DOC-{short_id}] LightRAG processing failed (non-critical)")
        
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
        rag_note_task.apply_async(kwargs={
            "user_id": note_metadata["user_id"],
            "note_type": note_metadata["note_type"], 
            "project_id": note_metadata["project_id"],
            "note_title": note_metadata["note_title"],
            "provider": note_metadata.get("provider"),
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
    new_doc_id = str(uuid.uuid4())
    
    try:
        # ——— Create New Document Entry + Copy Embeddings ——————————————————————————
        # Use global pool instead of local pool
        pool = get_global_sync_db_pool()
        conn = pool.getconn()
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # FIND source document info for copying
                cur.execute(
                    'SELECT total_chunks, total_batches FROM document_sources WHERE id = %s',
                    (existing_doc_id,)
                )
                source_info = cur.fetchone()
                
                if not source_info:
                    raise Exception(f"Source document {existing_doc_id} not found")
                
                # INSERT new document record with completed status
                cur.execute(
                    '''INSERT INTO document_sources 
                    (id, cdn_url, content_hash, project_id, content_tags, uploaded_by, 
                    vector_embed_status, filename, file_size_bytes, file_extension, 
                    total_chunks, total_batches, created_at, processing_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', # <-- FIXED HERE: unsupported format character ')' (0x29)
                    (new_doc_id, doc_data['cdn_url'], doc_data['content_hash'], 
                    project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                    ProcessingStatus.COMPLETE.value, doc_data['filename'], 
                    doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                    source_info['total_chunks'], source_info['total_batches'],
                    datetime.now(timezone.utc), Json(workflow_metadata))
                )
                conn.commit() # Commit the insert of the new document source
        finally:
            pool.putconn(conn)
        
        # ——— Copy Embeddings (reuse existing sync function) ————————————————————————
        copy_result = copy_embeddings_for_project_sync(
            existing_doc_id, 
            new_doc_id, 
            project_id, 
            workflow_metadata['user_id']
        )
        
        logger.info(f"♻️ [DOC-{new_doc_id[:8]}] Smart reuse complete: {copy_result['copied_count']} chunks")
        return {
            'doc_id': new_doc_id,
            'processing_type': 'REUSED',
            'status': 'COMPLETE',
            'chunks_reused': copy_result['copied_count'],
            'tokens_reused': copy_result['total_tokens']
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
