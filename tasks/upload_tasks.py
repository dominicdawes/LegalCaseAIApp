"""v6 By Gemini 2.5 Pro bc Claude hit rate/context limits"""

# ===== STANDARD LIBRARY IMPORTS =====
import os
import sys
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
import gevent
import gevent.socket
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
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch

# ===== CELERY & TASK QUEUE =====
from celery import chord, group
from celery.signals import worker_init, worker_shutdown
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
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.base import BaseDocumentLoader
from utils.document_loaders.loader_factory import get_loader_for, analyze_document_before_processing, get_high_performance_loader
from utils.document_loaders.performance import create_optimized_processor, BatchTokenCounter
from utils.metrics import MetricsCollector, Timer
from utils.connection_pool import ConnectionPoolManager
from utils.memory_manager import MemoryManager # Kept for health checks

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

# OpenAI configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_MAX_TOKENS_PER_BATCH = 8190 # Safety margin below the 8192 limit
EXPECTED_EMBEDDING_LEN = 1536
MAX_CONCURRENT_DOWNLOADS = 3 # A modest limit instead of 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Database (asyncpg)
DB_DSN = os.getenv("POSTGRES_DSN_POOL") # e.g., Supabase -> Connection -> Get Direct URL
# DB_POOL_MIN_SIZE = 5  # <-- if i had more compute
# DB_POOL_MAX_SIZE = 20
DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 5

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

# class StreamingDocumentProcessor:
#     """
#     DEPRECATED: Wrapper class to encapsulate streaming document processing
#     Maintains the TRUE IN-MEMORY STREAMING architecture
#     """
    
#     def __init__(self, file_buffer: io.BytesIO, filename: str):
#         self.file_buffer = file_buffer
#         self.filename = filename
#         self.loader = None
        
#     def get_loader(self) -> BaseDocumentLoader:
#         """Get the appropriate loader for this document"""
#         if self.loader is None:
#             self.loader = get_loader_for(filename=self.filename, file_like_object=self.file_buffer)
#         return self.loader
    
#     def stream_chunks(self, splitter: RecursiveCharacterTextSplitter) -> Iterator[Tuple[str, Dict]]:
#         """
#         Generator that yields (text, metadata) tuples for each chunk
#         Maintains true streaming - never loads entire document into memory at once
#         """
#         loader = self.get_loader()
        
#         # Reset buffer position
#         self.file_buffer.seek(0)
        
#         for page_num, page in enumerate(loader.stream_documents(self.file_buffer)):
#             # Split page into chunks
#             chunks = splitter.split_documents([page])
            
#             for chunk_idx, chunk in enumerate(chunks):
#                 cleaned_content = _clean_text(chunk.page_content)
#                 if cleaned_content:
#                     metadata = {
#                         **chunk.metadata,
#                         'page_number': page_num,
#                         'chunk_index': chunk_idx
#                     }
#                     yield cleaned_content, metadata

# ——— DB Pool Instances (Initialized once per worker) ————————————————————————————

# Global pool variable
db_pool: Optional[asyncpg.Pool] = None

# ——— 1. POOL CLEANUP FUNCTIONS ——————————————————————————————————————————————————

async def close_db_pool():
    """Safely close the database pool"""
    global db_pool
    if db_pool is not None:
        logger.info("🔄 Closing database pool...")
        try:
            await db_pool.close()
            logger.info("✅ Database pool closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing database pool: {e}")
        finally:
            db_pool = None

def sync_close_db_pool():
    """Synchronous wrapper for closing the pool"""
    try:
        if db_pool is not None:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Close the pool
            loop.run_until_complete(close_db_pool())
    except Exception as e:
        logger.error(f"❌ Error in sync pool cleanup: {e}")

# ——— 2. CELERY SIGNAL HANDLERS ————————————————————————————————————————————————————

# @worker_init.connect
def worker_init_handler(sender=None, **kwargs):
    """Called when Celery worker starts"""
    global db_pool
    logger.info("🚀 Celery worker initializing && db pool resetting...")
    
    # Force reset the global pool variable
    db_pool = None
    logger.info("🔄 Database pool reset on worker init")

# Global pool reset on worker init
worker_init.connect(worker_init_handler)

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when Celery worker shuts down"""
    logger.info("🛑 Celery worker shutting down...")
    sync_close_db_pool()

# ——— 3. SYSTEM SIGNAL HANDLERS ————————————————————————————————————————————————————

def signal_handler(signum, frame):
    """Handle system signals (SIGTERM, SIGINT)"""
    logger.info(f"🛑 Received signal {signum}, cleaning up...")
    sync_close_db_pool()

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register atexit handler as last resort
atexit.register(sync_close_db_pool)

# ——— 4. UPDATED get_async_db_pool FUNCTION ———————————————————————————————————————

# Get Asynchronous DB pool
async def get_async_db_pool() -> asyncpg.Pool:
    """Initializes and returns the asyncpg connection pool with connection testing."""
    global db_pool
    try:
        # Always close existing pool if it exists (future-proofing)
        if db_pool is not None:
            logger.info("🔄 Closing existing database pool...")
            await db_pool.close()
            db_pool = None
            logger.info("✅ Existing pool closed")
        
        if not DB_DSN:
            raise ValueError("POSTGRES_DSN environment variable not set.")
        
        # Log connection attempt (mask sensitive info)
        masked_dsn = DB_DSN
        if '@' in masked_dsn:
            parts = masked_dsn.split('@')
            if len(parts) == 2:
                # Show only host:port/db part
                masked_dsn = f"postgresql://***:***@{parts[1]}"
        logger.info(f"🔗 Attempting to connect to: {masked_dsn}")
        
        # Extract host and port for ping test
        import urllib.parse
        parsed = urllib.parse.urlparse(DB_DSN)
        host = parsed.hostname
        port = parsed.port or 5432
        logger.info(f"🎯 Target host: {host}:{port}")
        
        # Test basic network connectivity first
        try:
            logger.info(f"🔍 Testing network connectivity to {host}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Network connectivity OK to {host}:{port}")
            else:
                logger.error(f"❌ Network connectivity FAILED to {host}:{port} - Error code: {result}")
                if result == 111:
                    logger.error("❌ Connection refused - database service may not be running")
                elif result == 110:
                    logger.error("❌ Connection timeout - check firewall/network rules")
                elif result == 101:
                    logger.error("❌ Network unreachable - check network configuration")
        except Exception as net_error:
            logger.error(f"❌ Network test failed: {net_error}")
        
        # Create the connection pool with pgBouncer compatibility
        logger.info("🏊 Creating asyncpg connection pool...")
        db_pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            command_timeout=30,
            server_settings={
                'application_name': 'celery_worker_render',
            },
            timeout=60,  # Connection timeout
            statement_cache_size=0  # 🔑 CRITICAL: Disable prepared statement caching for pgBouncer compatibility
        )
        logger.info("✅ Database pool created successfully")
        
        # Test the connection with a ping
        logger.info("🏓 Testing database connection with ping...")
        async with db_pool.acquire() as conn:
            # Simple ping query
            ping_result = await conn.fetchval('SELECT 1 as ping')
            logger.info(f"🏓 Database ping result: {ping_result}")
            
            # Get database info
            db_version = await conn.fetchval('SELECT version()')
            logger.info(f"🗄️  Database version: {db_version[:100]}...")  # Truncate long version strings
            
            # Test current timestamp
            current_time = await conn.fetchval('SELECT NOW()')
            logger.info(f"🕐 Database time: {current_time}")
            
            # Check if our main table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'document_sources')"
            )
            logger.info(f"📋 Table 'document_sources' exists: {table_exists}")
            
        logger.info("✅ Database connection verified and working!")
        
    except socket.gaierror as e:
        logger.error(f"❌ DNS resolution failed for {host}: {e}")
        logger.error("💡 Check if the database hostname is correct and reachable")
        raise
    except OSError as e:
        if e.errno == 101:
            logger.error(f"❌ Network unreachable to {host}:{port}")
            logger.error("💡 Possible causes:")
            logger.error("   - Database server is down")
            logger.error("   - Firewall blocking connection")
            logger.error("   - Wrong host/port in connection string")
            logger.error("   - Network routing issues")
        elif e.errno == 111:
            logger.error(f"❌ Connection refused by {host}:{port}")
            logger.error("💡 Database service may not be running or not accepting connections")
        elif e.errno == 110:
            logger.error(f"❌ Connection timeout to {host}:{port}")
            logger.error("💡 Database may be overloaded or firewall is dropping packets")
        logger.error(f"❌ OS-level connection error: {e}")
        raise
    except asyncpg.InvalidAuthorizationSpecificationError as e:
        logger.error(f"❌ Database authentication failed: {e}")
        logger.error("💡 Check username/password in POSTGRES_DSN")
        raise
    except asyncpg.InvalidCatalogNameError as e:
        logger.error(f"❌ Database does not exist: {e}")
        logger.error("💡 Check database name in POSTGRES_DSN")
        raise
    except asyncpg.DuplicatePreparedStatementError as e:
        logger.error(f"❌ Prepared statement conflict (pgBouncer issue): {e}")
        logger.error("💡 This should not happen with statement_cache_size=0")
        # Close the problematic pool and retry once
        if db_pool is not None:
            await db_pool.close()
            db_pool = None
        raise
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        raise
    
    return db_pool

# Create a sync database connection pool for gevent
sync_db_pool = None

# Get Synchronous DB pool
def get_sync_db_pool():
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

# ——— 5. OPTIONAL: MANUAL POOL RESET TASK ————————————————————————————————

@celery_app.task(bind=True)
def reset_db_pool_task(self):
    """Manual task to reset the database pool"""
    try:
        sync_close_db_pool()
        return {"status": "success", "message": "Database pool reset successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to reset database pool: {e}")
        return {"status": "error", "message": str(e)}

# ——— 6. Retry Strategies ————————————————————————————————

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
    """
    [DEPRECATED] Async helper to update a document's status in the database (for asyncio)
    """
    logger.info(f"📋 Doc {doc_id[:8]}... → {status.value}")
    pool = await get_async_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE document_sources
            SET vector_embed_status = $1, error_message = $2, updated_at = NOW()
            WHERE id = $3
            """,
            status.value, error_message, uuid.UUID(doc_id)
        )

def _update_document_status_sync(doc_id: str, status: ProcessingStatus, error_message: str = None):
    """Synchronous version of document status update helper (for gevent)"""
    logger.info(f"📋 Doc {doc_id[:8]}... → {status.value}")
    pool = get_sync_db_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_sources
                SET vector_embed_status = %s, error_message = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (status.value, error_message, doc_id)
            )
            conn.commit()
    finally:
        pool.putconn(conn)

def create_http_session() -> requests.Session:
    """Helper to create a requests session with retry strategy"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=HTTP_RETRY_STRATEGY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Optional: Add a utility function to check embedding reuse stats
async def get_embedding_reuse_stats(project_id: str) -> Dict[str, Any]:
    """
    Get statistics about embedding reuse for a project
    """
    pool = await get_async_db_pool()
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

def get_clean_filename_from_url(url: str, extension: str) -> str:
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

# ——— Task 1: Ingest (Fully Async) ———————————————————————————————————————————  

@celery_app.task(bind=True)
def test_celery_log_task(self) -> str:
    """Test task with multiple logging methods to ensure visibility"""
    task_id = self.request.id

    
    # Method 2: Celery task logger
    logger.info(f"📊 [TASK_LOGGER] Task {task_id} - testing Info level task logger working")
    logger.warning(f"⚠️  [TASK_LOGGER] Task {task_id} - Warning level message")
    logger.error(f"❌ [TASK_LOGGER] Task {task_id} - Error level message (not a real error)")
    
    # Return a success message
    result = f"Task {task_id} completed successfully with all logging methods tested"
    print(f"✅ [PRINT] {result}")
    
    return result


@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_document_batch_workflow(self, file_urls: List[str], metadata: Dict[str, Any], create_note: bool = False) -> Dict[str, Any]:
    """
    [ORCHESTRATOR] Master workflow coordinator for batch document processing:
    - Handles mixed new/reused document scenarios intelligently
    - Uses Celery's group + chord pattern for distributed coordination
    - Ensures exactly ONE note generation per batch regardless of document mix
    - Scales horizontally without database coordination bottlenecks
    
    Workflow Steps:
    1. Analyze batch: separate new vs reused documents
    2. Process new documents via async pipeline (group of tasks)
    3. Process reused documents via lightweight sync tasks  
    4. Coordinate completion via chord → single note generation
    """
    batch_id = str(uuid.uuid4())
    metadata['create_note'] = create_note
    logger.info(f"🎯 [BATCH-{batch_id[:8]}] 🚀 Starting workflow for {len(file_urls)} documents")
    
    # ——— Phase 1: Concurrent Download + Classification ————————————————————————————
    # Download all documents and classify as new vs reused (async for performance)
    return asyncio.run(_execute_batch_workflow(batch_id, file_urls, metadata))

async def _execute_batch_workflow(batch_id: str, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    [ASYNC COORDINATOR] Executes the full batch workflow:
    - Downloads and classifies documents concurrently (smart reuse feature)
    - Builds appropriate Celery workflow based on document types
    - Returns workflow execution results
    """
    project_id = metadata['project_id']
    user_id = metadata['user_id']
    
    # ——— Step 1: Concurrent Document Analysis ————————————————————————————————————
    logger.info(f"🔍 [BATCH-{batch_id[:8]}] Analyzing document types...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        analysis_tasks = [
            _analyze_document_for_workflow(client, url, project_id, user_id) 
            for url in file_urls
        ]
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    
    # ——— Step 2: Classify Documents by Processing Type ———————————————————————————
    new_documents = []
    reused_documents = []
    duplicate_documents = []  # Track same-project duplicates separately
    failed_downloads = []
    
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
    logger.info(f"   ♻️  Reused documents: {len(reused_documents)}")
    logger.info(f"   📋 Duplicate documents: {len(duplicate_documents)}")
    logger.info(f"   ❌ Failed downloads: {len(failed_downloads)}")

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

    processing_tasks = []
    
    # Calculate processable documents (EXCLUDE duplicates and failures)
    processable_docs = len(new_documents) + len(reused_documents)
    
    # Early exit if nothing to process
    if processable_docs == 0:
        logger.warning(f"⚠️ [BATCH-{batch_id[:8]}] No processable documents - all duplicates or failed")
        
        # Still trigger note generation if only duplicates (documents ALREADY exist in CURRENT USER project)
        if len(duplicate_documents) > 0 and metadata.get('create_note'):
            logger.info(f"📝 [BATCH-{batch_id[:8]}] Triggering note for duplicate-only batch")
            return await _handle_duplicate_only_batch(batch_id, workflow_metadata, duplicate_documents)
        
        # Otherwise handle as complete failure
        handle_batch_failure.apply_async((batch_id, workflow_metadata, failed_downloads))
        return {
            'batch_id': batch_id,
            'status': 'BATCH_FAILED',
            'reason': 'No processable documents (all duplicates or failed)'
        }

    # ——— Path A: All-Reused ⚡Fast Track ———————————————————————————————————————
    if len(reused_documents) == processable_docs and len(reused_documents) > 0:     
        logger.info(f"⚡ [BATCH-{batch_id[:8]}] All-reused batch - fast track processing")
        
        # Still need full reused document pipeline for row copying
        for doc_info in reused_documents:
            task_sig = process_reused_document_task.s(
                doc_info['existing_doc_id'],
                doc_info['doc_data'], 
                doc_info['project_id'],
                workflow_metadata
            )
            processing_tasks.append(task_sig)
            
        # Same coordination, just optimized for reused-only scenario
        logger.info(f"🚀 [BATCH-{batch_id[:8]}] Launching coordinated REUSED workflow with {len(processing_tasks)} tasks")
        workflow = chord(
            group(processing_tasks),
            finalize_batch_and_create_note.s(batch_id, workflow_metadata).set(queue=FINAL_QUEUE)
        )
        
        # Execute the workflow
        chord_result = workflow.apply_async()

        return {
            'batch_id': batch_id,
            'workflow_id': chord_result.id,
            'processing_tasks': len(processing_tasks),
            'workflow_path': 'ALL_REUSED_FAST_TRACK',
            'status': 'WORKFLOW_LAUNCHED'
        }

    
    # ——— Path B: Mixed/New Documents Standard Workflow ——————————————————————————
    else:      
        logger.info(f"🔄 [BATCH-{batch_id[:8]}] Mixed batch - standard workflow processing")
        
        # Add new document processing tasks (full async pipeline)
        for doc_info in new_documents:
            task_sig = process_new_document_task.s(
                doc_info['doc_data'], 
                doc_info['project_id'], 
                workflow_metadata
            )
            processing_tasks.append(task_sig)
        
        # Add reused document processing tasks (lightweight sync operations)
        for doc_info in reused_documents:
            task_sig = process_reused_document_task.s(
                doc_info['existing_doc_id'],
                doc_info['doc_data'],
                doc_info['project_id'],
                workflow_metadata
            )
            processing_tasks.append(task_sig)

        # Same coordination pattern
        logger.info(f"🚀 [BATCH-{batch_id[:8]}] Launching coordinated MIXED/NEW workflow with {len(processing_tasks)} tasks")
        workflow = chord(
            group(processing_tasks),
            finalize_batch_and_create_note.s(batch_id, workflow_metadata).set(queue=FINAL_QUEUE)
        )
        
        # Execute the workflow
        chord_result = workflow.apply_async()
        
        return {
            'batch_id': batch_id,
            'workflow_id': chord_result.id,
            'processing_tasks': len(processing_tasks),
            'workflow_path': 'MIXED_NEW_STANDARD',
            'status': 'WORKFLOW_LAUNCHED'
        }

async def _handle_duplicate_only_batch(batch_id: str, workflow_metadata: Dict[str, Any], duplicate_docs: List[Dict]) -> Dict[str, Any]:
    """
    [DUPLICATE HANDLER] Handle batches containing only duplicate documents:
    - Documents already EXIST in this project, so no processing needed (It;s a type of USER dumb error)
    - Can still generate notes using existing document content
    """
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

# ——— 🔍 Document Analysis & Classification ——————————————————————————————————————————

async def _analyze_document_for_workflow(
    client: httpx.AsyncClient, 
    url: str, 
    project_id: str, 
    user_id: str
) -> Dict[str, Any]:
    """
    [ASYNC] Download and analyze document to determine processing type:
    - NEW: Requires full processing pipeline (parse → embed → store)
    - REUSED: Existing processed content can be copied (smart reuse)
    
    Returns document metadata with processing classification
    """
    try:
        # ——— Download & Prepare Document (same as existing logic) ——————————————————
        # keep existing _download_and_prep_doc logic...
        doc_data = await _download_and_prep_doc(client, url, project_id, user_id)
        if not doc_data:
            raise Exception(f"Failed to download document from {url}")
        
        # ——— Check Processing Type (sync DB lookup) ———————————————————————————————
        pool = get_sync_db_pool()
        conn = pool.getconn()
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                content_hash = doc_data['content_hash']
                
                # First: Check same project (SEARCH for DUPLICATE documents in the CURRENT PROJECT)
                cur.execute(
                    'SELECT id FROM document_sources WHERE content_hash = %s AND project_id = %s',
                    (content_hash, project_id)
                )
                same_project_doc = cur.fetchone()
                
                if same_project_doc:
                    return {
                        'processing_type': 'DUPLICATE',
                        'existing_doc_id': str(same_project_doc['id']),
                        'doc_data': doc_data,
                        'project_id': project_id
                    }
                
                # Second: Check for reusable processed content (SEARCH for matching CONTENT_HASH)
                cur.execute(
                    '''SELECT id, project_id, total_chunks 
                       FROM document_sources 
                       WHERE content_hash = %s AND vector_embed_status = %s AND total_chunks > 0
                       LIMIT 1''',
                    (content_hash, ProcessingStatus.COMPLETE.value)
                )
                existing_processed = cur.fetchone()
                
                if existing_processed:
                    return {
                        'processing_type': 'REUSED',
                        'existing_doc_id': str(existing_processed['id']),
                        'doc_data': doc_data,
                        'project_id': project_id,
                        'chunks_available': existing_processed['total_chunks']
                    }
                else:
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
    [ULTRA-OPTIMIZED] Also used in the ultra-low latency path for 100% reused documents
    
    Args:
        existing_source_id: Source ID of the already-processed document
        new_source_id: Source ID of the new document entry
        project_id: Target project ID
        user_id: User who uploaded the document
    
    Returns:
        Dict with copy statistics
    """
    pool = get_sync_db_pool()
    conn = pool.getconn()
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get vector embeddings from the existing document (via matching source_id)
            cur.execute(
                '''
                SELECT content, metadata, embedding, num_tokens 
                FROM document_vector_store 
                WHERE source_id = %s AND embedding IS NOT NULL
                ORDER BY created_at
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
            
            # Copy each existing embedding row to assocuate with the new user's project
            for embedding_row in existing_embeddings:
                records_to_insert.append((
                    str(uuid.uuid4()),               # id (converted to string)
                    new_source_id,                   # source_id (already string)
                    project_id,                      # project_id NEW DESTINATION (already string) 
                    embedding_row['content'],        # content   (reuse existing)
                    embedding_row['metadata'],       # metadata  (keep original)
                    embedding_row['embedding'],      # embedding (reuse existing)
                    embedding_row['num_tokens'],     # num_tokens
                    user_id,                         # user_id NEW USER (already string)
                    datetime.now(timezone.utc)       # created_at
                ))
                total_tokens += embedding_row['num_tokens'] or 0
            
            # Bulk insert the copied embeddings
            cur.executemany(
                '''INSERT INTO document_vector_store 
                (id, source_id, project_id, content, metadata, embedding, num_tokens, user_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
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
    Helper to download to memory, hash, stream to S3 & Store in AWS CLoudfront, and prep data.
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
            filename = get_clean_filename_from_url(url, ext)
            s3_key = f"{project_id}/{uuid.uuid4()}{ext}"
            
            # Stream directly to S3 && AWS CloudFront from the in-memory buffer
            upload_to_s3(s3_client, content_stream, s3_key)
            cdn_url = get_cloudfront_url(s3_key)
            
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

# ——— Parsing & Embedding: For one document → Parse, batch, dispatch to OpenAI for emmbedding (In-Memory & Token-Aware) ————————————————————————————————

def _parse_document_gevent_for_workflow(self, source_id: str, cdn_url: str, project_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    [WORKFLOW OPTIMIZED] Modified version of _parse_document_gevent for chord integration:
    - Removes the chord scheduling at the end (handled by workflow coordinator)
    - Returns parsing results directly for batch coordination
    - Keeps all the performance optimizations and gevent benefits
    """
    _update_document_status_sync(source_id, ProcessingStatus.PARSING)
    
    short_id = source_id[:8]
    file_buffer = io.BytesIO()
    doc_metrics = DocumentMetrics(doc_id=source_id)
    perf_summary = {}

    # Total `_parse_document_gevent` task Timer Wrapper
    with Timer() as total_timer:
        try:
            # Phase 1: Gevent-optimized streaming download
            with Timer() as download_timer:
                # Helper to create HTTP request sessions with retry strategy
                session = create_http_session()
                
                # Stream the file with gevent-friendly requests
                logger.info(f"🚀 Starting document streaming for {source_id}, with url: {cdn_url}")
                response = session.get(cdn_url, headers=DEFAULT_HEADERS, stream=True, timeout=120)
                response.raise_for_status()

                # Log response headers for debugging
                content_length = response.headers.get('content-length')
                content_type = response.headers.get('content-type', 'unknown')
                
                # Stream into memory buffer
                downloaded_bytes = 0
                chunk_count = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_buffer.write(chunk)
                        downloaded_bytes += len(chunk)
                        chunk_count += 1
                        
                        # Log progress every 5MB or 500 chunks
                        if chunk_count % 500 == 0 or downloaded_bytes % (1024 * 1024 * 5) == 0:
                            logger.info(f"📥 [PARSE-{short_id}] Downloaded {downloaded_bytes:,} bytes ({chunk_count} chunks)")
                        
                        gevent.sleep(0)

                file_buffer.seek(0)
                doc_metrics.download_time_ms = download_timer.elapsed_ms
                doc_metrics.file_size_bytes = file_buffer.getbuffer().nbytes
                perf_summary['download_ms'] = download_timer.elapsed_ms

            # Phase 2: Document Analysis & Optimal Loader Selection (CPU-bound, but fast)
            with Timer() as analysis_timer:
                # Quick analysis to choose best processing strategy
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
                
                # Add to performance bench summary dict
                perf_summary['analysis_ms'] = analysis_timer.elapsed_ms

            # Phase 3: Optimized Text Processing, Cleaning, & Chunking
            with Timer() as process_timer:
                # Create optimized processor (class uses "pseudo-Semantic Chunking" using RecursiveCharacterTextSplitter)
                processor = create_optimized_processor(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    performance_mode=processor_mode
                )
                
                # Stream processing with real-time metrics
                all_texts, all_metadatas = [], []
                
                logger.info(f"🔄 [PARSE-{short_id}] Starting streaming processing...")
                
                # Process document stream into chunks (and associated metadata)
                document_stream = loader.stream_documents(file_buffer)
                text_stream = processor.process_documents_streaming(document_stream, source_id)
                for chunk_text, chunk_metadata in text_stream:
                    all_texts.append(chunk_text)
                    all_metadatas.append(chunk_metadata)
                    
                    # Log progress every 200 chunks
                    if len(all_texts) % 200 == 0:
                        logger.info(f"✂️ [PARSE-{short_id}] Created {len(all_texts)} chunks")
                    
                    gevent.sleep(0)
                
                # [DEBUG] Print statement at the end of document chunking (i.e. the full 13.9MB)
                logger.info(f"⏱️ PARSING COMPLETE...")
                
                # Get processing performance summary
                processing_summary = processor.get_performance_summary()
                perf_summary.update({
                    'text_processing_ms': process_timer.elapsed_ms,
                    'total_chunks': len(all_texts),
                    'total_pages': processing_summary['total_pages'],
                    'chars_per_second': processing_summary['chars_per_second'],
                    'chunks_per_second': processing_summary['chunks_per_second']
                })
                
                logger.info(f"✅ [PARSE-{short_id}] Processing complete: {len(all_texts)} chunks from {processing_summary['total_pages']} pages")

            # Phase 4: Optimized Token Counting & Embedding Batching
            with Timer() as token_timer:
                # Batch token counting for efficiency
                token_counter = BatchTokenCounter(tokenizer)
                token_counts = token_counter.count_tokens_batch(all_texts)
                
                perf_summary['token_counting_ms'] = token_timer.elapsed_ms
                total_tokens = sum(token_counts)
                logger.info(f"🔢 [PARSE-{short_id}] Token counting: {total_tokens:,} tokens in {perf_summary['token_counting_ms']:.1f}ms")

            with Timer() as batch_timer:
                embedding_tasks = []
                current_batch_texts, current_batch_metas = [], []
                current_batch_tokens = 0

                for i, (text, token_count) in enumerate(zip(all_texts, token_counts)):
                    if current_batch_tokens + token_count > OPENAI_MAX_TOKENS_PER_BATCH and current_batch_texts:
                        task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
                        embedding_tasks.append(task_sig)
                        
                        if len(embedding_tasks) % 5 == 0:  # Log every 5 batches
                            logger.info(f"📦 [PARSE-{short_id}] Created {len(embedding_tasks)} batches")
                        
                        current_batch_texts, current_batch_metas, current_batch_tokens = [], [], 0

                    current_batch_texts.append(text)
                    current_batch_metas.append(all_metadatas[i])
                    current_batch_tokens += token_count

                if current_batch_texts:
                    task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
                    embedding_tasks.append(task_sig)

                perf_summary['batching_ms'] = batch_timer.elapsed_ms
                perf_summary['total_batches'] = len(embedding_tasks)

            # ——— Phase 5: Modified Database Update (no chord scheduling) ——————————————
            pool = get_sync_db_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Enhanced metrics for database
                    enhanced_metrics = {
                        **doc_metrics.to_dict(),
                        **perf_summary,
                        'processing_mode': processor_mode,
                        'loader_type': type(loader).__name__,
                        'workflow_integration': True  # Flag for workflow processing
                    }
                    
                    cur.execute(
                        """
                        UPDATE document_sources
                        SET vector_embed_status = %s, total_chunks = %s, total_batches = %s, 
                            processing_metadata = processing_metadata || %s::jsonb
                        WHERE id = %s""",
                        (ProcessingStatus.EMBEDDING.value, len(all_texts), 
                            len(embedding_tasks), json.dumps(enhanced_metrics), source_id)
                    )
                    conn.commit()
            finally:
                pool.putconn(conn)
            
            # ——— Schedule Embedding Tasks and Wait for Completion ————————————————————
            if embedding_tasks:
                logger.info(f"🚀 [PARSE-{short_id}] Waiting for {len(embedding_tasks)} embedding tasks...")
                
                try:
                    # Execute embeddings and wait for ALL to complete
                    embedding_group = group(embedding_tasks)
                    embedding_results = embedding_group.apply_async().get()  # Wait for completion
                    
                    # Analyze embedding results
                    successful_embeddings = [r for r in embedding_results if r and r.get('processed_count', 0) > 0]
                    failed_embeddings = len(embedding_tasks) - len(successful_embeddings)
                    
                    if len(successful_embeddings) == 0:
                        # All embeddings failed
                        logger.error(f"💥 [PARSE-{short_id}] All embedding batches failed")
                        _update_document_status_sync(source_id, ProcessingStatus.FAILED_EMBEDDING)
                        return {
                            'source_id': source_id,
                            'status': 'FAILED',
                            'error': 'All embedding batches failed',
                            'total_chunks': len(all_texts),
                            'failed_batches': len(embedding_tasks)
                        }
                    elif failed_embeddings > 0:
                        # Partial success
                        logger.warning(f"⚠️ [PARSE-{short_id}] Partial embedding success: {len(successful_embeddings)}/{len(embedding_tasks)} batches")
                        _update_document_status_sync(source_id, ProcessingStatus.PARTIAL)
                        return {
                            'source_id': source_id,
                            'status': 'PARTIAL',
                            'successful_batches': len(successful_embeddings),
                            'failed_batches': failed_embeddings,
                            'total_chunks': len(all_texts)
                        }
                    else:
                        # Complete success
                        logger.info(f"✅ [PARSE-{short_id}] All embeddings successful: {len(successful_embeddings)} batches")
                        _update_document_status_sync(source_id, ProcessingStatus.COMPLETE)
                        return {
                            'source_id': source_id,
                            'status': 'COMPLETE',
                            'total_chunks': len(all_texts),
                            'successful_batches': len(successful_embeddings),
                            'performance_metrics': perf_summary
                        }
                        
                except Exception as embedding_error:
                    # Embedding group execution failed
                    logger.error(f"💥 [PARSE-{short_id}] Embedding group execution failed: {embedding_error}")
                    _update_document_status_sync(source_id, ProcessingStatus.FAILED_EMBEDDING, str(embedding_error))
                    return {
                        'source_id': source_id,
                        'status': 'FAILED',
                        'error': f'Embedding group execution failed: {str(embedding_error)}',
                        'total_chunks': len(all_texts)
                    }
            else:
                # No embedding tasks created (edge case)
                logger.warning(f"⚠️ [PARSE-{short_id}] No embedding tasks created - empty document?")
                _update_document_status_sync(source_id, ProcessingStatus.COMPLETE)
                return {
                    'source_id': source_id,
                    'status': 'COMPLETE',
                    'total_chunks': 0,
                    'successful_batches': 0,
                    'performance_metrics': perf_summary
                }

        except Exception as e:
            logger.error(f"💥 [PARSE-{short_id}] WORKFLOW PARSING FAILED: {e}", exc_info=True)
            _update_document_status_sync(source_id, ProcessingStatus.FAILED_PARSING, str(e))
            raise

@celery_app.task(
    bind=True, 
    queue=PARSE_QUEUE, 
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': MAX_RETRIES, 'countdown': DEFAULT_RETRY_DELAY}
)

# ——— Task 3: Embed (Fully Async DB) ——————————————————————————————————————————

@celery_app.task(
    bind=True, 
    queue=EMBED_QUEUE, 
    acks_late=True,
    rate_limit=RATE_LIMIT,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': MAX_RETRIES, 'countdown': DEFAULT_RETRY_DELAY}
)
def embed_batch_task(self, source_id: str, project_id: str, texts: List[str], metadatas: List[Dict]):
    try:
        return _embed_batch_gevent(self, source_id, project_id, texts, metadatas)
    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed, letting Celery handle retry: {exc}")
        raise

def _embed_batch_gevent(self, source_id: str, project_id: str, texts: List[str], metadatas: List[Dict]):
    """
    Gevent-based embedding implementation - more reliable with Celery (does this have retry/circuit breaker logic?)
    """
    try:
        # 1. Generate Embeddings using sync OpenAI client
        logger.info(f"🤖 Generating embeddings...")
        
        # Use synchronous OpenAI client instead of async
        embeddings = embedding_model.embed_documents(texts)  # ← SYNC method (no await)
        
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
                str(uuid.uuid4()), str(uuid.UUID(source_id)), str(uuid.UUID(project_id)), text, 
                json.dumps(meta), vec, token_count, datetime.now(timezone.utc)
            ))

        if not records_to_insert:
            return {'processed_count': 0, 'token_count': 0}

        # 3. Use sync database pool (like Task 2)
        pool = get_sync_db_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                # Use executemany for bulk insert
                cur.executemany(
                    '''INSERT INTO document_vector_store 
                    (id, source_id, project_id, content, metadata, embedding, num_tokens, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                    records_to_insert
                )
                conn.commit()
        finally:
            pool.putconn(conn)
        
        return {'processed_count': len(records_to_insert), 'token_count': total_tokens}

    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed (attempt {self.request.retries + 1}), retrying: {exc}")
        raise self.retry(exc=exc, countdown=DEFAULT_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** self.request.retries))

# ——— Specialized Processing Tasks ————————————————————————————————————————————————

@celery_app.task(bind=True, queue=INGEST_QUEUE, acks_late=True)
def process_new_document_task(self, doc_data: Dict[str, Any], project_id: str, workflow_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    [NEW DOC PIPELINE] Process completely new document through full pipeline:
    - Insert document record `public.document_sources`
    - Pass document to `_parse_document_gevent_for_workflow` to → Fetch Loader → Parse → Embed → Store vectors
    - Returns processing results for workflow coordination
    """
    doc_id = str(uuid.uuid4())
    doc_data['id'] = doc_id
    
    try:
        # ——— Insert Document Record ———————————————————————————————————————————————
        pool = get_sync_db_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    '''INSERT INTO document_sources 
                    (id, cdn_url, content_hash, project_id, content_tags, uploaded_by, 
                    vector_embed_status, filename, file_size_bytes, file_extension, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (doc_id, doc_data['cdn_url'], doc_data['content_hash'], 
                    project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                    ProcessingStatus.PENDING.value, doc_data['filename'], 
                    doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                    datetime.now(timezone.utc))
                )
                conn.commit()
        finally:
            pool.putconn(conn)
        
        # ——— Launch Full Processing Pipeline ——————————————————————————————————————
        # Use existing _parse_document_gevent_for_workflow function to (select optimal loader → clean/chunk → embed)
        parse_result = _parse_document_gevent_for_workflow(
            self, doc_id, doc_data['cdn_url'], project_id, workflow_metadata
        )
        
        # ——— Check Parse Result Status and Return Appropriately ——————————————————
        if parse_result.get('status') == 'COMPLETE':
            logger.info(f"✅ [DOC-{doc_id[:8]}] New document processing complete")
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'COMPLETE',
                'chunks_created': parse_result.get('total_chunks', 0),
                'performance_metrics': parse_result.get('performance_metrics', {})
            }
        elif parse_result.get('status') == 'PARTIAL':
            logger.warning(f"⚠️ [DOC-{doc_id[:8]}] New document processing partially successful")
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'PARTIAL',
                'chunks_created': parse_result.get('total_chunks', 0),
                'successful_batches': parse_result.get('successful_batches', 0),
                'failed_batches': parse_result.get('failed_batches', 0)
            }
        else:  # FAILED
            logger.error(f"❌ [DOC-{doc_id[:8]}] New document processing failed")
            return {
                'doc_id': doc_id,
                'processing_type': 'NEW',
                'status': 'FAILED',
                'error': parse_result.get('error', 'Unknown parsing failure'),
                'chunks_created': parse_result.get('total_chunks', 0)
            }
        
    except Exception as e:
        logger.error(f"❌ [DOC-{doc_id[:8]}] New document processing failed: {e}")
        _update_document_status_sync(doc_id, ProcessingStatus.FAILED_PARSING, str(e))
        return {
            'doc_id': doc_id,
            'processing_type': 'NEW', 
            'status': 'FAILED',
            'error': str(e)
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
        pool = get_sync_db_pool()
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
                     total_chunks, total_batches, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (new_doc_id, doc_data['cdn_url'], doc_data['content_hash'], 
                     project_id, doc_data.get('content_tags', []), workflow_metadata['user_id'],
                     ProcessingStatus.COMPLETE.value, doc_data['filename'], 
                     doc_data['file_size_bytes'], os.path.splitext(doc_data['filename'])[1].lower(),
                     source_info['total_chunks'], source_info['total_batches'],
                     datetime.now(timezone.utc))
                )
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
        logger.error(f"❌ [DOC-{new_doc_id[:8]}] Reused document processing failed: {e}")
        return {
            'doc_id': new_doc_id,
            'processing_type': 'REUSED',
            'status': 'FAILED', 
            'error': str(e)
        }

# ——— Finalization: Batch Coordination & Note Generation ————————————————————————————————————————

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def finalize_batch_and_create_note(
    self, 
    processing_results: List[Dict[str, Any]], 
    batch_id: str, 
    workflow_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [BATCH COORDINATOR] Final coordination point for entire batch:
    - Analyzes all document processing results
    - Determines batch success/partial/failure status  
    - Triggers single RAG note generation with appropriate context
    - Handles all resilience cases (full success, partial, complete failure)
    
    This is the ONLY place where rag_note_task gets triggered per batch!
    """
    logger.info(f"🎯 [BATCH-{batch_id[:8]}] Finalizing batch with {len(processing_results)} results")
    
    # ——— Analyze Batch Results ————————————————————————————————————————————————————
    successful_docs = [r for r in processing_results if r and r.get('status') == 'COMPLETE']
    failed_docs = [r for r in processing_results if r and r.get('status') == 'FAILED']
    
    total_docs = workflow_metadata['total_documents']
    success_count = len(successful_docs)
    failure_count = len(failed_docs)
    
    # Calculate batch statistics
    total_chunks = sum(
        r.get('chunks_created', 0) + r.get('chunks_reused', 0) 
        for r in successful_docs
    )
    total_tokens_reused = sum(r.get('tokens_reused', 0) for r in successful_docs)
    
    # ——— Determine Batch Status ———————————————————————————————————————————————————
    if success_count == total_docs:
        batch_status = 'COMPLETE'
        note_context = 'All documents processed successfully'
    elif success_count > 0:
        batch_status = 'PARTIAL' 
        note_context = f'{success_count}/{total_docs} documents processed successfully'
    else:
        batch_status = 'FAILED'
        note_context = 'All documents failed to process'
    
    logger.info(f"📊 [BATCH-{batch_id[:8]}] Batch analysis:")
    logger.info(f"   ✅ Successful: {success_count}/{total_docs}")
    logger.info(f"   ❌ Failed: {failure_count}")
    logger.info(f"   📄 Total chunks: {total_chunks:,}")
    logger.info(f"   🔄 Status: {batch_status}")
    
    # ——— Trigger Note Generation (Based on Resilience Rules) ——————————————————————
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

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def handle_batch_failure(self, batch_id: str, workflow_metadata: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    """
    [FAILURE HANDLER] Handle complete batch failure scenarios:
    - Log comprehensive failure information
    - Could store failure record for debugging/monitoring
    - No note generation for complete failures
    """
    logger.error(f"💥 [BATCH-{batch_id[:8]}] Complete batch failure:")
    for i, error in enumerate(errors, 1):
        logger.error(f"   {i}. {error}")
    
    # Optional: Store failure record in database for monitoring
    # This could be useful for debugging and user feedback
    
    return {
        'batch_id': batch_id,
        'batch_status': 'COMPLETE_FAILURE',
        'errors': errors,
        'note_generation_triggered': False
    }

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
    # ——— Main Workflow Tasks ———————————————————————————————————————————————————————
    'process_document_batch_workflow',           # NEW: Main entry point (replaces process_document_task)
    'process_new_document_task',                 # NEW: Handle new documents in workflow
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
    '_analyze_document_for_workflow',            # NEW: Document classification
    '_parse_document_gevent_for_workflow',       # NEW: Workflow-optimized parsing
    '_handle_duplicate_only_batch',              # NEW: Handle all-duplicate scenarios
    
    # ——— System Management Tasks ———————————————————————————————————————————————————
    'test_celery_log_task',                      # KEEP: Testing functionality
    'system_health_check',                       # KEEP: If you have this
    'cleanup_orphaned_resources',                # KEEP: If you have this
    'optimize_embedding_performance',            # KEEP: If you have this
    'initialize_production_pipeline',            # KEEP: If you have this
    'validate_production_readiness'              # KEEP: If you have this
]
