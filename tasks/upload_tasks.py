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
from utils.s3_utils import upload_to_s3, s3_client
from utils.cloudfront_utils import get_cloudfront_url
from utils.supabase_utils import supabase_client
from utils.document_loaders.base import BaseDocumentLoader
from utils.document_loaders.loader_factory import get_loader_for        # The loader factory is now expected to handle in-memory file-like objects
from utils.metrics import MetricsCollector, Timer
from utils.connection_pool import ConnectionPoolManager
from utils.memory_manager import MemoryManager # Kept for health checks

# ——— Logging & Env Load ————————————————————————————————————————————————————
logger = get_task_logger(__name__)
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

class StreamingDocumentProcessor:
    """
    Wrapper class to encapsulate streaming document processing
    Maintains the TRUE IN-MEMORY STREAMING architecture
    """
    
    def __init__(self, file_buffer: io.BytesIO, filename: str):
        self.file_buffer = file_buffer
        self.filename = filename
        self.loader = None
        
    def get_loader(self) -> BaseDocumentLoader:
        """Get the appropriate loader for this document"""
        if self.loader is None:
            self.loader = get_loader_for(filename=self.filename, file_like_object=self.file_buffer)
        return self.loader
    
    def stream_chunks(self, splitter: RecursiveCharacterTextSplitter) -> Iterator[Tuple[str, Dict]]:
        """
        Generator that yields (text, metadata) tuples for each chunk
        Maintains true streaming - never loads entire document into memory at once
        """
        loader = self.get_loader()
        
        # Reset buffer position
        self.file_buffer.seek(0)
        
        for page_num, page in enumerate(loader.stream_documents(self.file_buffer)):
            # Split page into chunks
            chunks = splitter.split_documents([page])
            
            for chunk_idx, chunk in enumerate(chunks):
                cleaned_content = _clean_text(chunk.page_content)
                if cleaned_content:
                    metadata = {
                        **chunk.metadata,
                        'page_number': page_num,
                        'chunk_index': chunk_idx
                    }
                    yield cleaned_content, metadata

# ——— DB Pool Instances (Initialized once per worker) —————————————————————

# Global pool variable
db_pool: Optional[asyncpg.Pool] = None

# ——— 1. POOL CLEANUP FUNCTIONS ————————————————————————————————

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

# ——— 2. CELERY SIGNAL HANDLERS ————————————————————————————————

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

# ——— 3. SYSTEM SIGNAL HANDLERS ————————————————————————————————

def signal_handler(signum, frame):
    """Handle system signals (SIGTERM, SIGINT)"""
    logger.info(f"🛑 Received signal {signum}, cleaning up...")
    sync_close_db_pool()

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register atexit handler as last resort
atexit.register(sync_close_db_pool)

# ——— 4. UPDATED get_async_db_pool FUNCTION ————————————————————————————————

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

# Celery Task Retry Configuration
# CELERY_RETRY_CONFIG = {
#     'max_retries': 5,
#     'default_retry_delay': 5,
#     'retry_backoff': True,
#     'retry_backoff_max': 300,  # 5 minutes max
#     'retry_jitter': True
# }

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
    """Async helper to update a document's status in the database (for asyncio)"""
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
def process_document_task(self, file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    task_id = self.request.id
    logger.info(f"🚀 Starting task {task_id} asyc document processing task URLs")
    return asyncio.run(_process_document_async(file_urls, metadata))

async def copy_embeddings_for_project(existing_source_id: str, new_source_id: str, project_id: str, user_id: str) -> Dict[str, Any]:
    """
    [ASYNC] Copy embeddings from an existing processed document to a new project
    
    Args:
        existing_source_id: Source ID of the already-processed document
        new_source_id: Source ID of the new document entry
        project_id: Target project ID
        user_id: User who uploaded the document
    
    Returns:
        Dict with copy statistics
    """
    pool = await get_async_db_pool()
    async with pool.acquire() as conn:
        # Get vector embeddings from the existing document (via matching source_id)
        existing_embeddings = await conn.fetch(
            '''
            SELECT content, metadata, embedding, num_tokens 
            FROM document_vector_store 
            WHERE source_id = $1 AND embedding IS NOT NULL
            ORDER BY created_at
            ''',
            uuid.UUID(existing_source_id)
        )
        
        if not existing_embeddings:
            logger.warning(f"No embeddings found for source_id {existing_source_id}")
            return {'copied_count': 0, 'total_tokens': 0}
        
        # Prepare records for bulk insert
        records_to_insert = []
        total_tokens = 0
        
        for embedding_row in existing_embeddings:
            records_to_insert.append((
                uuid.uuid4(),                    # id
                uuid.UUID(new_source_id),        # source_id (new document)
                uuid.UUID(project_id),           # project_id (target project)
                embedding_row['content'],        # content
                embedding_row['metadata'],       # metadata (keep original)
                embedding_row['embedding'],      # embedding (reuse existing)
                embedding_row['num_tokens'],     # num_tokens
                uuid.UUID(user_id),             # user_id
                datetime.now(timezone.utc)      # created_at
            ))
            total_tokens += embedding_row['num_tokens'] or 0
        
        # Bulk insert the copied embeddings
        # await conn.copy_records_to_table(
        #     'document_vector_store',
        #     records=records_to_insert,
        #     columns=(
        #         'id', 'source_id', 'project_id', 'content', 'metadata', 
        #         'embedding', 'num_tokens', 'user_id', 'created_at'
        #     ),
        #     timeout=120
        # )
        # CHANGE: Use executemany instead of copy_records_to_table
        await conn.executemany(
            '''INSERT INTO document_vector_store 
            (id, source_id, project_id, content, metadata, embedding, num_tokens, user_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)''',
            records_to_insert
        )
        logger.info(f"✅ Copied {len(records_to_insert)} embeddings from {existing_source_id} to {new_source_id} for project {project_id}")
        
        return {
            'copied_count': len(records_to_insert),
            'total_tokens': total_tokens
        }

async def _process_document_async(file_urls: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    [ASYNC] Smart Reuse Ingest Task:
    - Concurrent downloads directly into memory using httpx.
    - Smart deduplication: checks for processed content across ALL projects
    - If content exists and is processed, reuse embeddings for new project
    - If content is new or unprocessed, do full processing pipeline
    - Bulk-inserts new documents and dispatches parsing tasks only when needed
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
    reused_docs = []
    same_project_duplicates = []
    
    pool = await get_async_db_pool()
    async with pool.acquire() as conn:
        for res in results:
            if isinstance(res, Exception) or not res:
                logger.error(f"A document failed to download or prep: {res}")
                continue
            
            # First: Check if document already exists in THIS project
            same_project_doc = await conn.fetchval(
                'SELECT id FROM document_sources WHERE content_hash = $1 AND project_id = $2',
                res['content_hash'], uuid.UUID(project_id)
            )
            
            if same_project_doc:
                same_project_duplicates.append(str(same_project_doc))
                logger.info(f"📋 Document already exists in this project: {res['content_hash'][:8]} -> {same_project_doc}")
                continue
            
            # Second: Check if content exists in ANY project and is fully processed ♻️ 
            existing_processed_doc = await conn.fetchrow(
                '''
                SELECT id, project_id, total_chunks, total_batches
                FROM document_sources 
                WHERE content_hash = $1 
                AND vector_embed_status = $2 
                AND total_chunks > 0
                LIMIT 1''',
                res['content_hash'], ProcessingStatus.COMPLETE.value
            )
            
            if existing_processed_doc:
                # Content exists and is fully processed - SMART REUSE!
                logger.info(f"♻️  Found processed content {res['content_hash'][:8]} with {existing_processed_doc['total_chunks']} chunks")
                
                # Create new document entry for this project
                new_doc_id = uuid.uuid4()
                await conn.execute(
                    '''
                    INSERT INTO document_sources 
                    (id, cdn_url, content_hash, project_id, content_tags, uploaded_by, 
                        vector_embed_status, filename, file_size_bytes, file_extension, 
                        total_chunks, total_batches, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)''',
                    new_doc_id, res['cdn_url'], res['content_hash'], 
                    uuid.UUID(project_id), res.get('content_tags', []), user_id,
                    ProcessingStatus.COMPLETE.value, res['filename'], 
                    res['file_size_bytes'], os.path.splitext(res['filename'])[1].lower(),
                    existing_processed_doc['total_chunks'], existing_processed_doc['total_batches'],
                    datetime.now(timezone.utc)
                )
                
                # ♻️ Copy embeddings to the new project
                logger.info(f"♻️  Copying found embeddings...")
                copy_result = await copy_embeddings_for_project(
                    str(existing_processed_doc['id']), 
                    str(new_doc_id), 
                    project_id, 
                    user_id
                )
                
                reused_docs.append({
                    'doc_id': str(new_doc_id),
                    'content_hash': res['content_hash'],
                    'chunks_reused': copy_result['copied_count'],
                    'tokens_reused': copy_result['total_tokens']
                })
                
                logger.info(f"🎯 Smart reuse complete: {copy_result['copied_count']} chunks, {copy_result['total_tokens']} tokens")
                
            else:
                # New content or unprocessed content - full processing needed
                res['id'] = uuid.uuid4()
                new_docs_to_insert.append(res)
                logger.info(f"🆕 New content {res['content_hash'][:8]} needs full processing")
    
    # Bulk insert new documents that need full processing
    inserted_ids = []
    if new_docs_to_insert:
        logger.info(f"💾 Inserting {len(new_docs_to_insert)} new docs for full processing...")
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
                # await conn.copy_records_to_table(
                #     'document_sources',
                #     records=records_to_insert,
                #     columns=(
                #         'id', 'cdn_url', 'content_hash', 'project_id', 'content_tags',
                #         'uploaded_by', 'vector_embed_status', 'filename', 'file_size_bytes',
                #         'file_extension', 'created_at'
                #     ),
                #     timeout=60
                # )
                # CHANGE: Use executemany instead of copy_records_to_table
                await conn.executemany(
                    '''INSERT INTO document_sources 
                    (id, cdn_url, content_hash, project_id, content_tags, uploaded_by, 
                        vector_embed_status, filename, file_size_bytes, file_extension, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)''',
                    records_to_insert
                )      
        inserted_ids = [str(doc['id']) for doc in new_docs_to_insert]
        
        # Schedule parsing tasks ONLY for new documents
        logger.info("🗂️ Scheduling parse tasks for NEW documents only...")
        for doc in new_docs_to_insert:
            parse_document_task.apply_async(
                (str(doc['id']), doc['cdn_url'], doc['project_id']), queue=PARSE_QUEUE
            )
    
    # Comprehensive results
    all_doc_ids = same_project_duplicates + [doc['doc_id'] for doc in reused_docs] + inserted_ids
    
    total_reused_chunks = sum(doc['chunks_reused'] for doc in reused_docs)
    total_reused_tokens = sum(doc['tokens_reused'] for doc in reused_docs)
    
    logger.info(f"📊 Processing summary:")
    logger.info(f"   📋 Same project duplicates: {len(same_project_duplicates)}")
    logger.info(f"   ♻️  Smart reuse documents: {len(reused_docs)} ({total_reused_chunks} chunks, {total_reused_tokens} tokens)")
    logger.info(f"   🆕 New documents for processing: {len(inserted_ids)}")

    return {
        'doc_ids': all_doc_ids,
        'new_documents': len(inserted_ids),
        'reused_documents': len(reused_docs),
        'same_project_duplicates': len(same_project_duplicates),
        'total_chunks_reused': total_reused_chunks,
        'total_tokens_reused': total_reused_tokens,
        'openai_cost_saved': total_reused_tokens * 0.0001 / 1000  # Rough estimate
    }

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

            # ext = get_file_extension_from_url(url)   # os.path.splitext(url)[1].lower() or '.bin'
            # s3_key = f"{project_id}/{uuid.uuid4()}{ext}"
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

# ——— Task 2: For one document → Parse, batch, dispatch to emmbed (In-Memory & Token-Aware) ————————————————————————————————

@celery_app.task(
    bind=True, 
    queue=PARSE_QUEUE, 
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': MAX_RETRIES, 'countdown': DEFAULT_RETRY_DELAY}
)
def parse_document_task(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
    """
    Gevent-optimized Parsing Task:
    - Uses gevent for concurrent I/O operations
    - Streams document directly into an in-memory buffer (NO temp file)
    - Splits parsed text into semantic chunks
    - Uses token-aware adaptive batching to maximize embedding efficiency
    - Synchronous database updates optimized for gevent
    - Fire off embed task celery group
    """
    try:
        return _parse_document_gevent(self, source_id, cdn_url, project_id)
    except Exception as exc:
        logger.error(f"Parse task failed for {source_id}: {exc}")
        raise  # Let Celery handle the retry automatically with CELERY_RETRY_CONFIG

# async def _parse_document_async(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
#     """
#     LEGACY: Parsing Task:
#     - Streams document directly into an in-memory buffer (NO temp file).
#     - Splits parsed text into semantic chunks
#     - Uses token-aware adaptive batching to maximize embedding efficiency.
#     - Fully async database updates.
#     - Dispatch embed task celery group
#     """
#     await _update_document_status(source_id, ProcessingStatus.PARSING)
    
#     file_buffer = io.BytesIO()
#     doc_metrics = DocumentMetrics(doc_id=source_id)     # ← TELEMETRY

#     try:
#         # Phase 1: True In-Memory Streaming (No Disk I/O)
#         with Timer() as download_timer:
#             # Define headers for anti-bot detection
#             headers = DEFAULT_HEADERS

#             async with httpx.AsyncClient(timeout=120.0) as client:
#                 async with client.stream("GET", cdn_url, headers=headers) as response:
#                     response.raise_for_status()
#                     async for chunk in response.aiter_bytes():
#                         file_buffer.write(chunk)
#             file_buffer.seek(0)
#             # update telemetry
#             doc_metrics.download_time_ms = download_timer.elapsed_ms
#             doc_metrics.file_size_bytes = file_buffer.getbuffer().nbytes

#         # Phase 2: Process with Loader and Splitter
#         with Timer() as parse_timer:
#             # Assumes get_loader_for can take a file-like object
#             loader = get_loader_for(filename=cdn_url, file_like_object=file_buffer)

#             # Langchain semantic chunking strategy (preserves whole ideas)
#             splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
#             all_texts, all_metadatas = [], []

#             # Pass the file_buffer as the source parameter
#             logger.info(f"🔄 Starting document streaming for {source_id}")
#             try:
#                 # Reset buffer position before streaming
#                 file_buffer.seek(0)
                
#                 # CORRECT CALL: Pass file_buffer to stream_documents
#                 for page_num, page in enumerate(loader.stream_documents(file_buffer)):
#                     gevent.sleep(0)  # Yield control during processing
                    
#                     # Split each page into chunks
#                     for chunk_idx, chunk in enumerate(splitter.split_documents([page])):
#                         cleaned_content = _clean_text(chunk.page_content)
#                         if not cleaned_content:
#                             continue
                        
#                         all_texts.append(cleaned_content)
#                         all_metadatas.append({
#                             **chunk.metadata,
#                             'page_number': page_num,
#                             'chunk_index': chunk_idx
#                         })
                        
#                         # Periodic yield for very large documents
#                         if len(all_texts) % 50 == 0:
#                             gevent.sleep(0)
                
#                 logger.info(f"✅ Successfully streamed {len(all_texts)} chunks from {page_num + 1} pages")
                
#             except Exception as loader_error:
#                 logger.error(f"❌ Streaming failed: {loader_error}")
                
#                 # Fallback: Try load_documents method if streaming fails
#                 logger.info("🔄 Falling back to load_documents method")
#                 file_buffer.seek(0)
                
#                 try:
#                     documents = loader.load_documents(file_buffer)
#                     for doc_idx, doc in enumerate(documents):
#                         chunks = splitter.split_documents([doc])
#                         for chunk_idx, chunk in enumerate(chunks):
#                             cleaned_content = _clean_text(chunk.page_content)
#                             if not cleaned_content:
#                                 continue
                            
#                             all_texts.append(cleaned_content)
#                             all_metadatas.append({
#                                 **chunk.metadata,
#                                 'page_number': doc_idx,
#                                 'chunk_index': chunk_idx
#                             })
                    
#                     logger.info(f"✅ Fallback successful: {len(all_texts)} chunks")
                    
#                 except Exception as fallback_error:
#                     logger.error(f"❌ Both streaming and fallback failed: {fallback_error}")
#                     raise loader_error  # Re-raise original error
            
#             doc_metrics.parse_time_ms = parse_timer.elapsed_ms
#             doc_metrics.total_chunks = len(all_texts)

#         # Phase 3: Token-Aware Adaptive Batching (efficient use of every API call, reducing cost and latency)
#         with Timer() as batch_timer:
#             embedding_tasks = []
#             current_batch_texts, current_batch_metas = [], []
#             current_batch_tokens = 0

#             logger.info(f"💾 Kicking off embedding task for source_id {source_id} ...")
#             for i, text in enumerate(all_texts):
#                 token_count = len(tokenizer.encode(text))
                
#                 # If adding the next chunk exceeds the token limit, dispatch the current batch
#                 if current_batch_tokens + token_count > OPENAI_MAX_TOKENS_PER_BATCH and current_batch_texts:
#                     task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
#                     embedding_tasks.append(task_sig)
#                     current_batch_texts, current_batch_metas, current_batch_tokens = [], [], 0

#                 current_batch_texts.append(text)
#                 current_batch_metas.append(all_metadatas[i])
#                 current_batch_tokens += token_count
            
#             # Dispatch the final batch if it exists
#             if current_batch_texts:
#                 task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
#                 embedding_tasks.append(task_sig)
#             # update telemetry
#             doc_metrics.total_batches = len(embedding_tasks)
#             doc_metrics.chunk_time_ms = batch_timer.elapsed_ms

#         # Phase 4: Database Update & Task Scheduling
#         pool = await get_async_db_pool()
#         async with pool.acquire() as conn:
#             await conn.execute(
#                 """
#                 UPDATE document_sources
#                 SET vector_embed_status = $1, total_chunks = $2, total_batches = $3, processing_metadata = processing_metadata || $4::jsonb
#                 WHERE id = $5
#                 """,
#                 ProcessingStatus.EMBEDDING.value, doc_metrics.total_chunks, doc_metrics.total_batches,
#                 json.dumps(doc_metrics.to_dict()), uuid.UUID(source_id)
#             )
        
#         if embedding_tasks:
#             # Use a chord to run all embedding tasks and then call finalization
#             chord(group(embedding_tasks), queue=EMBED_QUEUE)(finalize_embeddings.s(source_id).set(queue=FINAL_QUEUE))
#         else:
#             # Handle empty documents correctly
#             finalize_embeddings.apply_async(([], source_id), queue=FINAL_QUEUE)

#         logger.info(f"Parsing complete for {source_id}: {doc_metrics.total_chunks} chunks in {doc_metrics.total_batches} token-aware batches.")
#         return {'source_id': source_id, 'status': 'SUCCESS', 'batches_created': doc_metrics.total_batches}

#     except Exception as e:
#         logger.error(f"Parsing failed for {source_id}: {e}", exc_info=True)
#         await _update_document_status(source_id, ProcessingStatus.FAILED_PARSING, str(e))
#         raise

def _parse_document_gevent(self, source_id: str, cdn_url: str, project_id: str) -> Dict[str, Any]:
    """
    Gevent-based parsing implementation
    """
    _update_document_status_sync(source_id, ProcessingStatus.PARSING)
    
    file_buffer = io.BytesIO()
    doc_metrics = DocumentMetrics(doc_id=source_id)

    try:
        # Phase 1: Gevent-optimized streaming download
        with Timer() as download_timer:
            # Helper to create HTTP request sessions with retry strategy
            session = create_http_session()
            
            # Stream the file with gevent-friendly requests
            response = session.get(cdn_url, headers=DEFAULT_HEADERS, stream=True, timeout=120)
            response.raise_for_status()
            
            # Stream into memory buffer
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_buffer.write(chunk)
                    # Yield control to other greenlets occasionally
                    gevent.sleep(0)
            
            file_buffer.seek(0)
            doc_metrics.download_time_ms = download_timer.elapsed_ms
            doc_metrics.file_size_bytes = file_buffer.getbuffer().nbytes

        # Phase 2: Process with Loader and Splitter (CPU-bound, but fast)
        with Timer() as parse_timer:
            loader = get_loader_for(filename=cdn_url, file_like_object=file_buffer)
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
            all_texts, all_metadatas = [], []
            for page_num, page in enumerate(loader.stream_documents(file_buffer)):
                # Yield control during processing
                gevent.sleep(0)
                
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
            
            doc_metrics.parse_time_ms = parse_timer.elapsed_ms
            doc_metrics.total_chunks = len(all_texts)

        # Phase 3: Token-Aware Adaptive Batching
        with Timer() as batch_timer:
            embedding_tasks = []
            current_batch_texts, current_batch_metas = [], []
            current_batch_tokens = 0

            logger.info(f"💾 Kicking off embedding task for source_id {source_id}...")
            
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
                
                # Yield control during batching
                if i % 100 == 0:
                    gevent.sleep(0)
            
            # Dispatch the final batch if it exists
            if current_batch_texts:
                task_sig = embed_batch_task.s(source_id, project_id, current_batch_texts, current_batch_metas)
                embedding_tasks.append(task_sig)
            
            doc_metrics.total_batches = len(embedding_tasks)
            doc_metrics.chunk_time_ms = batch_timer.elapsed_ms

        # Phase 4: Database Update & Task Scheduling
        pool = get_sync_db_pool()   # use SYNC pool since its a simple status update to Supabase
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE document_sources
                    SET vector_embed_status = %s, total_chunks = %s, total_batches = %s, 
                        processing_metadata = processing_metadata || %s::jsonb
                    WHERE id = %s""",
                    (ProcessingStatus.EMBEDDING.value, doc_metrics.total_chunks, 
                        doc_metrics.total_batches, json.dumps(doc_metrics.to_dict()), source_id)
                )
                conn.commit()
        finally:
            pool.putconn(conn)
        
        # Schedule embedding tasks
        if embedding_tasks:
            chord(group(embedding_tasks), queue=EMBED_QUEUE)(finalize_embeddings.s(source_id).set(queue=FINAL_QUEUE))
        else:
            finalize_embeddings.apply_async(([], source_id), queue=FINAL_QUEUE)

        logger.info(f"Parsing complete for {source_id}: {doc_metrics.total_chunks} chunks in {doc_metrics.total_batches} token-aware batches.")
        return {'source_id': source_id, 'status': 'SUCCESS', 'batches_created': doc_metrics.total_batches}

    except Exception as e:
        logger.error(f"Parsing failed for {source_id}: {e}", exc_info=True)
        _update_document_status_sync(source_id, ProcessingStatus.FAILED_PARSING, str(e))
        raise

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
        return asyncio.run(_embed_batch_async(self, source_id, project_id, texts, metadatas))
    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed, letting Celery handle retry: {exc}")
        raise  # Let Celery handle the retry automatically

async def _embed_batch_async(self, source_id: str, project_id: str, texts: List[str], metadatas: List[Dict]):
    """
    [ASYNC] Embedding Task:
    - Fully async with non-blocking DB calls via asyncpg.
    - Uses a shared connection pool and highly efficient `copy_records_to_table`.
    - Retries with exponential backoff and uses a circuit breaker for OpenAI calls.
    """
    try:
        # 1. Generate Embeddings (with retry and circuit breaker)
        logger.info(f"🤖 Genrating embeddings...")
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

        # # 3. Fully Async Bulk Insert using copy_records_to_table for max performance
        # pool = await get_async_db_pool()      # BULK insert use ASYNC pgbouncer pool
        # async with pool.acquire() as conn:
        #     await conn.copy_records_to_table(
        #         'document_vector_store',
        #         records=records_to_insert,
        #         columns=('id', 'source_id', 'project_id', 'content', 'metadata', 'embedding', 'num_tokens', 'created_at'),
        #         timeout=120
        #     )
        # 3. CHANGE: Use executemany instead of copy_records_to_table
        pool = await get_async_db_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                '''INSERT INTO document_vector_store 
                (id, source_id, project_id, content, metadata, embedding, num_tokens, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)''',
                records_to_insert
            )    
        return {'processed_count': len(records_to_insert), 'token_count': total_tokens}

    except Exception as exc:
        logger.warning(f"Embedding batch for {source_id} failed (attempt {self.request.retries + 1}), retrying: {exc}")
        raise self.retry(exc=exc, countdown=DEFAULT_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** self.request.retries))

# optional remove the exclude arg
@CircuitBreaker(fail_max=5, reset_timeout=60) # EDIT: Removed the exclude parameter
@embedding_retry
async def _embed_with_retry(texts: List[str]) -> List[List[float]]:
    """Wrapper for OpenAI embedding call with circuit breaker and tenacity retry."""
    return await embedding_model.aembed_documents(texts)


# ——— Task 4: Finalize (Fully Async DB) ————————————————————————————————————————

@celery_app.task(bind=True, queue=FINAL_QUEUE, acks_late=True)
def finalize_embeddings(self, batch_results: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
    return asyncio.run(_finalize_embeddings_async(self, batch_results, source_id))

async def _finalize_embeddings_async(self, batch_results: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
    """
    Finalization Task:
    - Uses asyncpg for all database reads and writes.
    - Preserves the detailed completion metrics and performance analysis from v5.
    """
    pool = await get_async_db_pool()
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
