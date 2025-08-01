# tasks/database.py
"""
Database connection utilities - separate from celery_app to avoid circular imports
"""

import logging
import threading
import asyncio
import os
from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as aioredis
import psycopg2.pool
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Database pools (module-level globals)
db_pool = None
redis_pool = None
sync_db_pool = None

# Race condition prevention locks
_sync_pool_lock = threading.Lock()
_async_pool_init_lock = None
_redis_init_lock = None

# Configuration
DB_DSN = os.getenv("POSTGRES_DSN_POOL")
DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 5

REDIS_LABS_URL = (
    "redis://default:"
    + os.getenv("REDIS_PASSWORD")
    + "@"
    + os.getenv("REDIS_PUBLIC_ENDPOINT")
)

# ——— Pool Initialization Functions ————————————————————————————————————————————————

async def init_async_pools():
    """Initialize async pools once per worker with race condition protection"""
    global db_pool, redis_pool, _async_pool_init_lock, _redis_init_lock
    
    # Initialize locks if they don't exist (lazy initialization)
    if _async_pool_init_lock is None:
        _async_pool_init_lock = asyncio.Lock()
    if _redis_init_lock is None:
        _redis_init_lock = asyncio.Lock()
    
    # Database pool initialization with lock
    if not db_pool:
        async with _async_pool_init_lock:
            # Double-check pattern: another coroutine might have initialized it
            if not db_pool:
                db_pool = await asyncpg.create_pool(
                    dsn=DB_DSN,
                    min_size=DB_POOL_MIN_SIZE,
                    max_size=DB_POOL_MAX_SIZE,
                    command_timeout=30,
                    statement_cache_size=0,
                    server_settings={'application_name': 'celery_worker'}
                )
                logger.info("✅ [database] Global async DB pool initialized")
    
    # Redis pool initialization with lock
    if not redis_pool:
        async with _redis_init_lock:
            # Double-check pattern
            if not redis_pool:
                redis_pool = aioredis.ConnectionPool.from_url(
                    REDIS_LABS_URL,
                    max_connections=20,
                    decode_responses=True
                )
                logger.info("✅ [database] Global Redis pool initialized")

def init_sync_pool():
    """Initialize sync database pool for upload tasks with race condition protection"""
    global sync_db_pool
    
    # Use threading lock for sync pool
    with _sync_pool_lock:
        # Double-check pattern: another thread might have initialized it
        if not sync_db_pool:
            # Extract connection params from DSN for psycopg2
            parsed = urllib.parse.urlparse(DB_DSN)
            
            sync_db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],  # Remove leading '/'
                user=parsed.username,
                password=parsed.password,
                application_name='celery_sync_worker'
            )
            logger.info("✅ [database] Global sync DB pool initialized")

# ——— Pool Access Functions ————————————————————————————————————————————————————————

def get_global_async_db_pool():
    """Get the global async database pool"""
    return db_pool

def get_global_redis_pool():
    """Get the global Redis pool"""
    return redis_pool

def get_global_sync_db_pool():
    """Get the global sync database pool"""
    if not sync_db_pool:
        init_sync_pool()
    return sync_db_pool

# ——— Health Check Functions (using pool_utils) ——————————————————————————————————

async def check_db_pool_health() -> bool:
    """Check if global database pool is healthy"""
    from .pool_utils import check_async_db_pool_health
    return await check_async_db_pool_health(db_pool)

async def check_redis_pool_health() -> bool:
    """Check if global Redis pool is healthy"""
    from .pool_utils import check_redis_pool_health
    return await check_redis_pool_health(redis_pool)

def check_sync_pool_health() -> bool:
    """Check if global sync database pool is healthy"""
    from .pool_utils import check_sync_db_pool_health
    return check_sync_db_pool_health(sync_db_pool)

# ——— Context Managers (using pool_utils) ——————————————————————————————————————————

@asynccontextmanager
async def get_db_connection():
    """Get database connection from global pool"""
    if not db_pool:
        await init_async_pools()
    
    from .pool_utils import get_db_connection_from_pool
    async with get_db_connection_from_pool(db_pool) as conn:
        yield conn

@asynccontextmanager
async def get_redis_connection():
    """Get Redis connection from global pool"""
    if not redis_pool:
        await init_async_pools()
    
    from .pool_utils import get_redis_connection_from_pool
    async with get_redis_connection_from_pool(redis_pool) as redis:
        yield redis

# ——— Cleanup Functions ————————————————————————————————————————————————————————————

async def cleanup_async_pools():
    """Clean up async pools"""
    global db_pool, redis_pool
    
    if db_pool:
        await db_pool.close()
        db_pool = None
    
    if redis_pool:
        await redis_pool.disconnect()
        redis_pool = None

def cleanup_sync_pool():
    """Clean up sync pool"""
    global sync_db_pool
    
    if sync_db_pool:
        sync_db_pool.closeall()
        sync_db_pool = None