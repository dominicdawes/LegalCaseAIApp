# tasks/pool_utils.py
"""
Functions for DB and Redis pools
Shared pool utilities and health checks
Prevents circular imports between celery_app and task modules
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ——— Health Check Functions ——————————————————————————————————————————————————————

async def check_async_db_pool_health(pool) -> bool:
    """Check if async database pool is healthy"""
    if not pool or pool.is_closing():
        return False
    
    try:
        async with asyncio.timeout(5):
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
        return True
    except Exception as e:
        logger.warning(f"⚠️ Async DB pool health check failed: {e}")
        return False

async def check_redis_pool_health(pool) -> bool:
    """Check if Redis pool is healthy"""
    if not pool:
        return False
    try:
        async with asyncio.timeout(5):
            import redis.asyncio as aioredis
            async with aioredis.Redis(connection_pool=pool) as r:
                await r.ping()
        return True
    except Exception as e:
        logger.warning(f"⚠️ Redis pool health check failed: {e}")
        return False

def check_sync_db_pool_health(pool) -> bool:
    """Check if sync database pool is healthy"""
    if not pool:
        return False
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                cur.fetchone()
            return True
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.warning(f"⚠️ Sync DB pool health check failed: {e}")
        return False

# ——— Context Manager Helpers ——————————————————————————————————————————————————————

from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_connection_from_pool(pool):
    """Generic database connection context manager"""
    if not pool:
        raise RuntimeError("Database pool not initialized")
    
    async with pool.acquire() as conn:
        yield conn

@asynccontextmanager
async def get_redis_connection_from_pool(pool):
    """Generic Redis connection context manager"""
    if not pool:
        raise RuntimeError("Redis pool not initialized")
    
    import redis.asyncio as aioredis
    async with aioredis.Redis(connection_pool=pool) as redis:
        yield redis