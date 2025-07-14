import asyncio
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import httpx
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """Production-grade connection pool management"""
    
    def __init__(self, 
                 max_connections: int = 100,
                 max_keepalive_connections: int = 20,
                 timeout: float = 60.0,
                 max_retries: int = 3):
        
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Connection pools
        self._http_client: Optional[httpx.AsyncClient] = None
        self._sync_session: Optional[httpx.Client] = None
        
        # Pool statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connection_errors': 0,
            'requests_made': 0,
            'retries_attempted': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Health monitoring
        self._health_check_interval = 60  # seconds
        self._last_health_check = time.time()
        
    async def get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling"""
        if self._http_client is None or self._http_client.is_closed:
            with self._lock:
                if self._http_client is None or self._http_client.is_closed:
                    self._http_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            max_keepalive_connections=self.max_keepalive_connections
                        ),
                        follow_redirects=True,
                        verify=True
                    )
                    self.stats['connections_created'] += 1
                    logger.info("Created new async HTTP client")
        
        return self._http_client
    
    def get_sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client with connection pooling"""
        if self._sync_session is None or self._sync_session.is_closed:
            with self._lock:
                if self._sync_session is None or self._sync_session.is_closed:
                    self._sync_session = httpx.Client(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            max_keepalive_connections=self.max_keepalive_connections
                        ),
                        follow_redirects=True,
                        verify=True
                    )
                    self.stats['connections_created'] += 1
                    logger.info("Created new sync HTTP client")
        
        return self._sync_session
    
    @asynccontextmanager
    async def request_context(self, method: str, url: str, **kwargs):
        """Context manager for making HTTP requests with retry logic"""
        client = await self.get_async_client()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with client.stream(method, url, **kwargs) as response:
                    self.stats['requests_made'] += 1
                    if attempt > 0:
                        self.stats['retries_attempted'] += 1
                    
                    yield response
                    return
                    
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                self.stats['connection_errors'] += 1
                
                if attempt == self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise
                
                # Exponential backoff
                wait_time = (2 ** attempt) * 0.5
                logger.warning(f"Request attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connection pools"""
        current_time = time.time()
        
        # Only run health check if enough time has passed
        if current_time - self._last_health_check < self._health_check_interval:
            return {'skipped': True, 'reason': 'too_recent'}
        
        self._last_health_check = current_time
        
        health_status = {
            'timestamp': current_time,
            'async_client_status': 'unknown',
            'sync_client_status': 'unknown',
            'stats': self.stats.copy(),
            'issues': []
        }
        
        # Check async client
        try:
            if self._http_client and not self._http_client.is_closed:
                # Simple health check request
                test_response = await self._http_client.get('https://httpbin.org/status/200', timeout=5.0)
                health_status['async_client_status'] = 'healthy' if test_response.status_code == 200 else 'degraded'
            else:
                health_status['async_client_status'] = 'not_initialized'
        except Exception as e:
            health_status['async_client_status'] = 'error'
            health_status['issues'].append(f"Async client error: {e}")
        
        # Check sync client
        try:
            if self._sync_session and not self._sync_session.is_closed:
                test_response = self._sync_session.get('https://httpbin.org/status/200', timeout=5.0)
                health_status['sync_client_status'] = 'healthy' if test_response.status_code == 200 else 'degraded'
            else:
                health_status['sync_client_status'] = 'not_initialized'
        except Exception as e:
            health_status['sync_client_status'] = 'error'
            health_status['issues'].append(f"Sync client error: {e}")
        
        # Calculate success rate
        total_requests = self.stats['requests_made']
        total_errors = self.stats['connection_errors']
        
        if total_requests > 0:
            success_rate = ((total_requests - total_errors) / total_requests) * 100
            health_status['success_rate'] = success_rate
            
            if success_rate < 95:
                health_status['issues'].append(f"Low success rate: {success_rate:.1f}%")
        
        logger.info(f"Connection pool health check: {health_status}")
        return health_status
    
    async def close_all(self):
        """Close all connection pools"""
        if self._http_client:
            await self._http_client.aclose()
            logger.info("Closed async HTTP client")
        
        if self._sync_session:
            self._sync_session.close()
            logger.info("Closed sync HTTP client")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'stats': self.stats.copy(),
            'config': {
                'max_connections': self.max_connections,
                'max_keepalive_connections': self.max_keepalive_connections,
                'timeout': self.timeout,
                'max_retries': self.max_retries
            },
            'status': {
                'async_client_active': self._http_client is not None and not self._http_client.is_closed,
                'sync_client_active': self._sync_session is not None and not self._sync_session.is_closed
            }
        }

class DatabaseConnectionPool:
    """Database connection pool for high-performance database operations"""
    
    def __init__(self, connection_string: str, min_connections: int = 5, max_connections: int = 20):
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        # Connection pool
        self._pool: Optional[asyncio.Queue] = None
        self._connections_created = 0
        self._connections_in_use = 0
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'queries_executed': 0,
            'connection_errors': 0
        }
        
        self._lock = asyncio.Lock()
    
    async def initialize_pool(self):
        """Initialize the connection pool"""
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    try:
                        import asyncpg
                        
                        self._pool = asyncio.Queue(maxsize=self.max_connections)
                        
                        # Create minimum connections
                        for _ in range(self.min_connections):
                            conn = await asyncpg.connect(self.connection_string)
                            await self._pool.put(conn)
                            self._connections_created += 1
                            self.stats['connections_created'] += 1
                        
                        logger.info(f"Database connection pool initialized with {self.min_connections} connections")
                        
                    except Exception as e:
                        logger.error(f"Failed to initialize database pool: {e}")
                        raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        if self._pool is None:
            await self.initialize_pool()
        
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self._pool.get_nowait()
                self.stats['connections_reused'] += 1
            except asyncio.QueueEmpty:
                # Create new connection if under limit
                if self._connections_created < self.max_connections:
                    import asyncpg
                    conn = await asyncpg.connect(self.connection_string)
                    self._connections_created += 1
                    self.stats['connections_created'] += 1
                else:
                    # Wait for available connection
                    conn = await self._pool.get()
                    self.stats['connections_reused'] += 1
            
            self._connections_in_use += 1
            yield conn
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._connections_in_use -= 1
                # Return connection to pool
                try:
                    await self._pool.put(conn)
                except asyncio.QueueFull:
                    # Pool is full, close the connection
                    await conn.close()
                    self._connections_created -= 1
    
    async def execute_query(self, query: str, *args):
        """Execute a query using a pooled connection"""
        async with self.get_connection() as conn:
            self.stats['queries_executed'] += 1
            return await conn.fetch(query, *args)
    
    async def close_pool(self):
        """Close all connections in the pool"""
        if self._pool:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    await conn.close()
                except asyncio.QueueEmpty:
                    break
            
            logger.info("Database connection pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get database pool statistics"""
        return {
            'stats': self.stats.copy(),
            'connections_created': self._connections_created,
            'connections_in_use': self._connections_in_use,
            'pool_size': self._pool.qsize() if self._pool else 0,
            'config': {
                'min_connections': self.min_connections,
                'max_connections': self.max_connections
            }
        }