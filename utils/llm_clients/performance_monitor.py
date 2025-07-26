# utils/llm_clients/performance_monitor.py
"""
Performance monitoring utility for tracking and analyzing streaming chat performance.

Features:
- Real-time performance metrics collection
- Function-level timing decorators
- Redis-based metrics storage
- Performance alerting and anomaly detection
- Comprehensive reporting and analytics
- Memory and resource usage tracking
"""

import time
import asyncio
import json
import psutil
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from functools import wraps
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Core performance metrics data structure"""
    function_name: str
    execution_time: float
    start_time: str
    end_time: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # Chat-specific metrics
    tokens_processed: int = 0
    tokens_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Request context
    user_id: Optional[str] = None
    chat_session_id: Optional[str] = None
    message_id: Optional[str] = None
    model_name: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    active_connections: int = 0
    queue_size: int = 0

@dataclass
class AlertThreshold:
    """Performance alert threshold configuration"""
    metric_name: str
    threshold_value: float
    comparison: str = "greater_than"  # greater_than, less_than, equals
    window_minutes: int = 5
    min_occurrences: int = 3

class PerformanceMonitor:
    """
    High-performance monitoring system with real-time analytics
    """
    
    def __init__(self, redis_pool=None, alert_webhook=None):
        self.redis_pool = redis_pool
        self.alert_webhook = alert_webhook
        self.metrics_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        
        # Performance thresholds
        self.alert_thresholds = [
            AlertThreshold("execution_time", 10.0, "greater_than", 5, 3),  # 10s+ execution time
            AlertThreshold("memory_usage_mb", 1000.0, "greater_than", 5, 2),  # 1GB+ memory
            AlertThreshold("cpu_percent", 80.0, "greater_than", 5, 3),  # 80%+ CPU
            AlertThreshold("tokens_per_second", 5.0, "less_than", 10, 5),  # <5 tokens/sec
        ]
        
        # Cache for recent metrics
        self.recent_metrics = {}
        self.system_stats = {}
        
        # Background tasks
        self._flush_task = None
        self._system_monitor_task = None
        
    async def initialize(self):
        """Initialize monitoring system"""
        logger.info("📊 Initializing PerformanceMonitor...")
        
        # Start background tasks
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self._system_monitor_task = asyncio.create_task(self._system_monitor())
        
        logger.info("✅ PerformanceMonitor initialized")
    
    async def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("🛑 Shutting down PerformanceMonitor...")
        
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
        
        # Flush remaining metrics
        await self._flush_metrics()
        
        logger.info("✅ PerformanceMonitor shutdown complete")

    def performance_timer(
        self, 
        function_name: str = None,
        track_resources: bool = True,
        alert_on_slow: bool = True
    ):
        """
        Decorator for automatic performance monitoring
        
        Usage:
            @monitor.performance_timer("embedding_generation")
            async def generate_embedding(query):
                # Function implementation
                pass
        """
        def decorator(func: Callable) -> Callable:
            func_name = function_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_with_monitoring(
                        func, func_name, track_resources, alert_on_slow, *args, **kwargs
                    )
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return asyncio.run(self._execute_with_monitoring(
                        func, func_name, track_resources, alert_on_slow, *args, **kwargs
                    ))
                return sync_wrapper
        return decorator

    async def _execute_with_monitoring(
        self, 
        func: Callable, 
        func_name: str,
        track_resources: bool,
        alert_on_slow: bool,
        *args, 
        **kwargs
    ):
        """Execute function with comprehensive monitoring"""
        start_time = time.time()
        start_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Capture initial resource usage
        initial_memory = psutil.virtual_memory().percent if track_resources else 0
        initial_cpu = psutil.cpu_percent() if track_resources else 0
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            end_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Capture final resource usage
            final_memory = psutil.virtual_memory().percent if track_resources else 0
            final_cpu = psutil.cpu_percent() if track_resources else 0
            
            # Create metrics record
            metrics = PerformanceMetrics(
                function_name=func_name,
                execution_time=execution_time,
                start_time=start_timestamp,
                end_time=end_timestamp,
                success=True,
                memory_usage_mb=(final_memory - initial_memory) * psutil.virtual_memory().total / (1024**3),
                cpu_percent=(final_cpu + initial_cpu) / 2,
            )
            
            # Extract context from kwargs if available
            self._extract_context(metrics, kwargs)
            
            # Record metrics
            await self.record_metrics(metrics)
            
            # Check for alerts
            if alert_on_slow:
                await self._check_alerts(metrics)
            
            logger.debug(f"📊 {func_name}: {execution_time*1000:.1f}ms")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            end_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Record error metrics
            error_metrics = PerformanceMetrics(
                function_name=func_name,
                execution_time=execution_time,
                start_time=start_timestamp,
                end_time=end_timestamp,
                success=False,
                error_message=str(e),
            )
            
            await self.record_metrics(error_metrics)
            logger.error(f"❌ {func_name} failed after {execution_time*1000:.1f}ms: {e}")
            raise

    def _extract_context(self, metrics: PerformanceMetrics, kwargs: Dict):
        """Extract context information from function arguments"""
        # Try to extract common context fields
        context_fields = ['user_id', 'chat_session_id', 'message_id', 'model_name']
        
        for field in context_fields:
            if field in kwargs:
                setattr(metrics, field, kwargs[field])
        
        # Extract token information if available
        if 'content' in kwargs:
            content = kwargs['content']
            if isinstance(content, str):
                metrics.tokens_processed = len(content.split())
        
        # Calculate tokens per second
        if metrics.tokens_processed > 0 and metrics.execution_time > 0:
            metrics.tokens_per_second = metrics.tokens_processed / metrics.execution_time

    async def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Update recent metrics cache
        self.recent_metrics[metrics.function_name] = metrics
        
        # Flush if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            await self._flush_metrics()

    async def record_custom_metric(
        self,
        name: str,
        value: float,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
        chat_session_id: str = None
    ):
        """Record a custom metric"""
        custom_metrics = PerformanceMetrics(
            function_name=f"custom.{name}",
            execution_time=value,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
            user_id=user_id,
            chat_session_id=chat_session_id
        )
        
        await self.record_metrics(custom_metrics)

    async def _flush_metrics(self):
        """Flush metrics buffer to Redis"""
        if not self.metrics_buffer or not self.redis_pool:
            return
        
        try:
            import redis.asyncio as aioredis
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                # Batch insert metrics
                pipe = r.pipeline()
                
                for metrics in self.metrics_buffer:
                    metric_data = asdict(metrics)
                    pipe.lpush("performance_metrics", json.dumps(metric_data))
                
                # Keep only last 10,000 metrics
                pipe.ltrim("performance_metrics", 0, 9999)
                
                await pipe.execute()
                
            logger.debug(f"📊 Flushed {len(self.metrics_buffer)} metrics to Redis")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"❌ Failed to flush metrics: {e}")

    async def _periodic_flush(self):
        """Periodically flush metrics to prevent buffer overflow"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Periodic flush error: {e}")

    async def _system_monitor(self):
        """Monitor system-level metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Collect system metrics
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metrics = SystemMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage_percent=disk.percent
                )
                
                # Store in Redis
                if self.redis_pool:
                    await self._store_system_metrics(system_metrics)
                
                # Update local cache
                self.system_stats = asdict(system_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ System monitoring error: {e}")

    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in Redis"""
        try:
            import redis.asyncio as aioredis
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                await r.lpush("system_metrics", json.dumps(asdict(metrics)))
                await r.ltrim("system_metrics", 0, 2880)  # Keep 24 hours (30s intervals)
                
        except Exception as e:
            logger.error(f"❌ Failed to store system metrics: {e}")

    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check if metrics trigger any alerts"""
        for threshold in self.alert_thresholds:
            value = getattr(metrics, threshold.metric_name, None)
            
            if value is None:
                continue
            
            should_alert = False
            
            if threshold.comparison == "greater_than" and value > threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "less_than" and value < threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "equals" and abs(value - threshold.threshold_value) < 0.001:
                should_alert = True
            
            if should_alert:
                await self._trigger_alert(metrics, threshold, value)

    async def _trigger_alert(self, metrics: PerformanceMetrics, threshold: AlertThreshold, value: float):
        """Trigger performance alert"""
        alert_data = {
            "alert_type": "performance_threshold",
            "metric_name": threshold.metric_name,
            "threshold_value": threshold.threshold_value,
            "actual_value": value,
            "function_name": metrics.function_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": metrics.user_id,
            "chat_session_id": metrics.chat_session_id,
        }
        
        logger.warning(f"🚨 Performance Alert: {threshold.metric_name} = {value} (threshold: {threshold.threshold_value})")
        
        # Store alert in Redis
        if self.redis_pool:
            try:
                import redis.asyncio as aioredis
                async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                    await r.lpush("performance_alerts", json.dumps(alert_data))
                    await r.ltrim("performance_alerts", 0, 1000)  # Keep last 1000 alerts
            except Exception as e:
                logger.error(f"❌ Failed to store alert: {e}")
        
        # Send webhook notification if configured
        if self.alert_webhook:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(self.alert_webhook, json=alert_data)
            except Exception as e:
                logger.error(f"❌ Failed to send alert webhook: {e}")

    async def get_performance_summary(
        self, 
        function_name: str = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance summary for analysis"""
        if not self.redis_pool:
            return {"error": "Redis not available"}
        
        try:
            import redis.asyncio as aioredis
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                # Get metrics from Redis
                raw_metrics = await r.lrange("performance_metrics", 0, -1)
                
                metrics = []
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
                
                for raw_metric in raw_metrics:
                    try:
                        metric_data = json.loads(raw_metric)
                        metric_time = datetime.fromisoformat(metric_data['start_time'].replace('Z', '+00:00'))
                        
                        if metric_time >= cutoff_time:
                            if function_name is None or metric_data['function_name'] == function_name:
                                metrics.append(metric_data)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse metric: {e}")
                
                return self._calculate_summary_stats(metrics)
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return {"error": str(e)}

    def _calculate_summary_stats(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics"""
        if not metrics:
            return {"total_requests": 0}
        
        execution_times = [m['execution_time'] for m in metrics if m.get('execution_time')]
        success_rate = len([m for m in metrics if m.get('success', True)]) / len(metrics)
        
        function_stats = {}
        for metric in metrics:
            func_name = metric.get('function_name', 'unknown')
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'count': 0,
                    'total_time': 0,
                    'errors': 0
                }
            
            function_stats[func_name]['count'] += 1
            function_stats[func_name]['total_time'] += metric.get('execution_time', 0)
            if not metric.get('success', True):
                function_stats[func_name]['errors'] += 1
        
        # Calculate averages
        for func_name, stats in function_stats.items():
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['error_rate'] = stats['errors'] / stats['count']
        
        return {
            "total_requests": len(metrics),
            "success_rate": success_rate,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "p95_execution_time": self._percentile(execution_times, 95) if execution_times else 0,
            "p99_execution_time": self._percentile(execution_times, 99) if execution_times else 0,
            "function_breakdown": function_stats,
            "time_window_hours": len(metrics),
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]

    async def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time performance statistics"""
        current_time = datetime.now(timezone.utc)
        
        # Recent function performance (last 5 minutes)
        recent_metrics = [
            metrics for metrics in self.recent_metrics.values()
            if (current_time - datetime.fromisoformat(metrics.start_time.replace('Z', '+00:00'))).seconds < 300
        ]
        
        active_functions = len(set(m.function_name for m in recent_metrics))
        avg_response_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            "timestamp": current_time.isoformat(),
            "active_functions": active_functions,
            "recent_requests": len(recent_metrics),
            "avg_response_time_ms": avg_response_time * 1000,
            "system_stats": self.system_stats,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent metrics"""
        cache_metrics = [
            m for m in self.recent_metrics.values() 
            if m.cache_hit_rate > 0
        ]
        
        if not cache_metrics:
            return 0.0
        
        return sum(m.cache_hit_rate for m in cache_metrics) / len(cache_metrics)

    @asynccontextmanager
    async def timing_context(self, operation_name: str, **context):
        """Context manager for manual timing"""
        start_time = time.time()
        start_timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            yield
            
            execution_time = time.time() - start_time
            metrics = PerformanceMetrics(
                function_name=f"manual.{operation_name}",
                execution_time=execution_time,
                start_time=start_timestamp,
                end_time=datetime.now(timezone.utc).isoformat(),
                success=True,
                **context
            )
            
            await self.record_metrics(metrics)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = PerformanceMetrics(
                function_name=f"manual.{operation_name}",
                execution_time=execution_time,
                start_time=start_timestamp,
                end_time=datetime.now(timezone.utc).isoformat(),
                success=False,
                error_message=str(e),
                **context
            )
            
            await self.record_metrics(error_metrics)
            raise

    def record_error(self, function_name: str, error_message: str, **context):
        """Record an error occurrence"""
        error_metrics = PerformanceMetrics(
            function_name=function_name,
            execution_time=0,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            success=False,
            error_message=error_message,
            **context
        )
        
        # Add to buffer synchronously for immediate error recording
        self.metrics_buffer.append(error_metrics)

    async def export_metrics(
        self, 
        format_type: str = "json",
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Export metrics for external analysis"""
        summary = await self.get_performance_summary(time_window_hours=time_window_hours)
        real_time = await self.get_real_time_stats()
        
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "time_window_hours": time_window_hours,
            "summary": summary,
            "real_time_stats": real_time,
            "alert_thresholds": [asdict(t) for t in self.alert_thresholds],
        }
        
        if format_type == "json":
            return export_data
        elif format_type == "csv":
            # Convert to CSV format for spreadsheet analysis
            return self._convert_to_csv(export_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _convert_to_csv(self, data: Dict) -> str:
        """Convert metrics data to CSV format"""
        import csv
        import io
        
        output = io.StringIO()
        
        # Write summary stats
        writer = csv.writer(output)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Requests", data["summary"]["total_requests"]])
        writer.writerow(["Success Rate", data["summary"]["success_rate"]])
        writer.writerow(["Avg Execution Time", data["summary"]["avg_execution_time"]])
        writer.writerow(["P95 Execution Time", data["summary"]["p95_execution_time"]])
        writer.writerow(["P99 Execution Time", data["summary"]["p99_execution_time"]])
        
        return output.getvalue()

# Example usage and testing
async def example_usage():
    """Example of how to use PerformanceMonitor"""
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    await monitor.initialize()
    
    try:
        # Using decorator
        @monitor.performance_timer("test_function")
        async def test_function(duration: float):
            await asyncio.sleep(duration)
            return "completed"
        
        # Execute monitored function
        result = await test_function(0.1)
        print(f"Function result: {result}")
        
        # Using context manager
        async with monitor.timing_context("manual_operation", user_id="test123"):
            await asyncio.sleep(0.05)
            print("Manual operation completed")
        
        # Record custom metric
        await monitor.record_custom_metric(
            "custom_latency", 
            150.0, 
            metadata={"endpoint": "/api/test"},
            user_id="test123"
        )
        
        # Get performance summary
        await asyncio.sleep(1)  # Allow metrics to flush
        summary = await monitor.get_performance_summary()
        print(f"Performance Summary: {summary}")
        
        # Get real-time stats
        real_time = await monitor.get_real_time_stats()
        print(f"Real-time Stats: {real_time}")
        
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())