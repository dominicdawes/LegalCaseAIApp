""" 
# memory_manager.py
Advanced memory management class
"""

import psutil
import threading
import time
import logging
from contextlib import contextmanager
from typing import Optional
import gc

logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management for production workloads"""
    
    def __init__(self, max_usage_pct: float = 75.0, check_interval: int = 30):
        self.max_usage_pct = max_usage_pct
        self.check_interval = check_interval
        self.process = psutil.Process()
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Memory pressure thresholds
        self.warning_threshold = max_usage_pct * 0.8  # 60% if max is 75%
        self.critical_threshold = max_usage_pct * 0.95  # 71.25% if max is 75%
        
        # Statistics
        self.peak_usage = 0.0
        self.pressure_events = 0
        self.cleanup_calls = 0
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                current_usage = self.get_memory_usage_pct()
                
                with self._lock:
                    if current_usage > self.peak_usage:
                        self.peak_usage = current_usage
                
                if current_usage > self.critical_threshold:
                    logger.warning(f"Critical memory usage: {current_usage:.1f}%")
                    self.force_cleanup()
                elif current_usage > self.warning_threshold:
                    logger.info(f"High memory usage warning: {current_usage:.1f}%")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_usage_pct(self) -> float:
        """Get current memory usage percentage"""
        try:
            return self.process.memory_percent()
        except Exception:
            # Fallback to system memory if process memory fails
            return psutil.virtual_memory().percent
    
    def get_memory_info(self) -> dict:
        """Get detailed memory information"""
        try:
            process_memory = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_rss_mb': process_memory.rss / (1024 * 1024),
                'process_vms_mb': process_memory.vms / (1024 * 1024),
                'process_percent': self.process.memory_percent(),
                'system_total_gb': system_memory.total / (1024 * 1024 * 1024),
                'system_available_gb': system_memory.available / (1024 * 1024 * 1024),
                'system_percent': system_memory.percent,
                'peak_usage_pct': self.peak_usage
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
    
    def check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure"""
        current_usage = self.get_memory_usage_pct()
        is_pressure = current_usage > self.max_usage_pct
        
        if is_pressure:
            with self._lock:
                self.pressure_events += 1
        
        return is_pressure
    
    def wait_for_memory_available(self, timeout: int = 60, check_interval: float = 1.0) -> bool:
        """Wait for memory to become available"""
        start_time = time.time()
        
        while self.check_memory_pressure():
            if time.time() - start_time > timeout:
                logger.error(f"Memory timeout: waited {timeout}s, usage still {self.get_memory_usage_pct():.1f}%")
                return False
            
            # Try cleanup if we're waiting
            if time.time() - start_time > timeout / 2:
                self.force_cleanup()
            
            time.sleep(check_interval)
        
        return True
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        try:
            with self._lock:
                self.cleanup_calls += 1
            
            # Run garbage collection
            collected = gc.collect()
            
            # Additional cleanup for different generations
            for generation in range(3):
                gc.collect(generation)
            
            logger.info(f"Memory cleanup completed, collected {collected} objects")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    @contextmanager
    def memory_context(self, required_mb: float = 0, cleanup_on_exit: bool = True):
        """Context manager for memory-aware operations"""
        # Check if we have enough memory before starting
        if required_mb > 0:
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / (1024 * 1024)
            
            if available_mb < required_mb:
                # Try cleanup first
                self.force_cleanup()
                
                # Check again after cleanup
                system_memory = psutil.virtual_memory()
                available_mb = system_memory.available / (1024 * 1024)
                
                if available_mb < required_mb:
                    raise MemoryError(
                        f"Insufficient memory: need {required_mb:.1f}MB, "
                        f"have {available_mb:.1f}MB available"
                    )
        
        # Wait for memory pressure to subside
        if not self.wait_for_memory_available():
            raise MemoryError("Memory pressure timeout")
        
        start_memory = self.get_memory_usage_pct()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage_pct()
            duration = time.time() - start_time
            
            logger.debug(
                f"Memory context: {start_memory:.1f}% → {end_memory:.1f}% "
                f"in {duration:.2f}s"
            )
            
            # Cleanup on exit if requested and memory usage increased significantly
            if cleanup_on_exit and (end_memory - start_memory) > 10:
                self.force_cleanup()
    
    def estimate_available_memory_mb(self) -> float:
        """Estimate available memory for new operations"""
        system_memory = psutil.virtual_memory()
        available_mb = system_memory.available / (1024 * 1024)
        
        # Conservative estimate: only use 80% of available memory
        return available_mb * 0.8
    
    def get_stats(self) -> dict:
        """Get memory manager statistics"""
        with self._lock:
            return {
                'peak_usage_pct': self.peak_usage,
                'pressure_events': self.pressure_events,
                'cleanup_calls': self.cleanup_calls,
                'current_usage_pct': self.get_memory_usage_pct(),
                'monitoring_active': self._monitoring,
                'max_usage_threshold': self.max_usage_pct,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold
            }