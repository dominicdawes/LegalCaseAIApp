""" 
# metrics.py
Production-grade telemetry and server metrics
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import psutil
import logging

logger = logging.getLogger(__name__)

class Timer:
    """High-precision timer for performance measurements"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

class MetricsCollector:
    """Production-grade metrics collection with thread safety"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Performance metrics
        self.ingestion_metrics = deque(maxlen=max_history)
        self.batch_metrics = deque(maxlen=max_history)
        self.completion_metrics = deque(maxlen=max_history)
        
        # Counters
        self.counters = defaultdict(int)
        
        # Time series data
        self.embedding_times = deque(maxlen=max_history)
        self.batch_retries = defaultdict(int)
        
        # Performance history for optimization
        self.performance_history = deque(maxlen=100)
        
    def record_ingestion_metrics(self, total_documents: int, processed_documents: int, 
                                duplicate_documents: int, processing_time_ms: int):
        """Record document ingestion metrics"""
        with self._lock:
            metric = {
                'timestamp': datetime.now(timezone.utc),
                'total_documents': total_documents,
                'processed_documents': processed_documents,
                'duplicate_documents': duplicate_documents,
                'processing_time_ms': processing_time_ms,
                'throughput_docs_per_sec': processed_documents / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            }
            self.ingestion_metrics.append(metric)
    
    def record_batch_metrics(self, batch_id: str, source_id: str, chunk_count: int,
                           total_time_ms: int, embedding_time_ms: int, 
                           storage_time_ms: int, token_count: int):
        """Record batch processing metrics"""
        with self._lock:
            metric = {
                'timestamp': datetime.now(timezone.utc),
                'batch_id': batch_id,
                'source_id': source_id,
                'chunk_count': chunk_count,
                'total_time_ms': total_time_ms,
                'embedding_time_ms': embedding_time_ms,
                'storage_time_ms': storage_time_ms,
                'token_count': token_count,
                'tokens_per_second': token_count / (total_time_ms / 1000) if total_time_ms > 0 else 0,
                'chunks_per_second': chunk_count / (total_time_ms / 1000) if total_time_ms > 0 else 0
            }
            self.batch_metrics.append(metric)
            
            # Update performance history for optimization
            self.performance_history.append({
                'processing_time': total_time_ms,
                'memory_usage': psutil.virtual_memory().percent,
                'chunk_count': chunk_count
            })
    
    def record_completion_metrics(self, source_id: str, final_status: str,
                                total_chunks: int, processed_chunks: int,
                                total_time_ms: int, completion_rate: float):
        """Record document completion metrics"""
        with self._lock:
            metric = {
                'timestamp': datetime.now(timezone.utc),
                'source_id': source_id,
                'final_status': final_status,
                'total_chunks': total_chunks,
                'processed_chunks': processed_chunks,
                'total_time_ms': total_time_ms,
                'completion_rate': completion_rate,
                'success': final_status == 'COMPLETE'
            }
            self.completion_metrics.append(metric)
    
    def record_embedding_time(self, time_ms: int):
        """Record embedding API call time"""
        with self._lock:
            self.embedding_times.append({
                'timestamp': datetime.now(timezone.utc),
                'time_ms': time_ms
            })
    
    def increment_embedding_calls(self):
        """Increment embedding API call counter"""
        with self._lock:
            self.counters['embedding_calls'] += 1
    
    def increment_embedding_errors(self):
        """Increment embedding error counter"""
        with self._lock:
            self.counters['embedding_errors'] += 1
    
    def increment_batch_retries(self, batch_id: str):
        """Increment retry count for a batch"""
        with self._lock:
            self.batch_retries[batch_id] += 1
    
    def get_recent_performance_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics from the last N minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._lock:
            recent_batches = [
                m for m in self.batch_metrics 
                if m['timestamp'] > cutoff_time
            ]
            
            recent_completions = [
                m for m in self.completion_metrics
                if m['timestamp'] > cutoff_time
            ]
            
            if not recent_batches:
                return {'no_data': True}
            
            # Calculate averages
            avg_batch_time = sum(m['total_time_ms'] for m in recent_batches) / len(recent_batches)
            avg_tokens_per_sec = sum(m['tokens_per_second'] for m in recent_batches) / len(recent_batches)
            avg_memory_usage = sum(p['memory_usage'] for p in self.performance_history[-50:]) / min(50, len(self.performance_history))
            
            # Calculate success rates
            total_completions = len(recent_completions)
            successful_completions = sum(1 for m in recent_completions if m['success'])
            success_rate = (successful_completions / total_completions * 100) if total_completions > 0 else 0
            
            return {
                'avg_batch_time_ms': avg_batch_time,
                'avg_tokens_per_sec': avg_tokens_per_sec,
                'avg_memory_usage': avg_memory_usage,
                'success_rate': success_rate,
                'total_batches': len(recent_batches),
                'total_completions': total_completions,
                'embedding_calls': self.counters.get('embedding_calls', 0),
                'embedding_errors': self.counters.get('embedding_errors', 0)
            }
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance trends for optimization"""
        with self._lock:
            if len(self.performance_history) < 10:
                return {'insufficient_data': True}
            
            recent_data = list(self.performance_history)[-50:]
            
            avg_processing_time = sum(p['processing_time'] for p in recent_data) / len(recent_data)
            avg_memory_usage = sum(p['memory_usage'] for p in recent_data) / len(recent_data)
            avg_chunk_count = sum(p['chunk_count'] for p in recent_data) / len(recent_data)
            
            # Calculate trend (simple linear regression)
            if len(recent_data) >= 20:
                first_half = recent_data[:len(recent_data)//2]
                second_half = recent_data[len(recent_data)//2:]
                
                first_avg = sum(p['processing_time'] for p in first_half) / len(first_half)
                second_avg = sum(p['processing_time'] for p in second_half) / len(second_half)
                
                improvement_estimate = ((first_avg - second_avg) / first_avg * 100) if first_avg > 0 else 0
            else:
                improvement_estimate = 0
            
            return {
                'avg_processing_time': avg_processing_time,
                'avg_memory_usage': avg_memory_usage,
                'avg_chunk_count': avg_chunk_count,
                'improvement_estimate': improvement_estimate,
                'data_points': len(recent_data)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'timestamp': datetime.now(timezone.utc),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'embedding_calls_total': self.counters.get('embedding_calls', 0),
            'embedding_errors_total': self.counters.get('embedding_errors', 0),
            'active_batch_retries': len(self.batch_retries)
        }