from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta


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