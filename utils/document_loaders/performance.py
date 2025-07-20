# utils/document_loaders/performance.py

import re
import time
import logging
from typing import List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Track performance metrics during text processing"""
    total_pages: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0
    cleaning_time_ms: float = 0
    chunking_time_ms: float = 0
    token_counting_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms,
            "cleaning_time_ms": self.cleaning_time_ms,
            "chunking_time_ms": self.chunking_time_ms,
            "token_counting_time_ms": self.token_counting_time_ms,
            "chars_per_second": self.total_characters / max(self.processing_time_ms / 1000, 0.001),
            "chunks_per_second": self.total_chunks / max(self.processing_time_ms / 1000, 0.001),
        }

class OptimizedTextCleaner:
    """High-performance text cleaning with compiled regex patterns"""
    
    def __init__(self):
        # Compile regex patterns once for better performance
        self._patterns = {
            'null_chars': re.compile(r'[\x00\ufffd]'),
            'excess_whitespace': re.compile(r'\s+'),
            'line_breaks': re.compile(r'\n{3,}'),  # Replace 3+ line breaks with 2
            'unicode_cleanup': re.compile(r'[^\x20-\x7E\n\r\t\u00A0-\u024F\u1E00-\u1EFF]'),
            'trailing_spaces': re.compile(r' +$', re.MULTILINE),
        }
    
    def clean_text_fast(self, text: str) -> str:
        """
        Fast text cleaning optimized for speed
        """
        if not text or len(text.strip()) < 3:
            return ""
        
        # Apply cleaning patterns in order of frequency/impact
        text = self._patterns['null_chars'].sub('', text)
        text = self._patterns['excess_whitespace'].sub(' ', text)
        text = self._patterns['line_breaks'].sub('\n\n', text)
        text = self._patterns['trailing_spaces'].sub('', text)
        
        return text.strip()
    
    def clean_text_thorough(self, text: str) -> str:
        """
        More thorough cleaning for better quality (slightly slower)
        """
        if not text or len(text.strip()) < 3:
            return ""
        
        # Apply all cleaning patterns
        text = self._patterns['null_chars'].sub('', text)
        text = self._patterns['unicode_cleanup'].sub('', text)
        text = self._patterns['excess_whitespace'].sub(' ', text)
        text = self._patterns['line_breaks'].sub('\n\n', text)
        text = self._patterns['trailing_spaces'].sub('', text)
        
        # Additional normalization
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()

class StreamingTextProcessor:
    """
    High-performance streaming text processor with metrics
    """
    
    def __init__(self, 
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    cleaning_mode: str = "fast",
                    min_chunk_length: int = 50):
        """
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between chunks
            cleaning_mode: "fast" or "thorough"
            min_chunk_length: Minimum length for valid chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        
        # Initialize components
        self.cleaner = OptimizedTextCleaner()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],  # Optimized separators
            keep_separator=False
        )
        
        # Set cleaning function based on mode
        self.clean_text = (self.cleaner.clean_text_fast 
                            if cleaning_mode == "fast" 
                            else self.cleaner.clean_text_thorough)
        
        # Metrics tracking
        self.metrics = ProcessingMetrics()
    
    def process_documents_streaming(self, 
                                    documents: Iterator[Document],
                                    source_id: str = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Process documents in streaming fashion with performance tracking
        
        Yields:
            Tuple of (cleaned_text, metadata)
        """
        start_time = time.time()
        
        try:
            for page_num, document in enumerate(documents):
                # Track page processing
                self.metrics.total_pages += 1
                page_start_time = time.time()
                
                # Clean text
                clean_start = time.time()
                cleaned_content = self.clean_text(document.page_content)
                self.metrics.cleaning_time_ms += (time.time() - clean_start) * 1000
                
                if len(cleaned_content) < self.min_chunk_length:
                    logger.debug(f"Skipping short page {page_num + 1}: {len(cleaned_content)} chars")
                    continue
                
                self.metrics.total_characters += len(cleaned_content)
                
                # Create temporary document for chunking
                temp_doc = Document(
                    page_content=cleaned_content,
                    metadata={**document.metadata, "page_processing_time": time.time() - page_start_time}
                )
                
                # Split into chunks
                chunk_start = time.time()
                chunks = self.splitter.split_documents([temp_doc])
                self.metrics.chunking_time_ms += (time.time() - chunk_start) * 1000
                
                # Yield chunks
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.page_content.strip()) >= self.min_chunk_length:
                        self.metrics.total_chunks += 1
                        
                        # Enhanced metadata
                        chunk_metadata = {
                            **chunk.metadata,
                            "chunk_index": chunk_idx,
                            "chunk_length": len(chunk.page_content),
                            "source_id": source_id
                        }
                        
                        yield chunk.page_content, chunk_metadata
                
                # Log progress periodically
                if (page_num + 1) % 10 == 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.info(f"Processed {page_num + 1} pages, "
                                f"{self.metrics.total_chunks} chunks, "
                                f"{elapsed_ms:.1f}ms elapsed")
        
        finally:
            # Finalize metrics
            self.metrics.processing_time_ms = (time.time() - start_time) * 1000
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            **self.metrics.to_dict(),
            "avg_chars_per_chunk": (self.metrics.total_characters / 
                                    max(self.metrics.total_chunks, 1)),
            "avg_chunks_per_page": (self.metrics.total_chunks / 
                                    max(self.metrics.total_pages, 1)),
            "processing_efficiency": {
                "cleaning_percent": (self.metrics.cleaning_time_ms / 
                                   max(self.metrics.processing_time_ms, 1)) * 100,
                "chunking_percent": (self.metrics.chunking_time_ms / 
                                   max(self.metrics.processing_time_ms, 1)) * 100,
            }
        }

class BatchTokenCounter:
    """Optimized token counting with batch processing"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache = {}  # Simple cache for repeated text
        
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for a batch of texts with caching
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hash(text[:100])  # Hash first 100 chars for cache key
            if text_hash in self._cache:
                results.append(self._cache[text_hash])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            try:
                # Try batch encoding if available
                if hasattr(self.tokenizer, 'encode_batch'):
                    token_lists = self.tokenizer.encode_batch(uncached_texts)
                    token_counts = [len(tokens) for tokens in token_lists]
                else:
                    # Fallback to individual encoding
                    token_counts = [len(self.tokenizer.encode(text)) for text in uncached_texts]
                
                # Update results and cache
                for idx, count in zip(uncached_indices, token_counts):
                    results[idx] = count
                    # Cache result (with size limit)
                    if len(self._cache) < 1000:
                        text_hash = hash(uncached_texts[uncached_indices.index(idx)][:100])
                        self._cache[text_hash] = count
                        
            except Exception as e:
                logger.warning(f"Batch token counting failed: {e}")
                # Fallback to individual counting
                for idx, text in zip(uncached_indices, uncached_texts):
                    try:
                        count = len(self.tokenizer.encode(text))
                        results[idx] = count
                    except:
                        results[idx] = len(text) // 4  # Rough estimate
        
        return results
    
    def clear_cache(self):
        """Clear the token counting cache"""
        self._cache.clear()

# Convenience function for the main task
def create_optimized_processor(chunk_size: int = 1000, 
                                chunk_overlap: int = 200,
                                performance_mode: str = "balanced") -> StreamingTextProcessor:
    """
    Create an optimized text processor based on performance requirements
    
    Args:
        performance_mode: "fast", "balanced", or "quality"
    """
    if performance_mode == "fast":
        return StreamingTextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cleaning_mode="fast",
            min_chunk_length=30
        )
    elif performance_mode == "quality":
        return StreamingTextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cleaning_mode="thorough",
            min_chunk_length=100
        )
    else:  # balanced
        return StreamingTextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cleaning_mode="fast",
            min_chunk_length=50
        )