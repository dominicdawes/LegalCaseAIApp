# utils/llm_clients/citation_processor.py
"""
Citation processing utility for extracting, validating, and enriching citations
from LLM responses in real-time during streaming.

Features:
- Real-time citation extraction from streaming text
- Confidence scoring based on source quality
- Link preview fetching and caching
- Citation validation and deduplication
- Source type classification
"""

import re
import json
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Enhanced citation data structure"""
    id: str
    text: str
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    preview_image: Optional[str] = None
    source_type: str = "document"
    confidence: float = 1.0
    relevant_excerpt: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LinkPreview:
    """Link preview data structure"""
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    site_name: Optional[str] = None
    favicon: Optional[str] = None
    status_code: int = 200
    error: Optional[str] = None

class CitationProcessor:
    """
    High-performance citation processor with real-time extraction and enrichment
    """
    
    def __init__(self, redis_pool=None):
        self.redis_pool = redis_pool
        self.citation_patterns = [
            r'\[([^\]]+)\]\(citation:(\d+)\)',  # [text](citation:1)
            r'\[(\d+)\]',                       # [1] 
            r'\((\d+)\)',                       # (1)
            r'source\s*(\d+)',                  # source 1
        ]
        self.confidence_weights = {
            'academic': 0.95,
            'government': 0.9,
            'news': 0.85,
            'document': 0.8,
            'web': 0.7,
            'blog': 0.6,
            'social': 0.4,
        }
        
        # Session for HTTP requests
        self.http_session = None
        
    async def initialize(self):
        """Initialize async resources"""
        if not self.http_session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; CitationBot/1.0)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
            )
    
    async def cleanup(self):
        """Cleanup async resources"""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def extract_citations_from_streaming_text(
        self, 
        accumulated_text: str,
        relevant_chunks: List[Dict],
        seen_citations: set = None
    ) -> Tuple[List[Citation], set]:
        """
        Extract citations from streaming text in real-time
        
        Args:
            accumulated_text: The text accumulated so far
            relevant_chunks: Available source chunks for citation
            seen_citations: Set of already processed citation IDs
            
        Returns:
            Tuple of (new_citations, updated_seen_set)
        """
        if seen_citations is None:
            seen_citations = set()
        
        new_citations = []
        
        # Try each citation pattern
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, accumulated_text, re.IGNORECASE)
            
            for match in matches:
                citation_data = self._process_citation_match(match, relevant_chunks)
                
                if citation_data and citation_data.id not in seen_citations:
                    # Calculate confidence score
                    citation_data.confidence = self._calculate_confidence(citation_data)
                    
                    new_citations.append(citation_data)
                    seen_citations.add(citation_data.id)
                    
                    logger.info(f"📎 New citation extracted: {citation_data.id} ({citation_data.confidence:.2f})")
        
        return new_citations, seen_citations

    def _process_citation_match(
        self, 
        match: tuple, 
        relevant_chunks: List[Dict]
    ) -> Optional[Citation]:
        """Process a regex match into a Citation object"""
        
        # Handle different match patterns
        if len(match) == 2:
            # [text](citation:N) pattern
            text, citation_num = match
            citation_id = f"citation:{citation_num}"
        elif len(match) == 1:
            # [N] or (N) pattern
            citation_num = match[0]
            citation_id = f"citation:{citation_num}"
            text = f"Source {citation_num}"
        else:
            return None
        
        try:
            chunk_idx = int(citation_num) - 1  # Convert to 0-based index
            
            if 0 <= chunk_idx < len(relevant_chunks):
                chunk = relevant_chunks[chunk_idx]
                
                return Citation(
                    id=citation_id,
                    text=text,
                    url=chunk.get('url', ''),
                    title=chunk.get('title', ''),
                    description=chunk.get('description', ''),
                    source_type=self._classify_source_type(chunk),
                    relevant_excerpt=self._extract_relevant_excerpt(chunk.get('content', '')),
                    metadata={
                        'chunk_index': chunk_idx,
                        'similarity_score': chunk.get('similarity', 0.0),
                        'extraction_time': datetime.utcnow().isoformat()
                    }
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Invalid citation number: {citation_num} - {e}")
            return None

    def _classify_source_type(self, chunk: Dict) -> str:
        """Classify the source type based on URL and metadata"""
        url = chunk.get('url', '').lower()
        title = chunk.get('title', '').lower()
        
        # Academic sources
        if any(domain in url for domain in [
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            'ieee.org', 'acm.org', 'springer.com', 'elsevier.com'
        ]):
            return 'academic'
        
        # Government sources
        if url.endswith('.gov') or 'government' in title:
            return 'government'
        
        # News sources
        if any(domain in url for domain in [
            'cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'wsj.com'
        ]):
            return 'news'
        
        # Documentation/technical
        if any(keyword in url for keyword in [
            'docs.', 'documentation', 'manual', 'guide', 'tutorial'
        ]):
            return 'document'
        
        # Blog sources
        if any(keyword in url for keyword in [
            'blog', 'medium.com', 'dev.to', 'hashnode.com'
        ]):
            return 'blog'
        
        # Social media
        if any(domain in url for domain in [
            'twitter.com', 'linkedin.com', 'facebook.com', 'reddit.com'
        ]):
            return 'social'
        
        # Default to web
        return 'web'

    def _calculate_confidence(self, citation: Citation) -> float:
        """Calculate confidence score for a citation"""
        base_confidence = self.confidence_weights.get(citation.source_type, 0.7)
        
        # Adjust based on metadata
        similarity_score = citation.metadata.get('similarity_score', 0.0)
        
        # Boost for high similarity
        if similarity_score > 0.8:
            base_confidence += 0.1
        elif similarity_score < 0.6:
            base_confidence -= 0.1
        
        # Boost for complete information
        if citation.title and citation.description:
            base_confidence += 0.05
        
        # Ensure bounds
        return max(0.0, min(1.0, base_confidence))

    def _extract_relevant_excerpt(self, content: str, max_length: int = 200) -> str:
        """Extract a relevant excerpt from content"""
        if not content:
            return ""
        
        # Clean up the content
        cleaned = re.sub(r'\s+', ' ', content.strip())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # Try to find a good breaking point (sentence end)
        excerpt = cleaned[:max_length]
        last_period = excerpt.rfind('.')
        last_space = excerpt.rfind(' ')
        
        if last_period > max_length * 0.7:  # If period is not too early
            return excerpt[:last_period + 1]
        elif last_space > max_length * 0.8:  # If space is reasonable
            return excerpt[:last_space] + "..."
        else:
            return excerpt + "..."

    async def enrich_citations_with_previews(
        self, 
        citations: List[Citation],
        use_cache: bool = True
    ) -> List[Citation]:
        """
        Enrich citations with link previews and additional metadata
        """
        if not self.http_session:
            await self.initialize()
        
        enriched_citations = []
        
        for citation in citations:
            try:
                # Skip if no URL
                if not citation.url or citation.url.startswith('file://'):
                    enriched_citations.append(citation)
                    continue
                
                # Try cache first
                if use_cache and self.redis_pool:
                    cached_preview = await self._get_cached_preview(citation.url)
                    if cached_preview:
                        citation = self._apply_preview_to_citation(citation, cached_preview)
                        enriched_citations.append(citation)
                        continue
                
                # Fetch live preview
                preview = await self._fetch_link_preview(citation.url)
                
                if preview and not preview.error:
                    citation = self._apply_preview_to_citation(citation, preview)
                    
                    # Cache the result
                    if self.redis_pool:
                        await self._cache_preview(citation.url, preview)
                
                enriched_citations.append(citation)
                
            except Exception as e:
                logger.warning(f"⚠️ Preview enrichment failed for {citation.url}: {e}")
                enriched_citations.append(citation)
        
        logger.info(f"🔗 Enriched {len([c for c in enriched_citations if c.title])} citations with previews")
        return enriched_citations

    async def _fetch_link_preview(self, url: str) -> Optional[LinkPreview]:
        """Fetch link preview data from URL"""
        try:
            async with self.http_session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    return LinkPreview(
                        url=url,
                        status_code=response.status,
                        error=f"HTTP {response.status}"
                    )
                
                # Only process HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    return LinkPreview(url=url, error="Not HTML content")
                
                html = await response.text()
                return self._parse_html_preview(url, html)
                
        except asyncio.TimeoutError:
            return LinkPreview(url=url, error="Timeout")
        except Exception as e:
            return LinkPreview(url=url, error=str(e))

    def _parse_html_preview(self, url: str, html: str) -> LinkPreview:
        """Parse HTML to extract preview data"""
        preview = LinkPreview(url=url)
        
        # Simple regex-based parsing (for production, use BeautifulSoup)
        patterns = {
            'title': [
                r'<meta property="og:title" content="([^"]*)"',
                r'<meta name="twitter:title" content="([^"]*)"',
                r'<title>([^<]*)</title>'
            ],
            'description': [
                r'<meta property="og:description" content="([^"]*)"',
                r'<meta name="twitter:description" content="([^"]*)"',
                r'<meta name="description" content="([^"]*)"'
            ],
            'image': [
                r'<meta property="og:image" content="([^"]*)"',
                r'<meta name="twitter:image" content="([^"]*)"'
            ],
            'site_name': [
                r'<meta property="og:site_name" content="([^"]*)"'
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if value:
                        setattr(preview, field, value)
                        break
        
        # Clean up title
        if preview.title:
            preview.title = re.sub(r'\s+', ' ', preview.title)[:100]
        
        # Clean up description
        if preview.description:
            preview.description = re.sub(r'\s+', ' ', preview.description)[:300]
        
        # Make image URL absolute
        if preview.image and not preview.image.startswith('http'):
            preview.image = urljoin(url, preview.image)
        
        return preview

    def _apply_preview_to_citation(
        self, 
        citation: Citation, 
        preview: LinkPreview
    ) -> Citation:
        """Apply link preview data to citation"""
        if preview.title and not citation.title:
            citation.title = preview.title
        
        if preview.description and not citation.description:
            citation.description = preview.description
        
        if preview.image:
            citation.preview_image = preview.image
        
        if preview.site_name:
            citation.metadata['site_name'] = preview.site_name
        
        citation.metadata['preview_fetched'] = True
        citation.metadata['preview_status'] = preview.status_code
        
        return citation

    async def _get_cached_preview(self, url: str) -> Optional[LinkPreview]:
        """Get cached link preview"""
        if not self.redis_pool:
            return None
        
        try:
            import redis.asyncio as aioredis
            cache_key = f"link_preview:{hashlib.md5(url.encode()).hexdigest()}"
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                cached_data = await r.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return LinkPreview(**data)
        except Exception as e:
            logger.warning(f"⚠️ Cache lookup failed: {e}")
        
        return None

    async def _cache_preview(self, url: str, preview: LinkPreview):
        """Cache link preview data"""
        if not self.redis_pool:
            return
        
        try:
            import redis.asyncio as aioredis
            cache_key = f"link_preview:{hashlib.md5(url.encode()).hexdigest()}"
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as r:
                # Cache for 24 hours
                await r.setex(cache_key, 86400, json.dumps(asdict(preview)))
        except Exception as e:
            logger.warning(f"⚠️ Cache storage failed: {e}")

    def validate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Validate and clean citations"""
        valid_citations = []
        seen_urls = set()
        
        for citation in citations:
            # Skip duplicates
            if citation.url in seen_urls:
                continue
            
            # Skip invalid URLs
            if not self._is_valid_url(citation.url):
                logger.warning(f"⚠️ Invalid URL in citation: {citation.url}")
                continue
            
            # Skip low confidence citations
            if citation.confidence < 0.5:
                logger.warning(f"⚠️ Low confidence citation: {citation.id} ({citation.confidence:.2f})")
                continue
            
            valid_citations.append(citation)
            seen_urls.add(citation.url)
        
        logger.info(f"✅ Validated {len(valid_citations)}/{len(citations)} citations")
        return valid_citations

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def get_citation_summary(self, citations: List[Citation]) -> Dict[str, Any]:
        """Generate summary statistics for citations"""
        if not citations:
            return {"total": 0}
        
        source_types = {}
        confidence_scores = []
        
        for citation in citations:
            source_types[citation.source_type] = source_types.get(citation.source_type, 0) + 1
            confidence_scores.append(citation.confidence)
        
        return {
            "total": len(citations),
            "source_types": source_types,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "high_confidence_count": len([c for c in citations if c.confidence >= 0.8]),
            "with_previews": len([c for c in citations if c.preview_image or c.title])
        }

# Example usage
async def example_usage():
    """Example of how to use CitationProcessor"""
    processor = CitationProcessor()
    await processor.initialize()
    
    try:
        # Mock streaming text with citations
        streaming_text = "The latest research [shows promising results](citation:1) in 3D Gaussian Splatting [2], with 95% accuracy [3]."
        
        # Mock relevant chunks
        chunks = [
            {
                "content": "Research shows 95% accuracy in 3D rendering",
                "title": "3D Gaussian Splatting Research",
                "url": "https://example.com/research",
                "similarity": 0.9
            },
            {
                "content": "Performance improvements in rendering",
                "title": "Rendering Performance Study", 
                "url": "https://example.com/performance",
                "similarity": 0.85
            }
        ]
        
        # Extract citations
        citations, seen = await processor.extract_citations_from_streaming_text(
            streaming_text, chunks
        )
        
        # Enrich with previews
        enriched = await processor.enrich_citations_with_previews(citations)
        
        # Get summary
        summary = processor.get_citation_summary(enriched)
        
        print(f"Extracted {len(citations)} citations")
        print(f"Summary: {summary}")
        
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())