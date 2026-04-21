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
from fuzzywuzzy import fuzz
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Enhanced citation data structure supporting document chunks"""
    id: str
    text: str
    url: str = ""  # Optional for document citations
    title: Optional[str] = None
    description: Optional[str] = None
    preview_image: Optional[str] = None
    source_type: str = "document"
    confidence: float = 1.0
    relevant_excerpt: Optional[str] = None
    
    # 🆕 Document-specific fields
    page_number: Optional[int] = None
    source_id: Optional[str] = None
    document_title: Optional[str] = None
    chunk_index: Optional[int] = None
    similarity_score: Optional[float] = None
    
    # Metadata for future expansion
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    High-performance citation processor with real-time extraction and enrichment.
    
    WORKFLOW:
    1. RAG retrieval finds document chunks with metadata (page_number, source_id, etc.)
    2. System prompt instructs LLM to cite using specific formats: [Case Name, p. X]
    3. LLM generates response with natural citations: "...violated rights [Brown v. Board, p. 25]"
    4. Regex patterns detect citations in streaming LLM output
    5. Citations matched to original chunks by page_number/content
    6. Rich Citation objects created with document metadata + clickable URLs
    
    EXAMPLE FLOW:
    ```
    Input Chunk: {
        content: "The Supreme Court opinion in Brown states...",
        page_number: 26,
        source_id: "uuid-123",
        metadata: {"title": "Brown v. Board Opinion"}
    }
    
    LLM Output 🤖 : "...facilities are unequal [Brown v. Board, p. 26]"
    
    Detected Citation: {
        id: "case:Brown v. Board:p26",
        text: "Brown v. Board, p. 26", 
        page_number: 26,
        source_id: "uuid-123",
        url: "/documents/uuid-123#page=26",
        relevant_excerpt: "The Supreme Court opinion in Brown states..."
    }
    ```
    
    SUPPORTED PATTERNS:
    - [Case Name, p. X] → Legal case citations
    - [Document Title, p. X] → Document page references  
    - [1], [2] → Simple chunk references
    - Page X, p. X → Inline page mentions
    
    NOTE: No tool calling needed - pure regex extraction from LLM's natural language.
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

    def _parse_page_string(self, page_str: str) -> list[int]:
        """
        UTIL FUNCTION
        Parses a page string like "3-4" or "1, 3, 10" into a list of numbers.
        """
        pages = set()
        parts = page_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # It's a range, e.g., "3-4"
                try:
                    start, end = map(int, part.split('-'))
                    if end < start:
                        start, end = end, start # Handle "5-3"
                    pages.update(range(start, end + 1))
                except ValueError:
                    continue # Skip malformed range like "3-"
            else:
                # It's a single number
                try:
                    pages.add(int(part))
                except ValueError:
                    continue # Skip malformed part like "1,"
        return sorted(list(pages))

    def _parse_metadata(self, metadata_obj: Any) -> Dict:
            """Safely parse metadata that might be a dict or a JSON string. Fetched from Postgres (Supabase)"""
            if isinstance(metadata_obj, dict):
                return metadata_obj
            
            if isinstance(metadata_obj, str):
                try:
                    return json.loads(metadata_obj)
                    logger.debug(f"✅ Metadata JSON decoded properly: {metadata_obj[:50]}")
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Metadata JSONDecodeError: {metadata_obj[:50]}")
                    return {}
            
            return {}

# --- MAIN FUNCTIONS ----------------------- #

    # async def extract_citations_from_streaming_text(
    #     self, 
    #     accumulated_text: str,
    #     relevant_chunks: List[Dict],
    #     seen_citations: set = None
    # ) -> Tuple[List[Citation], set]:
    #     """
    #     [PROBABLY OBSOLETE]
    #     Extract citations from streaming text in real-time
        
    #     Args:
    #         accumulated_text: The text accumulated so far
    #         relevant_chunks: Available source chunks for citation
    #         seen_citations: Set of already processed citation IDs
            
    #     Returns:
    #         Tuple of (new_citations, updated_seen_set)
    #     """
    #     if seen_citations is None:
    #         seen_citations = set()
        
    #     new_citations = []
        
    #     # Try each citation pattern
    #     for pattern in self.citation_patterns:
    #         matches = re.findall(pattern, accumulated_text, re.IGNORECASE)
            
    #         for match in matches:
    #             citation_data = self._process_citation_match(match, relevant_chunks)
                
    #             if citation_data and citation_data.id not in seen_citations:
    #                 # Calculate confidence score
    #                 citation_data.confidence = self._calculate_confidence(citation_data)
                    
    #                 new_citations.append(citation_data)
    #                 seen_citations.add(citation_data.id)
                    
    #                 logger.info(f"📎 New citation extracted: {citation_data.id} ({citation_data.confidence:.2f})")
        
    #     return new_citations, seen_citations

    # def extract_document_citations_from_chunks(
    #     self, 
    #     accumulated_text: str,
    #     relevant_chunks: List[Dict],
    #     seen_citations: set = None
    # ) -> Tuple[List[Citation], set]:
    #     """
    #     [PROBABLY OBSOLETE]
    #     Extract document-based citations from streaming text
        
    #     Handles patterns like:
    #     - [1] or (1) -> references chunk 1
    #     - "Page 25" -> references specific page
    #     - [Case Name, p. 45] -> legal case citation
    #     """
    #     if seen_citations is None:
    #         seen_citations = set()
        
    #     new_citations = []
        
    #     # Pattern 1: Simple chunk references [1], [2], (1), etc.
    #     chunk_pattern = r'\[(\d+)\]|\((\d+)\)'
    #     chunk_matches = re.findall(chunk_pattern, accumulated_text)
        
    #     for match in chunk_matches:
    #         chunk_num = match[0] or match[1]  # Handle both [1] and (1) patterns
    #         citation_id = f"chunk:{chunk_num}"
            
    #         if citation_id not in seen_citations:
    #             citation = self._create_document_citation(
    #                 chunk_num, relevant_chunks, "chunk_reference"
    #             )
    #             if citation:
    #                 new_citations.append(citation)
    #                 seen_citations.add(citation_id)
        
    #     # Pattern 2: Page references "Page X", "p. X", "pg. X"
    #     page_pattern = r'(?:Page|p\.|pg\.)\s*(\d+)'
    #     page_matches = re.findall(page_pattern, accumulated_text, re.IGNORECASE)
        
    #     for page_num in page_matches:
    #         citation_id = f"page:{page_num}"
            
    #         if citation_id not in seen_citations:
    #             citation = self._find_citation_by_page(
    #                 int(page_num), relevant_chunks
    #             )
    #             if citation:
    #                 new_citations.append(citation)
    #                 seen_citations.add(citation_id)
        
    #     # Pattern 3: Legal case citations "Case Name, p. X"
    #     case_pattern = r'([A-Z][^,]+),\s*p\.\s*(\d+)'
    #     case_matches = re.findall(case_pattern, accumulated_text)
        
    #     for case_name, page_num in case_matches:
    #         citation_id = f"case:{case_name.strip()}:p{page_num}"
            
    #         if citation_id not in seen_citations:
    #             citation = self._find_legal_case_citation(
    #                 case_name.strip(), int(page_num), relevant_chunks
    #             )
    #             if citation:
    #                 new_citations.append(citation)
    #                 seen_citations.add(citation_id)
        
    #     return new_citations, seen_citations

    def extract_inline_citations_from_content(self, content: str) -> List[Dict]:
        """
        Description:
            Extract inline citations (e.g. [IRB_-_Hadley_v_Baxendale_1854_.pdf, p. 3]) from 
            streamed 📡 accumulated markdown content. If a multi-page citation is detected
            it is split before any excerpt matching take place.

            This method can now handle single-page and multipage references
            - [Doc.pdf, p. 5]
            - [Doc.pdf, pp. 3-4]
            - [Doc.pdf, pp. 1, 3, 10]

        Returns: 
            List of {text, page_number, document_name}

        NOTE: May cause the others functions (extract_citations_from_streaming_text, 
        extract_document_citations_from_chunks, _create_document_citation, 
        _find_citation_by_page, _extract_relevant_excerpt, etc...) to be obsolete
        """
        # Pattern: [text, p. number] or [text, Page number]
        pattern = r'\[([^,\]]+),\s*(?:p\.|Page)\s*(\d+)\]'
        
        citations_found = []
        for match in re.finditer(pattern, content):
            full_text = match.group(0)
            doc_name = match.group(1).strip()
            page_str = match.group(2).strip() # e.g., "5", "3-4", "1, 3, 10"
            
            # Use your new parser to get a list of pages
            pages = self._parse_page_string(page_str) # <--- YOUR NEW HELPER `_parse_page_string()`
            
            for page_num in pages:
                # Create a *separate* citation object for each page
                citations_found.append({
                    'full_text': full_text, # The *original* text match
                    'document_name': doc_name,
                    'page_number': int(page_num),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'is_multi_page_source': len(pages) > 1
                })
        
        logger.info(f"📚 Found {len(citations_found)} inline citation(s) in accumulated content")
        logger.info(f"  🙌 Extracted citation object:\n{citations_found}")
        return citations_found

    def match_inline_citations_to_chunks(
        self, 
        inline_citations: List[Dict],
        relevant_chunks: List[Dict]
    ) -> Dict[str, Citation]:
        """
        Match inline citations to document chunks using fuzzy logic (robust to LLM typos)
        
        Args:
        - list of inline citations
        - list of relevent chunk exerpts

        Returns: 
        - Dict mapping several citation_text -> Citation object

        NOTE: May cause the others functions (extract_citations_from_streaming_text, 
        extract_document_citations_from_chunks, _create_document_citation, 
        _find_citation_by_page, _extract_relevant_excerpt, etc...) to be obsolete
        """
        citation_map = {}
        # Ensure relevant_chunks is a list, even if empty
        relevant_chunks = relevant_chunks if relevant_chunks else []

        try: # Outer try block for the main loop over inline citations
            for inline_cite in inline_citations:
                try: # Inner try block for processing a single inline citation
                    # --- 🐞 DEBUG: Log the citation we're trying to match ---
                    logger.debug(f"--- Attempting to match Citation: {inline_cite.get('full_text', 'N/A')} ---")

                    # Extract required info, handle potential missing keys
                    doc_name = inline_cite.get('document_name')
                    page_num = inline_cite.get('page_number')
                    full_text = inline_cite.get('full_text')

                    if not doc_name or page_num is None or not full_text:
                        logger.warning(f"⚠️ Skipping invalid inline citation data: {inline_cite}")
                        continue # Skip to the next inline_cite

                    found_match = False # Flag to check if a match was found for this inline_cite

                    # --- Loop through retrieved chunks to find a match ---
                    for chunk_idx, chunk in enumerate(relevant_chunks):
                        try: # Innermost try block for processing a single chunk
                            # --- 🐞 DEBUG: Log chunk details ---
                            chunk_id = chunk.get('id', 'N/A')
                            logger.debug(f"  Comparing with Chunk Index: {chunk_idx}, Chunk ID: {chunk_id}")

                            # Safely get chunk page number and metadata
                            # page_number is a top-level DB column; fall back to metadata['page']
                            # for older chunks uploaded before the column was added.
                            chunk_page = chunk.get('page_number')
                            metadata = self._parse_metadata(chunk.get('metadata', {}))
                            if chunk_page is None:
                                chunk_page = metadata.get('page') or metadata.get('estimated_page')
                                if chunk_page is not None:
                                    try:
                                        chunk_page = int(chunk_page)
                                    except (ValueError, TypeError):
                                        chunk_page = None

                            # title is a top-level column returned by the HNSW JOIN query;
                            # metadata blob doesn't carry it, so check top-level first.
                            chunk_doc_name = (
                                chunk.get('title')
                                or metadata.get('title')
                                or metadata.get('filename')
                                or ''
                            )
                            source_id = chunk.get('source_id') # Get source_id for URL generation

                            # --- 🐞 DEBUG: Log comparison details ---
                            logger.info(
                                f"    Inline Cite Details: (doc_name='{doc_name}', page={page_num})"
                            )
                            logger.info(
                                f"    Chunk Details      : (chunk_docname='{chunk_doc_name}', page={chunk_page})"
                            )

                            # # === Perform matching (ALSOLUTE TEXT MATCHING) ================
                            # page_match = chunk_page == page_num
                            # # Robust, case-insensitive, bidirectional name matching
                            # doc_match = False
                            # if chunk_doc_name: # Ensure chunk_doc_name is not empty
                            #     doc_match = (doc_name.lower() in chunk_doc_name.lower() or
                            #                  chunk_doc_name.lower() in doc_name.lower())

                            # logger.info(f"    Page match: {page_match}, Doc match: {doc_match}")
                            
                            # === Perform matching (FUZZY LOGIC -- LLM Practical) =============
                            page_match = chunk_page == page_num
                            doc_match = False
                            similarity_threshold = 85 # Adjust as needed (e.g., 85%)

                            if doc_name and chunk_doc_name: # Check if both names exist
                                # Calculate similarity ratio (0-100)
                                ratio = fuzz.ratio(doc_name.lower(), chunk_doc_name.lower())
                                partial_ratio = fuzz.partial_ratio(doc_name.lower(), chunk_doc_name.lower())

                                # Consider it a match if either ratio is above the threshold
                                if ratio >= similarity_threshold or partial_ratio >= similarity_threshold:
                                    doc_match = True
                                    logger.info(f"    Fuzzy Match: ratio={ratio}, partial={partial_ratio} (Threshold={similarity_threshold})")
                                else:
                                    logger.info(f"    Fuzzy Match Failed: ratio={ratio}, partial={partial_ratio} (Threshold={similarity_threshold})")

                            logger.info(f"    Page match: {page_match}, Doc match (Fuzzy): {doc_match}")

                            # If match found, create Citation object
                            if page_match and doc_match:
                                logger.info(f"    ✅ MATCH FOUND for chunk {chunk_idx}")

                                # Generate unique citation ID (e.g., cite-MyDocument-p5)
                                citation_id = f"cite-{doc_name.replace('.pdf', '').replace('.docx', '').replace(' ', '-')}-p{page_num}"

                                # Generate internal URL, handle missing source_id gracefully
                                url = f"/documents/{source_id}#page={page_num}" if source_id else ""
                                if not url:
                                    logger.warning(f"⚠️ Could not generate URL for citation {citation_id} - missing source_id in chunk {chunk_id}")


                                citation = Citation(
                                    id=citation_id,
                                    title=chunk_doc_name, # Use matched chunk's title/filename
                                    text=full_text, # The original text from LLM, e.g., "[Doc.pdf, p. 5]"
                                    source_type="document",
                                    page_number=page_num,
                                    document_title=chunk_doc_name, # Redundant but fine for DB schema
                                    source_id=source_id,
                                    confidence=chunk.get('similarity', 0.8), # Use similarity score if available
                                    relevant_excerpt=self._extract_relevant_excerpt(
                                        chunk.get('content', ''), max_length=300
                                    ),
                                    url=url, # Generated internal URL
                                    metadata={
                                        'chunk_id': chunk.get('id'),
                                        # Use index from enumerate for robustness
                                        'chunk_index': chunk_idx,
                                        'similarity_score': chunk.get('similarity')
                                    }
                                )

                                # --- 🐞 DEBUG: Log the created object ---
                                logger.debug(f"    CREATED CITATION OBJECT: {citation}")

                                citation_map[full_text] = citation
                                found_match = True
                                break # Stop searching chunks for this inline_cite once matched

                            # --- 🐞 DEBUG: (Optional) Log if no match for this chunk ---
                            # else:
                            #     logger.debug(f"    ❌ NO MATCH for chunk {chunk_idx}")

                        except Exception as chunk_error:
                            logger.error(f"❌ Error processing chunk index {chunk_idx} (ID: {chunk.get('id', 'N/A')}): {chunk_error}", exc_info=False)
                            # Continue to the next chunk

                    # --- 🐞 DEBUG: Log if no match was found after checking all chunks ---
                    if not found_match:
                        logger.warning(f"⚠️ No matching chunk found for inline citation: {full_text}")

                except KeyError as ke:
                    logger.error(f"❌ Missing key in inline citation data '{inline_cite}': {ke}", exc_info=False)
                    # Continue to the next inline_cite
                except Exception as inner_error:
                    logger.error(f"❌ Error processing inline citation '{inline_cite.get('full_text', 'N/A')}': {inner_error}", exc_info=False)
                    # Continue to the next inline_cite

        except Exception as outer_error:
            logger.error(f"💥 Unexpected error during citation matching loop: {outer_error}", exc_info=True) # Log full traceback here
            # Depending on desired resilience, you could 'pass' or 'raise outer_error'
            # Returning the partially built map allows processing to continue.
            pass

        # --- 🐞 DEBUG: Log the final map ---
        logger.info(f"--- Finished Matching: Returning citation_map with {len(citation_map)} entries ---")
        return citation_map

    def expand_multi_page_citations(self, text: str) -> str:
        """
        Finds multi-page citations and expands them into individual, 
        fully-formatted links.
        
        Example:
        [Doc.pdf, pp. 3-4] 
        ...becomes...
        [Doc.pdf, p. 3](#cite-Doc-p3){.chat-citation-link} [Doc.pdf, p. 4](#cite-Doc-p4){.chat-citation-link}
        """
        
        def replacer(match):
            doc_name = match.group(1) # "IRB_-_Hadley_v_Baxendale_1854_.pdf"
            page_str = match.group(2) # "3-4" or "1, 3, 10"
            
            # Parse the page string into a list of integers
            pages =self._parse_page_string(page_str)
            
            links = []
            for page in pages:
                # --- Create the standard cite-id ---
                # "IRB_-_Hadley_v_Baxendale_1854_.pdf" -> "IRB_-_Hadley_v_Baxendale_1854_"
                cleaned_doc_name = doc_name.replace('.pdf', '').replace('.docx', '').replace(' ', '-')
                cite_id = f"cite-{cleaned_doc_name}-p{page}"
                
                # --- Build the full, correct markdown link ---
                display_text = f"[{doc_name}, p. {page}]"
                link_id = f"(#{cite_id})"
                link_class = "{.chat-citation-link}"
                
                links.append(f"{display_text}{link_id}{link_class}")
                
            # Join all the newly created links with a space
            return " ".join(links)

        # This regex finds:
        # \[           -> a literal [
        # ([^,\]]+)    -> Group 1: The document name (anything not a comma or bracket)
        # ,\s*pp\.\s* -> ", pp. " (with optional whitespace)
        # ([\d,\-\s]+) -> Group 2: The page numbers/ranges (digits, commas, dashes, whitespace)
        # \]           -> a literal ]
        pattern = r'\[([^,\]]+),\s*pp\.\s*([\d,\-\s]+)\]'
        
        return re.sub(pattern, replacer, text)

    def convert_inline_citations_to_markdown_links(
        self,
        content: str,
        citation_map: Dict[str, Citation]
    ) -> str:
        """
        Description:
        - Convert [Document.pdf, p. 5] to [Document.pdf, p. 5](#cite-id){.chat-citation-link} =tThis creates clickable citations that work with LinkPreviewManager and is 
        readable by the markdown-it-attrs plugin whuch uses the {.class} syntax
        
        - Cleans up any malformed or missed citations.

        Args:
        - content (str): [Document.pdf, p. 5]

        Returns:
        - modified_content (str): [Document.pdf, p. 5](#cite-id){.chat-citation-link}
        
        """
        modified_content = content
        
        # --- 1. Expand multi-page citations that were in the map ---
        # This handles the [Doc, pp. 3-4] case
        for citation_text, citation_obj in citation_map.items():
            if citation_obj.metadata.get('is_multi_page_source'):
                # This was a multi-page, but we only have one object for it
                continue # Skip, let step 2 handle it (...better to use the regex expansion here for consistency)
            
            # --- 2. Convert single-page citations from the map, add (#cite-id){.chat-citation-link}---
            safe_text = re.escape(citation_text)
            citation_link = f"[{citation_text.strip('[]')}](#{citation_obj.id}){{.chat-citation-link}}"
            modified_content = re.sub(safe_text, citation_link, modified_content)

        # --- 3. Run your new cleanup functions (from previous chat) ---
        # These will find and fix *anything* the LLM messed up.
        
        # First, expand any multi-page citations the map might have missed
        modified_content = self.expand_multi_page_citations(modified_content)
        
        # Second, fix any malformed single-page links
        modified_content = self.fix_malformed_citations(modified_content)
        
        return modified_content

    def _create_document_citation(
        self, chunk_num: str, relevant_chunks: List[Dict], citation_type: str
    ) -> Optional[Citation]:
        """Create citation from document chunk"""
        try:
            chunk_idx = int(chunk_num) - 1
            
            if 0 <= chunk_idx < len(relevant_chunks):
                chunk = relevant_chunks[chunk_idx]
                
                # Extract document metadata
                metadata = self._parse_metadata(chunk.get('metadata'))
                # metadata = chunk.get('metadata', {})
                
                return Citation(
                    id=f"chunk:{chunk_num}",
                    text=f"Document {chunk_num}",
                    source_type="document",
                    confidence=chunk.get('similarity', 0.8),
                    relevant_excerpt=self._extract_relevant_excerpt(
                        chunk.get('content', ''), max_length=150
                    ),
                    
                    # Document-specific fields
                    page_number=chunk.get('page_number'),
                    source_id=chunk.get('source_id'),
                    document_title=metadata.get('title') or metadata.get('filename'),
                    chunk_index=chunk_idx,
                    similarity_score=chunk.get('similarity'),
                    
                    # Build display URL for internal documents
                    url=self._build_document_url(chunk),
                    title=self._build_document_title(chunk),
                    description=f"Page {chunk.get('page_number', 'N/A')} - {metadata.get('title', 'Document')}",
                    
                    metadata={
                        'citation_type': citation_type,
                        'chunk_id': chunk.get('id'),
                        'project_id': chunk.get('project_id'),
                        'num_tokens': chunk.get('num_tokens'),
                        'extraction_time': datetime.utcnow().isoformat()
                    }
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Invalid chunk reference: {chunk_num} - {e}")
            return None

    def _find_citation_by_page(
        self, page_num: int, relevant_chunks: List[Dict]
    ) -> Optional[Citation]:
        """Find citation by page number across chunks"""
        
        # Look for chunk with matching page number
        for chunk in relevant_chunks:
            if chunk.get('page_number') == page_num:
                # metadata = chunk.get('metadata', {})
                metadata = self._parse_metadata(chunk.get('metadata'))
                
                return Citation(
                    id=f"page:{page_num}",
                    text=f"Page {page_num}",
                    source_type="document",
                    confidence=chunk.get('similarity', 0.8),
                    relevant_excerpt=self._extract_relevant_excerpt(
                        chunk.get('content', ''), max_length=200
                    ),
                    
                    page_number=page_num,
                    source_id=chunk.get('source_id'),
                    document_title=metadata.get('title') or metadata.get('filename'),
                    similarity_score=chunk.get('similarity'),
                    
                    url=self._build_document_url(chunk, page_num),
                    title=f"Page {page_num} - {metadata.get('title', 'Document')}",
                    description=f"Content from page {page_num}",
                    
                    metadata={
                        'citation_type': 'page_reference',
                        'chunk_id': chunk.get('id'),
                        'project_id': chunk.get('project_id')
                    }
                )
        
        return None

    def _find_legal_case_citation(
        self, case_name: str, page_num: int, relevant_chunks: List[Dict]
    ) -> Optional[Citation]:
        """Find legal case citation by name and page"""
        
        # Look for chunk containing case name and page
        for chunk in relevant_chunks:
            content = chunk.get('content', '').lower()
            # metadata = chunk.get('metadata', {})
            metadata = self._parse_metadata(chunk.get('metadata'))
            
            if (case_name.lower() in content and 
                chunk.get('page_number') == page_num):
                
                return Citation(
                    id=f"case:{case_name}:p{page_num}",
                    text=f"{case_name}, p. {page_num}",
                    source_type="legal_case",
                    confidence=chunk.get('similarity', 0.9),  # Higher for legal citations
                    relevant_excerpt=self._extract_case_excerpt(content, case_name),
                    
                    page_number=page_num,
                    source_id=chunk.get('source_id'),
                    document_title=case_name,
                    similarity_score=chunk.get('similarity'),
                    
                    url=self._build_document_url(chunk, page_num),
                    title=f"{case_name} (Page {page_num})",
                    description=f"Legal case citation from {case_name}",
                    
                    metadata={
                        'citation_type': 'legal_case',
                        'case_name': case_name,
                        'chunk_id': chunk.get('id'),
                        'project_id': chunk.get('project_id')
                    }
                )
        
        return None

    def _build_document_url(self, chunk: Dict, page_num: int = None) -> str:
        """Build internal document URL for navigation"""
        source_id = chunk.get('source_id')
        page = page_num or chunk.get('page_number')
        
        if source_id and page:
            return f"/documents/{source_id}#page={page}"
        elif source_id:
            return f"/documents/{source_id}"
        else:
            return ""

    def _build_document_title(self, chunk: Dict) -> str:
        """Build human-readable title for document citation"""
        metadata = self._parse_metadata(chunk.get('metadata'))
        title = metadata.get('title') or metadata.get('filename', 'Document')
        page = chunk.get('page_number')
        
        if page:
            return f"{title} (Page {page})"
        else:
            return title

    def _extract_case_excerpt(self, content: str, case_name: str) -> str:
        """Extract relevant excerpt around case name"""
        content_lower = content.lower()
        case_lower = case_name.lower()
        
        # Find case name position
        case_pos = content_lower.find(case_lower)
        if case_pos == -1:
            return content[:200] + "..."
        
        # Extract context around case name
        start = max(0, case_pos - 100)
        end = min(len(content), case_pos + len(case_name) + 100)
        
        excerpt = content[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
        
        return excerpt

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
        if len(enriched_citations) > 0: logger.info(f"🔗 Enriched citation idx 0: {enriched_citations[0]} ")
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

# --- FIXING AND VALIDATION -------#

    def validate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Validate and clean citations:
        This method handles both:
        - external urls
        - internal document citation urls 'document/source_id/#page_num'
        """
        valid_citations = []
        seen_urls = set()
        
        for citation in citations:
            # Skip duplicates
            if citation.url in seen_urls:
                continue

            # Skip URL validation for internal document links
            if citation.url and citation.url.startswith('/documents/'):
                valid_citations.append(citation)
                seen_urls.add(citation.url)
                logger.warning(f"📌 found internal document citation URL: {citation.url}")
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

    def fix_malformed_citations(self, text: str) -> str:
        """
        Cleans up single-page citations that are missing parts.
        - Adds missing {.chat-citation-link}
        - Converts raw [Doc, p. 3] links to the full format
        """
        
        # === 1. Fix Missing Class ===
        # Finds any "(#cite-...)" that is NOT immediately followed by a "{"
        # Uses a negative lookahead (?!\{)
        class_pattern = r'(\(\#cite-[^)]*\))(?!\{)'
        text = re.sub(class_pattern, r'\1{.chat-citation-link}', text)

        # === 2. Fix Missing ID & Class ===
        # This finds raw [Doc, p. 3] links that were never processed.
        
        def replacer_single(match):
            doc_name = match.group(1) # "IRB_-_Hadley_v_Baxendale_1854_.pdf"
            page_num = match.group(2) # "3"
            
            # --- Create the standard cite-id ---
            cleaned_doc_name = doc_name.replace('.pdf', '').replace('.docx', '').replace(' ', '-')
            cite_id = f"cite-{cleaned_doc_name}-p{page_num}"
            
            # --- Build the full, correct markdown link ---
            display_text = f"[{doc_name}, p. {page_num}]"
            link_id = f"(#{cite_id})"
            link_class = "{.chat-citation-link}"
            
            return f"{display_text}{link_id}{link_class}"

        # This regex finds:
        # \[           -> a literal [
        # ([^,\]]+)    -> Group 1: The document name
        # ,\s*p\.\s* -> ", p. "
        # (\d+)        -> Group 2: The page number
        # \]           -> a literal ]
        # (?![\(\{])   -> Negative lookahead: asserts the link is NOT followed
        #                by a ( or { (meaning it's unprocessed)
        id_class_pattern = r'\[([^,\]]+),\s*p\.\s*(\d+)\](?![\(\{])'
        text = re.sub(id_class_pattern, replacer_single, text)
        
        return text

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
