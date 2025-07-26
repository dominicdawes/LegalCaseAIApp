"""
# utils/llm_clients/llm_factory_enhanced.py
Enhanced LLM Factory with citation-aware prompt engineering
"""

import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from utils.llm_clients.anthropic_client import AnthropicClient
from utils.llm_clients.openai_client import OpenAIClient

@dataclass
class CitationPromptTemplate:
    """Template for citation-aware prompts"""
    system_prompt: str
    citation_format: str
    highlight_format: str
    context_template: str

class CitationAwareLLMFactory:
    """Enhanced LLM Factory with citation and streaming support"""
    
    CITATION_TEMPLATES = {
        "academic": CitationPromptTemplate(
            system_prompt="""You are an expert research assistant. When referencing information from the provided context, you MUST use the exact citation format: [descriptive text](citation:N) where N is the chunk number (1, 2, 3, etc.).

CITATION RULES:
- Every factual claim must be cited using [text](citation:N) format
- Use descriptive text that summarizes what you're citing
- Number citations sequentially starting from 1
- Only cite information that directly appears in the provided chunks
- For metrics or statistics, wrap them in **bold** for highlighting

EXAMPLE: 
"The latest framework [ELLMER](citation:1) achieves **95% success rate** in manipulation tasks, while [RoboDexVLM](citation:2) focuses on dexterous control."
""",
            citation_format="[{text}](citation:{num})",
            highlight_format="**{text}**",
            context_template="""RELEVANT CONTEXT:
{chunks}

CHAT HISTORY:
{history}

USER QUERY: {query}

Provide a comprehensive answer with proper citations and highlighted key metrics."""
        ),
        
        "conversational": CitationPromptTemplate(
            system_prompt="""You are a helpful AI assistant. Reference sources naturally using [descriptive text](citation:N) format. Make your response engaging and conversational while being accurate.

GUIDELINES:
- Use citations for factual claims: [source description](citation:N)
- Highlight important metrics with **bold**
- Keep a friendly, accessible tone
- Explain complex concepts clearly
""",
            citation_format="[{text}](citation:{num})",
            highlight_format="**{text}**",
            context_template="""Here's what I found that might help:

{chunks}

Previous conversation:
{history}

Your question: {query}

Let me give you a comprehensive answer:"""
        )
    }

    @classmethod
    def create_citation_aware_client(
        cls,
        provider: str,
        model_name: str,
        temperature: float = 0.7,
        citation_style: str = "academic",
        streaming: bool = True
    ) -> "CitationAwareLLMClient":
        """Create an LLM client with citation awareness"""
        
        # Get base client
        if provider.lower() == "anthropic":
            base_client = AnthropicClient(
                model_name=model_name,
                temperature=temperature,
                streaming=streaming,
                max_tokens=4000
            )
        elif provider.lower() == "openai":
            base_client = OpenAIClient(
                model_name=model_name,
                temperature=temperature,
                streaming=streaming
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        template = cls.CITATION_TEMPLATES.get(citation_style, cls.CITATION_TEMPLATES["academic"])
        
        return CitationAwareLLMClient(base_client, template)

class CitationAwareLLMClient:
    """Wrapper that adds citation awareness to any LLM client"""
    
    def __init__(self, base_client: Any, template: CitationPromptTemplate):
        self.base_client = base_client
        self.template = template
        self.model_name = getattr(base_client, 'model_name', 'unknown')
        
    async def stream_chat_with_citations(
        self,
        query: str,
        relevant_chunks: List[Dict],
        chat_history: List[Dict] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response with real-time citation processing
        """
        # Build context
        context = self._build_citation_context(query, relevant_chunks, chat_history or [])
        
        # Track citations and content
        accumulated_content = ""
        citation_map = {}
        current_citations = []
        
        try:
            # Stream from base client
            if hasattr(self.base_client, 'stream_chat'):
                # Native async streaming
                async for chunk in self.base_client.stream_chat(context, self.template.system_prompt):
                    accumulated_content += chunk
                    
                    # Process citations in real-time
                    new_citations, updated_content = self._process_citations_realtime(
                        chunk, accumulated_content, relevant_chunks, citation_map
                    )
                    
                    yield {
                        "type": "content_delta",
                        "content": chunk,
                        "accumulated_content": accumulated_content,
                        "new_citations": new_citations,
                        "processed_chunk": updated_content
                    }
                    
            else:
                # Simulate streaming for non-async clients
                response = self.base_client.chat(context)
                
                # Process in chunks
                chunk_size = 15
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    accumulated_content += chunk
                    
                    new_citations, updated_content = self._process_citations_realtime(
                        chunk, accumulated_content, relevant_chunks, citation_map
                    )
                    
                    yield {
                        "type": "content_delta", 
                        "content": chunk,
                        "accumulated_content": accumulated_content,
                        "new_citations": new_citations,
                        "processed_chunk": updated_content
                    }
                    
                    await asyncio.sleep(0.03)  # Simulate streaming delay
        
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
            return
            
        # Final processing
        final_citations = self._extract_all_citations(accumulated_content, relevant_chunks)
        
        yield {
            "type": "complete",
            "final_content": accumulated_content,
            "all_citations": final_citations,
            "total_citations": len(final_citations)
        }

    def _build_citation_context(
        self, 
        query: str, 
        chunks: List[Dict], 
        history: List[Dict]
    ) -> str:
        """Build context with numbered chunks for citations"""
        
        # Format chunks with numbers
        numbered_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[CHUNK {i}]\nSource: {chunk.get('title', 'Unknown')}\nURL: {chunk.get('url', 'N/A')}\nContent: {chunk.get('content', '')}\n"
            numbered_chunks.append(chunk_text)
        
        chunks_text = "\n".join(numbered_chunks)
        
        # Format chat history
        history_text = ""
        if history:
            for msg in history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user').title()
                content = msg.get('content', '')
                history_text += f"{role}: {content}\n"
        
        # Use template to build final context
        return self.template.context_template.format(
            chunks=chunks_text,
            history=history_text,
            query=query
        )

    def _process_citations_realtime(
        self,
        new_chunk: str,
        accumulated_content: str,
        relevant_chunks: List[Dict],
        citation_map: Dict
    ) -> tuple[List[Dict], str]:
        """Process citations in real-time as content streams"""
        
        # Look for citation patterns in the new chunk
        citation_pattern = r'\[([^\]]+)\]\(citation:(\d+)\)'
        matches = re.findall(citation_pattern, new_chunk)
        
        new_citations = []
        
        for text, citation_num in matches:
            citation_id = f"citation:{citation_num}"
            
            # Skip if already processed
            if citation_id in citation_map:
                continue
                
            # Create citation object
            chunk_idx = int(citation_num) - 1
            if chunk_idx < len(relevant_chunks):
                chunk = relevant_chunks[chunk_idx]
                
                citation = {
                    "id": citation_id,
                    "text": text,
                    "url": chunk.get('url', ''),
                    "title": chunk.get('title', ''),
                    "description": chunk.get('description', ''),
                    "source_type": chunk.get('source_type', 'document'),
                    "confidence": chunk.get('similarity', 0.8),
                    "relevant_excerpt": chunk.get('content', '')[:200] + "..."
                }
                
                citation_map[citation_id] = citation
                new_citations.append(citation)
        
        return new_citations, new_chunk

    def _extract_all_citations(
        self, 
        content: str, 
        relevant_chunks: List[Dict]
    ) -> List[Dict]:
        """Extract all citations from final content"""
        
        citation_pattern = r'\[([^\]]+)\]\(citation:(\d+)\)'
        matches = re.findall(citation_pattern, content)
        
        citations = []
        seen_citations = set()
        
        for text, citation_num in matches:
            citation_id = f"citation:{citation_num}"
            
            if citation_id in seen_citations:
                continue
                
            seen_citations.add(citation_id)
            
            chunk_idx = int(citation_num) - 1
            if chunk_idx < len(relevant_chunks):
                chunk = relevant_chunks[chunk_idx]
                
                citation = {
                    "id": citation_id,
                    "text": text,
                    "url": chunk.get('url', ''),
                    "title": chunk.get('title', ''),
                    "description": chunk.get('description', ''),
                    "source_type": chunk.get('source_type', 'document'),
                    "confidence": chunk.get('similarity', 0.8),
                    "relevant_excerpt": chunk.get('content', '')[:200] + "..."
                }
                citations.append(citation)
        
        return citations

    def chat(self, prompt: str) -> str:
        """Synchronous chat for compatibility"""
        return self.base_client.chat(prompt)

# Usage example
async def example_usage():
    """Example of how to use the enhanced factory"""
    
    # Create citation-aware client
    client = CitationAwareLLMFactory.create_citation_aware_client(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.7,
        citation_style="academic",
        streaming=True
    )
    
    # Mock relevant chunks
    chunks = [
        {
            "content": "ELLMER framework achieves 95% success rate in manipulation tasks",
            "title": "ELLMER Research Paper",
            "url": "https://example.com/ellmer",
            "source_type": "academic"
        },
        {
            "content": "RoboDexVLM combines vision-language models with dexterous manipulation",
            "title": "RoboDexVLM Documentation", 
            "url": "https://example.com/robodex",
            "source_type": "research"
        }
    ]
    
    # Stream response with citations
    async for update in client.stream_chat_with_citations(
        query="What are the latest frameworks for robotic manipulation?",
        relevant_chunks=chunks
    ):
        print(f"Update type: {update['type']}")
        if update['type'] == 'content_delta':
            print(f"New content: {update['content']}")
            if update['new_citations']:
                print(f"New citations: {update['new_citations']}")

if __name__ == "__main__":
    asyncio.run(example_usage())