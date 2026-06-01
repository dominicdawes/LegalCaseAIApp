# tasks/note_tasks.py

"""
High-performance async RAG AI note creation tasks (outlines, summaries, compare-contrast)
Modernized to use persistent event loop pattern from chat_tasks.py for better throughput.
Note genereation (with RAG) is done without token streaming `llm_client.chat` returns full answer in one go 

🆕 IMPROVEMENTS:
- Async/non-blocking operations with persistent event loop
- Connection pooling and batched operations  
- Enhanced performance monitoring and observability
- Clean separation of sync Celery tasks and async operations
- Maintains legacy interface compatibility
"""

# ===== STANDARD LIBRARY IMPORTS =====
import gc
import re
import logging
import os
import time
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# ===== DATABASE & ASYNC =====
import asyncio
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# ===== CELERY & TASK QUEUE =====
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from celery.utils.log import get_task_logger
try:
    from anthropic import NotFoundError as AnthropicNotFoundError, AuthenticationError as AnthropicAuthError
    _PERMANENT_LLM_ERRORS = (
        AnthropicNotFoundError, AnthropicAuthError,
        ModuleNotFoundError, ImportError, SyntaxError,
        # Python code bugs — retrying will never fix these:
        TypeError, AttributeError, NameError, KeyError, IndexError,
    )
except ImportError:
    _PERMANENT_LLM_ERRORS = (
        ModuleNotFoundError, ImportError, SyntaxError,
        TypeError, AttributeError, NameError, KeyError, IndexError,
    )

# ===== MACHINE LEARNING & TEXT PROCESSING =====  
import tiktoken
from langchain_openai import OpenAIEmbeddings

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app, run_async_in_worker
from tasks.database import (
    get_db_connection, 
    get_redis_connection, 
    get_global_async_db_pool, 
    get_global_redis_pool, 
    init_async_pools, 
    check_db_pool_health
)
from utils.prompt_utils import load_yaml_prompt, build_prompt_template_from_yaml
from utils.supabase_utils import (
    insert_note_supabase_record,
    supabase_client,
)
# LightRAG disabled — import lazily inside lightrag_note_generation_async when re-enabled
# from utils.lightrag.lightrag_utils import lightrag_integration
from utils.llm_clients.llm_factory import LLMFactory
from utils.llm_clients.performance_monitor import PerformanceMonitor
from utils.note_processing.flashcard_processor import FlashcardProcessor
from utils.note_processing.quiz_processor import QuizProcessor

# ——— Logging & Env Load ———————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
logger.propagate = False
load_dotenv()

# ——— Configuration & Constants ————————————————————————————————————————————————————
USE_LIGHTRAG_INTEGRATION = False
USE_LANGGRAPH_AGENT = os.getenv("USE_LANGGRAPH_AGENT", "false").lower() == "true"
USE_ATTACK_OUTLINE_AGENT = os.getenv("USE_ATTACK_OUTLINE_AGENT", "false").lower() == "true"
USE_CASE_BRIEF_AGENT = os.getenv("USE_CASE_BRIEF_AGENT", "false").lower() == "true"
USE_COLD_CALL_AGENT = os.getenv("USE_COLD_CALL_AGENT", "false").lower() == "true"
USE_FLASHCARD_AGENT = os.getenv("USE_FLASHCARD_AGENT", "false").lower() == "true"
USE_VOYAGE_EMBEDDINGS = os.getenv("USE_VOYAGE_EMBEDDINGS", "true").lower() == "true"  # corpus is always Voyage-indexed

# Queue configuration
NOTES_QUEUE = 'notes'
# PARSE_QUEUE = 'parsing'
# EMBED_QUEUE = 'embedding'
# FINAL_QUEUE = 'finalize'

# ——— Configuration & Constants ————————————————————————————————————————————————————

# Performance, Retries & Batching
MAX_RETRIES = 2  # Reduced from 5 — agent runs take 20+ min each; 2 retries = 3 total attempts max
RETRY_BACKOFF_MULTIPLIER = 2
DEFAULT_RETRY_DELAY = 5
RATE_LIMIT = '150/m'

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Note Type Mapping
NOTE_TYPE_YAML_MAP = {
    "outline": "case-outline-prompt.yaml",
    "exam_questions": "exam-questions-prompt.yaml", 
    "case_brief": "case-brief-prompt.yaml",
    "compare_contrast": "compare-contrast-prompt.yaml",
    "flashcards": "flashcards-prompt.yaml",
    "cold_call": "cold-call-prompt.yaml",
    "quiz": "quiz-prompt.yaml",
    "attack_outline": "attack-outline-prompt.yaml",
}

# ——— Enhanced Base Task Class ————————————————————————————————————————————————————

class BaseTaskWithRetry(Task):
    """Enhanced base task with automatic retries and better error handling"""
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_kwargs = {"max_retries": MAX_RETRIES}
    retry_jitter = True

# ——— Async Note Generation Manager ———————————————————————————————————————————————

class AsyncNoteManager:
    """
    High-performance async note generation manager.
    
    Similar to StreamingChatManager but optimized for note generation:
    - Parallel embedding and prompt loading
    - Async chunk retrieval with connection pooling
    - Performance monitoring and observability
    - Clean error handling and resource management
    """
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self._initialized = False
        self._embedding_cache = {}  # Simple in-memory cache for this worker
        
    async def initialize(self):
        """🔧 Initialize async resources with health checks"""
        if self._initialized:
            db_healthy = await check_db_pool_health()
            if db_healthy:
                logger.info("✅ AsyncNoteManager already initialized and healthy")
                return
            else:
                logger.warning("⚠️ Resources unhealthy, reinitializing...")
        
        try:
            # Initialize global pools
            await init_async_pools()
            
            # Verify pools are available
            db_pool = get_global_async_db_pool()
            redis_pool = get_global_redis_pool()
            
            if not db_pool:
                raise RuntimeError("Failed to initialize database pool")
                
            self._initialized = True
            logger.info("🚀 AsyncNoteManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ AsyncNoteManager initialization failed: {e}")
            self._initialized = False
            raise

    async def generate_note_async(
        self,
        note_id: str,
        user_id: str,
        note_type: str,
        project_id: str,
        note_title: str,
        provider: str,
        model_name: str,
        num_sources: int,
        temperature: float = 0.7,
        addtl_params: Optional[Dict] = None,
    ) -> str:
        """
        🚀 Main async note generation workflow with parallel processing

        Key improvements:
        - Parallel execution of embedding and prompt loading
        - Async chunk retrieval with connection pooling
        - Performance monitoring and detailed logging
        - Clean resource management
        """
        if not self._initialized:
            await self.initialize()

        if addtl_params is None:
            addtl_params = {}

        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting async note generation: {note_type} for project {project_id}")

            # ── Pipeline routing log ────────────────────────────────────────────────
            _agent_active = {
                "exam_questions": USE_LANGGRAPH_AGENT,
                "attack_outline": USE_ATTACK_OUTLINE_AGENT,
                "case_brief":     USE_CASE_BRIEF_AGENT,
                "cold_call":      USE_COLD_CALL_AGENT,
                "flashcards":     USE_FLASHCARD_AGENT,
            }
            if _agent_active.get(note_type, False):
                logger.info(f"🤖 Using {note_type} agent")
            else:
                logger.info(f"🕰️ Using {note_type} legacy")

            # ── LangGraph agentic path ──────────────────────────────────────────────
            if note_type == "exam_questions" and USE_LANGGRAPH_AGENT:
                return await self._generate_exam_questions_agent(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    note_title=note_title,
                    addtl_params=addtl_params or {},
                )

            if note_type == "attack_outline" and USE_ATTACK_OUTLINE_AGENT:
                return await self._generate_attack_outline_agent(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    note_title=note_title,
                    addtl_params=addtl_params or {},
                )

            if note_type == "case_brief" and USE_CASE_BRIEF_AGENT:
                return await self._generate_case_brief_agent(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    note_title=note_title,
                    addtl_params=addtl_params or {},
                )

            if note_type == "cold_call" and USE_COLD_CALL_AGENT:
                return await self._generate_cold_call_agent(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    note_title=note_title,
                    addtl_params=addtl_params or {},
                )

            if note_type == "flashcards" and USE_FLASHCARD_AGENT:
                return await self._generate_flashcard_agent(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    note_title=note_title,
                    addtl_params=addtl_params or {},
                )

            # 🆕 PARALLEL EXECUTION - Load prompt and generate embedding concurrently
            logger.info("⚡ Executing parallel tasks: prompt loading + embedding generation")
            
            prompt_task = asyncio.create_task(
                self._load_prompt_async(note_type)
            )
            embedding_task = asyncio.create_task(
                self._get_embedding_async(note_type)
            )
            # Quiz JSON is large: ~1000 tokens per question + 500 buffer
            num_questions = (addtl_params or {}).get('num_questions', 10)
            quiz_token_budget = (num_questions * 1000 + 500) if note_type == "quiz" else 4056
            llm_task = asyncio.create_task(
                self._setup_llm_client_async(provider, model_name, temperature, max_output_tokens=quiz_token_budget)
            )
            
            # Wait for all parallel tasks
            prompt_yaml, embedding, llm_client = await asyncio.gather(
                prompt_task, embedding_task, llm_task
            )
            
            setup_time = time.time() - start_time
            logger.info(f"📊 Parallel setup completed in {setup_time*1000:.0f}ms")

            await self._update_note_progress_async(note_id, "PROCESSING")

            # Async chunk retrieval (Naive RAG)
            retrieval_start = time.time()
            subset = addtl_params.get("subset_document_ids") or []
            relevant_chunks = await self._fetch_relevant_chunks_async(
                embedding, project_id, source_ids=subset,
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks in {retrieval_time*1000:.0f}ms")

            # Resolve referenced_sources for persistence — all project docs or the explicit subset
            if subset:
                referenced_sources = [uuid.UUID(sid) for sid in subset]
            else:
                async with get_db_connection() as conn:
                    src_rows = await conn.fetch(
                        "SELECT id FROM document_sources WHERE project_id = $1",
                        uuid.UUID(project_id),
                    )
                referenced_sources = [row["id"] for row in src_rows]
            
            # Build context and generate note (typicaly BASE_PROMPT + TEMPLATE + CHUNKS)
            generation_start = time.time()
            context = self._build_note_context(
                prompt_yaml, 
                relevant_chunks, 
                note_type, 
                addtl_params
            )
            
            # Generate note content
            # Strip preamble for markdown note types (LLMs sometimes prefix with intro prose)
            _markdown_note_types = {"attack_outline", "case_brief", "outline", "compare_contrast", "exam_questions"}
            note_content = await self._generate_note_content_async(
                llm_client, context, provider,
                strip_preamble=(note_type in _markdown_note_types),
            )
            generation_time = time.time() - generation_start
            
            # 🆕 Async note persistence: Nomal Notes (one block) vs Flashcards, Cold Calls (discrete-blocks)
            save_start = time.time()
            if note_type == "flashcards"  or note_type == "cold_call":
                # Special flashcard processing and storage
                deck_id, num_cards = await self._save_flashcard_deck_and_cards_async(
                    note_id=note_id,
                    project_id=project_id,
                    user_id=user_id,
                    deck_name=note_title,
                    llm_output=note_content,
                    is_essential=(addtl_params or {}).get('is_essential', False),
                    num_sources=num_sources,
                    referenced_sources=referenced_sources,
                )
                logger.info(f"🃏 Created flashcard deck {deck_id} with {num_cards} cards")

                save_metrics = {
                    "deck_id": str(deck_id),
                    "num_cards": num_cards,
                    "storage_type": "flashcards"
                }
            elif note_type=="quiz":
                # Special quiz processing and storage
                num_questions_requested = (addtl_params or {}).get('num_questions', 10)

                quiz_note_id, num_questions_saved = await self._save_quiz_and_questions_async(
                    note_id=note_id,
                    user_id=user_id,
                    llm_output=note_content,
                    num_questions_requested=num_questions_requested,
                    is_essential=(addtl_params or {}).get('is_essential', False),
                    num_sources=num_sources,
                    referenced_sources=referenced_sources,
                )
                save_metrics = {
                    "quiz_note_id": str(quiz_note_id),
                    "num_questions": num_questions_saved,
                    "storage_type": "quiz"
                }
            else:
                # Regular note types - save to notes table
                await self._save_note_async(
                    note_id=note_id,
                    note_type=note_type,
                    content=note_content,
                    is_essential=(addtl_params or {}).get('is_essential', False),
                    num_sources=1,
                    referenced_sources=referenced_sources,
                )
                save_metrics = {"storage_type": "regular_note"}
            
            save_time = time.time() - save_start
            
            # 🆕 Performance logging with flashcard-specific metrics
            total_time = time.time() - start_time
            performance_metrics = {
                "note_type": note_type,
                "project_id": project_id,
                "setup_time": setup_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "save_time": save_time,
                "total_time": total_time,
                "chunks_used": len(relevant_chunks),
                "content_length": len(note_content),
                **save_metrics  # Include flashcard-specific metrics
            }
            
            await self._log_performance_metrics(performance_metrics)
            
            logger.info(f"✅ {note_type.title()} generation completed in {total_time*1000:.0f}ms")
            return note_content
            
        except Exception as e:
            logger.error(f"❌ Async note generation failed: {e}", exc_info=True)
            await self._handle_note_error(str(e), note_id=note_id)
            raise
        finally:
            # Clean up large objects
            try:
                del relevant_chunks, note_content, context
            except NameError:
                pass
            gc.collect()

    # LightRAG disabled — re-enable by uncommenting this method and restoring
    # the lightrag_integration import at the top of the file.
    # async def lightrag_note_generation_async(self, ...): ...

    async def cleanup_note_async(
        self,
        note_id: str,
        user_id: str,
        provider: str,
        model_name: str,
        temperature: float = 0.5
    ) -> str:
        """
        Fetches a user's note, enhances it using an LLM, and updates it in the database.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        logger.info(f"✨ Starting note cleanup for note_id: {note_id}")

        try:
            original_content = ""
            # 1. Fetch the existing note content using the async pool
            async with get_db_connection() as conn:
                record = await conn.fetchrow(
                    "SELECT content_markdown FROM notes WHERE id = $1 AND user_id = $2",
                    uuid.UUID(note_id), uuid.UUID(user_id)
                )
                if not record or not record['content_markdown']:
                    logger.warning(f"⚠️ Note {note_id} not found or is empty. Aborting cleanup.")
                    return "Note not found or empty."
                
                original_content = record['content_markdown']

            # 2. Build the prompt for the LLM
            prompt = f"""
            You are an expert editor. Review the following user-written note and enhance it.
            Your tasks are to:
            - Correct any spelling, grammar, and punctuation errors.
            - Improve sentence structure for better clarity and flow.
            - Format the entire note using clean and readable Markdown.
            - Do not add any new information or change the original meaning of the note.
            - Retain the user's original intent and tone.
            
            Here is the note to clean up:
            ---
            {original_content}
            ---
            
            output_format: markdown
            
            important formatting: no triple tick wrapping (e.g. ``` ```) or explict language identifier (e.g. ```markdown ```)
            """
            # 3. Get LLM client and generate the cleaned content
            llm_client = await self._setup_llm_client_async(provider, model_name, temperature)
            cleaned_content = await self._generate_note_content_async(llm_client, prompt, provider)

            # NEW: Add this block to remove wrapping markdown code fences
            # This pattern finds a string that starts with ```, optionally followed by a language name,
            # captures everything in between (.+?), and ends with ```. It then replaces the
            # entire match with just the captured group. The re.DOTALL flag is crucial
            # to ensure that the '.' special character matches newlines.
            pattern = r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$"
            match = re.search(pattern, cleaned_content, re.DOTALL)
            if match:
                # If the pattern matches, extract the content from the capture group
                cleaned_content = match.group(1).strip()
            # END NEW BLOCK

            # 4. Update the note in the database within a transaction
            async with get_db_connection() as conn:
                async with conn.transaction():
                    await conn.execute(
                        """
                        UPDATE notes 
                        SET suggested_cleanup = $1, updated_at = NOW()
                        WHERE id = $2 AND user_id = $3
                        """,
                        cleaned_content, uuid.UUID(note_id), uuid.UUID(user_id)
                    )
            
            total_time = time.time() - start_time
            logger.info(f"✅ Note cleanup for {note_id} completed in {total_time:.2f}s.")
            return cleaned_content

        except Exception as e:
            logger.error(f"❌ Note cleanup failed for {note_id}: {e}", exc_info=True)
            # You could potentially log this error to the 'notes' table as well
            raise
            
    async def _load_prompt_async(self, note_type: str) -> tuple:
        """Returns the full yaml dict and all keys"""
        
        yaml_file = NOTE_TYPE_YAML_MAP.get(note_type)
        if not yaml_file:
            raise ValueError(f"Unknown note_type: {note_type}")
        
        logger.info(f"📋 Loading prompt YAML for NOTE TYPE: {note_type}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        yaml_dict = await loop.run_in_executor(
            None, load_yaml_prompt, yaml_file
        )
        
        # No longer need to extract parts here, just return the whole thing
        logger.info(f"✅ YAML prompt config loaded successfully for {note_type}")
        return yaml_dict

    async def _get_embedding_async(self, note_type: str) -> List[float]:
        """Async 'note prompt' embedding generation with caching
        Description:
            Converts baseline prompts to vector representation. Note embeddings are not seen by the LLM only vector-similarity lookup.
            Base prompts should be instructive/SEO based to to match with the most relevant chunk vectors. 
            i.e. packed with domain specific search terms: "core legal concepts, black letter law, claim, olding, etc..."
        """
        
        if note_type in self._embedding_cache:
            logger.info(f"🎯 Cache HIT for {note_type} embedding")
            return self._embedding_cache[note_type]
        
        # Load prompt to get base query
        yaml_file = NOTE_TYPE_YAML_MAP.get(note_type)
        yaml_dict = load_yaml_prompt(yaml_file)
        
        # Look for the new key. Fallback to base_prompt for older files.
        retrieval_query = yaml_dict.get("retrieval_prompt", yaml_dict.get("base_prompt"))

        if not retrieval_query:
            logger.error(f"No retrieval_prompt or base_prompt found in {yaml_file}")
            raise ValueError(f"Missing prompt for embedding in {note_type}")

        logger.info(f"🤖 Generating embedding for {note_type}")
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._generate_embedding_sync, retrieval_query
        )
        
        self._embedding_cache[note_type] = embedding
        logger.info(f"💾 Cached embedding for {note_type}")
        
        return embedding
    
    def _generate_embedding_sync(self, query: str) -> List[float]:
        """Synchronous embedding generation for thread pool"""
        embedder = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            max_retries=3,
            request_timeout=60
        )
        return embedder.embed_query(query)

    async def _setup_llm_client_async(self, provider: str, model_name: str, temperature: float, max_output_tokens: int = 4056):
        """🆕 Async LLM client setup"""
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None,
            lambda: LLMFactory.get_client_for(provider, model_name, temperature, False, max_output_tokens=max_output_tokens)
        )
        logger.info(f"🤖 LLM client setup: {provider}/{model_name} (max_output_tokens={max_output_tokens})")
        return client

    async def _fetch_relevant_chunks_async(
        self,
        embedding: List[float],
        project_id: str,
        k: int = 10,
        source_ids: List[str] = None,
    ) -> List[Dict]:
        """🆕 Async chunk retrieval with connection pooling
        Args:
        - embedding (List): vectorized question prompt
        - k (int): Top k relevant chunks, (default is 10)
        - source_ids (List[str]): optional subset of document_sources UUIDs to restrict retrieval
        """
        vector_str = '[' + ','.join(map(str, embedding)) + ']'

        async with get_db_connection() as conn:
            if source_ids:
                uuid_list = [uuid.UUID(sid) for sid in source_ids]
                rows = await conn.fetch(
                    "SELECT * FROM match_document_chunks_hnsw($1, $2, $3, NULL, $4)",
                    project_id, vector_str, k, uuid_list,
                )
                logger.info(f"🎯 Subset retrieval ({len(source_ids)} docs): {len(rows)} chunks")
            else:
                rows = await conn.fetch(
                    "SELECT * FROM match_document_chunks_hnsw($1, $2, $3)",
                    project_id, vector_str, k,
                )
                logger.info(f"🎯 Project-wide retrieval: {len(rows)} chunks")

        return [dict(row) for row in rows]

    def _build_note_context(
        self, 
        prompt_yaml: Dict,
        relevant_chunks: List[Dict], 
        note_type: str, 
        addtl_params: Dict
    ) -> str:
        """
        Build context for note generation with smart parameter 
        handling (organized by page numbers)
        """
        # 1. Build Page Context (Standard for all)
        pages_dict = {}
        for chunk in relevant_chunks:
            page_num = chunk.get('page_number') or 'Unknown'
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(chunk)
        
        page_contexts = []
        for page_num in sorted(pages_dict.keys(), key=lambda x: x if isinstance(x, int) else float('inf')):
            chunks_on_page = pages_dict[page_num]
            page_content = "\n".join(chunk["content"] for chunk in chunks_on_page)
            page_contexts.append(
                f"=== Page {page_num} ===\n"
                f"Source: {chunks_on_page[0].get('title', 'Unknown Document')}\n"
                f"{page_content}\n"
            )
        chunk_context = "\n".join(page_contexts)
    
        # 2. Extract Standard Keys
        # NOW STANDARDIZED: Every YAML must have system_prompt and template
        system_prompt = prompt_yaml.get("system_prompt", "You are a helpful AI research assistant.")
        prompt_template = build_prompt_template_from_yaml(prompt_yaml)
        
        # 3. Handle Variable Injection (Custom Logic only where vars differ)
        if note_type == "exam_questions":
            # Special Case: Exam questions needs example injection
            example = prompt_yaml.get("example_issue_spotter", "")
            num_questions = addtl_params.get("num_questions", 5)
            formatted_template = prompt_template.format(
                context=chunk_context,
                n_questions=num_questions
            )
            # Inject example between system and template
            final_context = f"{system_prompt}\n\n## GOLD-STANDARD EXAMPLE\n{example}\n\n## YOUR TASK\n{formatted_template}"

        elif note_type in ["flashcards", "cold_call", "quiz"]:
            # Special Case: Numeric parameters
            count_param = "num_questions" if note_type == "quiz" else "num_cards"
            count_val = addtl_params.get(count_param, 10)
            
            # Dynamic kwarg unpacking for format
            fmt_args = {"context": chunk_context}
            fmt_args[count_param] = count_val
            
            formatted_template = prompt_template.format(**fmt_args)
            final_context = f"{system_prompt}\n\n{formatted_template}"

        else:
            # Standard Case (Attack Outline, Case Brief, Outline, Summary)
            # Relies purely on standardized YAML keys
            formatted_template = prompt_template.format(context=chunk_context)
            final_context = f"{system_prompt}\n\n{formatted_template}"
        
        logger.info(f"📝 Built context: {len(final_context)} characters")
        return final_context

    @staticmethod
    def _strip_llm_preamble(content: str) -> str:
        """Strip any intro prose before the first Markdown header.

        LLMs sometimes prefix responses with 'Here is your outline...' etc.
        This finds the first line starting with '#' and discards everything before it.
        """
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                stripped = "\n".join(lines[i:]).strip()
                if i > 0:
                    logger.debug(f"🧹 Stripped {i} preamble line(s) from LLM response.")
                return stripped
        # No header found — return as-is (e.g. plain-text note types)
        return content.strip()

    async def _generate_note_content_async(
        self, llm_client, context: str, provider: str, strip_preamble: bool = False
    ) -> str:
        """🆕 Async note content generation"""

        logger.info(f"🧠 Generating note content with {provider}")

        # Run LLM generation in thread pool
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, llm_client.chat, context
        )

        if strip_preamble:
            content = self._strip_llm_preamble(content)

        logger.info(f"✅ Generated {len(content)} characters of content")
        return content

    async def _save_note_async(
        self,
        note_id: str,
        note_type: str,
        content: str,
        is_essential: bool,
        num_sources: int,
        referenced_sources: List = None,
    ):
        """Async note persistence — updates the stub row created at request time."""

        logger.info(f"💾 Saving {note_type} note to database")

        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE notes SET
                    content_markdown        = $1,
                    is_generated            = $2,
                    is_shareable            = $3,
                    is_essential            = $4,
                    num_sources_based_on    = $5,
                    referenced_sources      = $6,
                    note_progress_status    = 'COMPLETE',
                    error_message           = NULL
                WHERE id = $7
                """,
                content, True, False, is_essential, num_sources,
                [str(s) for s in referenced_sources] if referenced_sources else [], note_id,
            )

        logger.info(f"✅ Note saved successfully")

    async def _save_flashcard_deck_and_cards_async(
        self,
        note_id: str,
        project_id: str,
        user_id: str,
        deck_name: str,
        llm_output: str,
        is_essential: bool,
        num_sources: int,
        referenced_sources: List = None,
    ) -> tuple:
        """
        🆕 Async flashcard processing and database insertion
        
        Args:
            project_id: The project ID
            user_id: The user ID  
            deck_name: Name for the flashcard deck
            llm_output: Raw LLM response containing flashcard content
            
        Returns:
            Tuple of (deck_id, num_cards)
        """
        try:
            logger.info(f"🃏 Processing flashcards for deck: {deck_name}")
            
            # Initialize processor in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            processor = FlashcardProcessor()
            
            # Parse the LLM output in thread pool
            deck_data, cards_list = await loop.run_in_executor(
                None, processor.parse_flashcard_content, llm_output, deck_name
            )
            
            # Validate the parsed data
            is_valid = await loop.run_in_executor(
                None, processor.validate_flashcard_data, deck_data, cards_list
            )
            
            if not is_valid:
                raise ValueError("Invalid flashcard data structure")
            
            logger.info(f"📋 Parsed {len(cards_list)} flashcards for deck: {deck_name}")
            
            # Use async database connection for both operations
            async with get_db_connection() as conn:
                # Start transaction
                async with conn.transaction():
                    # Update the stub note row created at request time
                    await conn.execute(
                        """
                        UPDATE notes SET
                            description          = $1,
                            num_cards            = $2,
                            is_active            = $3,
                            is_essential         = $4,
                            num_sources_based_on = $5,
                            referenced_sources   = $6,
                            note_progress_status = 'COMPLETE'
                        WHERE id = $7
                        """,
                        deck_data['description'], deck_data['num_cards'],
                        deck_data.get('is_active', True), is_essential, num_sources,
                        [str(s) for s in referenced_sources] if referenced_sources else [], note_id,
                    )
                    deck_id = note_id

                    logger.info(f"✅ Updated flashcard deck stub with ID: {deck_id}")
                    
                    # Prepare individual cards for batch insert
                    if cards_list:
                        # Build bulk insert query
                        card_values = []
                        for card in cards_list:
                            card_values.extend([
                                str(uuid.uuid4()),  # id
                                deck_id,            # deck_id (is just a the id from public.notes)
                                user_id,            # user_id
                                project_id,         # project_id
                                card['front_content'],  # front_content
                                card['back_content'],   # back_content
                                card['card_order'],     # card_order
                                card['created_at'],     # created_at
                                card.get('is_active', True)  # is_active
                            ])
                        
                        # Batch insert individual cards
                        num_cards = len(cards_list)
                        placeholders = []
                        
                        for i in range(num_cards):
                            base = i * 9 + 1  # 9 fields per card
                            placeholders.append(
                                f"(${base}, ${base+1}, ${base+2}, ${base+3}, "
                                f"${base+4}, ${base+5}, ${base+6}, ${base+7}, ${base+8})"
                            )
                        
                        query = f"""
                            INSERT INTO individual_cards (
                                id, deck_id, user_id, project_id, front_content, 
                                back_content, card_order, created_at, is_active
                            ) VALUES {', '.join(placeholders)}
                        """
                        
                        await conn.execute(query, *card_values)
                        logger.info(f"✅ Inserted {num_cards} individual flashcards")
                        
                        return deck_id, num_cards
                    else:
                        logger.warning("⚠️ No cards to insert")
                        return deck_id, 0
            
        except Exception as e:
            logger.error(f"❌ Error saving flashcard deck and cards: {e}", exc_info=True)
            raise
    
    async def _save_quiz_and_questions_async(
        self,
        note_id: str,
        user_id: str,
        llm_output: str,
        num_questions_requested: int,
        is_essential: bool,
        num_sources: int,
        referenced_sources: List = None,
    ) -> tuple:
        """
        🆕 Async quiz processing and database insertion.

        Parses LLM JSON output and saves data to `notes`, `quiz_questions`,
        and `quiz_answers` tables within a single transaction.

        Args:
            project_id: The project ID
            user_id: The user ID
            quiz_title: Name for the quiz (saved to notes.title)
            llm_output: Raw LLM JSON response string
            num_questions_requested: The number of questions asked for
            is_essential: Flag for essential notes

        Returns:
            Tuple of (quiz_note_id, num_questions_saved)
        """
        try:
            logger.info(f"🧠 Processing quiz note {note_id[:8]}…")

            # 1. Initialize processor and parse/validate content
            # This part is synchronous but fast (no I/O)
            processor = QuizProcessor()

            # 🆕 DEBUG PRINT: Log the first 2000 chars of the output
            logger.info(f"Raw LLM Output ({len(llm_output)} chars) received for quiz:\n{llm_output[:2000]}...")

            # 🆕 Use the new "Aggressive Coercion" method
            # This one method replaces both parse_quiz_content and validate_quiz_data
            quiz_data = processor.parse_and_salvage_quiz(llm_output)
            # 🛑 REMOVED: processor.validate_quiz_data(quiz_data)

            questions_list = quiz_data.get("questions", [])
            num_questions_saved = len(questions_list)

            # 💡 OPTIONAL: Add this log for comparison
            logger.info(f"Successfully salvaged {num_questions_saved} out of {num_questions_requested} requested questions.")

            if num_questions_saved == 0:
                raise ValueError("No questions found in parsed quiz data.")

            logger.info(f"💾 Saving {num_questions_saved} quiz questions to DB")

            # 2. Use async database connection for all DB operations
            async with get_db_connection() as conn:
                # Start a single transaction for all inserts
                async with conn.transaction():
                    
                    # 3. Update the stub notes row created at request time
                    await conn.execute(
                        """
                        UPDATE notes SET
                            num_questions        = $1,
                            is_generated         = $2,
                            is_essential         = $3,
                            num_sources_based_on = $4,
                            referenced_sources   = $5,
                            note_progress_status = 'COMPLETE'
                        WHERE id = $6
                        """,
                        num_questions_saved, True, is_essential, num_sources,
                        [str(s) for s in referenced_sources] if referenced_sources else [], note_id,
                    )
                    quiz_note_id = note_id

                    logger.info(f"✅ Updated quiz note stub with ID: {quiz_note_id}")

                    # 4. Loop and insert all questions and their answers
                    for question_data in questions_list:
                        # Insert quiz_questions record
                        question_db_id = await conn.fetchval(
                            """
                            INSERT INTO quiz_questions (
                                id, quiz_id, user_id, question_text, hint, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                            RETURNING id
                            """,
                            str(uuid.uuid4()), quiz_note_id, user_id,
                            question_data['question_text'], question_data['hint'],
                            datetime.now(timezone.utc)
                        )

                        if not question_db_id:
                            raise Exception(f"Failed to insert question: {question_data['question_text']}")

                        # 5. Prepare and batch insert answers for this question
                        answers_data = question_data.get("answers", [])
                        if answers_data:
                            answer_records = [
                                (
                                    str(uuid.uuid4()),  # id
                                    question_db_id,     # question_id
                                    ans['answer_choice_text'], # answer_choice_text
                                    ans['is_correct'],  # is_correct
                                    ans['feedback'],    # feedback
                                    datetime.now(timezone.utc) # created_at
                                ) for ans in answers_data
                            ]
                            
                            await conn.copy_records_to_table(
                                'quiz_answers',
                                records=answer_records,
                                columns=('id', 'question_id', 'answer_choice_text', 'is_correct', 'feedback', 'created_at')
                            )

                    logger.info(f"✅ Successfully inserted {num_questions_saved} questions and all their answers.")
                    return quiz_note_id, num_questions_saved

        except Exception as e:
            logger.error(f"❌ Error saving quiz and questions: {e}", exc_info=True)
            # 🆕 DEBUG PRINT: On error, log the *full* problematic output
            if "llm_output" in locals():
                logger.error(f"--- FULL FAILED LLM OUTPUT ---\n{llm_output}\n--- END FAILED LLM OUTPUT ---")
            raise

    async def _log_performance_metrics(self, metrics: Dict):
        """🆕 Log performance metrics for monitoring"""
        try:
            async with get_redis_connection() as r:
                metric_data = {
                    **metrics,
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_type": "note_generation"
                }
                
                await r.lpush("note_performance_metrics", json.dumps(metric_data))
                await r.ltrim("note_performance_metrics", 0, 1000)
                
            logger.info(f"📊 Performance: {metrics['total_time']*1000:.0f}ms total, {metrics['chunks_used']} chunks used")
            
        except Exception as e:
            logger.warning(f"⚠️ Metrics logging failed: {e}")

    async def _update_note_progress_async(self, note_id: str, status: str, error_message: str = None):
        """Update note_progress_status (and optionally error_message) on the existing stub row."""
        try:
            async with get_db_connection() as conn:
                if error_message:
                    await conn.execute(
                        "UPDATE notes SET note_progress_status = $1, error_message = $2 WHERE id = $3",
                        status, error_message[:2000], note_id
                    )
                else:
                    await conn.execute(
                        "UPDATE notes SET note_progress_status = $1 WHERE id = $2",
                        status, note_id
                    )
            logger.info(f"📡 Note {note_id[:8]}… progress → {status}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update note progress [{status}] for {note_id}: {e}")

    async def _generate_exam_questions_agent(
        self,
        note_id: str,
        user_id: str,
        project_id: str,
        note_title: str,
        addtl_params: Dict,
    ) -> str:
        """
        Route exam_questions to the LangGraph agentic pipeline when
        USE_LANGGRAPH_AGENT=true.  Falls back to the standard RAG path
        on import error so a missing langgraph install never breaks prod.
        """
        try:
            from agents.exam_questions.graph import run_exam_agent
        except ImportError:
            logger.warning("langgraph not installed — falling back to standard RAG path")
            return await self.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type="exam_questions",
                project_id=project_id,
                note_title=note_title,
                provider="anthropic",
                model_name="claude-opus-4-7",
                num_sources=10,
                addtl_params=addtl_params,
            )

        n_questions = int(addtl_params.get("num_questions", 5))
        source_ids = addtl_params.get("source_ids") or []

        # If no explicit source_ids, fetch all for the project
        if not source_ids:
            async with get_db_connection() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    uuid.UUID(project_id),
                )
            source_ids = [str(r["id"]) for r in rows]

        await self._update_note_progress_async(note_id, "PROCESSING")

        # ── Ledger: initialise job + run ──────────────────────────────────────
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        job_uuid = uuid.UUID(note_id)
        run_meta = None
        _ledger_ok = False
        try:
            await ledger.ensure_job(
                job_id=job_uuid,
                project_id=project_id,
                source_ids=source_ids,
                job_type="exam_questions",
            )
            run_meta = await ledger.initialize_run(
                job_id=job_uuid,
                graph_name="exam_questions",
            )
            agent_thread_id = run_meta.langgraph_thread_id
            agent_run_id    = str(run_meta.run_id)
            _ledger_ok = True
        except Exception as ledger_exc:
            logger.warning(f"Ledger init failed (non-fatal, running without persistence): {ledger_exc}")
            agent_thread_id = note_id
            agent_run_id    = None

        # ── Run the agent ─────────────────────────────────────────────────────
        try:
            final_state = await run_exam_agent(
                request=note_title,
                project_id=project_id,
                source_ids=source_ids,
                n_questions=n_questions,
                use_voyage=USE_VOYAGE_EMBEDDINGS,
                thread_id=agent_thread_id,
                job_id=note_id if _ledger_ok else "",
                run_id=agent_run_id,
                user_id=user_id,
            )
            markdown = final_state.get("final_output") or ""
        except Exception as e:
            if run_meta:
                try:
                    await ledger.mark_run_failed(run_meta.run_id, e)
                    await ledger.set_job_status(job_uuid, "failed")
                except Exception:
                    pass
            logger.error(f"LangGraph agent failed: {e}", exc_info=True)
            raise

        # ── Ledger: mark success ──────────────────────────────────────────────
        if run_meta:
            try:
                await ledger.complete_run(run_meta.run_id)
                await ledger.set_job_status(job_uuid, "succeeded")
            except Exception as ledger_exc:
                logger.warning(f"Ledger completion update failed (non-fatal): {ledger_exc}")

        # ── Update notes stub with exam card count ────────────────────────────
        # exam_card_writer nodes (parallel fan-out) already persisted the
        # individual rows into exam_questions + exam_answers.  Here we just
        # stamp the parent notes row with the final count and status.
        persisted_ids = final_state.get("persisted_question_ids") or []
        num_persisted = len(persisted_ids)

        ref_sources = [str(sid) for sid in source_ids] if source_ids else []
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE notes SET
                    num_questions        = $1,
                    is_generated         = $2,
                    is_essential         = $3,
                    num_sources_based_on = $4,
                    referenced_sources   = $5::uuid[],
                    note_progress_status = 'COMPLETE',
                    error_message        = NULL
                WHERE id = $6
                """,
                num_persisted, True, False, len(source_ids), ref_sources, note_id,
            )

        logger.info(
            f"✅ Exam agent persisted {num_persisted} question+answer pairs "
            f"for note {note_id[:8]}…"
        )
        return markdown

    async def _generate_attack_outline_agent(
        self,
        note_id: str,
        user_id: str,
        project_id: str,
        note_title: str,
        addtl_params: Dict,
    ) -> str:
        """
        Route attack_outline to the LangGraph agentic pipeline when
        USE_ATTACK_OUTLINE_AGENT=true.  Falls back to the standard RAG path
        on import error so a missing langgraph install never breaks prod.
        """
        try:
            from agents.attack_outline.graph import run_attack_outline_agent
        except ImportError:
            logger.warning("langgraph not installed — falling back to standard RAG path for attack_outline")
            return await self.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type="attack_outline",
                project_id=project_id,
                note_title=note_title,
                provider="anthropic",
                model_name="claude-opus-4-7",
                num_sources=10,
                addtl_params=addtl_params,
            )

        source_ids = addtl_params.get("source_ids") or []

        # If no explicit source_ids, fetch all for the project
        if not source_ids:
            async with get_db_connection() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    uuid.UUID(project_id),
                )
            source_ids = [str(r["id"]) for r in rows]

        await self._update_note_progress_async(note_id, "PROCESSING")

        # ── Ledger: initialise job + run ──────────────────────────────────────
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        job_uuid = uuid.UUID(note_id)
        run_meta = None
        _ledger_ok = False          # tracks whether ensure_job committed a row
        try:
            await ledger.ensure_job(
                job_id=job_uuid,
                project_id=project_id,
                source_ids=source_ids,
                job_type="attack_outline",
            )
            run_meta = await ledger.initialize_run(
                job_id=job_uuid,
                graph_name="attack_outline",
            )
            agent_thread_id = run_meta.langgraph_thread_id
            agent_run_id    = str(run_meta.run_id)
            _ledger_ok = True
        except Exception as ledger_exc:
            # Ledger is down — continue without it, but blank out job_id so
            # _try_save_artifact calls inside nodes no-op instead of hammering
            # the DB with artifact saves that will FK-fail (no agent_jobs row).
            logger.warning(f"Ledger init failed (non-fatal): {ledger_exc}")
            agent_thread_id = note_id
            agent_run_id    = None

        # ── Run the agent ─────────────────────────────────────────────────────
        try:
            final_state = await run_attack_outline_agent(
                request=note_title,
                project_id=project_id,
                source_ids=source_ids,
                use_voyage=USE_VOYAGE_EMBEDDINGS,
                thread_id=agent_thread_id,
                job_id=note_id if _ledger_ok else "",
                run_id=agent_run_id,
                user_id=user_id,
            )
            markdown = final_state.get("final_output") or ""
        except Exception as e:
            if run_meta:
                try:
                    await ledger.mark_run_failed(run_meta.run_id, e)
                    await ledger.set_job_status(job_uuid, "failed")
                except Exception:
                    pass
            logger.error(f"Attack outline agent failed: {e}", exc_info=True)
            raise

        # ── Ledger: mark success ──────────────────────────────────────────────
        if run_meta:
            try:
                await ledger.complete_run(run_meta.run_id)
                await ledger.set_job_status(job_uuid, "succeeded")
            except Exception as ledger_exc:
                logger.warning(f"Ledger completion update failed (non-fatal): {ledger_exc}")

        # ── Persist the outline markdown to the notes stub ────────────────────
        # Pass source_ids as strings with an explicit ::uuid[] cast so asyncpg
        # sends a text array and Postgres handles the str→uuid coercion.
        # Passing uuid.UUID objects directly triggers "expected str, got UUID"
        # when the array codec infers text[] from the Python list type.
        ref_sources = [str(sid) for sid in source_ids] if source_ids else []
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE notes SET
                    content_markdown     = $1,
                    is_generated         = $2,
                    is_essential         = $3,
                    num_sources_based_on = $4,
                    referenced_sources   = $5::uuid[],
                    note_progress_status = 'COMPLETE',
                    error_message        = NULL
                WHERE id = $6
                """,
                markdown, True, False, len(source_ids), ref_sources, note_id,
            )

        logger.info(f"✅ Attack outline agent completed for note {note_id[:8]}…")
        return markdown

    async def _generate_case_brief_agent(
        self,
        note_id: str,
        user_id: str,
        project_id: str,
        note_title: str,
        addtl_params: Dict,
    ) -> str:
        """
        Route case_brief to the LangGraph agentic pipeline when
        USE_CASE_BRIEF_AGENT=true.  Falls back to the standard RAG path
        on import error so a missing langgraph install never breaks prod.
        """
        try:
            from agents.case_brief.graph import run_case_brief_agent
        except ImportError:
            logger.warning("langgraph not installed — falling back to standard RAG path for case_brief")
            return await self.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type="case_brief",
                project_id=project_id,
                note_title=note_title,
                provider="anthropic",
                model_name="claude-opus-4-7",
                num_sources=10,
                addtl_params=addtl_params,
            )

        source_ids = addtl_params.get("source_ids") or []

        # If no explicit source_ids, fetch all for the project
        if not source_ids:
            async with get_db_connection() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    uuid.UUID(project_id),
                )
            source_ids = [str(r["id"]) for r in rows]

        await self._update_note_progress_async(note_id, "PROCESSING")

        # ── Ledger: initialise job + run ──────────────────────────────────────
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        job_uuid = uuid.UUID(note_id)
        run_meta = None
        _ledger_ok = False
        try:
            await ledger.ensure_job(
                job_id=job_uuid,
                project_id=project_id,
                source_ids=source_ids,
                job_type="case_brief",
            )
            run_meta = await ledger.initialize_run(
                job_id=job_uuid,
                graph_name="case_brief",
            )
            agent_thread_id = run_meta.langgraph_thread_id
            agent_run_id    = str(run_meta.run_id)
            _ledger_ok = True
        except Exception as ledger_exc:
            logger.warning(f"Ledger init failed (non-fatal): {ledger_exc}")
            agent_thread_id = note_id
            agent_run_id    = None

        # ── Run the agent ─────────────────────────────────────────────────────
        try:
            final_state = await run_case_brief_agent(
                request=note_title,
                project_id=project_id,
                source_ids=source_ids,
                use_voyage=USE_VOYAGE_EMBEDDINGS,
                thread_id=agent_thread_id,
                job_id=note_id if _ledger_ok else "",
                run_id=agent_run_id,
                user_id=user_id,
            )
            markdown = final_state.get("final_output") or ""
        except Exception as e:
            if run_meta:
                try:
                    await ledger.mark_run_failed(run_meta.run_id, e)
                    await ledger.set_job_status(job_uuid, "failed")
                except Exception:
                    pass
            logger.error(f"Case brief agent failed: {e}", exc_info=True)
            raise

        # ── Ledger: mark success ──────────────────────────────────────────────
        if run_meta:
            try:
                await ledger.complete_run(run_meta.run_id)
                await ledger.set_job_status(job_uuid, "succeeded")
            except Exception as ledger_exc:
                logger.warning(f"Ledger completion update failed (non-fatal): {ledger_exc}")

        # ── Persist the brief markdown to the notes stub ──────────────────────
        ref_sources = [str(sid) for sid in source_ids] if source_ids else []
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE notes SET
                    content_markdown     = $1,
                    is_generated         = $2,
                    is_essential         = $3,
                    num_sources_based_on = $4,
                    referenced_sources   = $5::uuid[],
                    note_progress_status = 'COMPLETE',
                    error_message        = NULL
                WHERE id = $6
                """,
                markdown, True, False, len(source_ids), ref_sources, note_id,
            )

        logger.info(f"✅ Case brief agent completed for note {note_id[:8]}…")
        return markdown

    async def _generate_flashcard_agent(
        self,
        note_id: str,
        user_id: str,
        project_id: str,
        note_title: str,
        addtl_params: Dict,
    ) -> str:
        """Route flashcards to the LangGraph agentic pipeline when USE_FLASHCARD_AGENT=true."""
        try:
            from agents.flashcards.graph import run_flashcard_agent
        except ImportError:
            logger.warning("langgraph not installed — falling back to standard RAG path for flashcards")
            return await self.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type="flashcards",
                project_id=project_id,
                note_title=note_title,
                provider="anthropic",
                model_name="claude-opus-4-7",
                num_sources=10,
                addtl_params=addtl_params,
            )

        source_ids = addtl_params.get("document_ids") or []
        num_cards = int(addtl_params.get("num_cards", 10))
        is_essential = addtl_params.get("is_essential", False)

        if not source_ids:
            async with get_db_connection() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    uuid.UUID(project_id),
                )
            source_ids = [str(r["id"]) for r in rows]

        await self._update_note_progress_async(note_id, "PROCESSING")

        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        job_uuid = uuid.UUID(note_id)
        run_meta = None
        try:
            await ledger.ensure_job(
                job_id=job_uuid,
                project_id=project_id,
                source_ids=source_ids,
                job_type="flashcards",
            )
            run_meta = await ledger.initialize_run(
                job_id=job_uuid,
                graph_name="flashcards",
            )
            agent_thread_id = run_meta.langgraph_thread_id
            agent_run_id    = str(run_meta.run_id)
        except Exception as ledger_exc:
            logger.warning(f"Ledger init failed (non-fatal): {ledger_exc}")
            agent_thread_id = note_id
            agent_run_id    = None

        try:
            final_state = await run_flashcard_agent(
                request=note_title,
                project_id=project_id,
                source_ids=source_ids,
                num_cards=num_cards,
                use_voyage=USE_VOYAGE_EMBEDDINGS,
                is_essential=is_essential,
                thread_id=agent_thread_id,
                job_id=note_id,
                run_id=agent_run_id,
                user_id=user_id,
            )
        except Exception as e:
            if run_meta:
                try:
                    await ledger.mark_run_failed(run_meta.run_id, e)
                    await ledger.set_job_status(job_uuid, "failed")
                except Exception:
                    pass
            logger.error(f"Flashcard agent failed: {e}", exc_info=True)
            raise

        if run_meta:
            try:
                await ledger.complete_run(run_meta.run_id)
                await ledger.set_job_status(job_uuid, "succeeded")
            except Exception as ledger_exc:
                logger.warning(f"Ledger completion update failed (non-fatal): {ledger_exc}")

        num_cards_persisted = len(final_state.get("accepted_card_ids") or [])
        logger.info(f"🃏 Flashcard agent completed for note {note_id[:8]}… — {num_cards_persisted} cards")
        return final_state.get("final_output") or ""

    async def _generate_cold_call_agent(
        self,
        note_id: str,
        user_id: str,
        project_id: str,
        note_title: str,
        addtl_params: Dict,
    ) -> str:
        """
        Route cold_call to the LangGraph agentic pipeline when
        USE_COLD_CALL_AGENT=true.  Falls back to the standard RAG path
        on import error so a missing langgraph install never breaks prod.
        """
        try:
            from agents.cold_call.graph import run_cold_call_agent
        except ImportError:
            logger.warning("langgraph not installed — falling back to standard RAG path for cold_call")
            return await self.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type="cold_call",
                project_id=project_id,
                note_title=note_title,
                provider="anthropic",
                model_name="claude-opus-4-7",
                num_sources=10,
                addtl_params=addtl_params,
            )

        source_ids = addtl_params.get("source_ids") or []
        requested_sequence_count = int(addtl_params.get("num_sequences", 5))
        target_difficulty = addtl_params.get("target_difficulty", "day_one_t14")

        # If no explicit source_ids, fetch all for the project
        if not source_ids:
            async with get_db_connection() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM document_sources WHERE project_id = $1",
                    uuid.UUID(project_id),
                )
            source_ids = [str(r["id"]) for r in rows]

        await self._update_note_progress_async(note_id, "PROCESSING")

        # ── Ledger: initialise job + run ──────────────────────────────────────
        from agents.ledger import AgentLedgerService
        ledger = AgentLedgerService()
        job_uuid = uuid.UUID(note_id)
        run_meta = None
        try:
            await ledger.ensure_job(
                job_id=job_uuid,
                project_id=project_id,
                source_ids=source_ids,
                job_type="cold_call",
            )
            run_meta = await ledger.initialize_run(
                job_id=job_uuid,
                graph_name="cold_call",
            )
            agent_thread_id = run_meta.langgraph_thread_id
            agent_run_id    = str(run_meta.run_id)
        except Exception as ledger_exc:
            logger.warning(f"Ledger init failed (non-fatal): {ledger_exc}")
            agent_thread_id = note_id
            agent_run_id    = None

        # ── Run the agent ─────────────────────────────────────────────────────
        try:
            final_state = await run_cold_call_agent(
                request=note_title,
                project_id=project_id,
                source_ids=source_ids,
                requested_sequence_count=requested_sequence_count,
                target_difficulty=target_difficulty,
                use_voyage=USE_VOYAGE_EMBEDDINGS,
                thread_id=agent_thread_id,
                job_id=note_id,
                run_id=agent_run_id,
                user_id=user_id,
                note_id=note_id,
            )
            markdown = final_state.get("final_output") or ""
        except Exception as e:
            if run_meta:
                try:
                    await ledger.mark_run_failed(run_meta.run_id, e)
                    await ledger.set_job_status(job_uuid, "failed")
                except Exception:
                    pass
            logger.error(f"Cold call agent failed: {e}", exc_info=True)
            raise

        # ── Ledger: mark success ──────────────────────────────────────────────
        if run_meta:
            try:
                await ledger.complete_run(run_meta.run_id)
                await ledger.set_job_status(job_uuid, "succeeded")
            except Exception as ledger_exc:
                logger.warning(f"Ledger completion update failed (non-fatal): {ledger_exc}")

        # ── Persist the markdown summary to the notes stub ────────────────────
        ref_sources = [str(sid) for sid in source_ids] if source_ids else []
        export_result = final_state.get("export_result") or {}
        num_sequences = export_result.get("question_sequences_exported", 0)

        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE notes SET
                    content_markdown     = $1,
                    is_generated         = $2,
                    is_essential         = $3,
                    num_sources_based_on = $4,
                    referenced_sources   = $5::uuid[],
                    note_progress_status = 'COMPLETE',
                    error_message        = NULL
                WHERE id = $6
                """,
                markdown, True, False, len(source_ids), ref_sources, note_id,
            )

        logger.info(
            f"✅ Cold call agent completed for note {note_id[:8]}… "
            f"({num_sequences} sequences exported)"
        )
        return markdown

    async def _handle_note_error(
        self, error_message: str, note_id: Optional[str] = None,
    ):
        # Update the existing stub row — never INSERT a new one
        if note_id:
            await self._update_note_progress_async(note_id, "ERROR", error_message=error_message)
        else:
            logger.warning(f"⚠️ _handle_note_error called without note_id — error not persisted: {error_message[:200]}")

# ——— Global Manager Instance ————————————————————————————————————————————————————

note_manager = AsyncNoteManager()

# ——— [MAIN] Celery Task (Clean Interface) ———————————————————————————————————————

@celery_app.task(
    bind=True, 
    base=BaseTaskWithRetry,
    queue='notes',
    acks_late=True,
    rate_limit=RATE_LIMIT
)
def rag_note_task(
    self,
    user_id: str,
    note_type: str,
    project_id: str,
    note_title: str,
    provider: str,
    model_name: str,
    num_sources: int = 1,
    temperature: float = 0.7,
    addtl_params: Optional[Dict] = None,
    note_id: Optional[str] = None,   # trailing optional — backward-compatible with old dispatches
):
    """
    🚀 Enhanced RAG note generation task with async event loop
    
    🔄 CRITICAL: Uses run_async_in_worker() to execute async code 
    in the persistent worker event loop, preventing asyncio conflicts.
    
    🆕 IMPROVEMENTS:
    - Async/non-blocking operations for better throughput
    - Parallel execution of setup tasks (70% faster)
    - Connection pooling and batched operations
    - Enhanced performance monitoring and observability
    - Clean error handling and resource management
    
    🔄 MAINTAINS LEGACY INTERFACE:
    - Same function signature as original
    - Same FastAPI integration
    - Same error handling patterns
    """
    
    # Backward-compatibility: old dispatches (pre-observability) won't include note_id.
    # Generate one so the pipeline has a consistent ID even without a stub row.
    if note_id is None:
        note_id = str(uuid.uuid4())
        logger.warning(f"⚠️ note_id not provided — generated fallback {note_id[:8]}… (stub row not created)")

    # Reset status to PROCESSING on each attempt so the frontend doesn't
    # flash ERROR while the task is waiting between retries.
    if self.request.retries > 0:
        try:
            supabase_client.table("notes").update({
                "note_progress_status": "PROCESSING",
                "error_message": None,  # clear stale error from previous attempt
            }).eq("id", note_id).execute()
        except Exception:
            pass

    try:
        # Set explicit start time metadata
        task_id = self.request.id
        logger.info(f"🎯 Starting note task {task_id} for project: {project_id}")

        self.update_state(
            state="STARTED", 
            meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # 🔥 CRITICAL: Execute async workflow in persistent event loop
        result = run_async_in_worker(
            note_manager.generate_note_async(
                note_id=note_id,
                user_id=user_id,
                note_type=note_type,
                project_id=project_id,
                note_title=note_title,
                provider=provider,
                model_name=model_name,
                num_sources=num_sources,
                temperature=temperature,
                addtl_params=addtl_params,
            )
        )
        
        logger.info(f"✅ Note task {task_id} completed successfully")
        return "RAG Note Task success"

    except Exception as e:
        logger.error(f"❌ Note task for {note_type} failed: {e}", exc_info=True)

        # Permanent errors (bad model name, invalid API key, etc.) — never worth retrying.
        if _PERMANENT_LLM_ERRORS and isinstance(e, _PERMANENT_LLM_ERRORS):
            logger.error(f"💀 Permanent LLM error for note {note_id[:8]}, skipping retries: {e}")
            raise RuntimeError(str(e)) from e

        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            error_msg = (
                f"Note generation timed out and failed after {self.max_retries} retries. "
                f"The task repeatedly encountered errors and could not complete. "
                f"Last error: {str(e)[:300]}"
            )
            logger.error(f"💀 Note {note_id[:8]} exhausted {self.max_retries} retries: {e}")
            try:
                supabase_client.table("notes").update({
                    "note_progress_status": "MAX_RETRY_ERROR",
                    "error_message": error_msg[:2000],
                }).eq("id", note_id).execute()
            except Exception as upd_err:
                logger.warning(f"⚠️ Could not update note to MAX_RETRY_ERROR: {upd_err}")
            raise RuntimeError(error_msg) from e
    
    finally:
        # Clean up
        gc.collect()

@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    queue='notes', # Or a different queue if you prefer
    acks_late=True,
    rate_limit=RATE_LIMIT # Adjust as needed
)
def cleanup_note_task(
    self,
    note_id: str,
    user_id: str,
    provider: str,
    model_name: str,
    temperature: float = 0.5,
):
    """
    Celery task to clean up and enhance a user's existing note.
    """
    try:
        task_id = self.request.id
        logger.info(f"🚀 Starting cleanup_note_task {task_id} for note: {note_id}")
        self.update_state(
            state="STARTED",
            meta={"start_time": datetime.now(timezone.utc).isoformat()}
        )

        # Execute the async workflow in the worker's persistent event loop
        result = run_async_in_worker(
            note_manager.cleanup_note_async(
                note_id=note_id,
                user_id=user_id,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
            )
        )
        
        logger.info(f"✅ Cleanup task {task_id} completed successfully.")
        return "Note cleanup successful."

    except Exception as e:
        logger.error(f"❌ Cleanup task for note {note_id} failed: {e}", exc_info=True)
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            raise RuntimeError(
                f"Note cleanup failed permanently for note {note_id} after {self.max_retries} retries: {e}"
            ) from e

# ——— Legacy Support Functions (For Backward Compatibility) ——————————————————————

def fetch_relevant_chunks(query_embedding, project_id, match_count=10):
    """🔄 Legacy function maintained for backward compatibility"""
    logger.warning("⚠️ Using legacy fetch_relevant_chunks - consider upgrading")
    
    try:
        response = supabase_client.rpc(
            "match_document_chunks_hnsw",
            {
                "p_project_id": project_id,
                "p_query": query_embedding,
                "p_k": match_count,
            },
        ).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching relevant chunks: {e}", exc_info=True)
        raise

def save_note(project_id, user_id, note_type, note_title, content):
    """🔄 Legacy function maintained for backward compatibility"""
    logger.warning("⚠️ Using legacy save_note - consider upgrading")
    
    insert_note_supabase_record(
        client=supabase_client,
        table_name="notes",
        user_id=user_id,
        project_id=project_id,
        note_title=note_title,
        content_markdown=content,
        note_type=note_type,
        is_generated=True,
        is_shareable=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        num_sources=1,
    )

def trim_context_length(full_context, query, relevant_chunks, model_name, max_tokens):
    """🔄 Legacy function maintained with enhanced tokenizer support"""
    model_to_encoding = {
        "o4-mini": "o200k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "claude-3-5-sonnet": "cl100k_base",
    }
    
    encoding_name = model_to_encoding.get(model_name, "cl100k_base")
    
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    history = full_context
    while len(tokenizer.encode(history)) > max_tokens and relevant_chunks:
        relevant_chunks.pop()
        chunk_context = "\n\n".join(c["content"] for c in relevant_chunks)
        history = (
            f"Relevant Context:\n{chunk_context}\n\nUser Query: {query}\nAssistant:"
        )
    
    return history
