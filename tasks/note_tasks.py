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

# ===== MACHINE LEARNING & TEXT PROCESSING =====  
import tiktoken
from langchain_community.embeddings import OpenAIEmbeddings

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
    log_llm_error,
)
from utils.lightrag.lightrag_utils import lightrag_integration
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

# Queue configuration
NOTES_QUEUE = 'notes'
# PARSE_QUEUE = 'parsing'
# EMBED_QUEUE = 'embedding'
# FINAL_QUEUE = 'finalize'

# ——— Configuration & Constants ————————————————————————————————————————————————————

# Performance, Retries & Batching
MAX_RETRIES = 5
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
        user_id: str,
        note_type: str,
        project_id: str,
        note_title: str,
        provider: str,
        model_name: str,
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
        note_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🎯 Starting async note generation: {note_type} for project {project_id}")
            
            # 🆕 PARALLEL EXECUTION - Load prompt and generate embedding concurrently
            logger.info("⚡ Executing parallel tasks: prompt loading + embedding generation")
            
            prompt_task = asyncio.create_task(
                self._load_prompt_async(note_type)
            )
            embedding_task = asyncio.create_task(
                self._get_embedding_async(note_type)
            )
            llm_task = asyncio.create_task(
                self._setup_llm_client_async(provider, model_name, temperature)
            )
            
            # Wait for all parallel tasks
            prompt_yaml, embedding, llm_client = await asyncio.gather(
                prompt_task, embedding_task, llm_task
            )
            
            setup_time = time.time() - start_time
            logger.info(f"📊 Parallel setup completed in {setup_time*1000:.0f}ms")
            
            # 🆕 Async chunk retrieval (Naive RAG)
            retrieval_start = time.time()
            relevant_chunks = await self._fetch_relevant_chunks_async(
                embedding, project_id
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks in {retrieval_time*1000:.0f}ms")
            
            # 🆕 Build context and generate note
            generation_start = time.time()
            context = self._build_note_context(
                prompt_yaml, 
                relevant_chunks, 
                note_type, 
                addtl_params
            )
            
            # Generate note content
            note_content = await self._generate_note_content_async(
                llm_client, context, provider
            )
            generation_time = time.time() - generation_start
            
            # 🆕 Async note persistence: Nomal Notes (one block) vs Flashcards, Cold Calls (discrete-blocks)
            save_start = time.time()
            if note_type == "flashcards"  or note_type == "cold_call":  
                # Special flashcard processing and storage
                deck_id, num_cards = await self._save_flashcard_deck_and_cards_async(
                    project_id=project_id,
                    user_id=user_id,
                    deck_name=note_title,
                    llm_output=note_content,
                    is_essential=(addtl_params or {}).get('is_essential', False),
                )
                logger.info(f"🃏 Created flashcard deck {deck_id} with {num_cards} cards")
                
                # Store success metrics for flashcards
                save_metrics = {
                    "deck_id": str(deck_id),        # <-- Fix json.dump issue 
                    "num_cards": num_cards,
                    "storage_type": "flashcards"
                }
            elif note_type=="quiz":
                # Special quiz processing and storage
                num_questions_requested = (addtl_params or {}).get('num_questions', 15)
                
                quiz_note_id, num_questions_saved = await self._save_quiz_and_questions_async(
                    project_id=project_id,
                    user_id=user_id,
                    quiz_title=note_title,
                    llm_output=note_content,
                    num_questions_requested=num_questions_requested,
                    is_essential=(addtl_params or {}).get('is_essential', False),
                )
                save_metrics = {
                    "quiz_note_id": str(quiz_note_id),
                    "num_questions": num_questions_saved,
                    "storage_type": "quiz"
                }
            else:
                # Regular note types - save to notes table
                await self._save_note_async(
                    project_id, 
                    user_id, 
                    note_type, 
                    note_title, 
                    note_content,
                    is_essential=(addtl_params or {}).get('is_essential', False),
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
            await self._handle_note_error(project_id, user_id, note_type, str(e))
            raise
        finally:
            # Clean up large objects
            try:
                del relevant_chunks, note_content, context
            except NameError:
                pass
            gc.collect()

    async def lightrag_note_generation_async(
        self,
        user_id: str,
        note_type: str,
        project_id: str,
        note_title: str,
        provider: str,
        model_name: str,
        temperature: float = 0.7,
        addtl_params: Optional[Dict] = None,
    ) -> str:
        """
        Enhanced version of your note generation that uses LightRAG
        
        This would supplement your existing generate_note_async function in note_tasks.py
        """
        
        if addtl_params is None:
            addtl_params = {}
        
        try:
            logger.info(f"🎯 Starting enhanced note generation: {note_type} for project {project_id}")
            
            # ——— 1. Load Prompt (Your Existing Logic) ————————————————————————————————
            from note_tasks import NOTE_TYPE_YAML_MAP, load_yaml_prompt, build_prompt_template_from_yaml
            
            yaml_file = NOTE_TYPE_YAML_MAP.get(note_type)
            yaml_dict = load_yaml_prompt(yaml_file)
            base_query = yaml_dict.get("base_prompt")
            prompt_template = build_prompt_template_from_yaml(yaml_dict)
            
            # ——— 2. Enhanced Retrieval with LightRAG ——————————————————————————————————
            logger.info("🔍 Using enhanced LightRAG retrieval...")
            
            try:
                # Try LightRAG first
                enhanced_context = await lightrag_integration.enhance_note_generation(
                    query=base_query,
                    project_id=project_id,
                    note_type=note_type,
                    retrieval_mode="hybrid"  # Use hybrid mode for best results
                )
                
                logger.info(f"✅ LightRAG retrieval successful - context: {len(enhanced_context)} chars")
                
            except Exception as e:
                logger.warning(f"⚠️ LightRAG retrieval failed, falling back to vector search: {e}")
                
                # Fallback to your existing vector search
                from note_tasks import _fetch_relevant_chunks_async, _get_embedding_async
                
                embedding = await _get_embedding_async(note_type)
                relevant_chunks = await _fetch_relevant_chunks_async(embedding, project_id)
                
                # Build context the traditional way
                chunk_context = "\n\n".join(chunk["content"] for chunk in relevant_chunks)
                enhanced_context = f"=== Retrieved Context ===\n{chunk_context}"
            
            # ——— 3. Build Final Context ——————————————————————————————————————————————
            context = self._build_note_context(
                prompt_template, 
                enhanced_context, 
                note_type, 
                addtl_params
            )
            
            # ——— 4. Generate Note (Your Existing Logic) ———————————————————————————————
            from utils.llm_clients.llm_factory import LLMFactory
            
            llm_client = LLMFactory.get_client_for(provider, model_name, temperature, False)
            note_content = await asyncio.get_event_loop().run_in_executor(
                None, llm_client.chat, context
            )
            
            logger.info(f"✅ Enhanced note generation completed: {len(note_content)} chars")
            
            return note_content
            
        except Exception as e:
            logger.error(f"❌ Enhanced note generation failed: {e}")
            raise

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
        
        logger.info(f"📋 Loading prompt for note type: {note_type}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        yaml_dict = await loop.run_in_executor(
            None, load_yaml_prompt, yaml_file
        )
        
        # No longer need to extract parts here, just return the whole thing
        logger.info(f"✅ YAML prompt config loaded successfully for {note_type}")
        return yaml_dict

    async def _get_embedding_async(self, note_type: str) -> List[float]:
        """🆕 Async 'note prompt' embedding generation with caching"""
        
        # Use note_type as cache key (embeddings are similar for same note types)
        if note_type in self._embedding_cache:
            logger.info(f"🎯 Cache HIT for {note_type} embedding")
            return self._embedding_cache[note_type]
        
        # Load prompt to get base query
        yaml_file = NOTE_TYPE_YAML_MAP.get(note_type)
        yaml_dict = load_yaml_prompt(yaml_file)
        base_query = yaml_dict.get("base_prompt")
        
        logger.info(f"🤖 Generating embedding for {note_type}")
        
        # Generate embedding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._generate_embedding_sync, base_query
        )
        
        # Cache for future use
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

    async def _setup_llm_client_async(self, provider: str, model_name: str, temperature: float):
        """🆕 Async LLM client setup"""
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(
            None, 
            LLMFactory.get_client_for,
            provider, model_name, temperature, False  # streaming=False for notes
        )
        logger.info(f"🤖 LLM client setup: {provider}/{model_name}")
        return client

    async def _fetch_relevant_chunks_async(
        self, embedding: List[float], project_id: str, k: int = 10
    ) -> List[Dict]:
        """🆕 Async chunk retrieval with connection pooling"""
        
        # Convert to pgvector format
        vector_str = '[' + ','.join(map(str, embedding)) + ']'
        
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM match_document_chunks_hnsw($1, $2, $3)",
                project_id, vector_str, k
            )
        
        chunks = [dict(row) for row in rows]
        logger.info(f"🎯 Retrieved {len(chunks)} relevant chunks")
        return chunks

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
            # Build page number dictionary (for note citations etc...)
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
            

            # Format yaml prompt dictionary
            base_prompt = prompt_yaml.get("base_prompt", "")        # directly accesses the yaml dict
            prompt_template = build_prompt_template_from_yaml(prompt_yaml)      # fetched the full dict (so unique keys are included "example_issue_spotter"...)
            
            # Handle note-type specific YAML parameters
            if note_type == "exam_questions":
                example = prompt_yaml.get("example_issue_spotter", "")
                num_questions = addtl_params.get("num_questions", 15)
                
                final_request = prompt_template.format(
                    context=chunk_context, 
                    n_questions=num_questions
                )
                # Assemble with the special example
                context = f"{base_prompt}\n\n## GOLD-STANDARD EXAMPLE\n{example}\n\n## YOUR TASK\n{final_request}"

            elif note_type in ["flashcards", "cold_call"]:
                # Use "num_cards" as the consistent parameter key
                num_cards = addtl_params.get("num_cards", 10)
                
                # Curly brace "filling in" vars
                formatted_template = prompt_template.format(
                    context=chunk_context, 
                    num_cards=num_cards
                )
                # CORRECTLY assemble the prompt using the base_prompt
                context = f"{base_prompt}\n\n{formatted_template}"
                
            else:
                # For other note types, still combine base_prompt and template
                # Curly brace "filling in" vars
                formatted_template = prompt_template.format(context=chunk_context)
                context = f"{base_prompt}\n\n{formatted_template}"
            
            logger.info(f"📝 Built context: {len(context)} characters")
            return context

    async def _generate_note_content_async(
        self, llm_client, context: str, provider: str
    ) -> str:
        """🆕 Async note content generation"""
        
        logger.info(f"🧠 Generating note content with {provider}")
        
        # Run LLM generation in thread pool
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, llm_client.chat, context
        )
        
        logger.info(f"✅ Generated {len(content)} characters of content")
        return content

    async def _save_note_async(
        self, 
        project_id: str, 
        user_id: str, 
        note_type: str, 
        note_title: str, 
        content: str,
        is_essential: bool
    ):
        """🆕 Async note persistence with connection pooling"""
        
        logger.info(f"💾 Saving {note_type} note to database")
        
        async with get_db_connection() as conn:
            await conn.execute(
                """
                INSERT INTO notes (
                    id, user_id, project_id, title, content_markdown, 
                    note_type, is_generated, is_shareable, created_at, is_essential
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                str(uuid.uuid4()), user_id, project_id, note_title, content,
                note_type, True, False, datetime.now(timezone.utc), is_essential
            )
        
        logger.info(f"✅ Note saved successfully")

    async def _save_flashcard_deck_and_cards_async(
        self, 
        project_id: str, 
        user_id: str, 
        deck_name: str, 
        llm_output: str,
        is_essential: bool
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
                    # Insert deck record first
                    deck_id = await conn.fetchval(
                        """
                        INSERT INTO notes (
                            id, user_id, project_id, title, note_type, description, 
                            num_cards, created_at, is_active, is_essential
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                        """,
                        str(uuid.uuid4()), user_id, project_id, 
                        deck_data['deck_name'], "flashcards", deck_data['description'],
                        deck_data['num_cards'], deck_data['created_at'],
                        deck_data.get('is_active', True), is_essential
                    )
                    
                    if not deck_id:
                        raise Exception("Failed to insert flashcard deck")
                    
                    logger.info(f"✅ Inserted flashcard deck with ID: {deck_id}")
                    
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
        project_id: str,
        user_id: str,
        quiz_title: str,
        llm_output: str,
        num_questions_requested: int,
        is_essential: bool
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
            logger.info(f"🧠 Processing quiz: {quiz_title}")

            # 1. Initialize processor and parse/validate content
            # This part is synchronous but fast (no I/O)
            processor = QuizProcessor()
            quiz_data = processor.parse_quiz_content(llm_output)
            processor.validate_quiz_data(quiz_data)

            questions_list = quiz_data.get("questions", [])
            num_questions_saved = len(questions_list)

            if num_questions_saved == 0:
                raise ValueError("No questions found in parsed quiz data.")

            logger.info(f"💾 Saving {num_questions_saved} quiz questions to DB for: {quiz_title}")

            # 2. Use async database connection for all DB operations
            async with get_db_connection() as conn:
                # Start a single transaction for all inserts
                async with conn.transaction():
                    
                    # 3. Insert the main 'notes' record (type 'quiz')
                    quiz_note_id = await conn.fetchval(
                        """
                        INSERT INTO notes (
                            id, user_id, project_id, title, note_type,
                            num_questions, created_at, is_generated, is_essential
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                        str(uuid.uuid4()), user_id, project_id,
                        quiz_title, "quiz", num_questions_saved,
                        datetime.now(timezone.utc), True, is_essential
                    )

                    if not quiz_note_id:
                        raise Exception("Failed to insert quiz 'notes' record.")

                    logger.info(f"✅ Inserted quiz note record with ID: {quiz_note_id}")

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
                            question_data['questionText'], question_data['hint'],
                            datetime.now(timezone.utc)
                        )

                        if not question_db_id:
                            raise Exception(f"Failed to insert question: {question_data['questionText']}")

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

    async def _handle_note_error(
        self, project_id: str, user_id: str, note_type: str, error_message: str
    ):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: log_llm_error(
                    supabase_client,
                    table_name="notes",
                    task_name="async_note_generation",
                    error_message=error_message,
                    project_id=project_id,
                    user_id=user_id,
                    note_type=note_type,
                ),
            )
        except Exception as e:
            logger.warning(f"⚠️ Error logging failed: {e}")

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
    temperature: float = 0.7,
    addtl_params: Optional[Dict] = None,
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
                user_id=user_id,
                note_type=note_type,
                project_id=project_id,
                note_title=note_title,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                addtl_params=addtl_params,
            )
        )
        
        logger.info(f"✅ Note task {task_id} completed successfully")
        return "RAG Note Task success"

    except Exception as e:
        logger.error(f"❌ Note task failed: {e}", exc_info=True)
        
        # Log error synchronously
        log_llm_error(
            client=supabase_client,
            table_name="notes",
            task_name="rag_note_task",
            error_message=str(e),
            project_id=project_id,
            user_id=user_id,
        )
        
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            raise RuntimeError(
                f"Note creation failed permanently after {self.max_retries} retries: {e}"
            ) from e
    
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
