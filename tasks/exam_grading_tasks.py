# tasks/exam_grading_tasks.py

"""
Async Exam Grading Pipeline
Orchestrates Ingest -> Rubric Generation -> RAG Grading -> Ideal Answer -> Feedback
"""

import asyncio
import json
import logging
import uuid
import time
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from celery import Task
from celery.utils.log import get_task_logger

# ===== PROJECT MODULES =====
from tasks.celery_app import celery_app, run_async_in_worker
from tasks.database import get_db_connection, get_global_async_db_pool, init_async_pools
from tasks.upload_tasks import _process_document_async_workflow
from tasks.note_tasks import note_manager # Reuse embedding/chunk fetching logic

from utils.llm_clients.llm_factory import LLMFactory
from utils.prompt_utils import load_yaml_prompt
from utils.supabase_utils import log_llm_error, supabase_client
from utils.grade_utils import calculate_letter_grade
from utils.exam_processor.exam_grading_processor import ExamGradingProcessor

# ——— CONFIG ———————————————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
MAIN_PROMPT_FILE = "exam-feedback.yaml"
# The loader already assumes we are in the 'prompts' directory
RUBRIC_DIR = "rubrics/"

# Standard limit for JSON tasks (Rubric, Feedback)
DEFAULT_MAX_TOKENS = 4096 

class ExamGradingStatus(Enum):
    INITIALIZED = "INITIALIZED"
    PROCESSING_OUTLINE = "PROCESSING_OUTLINE"
    ANALYZING_QUESTION = "ANALYZING_QUESTION"
    GENERATING_SOLUTION = "GENERATING_SOLUTION"
    GRADING_AND_FEEDBACK = "GRADING_AND_FEEDBACK"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

class AsyncExamGradingManager:
    """
    Manages the sequential but async steps of grading a law exam.
    Uses persistent event loop for non-blocking I/O.
    """
    
    def __init__(self):
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await init_async_pools()
            self._initialized = True

    def _get_dynamic_max_tokens(self, model_name: str) -> int:
        """
        Determines the safe amx output token limit based on model capabilities.
        e.g. gemini-3-pro=32k, gpt-4o=16k
        """
        name = model_name.lower()
        
        if "gpt-5" in name:
            return 128000
        if "gemini-3" in name: # Pro and Flash Previews
            return 32000
        if "gpt-4o" in name:
            return 16384
        if "claude" in name and "sonnet" in name:
            return 8192 # Standard API max for Sonnet 3.5
        
        return 8192 # Safe high default for other models
    
    async def run_grading_workflow(
        self,
        user_id: str,
        project_id: str,
        question: str,
        user_answer: str,
        question_type: str, 
        professor_example: Optional[str],
        outline_url: Optional[str],
        model_name: str,
        model_provider: str,
    ):
        """Main Orchestrator for the Grading Pipeline"""
        
        await self.initialize()
        grading_id = str(uuid.uuid4())
        start_time = time.time()

        # Check question_type formatting
        if question_type == "fact_pattern":
            yaml_question_type = "fact-pattern"
        
        try:
            # 1. LOAD PROMPTS (Main + Specific Rubric)
            main_prompts = await self._load_yaml(MAIN_PROMPT_FILE)
            rubric_prompts = await self._load_rubric(yaml_question_type)

            # 2. INITIALIZE DB RECORD
            # We persist professor_example immediately so it is safe
            await self._update_status(grading_id, ExamGradingStatus.INITIALIZED, {
                "user_id": user_id, 
                "project_id": project_id,
                "question": question, 
                "user_answer": user_answer,
                "professor_example": professor_example,
                "question_type": question_type,
                "created_at": datetime.now(timezone.utc)
            }, create_new=True)

            # 3. OPTIONAL: PROCESS OUTLINE (INGESTION)
            outline_doc_id = None
            if outline_url:
                await self._update_status(grading_id, ExamGradingStatus.PROCESSING_OUTLINE)
                outline_doc_id = await self._ingest_outline(
                    user_id, 
                    project_id, 
                    outline_url, 
                    grading_id
                )
            
            # 4. FETCH RAG CONTEXT (If Outline Exists)
            rag_context = ""
            if outline_doc_id:
                # Reuse embedding logic from note_manager to get context for the question
                embedding = await note_manager._get_embedding_async("outline") # Generic lookup prompt
                # Fetch chunks scoped to this project
                chunks = await note_manager._fetch_relevant_chunks_async(embedding, project_id, k=7)
                rag_context = "\n\n".join([c['content'] for c in chunks])

            # 5. ANALYZE QUESTION & GENERATE RUBRIC
            await self._update_status(grading_id, ExamGradingStatus.ANALYZING_QUESTION)
            rubric_data = await self._analyze_question(
                question, 
                model_name, 
                model_provider, 
                main_prompts
            )
            
            # 6. GENERATE IDEAL ANSWER
            # Independent of student answer, useful for "Compare/Contrast"
            await self._update_status(grading_id, ExamGradingStatus.GENERATING_SOLUTION)
            ideal_answer = await self._generate_ideal_answer(
                question, 
                rubric_data['rubric_markdown'], 
                model_name, 
                model_provider, 
                main_prompts
            )
            
            # Save intermediate results
            async with get_db_connection() as conn:
                await conn.execute(
                    """UPDATE exam_grading SET 
                       course_type = $1, grading_rubric = $2, llm_ideal_answer = $3
                       WHERE id = $4""",
                    rubric_data['course_type'], rubric_data['rubric_markdown'], ideal_answer, grading_id
                )

            # 7. GRADING & GRANULAR FEEDBACK
            # 🤓 This is where we use the ExamGradingProcessor to ensure valid JSON
            await self._update_status(grading_id, ExamGradingStatus.GRADING_AND_FEEDBACK)
            
            feedback_data = await self._generate_granular_feedback(
                question=question,
                user_answer=user_answer,
                professor_example=professor_example, # Pass the benchmark
                ideal_answer=ideal_answer,
                rag_context=rag_context,
                course_type=rubric_data['course_type'],
                rubric_prompts=rubric_prompts, # Contains specific formatting rules
                main_prompts=main_prompts,
                model_name=model_name,
                model_provider=model_provider
            )

            # 8. DETERMINISTIC LETTER GRADE
            percentage = feedback_data.get('exam_grade_percentage', 0)
            letter_grade = calculate_letter_grade(percentage)

            # 9. FINALIZING & SAVING
            await self._update_status(grading_id, ExamGradingStatus.FINALIZING)
            
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE exam_grading SET 
                        exam_grade_percentage = $1,
                        exam_grade_letter = $2,
                        feedback_overview = $3,
                        feedback_breakdown = $4,
                        feedback_strength = $5,
                        feedback_weakness = $6,
                        grading_progress = $7,
                        updated_at = NOW()
                    WHERE id = $8
                    """,
                    percentage,
                    letter_grade,
                    feedback_data.get('feedback_overview'),
                    feedback_data.get('feedback_breakdown'),
                    feedback_data.get('feedback_strength'),
                    feedback_data.get('feedback_weakness'),
                    ExamGradingStatus.COMPLETE.value,
                    grading_id
                )
            
            logger.info(f"✅ Exam Grading {grading_id} Complete. Score: {percentage}% ({letter_grade}) in {time.time() - start_time:.2f}s")
            return grading_id

        except Exception as e:
            logger.error(f"❌ Exam Grading Failed: {e}", exc_info=True)
            await self._update_status(grading_id, ExamGradingStatus.ERROR, {"error_msg": str(e)})
            raise

    # ——— Helper Methods: Workflow Logic ——————————————————————————————————————

    async def _ingest_outline(self, user_id: str, project_id: str, cdn_url: str, grading_id: str) -> str:
        """
        Purpose: This block handles the optional ingestion of a specific 
        course outline (like a syllabus or class notes PDF) that the user might have provided.

        Funtions:
        - Reuses upload_tasks logic to ingest outline.
        - Updates exam_grading table with doc ID (The Linkage Strategy).
        """
        logger.info("📄 Ingesting Outline for Exam Grading...")
        
        # Mocking doc_data required for wrapper
        doc_data = {
            "filename": "Exam_Outline_Reference.pdf",
            "cdn_url": cdn_url,
            "content_hash": str(uuid.uuid4()), # Force processing
            "file_size_bytes": 0, 
        }
        
        workflow_metadata = {
            "user_id": user_id,
            "project_id": project_id,
            "is_essential": False
        }

        # Reuse existing async workflow from upload_tasks
        result = await _process_document_async_workflow(
            str(uuid.uuid4()), doc_data, project_id, workflow_metadata
        )
        
        if result['status'] == 'FAILED':
            raise Exception(f"Outline ingestion failed: {result.get('error')}")

        doc_id = result['doc_id']

        # Link this document to the exam_grading table
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE exam_grading 
                SET outline = $1, outline_document_id = $2
                WHERE id = $3
                """,
                cdn_url, uuid.UUID(doc_id), grading_id
            )
            
        logger.info(f"✅ Outline ingested and linked: {doc_id}")
        return doc_id

    # ——— Helper Methods: LLM Interactions ————————————————————————————————————

    async def _analyze_question(
        self, 
        question: str, 
        model_name: str, 
        model_provider: str, 
        prompts: str, 
    ) -> Dict:
        """Step 5: Infer Course Type"""
        llm = LLMFactory.get_client_for(
            provider=model_provider, 
            model_name=model_name, 
            temperature=0.2, 
            streaming=False,
            max_output_tokens=DEFAULT_MAX_TOKENS
        )
        prompt = f"{prompts['system_prompt']}\n\n{prompts['rubric_generation_prompt']}".format(
            question=question
        )
        response = await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)
        return self._parse_json_simple(response)

    async def _generate_ideal_answer(
        self, 
        question: str,  
        rubric: str, 
        model_name: str, 
        model_provider: str,  
        prompts: str, 
    ) -> str:
        
        # Determine max tokens based on the specific model
        dynamic_max = self._get_dynamic_max_tokens(model_name)
        logger.info(f"⚖️ Using limit of {dynamic_max} tokens for Ideal Answer generation ({model_name})")

        """Step 6: Generate Model 'Legalnote' Answer"""
        llm = LLMFactory.get_client_for(
            provider=model_provider, 
            model_name=model_name, 
            temperature=0.3, 
            streaming=False,
            max_output_tokens=dynamic_max
        )
        prompt = prompts['ideal_answer_prompt'].format(
            question=question,
            rubric=rubric
        )
        return await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)

    async def _generate_granular_feedback(
        self, question, user_answer, professor_example, ideal_answer, rag_context, 
        course_type, rubric_prompts, main_prompts, model_name, model_provider
    ) -> Dict:
        """
        Step 7: The Big One... ALL FEEDBACK
        Generates score (89/A-) and all feedback sections in one JSON object.
        """
        llm = LLMFactory.get_client_for(
            provider=model_provider, 
            model_name=model_name, 
            temperature=0.2, 
            streaming=False,
            max_output_tokens=DEFAULT_MAX_TOKENS
        )
        
        if professor_example:
            # Case A: Professor Example Provided
            final_prompt = main_prompts['grading_feedback_prompt_with_professor'].format(
                question=question,
                course_type=course_type,
                user_answer=user_answer,
                professor_example=professor_example, # <--- Used here
                ideal_answer=ideal_answer,
                rag_context=rag_context or "No specific outline provided.",
                grading_criteria=rubric_prompts['grading_criteria'],
                breakdown_format_instruction=rubric_prompts['breakdown_format_instruction']
            )
        else:
            # Case B: No Professor Example (Use Base Prompt)
            final_prompt = main_prompts['grading_feedback_prompt_base'].format(
                question=question,
                course_type=course_type,
                user_answer=user_answer,
                # professor_example is omitted from format arguments as the prompt won't have the placeholder
                ideal_answer=ideal_answer,
                rag_context=rag_context or "No specific outline provided.",
                grading_criteria=rubric_prompts['grading_criteria'],
                breakdown_format_instruction=rubric_prompts['breakdown_format_instruction']
            )
        
        # Call LLM
        raw_response = await asyncio.get_event_loop().run_in_executor(None, llm.chat, final_prompt)
        
        # Process and Validate Response using Robust Processor
        return ExamGradingProcessor.process_and_validate_json(raw_response)

    # ——— Utilities —————————————————————————————————————————————————————————

    def _parse_json_simple(self, response_text: str) -> Dict:
        """Simple parser for the non-critical rubric generation step"""
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError:
            # Fallback if LLM didn't output strict JSON for rubric
            return {
                "course_type": "General Law",
                "rubric_markdown": response_text
            }

    async def _load_yaml(self, filepath: str) -> Dict:
        return await asyncio.get_event_loop().run_in_executor(
            None, load_yaml_prompt, filepath
        )
        
    async def _load_rubric(self, question_type: str) -> Dict:
        """Loads specific rubric or defaults to fact-pattern"""
        rubric_file = f"{RUBRIC_DIR}{question_type}.yaml"
        try:
            return await self._load_yaml(rubric_file)
        except Exception:
            logger.warning(f"Rubric {question_type} not found, defaulting to fact-pattern")
            return await self._load_yaml(f"{RUBRIC_DIR}fact-pattern.yaml")

    async def _update_status(self, grading_id, status, extra_data=None, create_new=False):
        """Updates status and handles the Professor Example insert on creation"""
        async with get_db_connection() as conn:
            if create_new:
                await conn.execute(
                    """
                    INSERT INTO exam_grading (
                        id, user_id, project_id, question, user_answer, 
                        professor_example, question_type, grading_progress, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    grading_id, 
                    uuid.UUID(extra_data['user_id']), 
                    uuid.UUID(extra_data['project_id']), 
                    extra_data['question'], 
                    extra_data['user_answer'], 
                    extra_data.get('professor_example'), 
                    extra_data['question_type'], 
                    status.value, 
                    extra_data['created_at']
                )
            elif status == ExamGradingStatus.ERROR:
                await conn.execute(
                    "UPDATE exam_grading SET grading_progress = $1, error_msg = $2 WHERE id = $3",
                    status.value, extra_data.get("error_msg"), grading_id
                )
            else:
                await conn.execute(
                    "UPDATE exam_grading SET grading_progress = $1 WHERE id = $2",
                    status.value, grading_id
                )

# ——— Global Instance —————————————————————————————————————————————————————

exam_manager = AsyncExamGradingManager()

# ——— Celery Task —————————————————————————————————————————————————————————

@celery_app.task(bind=True, queue='notes', acks_late=True)
def grade_exam_question_workflow(
    self,
    user_id: str,
    project_id: Optional[str],
    question: str,
    user_answer: str,
    question_type: str = "fact_pattern",
    professor_example: Optional[str] = None,
    outline_url: Optional[str] = None,
    model_name: str = "gpt-4o",
    model_provider: str = "opwnai"
):
    """
    Celery task wrapper for the async grading workflow.
    Executes in the persistent worker loop to avoid event loop conflicts.
    """

# Generate a new UUID if project_id is missing
    if project_id is None: 
        # uuid4() creates a random UUID, standard for Supabase/Postgres Primary Keys
        project_id = str(uuid.uuid4())

    try:
        self.update_state(state="STARTED")
        
        grading_id = run_async_in_worker(
            exam_manager.run_grading_workflow(
                user_id=user_id,
                project_id=project_id,
                question=question,
                user_answer=user_answer,
                question_type=question_type,
                professor_example=professor_example,
                outline_url=outline_url,
                model_name=model_name,
                model_provider=model_provider
            )
        )
        
        return {"grading_id": grading_id, "status": "SUCCESS"}

    except Exception as e:
        logger.error(f"Exam grading task failed: {e}")
        log_llm_error(
            supabase_client, 
            table_name="exam_grading", 
            task_name="grade_exam_question_workflow",
            error_message=str(e),
            project_id=project_id,
            user_id=user_id,
            record_id=grading_id # <--- Pass the UUID here!
        )
        raise