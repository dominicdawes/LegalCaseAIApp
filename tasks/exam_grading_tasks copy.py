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

# ——— Configuration ————————————————————————————————————————————————————————————
logger = get_task_logger(__name__)
EXAM_PROMPT_FILE = "exam-feedback.yaml"

class ExamGradingStatus(Enum):
    INITIALIZED = "INITIALIZED"
    PROCESSING_OUTLINE = "PROCESSING_OUTLINE" # Ingesting PDF
    ANALYZING_QUESTION = "ANALYZING_QUESTION" # Inferring Law type & Rubric
    GRADING_RESPONSE = "GRADING_RESPONSE"     # Assigning Letter Grade
    GENERATING_SOLUTION = "GENERATING_SOLUTION" # Writing Ideal Answer
    FINALIZING = "FINALIZING"                 # Writing Feedback
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

class AsyncExamGradingManager:
    """
    Manages the sequential but async steps of grading a law exam.
    """
    
    def __init__(self):
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await init_async_pools()
            self._initialized = True

    async def run_grading_workflow(
        self,
        user_id: str,
        project_id: str,
        question: str,
        user_answer: str,
        professor_example: Optional[str],
        outline_url: Optional[str],
        model_name: str
    ):
        """Main Orchestrator for the Grading Pipeline"""
        
        await self.initialize()
        grading_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Load Prompts
        prompts = await self._load_prompts()
        
        try:
            # 1. INITIALIZE DB RECORD
            await self._update_status(grading_id, ExamGradingStatus.INITIALIZED, {
                "user_id": user_id,
                "project_id": project_id,
                "question": question,
                "user_answer": user_answer,
                "created_at": datetime.now(timezone.utc)
            }, create_new=True)

            # 2. OPTIONAL: PROCESS OUTLINE (INGESTION)
            outline_doc_id = None
            if outline_url:
                await self._update_status(grading_id, ExamGradingStatus.PROCESSING_OUTLINE)
                outline_doc_id = await self._ingest_outline(
                    user_id, project_id, outline_url, grading_id
                )
            
            # 3. ANALYZE QUESTION & GENERATE RUBRIC
            await self._update_status(grading_id, ExamGradingStatus.ANALYZING_QUESTION)
            rubric_data = await self._generate_rubric(
                question, model_name, prompts
            )
            
            # Update DB with Rubric & Course Type
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE exam_grading 
                    SET grading_rubric = $1, course_type = $2
                    WHERE id = $3
                    """,
                    rubric_data['rubric_markdown'], 
                    rubric_data['course_type'], 
                    grading_id
                )

            # 4. FETCH RAG CONTEXT (If Outline Exists)
            rag_context = ""
            if outline_doc_id:
                # We reuse the embedding/search logic from NoteManager
                # We first need the embedding for the QUESTION to find relevant parts of the OUTLINE
                embedding = await note_manager._get_embedding_async("outline") # Reuse generic outline embedding prompt
                
                # Perform Vector Search (scoped to project_id, which the outline is now attached to)
                # We specifically look for chunks from the document we just uploaded if possible, 
                # or just general project context. Given the prompt, let's look at project context.
                chunks = await note_manager._fetch_relevant_chunks_async(embedding, project_id, k=7)
                rag_context = "\n\n".join([c['content'] for c in chunks])

            # 5. GRADE RESPONSE
            await self._update_status(grading_id, ExamGradingStatus.GRADING_RESPONSE)
            letter_grade = await self._grade_student_answer(
                question, user_answer, rubric_data['rubric_markdown'], 
                professor_example, rag_context, model_name, prompts
            )
            
            async with get_db_connection() as conn:
                await conn.execute(
                    "UPDATE exam_grading SET letter_grade = $1 WHERE id = $2",
                    letter_grade, grading_id
                )

            # 6. GENERATE IDEAL ANSWER
            await self._update_status(grading_id, ExamGradingStatus.GENERATING_SOLUTION)
            ideal_answer = await self._generate_ideal_answer(
                question, rubric_data['rubric_markdown'], model_name, prompts
            )
            
            async with get_db_connection() as conn:
                await conn.execute(
                    "UPDATE exam_grading SET llm_ideal_answer = $1 WHERE id = $2",
                    ideal_answer, grading_id
                )

            # 7. GENERATE FEEDBACK (COMPARE & CONTRAST)
            await self._update_status(grading_id, ExamGradingStatus.FINALIZING)
            feedback = await self._generate_feedback(
                user_answer, ideal_answer, model_name, prompts
            )

            # 8. COMPLETE
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE exam_grading 
                    SET feedback = $1, grading_progress = $2, updated_at = NOW()
                    WHERE id = $3
                    """,
                    feedback, ExamGradingStatus.COMPLETE.value, grading_id
                )
            
            logger.info(f"✅ Exam Grading {grading_id} Complete in {time.time() - start_time:.2f}s")
            return grading_id

        except Exception as e:
            logger.error(f"❌ Exam Grading Failed: {e}", exc_info=True)
            await self._update_status(grading_id, ExamGradingStatus.ERROR, {"error_msg": str(e)})
            raise

    # ——— Helper Methods ——————————————————————————————————————————————————————

    async def _update_status(self, grading_id: str, status: ExamGradingStatus, extra_data: Dict = None, create_new: bool = False):
        """Updates the status enum and optional columns in postgres"""
        async with get_db_connection() as conn:
            if create_new:
                # Initial Insert
                await conn.execute(
                    """
                    INSERT INTO exam_grading (
                        id, user_id, project_id, question, user_answer, 
                        grading_progress, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    grading_id, 
                    uuid.UUID(extra_data['user_id']), 
                    uuid.UUID(extra_data['project_id']), 
                    extra_data['question'], 
                    extra_data['user_answer'], 
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

    async def _ingest_outline(self, user_id: str, project_id: str, cdn_url: str, grading_id: str) -> str:
        """
        Reuses upload_tasks logic to ingest outline.
        Updates exam_grading table with doc ID.
        """
        logger.info("📄 Ingesting Outline for Exam Grading...")
        
        # Mocking doc_data required for wrapper
        doc_data = {
            "filename": "Exam_Outline_Reference.pdf",
            "cdn_url": cdn_url,
            "content_hash": str(uuid.uuid4()), # We force processing since we don't calculate hash here
            "file_size_bytes": 0, 
        }
        
        workflow_metadata = {
            "user_id": user_id,
            "project_id": project_id,
            "is_essential": False # Standard processing
        }

        # Reuse existing async workflow from upload_tasks
        # This inserts into document_sources & document_vector_store
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

    async def _generate_rubric(self, question: str, model_name: str, prompts: Dict) -> Dict:
        """Step 3: Infer Course & Generate Rubric"""
        llm = LLMFactory.get_client_for("openai", model_name, temperature=0.2, streaming=False)
        
        prompt = f"{prompts['system_prompt']}\n\n{prompts['rubric_generation_prompt']}".format(
            question=question
        )
        
        response = await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)
        
        # Clean markdown wrappers if present to get raw JSON
        clean_json = response.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_json)
            return {
                "course_type": data.get("course_type", "General Law"),
                "rubric_markdown": data.get("rubric_markdown", response) # Fallback
            }
        except json.JSONDecodeError:
             # Fallback if LLM didn't output strict JSON
            return {
                "course_type": "Unspecified Law",
                "rubric_markdown": response
            }

    async def _grade_student_answer(self, question, answer, rubric, prof_example, context, model_name, prompts) -> str:
        """Step 5: Assign Letter Grade"""
        llm = LLMFactory.get_client_for("openai", model_name, temperature=0.0, streaming=False)
        
        prompt = prompts['grading_prompt'].format(
            context_chunks=context or "No external outline provided.",
            question=question,
            rubric=rubric,
            professor_example=prof_example or "N/A",
            user_answer=answer
        )
        
        return await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)

    async def _generate_ideal_answer(self, question, rubric, model_name, prompts) -> str:
        """Step 6: Generate Model Answer"""
        llm = LLMFactory.get_client_for("openai", model_name, temperature=0.3, streaming=False)
        
        prompt = prompts['ideal_answer_prompt'].format(
            question=question,
            rubric=rubric
        )
        
        return await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)

    async def _generate_feedback(self, user_answer, ideal_answer, model_name, prompts) -> str:
        """Step 7: Compare & Contrast Feedback"""
        llm = LLMFactory.get_client_for("openai", model_name, temperature=0.5, streaming=False)
        
        prompt = prompts['feedback_prompt'].format(
            user_answer=user_answer,
            ideal_answer=ideal_answer
        )
        
        return await asyncio.get_event_loop().run_in_executor(None, llm.chat, prompt)

    async def _load_prompts(self) -> Dict:
        """Loads YAML prompts"""
        return await asyncio.get_event_loop().run_in_executor(
            None, load_yaml_prompt, EXAM_PROMPT_FILE
        )

# ——— Global Instance —————————————————————————————————————————————————————————

exam_manager = AsyncExamGradingManager()

# ——— Celery Task —————————————————————————————————————————————————————————————

@celery_app.task(bind=True, queue='notes', acks_late=True)
def grade_exam_question_workflow(
    self,
    user_id: str,
    project_id: str,
    question: str,
    user_answer: str,
    professor_example: Optional[str] = None,
    outline_url: Optional[str] = None,
    model_name: str = "gpt-4o"
):
    """
    Celery task wrapper for the async grading workflow.
    """
    try:
        self.update_state(state="STARTED")
        
        # Execute async workflow in persistent worker loop
        grading_id = run_async_in_worker(
            exam_manager.run_grading_workflow(
                user_id=user_id,
                project_id=project_id,
                question=question,
                user_answer=user_answer,
                professor_example=professor_example,
                outline_url=outline_url,
                model_name=model_name
            )
        )
        
        return {"grading_id": grading_id, "status": "SUCCESS"}

    except Exception as e:
        logger.error(f"Exam grading task failed: {e}")
        # Log to Supabase error table
        log_llm_error(
            supabase_client, 
            table_name="exam_grading", 
            task_name="grade_exam_question_workflow",
            error_message=str(e),
            project_id=project_id,
            user_id=user_id
        )
        raise