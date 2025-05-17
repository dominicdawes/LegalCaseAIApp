'''
Main FastAPI script, this is the heart of the web service
'''

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Dict, Any, Literal
import uuid
from fastapi import FastAPI, HTTPException, Query
import logging
import redis.asyncio as aioredis 

# utils imports
from utils.supabase_utils import supabase_client
from utils.pdf_utils import extract_text_from_pdf

# Celery task imports
from celery import chain, chord, group, states
from celery.result import AsyncResult
from tasks.podcast_generate_tasks import validate_and_generate_audio_task, generate_dialogue_only_task
from tasks.upload_tasks import process_pdf_task
from tasks.test_tasks import addition_task
from tasks.chat_streaming_tasks import rag_chat_streaming_task
from tasks.chat_tasks import rag_chat_task, persist_user_query
from tasks.note_tasks import rag_note_task 

# Configure logging (basic example, adjust as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Init FastAPI app & Redis pub.sub
app = FastAPI()
REDIS_LABS_URL = os.getenv("REDIS_LABS_URL_AND_PASS")  # REDIS_LABS_URL
redis_pub = aioredis.from_url(REDIS_LABS_URL, decode_responses=True)
# celery_app = Celery('tasks', broker='redis://localhost:6379/0')

origins = [
    "https://app.weweb.io",  # Replace with the actual WeWeb domain if different
    "https://editor.weweb.io",
    # Add other domains as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== PYDANTIC MODELS TEST ======== #
class Numbers(BaseModel):
    x: int
    y: int
    
# Define Pydantic models for the responses
class SumResponse(BaseModel):
    sum: int

# [SANITY CHECK] Request model for addition sanity
class AdditionRequest(BaseModel):
    x: int
    y: int

class PDFCaptureResponse(BaseModel):
    uuid: str
    url: str

# ======== PYDANTIC MODELS PROD ======== #

# <---- Define Pydantic model for the PDF extraction response ----> #
class PDFExtractResponse(BaseModel):
    filename: str
    combined_text: str

class PDFExtractBatchResponse(BaseModel):
    results: List[PDFExtractResponse]

class PDFRequest(BaseModel):
    ''' WeWeb specific pydantic struct '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

# Pydantic model for the response
class PDFResponse(BaseModel):
    '''
    Response for the `` endpoint
    Format (seen in PostMan):
    {
        "embedding_task_id": "some_task_id"
    }
    '''
    audio_task_id: str
    embedding_task_id: str

# <---- Models for new RAG pipeline kickoff  ----> #
class NewRagPipelineRequest(BaseModel):
    ''' 
    Pydantic struct for `POST/new-rag-project/` to kick off of a new RAG project
    Files → S3/CF upload → Vector embedding
    '''
    files: List[str]  # List of URLs or file paths of the PDFs
    metadata: Dict[str, Any]  # A dictionary for any metadata information

class NewRagPipelineResponse(BaseModel):
    '''
    Response for the `POST/new-rag-project/` endpoint
    Format (seen in PostMan):
    {
        "embedding_task_id": "some_task_id"
    }
    '''
    embedding_task_id: str      # from vector embedding task

# <---- Models for new RAG pipeline nested Celery task staus checking ----> #
class EmbeddingStatusRequest(BaseModel):
    ''' Pydantic struct for `POST/embedding-pipeline-task-status` request checking on chunk & embed task for a new RAG project '''
    source_ids: List[str]  # List of uuids of documents being processed for RAG

class DocumentStatus(BaseModel):
    '''Helper struct for `POST/embedding-pipeline-task-status` endpoint '''
    id: str
    status: str

class EmbeddingStatusResponse(BaseModel):
    '''
    Response body for `POST/embedding-pipeline-task-status` status checker
    Format (seen in PostMan):
    {
        "status": "WORKFLOW_IN_PROGRESS",
        "message": "some message...",
        "document_statuses": ["COMPLETE", "PENDING"]
    }
    '''
    status: Literal["WORKFLOW_IN_PROGRESS","WORKFLOW_COMPLETE","WORKFLOW_ERROR"]
    message: str
    document_statuses: List[DocumentStatus]

# Chained RAG pipeline response model
class NewRagAndNoteResponse(BaseModel):
    workflow_id: str             # the chain’s overall task ID

class SourceIdsRequest(BaseModel):
    """Request body model for the status endpoint for checking the RAG pipeline upload."""
    source_ids: List[str]

# <---- Models for AI Chatbot assistant (with RAG) ----> #
class RagQueryRequest(BaseModel):  
    '''
    Pydantic request model for for `POST/rag-chat/` endpoiont
    '''
    user_id: str
    chat_session_id: str
    query: str
    project_id: str
    model_name: str

class RagQueryResponse(BaseModel):  
    '''
    Response body for `POST/rag-chat/` status checker
    Format (seen in PostMan):
    {
        "user_id",
        "chat_session_id",
        "query",
        "project_id",
        "model_name"

    }
    '''
    user_id: str
    chat_session_id: str
    query: str
    # document_ids: List[str]
    project_id: str
    model_name: str

class RagRegenerateRequest(BaseModel):  
    '''
    Pydantic request model for for `POST/rag-chat/regenerate` endpoiont
    Mainly uses the "FAILED" messages row.id to retry that query
    '''
    message_id: str        # ← public.messages.id
    chat_session_id: str
    project_id: str
    model_name: str

# ================================================ #
#                  WEBSOCKETS
# ================================================ #

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(session_id: str, websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_pub.pubsub()
    await pubsub.subscribe(f"chat_result:{session_id}")

    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_text(message['data'])
    except WebSocketDisconnect:
        await pubsub.unsubscribe(f"chat_result:{session_id}")


# ================================================ #
#               TEST ENDPOINTS
# ================================================ #

# Root endpoint
@app.get("/")
async def root():
    return {"success": "Hello Server LegalNoteAI FastAPI App"}

# Endpoint for sanity check 
@app.post("/sum_two_num/")
async def sum_two_num(numbers: AdditionRequest):
    print("execute 2 sum")
    task = addition_task.delay(numbers.x, numbers.y)
    print(f"Task submitted: {task.id}")
    return {"task_id": task.id}

# Endpoint for capturing PDF info (sanity check endpoint)
@app.get("/pdf_capture", response_model=PDFCaptureResponse)
async def pdf_capture(file: str):
    unique_id = str(uuid.uuid4())  # Generate a unique UUID
    return {"uuid": unique_id, "url": file}

# Endpoint for extracting a single PDF's content
@app.get("/pdf_extract", response_model=PDFExtractResponse)
async def pdf_extract(file: str):
    try:
        # Extract text using the utility function
        combined_text = extract_text_from_pdf(file)
        return {"filename": file.split("/")[-1], "combined_text": combined_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint for extracting content from multiple PDFs
@app.post("/pdf_extract_batch", response_model=PDFExtractBatchResponse)
async def pdf_extract_batch(files: List[str]):
    results = []
    
    for file_url in files:
        try:
            # Extract text using the utility function
            combined_text = extract_text_from_pdf(file_url)
            filename = file_url.split("/")[-1]
            results.append(PDFExtractResponse(filename=filename, combined_text=combined_text))
        
        except Exception as e:
            # Append an error message for the current file
            results.append(PDFExtractResponse(filename=file_url, combined_text=f"Error: {str(e)}"))
    
    return {"results": results}

# POST endpoint for addition using Celery
@app.post("/celery_test_addition/")
async def celery_test_addition(request: AdditionRequest):
    try:
        # Enqueue the Celery task
        print(f"API DEBUG: {[request.x, request.y]}")
        task = addition_task.apply_async(args=[request.x, request.y])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================ #
#                RAG CHAT ENDPOINTS
# ================================================ #
@app.post("/new-rag-project/", response_model=NewRagPipelineResponse)
async def create_new_rag_project(
    request: NewRagPipelineRequest, 
    background_tasks: BackgroundTasks
):
    '''
    Endpoint to create a RAG pipeline for user project:
        - Uploads PDFs to AWS S3 and Supabase
        - Initiates chunking and embedding tasks
        - Returns task_id and source_ids for status monitoring
        
    Request contains:
        request.files (List): list of pdf file links
        request.metadata (json): {
            user_id: UUID
            project_id: UUID
            model_name: string (optional) 
            note_type: string
        }
    '''
    try:
        # Log request information
        logger.info(f"Starting new RAG project with {len(request.files)} files for project {request.metadata.get('project_id')}")
        
        # Data validation
        if not request.files:
            raise HTTPException(status_code=400, detail="No files provided in request")
            
        if not request.metadata.get('project_id'):
            raise HTTPException(status_code=400, detail="project_id is required in metadata")
            
        if not request.metadata.get('user_id'):
            raise HTTPException(status_code=400, detail="user_id is required in metadata")
        
        # Apply async job to process PDFs
        job = process_pdf_task.apply_async(
            args=[request.files, request.metadata]
        )
        
        # The task will return source_ids when it completes initial DB insertion
        # But for now, return the job ID for immediate status monitoring
        logger.info(f"Started RAG project task with ID: {job.id}")
        
        return {
            "embedding_task_id": job.id,
            "message": f"Processing {len(request.files)} files, check status with job ID"
        }
    except Exception as e:
        logger.error(f"Error creating new RAG project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating RAG project: {str(e)}")
    
@app.post("/new-rag-and-notes/", response_model=NewRagAndNoteResponse)
async def create_new_rag_project_and_gen_notes(
    request: NewRagPipelineRequest, 
    background_tasks: BackgroundTasks
):
    '''
    Endpoint to create BOTH a RAG pipeline and a "quick action" AI note, chaining 2 celery notes together
        - process_pdf_task():
            input: request.files, request.metadata
            returns: None

        - rag_note_task():
            input: request.metadata
            returns: None
        
    Request contains:
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # 1) Kick off a chain: upload → note
        workflow = chain(
            process_pdf_task.s(request.files, request.metadata),
            rag_note_task.s(
                request.metadata["user_id"],
                request.metadata["chat_session_id"],
                request.metadata["summary_query"],  # e.g. your custom "Based on the docs…" prompt
                request.metadata["project_id"],
                request.metadata["model_name"]
            )
        )
        result: AsyncResult = workflow.apply_async()
        return {"workflow_id": result.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-to-rag-project/", response_model=NewRagPipelineResponse)
async def append_sources_to_project(request: NewRagPipelineRequest, background_tasks: BackgroundTasks):
    '''
    @TODO HAVE NOT STARTED !!!
    Endpoint to add new sources to an existing RAG pipeline for AI notes in one call:
        - process_pdf_task:
            input: request.files, request.metadata
            returns: source_ids
        
    Request contains
    project_id
        request.files (List): list of pdf file links
        request.metadata (json): {
            project_id:
        }
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)        # Upload PDFs to AWS S3, and Supabase
        # validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            # validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_project_documents_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        # validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            # "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat/original")
async def rag_chat_original(request: RagQueryRequest):
    """Endpoint for the rag query responses (token streaming not enabled)"""
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_name
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat/")
async def rag_chat(request: RagQueryRequest):
    """
    Endpoint for the rag query responses (token streaming not enabled)
    Uses Celery chain to perform persist → rag back-to-back
    
    **FYI chords are used to fire off several task in parallel, then call one callback with all their results

    Raw json:
    ----------
    {
        "user_id": "ae3df626-12cc-40c1-9d40-1d8249deea2e",
        "chat_session_id": "...",
        "query": "...",
        "project_id": "...",
        "model_name": "..." 
    }
    """
    try:
        # Build a chain: persist → rag
        job = chain(
            persist_user_query.s(
                request.user_id,
                request.chat_session_id,
                request.query,
                request.project_id,
                request.model_name
            ),
            # the return value of persist_user_query (i.e. message_id) ...
            # will be passed as the FIRST arg to rag_chat_task
            rag_chat_task.s(
                request.user_id,
                request.chat_session_id,
                request.query,
                request.project_id,
                request.model_name
            )
        ).apply_async()
        # Return the task ID to the client
        return {"task_id": job.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat-stream/")
async def rag_chat_stream(request: RagQueryRequest):
    """Endpoint for the rag query responses (token streaming not ENABLED), still a WORK IN PROGRESS !!! """
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_streaming_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_name
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-chat/regenerate/")
async def rag_chat_regenerate(request: RagRegenerateRequest):
    """
    Endpoint for regenerating a "FAILED" RAG query (token streaming not ENABLED)
    
    Raw json:
    ----------
    {
        "user_id": "...",
        "chat_session_id": "...",
        "query": "...",
        "project_id": "...",
        "model_name": "..." 
    }
    """
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_streaming_task.apply_async(args=[
            request.user_id,
            request.message_id,
            request.query,
            request.project_id,
            request.model_name
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag-chat-stream/regenerate/")
async def rag_chat_stream_regenerate(request: RagQueryRequest):
    """@TODO Endpoint for regenerating a failed RAG query (token streaming IS ENABLED), still a WORK IN PROGRESS !!! """
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_streaming_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_name
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/pdf-upload-task-status/{task_id}")
async def get_pdf_upload_status(task_id: str):
    """
    Status checker for process_pdf_task(self, files, metadata=None)

    Returns JSON with:
        - status: PENDING|STARTED|RETRY|SUCCESS|FAILURE
        - on FAILURE: exc_type + exc_message + traceback
        - on PROGRESS: stage, message, processed_files, total_files
        - on SUCCESS: result payload + elapsed_time
    """
    try:
        result = AsyncResult(task_id)
        state  = result.state
        info   = result.info if isinstance(result.info, dict) else {}
        now    = datetime.now(timezone.utc)

        # Compute elapsed_time if start_time was set
        elapsed = None
        if info.get("start_time"):
            st = datetime.fromisoformat(info["start_time"])
            if st.tzinfo is None:
                st = st.replace(tzinfo=timezone.utc)
            elapsed = (now - st).total_seconds()

        if state == states.FAILURE:
            return {
                "task_id":      task_id,
                "status":       "FAILURE",
                "error_type":   info.get("exc_type"),
                "error_message": info.get("exc_message"),
                "traceback":    result.traceback
            }

        if state in (states.PENDING, states.STARTED, states.RETRY):
            return {
                "task_id":        task_id,
                "status":         "IN_PROGRESS",
                "stage":          info.get("stage"),
                "message":        info.get("message"),
                "processed":      info.get("processed_files"),
                "total":          info.get("total_files"),
                "elapsed_time_s": elapsed
            }

        if state == states.SUCCESS:
            return {
                "task_id":       task_id,
                "status":        "SUCCESS",
                "result":        result.result,      # the dict returned by process_pdf_task
                "elapsed_time_s": elapsed
            }

        # catch-all
        return {
            "task_id": task_id,
            "status":  state,
            "info":    info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check error: {e}")

@app.post(
    "/embedding-pipeline-task-status",
    response_model=EmbeddingStatusResponse,
    summary="Check embedding status for a batch of documents"
)
async def embedding_pipeline_task_status(
    req: EmbeddingStatusRequest
):
    """
    Poll Supabase for the vector_embed_status of each document in `req.source_ids`.
    Returns an overall workflow status plus per-document statuses.
    """
    try:
        # 1) Fetch current statuses
        resp = (
            supabase_client
                .table("document_sources")
                .select("id, vector_embed_status")
                .in_("id", req.source_ids)
                .execute()
        )
        items = resp.data or []

        # 2) Build per-document status list
        doc_statuses = [
            DocumentStatus(id=row["id"], status=row["vector_embed_status"])
            for row in items
        ]

        # 3) Decide overall workflow status
        statuses = {d.status for d in doc_statuses}
        if "FAILED" in statuses:
            overall = "WORKFLOW_ERROR"
            msg = "One or more documents failed to embed."
        elif statuses - {"COMPLETE"}:
            overall = "WORKFLOW_IN_PROGRESS"
            msg = "Document embedding is in progress."
        else:
            overall = "WORKFLOW_COMPLETE"
            msg = "All documents have been embedded."

        return EmbeddingStatusResponse(
            status=overall,
            message=msg,
            document_statuses=doc_statuses
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch embedding statuses: {e}"
        )

@app.get("/rag-chat-status/{task_id}")
async def get_rag_chat_status(task_id: str):
    """
    Check task status and result of the rag_chat_task.run() task.
    Includes timeout if stuck in PENDING > 20s    
    """
    # A proxy object that allows you to fetch the status and result of any Celery task
    result = AsyncResult(task_id)

    # Retrieve task metadata
    task_meta = result.info if isinstance(result.info, dict) else {}
    now = datetime.now(timezone.utc)

    # Use explicitly set start_time if available
    start_time_str = task_meta.get("start_time")
    if start_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed = (now - start_time).total_seconds()
        except Exception:
            elapsed = None
    else:
        elapsed = None  # fallback if broker doesn't store this

    if result.state == "PENDING":
        if elapsed and elapsed > 20:
            return {
                "task_id": task_id,
                "status": "TIMEOUT",
                "elapsed_time": elapsed,
                "message": "RAG task has exceeded 20 seconds. You may retry."
            }
        return {"task_id": task_id, "status": "PENDING", "elapsed_time": elapsed}
    elif result.state == "SUCCESS":
        return {"task_id": task_id, "status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "FAILURE", "error": str(result.result)}
    else:
        return {"task_id": task_id, "status": result.state}

# ================================================ #
#          RAG GENERATIVE NOTES ENDPOINTS
# ================================================ #

@app.post("/generate-rag-note/")
async def generate_rag_note(request: RagQueryRequest):
    """
    postman raw json body:
    {
        "user_id": "",
        "chat_session_id": "",
        "query": "",
        "project_id": "",
        "model_name": "" 
    }
    """
    try:
        # Trigger the RAG task asynchronously and add it to the queue
        task = rag_chat_task.apply_async(args=[
            request.user_id,
            request.chat_session_id,
            request.query,
            request.project_id,
            request.model_name
        ])
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================================================ #
#              PODCAST API ENDPOINTS
# ================================================ #

@app.post("/pdf-to-dialogue/", response_model=PDFResponse)
async def pdf_to_dialogue(request: PDFRequest, background_tasks: BackgroundTasks):
    '''
    Endpoint to create both a pdf/RAG pipeline and create an AI podcast in one call:
        - process_pdf_task 
        - validate_and_generate_audio_task
    '''
    try:
        # Create signatures for the tasks
        process_pdf_task_signature = process_pdf_task.s(request.files, request.metadata)
        validate_and_generate_audio_task_signature = validate_and_generate_audio_task.s(request.files, request.metadata)

        # Create the group
        task_group = group(
            process_pdf_task_signature,
            validate_and_generate_audio_task_signature
        )

        # Create the chord with the group and callback
        task_chord = chord(task_group)(insert_sources_media_association_task.s())

        # The result of the chord is the AsyncResult of the callback task
        chord_result = task_chord

        # Get task IDs
        process_pdf_task_id = chord_result.parent.results[0].id
        validate_and_generate_audio_task_id = chord_result.parent.results[1].id
        insert_task_id = chord_result.id

        # Return the task IDs to the client
        return {
            "audio_task_id": validate_and_generate_audio_task_id,
            "embedding_task_id": process_pdf_task_id,
            "insert_task_id": insert_task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Endpoint to process PDFs and generate dialogue transcript
@app.post("/pdf-to-dialogue-transcript/", response_model=PDFResponse)
async def pdf_to_dialogue_transcript(request: PDFRequest, background_tasks: BackgroundTasks):
    try:
        # Enqueue the Celery task for dialogue generation
        print(f"API DEBUG: {request.files}")
        task = generate_dialogue_only_task.apply_async(
            args=[request.files]
        )
        
        # Return the task ID to the client
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Combined endpoint to check the status of any Celery task
@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    Purpose:
        A generic Celery task monitor. It can check the status of any Celery task, including:
        - rag_chat_task
        - pdf_to_dialogue_task
        - addition_task
        ...or any other task your system may launch
        No custom behavior or message formatting per task type.

    Limitations:
        No custom behavior or message formatting per task type.
    """
    try:
        # A proxy object that allows you to fetch the status and result of any Celery task
        task_result = AsyncResult(task_id)

        # Retrieve task metadata
        task_meta = task_result.info if isinstance(task_result.info, dict) else {}
        start_time_str = task_meta.get('start_time')
        elapsed_time = None

        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        if task_result.state == 'PENDING':
            return {
                "task_id": task_id,
                "status": "PENDING",
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'SUCCESS':
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": task_result.result,
                "elapsed_time": elapsed_time
            }
        elif task_result.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "error": str(task_result.result),
                "elapsed_time": elapsed_time
            }
        else:
            return {
                "task_id": task_id,
                "status": task_result.state,
                "elapsed_time": elapsed_time
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    except Exception as e:
        # Handle any unexpected exceptions
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

# ================================================ #
#              DEV-TOOLS ENDPOINTS
# ================================================ #

@app.get("/debug-task/{task_id}")
async def debug_task(task_id: str):
    """
    Dev-only endpoint to inspect any Celery task’s state, metadata, and result.
    WARNING: Don’t expose this in production without access control.
    """
    try:
        result = AsyncResult(task_id)
        task_meta = result.info if isinstance(result.info, dict) else {}

        return {
            "task_id": task_id,
            "state": result.state,
            "is_ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "result": str(result.result),  # may be a full dict or traceback
            "metadata": task_meta,
            "traceback": result.traceback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving task: {str(e)}")
