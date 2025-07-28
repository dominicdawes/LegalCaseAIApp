# app/ws_handlers/endpoints.py

import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from celery import chain

from .connection_manager import manager
from .handlers import (
    listen_redis_and_forward, 
    receive_websocket_messages,
    handle_chat_message
)

logger = logging.getLogger(__name__)

def setup_websocket_routes(app: FastAPI, redis_pub):
    """Setup all WebSocket routes on the FastAPI app
    session_id: chat session id for that particular convo (stored in Supabase table `chat_sessions`)
    """
    
    @app.websocket("/ws/chat/{session_id}")
    async def websocket_chat_enhanced(websocket: WebSocket, session_id: str, user_id: str = None):
        """
        Enhanced WebSocket endpoint for real-time chat streaming
        
        Usage from WeWeb:
        const ws = new WebSocket(`wss://your-app.onrender.com/ws/chat/${sessionId}?user_id=${userId}`);
        """
        await manager.connect(websocket, session_id, user_id)
        
        # Subscribe to Redis channels for this session
        pubsub = redis_pub.pubsub()
        await pubsub.subscribe(f"chat:{session_id}")
        
        try:
            # Create tasks for both directions
            listen_task = asyncio.create_task(
                listen_redis_and_forward(pubsub, session_id)
            )
            receive_task = asyncio.create_task(
                receive_websocket_messages(websocket, session_id, user_id)
            )
            
            # Wait for either task to complete (usually due to disconnect)
            done, pending = await asyncio.wait(
                [listen_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
        except WebSocketDisconnect:
            logger.info(f"🔌 WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"❌ WebSocket error for session {session_id}: {e}")
        finally:
            await pubsub.unsubscribe(f"chat:{session_id}")
            await pubsub.close()
            await manager.disconnect(websocket, session_id, user_id)

    @app.websocket("/ws/status/{user_id}")
    async def websocket_system_status(websocket: WebSocket, user_id: str):
        """
        WebSocket for system-wide status updates (tasks, notifications, etc.)
        """
        await websocket.accept()
        
        # Subscribe to user-specific status channel
        pubsub = redis_pub.pubsub()
        await pubsub.subscribe(f"user:{user_id}")
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    await websocket.send_text(message['data'])
                    
        except WebSocketDisconnect:
            logger.info(f"🔌 Status WebSocket disconnected for user {user_id}")
        finally:
            await pubsub.unsubscribe(f"user:{user_id}")
            await pubsub.close()

    @app.get("/ws/health")
    async def websocket_health():
        """Health check for WebSocket infrastructure"""
        try:
            # Check Redis connection
            await redis_pub.ping()
            
            # Get connection stats
            stats = manager.get_stats()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                **stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # Enhanced REST endpoint that works with WebSocket
    from tasks.chat_tasks import persist_user_query, rag_chat_task
    from pydantic import BaseModel
    
    # You'll need to import or redefine your RagQueryRequest model
    class RagQueryRequest(BaseModel):
        user_id: str
        chat_session_id: str
        query: str
        project_id: str
        provider: str
        model_name: str
        temperature: float

    @app.post("/rag-chat-websocket/")
    async def rag_chat_websocket(request: RagQueryRequest):
        """Enhanced RAG chat endpoint that works with WebSocket streaming"""
        try:
            # Send immediate acknowledgment via WebSocket
            await manager.send_to_session(request.chat_session_id, {
                "type": "processing_started",
                "query": request.query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trigger RAG workflow
            job = chain(
                persist_user_query.s(
                    request.user_id, request.chat_session_id, request.query,
                    request.project_id, request.model_name
                ),
                rag_chat_task.s(
                    request.user_id, request.chat_session_id, request.query,
                    request.project_id, request.provider, request.model_name, request.temperature
                )
            ).apply_async()
            
            # Send task info via WebSocket
            await manager.send_to_session(request.chat_session_id, {
                "type": "task_queued",
                "task_id": job.id,
                "estimated_time": "2-5 seconds",
                "timestamp": datetime.now().isoformat()
            })
            
            return {"task_id": job.id, "websocket_channel": request.chat_session_id}
            
        except Exception as e:
            # Send error via WebSocket
            await manager.send_to_session(request.chat_session_id, {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise HTTPException(status_code=400, detail=str(e))