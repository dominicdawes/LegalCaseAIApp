# app/ws_handlers/connection_manager.py

import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str = None):
        """Connect a WebSocket to a chat session"""
        await websocket.accept()
        
        # Add to session connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        
        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        logger.info(f"✅ WebSocket connected to session {session_id} (user: {user_id})")
        
    async def disconnect(self, websocket: WebSocket, session_id: str, user_id: str = None):
        """Disconnect a WebSocket from a chat session"""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        # Clean up user sessions
        if user_id and user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
                
        logger.info(f"❌ WebSocket disconnected from session {session_id}")
    
    async def send_to_session(self, session_id: str, message: dict):
        """Send message to all connections in a session"""
        if session_id in self.active_connections:
            dead_connections = set()
            
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send to connection: {e}")
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead_conn in dead_connections:
                self.active_connections[session_id].discard(dead_conn)
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to all sessions for a user"""
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id]:
                await self.send_to_session(session_id, message)

    def get_stats(self):
        """Get connection statistics"""
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        return {
            "active_sessions": len(self.active_connections),
            "total_connections": total_connections,
            "active_users": len(self.user_sessions)
        }

# Global connection manager instance
manager = ConnectionManager()