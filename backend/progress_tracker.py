import logging
from typing import Dict, Any, Optional
from fastapi import WebSocket
import json
import asyncio

logger = logging.getLogger("progress_tracker")

class ProgressTracker:
    """
    Tracks progress of long-running operations and sends updates to clients.
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, session_id: str) -> None:
        """
        Start a new progress tracking session.
        
        Args:
            session_id: Unique identifier for the session
        """
        self.active_sessions[session_id] = {
            "progress": 0,
            "stage": "Initializing",
            "completed": False,
            "websocket": None
        }
        logger.info(f"Started progress tracking session: {session_id}")
    
    def set_websocket(self, session_id: str, websocket: WebSocket) -> None:
        """
        Associate a WebSocket connection with a session.
        
        Args:
            session_id: Session identifier
            websocket: WebSocket connection to send updates to
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["websocket"] = websocket
            logger.info(f"WebSocket connected for session: {session_id}")
    
    async def update_progress(self, session_id: str, progress: float, stage: str) -> None:
        """
        Update progress for a session and send notification to client.
        
        Args:
            session_id: Session identifier
            progress: Progress value (0-100)
            stage: Current processing stage description
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to update non-existent session: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        session["progress"] = progress
        session["stage"] = stage
        
        # Send update via WebSocket if available
        if session["websocket"] is not None:
            try:
                await session["websocket"].send_text(json.dumps({
                    "type": "progress",
                    "data": {
                        "progress": progress,
                        "stage": stage
                    }
                }))
                logger.debug(f"Sent progress update for session {session_id}: {progress}%, {stage}")
            except Exception as e:
                logger.error(f"Error sending progress update: {e}")
    
    async def complete_session(self, session_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a session as completed and send final result.
        
        Args:
            session_id: Session identifier
            result: Optional result data to send to client
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to complete non-existent session: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        session["completed"] = True
        session["progress"] = 100
        
        # Send completion notification via WebSocket if available
        if session["websocket"] is not None:
            try:
                await session["websocket"].send_text(json.dumps({
                    "type": "complete",
                    "data": result or {}
                }))
                logger.info(f"Completed session: {session_id}")
            except Exception as e:
                logger.error(f"Error sending completion notification: {e}")
        
        # Clean up session after a delay
        asyncio.create_task(self._cleanup_session(session_id))
    
    async def _cleanup_session(self, session_id: str, delay: int = 60) -> None:
        """
        Clean up session data after a delay.
        
        Args:
            session_id: Session identifier
            delay: Delay in seconds before cleanup
        """
        await asyncio.sleep(delay)
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")

# Global instance
progress_tracker = ProgressTracker()
