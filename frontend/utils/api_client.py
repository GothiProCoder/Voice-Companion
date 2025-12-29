"""
GuppShupp API Client
====================

HTTP client for communicating with the GuppShupp FastAPI backend.
Includes SSE support for streaming chat responses with heartbeat.

Features:
    - Authentication (signup, login, logout)
    - SSE streaming for chat with heartbeat handling
    - Automatic token management
    - Error handling with retries
    - Conversation history

Author: GuppShupp Team
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Generator, Callable
from uuid import UUID
import uuid

import httpx
import sseclient

from frontend.config import config

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class User:
    """User data from API."""
    id: str
    username: str
    email: str
    display_name: Optional[str] = None
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    is_active: bool = True


@dataclass
class ChatMessage:
    """Single chat message."""
    id: str
    role: str  # "user" or "assistant"
    text: str
    audio_base64: Optional[str] = None
    emotion: Optional[str] = None
    timestamp: Optional[datetime] = None
    processing_time_ms: Optional[int] = None


@dataclass
class SSEEvent:
    """Server-Sent Event from chat endpoint."""
    event_type: str  # heartbeat, progress, complete, error
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# API CLIENT
# =============================================================================


class GuppShuppClient:
    """
    HTTP client for GuppShupp API.
    
    Handles authentication, SSE streaming, and API calls.
    
    Example:
        client = GuppShuppClient()
        
        # Login
        user = client.login("username", "password")
        
        # Chat with SSE
        for event in client.chat_stream(session_id, audio_base64):
            if event.event_type == "heartbeat":
                print("Processing...")
            elif event.event_type == "complete":
                print(event.data["response"]["ai_response_text"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize API client.
        
        Args:
            base_url: API base URL (default from config)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or config.api_url
        self.timeout = timeout
        self.session_token: Optional[str] = None
        self.current_user: Optional[User] = None
        
        # HTTP client
        self._client = httpx.Client(timeout=timeout)
    
    def _get_headers(self, include_auth: bool = True) -> Dict[str, str]:
        """Get request headers with optional auth token."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Bypass ngrok free tier interstitial page
            "ngrok-skip-browser-warning": "true",
        }
        if include_auth and self.session_token:
            headers["X-Session-Token"] = self.session_token
        return headers
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"message": response.text}
        
        if response.status_code >= 400:
            error_msg = data.get("detail") or data.get("message", "Unknown error")
            raise APIError(
                status_code=response.status_code,
                message=error_msg,
                retryable=response.status_code >= 500
            )
        
        return data
    
    # =========================================================================
    # HEALTH
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self._client.get(f"{self.base_url}/health")
        return self._handle_response(response)
    
    def is_healthy(self) -> bool:
        """Check if API is healthy."""
        try:
            result = self.health_check()
            return result.get("status") in ["healthy", "degraded"]
        except Exception:
            return False
    
    # =========================================================================
    # AUTHENTICATION
    # =========================================================================
    
    def signup(
        self,
        username: str,
        email: str,
        password: str,
        display_name: Optional[str] = None
    ) -> User:
        """
        Register new user.
        
        Args:
            username: Unique username
            email: Email address
            password: Password (min 6 chars)
            display_name: Optional display name
            
        Returns:
            User object
            
        Raises:
            APIError: If signup fails
        """
        response = self._client.post(
            f"{self.base_url}/auth/signup",
            headers=self._get_headers(include_auth=False),
            json={
                "username": username,
                "email": email,
                "password": password,
                "display_name": display_name
            }
        )
        
        data = self._handle_response(response)
        
        # Store session token
        self.session_token = data["session_token"]
        
        # Create user object
        user_data = data["user"]
        self.current_user = User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            display_name=user_data.get("display_name"),
            created_at=user_data.get("created_at"),
            is_active=user_data.get("is_active", True)
        )
        
        logger.info(f"Signup successful: {self.current_user.username}")
        return self.current_user
    
    def login(self, username: str, password: str) -> User:
        """
        Login user.
        
        Args:
            username: Username or email
            password: Password
            
        Returns:
            User object
            
        Raises:
            APIError: If login fails
        """
        response = self._client.post(
            f"{self.base_url}/auth/login",
            headers=self._get_headers(include_auth=False),
            json={
                "username": username,
                "password": password
            }
        )
        
        data = self._handle_response(response)
        
        # Store session token
        self.session_token = data["session_token"]
        
        # Create user object
        user_data = data["user"]
        self.current_user = User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            display_name=user_data.get("display_name"),
            created_at=user_data.get("created_at"),
            last_login=user_data.get("last_login"),
            is_active=user_data.get("is_active", True)
        )
        
        logger.info(f"Login successful: {self.current_user.username}")
        return self.current_user
    
    def logout(self) -> bool:
        """
        Logout current user.
        
        Returns:
            True if successful
        """
        if not self.session_token:
            return True
        
        try:
            response = self._client.post(
                f"{self.base_url}/auth/logout",
                headers=self._get_headers()
            )
            self._handle_response(response)
        except Exception as e:
            logger.warning(f"Logout error (ignored): {e}")
        
        self.session_token = None
        self.current_user = None
        logger.info("Logged out")
        return True
    
    def get_profile(self) -> User:
        """Get current user profile."""
        response = self._client.get(
            f"{self.base_url}/auth/me",
            headers=self._get_headers()
        )
        
        data = self._handle_response(response)
        
        self.current_user = User(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            display_name=data.get("display_name"),
            created_at=data.get("created_at"),
            last_login=data.get("last_login"),
            is_active=data.get("is_active", True)
        )
        
        return self.current_user
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.session_token is not None
    
    # =========================================================================
    # CHAT (SSE STREAMING)
    # =========================================================================
    
    def chat_stream(
        self,
        session_id: str,
        audio_base64: str,
        audio_format: str = "wav",
        session_context: Optional[Dict[str, Any]] = None,
        on_heartbeat: Optional[Callable[[SSEEvent], None]] = None,
        on_progress: Optional[Callable[[SSEEvent], None]] = None,
    ) -> Generator[SSEEvent, None, None]:
        """
        Send audio and stream response via SSE.
        
        This is a generator that yields SSE events as they arrive.
        The stream includes heartbeats every 10 seconds.
        
        Args:
            session_id: Chat session UUID
            audio_base64: Base64-encoded audio
            audio_format: Audio format (wav, mp3, webm)
            session_context: Optional session context
            on_heartbeat: Optional callback for heartbeat events
            on_progress: Optional callback for progress events
            
        Yields:
            SSEEvent objects (heartbeat, progress, complete, error)
            
        Example:
            for event in client.chat_stream(session_id, audio_b64):
                if event.event_type == "complete":
                    response = event.data["response"]
                    print(response["ai_response_text"])
        """
        if not self.session_token:
            raise APIError(401, "Not authenticated", retryable=False)
        
        url = f"{self.base_url}/conversations/chat"
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        
        payload = {
            "session_id": session_id,
            "audio_base64": audio_base64,
            "audio_format": audio_format,
            "session_context": session_context or {}
        }
        
                # ✅ ENHANCED DEBUG LOGGING
        logger.info("="*80)
        logger.info("🔍 API CLIENT - SENDING TO BACKEND")
        logger.info("="*80)
        logger.info(f"📊 URL: {url}")
        logger.info(f"📊 Session ID: {session_id}")
        logger.info(f"📊 Audio format: {audio_format}")
        logger.info(f"📊 Audio base64 type: {type(audio_base64)}")
        logger.info(f"📊 Audio base64 length: {len(audio_base64) if isinstance(audio_base64, str) else 0}")
        logger.info(f"🔍 First 50 chars: {audio_base64[:50] if isinstance(audio_base64, str) else 'NOT A STRING!'}")
        
        # Detect format from base64
        if isinstance(audio_base64, str) and len(audio_base64) > 100:
            try:
                decoded_preview = base64.b64decode(audio_base64[:100])
                if decoded_preview[:4] == b'RIFF':
                    logger.info("✅ Detected: WAV format (from base64)")
                elif decoded_preview[:3] == b'ID3':
                    logger.warning("⚠️ Detected: MP3 with ID3 tags (from base64)")
                elif decoded_preview[:2] in [b'\xff\xfb', b'\xff\xfa']:
                    logger.info("✅ Detected: MP3 format (from base64)")
                else:
                    logger.warning(f"⚠️ Unknown format. First 4 bytes: {decoded_preview[:4].hex()}")
            except Exception as e:
                logger.warning(f"⚠️ Could not decode preview: {e}")
        
        logger.info("📤 Sending POST request...")

        # ⚠️ DYNAMIC HEARTBEAT TIMEOUT CONFIG
        HEARTBEAT_MAX_WAIT_SECONDS = 30  # Max time without heartbeat before timeout
        
        try:
            # Use longer initial timeout for connection, shorter for reads
            # The heartbeat system handles the actual timeout logic
            with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(
                    connect=30.0,      # Connection timeout
                    read=35.0,         # Read timeout (slightly longer than heartbeat interval)
                    write=30.0,        # Write timeout
                    pool=10.0,         # Pool acquisition timeout
                )
            ) as response:
                # ✅ LOG RESPONSE STATUS
                logger.info(f"✅ Backend responded: HTTP {response.status_code}")
                
                if response.status_code >= 400:
                    # Try to read error body
                    error_body = ""
                    for chunk in response.iter_text():
                        error_body += chunk
                        if len(error_body) > 1000:
                            break
                    
                    try:
                        error_data = json.loads(error_body)
                        error_msg = error_data.get("detail", error_body[:100])
                    except json.JSONDecodeError:
                        error_msg = error_body[:100]
                    
                    raise APIError(
                        response.status_code,
                        error_msg,
                        retryable=response.status_code >= 500
                    )
                
                # ✅ Parse SSE stream with DYNAMIC heartbeat timeout
                logger.info("📡 Parsing SSE stream with dynamic heartbeat timeout...")
                
                # Track last activity time for heartbeat timeout
                last_activity_time = time.time()
                
                client = sseclient.SSEClient(response.iter_bytes())
                for sse_event in client.events():
                    
                    # ✅ DYNAMIC TIMEOUT CHECK: Error if no activity for too long
                    current_time = time.time()
                    time_since_last_activity = current_time - last_activity_time
                    
                    if time_since_last_activity > HEARTBEAT_MAX_WAIT_SECONDS:
                        logger.error(f"❌ Heartbeat timeout! No activity for {time_since_last_activity:.1f}s")
                        raise APIError(
                            408,
                            f"Connection lost - no heartbeat for {int(time_since_last_activity)} seconds",
                            retryable=True
                        )
                    
                    # Reset activity timer on ANY event
                    last_activity_time = current_time
                    
                    # ✅ LOG EACH SSE EVENT
                    logger.debug(f"📨 SSE Event: type={sse_event.event}, data_length={len(sse_event.data) if sse_event.data else 0}")
                    
                    try:
                        event_type = sse_event.event or "message"
                        data = json.loads(sse_event.data) if sse_event.data else {}
                        
                        # ✅ LOG EVENT TYPE
                        if event_type == "heartbeat":
                            elapsed_ms = data.get("elapsed_ms", 0)
                            phase = data.get("phase", "unknown")
                            logger.info(f"💓 Heartbeat: phase={phase}, elapsed={elapsed_ms}ms")
                        elif event_type == "complete":
                            logger.info(f"✅ SSE Complete event received")
                        elif event_type == "error":
                            logger.error(f"❌ SSE Error event: {data.get('message', 'Unknown')}")
                        elif event_type == "progress":
                            logger.info(f"📊 SSE Progress: {data.get('message', '')}")
                        
                        event = SSEEvent(
                            event_type=event_type,
                            data=data,
                            timestamp=datetime.now()
                        )
                        
                        # Call optional callbacks
                        if event_type == "heartbeat" and on_heartbeat:
                            on_heartbeat(event)
                        elif event_type == "progress" and on_progress:
                            on_progress(event)
                        
                        yield event
                        
                        # Stop on complete or error
                        if event_type in ["complete", "error"]:
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {e}")
                        continue
                        
        except httpx.TimeoutException as e:
            logger.error(f"❌ HTTP Timeout: {e}")
            raise APIError(408, f"Request timeout: {str(e)}", retryable=True)
        except httpx.ConnectError:
            raise APIError(503, "Connection failed", retryable=True)

    
    def chat(
        self,
        session_id: str,
        audio_base64: str,
        audio_format: str = "wav",
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send audio and wait for complete response.
        
        This is a convenience method that handles the SSE stream
        internally and returns the final response.
        
        Args:
            session_id: Chat session UUID
            audio_base64: Base64-encoded audio
            audio_format: Audio format
            session_context: Optional session context
            
        Returns:
            Complete ChatResponse dict
            
        Raises:
            APIError: If chat fails
        """
        for event in self.chat_stream(
            session_id,
            audio_base64,
            audio_format,
            session_context
        ):
            if event.event_type == "complete":
                return event.data.get("response", event.data)
            elif event.event_type == "error":
                raise APIError(
                    500,
                    event.data.get("message", "Unknown error"),
                    retryable=event.data.get("retryable", True)
                )
        
        raise APIError(500, "No response received", retryable=True)
    
    # =========================================================================
    # SESSIONS
    # =========================================================================
    
    def create_session(self, title: Optional[str] = None) -> str:
        """
        Create new chat session.
        
        Args:
            title: Optional session title
            
        Returns:
            New session UUID
        """
        response = self._client.post(
            f"{self.base_url}/conversations/sessions",
            headers=self._get_headers(),
            json={"title": title} if title else {}
        )
        
        data = self._handle_response(response)
        return data["session_id"]
    
    def list_sessions(self, limit: int = 20) -> list:
        """
        List user's chat sessions.
        
        Args:
            limit: Max sessions to return
            
        Returns:
            List of session info dicts
        """
        response = self._client.get(
            f"{self.base_url}/conversations/sessions",
            headers=self._get_headers(),
            params={"limit": limit}
        )
        
        data = self._handle_response(response)
        return data.get("sessions", [])
    
    def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        include_audio: bool = False
    ) -> dict:
        """
        Get conversation history for session with optional audio.
        
        Args:
            session_id: Session UUID
            limit: Max items to return
            offset: Pagination offset
            include_audio: Whether to include audio data (larger response)
            
        Returns:
            Dict with 'items', 'total_count', 'has_more'
        """
        response = self._client.get(
            f"{self.base_url}/conversations/sessions/{session_id}/conversations",
            headers=self._get_headers(),
            params={
                "limit": limit,
                "offset": offset,
                "include_audio": include_audio
            },
            timeout=60.0  # Longer timeout when fetching audio
        )
        
        data = self._handle_response(response)
        return data
    
    def get_audio_url(self, conversation_id: str) -> str:
        """
        Get the streaming audio URL for a conversation.
        
        NOTE: This URL requires authentication - use get_conversation_audio() 
        which handles auth and returns playable base64.
        """
        return f"{self.base_url}/conversations/conversations/{conversation_id}/audio"
    
    def get_conversation_audio(self, conversation_id: str) -> Optional[str]:
        """
        Fetch audio for a conversation and return as base64 for playback.
        
        This method handles authentication automatically and returns the audio
        in a format that st.audio() can play directly.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Base64-encoded audio string ("data:audio/ogg;base64,..."), or None if not available
        """
        import base64
        
        try:
            response = self._client.get(
                f"{self.base_url}/conversations/conversations/{conversation_id}/audio",
                headers=self._get_headers(),  # Include JWT auth token
                timeout=30.0
            )
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                logger.warning(f"Audio fetch failed: {response.status_code}")
                return None
            
            # Convert audio bytes to base64 data URI for st.audio()
            audio_bytes = response.content
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Return as data URI that st.audio() can play directly
            return f"data:audio/ogg;base64,{audio_b64}"
            
        except Exception as e:
            logger.warning(f"Failed to fetch audio for {conversation_id}: {e}")
            return None
    
    
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def close(self):
        """Close HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# EXCEPTIONS
# =============================================================================


class APIError(Exception):
    """API error with status code and retry information."""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        retryable: bool = False
    ):
        self.status_code = status_code
        self.message = message
        self.retryable = retryable
        super().__init__(f"[{status_code}] {message}")


# =============================================================================
# UTILITIES
# =============================================================================


def encode_audio_file(file_path: str) -> str:
    """Encode audio file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def encode_audio_bytes(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64."""
    return base64.b64encode(audio_bytes).decode("ascii")


def generate_session_id() -> str:
    """Generate new session UUID."""
    return str(uuid.uuid4())
