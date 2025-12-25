"""
GuppShupp Conversation Schemas
==============================

Pydantic models for chat and conversation endpoints.
Includes SSE event schemas for heartbeat-based communication.

Author: GuppShupp Team
"""

import base64
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "webm", "ogg", "m4a"]
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_AUDIO_DURATION_SECONDS = 120  # 2 minutes


# =============================================================================
# PHASE TIMINGS
# =============================================================================


class PhaseTimings(BaseModel):
    """
    Detailed timing breakdown for each workflow phase.
    Useful for performance monitoring and debugging.
    """
    
    audio_analysis_ms: int = Field(
        ...,
        description="Phase 1: Whisper ASR + emotion detection",
        ge=0
    )
    
    context_preparation_ms: int = Field(
        ...,
        description="Phase 2: Memory retrieval + session context",
        ge=0
    )
    
    llm_generation_ms: int = Field(
        ...,
        description="Phase 3: Gemini LLM response generation",
        ge=0
    )
    
    tts_generation_ms: int = Field(
        ...,
        description="Phase 4: Parler TTS audio synthesis",
        ge=0
    )
    
    database_persistence_ms: int = Field(
        ...,
        description="Phase 5: Database storage",
        ge=0
    )


# =============================================================================
# SESSION INFO
# =============================================================================


class SessionInfo(BaseModel):
    """
    Information about a chat session.
    """
    
    session_id: UUID = Field(
        ...,
        description="Unique session identifier"
    )
    
    user_id: UUID = Field(
        ...,
        description="Owner user ID"
    )
    
    created_at: datetime = Field(
        ...,
        description="Session creation timestamp"
    )
    
    last_activity: datetime = Field(
        ...,
        description="Last message timestamp"
    )
    
    message_count: int = Field(
        ...,
        description="Total messages in session",
        ge=0
    )
    
    title: Optional[str] = Field(
        None,
        description="Session title (auto-generated from first message)"
    )


# =============================================================================
# CHAT REQUEST
# =============================================================================


class ChatRequest(BaseModel):
    """
    Request to process audio and generate AI response.
    
    Audio should be base64-encoded WAV, MP3, or WebM.
    Session context can include TTS preferences like speaker gender.
    
    Example:
        {
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "audio_base64": "UklGRi4AAABXQVZFZm10...",
            "audio_format": "wav",
            "session_context": {
                "current_tts_speaker": "Rohit",
                "voice_preferences": {"gender": "male"}
            }
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "audio_base64": "UklGRi4AAABXQVZFZm10...",
                "audio_format": "wav",
                "session_context": {
                    "current_tts_speaker": "Rohit",
                    "voice_preferences": {"gender": "male"}
                }
            }
        }
    )
    
    session_id: UUID = Field(
        ...,
        description="Chat session identifier (create new UUID for new session)"
    )
    
    audio_base64: str = Field(
        ...,
        description="Base64-encoded audio data",
        min_length=100  # Minimum reasonable audio size
    )
    
    audio_format: Literal["wav", "mp3", "webm", "ogg", "m4a"] = Field(
        "wav",
        description="Audio format"
    )
    
    session_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Session context including TTS preferences"
    )
    
    @field_validator("audio_base64")
    @classmethod
    def validate_audio_base64(cls, v: str) -> str:
        """Validate base64 encoding and size."""
        # Remove any whitespace or newlines
        v = v.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        
        # Check if it's valid base64
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        
        # Check size
        if len(decoded) > MAX_AUDIO_SIZE_BYTES:
            raise ValueError(
                f"Audio too large. Maximum size: {MAX_AUDIO_SIZE_BYTES // (1024*1024)} MB"
            )
        
        if len(decoded) < 100:
            raise ValueError("Audio data too small - possibly corrupted")
        
        return v
    
    @model_validator(mode="after")
    def validate_session_context(self):
        """Ensure session_context has valid structure if provided."""
        if self.session_context is not None:
            # Validate known keys
            allowed_keys = {
                "current_tts_speaker",
                "session_language", 
                "voice_preferences"
            }
            for key in self.session_context.keys():
                if key not in allowed_keys:
                    # Allow extra keys for extensibility, just log warning
                    pass
        return self


# =============================================================================
# SAFETY FLAGS
# =============================================================================


class SafetyFlagsResponse(BaseModel):
    """Safety assessment from the AI response."""
    
    crisis_risk: Literal["low", "medium", "high"] = Field(
        ...,
        description="Self-harm/crisis risk level"
    )
    
    self_harm_mentioned: bool = Field(
        False,
        description="Whether self-harm was mentioned"
    )
    
    abuse_mentioned: bool = Field(
        False,
        description="Whether abuse was mentioned"
    )
    
    medical_concern: bool = Field(
        False,
        description="Whether medical issues were mentioned"
    )
    
    flagged_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that triggered safety flags"
    )


# =============================================================================
# CHAT RESPONSE
# =============================================================================


class ChatResponse(BaseModel):
    """
    Complete AI response with audio.
    
    Contains transcription, AI response, emotion analysis,
    TTS audio, and processing metadata.
    
    Example:
        {
            "request_id": "req_1735039200_abc123",
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "conversation_id": "456e7890-...",
            "user_input_text": "Hello, mera naam Rahul hai",
            "detected_language": "hi",
            "ai_response_text": "Namaste Rahul! Main Guppu hoon...",
            "response_language": "hi",
            "detected_emotion": "neutral",
            "emotion_confidence": 0.85,
            "sentiment": "positive",
            "response_audio_base64": "UklGRi4AAABXQVZFZm10...",
            "response_duration_seconds": 3.5,
            "tts_speaker": "Rohit",
            "total_processing_time_ms": 5432,
            "phase_timings": { ... },
            "safety_passed": true,
            "safety_action": "continue"
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_1735039200_abc123",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "conversation_id": "456e7890-e89b-12d3-a456-426614174000",
                "user_input_text": "Hello, mera naam Rahul hai",
                "detected_language": "hi",
                "ai_response_text": "Namaste Rahul! Main Guppu hoon, tumhara dost!",
                "response_language": "hi",
                "detected_emotion": "neutral",
                "emotion_confidence": 0.85,
                "sentiment": "positive",
                "detected_intent": "greeting",
                "intent_confidence": 0.92,
                "response_audio_base64": "UklGRi4AAABXQVZFZm10...",
                "response_duration_seconds": 3.5,
                "tts_speaker": "Rohit",
                "total_processing_time_ms": 5432,
                "safety_passed": True,
                "safety_action": "continue"
            }
        }
    )
    
    # Identifiers
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )
    
    session_id: UUID = Field(
        ...,
        description="Chat session identifier"
    )
    
    conversation_id: UUID = Field(
        ...,
        description="Stored conversation record ID"
    )
    
    # Transcription
    user_input_text: str = Field(
        ...,
        description="Transcribed user audio"
    )
    
    detected_language: str = Field(
        ...,
        description="Detected input language code",
        examples=["hi", "en", "hi-en", "ta"]
    )
    
    # AI Response
    ai_response_text: str = Field(
        ...,
        description="AI-generated response text"
    )
    
    response_language: str = Field(
        ...,
        description="Response language code"
    )
    
    # Emotion Analysis
    detected_emotion: str = Field(
        ...,
        description="Detected user emotion",
        examples=["neutral", "joy", "sadness", "anger", "fear", "anxious"]
    )
    
    emotion_confidence: float = Field(
        ...,
        description="Emotion detection confidence",
        ge=0.0,
        le=1.0
    )
    
    sentiment: str = Field(
        ...,
        description="Overall sentiment",
        examples=["positive", "negative", "neutral", "mixed"]
    )
    
    detected_intent: str = Field(
        ...,
        description="Detected user intent",
        examples=["greeting", "question", "venting", "seeking_advice"]
    )
    
    intent_confidence: float = Field(
        ...,
        description="Intent detection confidence",
        ge=0.0,
        le=1.0
    )
    
    # Audio Response
    response_audio_base64: str = Field(
        ...,
        description="Base64-encoded WAV audio response"
    )
    
    response_duration_seconds: float = Field(
        ...,
        description="Audio response duration",
        ge=0.0
    )
    
    tts_speaker: str = Field(
        ...,
        description="TTS speaker name used",
        examples=["Rohit", "Divya", "Thoma", "Mary"]
    )
    
    # Processing Metadata
    total_processing_time_ms: int = Field(
        ...,
        description="Total end-to-end processing time",
        ge=0
    )
    
    phase_timings: Optional[PhaseTimings] = Field(
        None,
        description="Detailed per-phase timings"
    )
    
    # Safety
    safety_passed: bool = Field(
        ...,
        description="Whether safety checks passed"
    )
    
    safety_action: Literal["continue", "escalate", "block"] = Field(
        ...,
        description="Recommended safety action"
    )
    
    safety_flags: Optional[SafetyFlagsResponse] = Field(
        None,
        description="Detailed safety assessment"
    )
    
    # Memory
    memories_extracted: int = Field(
        0,
        description="Number of memories extracted from conversation",
        ge=0
    )


# =============================================================================
# SSE EVENTS
# =============================================================================


class SSEHeartbeat(BaseModel):
    """
    SSE heartbeat event sent every 10 seconds.
    Resets the client's timeout counter.
    
    Example:
        event: heartbeat
        data: {"timestamp": "2025-12-24T15:30:00Z", "phase": "llm_generation"}
    """
    
    type: Literal["heartbeat"] = "heartbeat"
    
    timestamp: datetime = Field(
        ...,
        description="Heartbeat timestamp"
    )
    
    phase: Optional[str] = Field(
        None,
        description="Current processing phase",
        examples=["audio_analysis", "context_preparation", "llm_generation", "tts_generation", "database_persistence"]
    )
    
    elapsed_ms: Optional[int] = Field(
        None,
        description="Elapsed time since request started",
        ge=0
    )


class SSEProgress(BaseModel):
    """
    SSE progress event with phase information.
    
    Example:
        event: progress
        data: {"phase": "llm_generation", "message": "Generating response...", "progress_percent": 60}
    """
    
    type: Literal["progress"] = "progress"
    
    phase: str = Field(
        ...,
        description="Current processing phase"
    )
    
    message: str = Field(
        ...,
        description="Human-readable progress message"
    )
    
    progress_percent: Optional[int] = Field(
        None,
        description="Approximate progress percentage",
        ge=0,
        le=100
    )
    
    timestamp: datetime = Field(
        ...,
        description="Event timestamp"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Phase-specific details"
    )


class SSEComplete(BaseModel):
    """
    SSE completion event with full response.
    
    Example:
        event: complete
        data: {"response": ChatResponse}
    """
    
    type: Literal["complete"] = "complete"
    
    response: ChatResponse = Field(
        ...,
        description="Complete chat response"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Completion timestamp"
    )


class SSEError(BaseModel):
    """
    SSE error event.
    
    Example:
        event: error
        data: {"error": "rate_limit", "message": "...", "retryable": true}
    """
    
    type: Literal["error"] = "error"
    
    error: str = Field(
        ...,
        description="Error type identifier"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    retryable: bool = Field(
        ...,
        description="Whether client should retry"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request ID for debugging"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Error timestamp"
    )


# Union type for all SSE events
SSEEvent = Union[SSEHeartbeat, SSEProgress, SSEComplete, SSEError]


# =============================================================================
# CONVERSATION HISTORY
# =============================================================================


class ConversationHistoryRequest(BaseModel):
    """Request parameters for conversation history."""
    
    session_id: UUID = Field(
        ...,
        description="Session to fetch history for"
    )
    
    limit: int = Field(
        20,
        description="Maximum items to return",
        ge=1,
        le=100
    )
    
    offset: int = Field(
        0,
        description="Pagination offset",
        ge=0
    )
    
    include_audio: bool = Field(
        False,
        description="Whether to include audio data (increases response size)"
    )


class ConversationHistoryItem(BaseModel):
    """Single conversation in history."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(..., description="Conversation ID")
    
    user_input_text: str = Field(..., description="User's message")
    
    ai_response_text: str = Field(..., description="AI's response")
    
    detected_emotion: str = Field(..., description="Detected emotion")
    
    sentiment: str = Field(..., description="Sentiment")
    
    created_at: datetime = Field(..., description="Timestamp")
    
    has_audio: bool = Field(
        True,
        description="Whether audio is available"
    )
    
    response_duration_seconds: Optional[float] = Field(
        None,
        description="Response audio duration"
    )
    
    tts_speaker: Optional[str] = Field(
        None,
        description="TTS speaker used"
    )
    
    # Optional audio (only if include_audio=True in request)
    response_audio_base64: Optional[str] = Field(
        None,
        description="Audio data (if requested)"
    )


class ConversationHistoryResponse(BaseModel):
    """Paginated conversation history response."""
    
    items: List[ConversationHistoryItem] = Field(
        ...,
        description="Conversation items"
    )
    
    total_count: int = Field(
        ...,
        description="Total conversations in session",
        ge=0
    )
    
    session_id: UUID = Field(
        ...,
        description="Session identifier"
    )
    
    has_more: bool = Field(
        ...,
        description="Whether more items exist"
    )


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""
    
    title: Optional[str] = Field(
        None,
        description="Optional session title",
        max_length=200
    )
    
    initial_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Initial session context"
    )


class CreateSessionResponse(BaseModel):
    """Response with new session details."""
    
    session_id: UUID = Field(..., description="New session ID")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    
    message: str = Field(
        "Session created successfully",
        description="Status message"
    )


class ListSessionsResponse(BaseModel):
    """List of user's chat sessions."""
    
    sessions: List[SessionInfo] = Field(
        ...,
        description="User's sessions"
    )
    
    total_count: int = Field(
        ...,
        description="Total session count",
        ge=0
    )
