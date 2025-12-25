"""
GuppShupp API Schemas Package
=============================

Comprehensive Pydantic models for FastAPI request/response validation.
All schemas are production-grade with proper validation, documentation, and examples.

Author: GuppShupp Team
Version: 1.0.0
"""

from backend.schemas.common import (
    ErrorResponse,
    HealthResponse,
    ServiceStatus,
    PaginatedResponse,
    BaseTimestampMixin,
)

from backend.schemas.auth import (
    SignupRequest,
    LoginRequest,
    UserResponse,
    AuthResponse,
    LogoutResponse,
)

from backend.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    SSEHeartbeat,
    SSEProgress,
    SSEComplete,
    SSEError,
    ConversationHistoryRequest,
    ConversationHistoryItem,
    ConversationHistoryResponse,
    SessionInfo,
    PhaseTimings,
)

__all__ = [
    # Common
    "ErrorResponse",
    "HealthResponse", 
    "ServiceStatus",
    "PaginatedResponse",
    "BaseTimestampMixin",
    # Auth
    "SignupRequest",
    "LoginRequest",
    "UserResponse",
    "AuthResponse",
    "LogoutResponse",
    # Conversation
    "ChatRequest",
    "ChatResponse",
    "SSEHeartbeat",
    "SSEProgress", 
    "SSEComplete",
    "SSEError",
    "ConversationHistoryRequest",
    "ConversationHistoryItem",
    "ConversationHistoryResponse",
    "SessionInfo",
    "PhaseTimings",
]
