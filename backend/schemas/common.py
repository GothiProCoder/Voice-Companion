"""
GuppShupp Common Schemas
========================

Shared Pydantic models used across all API endpoints.
Includes error responses, health checks, and base mixins.

Author: GuppShupp Team
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# GENERIC TYPE FOR PAGINATION
# =============================================================================

T = TypeVar("T")


# =============================================================================
# BASE MIXINS
# =============================================================================


class BaseTimestampMixin(BaseModel):
    """Mixin providing standardized timestamp fields."""
    
    created_at: datetime = Field(
        ...,
        description="Record creation timestamp (UTC)"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp (UTC)"
    )


# =============================================================================
# ERROR RESPONSES
# =============================================================================


class ErrorResponse(BaseModel):
    """
    Standardized error response format.
    
    Used across all API endpoints for consistent error handling.
    The `retryable` flag helps clients decide if they should retry the request.
    
    Example:
        {
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please wait 30 seconds.",
            "request_id": "req_abc123",
            "retryable": true,
            "details": {"retry_after_seconds": 30}
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid audio format. Supported: wav, mp3, webm",
                "request_id": "req_1735039200_abc123",
                "retryable": False,
                "details": {"field": "audio_format", "received": "ogg"}
            }
        }
    )
    
    error: str = Field(
        ...,
        description="Error type identifier (snake_case)",
        examples=["validation_error", "rate_limit_exceeded", "server_error", "auth_failed"]
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        max_length=500
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier for debugging",
        pattern=r"^req_[a-zA-Z0-9_]+$"
    )
    
    retryable: bool = Field(
        ...,
        description="Whether the client should retry this request"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context-specific error details"
    )


# =============================================================================
# HEALTH CHECK RESPONSES
# =============================================================================


class ServiceStatus(BaseModel):
    """Individual service health status."""
    
    name: str = Field(..., description="Service name")
    healthy: bool = Field(..., description="Whether service is operational")
    latency_ms: Optional[int] = Field(None, description="Last check latency in milliseconds")
    message: Optional[str] = Field(None, description="Status message or error")
    last_check: Optional[datetime] = Field(None, description="Last health check timestamp")


class HealthResponse(BaseModel):
    """
    Comprehensive health check response.
    
    Provides detailed status of all system components for monitoring
    and debugging purposes.
    
    Example:
        {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.5,
            "services": {
                "database": {"name": "database", "healthy": true, "latency_ms": 5},
                "gemini": {"name": "gemini", "healthy": true, "latency_ms": 150},
                "whisper": {"name": "whisper", "healthy": true, "message": "Model loaded"},
                "tts": {"name": "tts", "healthy": true, "message": "Model loaded"},
                "memory": {"name": "memory", "healthy": true, "message": "IndicBERT loaded"}
            },
            "timestamp": "2025-12-24T15:30:00Z"
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "services": {
                    "database": {"name": "database", "healthy": True, "latency_ms": 5},
                    "gemini": {"name": "gemini", "healthy": True}
                },
                "timestamp": "2025-12-24T15:30:00Z"
            }
        }
    )
    
    status: str = Field(
        ...,
        description="Overall system status",
        pattern="^(healthy|degraded|unhealthy)$"
    )
    
    version: str = Field(
        ...,
        description="API version string",
        examples=["1.0.0", "1.2.3-beta"]
    )
    
    uptime_seconds: float = Field(
        ...,
        description="Server uptime in seconds",
        ge=0
    )
    
    services: Dict[str, ServiceStatus] = Field(
        ...,
        description="Health status of individual services"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp (UTC)"
    )
    
    environment: Optional[str] = Field(
        None,
        description="Deployment environment",
        examples=["development", "staging", "production"]
    )


# =============================================================================
# PAGINATION
# =============================================================================


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response wrapper.
    
    Provides consistent pagination metadata across all list endpoints.
    """
    
    items: List[T] = Field(
        ...,
        description="List of items for current page"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of items across all pages",
        ge=0
    )
    
    page: int = Field(
        ...,
        description="Current page number (1-indexed)",
        ge=1
    )
    
    page_size: int = Field(
        ...,
        description="Number of items per page",
        ge=1,
        le=100
    )
    
    has_next: bool = Field(
        ...,
        description="Whether there are more pages"
    )
    
    has_previous: bool = Field(
        ...,
        description="Whether there are previous pages"
    )


# =============================================================================
# REQUEST METADATA
# =============================================================================


class RequestMetadata(BaseModel):
    """
    Request context metadata for logging and tracing.
    Automatically populated by middleware.
    """
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )
    
    user_id: Optional[UUID] = Field(
        None,
        description="Authenticated user ID"
    )
    
    session_id: Optional[UUID] = Field(
        None,
        description="Chat session ID"
    )
    
    received_at: datetime = Field(
        ...,
        description="Request received timestamp"
    )
    
    client_ip: Optional[str] = Field(
        None,
        description="Client IP address"
    )
    
    user_agent: Optional[str] = Field(
        None,
        description="Client user agent string"
    )
