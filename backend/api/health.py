"""
GuppShupp Health Check Endpoints
================================

Health and readiness endpoints for monitoring and orchestration.
Provides detailed service status for debugging and alerting.

Endpoints:
    GET /health - Basic liveness check
    GET /health/ready - Full readiness check with service status
    GET /health/services - Detailed service status

Author: GuppShupp Team
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.api.deps import get_db
from backend.schemas.common import HealthResponse, ServiceStatus
from backend.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

# Track server start time for uptime calculation
_server_start_time = time.time()


# =============================================================================
# HEALTH CHECK UTILITIES
# =============================================================================


async def check_database_health(db: Session) -> ServiceStatus:
    """Check database connectivity and latency."""
    start = time.time()
    try:
        # Simple query to verify connection
        db.execute(text("SELECT 1"))
        latency_ms = int((time.time() - start) * 1000)
        
        return ServiceStatus(
            name="database",
            healthy=True,
            latency_ms=latency_ms,
            message="PostgreSQL connected",
            last_check=datetime.now(timezone.utc)
        )
    except Exception as e:
        return ServiceStatus(
            name="database",
            healthy=False,
            latency_ms=None,
            message=f"Connection failed: {str(e)[:100]}",
            last_check=datetime.now(timezone.utc)
        )


async def check_gemini_health() -> ServiceStatus:
    """Check Gemini API availability (lightweight check)."""
    try:
        # Import and check if client is configured
        from google import genai
        
        # Check if API key is configured
        api_key = config.gemini.api_key if hasattr(config, 'gemini') else None
        if not api_key:
            return ServiceStatus(
                name="gemini",
                healthy=False,
                message="API key not configured",
                last_check=datetime.now(timezone.utc)
            )
        
        return ServiceStatus(
            name="gemini",
            healthy=True,
            message="API key configured",
            last_check=datetime.now(timezone.utc)
        )
    except ImportError:
        return ServiceStatus(
            name="gemini",
            healthy=False,
            message="google-genai not installed",
            last_check=datetime.now(timezone.utc)
        )
    except Exception as e:
        return ServiceStatus(
            name="gemini",
            healthy=False,
            message=str(e)[:100],
            last_check=datetime.now(timezone.utc)
        )


async def check_whisper_health() -> ServiceStatus:
    """Check if Whisper ASR model is available."""
    try:
        from backend.services.whisper_asr import WhisperASRService
        
        return ServiceStatus(
            name="whisper",
            healthy=True,
            message="Service module available",
            last_check=datetime.now(timezone.utc)
        )
    except ImportError as e:
        return ServiceStatus(
            name="whisper",
            healthy=False,
            message=f"Import failed: {str(e)[:50]}",
            last_check=datetime.now(timezone.utc)
        )


async def check_tts_health() -> ServiceStatus:
    """Check if TTS service is available."""
    try:
        from backend.services.parler_tts_module import ParlerTTSService
        
        return ServiceStatus(
            name="tts",
            healthy=True,
            message="Service module available",
            last_check=datetime.now(timezone.utc)
        )
    except ImportError as e:
        return ServiceStatus(
            name="tts",
            healthy=False,
            message=f"Import failed: {str(e)[:50]}",
            last_check=datetime.now(timezone.utc)
        )


async def check_memory_health() -> ServiceStatus:
    """Check if IndicBERT memory service is available."""
    try:
        from backend.services.indicbert_memory import IndicBERTMemoryService
        
        return ServiceStatus(
            name="memory",
            healthy=True,
            message="Service module available",
            last_check=datetime.now(timezone.utc)
        )
    except ImportError as e:
        return ServiceStatus(
            name="memory",
            healthy=False,
            message=f"Import failed: {str(e)[:50]}",
            last_check=datetime.now(timezone.utc)
        )


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Basic liveness check",
    description="Simple health check that returns immediately. Use this for load balancer health probes."
)
async def health_check():
    """
    Basic liveness check.
    
    Returns 200 OK if the server is running.
    Does not check external dependencies.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Full readiness check",
    description="Comprehensive health check including all services. Use for deployment readiness."
)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Full readiness check with service status.
    
    Checks all critical services:
    - Database connectivity
    - Gemini API configuration
    - Whisper ASR availability
    - TTS service availability
    - Memory service availability
    
    Returns:
        HealthResponse with detailed service status
    """
    # Check all services
    db_status = await check_database_health(db)
    gemini_status = await check_gemini_health()
    whisper_status = await check_whisper_health()
    tts_status = await check_tts_health()
    memory_status = await check_memory_health()
    
    services = {
        "database": db_status,
        "gemini": gemini_status,
        "whisper": whisper_status,
        "tts": tts_status,
        "memory": memory_status,
    }
    
    # Determine overall status
    critical_services = ["database", "gemini"]
    all_healthy = all(s.healthy for s in services.values())
    critical_healthy = all(services[name].healthy for name in critical_services)
    
    if all_healthy:
        overall_status = "healthy"
    elif critical_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    # Calculate uptime
    uptime_seconds = time.time() - _server_start_time
    
    return HealthResponse(
        status=overall_status,
        version=getattr(config, 'version', '1.0.0'),
        uptime_seconds=uptime_seconds,
        services=services,
        timestamp=datetime.now(timezone.utc),
        environment=getattr(config, 'environment', 'development')
    )


@router.get(
    "/services",
    response_model=Dict[str, ServiceStatus],
    summary="Detailed service status",
    description="Get detailed status for each individual service."
)
async def service_status(db: Session = Depends(get_db)):
    """Get detailed status of all services."""
    return {
        "database": await check_database_health(db),
        "gemini": await check_gemini_health(),
        "whisper": await check_whisper_health(),
        "tts": await check_tts_health(),
        "memory": await check_memory_health(),
    }


@router.get(
    "/ping",
    response_model=Dict[str, str],
    summary="Quick ping endpoint",
    description="Ultra-lightweight endpoint for network latency testing."
)
async def ping():
    """Quick ping for latency testing."""
    return {"pong": datetime.now(timezone.utc).isoformat()}
