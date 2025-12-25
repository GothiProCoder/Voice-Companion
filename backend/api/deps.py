"""
GuppShupp API Dependencies
==========================

Shared FastAPI dependencies for database sessions, authentication,
request context, and rate limiting.

Usage:
    from backend.api.deps import get_db, get_current_user, get_request_context
    
    @router.post("/endpoint")
    async def endpoint(
        db: Session = Depends(get_db),
        user: User = Depends(get_current_user),
        ctx: RequestContext = Depends(get_request_context)
    ):
        ...

Author: GuppShupp Team
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Generator
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from backend.database.database import SessionLocal
from backend.database.models import User
from backend.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY SCHEME
# =============================================================================

# Session token header scheme
session_token_header = APIKeyHeader(
    name="X-Session-Token",
    auto_error=False,
    description="Session token obtained from /auth/login"
)


# =============================================================================
# DATABASE DEPENDENCY
# =============================================================================


def get_db() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup.
    
    Yields:
        SQLAlchemy Session
        
    Example:
        @router.get("/users")
        async def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database error, rolled back: {e}")
        raise
    finally:
        db.close()



# =============================================================================
# REQUEST CONTEXT
# =============================================================================


@dataclass
class RequestContext:
    """
    Request context with tracing information.
    
    Automatically populated for each request and used throughout
    the request lifecycle for logging and tracing.
    
    Attributes:
        request_id: Unique identifier for this request
        user_id: Authenticated user ID (if any)
        session_id: Chat session ID (if provided)
        received_at: Request received timestamp
        client_ip: Client IP address
        user_agent: Client user agent string
    """
    
    request_id: str
    received_at: datetime
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_log_context(self) -> Dict[str, Any]:
        """Get context dict for structured logging."""
        return {
            "request_id": self.request_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "client_ip": self.client_ip,
        }
    
    def elapsed_ms(self) -> int:
        """Get elapsed time since request received."""
        return int((datetime.now(timezone.utc) - self.received_at).total_seconds() * 1000)


def _generate_request_id() -> str:
    """Generate unique request ID with timestamp prefix."""
    timestamp = int(time.time())
    unique = uuid4().hex[:8]
    return f"req_{timestamp}_{unique}"


async def get_request_context(request: Request) -> RequestContext:
    """
    Create request context from incoming request.
    
    Extracts client information and generates unique request ID
    for tracing purposes.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        RequestContext with request metadata
    """
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = _generate_request_id()
    
    # Get client IP (handle proxies)
    client_ip = request.headers.get("X-Forwarded-For")
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else None
    
    # Get user agent
    user_agent = request.headers.get("User-Agent")
    
    context = RequestContext(
        request_id=request_id,
        received_at=datetime.now(timezone.utc),
        client_ip=client_ip,
        user_agent=user_agent[:200] if user_agent else None,  # Truncate long UAs
    )
    
    # Store in request state for access elsewhere
    request.state.context = context
    
    return context


# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(session_token_header),
) -> User:
    """
    Get current authenticated user from session token.
    
    Validates the session token in the X-Session-Token header
    and returns the associated user.
    
    Args:
        request: FastAPI Request
        db: Database session
        token: Session token from header
        
    Returns:
        Authenticated User object
        
    Raises:
        HTTPException 401: If token is missing or invalid
        HTTPException 403: If account is deactivated
        
    Example:
        @router.get("/profile")
        async def get_profile(user: User = Depends(get_current_user)):
            return {"username": user.username}
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please login.",
            headers={"WWW-Authenticate": "X-Session-Token"},
        )
    
    # Look up user by session token
    user = db.query(User).filter(
        User.session_token == token,
        User.is_active == True
    ).first()
    
    if not user:
        logger.warning(f"Invalid session token attempted: {token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please login again.",
            headers={"WWW-Authenticate": "X-Session-Token"},
        )
    
    # Update request context with user ID
    if hasattr(request.state, "context"):
        request.state.context.user_id = user.id
    
    logger.debug(f"Authenticated user: {user.username}")
    return user


async def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(session_token_header),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Use this for endpoints that work with or without authentication,
    but may provide enhanced functionality when authenticated.
    
    Args:
        request: FastAPI Request
        db: Database session
        token: Session token from header
        
    Returns:
        User object if authenticated, None otherwise
        
    Example:
        @router.get("/public-data")
        async def get_data(user: Optional[User] = Depends(get_optional_user)):
            if user:
                return {"data": "personalized"}
            return {"data": "generic"}
    """
    if not token:
        return None
    
    user = db.query(User).filter(
        User.session_token == token,
        User.is_active == True
    ).first()
    
    if user and hasattr(request.state, "context"):
        request.state.context.user_id = user.id
    
    return user


# =============================================================================
# RATE LIMITING
# =============================================================================


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Tracks request counts per IP/user with sliding window.
    For production, consider Redis-based implementation.
    
    Attributes:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
    
    def _cleanup_old_requests(self, key: str, current_time: float):
        """Remove requests outside the current window."""
        cutoff = current_time - self.window_seconds
        self._requests[key] = [
            t for t in self._requests[key] if t > cutoff
        ]
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed for this key.
        
        Args:
            key: Identifier (IP address, user ID, etc.)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        self._cleanup_old_requests(key, current_time)
        
        if len(self._requests[key]) >= self.max_requests:
            return False
        
        self._requests[key].append(current_time)
        return True
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        current_time = time.time()
        self._cleanup_old_requests(key, current_time)
        return max(0, self.max_requests - len(self._requests[key]))
    
    def get_reset_time(self, key: str) -> int:
        """Get seconds until rate limit resets."""
        if not self._requests[key]:
            return 0
        oldest = min(self._requests[key])
        reset_time = oldest + self.window_seconds - time.time()
        return max(0, int(reset_time))


# Global rate limiter instances
_chat_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 req/min
_auth_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)  # 10 req/min


async def check_chat_rate_limit(
    request: Request,
    user: User = Depends(get_current_user)
) -> None:
    """
    Check rate limit for chat endpoints.
    
    Raises:
        HTTPException 429: If rate limit exceeded
    """
    key = str(user.id)
    
    if not _chat_rate_limiter.is_allowed(key):
        reset_time = _chat_rate_limiter.get_reset_time(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
            headers={
                "Retry-After": str(reset_time),
                "X-RateLimit-Limit": str(_chat_rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
            }
        )


async def check_auth_rate_limit(request: Request) -> None:
    """
    Check rate limit for auth endpoints.
    Limits by IP address to prevent brute force.
    
    Raises:
        HTTPException 429: If rate limit exceeded
    """
    # Get client IP
    client_ip = request.headers.get("X-Forwarded-For")
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    if not _auth_rate_limiter.is_allowed(client_ip):
        reset_time = _auth_rate_limiter.get_reset_time(client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many authentication attempts. Try again in {reset_time} seconds.",
            headers={"Retry-After": str(reset_time)}
        )


# =============================================================================
# UTILITY DEPENDENCIES
# =============================================================================


def get_user_id_from_request(request: Request) -> Optional[UUID]:
    """Extract user ID from request context if available."""
    if hasattr(request.state, "context") and request.state.context.user_id:
        return request.state.context.user_id
    return None
