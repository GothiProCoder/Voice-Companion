"""
GuppShupp Authentication Endpoints
==================================

User authentication endpoints for signup, login, logout, and profile management.
Uses bcrypt for password hashing and simple session tokens for auth.

Endpoints:
    POST /auth/signup - Register new user
    POST /auth/login - Authenticate and get session token
    POST /auth/logout - Invalidate session token
    GET /auth/me - Get current user profile
    PUT /auth/profile - Update user profile

Author: GuppShupp Team
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.api.deps import (
    get_db,
    get_current_user,
    get_request_context,
    check_auth_rate_limit,
    RequestContext,
)
from backend.database.models import User
from backend.schemas.auth import (
    SignupRequest,
    LoginRequest,
    UserResponse,
    AuthResponse,
    LogoutResponse,
    PasswordChangeRequest,
)
from backend.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# PASSWORD UTILITIES
# =============================================================================


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt with secure settings.
    
    Args:
        password: Plain text password
        
    Returns:
        bcrypt hash string
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against bcrypt hash.
    
    Args:
        password: Plain text password to verify
        hashed: bcrypt hash to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def generate_session_token() -> str:
    """Generate cryptographically secure session token."""
    return secrets.token_hex(32)  # 64 character hex string


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

# Session token expiry (24 hours)
SESSION_EXPIRY_HOURS = 24


def create_session(user: User, db: Session) -> tuple[str, datetime]:
    """
    Create new session for user.
    
    Args:
        user: User object
        db: Database session
        
    Returns:
        Tuple of (session_token, expires_at)
    """
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_EXPIRY_HOURS)
    
    # Update user with new session token
    user.session_token = token
    user.last_login = datetime.now(timezone.utc)
    user.last_active = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"Session created for user: {user.username}")
    return token, expires_at


def invalidate_session(user: User, db: Session) -> None:
    """Invalidate user's current session."""
    user.session_token = None
    db.commit()
    logger.info(f"Session invalidated for user: {user.username}")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with username, email, and password.",
    responses={
        201: {"description": "User created successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    }
)
async def signup(
    request: SignupRequest,
    req: Request,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context),
    _: None = Depends(check_auth_rate_limit),
):
    """
    Register a new user account.
    
    - **username**: Unique username (3-50 chars, alphanumeric + underscore/hyphen)
    - **email**: Valid email address
    - **password**: Password (min 6 characters)
    - **display_name**: Optional display name
    
    Returns session token for immediate login after signup.
    """
    logger.info(f"[{ctx.request_id}] Signup attempt for username: {request.username}")
    
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken. Please choose a different one."
        )
    
    # Check if email already exists
    if request.email:
        existing_email = db.query(User).filter(User.email == request.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered. Please use a different email or login."
            )
    
    # Create new user
    try:
        hashed_password = hash_password(request.password)
        
        new_user = User(
            username=request.username,
            email=request.email,
            password_hash=hashed_password,
            display_name=request.display_name or request.username,
            is_active=True,
            preferred_language='hi',  # Default to Hindi
            preferences={},
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"[{ctx.request_id}] User created: {new_user.username} ({new_user.id})")
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"[{ctx.request_id}] Signup integrity error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists."
        )
    except Exception as e:
        db.rollback()
        logger.error(f"[{ctx.request_id}] Signup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account. Please try again."
        )
    
    # Create session (auto-login after signup)
    token, expires_at = create_session(new_user, db)
    
    return AuthResponse(
        user=UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            display_name=new_user.display_name,
            created_at=new_user.created_at,
            last_login=new_user.last_login,
            is_active=new_user.is_active,
        ),
        session_token=token,
        expires_at=expires_at,
        message="Account created successfully! Welcome to GuppShupp!"
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login user",
    description="Authenticate with username/email and password.",
    responses={
        200: {"description": "Login successful"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    }
)
async def login(
    request: LoginRequest,
    req: Request,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context),
    _: None = Depends(check_auth_rate_limit),
):
    """
    Authenticate user and get session token.
    
    - **username**: Username or email
    - **password**: User's password
    
    Returns session token to be used in X-Session-Token header.
    """
    logger.info(f"[{ctx.request_id}] Login attempt for: {request.username}")
    
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == request.username) | (User.email == request.username)
    ).first()
    
    if not user:
        logger.warning(f"[{ctx.request_id}] Login failed - user not found: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    
    # Check if account is active
    if not user.is_active:
        logger.warning(f"[{ctx.request_id}] Login failed - account deactivated: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated. Please contact support."
        )
    
    # Verify password
    if not user.password_hash:
        logger.warning(f"[{ctx.request_id}] Login failed - no password set: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    
    if not verify_password(request.password, user.password_hash):
        logger.warning(f"[{ctx.request_id}] Login failed - wrong password: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    
    # Create session
    token, expires_at = create_session(user, db)
    
    logger.info(f"[{ctx.request_id}] Login successful: {user.username}")
    
    return AuthResponse(
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            display_name=user.display_name,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active,
        ),
        session_token=token,
        expires_at=expires_at,
        message="Login successful! Welcome back!"
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout user",
    description="Invalidate current session token.",
)
async def logout(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Logout current user and invalidate session.
    
    Requires valid session token in X-Session-Token header.
    """
    logger.info(f"[{ctx.request_id}] Logout: {user.username}")
    
    invalidate_session(user, db)
    
    return LogoutResponse(
        success=True,
        message="Logged out successfully. See you again!"
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Get profile of currently authenticated user.",
)
async def get_profile(
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get current user's profile.
    
    Requires valid session token in X-Session-Token header.
    """
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        last_login=user.last_login,
        is_active=user.is_active,
    )


@router.put(
    "/profile",
    response_model=UserResponse,
    summary="Update user profile",
    description="Update current user's profile settings.",
)
async def update_profile(
    display_name: Optional[str] = None,
    preferred_language: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Update current user's profile.
    
    - **display_name**: New display name (optional)
    - **preferred_language**: Preferred language code (optional)
    """
    logger.info(f"[{ctx.request_id}] Profile update for: {user.username}")
    
    if display_name is not None:
        user.display_name = display_name[:100]  # Limit length
    
    if preferred_language is not None:
        user.preferred_language = preferred_language[:10]
    
    user.last_active = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        last_login=user.last_login,
        is_active=user.is_active,
    )


@router.post(
    "/change-password",
    response_model=LogoutResponse,
    summary="Change password",
    description="Change current user's password.",
)
async def change_password(
    request: PasswordChangeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ctx: RequestContext = Depends(get_request_context),
    _: None = Depends(check_auth_rate_limit),
):
    """
    Change password for current user.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (min 6 characters)
    
    After password change, session remains valid.
    """
    logger.info(f"[{ctx.request_id}] Password change for: {user.username}")
    
    # Verify current password
    if not user.password_hash or not verify_password(request.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect."
        )
    
    # Hash and set new password
    user.password_hash = hash_password(request.new_password)
    db.commit()
    
    logger.info(f"[{ctx.request_id}] Password changed successfully: {user.username}")
    
    return LogoutResponse(
        success=True,
        message="Password changed successfully!"
    )
