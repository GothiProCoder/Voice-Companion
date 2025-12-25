"""
GuppShupp Authentication Schemas
================================

Pydantic models for user authentication endpoints.
Includes signup, login, logout, and user profile schemas.

Author: GuppShupp Team
"""

import re
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict


# =============================================================================
# VALIDATORS
# =============================================================================


def validate_password_strength(password: str) -> str:
    """
    Validate password meets minimum security requirements.
    
    Requirements:
    - Minimum 6 characters
    - Maximum 128 characters
    """
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long")
    if len(password) > 128:
        raise ValueError("Password must not exceed 128 characters")
    return password


def validate_username(username: str) -> str:
    """
    Validate username format.
    
    Requirements:
    - 3-50 characters
    - Alphanumeric, underscores, hyphens only
    - Must start with a letter
    """
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if len(username) > 50:
        raise ValueError("Username must not exceed 50 characters")
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', username):
        raise ValueError(
            "Username must start with a letter and contain only "
            "letters, numbers, underscores, and hyphens"
        )
    return username


# =============================================================================
# SIGNUP
# =============================================================================


class SignupRequest(BaseModel):
    """
    User registration request.
    
    Example:
        {
            "username": "rahul_sharma",
            "email": "rahul@example.com",
            "password": "securepass123",
            "display_name": "Rahul Sharma"
        }
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "username": "rahul_sharma",
                "email": "rahul@example.com",
                "password": "securepass123",
                "display_name": "Rahul Sharma"
            }
        }
    )
    
    username: str = Field(
        ...,
        description="Unique username (3-50 chars, alphanumeric)",
        min_length=3,
        max_length=50,
        examples=["rahul_sharma", "priya123"]
    )
    
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        examples=["rahul@example.com"]
    )
    
    password: str = Field(
        ...,
        description="Password (min 6 characters)",
        min_length=6,
        max_length=128,
        examples=["securepass123"]
    )
    
    display_name: Optional[str] = Field(
        None,
        description="Display name (optional, defaults to username)",
        max_length=100,
        examples=["Rahul Sharma"]
    )
    
    @field_validator("username")
    @classmethod
    def validate_username_format(cls, v: str) -> str:
        return validate_username(v)
    
    @field_validator("password")
    @classmethod
    def validate_password_format(cls, v: str) -> str:
        return validate_password_strength(v)


# =============================================================================
# LOGIN
# =============================================================================


class LoginRequest(BaseModel):
    """
    User login request.
    
    Supports login via username or email.
    
    Example:
        {
            "username": "rahul_sharma",
            "password": "securepass123"
        }
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "username": "rahul_sharma",
                "password": "securepass123"
            }
        }
    )
    
    username: str = Field(
        ...,
        description="Username or email",
        min_length=3,
        max_length=100,
        examples=["rahul_sharma", "rahul@example.com"]
    )
    
    password: str = Field(
        ...,
        description="User password",
        min_length=1,
        max_length=128
    )


# =============================================================================
# USER RESPONSE
# =============================================================================


class UserResponse(BaseModel):
    """
    User profile response.
    
    Returned after successful authentication or profile fetch.
    Does NOT include sensitive data like password hash.
    
    Example:
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "rahul_sharma",
            "email": "rahul@example.com",
            "display_name": "Rahul Sharma",
            "created_at": "2025-12-24T10:30:00Z",
            "last_login": "2025-12-24T15:30:00Z",
            "is_active": true
        }
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "username": "rahul_sharma",
                "email": "rahul@example.com",
                "display_name": "Rahul Sharma",
                "created_at": "2025-12-24T10:30:00Z",
                "last_login": "2025-12-24T15:30:00Z",
                "is_active": True
            }
        }
    )
    
    id: UUID = Field(
        ...,
        description="Unique user identifier"
    )
    
    username: str = Field(
        ...,
        description="Username"
    )
    
    email: str = Field(
        ...,
        description="Email address"
    )
    
    display_name: Optional[str] = Field(
        None,
        description="Display name"
    )
    
    created_at: datetime = Field(
        ...,
        description="Account creation timestamp"
    )
    
    last_login: Optional[datetime] = Field(
        None,
        description="Last login timestamp"
    )
    
    is_active: bool = Field(
        True,
        description="Whether account is active"
    )


# =============================================================================
# AUTH RESPONSE
# =============================================================================


class AuthResponse(BaseModel):
    """
    Authentication response with session token.
    
    Returned after successful login or signup.
    The session_token should be stored securely and sent
    in the X-Session-Token header for authenticated requests.
    
    Example:
        {
            "user": { ... UserResponse ... },
            "session_token": "abc123def456...",
            "expires_at": "2025-12-25T15:30:00Z",
            "message": "Login successful"
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "username": "rahul_sharma",
                    "email": "rahul@example.com",
                    "display_name": "Rahul Sharma",
                    "created_at": "2025-12-24T10:30:00Z",
                    "is_active": True
                },
                "session_token": "a1b2c3d4e5f6g7h8i9j0",
                "expires_at": "2025-12-25T15:30:00Z",
                "message": "Login successful"
            }
        }
    )
    
    user: UserResponse = Field(
        ...,
        description="User profile data"
    )
    
    session_token: str = Field(
        ...,
        description="Session token for authentication (use in X-Session-Token header)",
        min_length=20,
        max_length=64
    )
    
    expires_at: datetime = Field(
        ...,
        description="Token expiration timestamp"
    )
    
    message: str = Field(
        "Authentication successful",
        description="Success message"
    )


# =============================================================================
# LOGOUT
# =============================================================================


class LogoutResponse(BaseModel):
    """
    Logout confirmation response.
    
    Example:
        {
            "success": true,
            "message": "Logged out successfully"
        }
    """
    
    success: bool = Field(
        True,
        description="Whether logout was successful"
    )
    
    message: str = Field(
        "Logged out successfully",
        description="Status message"
    )


# =============================================================================
# PASSWORD CHANGE
# =============================================================================


class PasswordChangeRequest(BaseModel):
    """
    Password change request (for authenticated users).
    
    Example:
        {
            "current_password": "oldpass123",
            "new_password": "newpass456"
        }
    """
    
    current_password: str = Field(
        ...,
        description="Current password for verification",
        min_length=1
    )
    
    new_password: str = Field(
        ...,
        description="New password",
        min_length=6,
        max_length=128
    )
    
    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return validate_password_strength(v)
