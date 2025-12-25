"""
GuppShupp API Package
=====================

FastAPI routers and endpoint handlers.

Author: GuppShupp Team
"""

from backend.api.deps import (
    get_db,
    get_current_user,
    get_optional_user,
    RequestContext,
    get_request_context,
)

__all__ = [
    "get_db",
    "get_current_user",
    "get_optional_user",
    "RequestContext",
    "get_request_context",
]
