"""
GuppShupp API Router
====================

Aggregates all API routers and provides versioned API endpoints.

Author: GuppShupp Team
"""

from fastapi import APIRouter

from backend.api.health import router as health_router
from backend.api.auth import router as auth_router
from backend.api.conversation import router as conversation_router


# =============================================================================
# API ROUTER AGGREGATION
# =============================================================================

# Main API router with version prefix
api_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(conversation_router)


# =============================================================================
# ROUTER INFORMATION
# =============================================================================

def get_router_info():
    """Get information about all registered routes."""
    routes = []
    for route in api_router.routes:
        routes.append({
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, 'methods') else [],
            "name": route.name if hasattr(route, 'name') else None,
        })
    return routes
