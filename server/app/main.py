import asyncio
import logging
import sys
from typing import cast

# Raise Python 3.11+ integer string conversion limit for large composites
# Default is 4300 digits; ECM works with numbers that can exceed this
sys.set_int_max_str_digits(100000)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ExceptionHandler

from .config import get_settings
from .dependencies import AdminAuthRedirect
from .api.v1.router import v1_router
from .utils.html_helpers import get_unauthorized_redirect_html


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request timeouts and prevent slow requests from accumulating.

    For memory-constrained environments, this prevents a few slow requests
    from tying up all available connections/memory.

    Admin endpoints get a longer timeout since some operations (like t-level
    recalculation) can take longer.
    """

    def __init__(self, app, timeout_seconds: int = 60, admin_timeout_seconds: int = 300):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.admin_timeout_seconds = admin_timeout_seconds

    async def dispatch(self, request: Request, call_next):
        # Admin endpoints get longer timeout (5 min vs 60s)
        path = request.url.path
        if "/admin/" in path:
            timeout = self.admin_timeout_seconds
        else:
            timeout = self.timeout_seconds

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logging.warning(f"Request timeout ({timeout}s): {request.method} {path}")
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout - please try again"}
            )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Schema is owned by Alembic — see Dockerfile CMD which runs
# `alembic upgrade head` before uvicorn. Do NOT call Base.metadata.create_all()
# here; it papers over missing migrations and lets the live schema drift
# silently from migration history.

# Get settings
settings = get_settings()

# Initialize rate limiter with real client IP (behind Cloudflare/reverse proxy)
from .rate_limit import get_real_client_ip
limiter = Limiter(key_func=get_real_client_ip)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

# Add rate limiter to app state
app.state.limiter = limiter
# slowapi types its handler as taking RateLimitExceeded, but starlette types
# the registry as taking Exception, so the narrower signature is not assignable.
# Correct at runtime -- starlette only dispatches this handler for the
# exception class it is registered against.
app.add_exception_handler(
    RateLimitExceeded, cast(ExceptionHandler, _rate_limit_exceeded_handler)
)

# Admin HTML auth redirect: when a dashboard page fails auth, redirect to login
@app.exception_handler(AdminAuthRedirect)
async def admin_auth_redirect_handler(request: Request, exc: AdminAuthRedirect):
    return HTMLResponse(content=get_unauthorized_redirect_html())

# Add CORS middleware for web client access
# Note: allow_credentials=False because we accept requests from any origin
# and don't use cookie-based authentication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (public API)
    allow_credentials=False,  # No cookie-based auth used
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timeout middleware to prevent slow requests from tying up resources
# 60 second timeout is generous but prevents runaway requests
app.add_middleware(TimeoutMiddleware, timeout_seconds=60)

# Include API routers
app.include_router(v1_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ecm-distributed-api"}

@app.get("/")
async def root():
    return {
        "service": "ECM Distributed Factorization API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/robots.txt")
async def robots_txt():
    """Disallow crawlers from expensive detail pages."""
    content = (
        "User-agent: *\n"
        "Disallow: /api/v1/dashboard/composites/\n"
        "Disallow: /api/v1/admin/\n"
        "Allow: /api/v1/dashboard/\n"
        "Allow: /api/v1/dashboard/testing-status\n"
    )
    return Response(content=content, media_type="text/plain")

@app.get("/favicon.ico")
async def favicon():
    """Return a simple SVG favicon"""
    # Simple "E" icon for ECM
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" fill="#667eea"/>
        <text x="50" y="75" font-family="Arial" font-size="70" font-weight="bold" fill="white" text-anchor="middle">E</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")
