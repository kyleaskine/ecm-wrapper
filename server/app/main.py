from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import engine
from .models.base import Base
from .api.v1.router import v1_router

# Create database tables
Base.metadata.create_all(bind=engine)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

# Add CORS middleware for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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