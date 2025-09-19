import os
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = Field(
        default=os.getenv("DATABASE_URL", "postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed"),
        description="PostgreSQL connection string"
    )
    
    # API
    api_title: str = "ECM Distributed Factorization API"
    api_version: str = "1.0.0"
    api_description: str = "API for coordinating distributed integer factorization"
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    reload: bool = False
    
    # Work assignment
    default_work_timeout_minutes: int = Field(default=60, ge=1, le=1440, description="Work timeout in minutes")
    max_work_items_per_client: int = Field(default=5, ge=1, le=100, description="Max work items per client")
    
    # Security (for future auth)
    secret_key: str = Field(
        default=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        min_length=16,
        description="Secret key for cryptographic operations"
    )
    
    @validator("database_url")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://") and not v.startswith("postgresql+psycopg2://"):
            raise ValueError("database_url must be a PostgreSQL connection string")
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if v == "dev-secret-key-change-in-production":
            import warnings
            warnings.warn("Using default secret key - change for production!", UserWarning)
        return v
    
    class Config:
        env_file = ".env"
        validate_assignment = True

@lru_cache()
def get_settings():
    return Settings()