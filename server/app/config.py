import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, cast

from pydantic import Field, validator
from pydantic_settings import BaseSettings

def read_secret_file(file_path: str) -> Optional[str]:
    """Read secret from file if it exists."""
    try:
        path = Path(file_path)
        if path.exists():
            return path.read_text(encoding='utf-8').strip()
    except Exception:
        pass
    return None

def get_database_url() -> str:
    """Construct database URL from environment or secret files."""
    # Check if full DATABASE_URL is provided (env var or secret file)
    url_file = os.getenv("DATABASE_URL_FILE")
    if url_file:
        url = read_secret_file(url_file)
        if url:
            return url
    if "DATABASE_URL" in os.environ:
        url = os.getenv("DATABASE_URL")
        if url:
            return url

    # Build from components (for Docker secrets)
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ecm_distributed")
    user = os.getenv("POSTGRES_USER", "ecm_user")

    # Try to read password from secret file, fallback to env var
    password_file = os.getenv("POSTGRES_PASSWORD_FILE")
    password = read_secret_file(password_file) if password_file else os.getenv("POSTGRES_PASSWORD", "ecm_password")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

class Settings(BaseSettings):
    # Database
    database_url: str = Field(
        default_factory=get_database_url,
        description="PostgreSQL connection string"
    )

    # Environment ("development" or "production"). When set to "production"
    # the secret_key/admin_api_key validators raise on the default values
    # instead of warning, so a misconfigured deploy fails fast.
    environment: str = Field(
        default=os.getenv("ENV", "development"),
        description="Deployment environment: 'development' or 'production'"
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
    max_work_items_per_client: int = Field(default=12, ge=1, le=100, description="Max work items per client")

    # T-level calculation
    t_level_binary_path: str = Field(
        default=os.getenv("T_LEVEL_BINARY_PATH", "/app/bin/t-level"),
        description="Path to t-level executable binary"
    )
    # "library" computes t-levels in-process with the t-level package and a
    # cache (app/services/t_level_library.py). "binary" spawns
    # t_level_binary_path for every calculation, which was the behaviour before
    # 2026-11; it is kept so a rollback is one environment variable
    # (T_LEVEL_BACKEND=binary), and with it the package is never imported. The
    # binary is also used automatically if the package fails to import, or if
    # its probability database raises an error.
    t_level_backend: Literal["library", "binary"] = Field(
        default="library",
        description="T-level backend: 'library' (in-process, cached) or 'binary' (subprocess)"
    )

    # Residue storage
    residue_storage_path: str = Field(
        default=os.getenv("RESIDUE_STORAGE_PATH", "data/residues"),
        description="Path to store ECM residue files for two-stage decoupling"
    )

    # Trusted reverse proxies. Comma-separated list of IPs whose
    # CF-Connecting-IP / X-Forwarded-For headers we will honor. Requests
    # from anywhere else have those headers ignored (prevents IP spoofing
    # against the rate limiter when uvicorn is reachable outside nginx).
    trusted_proxies: str = Field(
        default=os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1"),
        description="Comma-separated list of trusted proxy IPs"
    )

    @property
    def trusted_proxies_set(self) -> set:
        return {ip.strip() for ip in self.trusted_proxies.split(",") if ip.strip()}

    # Security
    secret_key: str = Field(
        default_factory=lambda: cast(str, (
            (read_secret_file(secret_file) if (secret_file := os.getenv("SECRET_KEY_FILE")) else None)
            or os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        )),
        min_length=16,
        description="Secret key for cryptographic operations"
    )

    admin_api_key: str = Field(
        default_factory=lambda: cast(str, (
            (read_secret_file(api_key_file) if (api_key_file := os.getenv("ADMIN_API_KEY_FILE")) else None)
            or os.getenv("ADMIN_API_KEY", "dev-admin-key-change-in-production")
        )),
        min_length=16,
        description="API key for admin endpoints"
    )

    @validator("database_url")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://") and not v.startswith("postgresql+psycopg2://"):
            raise ValueError("database_url must be a PostgreSQL connection string")
        return v

    @validator("secret_key")
    def validate_secret_key(cls, v, values):
        if v == "dev-secret-key-change-in-production":
            if values.get("environment") == "production":
                raise ValueError(
                    "SECRET_KEY is the default placeholder but ENV=production — "
                    "refusing to start. Set SECRET_KEY (or SECRET_KEY_FILE) to a real value."
                )
            import warnings
            warnings.warn("Using default secret key - change for production!", UserWarning)
        return v

    @validator("admin_api_key")
    def validate_admin_api_key(cls, v, values):
        if v == "dev-admin-key-change-in-production":
            if values.get("environment") == "production":
                raise ValueError(
                    "ADMIN_API_KEY is the default placeholder but ENV=production — "
                    "refusing to start. Set ADMIN_API_KEY (or ADMIN_API_KEY_FILE) to a real value."
                )
            import warnings
            warnings.warn("Using default admin API key - change for production!", UserWarning)
        return v

    class Config:
        env_file = ".env"
        validate_assignment = True

@lru_cache()
def get_settings():
    return Settings()
