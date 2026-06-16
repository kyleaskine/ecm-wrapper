from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from .config import get_settings

settings = get_settings()

# Create SQLAlchemy engine with tuned pool settings for low-memory environments
# Need enough connections to handle frontend page-load bursts (which can fire
# 10-30+ concurrent API requests from RSC prefetching) plus background workers.
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=5,          # 5 persistent connections (handles steady-state load)
    max_overflow=15,      # Up to 15 extra under burst (20 total, matching uvicorn --limit-concurrency 20
                          # so a full request burst can't starve on connections; overflow conns close on release)
    pool_timeout=10,      # Fail fast (10s) instead of queueing for 30s — prevents cascading pileup
    pool_recycle=1800,    # Recycle connections every 30 min to prevent stale connections
    pool_pre_ping=True,   # Verify connections are alive before using
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Register the after_commit / after_rollback listeners that defer residue
# file deletion until the owning transaction commits. Imported here so the
# listeners attach to the Session class as soon as the DB layer loads.
from .utils import file_cleanup  # noqa: E402,F401

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        # Always rollback before closing to clear any failed/uncommitted transaction.
        # This prevents PendingRollbackError when a connection with a dirty transaction
        # is returned to the pool (e.g., after a request timeout or unhandled exception).
        # rollback() is a no-op if the transaction was already committed.
        db.rollback()
        db.close()
