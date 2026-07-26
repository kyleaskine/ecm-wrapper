"""
Pytest configuration and fixtures for server tests.

Provides:
- In-memory SQLite database for fast isolated tests
- FastAPI test client
- Helper functions for test data setup
"""
import sys
from contextlib import contextmanager
from pathlib import Path

# Add server directory to Python path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.models.base import Base
from app.database import get_db
from app.main import app

# Import all models to ensure they're registered with Base.metadata
from app.models.composites import Composite
from app.models.projects import Project, ProjectComposite
from app.models.attempts import ECMAttempt
from app.models.factors import Factor
from app.models.work_assignments import WorkAssignment
from app.models.clients import Client
from app.models.residues import ECMResidue


# Module-level shared engine using StaticPool
_engine = None
_TestingSessionLocal = None


def get_test_engine():
    """Get or create the shared test engine."""
    global _engine, _TestingSessionLocal
    if _engine is None:
        _engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        # Enable foreign keys for SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    assert _TestingSessionLocal is not None
    return _engine, _TestingSessionLocal


@pytest.fixture(scope="function", autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state before each test.

    The limiters are process-global and key on the shared 'testclient' IP,
    so without a reset the suite's cumulative submissions trip the
    30/minute limit on /submit_result partway through a run.
    """
    from app.api.v1 import submit as submit_module
    from app.api.v1 import web as web_module
    from app.main import limiter as app_limiter

    for limiter in (submit_module.limiter, web_module.limiter, app_limiter):
        limiter.reset()
    yield


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Reset database before each test."""
    engine, _ = get_test_engine()

    # Drop all tables explicitly (in reverse dependency order)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ecm_residues"))
        conn.execute(text("DROP TABLE IF EXISTS factors"))
        conn.execute(text("DROP TABLE IF EXISTS work_assignments"))
        conn.execute(text("DROP TABLE IF EXISTS ecm_attempts"))
        conn.execute(text("DROP TABLE IF EXISTS project_composites"))
        conn.execute(text("DROP TABLE IF EXISTS projects"))
        conn.execute(text("DROP TABLE IF EXISTS clients"))
        conn.execute(text("DROP TABLE IF EXISTS composites"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield

    # Cleanup after test
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ecm_residues"))
        conn.execute(text("DROP TABLE IF EXISTS factors"))
        conn.execute(text("DROP TABLE IF EXISTS work_assignments"))
        conn.execute(text("DROP TABLE IF EXISTS ecm_attempts"))
        conn.execute(text("DROP TABLE IF EXISTS project_composites"))
        conn.execute(text("DROP TABLE IF EXISTS projects"))
        conn.execute(text("DROP TABLE IF EXISTS clients"))
        conn.execute(text("DROP TABLE IF EXISTS composites"))
        conn.commit()


@pytest.fixture(scope="function")
def db_session():
    """Create a database session for testing."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    """Create a FastAPI test client with database override."""
    _, TestingSessionLocal = get_test_engine()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@contextmanager
def admin_auth():
    """Satisfy admin authentication for the duration of the block.

    Overrides both admin dependencies: verify_admin_key (JSON endpoints,
    401 on failure) and verify_admin_key_html (HTML pages, which raise
    AdminAuthRedirect instead). Restores any previous override on exit.
    """
    from app.dependencies import verify_admin_key, verify_admin_key_html

    previous = {
        dep: app.dependency_overrides.get(dep)
        for dep in (verify_admin_key, verify_admin_key_html)
    }
    for dep in previous:
        app.dependency_overrides[dep] = lambda: True
    try:
        yield
    finally:
        for dep, prior in previous.items():
            if prior is None:
                app.dependency_overrides.pop(dep, None)
            else:
                app.dependency_overrides[dep] = prior


@pytest.fixture(scope="function")
def admin_client(client):
    """TestClient with admin authentication satisfied for the whole test."""
    with admin_auth():
        yield client


def create_work_assignment(composite_id: int, client_id: str,
                           method: str = "ecm",
                           work_id: str = "wa-test-0001",
                           b1: int = 50000, b2: int = 5000000,
                           curves: int = 100,
                           status: str = "assigned") -> str:
    """Helper to create a work assignment in the test database.

    Note: at most one ACTIVE (assigned/claimed/running) assignment may exist
    per composite - enforced by a partial unique index. Pass an inactive
    status (e.g. 'completed') to stack multiple assignments on one composite.
    """
    from datetime import datetime, timedelta
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        assignment = WorkAssignment(
            id=work_id,
            composite_id=composite_id,
            client_id=client_id,
            method=method,
            b1=b1,
            b2=b2,
            curves_requested=curves,
            expires_at=datetime.utcnow() + timedelta(days=1),
            status=status,
        )
        db.add(assignment)
        db.commit()
        return work_id
    finally:
        db.close()


def create_composite(number: str, **kwargs) -> dict:
    """Helper to create a composite in the test database."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        composite = Composite(
            number=number,
            current_composite=kwargs.get("current_composite", number),
            digit_length=kwargs.get("digit_length", len(number)),
            target_t_level=kwargs.get("target_t_level", 20.0),
            current_t_level=kwargs.get("current_t_level", 0.0),
            priority=kwargs.get("priority", 5),
            is_active=kwargs.get("is_active", True),
        )
        db.add(composite)
        db.commit()
        db.refresh(composite)
        result = {
            "id": composite.id,
            "number": composite.number,
            "current_composite": composite.current_composite,
            "digit_length": composite.digit_length,
        }
        return result
    finally:
        db.close()
