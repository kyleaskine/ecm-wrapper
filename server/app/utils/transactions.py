"""
Transaction management utilities for database operations.

This module provides utilities for managing database transactions consistently
across the application, ensuring data integrity and proper rollback handling.

Key concepts:
- Services should NOT commit - they use db.flush() to make changes visible
- Routes control transaction boundaries using these utilities
- All-or-nothing guarantee for multi-step operations
"""

import logging
from contextlib import contextmanager
from typing import Callable, TypeVar, Any, Optional
from functools import wraps

from sqlalchemy.orm import Session
from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar('T')


@contextmanager
def transaction_scope(db: Session, logger_name: Optional[str] = None):
    """
    Context manager for database transactions with automatic rollback on error.

    Usage:
        with transaction_scope(db, "bulk_upload") as tx:
            # Do multiple operations
            service.create_composite(db, number1)
            service.create_composite(db, number2)
            # Commit happens automatically on success
            # Rollback happens automatically on exception

    Args:
        db: SQLAlchemy database session
        logger_name: Optional name for logging context

    Yields:
        The database session

    Example:
        with transaction_scope(db, "create_work") as tx:
            composite = composite_service.get_or_create_composite(db, number)
            work = work_service.create_work_assignment(db, composite.id)
            # Both operations committed together
    """
    log = logging.getLogger(logger_name) if logger_name else logger

    try:
        yield db
        db.commit()
        log.debug("Transaction committed successfully")
    except HTTPException:
        # HTTPExceptions should propagate with their status codes
        db.rollback()
        log.warning("Transaction rolled back due to HTTPException")
        raise
    except ValueError as e:
        # Validation errors - safe to expose to client
        db.rollback()
        log.warning(f"Transaction rolled back due to validation error: {e}")
        raise
    except Exception as e:
        # Unexpected errors - rollback and log
        db.rollback()
        log.error(f"Transaction rolled back due to error: {type(e).__name__}: {e}", exc_info=True)
        raise


@contextmanager
def batch_transaction_scope(
    db: Session,
    batch_size: int = 100,
    logger_name: Optional[str] = None
):
    """
    Context manager for batch operations with periodic commits.

    Use this for bulk operations where you want to commit every N items
    to avoid long-running transactions, while still maintaining some
    atomicity guarantees.

    Usage:
        with batch_transaction_scope(db, batch_size=100) as batch:
            for i, item in enumerate(items):
                process_item(db, item)

                # Commit every batch_size items
                if (i + 1) % batch_size == 0:
                    batch.checkpoint()

            # Final commit for remaining items
            batch.finalize()

    Args:
        db: SQLAlchemy database session
        batch_size: Number of items to process before committing
        logger_name: Optional name for logging context

    Yields:
        BatchContext object with checkpoint() and finalize() methods
    """
    log = logging.getLogger(logger_name) if logger_name else logger

    class BatchContext:
        def __init__(self, session: Session):
            self.session = session
            self.committed_count = 0
            self.error_count = 0

        def checkpoint(self):
            """Commit the current batch of changes."""
            try:
                self.session.commit()
                self.committed_count += 1
                log.debug(f"Batch checkpoint {self.committed_count} committed")
            except Exception as e:
                self.session.rollback()
                self.error_count += 1
                log.error(f"Batch checkpoint failed: {e}")
                raise

        def finalize(self):
            """Final commit for any remaining changes."""
            try:
                self.session.commit()
                log.info(f"Batch finalized: {self.committed_count} checkpoints committed")
            except Exception as e:
                self.session.rollback()
                log.error(f"Batch finalization failed: {e}")
                raise

    ctx = BatchContext(db)
    try:
        yield ctx
    except Exception as e:
        db.rollback()
        log.error(f"Batch operation failed with {ctx.error_count} errors: {e}")
        raise


def transactional(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for route handlers that need transaction management.

    This decorator wraps the entire function in a transaction, automatically
    committing on success and rolling back on any exception.

    Usage:
        @router.post("/composites")
        @transactional
        async def create_composite(
            data: CreateRequest,
            db: Session = Depends(get_db)
        ):
            # Multiple service calls in one transaction
            composite = composite_service.create(db, data)
            project = project_service.add_to_project(db, composite.id)
            return composite

    Args:
        func: The route handler function to wrap

    Returns:
        Wrapped function with transaction management

    Note:
        The function must have a 'db' parameter of type Session.
        Commits happen automatically on success.
        Rollbacks happen automatically on exception.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract db session from kwargs
        db = kwargs.get('db')
        if db is None:
            # Try to find it in args by checking for Session type
            for arg in args:
                if isinstance(arg, Session):
                    db = arg
                    break

        if db is None:
            raise ValueError(
                f"@transactional decorator requires a 'db' parameter of type Session. "
                f"Function: {func.__name__}"
            )

        try:
            result = await func(*args, **kwargs)
            db.commit()
            logger.debug(f"Transaction committed for {func.__name__}")
            return result
        except HTTPException:
            db.rollback()
            logger.warning(f"Transaction rolled back for {func.__name__} (HTTPException)")
            raise
        except ValueError as e:
            db.rollback()
            logger.warning(f"Transaction rolled back for {func.__name__} (ValueError): {e}")
            raise
        except Exception as e:
            db.rollback()
            logger.error(
                f"Transaction rolled back for {func.__name__} ({type(e).__name__}): {e}",
                exc_info=True
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract db session from kwargs
        db = kwargs.get('db')
        if db is None:
            # Try to find it in args by checking for Session type
            for arg in args:
                if isinstance(arg, Session):
                    db = arg
                    break

        if db is None:
            raise ValueError(
                f"@transactional decorator requires a 'db' parameter of type Session. "
                f"Function: {func.__name__}"
            )

        try:
            result = func(*args, **kwargs)
            db.commit()
            logger.debug(f"Transaction committed for {func.__name__}")
            return result
        except HTTPException:
            db.rollback()
            logger.warning(f"Transaction rolled back for {func.__name__} (HTTPException)")
            raise
        except ValueError as e:
            db.rollback()
            logger.warning(f"Transaction rolled back for {func.__name__} (ValueError): {e}")
            raise
        except Exception as e:
            db.rollback()
            logger.error(
                f"Transaction rolled back for {func.__name__} ({type(e).__name__}): {e}",
                exc_info=True
            )
            raise

    # Return appropriate wrapper based on whether function is async
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class TransactionManager:
    """
    Helper class for managing complex transaction scenarios.

    Use this when you need more control than the decorator or context manager provide.

    Example:
        tx = TransactionManager(db)

        try:
            # Step 1
            composite = tx.execute(composite_service.create, number)

            # Step 2
            project = tx.execute(project_service.create, name)

            # Commit everything
            tx.commit()
        except Exception:
            tx.rollback()
            raise
    """

    def __init__(self, db: Session, auto_commit: bool = False):
        """
        Initialize transaction manager.

        Args:
            db: Database session
            auto_commit: If True, automatically commit after each execute()
        """
        self.db = db
        self.auto_commit = auto_commit
        self.operations: list[str] = []

    def execute(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an operation within the transaction.

        Args:
            operation: Function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation
        """
        op_name = operation.__name__ if hasattr(operation, '__name__') else str(operation)
        self.operations.append(op_name)

        try:
            result = operation(*args, **kwargs)
            if self.auto_commit:
                self.db.commit()
                logger.debug(f"Auto-committed after {op_name}")
            return result
        except Exception as e:
            logger.error(f"Operation {op_name} failed: {e}")
            raise

    def commit(self):
        """Commit the transaction."""
        try:
            self.db.commit()
            logger.info(f"Transaction committed: {len(self.operations)} operations")
        except Exception as e:
            logger.error(f"Commit failed after operations: {self.operations}")
            raise

    def rollback(self):
        """Rollback the transaction."""
        self.db.rollback()
        logger.warning(f"Transaction rolled back: {len(self.operations)} operations undone")


def ensure_no_pending_changes(db: Session) -> None:
    """
    Verify that the session has no uncommitted changes.

    Useful for debugging transaction issues. Call this at the start of
    a function to ensure clean state.

    Args:
        db: Database session to check

    Raises:
        RuntimeError: If session has uncommitted changes
    """
    if db.new or db.dirty or db.deleted:
        new_count = len(db.new)
        dirty_count = len(db.dirty)
        deleted_count = len(db.deleted)

        error_msg = (
            f"Session has uncommitted changes: "
            f"{new_count} new, {dirty_count} modified, {deleted_count} deleted objects. "
            f"This indicates a transaction management bug."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
