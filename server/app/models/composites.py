from typing import Optional
from sqlalchemy import String, Boolean, Text, Float, Index, Computed, text
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin

# Composites /ecm-work can assign (predicate of the partial work indexes)
_ECM_WORK = "is_active AND NOT is_fully_factored AND ecm_progress < 1.0"


class Composite(Base, TimestampMixin):
    __tablename__ = "composites"

    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(Text, nullable=False, unique=True)  # Original mathematical form (e.g., "2^1223-1")
    current_composite: Mapped[str] = mapped_column(Text, nullable=False)  # Current composite being factored (gets smaller as we find factors) - hash index added in migration
    digit_length: Mapped[int] = mapped_column(nullable=False, index=True)

    # SNFS tracking
    has_snfs_form: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Whether number has SNFS polynomial form
    snfs_difficulty: Mapped[Optional[int]] = mapped_column(nullable=True)  # GNFS-equivalent digit count for SNFS numbers

    # Status fields
    is_complete: Mapped[Optional[bool]] = mapped_column(Boolean, default=None, nullable=True)  # Marks composite as sufficiently complete for OPN purposes
    is_fully_factored: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)  # Whether composite is available for work assignment

    # T-level ECM progress tracking
    target_t_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)  # Target t-level to achieve
    current_t_level: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # Current t-level achieved (includes prior_t_level if set)
    prior_t_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)  # T-level from work done before import
    ecm_progress: Mapped[Optional[float]] = mapped_column(
        Float,
        Computed("current_t_level / NULLIF(target_t_level, 0)"),
        nullable=True,
        index=True
    )  # Generated column: current/target ratio (NULL if no target). Read-only.

    # Work priority
    priority: Mapped[int] = mapped_column(default=0, nullable=False, index=True)

    # Add indexes for common queries
    # Note: current_composite has a PostgreSQL hash index created via migration
    # (not defined here because hash indexes are PostgreSQL-specific and break SQLite tests)
    __table_args__ = (
        Index('ix_composites_factored_status', 'is_fully_factored', 'is_complete'),
        Index('ix_composites_t_level_progress', 'target_t_level', 'current_t_level'),
        Index('ix_composites_priority_work', 'priority', 'is_fully_factored'),
        Index('ix_composites_active_status', 'is_active', 'is_fully_factored'),  # For work assignment queries
        # Work assignment (/ecm-work standard and progressive, /p1-work). The
        # queries' filters imply these predicates, so the planner skips the
        # factored composites instead of discarding them row by row.
        Index('ix_composites_ecm_work_standard', 'target_t_level', 'created_at',
              postgresql_where=text(_ECM_WORK)),
        Index('ix_composites_ecm_work_progressive', 'current_t_level', 'target_t_level', 'digit_length',
              postgresql_where=text(_ECM_WORK)),
        Index('ix_composites_p1_work', 'target_t_level', 'created_at',
              postgresql_where=text("is_active AND NOT is_fully_factored")),
        Index('ix_composites_created_at', 'created_at'),  # newest-first listings
    )