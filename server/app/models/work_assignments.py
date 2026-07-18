from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, ForeignKey, DateTime, Text, Index, BigInteger, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .composites import Composite


class WorkAssignment(Base, TimestampMixin):
    __tablename__ = "work_assignments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # UUID for work assignment
    composite_id: Mapped[int] = mapped_column(ForeignKey("composites.id"), nullable=False)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Work details
    method: Mapped[str] = mapped_column(String(50), nullable=False)  # 'ecm', 'pm1', 'pp1'
    b1: Mapped[int] = mapped_column(BigInteger, nullable=False)
    b2: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    curves_requested: Mapped[int] = mapped_column(nullable=False)

    # Status tracking
    status: Mapped[str] = mapped_column(String(20), default='assigned', nullable=False)  # 'assigned', 'claimed', 'running', 'completed', 'failed', 'timeout'
    priority: Mapped[int] = mapped_column(default=0, nullable=False, index=True)  # Higher = more priority

    # Timing
    assigned_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Progress tracking
    curves_completed: Mapped[int] = mapped_column(default=0, nullable=False)
    progress_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_progress_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    composite: Mapped["Composite"] = relationship("Composite")

    @property
    def is_expired(self) -> bool:
        """Check if this work assignment has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def estimated_time_minutes(self) -> int:
        """Estimate completion time based on method and parameters."""
        if self.method == 'ecm':
            # ECM time estimation: roughly 1 minute per curve at B1=50K
            # Scale by B1 value (higher B1 = longer time)
            base_time = self.curves_requested * (self.b1 / 50000)
            return max(5, int(base_time))  # Minimum 5 minutes
        elif self.method in ['pm1', 'pp1']:
            # P-1/P+1 typically longer per attempt
            return max(10, int(self.b1 / 100000))  # Scale with B1
        else:
            return 30  # Default for unknown methods

    def extend_deadline(self, minutes: Optional[int] = None):
        """Extend the work assignment deadline."""
        if minutes is None:
            minutes = self.estimated_time_minutes
        self.expires_at = datetime.utcnow() + timedelta(minutes=minutes)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_work_assignments_client_status', 'client_id', 'status'),
        Index('ix_work_assignments_status_priority', 'status', 'priority'),
        Index('ix_work_assignments_expires_at', 'expires_at'),
        Index('ix_work_assignments_composite_method', 'composite_id', 'method'),
        # At most one active assignment per composite. The work-request
        # endpoints exclude busy composites with NOT EXISTS filters, but those
        # are evaluated against the statement snapshot and can be stale when
        # a concurrent request commits mid-query, so the database must be the
        # final arbiter.
        Index(
            'uq_work_assignments_one_active_per_composite',
            'composite_id',
            unique=True,
            postgresql_where=text("status IN ('assigned', 'claimed', 'running')"),
            sqlite_where=text("status IN ('assigned', 'claimed', 'running')"),
        ),
    )