from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Text, Index, BigInteger
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from datetime import datetime, timedelta

class WorkAssignment(Base, TimestampMixin):
    __tablename__ = "work_assignments"

    id = Column(String(64), primary_key=True)  # UUID for work assignment
    composite_id = Column(Integer, ForeignKey("composites.id"), nullable=False)
    client_id = Column(String(255), nullable=False, index=True)

    # Work details
    method = Column(String(50), nullable=False)  # 'ecm', 'pm1', 'pp1'
    b1 = Column(BigInteger, nullable=False)
    b2 = Column(BigInteger, nullable=True)
    curves_requested = Column(Integer, nullable=False)

    # Status tracking
    status = Column(String(20), default='assigned', nullable=False)  # 'assigned', 'claimed', 'running', 'completed', 'failed', 'timeout'
    priority = Column(Integer, default=0, nullable=False, index=True)  # Higher = more priority

    # Timing
    assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    claimed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Progress tracking
    curves_completed = Column(Integer, default=0, nullable=False)
    progress_message = Column(Text, nullable=True)
    last_progress_at = Column(DateTime, nullable=True)

    # Relationships
    composite = relationship("Composite")

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

    def extend_deadline(self, minutes: int = None):
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
    )