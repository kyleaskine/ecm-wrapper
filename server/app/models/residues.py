from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, BigInteger, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .composites import Composite
    from .attempts import ECMAttempt


class ECMResidue(Base, TimestampMixin):
    """
    Tracks ECM stage 1 residue files for decoupled two-stage ECM processing.

    Workflow:
    1. GPU worker completes stage 1, submits results (gets immediate t-level credit)
    2. GPU worker uploads residue file, creating this record
    3. CPU worker requests stage 2 work, claims this residue
    4. CPU worker downloads file, runs stage 2, submits results
    5. CPU worker marks residue complete, file is deleted, stage 1 attempt is superseded
    """
    __tablename__ = "ecm_residues"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    composite_id: Mapped[int] = mapped_column(ForeignKey("composites.id"), nullable=False)

    # Who uploaded this residue
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Link to the original stage 1 ECM attempt
    # This will be marked as superseded when stage 2 completes
    stage1_attempt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ecm_attempts.id"), nullable=True)

    # ECM parameters (parsed from residue file)
    b1: Mapped[int] = mapped_column(BigInteger, nullable=False)
    parametrization: Mapped[int] = mapped_column(nullable=False)  # 0, 1, 2, or 3
    curve_count: Mapped[int] = mapped_column(nullable=False)

    # File storage info
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)  # Filesystem path
    file_size_bytes: Mapped[int] = mapped_column(nullable=False)
    # SHA-256 of file content. Unique: the checksum is an authoritative
    # identity for submission resolution and completion authorization, so
    # two rows must never share one (concurrent identical uploads race past
    # the pre-insert check; the constraint catches the loser).
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Lifecycle status
    # available: ready for stage 2 work
    # claimed: assigned to a client for stage 2
    # completed: stage 2 finished, file deleted
    # expired: unclaimed too long, cleaned up
    status: Mapped[str] = mapped_column(String(20), default='available', nullable=False)

    # Timing
    # expires_at is only set when claimed (claim timeout). None = no time-based expiration.
    # Residues are cleaned up when their composite is factored, not by time.
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    claimed_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Client ID of stage 2 worker
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    composite: Mapped["Composite"] = relationship("Composite")
    stage1_attempt: Mapped[Optional["ECMAttempt"]] = relationship("ECMAttempt", foreign_keys=[stage1_attempt_id])

    @classmethod
    def default_claim_timeout(cls) -> datetime:
        """Default claim timeout: 3 days from now."""
        return datetime.utcnow() + timedelta(days=3)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_ecm_residues_status', 'status'),
        Index('ix_ecm_residues_composite_status', 'composite_id', 'status'),
        Index('ix_ecm_residues_expires_at', 'expires_at'),
        Index('ix_ecm_residues_created_at', 'created_at'),
    )
