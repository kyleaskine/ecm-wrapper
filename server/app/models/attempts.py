from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey, Float, BigInteger, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin
import hashlib

if TYPE_CHECKING:
    from .composites import Composite


class ECMAttempt(Base, TimestampMixin):
    __tablename__ = "ecm_attempts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    composite_id: Mapped[int] = mapped_column(ForeignKey("composites.id"), nullable=False)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True, index=True)  # IPv4 or IPv6

    # Method and parameters
    method: Mapped[str] = mapped_column(String(50), nullable=False)  # 'ecm', 'pm1', 'pp1', 'qs', 'nfs'
    b1: Mapped[int] = mapped_column(BigInteger, nullable=False)
    b2: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)  # Optional stage 2
    parametrization: Mapped[Optional[int]] = mapped_column(nullable=True)  # ECM parametrization type (0, 1, 2, or 3)
    curves_requested: Mapped[int] = mapped_column(nullable=False)
    curves_completed: Mapped[int] = mapped_column(nullable=False)

    # Duplicate detection hash
    work_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True, unique=True)  # SHA-256 hash

    # Results
    factor_found: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # NULL if no factor found
    execution_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Program info
    program: Mapped[str] = mapped_column(String(50), nullable=False)  # 'gmp-ecm', 'yafu', etc
    program_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Status and metadata
    status: Mapped[str] = mapped_column(String(20), default='completed', nullable=False)  # 'running', 'completed', 'failed', 'timeout'
    raw_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Store full output for debugging

    # Work assignment fields
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Stage supersession tracking (for decoupled two-stage ECM)
    # When stage 2 completes, it supersedes the stage 1 attempt
    superseded_by: Mapped[Optional[int]] = mapped_column(ForeignKey("ecm_attempts.id"), nullable=True)

    # Residue file checksum (for stage 2 work from residue pool)
    # SHA-256 of the residue file, used to detect orphaned attempts when
    # another client completes the same residue
    residue_checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # Relationships
    composite: Mapped["Composite"] = relationship("Composite")
    superseding_attempt: Mapped[Optional["ECMAttempt"]] = relationship("ECMAttempt", remote_side=[id], uselist=False)

    @classmethod
    def generate_work_hash(cls, composite: str, method: str, b1: int, b2: Optional[int] = None,
                          parametrization: Optional[int] = None, sigma: Optional[int] = None, curves: Optional[int] = None) -> str:
        """Generate a hash to detect duplicate work.

        Only considers work duplicate if parametrization and sigma values are explicitly the same.
        Missing values are treated as unique work (different random seeds).

        Args:
            parametrization: ECM parametrization type (1, 2, or 3)
            sigma: The actual sigma value used
        """
        # Include key parameters that define the work
        hash_input = f"{composite}:{method}:{b1}:{b2 or 0}"

        # Add method-specific parameters - ONLY if sigma is present
        # Missing sigma means different random seeds, so should be unique
        if method.lower() == 'ecm' and sigma is not None:
            hash_input += f":{parametrization or 3}:{sigma}"
        else:
            # For work without sigma (YAFU, failed extraction, etc.),
            # include timestamp to ensure uniqueness
            import time
            hash_input += f":unique:{int(time.time() * 1000000)}"  # microsecond precision

        # For multi-curve work, include curve count to allow different batch sizes
        if curves and curves > 1:
            hash_input += f":curves:{curves}"

        return hashlib.sha256(hash_input.encode()).hexdigest()

    def set_work_hash(self, composite_number: str, sigma: Optional[int] = None):
        """Set the work hash for this attempt.

        Args:
            composite_number: The composite being factored
            sigma: The actual sigma value (not stored in attempts, only used for hashing)
        """
        self.work_hash = self.generate_work_hash(
            composite_number, self.method, self.b1, self.b2,
            self.parametrization, sigma, self.curves_requested
        )

    # Indexes for common queries
    # Note: work_hash index is created automatically via unique=True on the column
    __table_args__ = (
        Index('ix_ecm_attempts_composite_method', 'composite_id', 'method'),
        Index('ix_ecm_attempts_client_status', 'client_id', 'status'),
        Index('ix_ecm_attempts_factor_found', 'factor_found'),
        Index('ix_ecm_attempts_created_at', 'created_at'),  # dashboard 24h counts / recent-first ordering
    )