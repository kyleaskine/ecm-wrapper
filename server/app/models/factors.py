from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey, Boolean, BigInteger, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .composites import Composite
    from .attempts import ECMAttempt


class Factor(Base, TimestampMixin):
    __tablename__ = "factors"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    composite_id: Mapped[int] = mapped_column(ForeignKey("composites.id"), nullable=False)
    factor: Mapped[str] = mapped_column(Text, nullable=False)  # Store as string for arbitrary precision
    is_prime: Mapped[Optional[bool]] = mapped_column(Boolean, default=None, nullable=True)  # NULL until primality tested
    found_by_attempt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ecm_attempts.id"), nullable=True)
    sigma: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Sigma value that found this factor (ECM only) - Text to support large param 0 values
    group_order: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Elliptic curve group order (ECM only)
    group_order_factorization: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Factorization of group order

    # Relationships
    composite: Mapped["Composite"] = relationship("Composite")
    attempt: Mapped[Optional["ECMAttempt"]] = relationship("ECMAttempt")

    # Ensure no duplicate factors per composite
    __table_args__ = (
        UniqueConstraint('composite_id', 'factor', name='unique_composite_factor'),
        Index('ix_factors_created_at', 'created_at'),  # dashboard 24h counts / recent-first ordering
    )
