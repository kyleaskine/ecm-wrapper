from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin

class Factor(Base, TimestampMixin):
    __tablename__ = "factors"

    id = Column(Integer, primary_key=True, index=True)
    composite_id = Column(Integer, ForeignKey("composites.id"), nullable=False)
    factor = Column(Text, nullable=False)  # Store as string for arbitrary precision
    is_prime = Column(Boolean, default=None, nullable=True)  # NULL until primality tested
    found_by_attempt_id = Column(Integer, ForeignKey("ecm_attempts.id"), nullable=True)
    
    # Relationships
    composite = relationship("Composite")
    attempt = relationship("ECMAttempt")
    
    # Ensure no duplicate factors per composite
    __table_args__ = (
        UniqueConstraint('composite_id', 'factor', name='unique_composite_factor'),
    )