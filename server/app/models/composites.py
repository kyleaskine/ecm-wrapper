from sqlalchemy import Column, Integer, String, Boolean, Text, Float, Index
from .base import Base, TimestampMixin

class Composite(Base, TimestampMixin):
    __tablename__ = "composites"

    id = Column(Integer, primary_key=True, index=True)
    number = Column(Text, nullable=False, unique=True)  # Store as string for arbitrary precision
    digit_length = Column(Integer, nullable=False, index=True)

    # Status fields
    is_prime = Column(Boolean, default=None, nullable=True)  # NULL until determined
    is_fully_factored = Column(Boolean, default=False, nullable=False)

    # T-level ECM progress tracking
    target_t_level = Column(Float, nullable=True, index=True)  # Target t-level to achieve
    current_t_level = Column(Float, default=0.0, nullable=False)  # Current t-level achieved

    # Work priority
    priority = Column(Integer, default=0, nullable=False, index=True)

    # Add indexes for common queries
    __table_args__ = (
        Index('ix_composites_factored_status', 'is_fully_factored', 'is_prime'),
        Index('ix_composites_t_level_progress', 'target_t_level', 'current_t_level'),
        Index('ix_composites_priority_work', 'priority', 'is_fully_factored'),
    )