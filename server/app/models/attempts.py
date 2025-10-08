from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float, BigInteger, DateTime, Index
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
import hashlib

class ECMAttempt(Base, TimestampMixin):
    __tablename__ = "ecm_attempts"

    id = Column(Integer, primary_key=True, index=True)
    composite_id = Column(Integer, ForeignKey("composites.id"), nullable=False)
    client_id = Column(String(255), nullable=False, index=True)
    client_ip = Column(String(45), nullable=True, index=True)  # IPv4 or IPv6
    
    # Method and parameters
    method = Column(String(50), nullable=False)  # 'ecm', 'pm1', 'pp1', 'qs', 'nfs'
    b1 = Column(BigInteger, nullable=False)
    b2 = Column(BigInteger, nullable=True)  # Optional stage 2
    parametrization = Column(Integer, nullable=True)  # ECM parametrization type (0, 1, 2, or 3)
    curves_requested = Column(Integer, nullable=False)
    curves_completed = Column(Integer, nullable=False)
    
    # Duplicate detection hash
    work_hash = Column(String(64), nullable=True, index=True, unique=True)  # SHA-256 hash
    
    # Results
    factor_found = Column(Text, nullable=True)  # NULL if no factor found
    execution_time_seconds = Column(Float, nullable=True)
    
    # Program info
    program = Column(String(50), nullable=False)  # 'gmp-ecm', 'yafu', etc
    program_version = Column(String(50), nullable=True)
    
    # Status and metadata
    status = Column(String(20), default='completed', nullable=False)  # 'running', 'completed', 'failed', 'timeout'
    raw_output = Column(Text, nullable=True)  # Store full output for debugging
    
    # Work assignment fields
    assigned_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    
    # Relationships
    composite = relationship("Composite")
    
    @classmethod
    def generate_work_hash(cls, composite: str, method: str, b1: int, b2: int = None,
                          parametrization: int = None, sigma: int = None, curves: int = None) -> str:
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
    
    def set_work_hash(self, composite_number: str, sigma: int = None):
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
    __table_args__ = (
        Index('ix_ecm_attempts_composite_method', 'composite_id', 'method'),
        Index('ix_ecm_attempts_client_status', 'client_id', 'status'),
        Index('ix_ecm_attempts_factor_found', 'factor_found'),
        Index('ix_ecm_attempts_work_hash', 'work_hash'),
    )