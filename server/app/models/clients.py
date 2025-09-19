from sqlalchemy import Column, String, Integer, Float, DateTime, func
from .base import Base, TimestampMixin

class Client(Base, TimestampMixin):
    __tablename__ = "clients"

    id = Column(String(255), primary_key=True)  # Client identifier
    machine_name = Column(String(255), nullable=True)
    
    # Machine specifications for work allocation
    cpu_cores = Column(Integer, nullable=True)
    memory_gb = Column(Integer, nullable=True)
    avg_curves_per_hour = Column(Float, nullable=True)  # Performance metric
    
    # Status tracking
    last_seen = Column(DateTime, server_default=func.now(), nullable=False)
    status = Column(String(20), default='unknown', nullable=False)  # 'active', 'idle', 'offline', 'error'