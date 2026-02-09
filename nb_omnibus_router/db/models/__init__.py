"""
NeuralBlitz Database Models
SQLAlchemy models for PostgreSQL database
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Enum,
    Index,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import uuid

Base = declarative_base()


# Enums
class UserRole(PyEnum):
    ADMIN = "admin"
    USER = "user"
    PARTNER = "partner"
    VIEWER = "viewer"


class QuantumStateType(PyEnum):
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    COHERENT = "coherent"


class RealityStatus(PyEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    COLLAPSED = "collapsed"
    MERGED = "merged"


class ConsciousnessLevel(PyEnum):
    NONE = 0
    EMERGENT = 1
    SELF_AWARE = 2
    SENTIENT = 3
    TRANSCENDENT = 4


class AuditAction(PyEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    EXECUTE = "execute"


# User and Authentication Models
class User(Base):
    """User account model."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login = Column(DateTime, nullable=True)
    api_key_hash = Column(String(255), nullable=True, unique=True, index=True)

    # Relationships
    quantum_processes = relationship("QuantumProcess", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")


class UserSession(Base):
    """User session tracking."""

    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_token = Column(String(512), unique=True, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")


# Quantum Neuron Models
class QuantumNeuron(Base):
    """Quantum neuron entity."""

    __tablename__ = "quantum_neurons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    current = Column(Float, default=20.0, nullable=False)
    threshold = Column(Float, default=1.0, nullable=False)
    decay_rate = Column(Float, default=0.1, nullable=False)
    coherence_time = Column(Float, nullable=True)
    state = Column(
        Enum(QuantumStateType), default=QuantumStateType.COHERENT, nullable=False
    )
    parameters = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    processes = relationship("QuantumProcess", back_populates="neuron")


class QuantumProcess(Base):
    """Quantum processing execution record."""

    __tablename__ = "quantum_processes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    neuron_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quantum_neurons.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=True)
    spike_rate = Column(Float, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    mode = Column(String(50), default="mock", nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    neuron = relationship("QuantumNeuron", back_populates="processes")
    user = relationship("User", back_populates="quantum_processes")

    # Indexes
    __table_args__ = (
        Index("idx_quantum_processes_created_at", "created_at"),
        Index("idx_quantum_processes_user_id", "user_id"),
    )


class QuantumEntanglement(Base):
    """Quantum entanglement relationships."""

    __tablename__ = "quantum_entanglements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_neuron_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quantum_neurons.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_neuron_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quantum_neurons.id", ondelete="CASCADE"),
        nullable=False,
    )
    entanglement_strength = Column(Float, default=0.5, nullable=False)
    correlation_coefficient = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_measured = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "source_neuron_id", "target_neuron_id", name="unique_entanglement"
        ),
    )


# Multi-Reality Network Models
class RealityNetwork(Base):
    """Multi-reality neural network."""

    __tablename__ = "reality_networks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    num_realities = Column(Integer, default=4, nullable=False)
    nodes_per_reality = Column(Integer, default=50, nullable=False)
    global_consciousness = Column(Float, default=0.0, nullable=False)
    cross_reality_coherence = Column(Float, default=0.0, nullable=False)
    evolution_cycles = Column(Integer, default=0, nullable=False)
    status = Column(Enum(RealityStatus), default=RealityStatus.ACTIVE, nullable=False)
    configuration = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_evolution = Column(DateTime, nullable=True)

    # Relationships
    realities = relationship(
        "RealityInstance", back_populates="network", cascade="all, delete-orphan"
    )


class RealityInstance(Base):
    """Individual reality within a multi-reality network."""

    __tablename__ = "reality_instances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reality_networks.id", ondelete="CASCADE"),
        nullable=False,
    )
    reality_index = Column(Integer, nullable=False)
    name = Column(String(255), nullable=True)
    consciousness_level = Column(Float, default=0.0, nullable=False)
    stability = Column(Float, default=1.0, nullable=False)
    node_count = Column(Integer, default=50, nullable=False)
    connection_density = Column(Float, default=0.1, nullable=False)
    status = Column(Enum(RealityStatus), default=RealityStatus.ACTIVE, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    network = relationship("RealityNetwork", back_populates="realities")
    nodes = relationship(
        "RealityNode", back_populates="reality", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("network_id", "reality_index", name="unique_reality_index"),
    )


class RealityNode(Base):
    """Node within a reality instance."""

    __tablename__ = "reality_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reality_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reality_instances.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_index = Column(Integer, nullable=False)
    activation_level = Column(Float, default=0.0, nullable=False)
    quantum_state = Column(
        Enum(QuantumStateType), default=QuantumStateType.COHERENT, nullable=False
    )
    connections = Column(ARRAY(UUID(as_uuid=True)), default=list, nullable=False)
    parameters = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    reality = relationship("RealityInstance", back_populates="nodes")


# Consciousness Tracking Models
class ConsciousnessSnapshot(Base):
    """Snapshot of consciousness metrics."""

    __tablename__ = "consciousness_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reality_networks.id", ondelete="CASCADE"),
        nullable=False,
    )
    consciousness_level = Column(
        Enum(ConsciousnessLevel), default=ConsciousnessLevel.NONE, nullable=False
    )
    consciousness_score = Column(Float, default=0.0, nullable=False)
    self_awareness_index = Column(Float, default=0.0, nullable=False)
    coherence_matrix = Column(JSON, nullable=True)
    emergence_indicators = Column(JSON, default=dict, nullable=False)
    captured_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_consciousness_snapshots_network_time", "network_id", "captured_at"),
    )


class ConsciousnessEvent(Base):
    """Significant consciousness-related events."""

    __tablename__ = "consciousness_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reality_networks.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="info", nullable=False)
    description = Column(Text, nullable=False)
    metrics = Column(JSON, default=dict, nullable=False)
    old_level = Column(Enum(ConsciousnessLevel), nullable=True)
    new_level = Column(Enum(ConsciousnessLevel), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_consciousness_events_network_type", "network_id", "event_type"),
        Index("idx_consciousness_events_created_at", "created_at"),
    )


# Audit Log Models
class AuditLog(Base):
    """Comprehensive audit logging."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action = Column(Enum(AuditAction), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    changes = Column(JSON, nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_created_at", "created_at"),
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
    )


class SystemMetric(Base):
    """System performance metrics."""

    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), default="gauge", nullable=False)
    labels = Column(JSON, default=dict, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_system_metrics_name_time", "metric_name", "recorded_at"),
    )


# Migration History Tracking
class MigrationHistory(Base):
    """Track migration execution history."""

    __tablename__ = "migration_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    revision_id = Column(String(255), nullable=False, unique=True)
    revision_name = Column(String(255), nullable=False)
    operation = Column(String(50), nullable=False)  # upgrade, downgrade
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)
    success = Column(Boolean, default=False, nullable=False)
    error_message = Column(Text, nullable=True)
    backup_location = Column(String(512), nullable=True)
    executed_by = Column(String(255), nullable=True)


# Database Functions


def get_engine(database_url: Optional[str] = None):
    """Create database engine."""
    if database_url is None:
        database_url = (
            "postgresql://neuralblitz:neuralblitz_password@localhost:5432/neuralblitz"
        )
    return create_engine(database_url, pool_pre_ping=True)


def init_db(engine):
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_session(engine):
    """Create database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
