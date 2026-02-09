"""
Quantum Neuron Models
Quantum neuron states, history, and state management
"""

import enum
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Index,
    UniqueConstraint,
    Enum,
    CheckConstraint,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, BYTEA, TSVECTOR
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base import Base, TimestampMixin, UUIDMixin, AuditMixin, MetadataMixin


class QuantumState(str, enum.Enum):
    """Quantum neuron state types"""

    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    COHERENT = "coherent"
    DECOHERENT = "decoherent"
    TUNNELING = "tunneling"


class NeuronType(str, enum.Enum):
    """Quantum neuron types"""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    MEMORY = "memory"
    ATTENTION = "attention"
    RECURRENT = "recurrent"


class QuantumNeuron(Base, UUIDMixin, TimestampMixin, AuditMixin, MetadataMixin):
    """
    Current quantum neuron state
    Stores the latest state of each neuron in the quantum network
    """

    __tablename__ = "quantum_neurons"
    __table_args__ = (
        Index("ix_quantum_neurons_network", "network_id"),
        Index("ix_quantum_neurons_state", "current_state"),
        Index("ix_quantum_neurons_type_state", "neuron_type", "current_state"),
        Index("ix_quantum_neurons_coherence", "coherence_score"),
        Index("ix_quantum_neurons_active", "is_active", "current_state"),
        UniqueConstraint("neuron_id", "network_id", name="uq_neuron_network"),
        CheckConstraint("coherence_score >= 0 AND coherence_score <= 1", name="ck_coherence_range"),
        CheckConstraint("energy_level >= 0", name="ck_energy_positive"),
        {"comment": "Current quantum neuron states"},
    )

    # Identification
    neuron_id = Column(String(100), nullable=False, comment="Unique neuron identifier")
    network_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reality_networks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent reality network",
    )

    # Classification
    neuron_type = Column(
        Enum(NeuronType, name="neuron_type_enum"),
        nullable=False,
        default=NeuronType.HIDDEN,
        comment="Neuron classification",
    )
    layer_index = Column(Integer, nullable=False, default=0, comment="Layer position in network")

    # Quantum state
    current_state = Column(
        Enum(QuantumState, name="quantum_state_enum"),
        nullable=False,
        default=QuantumState.SUPERPOSITION,
        comment="Current quantum state",
    )
    previous_state = Column(
        Enum(QuantumState, name="quantum_state_enum_prev"),
        nullable=True,
        comment="Previous quantum state",
    )

    # Quantum properties
    coherence_score = Column(
        Numeric(10, 9), nullable=False, default=0.5, comment="Quantum coherence level (0-1)"
    )
    energy_level = Column(
        Numeric(20, 10), nullable=False, default=0.0, comment="Neuron energy level"
    )
    phase_angle = Column(Numeric(10, 6), nullable=True, comment="Quantum phase angle in radians")

    # State vector (compressed quantum state)
    state_vector = Column(BYTEA, nullable=True, comment="Serialized quantum state vector")
    state_vector_dims = Column(ARRAY(Integer), nullable=True, comment="State vector dimensions")

    # Entanglement
    entangled_with = Column(
        ARRAY(UUID(as_uuid=True)), default=list, nullable=False, comment="IDs of entangled neurons"
    )
    entanglement_strength = Column(
        Numeric(10, 9), nullable=True, comment="Entanglement correlation strength"
    )

    # Activation
    activation_value = Column(
        Numeric(10, 9), nullable=True, comment="Current activation (after measurement)"
    )
    activation_probability = Column(
        Numeric(10, 9), nullable=True, comment="Probability of activation"
    )

    # Status
    is_active = Column(Boolean, default=True, nullable=False, comment="Neuron active status")
    last_measured_at = Column(
        DateTime(timezone=True), nullable=True, comment="Last measurement timestamp"
    )
    measurement_count = Column(
        Integer, default=0, nullable=False, comment="Total measurements performed"
    )

    # Performance metrics
    spike_rate = Column(Numeric(10, 4), nullable=True, comment="Spiking frequency (Hz)")
    learning_rate = Column(Numeric(10, 9), nullable=True, comment="Adaptive learning rate")

    # Configuration
    quantum_params = Column(
        JSONB, default=dict, nullable=False, comment="Quantum-specific parameters"
    )

    # Full-text search
    search_vector = Column(TSVECTOR, nullable=True, comment="Full-text search vector")

    # Relationships
    network = relationship("RealityNetwork", back_populates="neurons")
    history = relationship(
        "QuantumNeuronHistory",
        back_populates="neuron",
        order_by="QuantumNeuronHistory.created_at.desc()",
    )

    @validates("coherence_score")
    def validate_coherence(self, key, value):
        """Ensure coherence is between 0 and 1"""
        if value is not None:
            return max(0.0, min(1.0, float(value)))
        return value

    def __repr__(self):
        return f"<QuantumNeuron(id={self.neuron_id}, state={self.current_state})>"


class QuantumNeuronHistory(Base, UUIDMixin, TimestampMixin):
    """
    Historical quantum neuron states
    Time-series data with partitioning by date
    """

    __tablename__ = "quantum_neuron_history"
    __table_args__ = (
        Index("ix_qn_history_neuron", "neuron_id"),
        Index("ix_qn_history_state", "state"),
        Index("ix_qn_history_created", "created_at"),
        Index("ix_qn_history_neuron_created", "neuron_id", "created_at"),
        Index("ix_qn_history_coherence", "coherence_score", "created_at"),
        {
            "comment": "Historical quantum neuron states (partitioned by week)",
            "postgresql_partition_by": "RANGE (created_at)",
        },
    )

    neuron_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quantum_neurons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to quantum neuron",
    )

    # Snapshot data
    state = Column(
        Enum(QuantumState, name="quantum_state_hist_enum"),
        nullable=False,
        comment="Quantum state at this point",
    )
    coherence_score = Column(Numeric(10, 9), nullable=False, comment="Coherence at this point")
    energy_level = Column(Numeric(20, 10), nullable=False, comment="Energy level at this point")
    phase_angle = Column(Numeric(10, 6), nullable=True, comment="Phase angle at this point")

    # State snapshot
    state_vector = Column(BYTEA, nullable=True, comment="Serialized state vector")
    activation_value = Column(Numeric(10, 9), nullable=True, comment="Activation at this point")

    # Event metadata
    event_type = Column(String(50), nullable=False, comment="Type of state change event")
    event_description = Column(Text, nullable=True, comment="Description of the event")
    trigger_source = Column(String(100), nullable=True, comment="Source that triggered the change")

    # Performance snapshot
    metrics_snapshot = Column(
        JSONB, default=dict, nullable=False, comment="Performance metrics snapshot"
    )

    # Relationships
    neuron = relationship("QuantumNeuron", back_populates="history")

    def __repr__(self):
        return f"<QuantumNeuronHistory(neuron={self.neuron_id}, state={self.state})>"


class QuantumStateTransition(Base, UUIDMixin, TimestampMixin):
    """
    Track quantum state transitions
    Used for analyzing state evolution patterns
    """

    __tablename__ = "quantum_state_transitions"
    __table_args__ = (
        Index("ix_qst_neuron", "neuron_id"),
        Index("ix_qst_from_to", "from_state", "to_state"),
        Index("ix_qst_probability", "transition_probability"),
        Index("ix_qst_created", "created_at"),
        {"comment": "Quantum state transition records"},
    )

    neuron_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quantum_neurons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Neuron that transitioned",
    )

    from_state = Column(
        Enum(QuantumState, name="qst_from_state_enum"), nullable=False, comment="Source state"
    )
    to_state = Column(
        Enum(QuantumState, name="qst_to_state_enum"), nullable=False, comment="Target state"
    )

    transition_probability = Column(
        Numeric(10, 9), nullable=False, comment="Probability of this transition"
    )
    transition_time = Column(
        Numeric(20, 10), nullable=True, comment="Time taken for transition (seconds)"
    )

    # Context
    trigger_event = Column(String(100), nullable=True, comment="Event that triggered transition")
    network_conditions = Column(
        JSONB, default=dict, nullable=False, comment="Network conditions during transition"
    )

    def __repr__(self):
        return f"<QuantumStateTransition({self.from_state.value} -> {self.to_state.value})>"
