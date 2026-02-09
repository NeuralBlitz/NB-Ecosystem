"""2026_02_09_1200-002_quantum_neuron_tables

Quantum neuron tables - Core quantum computing entities

Revision ID: 002
Revises: 001
Create Date: 2026-02-09 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create quantum neuron tables."""

    # Create quantum state type enum
    quantum_state_enum = postgresql.ENUM(
        "SUPERPOSITION", "ENTANGLED", "COLLAPSED", "COHERENT", name="quantumstatetype"
    )
    quantum_state_enum.create(op.get_bind())

    # Create quantum_neurons table
    op.create_table(
        "quantum_neurons",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("current", sa.Float(), server_default="20.0", nullable=False),
        sa.Column("threshold", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("decay_rate", sa.Float(), server_default="0.1", nullable=False),
        sa.Column("coherence_time", sa.Float(), nullable=True),
        sa.Column(
            "state",
            sa.Enum(
                "SUPERPOSITION",
                "ENTANGLED",
                "COLLAPSED",
                "COHERENT",
                name="quantumstatetype",
            ),
            server_default="COHERENT",
            nullable=False,
        ),
        sa.Column("parameters", postgresql.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index("idx_quantum_neurons_name", "quantum_neurons", ["name"])
    op.create_index("idx_quantum_neurons_state", "quantum_neurons", ["state"])
    op.create_index("idx_quantum_neurons_created_at", "quantum_neurons", ["created_at"])
    op.create_index("idx_quantum_neurons_active", "quantum_neurons", ["is_active"])

    # Create quantum_processes table
    op.create_table(
        "quantum_processes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("neuron_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("input_data", postgresql.JSON(), nullable=False),
        sa.Column("output_data", postgresql.JSON(), nullable=True),
        sa.Column("spike_rate", sa.Float(), nullable=True),
        sa.Column("execution_time_ms", sa.Float(), nullable=True),
        sa.Column("mode", sa.String(50), server_default="mock", nullable=False),
        sa.Column("status", sa.String(50), server_default="pending", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["neuron_id"], ["quantum_neurons.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for quantum_processes
    op.create_index(
        "idx_quantum_processes_neuron_id", "quantum_processes", ["neuron_id"]
    )
    op.create_index("idx_quantum_processes_user_id", "quantum_processes", ["user_id"])
    op.create_index(
        "idx_quantum_processes_created_at", "quantum_processes", ["created_at"]
    )
    op.create_index("idx_quantum_processes_status", "quantum_processes", ["status"])
    op.create_index("idx_quantum_processes_mode", "quantum_processes", ["mode"])

    # Create quantum_entanglements table
    op.create_table(
        "quantum_entanglements",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("source_neuron_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_neuron_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "entanglement_strength", sa.Float(), server_default="0.5", nullable=False
        ),
        sa.Column("correlation_coefficient", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("last_measured", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.ForeignKeyConstraint(
            ["source_neuron_id"], ["quantum_neurons.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_neuron_id"], ["quantum_neurons.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_neuron_id", "target_neuron_id", name="unique_entanglement"
        ),
    )

    # Create indexes
    op.create_index(
        "idx_quantum_entanglements_source",
        "quantum_entanglements",
        ["source_neuron_id"],
    )
    op.create_index(
        "idx_quantum_entanglements_target",
        "quantum_entanglements",
        ["target_neuron_id"],
    )
    op.create_index(
        "idx_quantum_entanglements_active", "quantum_entanglements", ["is_active"]
    )

    # Create triggers for updated_at
    op.execute("""
        CREATE TRIGGER update_quantum_neurons_updated_at
            BEFORE UPDATE ON quantum_neurons
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

    # Add comment for documentation
    op.execute(
        "COMMENT ON TABLE quantum_neurons IS 'Core quantum neuron entities for spiking neural network simulation';"
    )
    op.execute(
        "COMMENT ON TABLE quantum_processes IS 'Execution records for quantum processing operations';"
    )
    op.execute(
        "COMMENT ON TABLE quantum_entanglements IS 'Quantum entanglement relationships between neurons';"
    )


def downgrade() -> None:
    """Downgrade schema - Remove quantum neuron tables."""

    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS update_quantum_neurons_updated_at ON quantum_neurons;"
    )

    # Drop quantum_entanglements
    op.drop_index(
        "idx_quantum_entanglements_active", table_name="quantum_entanglements"
    )
    op.drop_index(
        "idx_quantum_entanglements_target", table_name="quantum_entanglements"
    )
    op.drop_index(
        "idx_quantum_entanglements_source", table_name="quantum_entanglements"
    )
    op.drop_table("quantum_entanglements")

    # Drop quantum_processes
    op.drop_index("idx_quantum_processes_mode", table_name="quantum_processes")
    op.drop_index("idx_quantum_processes_status", table_name="quantum_processes")
    op.drop_index("idx_quantum_processes_created_at", table_name="quantum_processes")
    op.drop_index("idx_quantum_processes_user_id", table_name="quantum_processes")
    op.drop_index("idx_quantum_processes_neuron_id", table_name="quantum_processes")
    op.drop_table("quantum_processes")

    # Drop quantum_neurons
    op.drop_index("idx_quantum_neurons_active", table_name="quantum_neurons")
    op.drop_index("idx_quantum_neurons_created_at", table_name="quantum_neurons")
    op.drop_index("idx_quantum_neurons_state", table_name="quantum_neurons")
    op.drop_index("idx_quantum_neurons_name", table_name="quantum_neurons")
    op.drop_table("quantum_neurons")

    # Drop enum
    op.execute("DROP TYPE IF EXISTS quantumstatetype;")
