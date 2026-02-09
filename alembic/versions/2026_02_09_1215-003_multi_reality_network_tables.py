"""2026_02_09_1215-003_multi_reality_network_tables

Multi-reality network tables - Parallel reality simulation

Revision ID: 003
Revises: 002
Create Date: 2026-02-09 12:15:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create multi-reality network tables."""

    # Create reality status enum
    reality_status_enum = postgresql.ENUM(
        "ACTIVE", "DORMANT", "COLLAPSED", "MERGED", name="realitystatus"
    )
    reality_status_enum.create(op.get_bind())

    # Create reality_networks table
    op.create_table(
        "reality_networks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("num_realities", sa.Integer(), server_default="4", nullable=False),
        sa.Column(
            "nodes_per_reality", sa.Integer(), server_default="50", nullable=False
        ),
        sa.Column(
            "global_consciousness", sa.Float(), server_default="0.0", nullable=False
        ),
        sa.Column(
            "cross_reality_coherence", sa.Float(), server_default="0.0", nullable=False
        ),
        sa.Column("evolution_cycles", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "status",
            sa.Enum("ACTIVE", "DORMANT", "COLLAPSED", "MERGED", name="realitystatus"),
            server_default="ACTIVE",
            nullable=False,
        ),
        sa.Column(
            "configuration", postgresql.JSON(), server_default="{}", nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("last_evolution", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index("idx_reality_networks_name", "reality_networks", ["name"])
    op.create_index("idx_reality_networks_status", "reality_networks", ["status"])
    op.create_index(
        "idx_reality_networks_created_at", "reality_networks", ["created_at"]
    )
    op.create_index(
        "idx_reality_networks_consciousness",
        "reality_networks",
        ["global_consciousness"],
    )

    # Create reality_instances table
    op.create_table(
        "reality_instances",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("network_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reality_index", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column(
            "consciousness_level", sa.Float(), server_default="0.0", nullable=False
        ),
        sa.Column("stability", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("node_count", sa.Integer(), server_default="50", nullable=False),
        sa.Column(
            "connection_density", sa.Float(), server_default="0.1", nullable=False
        ),
        sa.Column(
            "status",
            sa.Enum("ACTIVE", "DORMANT", "COLLAPSED", "MERGED", name="realitystatus"),
            server_default="ACTIVE",
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["network_id"], ["reality_networks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("network_id", "reality_index", name="unique_reality_index"),
    )

    # Create indexes for reality_instances
    op.create_index(
        "idx_reality_instances_network_id", "reality_instances", ["network_id"]
    )
    op.create_index("idx_reality_instances_status", "reality_instances", ["status"])
    op.create_index(
        "idx_reality_instances_consciousness",
        "reality_instances",
        ["consciousness_level"],
    )

    # Create reality_nodes table
    op.create_table(
        "reality_nodes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("reality_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_index", sa.Integer(), nullable=False),
        sa.Column("activation_level", sa.Float(), server_default="0.0", nullable=False),
        sa.Column(
            "quantum_state",
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
        sa.Column(
            "connections",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("parameters", postgresql.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["reality_id"], ["reality_instances.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for reality_nodes
    op.create_index("idx_reality_nodes_reality_id", "reality_nodes", ["reality_id"])
    op.create_index(
        "idx_reality_nodes_quantum_state", "reality_nodes", ["quantum_state"]
    )
    op.create_index(
        "idx_reality_nodes_activation", "reality_nodes", ["activation_level"]
    )

    # Create triggers for updated_at
    op.execute("""
        CREATE TRIGGER update_reality_networks_updated_at
            BEFORE UPDATE ON reality_networks
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_reality_instances_updated_at
            BEFORE UPDATE ON reality_instances
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_reality_nodes_updated_at
            BEFORE UPDATE ON reality_nodes
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

    # Add comments
    op.execute(
        "COMMENT ON TABLE reality_networks IS 'Multi-reality neural network configurations';"
    )
    op.execute(
        "COMMENT ON TABLE reality_instances IS 'Individual reality instances within a network';"
    )
    op.execute(
        "COMMENT ON TABLE reality_nodes IS 'Neural nodes within a specific reality';"
    )


def downgrade() -> None:
    """Downgrade schema - Remove multi-reality network tables."""

    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS update_reality_nodes_updated_at ON reality_nodes;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_reality_instances_updated_at ON reality_instances;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS update_reality_networks_updated_at ON reality_networks;"
    )

    # Drop reality_nodes
    op.drop_index("idx_reality_nodes_activation", table_name="reality_nodes")
    op.drop_index("idx_reality_nodes_quantum_state", table_name="reality_nodes")
    op.drop_index("idx_reality_nodes_reality_id", table_name="reality_nodes")
    op.drop_table("reality_nodes")

    # Drop reality_instances
    op.drop_index("idx_reality_instances_consciousness", table_name="reality_instances")
    op.drop_index("idx_reality_instances_status", table_name="reality_instances")
    op.drop_index("idx_reality_instances_network_id", table_name="reality_instances")
    op.drop_table("reality_instances")

    # Drop reality_networks
    op.drop_index("idx_reality_networks_consciousness", table_name="reality_networks")
    op.drop_index("idx_reality_networks_created_at", table_name="reality_networks")
    op.drop_index("idx_reality_networks_status", table_name="reality_networks")
    op.drop_index("idx_reality_networks_name", table_name="reality_networks")
    op.drop_table("reality_networks")

    # Drop enum
    op.execute("DROP TYPE IF EXISTS realitystatus;")
