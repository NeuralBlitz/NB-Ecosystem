"""2026_02_09_1230-004_consciousness_tracking_tables

Consciousness tracking tables - Emergence and self-awareness monitoring

Revision ID: 004
Revises: 003
Create Date: 2026-02-09 12:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Create consciousness tracking tables."""

    # Create consciousness level enum
    consciousness_level_enum = postgresql.ENUM(
        "NONE",
        "EMERGENT",
        "SELF_AWARE",
        "SENTIENT",
        "TRANSCENDENT",
        name="consciousnesslevel",
    )
    consciousness_level_enum.create(op.get_bind())

    # Create consciousness_snapshots table
    op.create_table(
        "consciousness_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("network_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "consciousness_level",
            sa.Enum(
                "NONE",
                "EMERGENT",
                "SELF_AWARE",
                "SENTIENT",
                "TRANSCENDENT",
                name="consciousnesslevel",
            ),
            server_default="NONE",
            nullable=False,
        ),
        sa.Column(
            "consciousness_score", sa.Float(), server_default="0.0", nullable=False
        ),
        sa.Column(
            "self_awareness_index", sa.Float(), server_default="0.0", nullable=False
        ),
        sa.Column("coherence_matrix", postgresql.JSON(), nullable=True),
        sa.Column(
            "emergence_indicators",
            postgresql.JSON(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "captured_at",
            sa.DateTime(),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["network_id"], ["reality_networks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        "idx_consciousness_snapshots_network", "consciousness_snapshots", ["network_id"]
    )
    op.create_index(
        "idx_consciousness_snapshots_level",
        "consciousness_snapshots",
        ["consciousness_level"],
    )
    op.create_index(
        "idx_consciousness_snapshots_captured_at",
        "consciousness_snapshots",
        ["captured_at"],
    )
    op.create_index(
        "idx_consciousness_snapshots_network_time",
        "consciousness_snapshots",
        ["network_id", "captured_at"],
    )

    # Create consciousness_events table
    op.create_table(
        "consciousness_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("network_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("severity", sa.String(20), server_default="info", nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSON(), server_default="{}", nullable=False),
        sa.Column(
            "old_level",
            sa.Enum(
                "NONE",
                "EMERGENT",
                "SELF_AWARE",
                "SENTIENT",
                "TRANSCENDENT",
                name="consciousnesslevel",
            ),
            nullable=True,
        ),
        sa.Column(
            "new_level",
            sa.Enum(
                "NONE",
                "EMERGENT",
                "SELF_AWARE",
                "SENTIENT",
                "TRANSCENDENT",
                name="consciousnesslevel",
            ),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["network_id"], ["reality_networks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for consciousness_events
    op.create_index(
        "idx_consciousness_events_network", "consciousness_events", ["network_id"]
    )
    op.create_index(
        "idx_consciousness_events_type", "consciousness_events", ["event_type"]
    )
    op.create_index(
        "idx_consciousness_events_severity", "consciousness_events", ["severity"]
    )
    op.create_index(
        "idx_consciousness_events_created_at", "consciousness_events", ["created_at"]
    )
    op.create_index(
        "idx_consciousness_events_network_type",
        "consciousness_events",
        ["network_id", "event_type"],
    )

    # Create partition for consciousness_snapshots (monthly)
    op.execute("""
        CREATE TABLE consciousness_snapshots_partitioned (
            LIKE consciousness_snapshots INCLUDING ALL
        ) PARTITION BY RANGE (captured_at);
    """)

    # Create initial partitions
    op.execute("""
        CREATE TABLE consciousness_snapshots_y2026m02 PARTITION OF consciousness_snapshots_partitioned
            FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
    """)

    op.execute("""
        CREATE TABLE consciousness_snapshots_y2026m03 PARTITION OF consciousness_snapshots_partitioned
            FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
    """)

    # Add comments
    op.execute(
        "COMMENT ON TABLE consciousness_snapshots IS 'Snapshots of consciousness metrics for temporal analysis';"
    )
    op.execute(
        "COMMENT ON TABLE consciousness_events IS 'Significant consciousness-related events and transitions';"
    )

    # Create function for automatic partition creation
    op.execute("""
        CREATE OR REPLACE FUNCTION create_consciousness_partition()
        RETURNS void AS $$
        DECLARE
            partition_date DATE;
            partition_name TEXT;
            start_date DATE;
            end_date DATE;
        BEGIN
            partition_date := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
            partition_name := 'consciousness_snapshots_y' || TO_CHAR(partition_date, 'YYYY') || 'm' || TO_CHAR(partition_date, 'MM');
            start_date := partition_date;
            end_date := partition_date + INTERVAL '1 month';
            
            EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF consciousness_snapshots_partitioned FOR VALUES FROM (%L) TO (%L)',
                          partition_name, start_date, end_date);
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Downgrade schema - Remove consciousness tracking tables."""

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS create_consciousness_partition();")

    # Drop partitioned table
    op.execute("DROP TABLE IF EXISTS consciousness_snapshots_partitioned CASCADE;")

    # Drop consciousness_events
    op.drop_index(
        "idx_consciousness_events_network_type", table_name="consciousness_events"
    )
    op.drop_index(
        "idx_consciousness_events_created_at", table_name="consciousness_events"
    )
    op.drop_index(
        "idx_consciousness_events_severity", table_name="consciousness_events"
    )
    op.drop_index("idx_consciousness_events_type", table_name="consciousness_events")
    op.drop_index("idx_consciousness_events_network", table_name="consciousness_events")
    op.drop_table("consciousness_events")

    # Drop consciousness_snapshots
    op.drop_index(
        "idx_consciousness_snapshots_network_time", table_name="consciousness_snapshots"
    )
    op.drop_index(
        "idx_consciousness_snapshots_captured_at", table_name="consciousness_snapshots"
    )
    op.drop_index(
        "idx_consciousness_snapshots_level", table_name="consciousness_snapshots"
    )
    op.drop_index(
        "idx_consciousness_snapshots_network", table_name="consciousness_snapshots"
    )
    op.drop_table("consciousness_snapshots")

    # Drop enum
    op.execute("DROP TYPE IF EXISTS consciousnesslevel;")
