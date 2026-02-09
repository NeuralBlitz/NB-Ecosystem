"""
Base Model Configuration
Provides common mixins and base class for all models
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, MetaData, String, event
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy.sql import func

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)


class TimestampMixin:
    """Adds created_at and updated_at timestamps"""

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Record creation timestamp",
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Record last update timestamp",
    )


class UUIDMixin:
    """Adds UUID primary key"""

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        comment="Unique identifier",
    )


class AuditMixin:
    """Adds audit fields for tracking changes"""

    created_by = Column(
        UUID(as_uuid=True), nullable=True, index=True, comment="User who created the record"
    )
    updated_by = Column(
        UUID(as_uuid=True), nullable=True, index=True, comment="User who last updated the record"
    )
    deleted_at = Column(
        DateTime(timezone=True), nullable=True, index=True, comment="Soft delete timestamp"
    )
    deleted_by = Column(UUID(as_uuid=True), nullable=True, comment="User who deleted the record")
    version = Column(
        String(50), default="1.0.0", nullable=False, comment="Record version for optimistic locking"
    )


class MetadataMixin:
    """Adds metadata JSONB column"""

    metadata = Column(JSONB, default=dict, nullable=False, comment="Additional metadata as JSON")


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID"""
    return uuid.uuid4()


# Event listeners for automatic timestamp updates
@event.listens_for(Base, "before_update", propagate=True)
def update_timestamp(mapper, connection, target):
    """Automatically update updated_at on modification"""
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.utcnow()
