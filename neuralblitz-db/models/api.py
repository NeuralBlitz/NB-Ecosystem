"""
API Key and Partner Configuration Models
Authentication, rate limiting, and partner integrations
"""

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    Index,
    UniqueConstraint,
    Enum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, INET, CIDR
from sqlalchemy.orm import relationship, validates

from .base import Base, TimestampMixin, UUIDMixin, AuditMixin, MetadataMixin


class APIKeyScope(str, enum.Enum):
    """API key scope permissions"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    QUANTUM_READ = "quantum:read"
    QUANTUM_WRITE = "quantum:write"
    REALITY_READ = "reality:read"
    REALITY_WRITE = "reality:write"
    CONSCIOUSNESS_READ = "consciousness:read"
    CONSCIOUSNESS_WRITE = "consciousness:write"
    METRICS_READ = "metrics:read"
    AUDIT_READ = "audit:read"


class APIKeyStatus(str, enum.Enum):
    """API key status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"


class RateLimitTier(str, enum.Enum):
    """Rate limit tiers"""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class APIKey(Base, UUIDMixin, TimestampMixin, AuditMixin):
    """API keys for external access"""

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash", unique=True),
        Index("ix_api_keys_user_status", "user_id", "status"),
        Index("ix_api_keys_partner", "partner_id"),
        Index("ix_api_keys_expires_at", "expires_at"),
        {"comment": "API keys for external authentication"},
    )

    # Key identification
    key_hash = Column(String(255), nullable=False, unique=True, comment="SHA256 hash of API key")
    key_prefix = Column(
        String(20), nullable=False, comment="First 8 chars of key for identification"
    )
    name = Column(String(100), nullable=False, comment="Human-readable key name")
    description = Column(Text, nullable=True, comment="Key description")

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Owner user (null for service keys)",
    )
    partner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("partner_configs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Associated partner configuration",
    )

    # Permissions
    scopes = Column(
        ARRAY(Enum(APIKeyScope, name="api_key_scope_enum")),
        default=list,
        nullable=False,
        comment="Granted permission scopes",
    )
    status = Column(
        Enum(APIKeyStatus, name="api_key_status_enum"),
        nullable=False,
        default=APIKeyStatus.ACTIVE,
        comment="Key status",
    )

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True, comment="Key expiration date")
    last_used_at = Column(DateTime(timezone=True), nullable=True, comment="Last usage timestamp")

    # Rate limiting
    rate_limit_tier = Column(
        Enum(RateLimitTier, name="rate_limit_tier_enum"),
        nullable=False,
        default=RateLimitTier.FREE,
        comment="Rate limiting tier",
    )
    custom_rate_limit = Column(JSONB, nullable=True, comment="Custom rate limit overrides")

    # IP restrictions
    allowed_ips = Column(ARRAY(CIDR), nullable=True, comment="Allowed IP ranges")
    blocked_ips = Column(ARRAY(CIDR), default=list, nullable=False, comment="Blocked IP ranges")

    # Usage tracking
    total_requests = Column(Integer, default=0, nullable=False, comment="Total request count")
    requests_this_hour = Column(
        Integer, default=0, nullable=False, comment="Requests in current hour"
    )
    requests_today = Column(Integer, default=0, nullable=False, comment="Requests today")

    # Relationships
    user = relationship("User", back_populates="api_keys")
    partner = relationship("PartnerConfig", back_populates="api_keys")
    rate_limits = relationship("RateLimit", back_populates="api_key", cascade="all, delete-orphan")

    @property
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def __repr__(self):
        return f"<APIKey(name={self.name}, status={self.status})>"


class PartnerConfig(Base, UUIDMixin, TimestampMixin, AuditMixin, MetadataMixin):
    """Third-party partner configurations"""

    __tablename__ = "partner_configs"
    __table_args__ = (
        Index("ix_partner_configs_name", "name", unique=True),
        Index("ix_partner_configs_status", "status"),
        Index("ix_partner_configs_type", "partner_type"),
        {"comment": "Third-party partner configurations"},
    )

    # Basic info
    name = Column(String(100), nullable=False, unique=True, comment="Partner name")
    partner_type = Column(
        String(50), nullable=False, comment="Partner type (cloud, enterprise, academic)"
    )
    description = Column(Text, nullable=True, comment="Partner description")

    # Contact information
    contact_email = Column(String(255), nullable=False, comment="Primary contact email")
    contact_name = Column(String(200), nullable=True, comment="Primary contact name")
    technical_contact = Column(
        JSONB, default=dict, nullable=False, comment="Technical contact information"
    )

    # Status
    status = Column(String(20), default="active", nullable=False, comment="Partner status")
    contract_start = Column(DateTime(timezone=True), nullable=True, comment="Contract start date")
    contract_end = Column(DateTime(timezone=True), nullable=True, comment="Contract end date")

    # Configuration
    webhook_url = Column(String(500), nullable=True, comment="Webhook endpoint URL")
    webhook_secret = Column(String(255), nullable=True, comment="Webhook signature secret")
    callback_urls = Column(
        ARRAY(String), default=list, nullable=False, comment="Allowed callback URLs"
    )

    # Integration settings
    integration_config = Column(
        JSONB, default=dict, nullable=False, comment="Integration-specific configuration"
    )
    feature_access = Column(
        ARRAY(String), default=list, nullable=False, comment="Enabled features for partner"
    )

    # Rate limiting
    default_rate_limit = Column(JSONB, nullable=True, comment="Default rate limit configuration")

    # Relationships
    api_keys = relationship("APIKey", back_populates="partner")

    def __repr__(self):
        return f"<PartnerConfig(name={self.name}, type={self.partner_type})>"


class RateLimit(Base, UUIDMixin, TimestampMixin):
    """Rate limit tracking per API key"""

    __tablename__ = "rate_limits"
    __table_args__ = (
        Index("ix_rate_limits_api_key_window", "api_key_id", "window_start"),
        Index("ix_rate_limits_api_key_endpoint", "api_key_id", "endpoint"),
        {"comment": "Rate limit usage tracking"},
    )

    api_key_id = Column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to API key",
    )

    # Window tracking
    window_start = Column(
        DateTime(timezone=True), nullable=False, comment="Rate limit window start"
    )
    window_end = Column(DateTime(timezone=True), nullable=False, comment="Rate limit window end")

    # Request tracking
    request_count = Column(Integer, default=0, nullable=False, comment="Requests in window")
    limit_reached = Column(Boolean, default=False, nullable=False, comment="Limit was reached")

    # Endpoint-specific tracking
    endpoint = Column(String(200), nullable=True, comment="API endpoint path")
    method = Column(String(10), nullable=True, comment="HTTP method")

    # Violations
    violations = Column(Integer, default=0, nullable=False, comment="Rate limit violations")

    # Relationships
    api_key = relationship("APIKey", back_populates="rate_limits")

    def __repr__(self):
        return f"<RateLimit(api_key={self.api_key_id}, count={self.request_count})>"
