"""
User Account Models
Authentication, sessions, and user preferences
"""

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Index,
    UniqueConstraint,
    CheckConstraint,
    Integer,
    Enum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, INET
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from .base import Base, TimestampMixin, UUIDMixin, AuditMixin, MetadataMixin


class UserRole(str, enum.Enum):
    """User role enumeration"""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API = "api"


class UserStatus(str, enum.Enum):
    """User account status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending_verification"
    DELETED = "deleted"


class User(Base, UUIDMixin, TimestampMixin, AuditMixin, MetadataMixin):
    """User account model with authentication"""

    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email_active", "email", "status", postgresql_where="deleted_at IS NULL"),
        Index("ix_users_role_status", "role", "status"),
        Index("ix_users_created_at_brin", "created_at", postgresql_using="brin"),
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint(
            "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'",
            name="ck_users_email_format",
        ),
        {"comment": "User accounts and authentication"},
    )

    # Authentication fields
    email = Column(String(255), nullable=False, index=True, comment="User email address")
    password_hash = Column(String(255), nullable=False, comment="Bcrypt hashed password")

    # Profile fields
    first_name = Column(String(100), nullable=False, comment="First name")
    last_name = Column(String(100), nullable=False, comment="Last name")
    display_name = Column(
        String(200), nullable=True, comment="Display name (auto-generated if null)"
    )

    # Status and role
    role = Column(
        Enum(UserRole, name="user_role_enum"),
        nullable=False,
        default=UserRole.VIEWER,
        comment="User role for RBAC",
    )
    status = Column(
        Enum(UserStatus, name="user_status_enum"),
        nullable=False,
        default=UserStatus.PENDING,
        comment="Account status",
    )

    # Security fields
    email_verified = Column(
        Boolean, default=False, nullable=False, comment="Email verification status"
    )
    email_verified_at = Column(
        DateTime(timezone=True), nullable=True, comment="Email verification timestamp"
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True, comment="Last successful login")
    last_login_ip = Column(INET, nullable=True, comment="IP address of last login")
    failed_login_attempts = Column(
        Integer, default=0, nullable=False, comment="Consecutive failed login attempts"
    )
    locked_until = Column(
        DateTime(timezone=True), nullable=True, comment="Account lockout expiration"
    )
    password_changed_at = Column(
        DateTime(timezone=True), nullable=True, comment="Last password change"
    )

    # MFA fields
    mfa_enabled = Column(
        Boolean, default=False, nullable=False, comment="Multi-factor authentication enabled"
    )
    mfa_secret = Column(String(255), nullable=True, comment="Encrypted MFA secret")
    mfa_backup_codes = Column(
        ARRAY(String), default=list, nullable=False, comment="Hashed MFA backup codes"
    )

    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship(
        "UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    @validates("email")
    def validate_email(self, key, email):
        """Normalize email address"""
        return email.lower().strip() if email else email

    @property
    def full_name(self) -> str:
        """Return full name"""
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class UserSession(Base, UUIDMixin, TimestampMixin):
    """Active user sessions for authentication"""

    __tablename__ = "user_sessions"
    __table_args__ = (
        Index(
            "ix_user_sessions_user_active",
            "user_id",
            "is_active",
            postgresql_where="is_active = true",
        ),
        Index("ix_user_sessions_expires_at", "expires_at"),
        Index("ix_user_sessions_refresh_token", "refresh_token_hash"),
        {"comment": "Active user sessions"},
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user",
    )

    # Session tokens (hashed for security)
    access_token_hash = Column(
        String(255), nullable=False, unique=True, comment="SHA256 hash of access token"
    )
    refresh_token_hash = Column(
        String(255), nullable=False, unique=True, comment="SHA256 hash of refresh token"
    )

    # Session metadata
    expires_at = Column(DateTime(timezone=True), nullable=False, comment="Session expiration")
    refresh_expires_at = Column(
        DateTime(timezone=True), nullable=False, comment="Refresh token expiration"
    )
    is_active = Column(Boolean, default=True, nullable=False, comment="Session active status")
    revoked_at = Column(
        DateTime(timezone=True), nullable=True, comment="Session revocation timestamp"
    )

    # Device information
    ip_address = Column(INET, nullable=True, comment="Client IP address")
    user_agent = Column(Text, nullable=True, comment="Client user agent")
    device_info = Column(JSONB, default=dict, nullable=False, comment="Parsed device information")

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, active={self.is_active})>"


class UserPreference(Base, TimestampMixin):
    """User preferences and settings"""

    __tablename__ = "user_preferences"
    __table_args__ = (
        Index("ix_user_preferences_user", "user_id", unique=True),
        {"comment": "User preferences and settings"},
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Reference to user",
    )

    # UI preferences
    theme = Column(String(20), default="system", nullable=False, comment="UI theme preference")
    language = Column(String(10), default="en", nullable=False, comment="Preferred language")
    timezone = Column(String(50), default="UTC", nullable=False, comment="Preferred timezone")

    # Notification preferences
    email_notifications = Column(
        Boolean, default=True, nullable=False, comment="Email notifications enabled"
    )
    push_notifications = Column(
        Boolean, default=True, nullable=False, comment="Push notifications enabled"
    )
    notification_frequency = Column(
        String(20), default="immediate", nullable=False, comment="Notification frequency"
    )

    # Dashboard preferences
    dashboard_layout = Column(
        JSONB, default=dict, nullable=False, comment="Dashboard layout configuration"
    )
    default_view = Column(
        String(50), default="overview", nullable=False, comment="Default dashboard view"
    )

    # Feature flags
    beta_features = Column(Boolean, default=False, nullable=False, comment="Beta features enabled")
    experimental_features = Column(
        ARRAY(String), default=list, nullable=False, comment="Enabled experimental features"
    )

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"
