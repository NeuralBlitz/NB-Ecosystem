"""
Enhanced Authentication Module
Generated: 2026-02-08
"""

from fastapi import APIRouter, HTTPException, Depends, Security, status
from fastapi.security import (
    APIKeyHeader,
    OAuth2PasswordBearer,
    HTTPAuthorizationCredentials,
)
from fastapi.security.api_key import APIKey
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import hashlib
import secrets
import yaml
from pathlib import Path

router = APIRouter()

# Security schemes
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
BEARER_TOKEN = HTTPBearer(auto_error=True)
OAUTH2 = OAuth2PasswordBearer(tokenUrl="token", auto_error=True)


@dataclass
class Partner:
    """Partner information."""

    partner_id: str
    api_key: str
    name: str
    tier: str
    active: bool = True
    permissions: List[str] = field(default_factory=list)
    rate_limit: int = 100
    quota_remaining: int = 1000000
    created_at: str = ""
    expires_at: str = ""
    contact: str = ""
    last_access: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None


class AuthenticationError(Exception):
    """Authentication error."""

    def __init__(self, message: str, code: str = "auth_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class RateLimitExceeded(Exception):
    """Rate limit exceeded."""

    def __init__(self, partner_id: str, limit: int):
        self.partner_id = partner_id
        self.limit = limit
        super().__init__(f"Rate limit exceeded for {partner_id}: {limit} requests")


class AuthenticationManager:
    """Manages authentication and authorization."""

    def __init__(self, config_path: str = "config/partners.yaml"):
        self.partners: Dict[str, Partner] = {}
        self.api_keys: Dict[str, str] = {}  # key -> partner_id
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.load_partners(config_path)

    def load_partners(self, config_path: str):
        """Load partners from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        for partner_id, config in data.get("partners", {}).items():
            partner = Partner(
                partner_id=partner_id,
                api_key=config.get("api_key", ""),
                name=config.get("name", partner_id),
                tier=config.get("tier", "basic"),
                active=config.get("active", True),
                permissions=config.get("permissions", []),
                rate_limit=config.get("rate_limit", 100),
                quota_remaining=config.get("quota_remaining", 1000000),
                created_at=config.get("created_at", ""),
                expires_at=config.get("expires_at", ""),
                contact=config.get("contact", ""),
            )
            self.partners[partner_id] = partner
            self.api_keys[partner.api_key] = partner_id

    def validate_api_key(self, api_key: str) -> Partner:
        """Validate API key and return partner."""
        if not api_key:
            raise AuthenticationError("API key is required", code="missing_api_key")

        partner_id = self.api_keys.get(api_key)
        if not partner_id:
            raise AuthenticationError("Invalid API key", code="invalid_api_key")

        partner = self.partners.get(partner_id)
        if not partner:
            raise AuthenticationError("Partner not found", code="partner_not_found")

        if not partner.active:
            raise AuthenticationError(
                "Partner account is deactivated", code="partner_inactive"
            )

        if partner.locked_until and datetime.now() < partner.locked_until:
            raise AuthenticationError(
                f"Account locked until {partner.locked_until}", code="account_locked"
            )

        return partner

    def check_rate_limit(self, partner: Partner) -> bool:
        """Check if partner has exceeded rate limit."""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)

        # Clean old entries
        if partner.partner_id in self.rate_limits:
            self.rate_limits[partner.partner_id] = [
                t for t in self.rate_limits[partner.partner_id] if t > window_start
            ]

        # Check limit
        if len(self.rate_limits.get(partner.partner_id, [])) >= partner.rate_limit:
            partner.failed_attempts += 1
            if partner.failed_attempts >= 5:
                partner.locked_until = now + timedelta(minutes=30)
            return False

        # Record request
        if partner.partner_id not in self.rate_limits:
            self.rate_limits[partner.partner_id] = []
        self.rate_limits[partner.partner_id].append(now)

        return True

    def check_quota(self, partner: Partner, amount: int = 1) -> bool:
        """Check if partner has sufficient quota."""
        if partner.quota_remaining < amount:
            raise AuthenticationError(
                f"Quota exceeded. Remaining: {partner.quota_remaining}",
                code="quota_exceeded",
            )
        return True

    def use_quota(self, partner: Partner, amount: int = 1):
        """Deduct from partner's quota."""
        partner.quota_remaining -= amount
        partner.last_access = datetime.now()


# Initialize authentication manager
auth_manager = AuthenticationManager()


async def verify_api_key(api_key: APIKey = Security(API_KEY_HEADER)) -> Partner:
    """Verify API key and return partner info."""
    try:
        partner = auth_manager.validate_api_key(api_key)

        # Check rate limit
        if not auth_manager.check_rate_limit(partner):
            raise RateLimitExceeded(partner.partner_id, partner.rate_limit)

        # Check quota
        auth_manager.check_quota(partner)

        return partner

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "ApiKey"},
        )
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {e.limit} requests/minute",
        )


async def verify_permission(partner: Partner, permission: str):
    """Verify partner has required permission."""
    if permission not in partner.permissions and "*" not in partner.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}",
        )


def generate_api_key() -> str:
    """Generate new secure API key."""
    return f"nb_{secrets.token_hex(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


@router.post("/auth/validate")
async def validate_credentials(partner: Partner = Depends(verify_api_key)):
    """Validate credentials and return partner info."""
    return {
        "valid": True,
        "partner_id": partner.partner_id,
        "name": partner.name,
        "tier": partner.tier,
        "permissions": partner.permissions,
    }


@router.post("/auth/refresh")
async def refresh_api_key(current_key: str, partner: Partner = Depends(verify_api_key)):
    """Refresh API key (key rotation)."""
    new_key = generate_api_key()

    # Remove old key
    del auth_manager.api_keys[partner.api_key]

    # Add new key
    partner.api_key = new_key
    auth_manager.api_keys[new_key] = partner.partner_id

    return {"new_api_key": new_key, "message": "API key rotated successfully"}


@router.get("/auth/status")
async def auth_status(partner: Partner = Depends(verify_api_key)):
    """Get authentication status and usage."""
    return {
        "partner_id": partner.partner_id,
        "name": partner.name,
        "tier": partner.tier,
        "quota_remaining": partner.quota_remaining,
        "rate_limit": partner.rate_limit,
        "last_access": partner.last_access.isoformat() if partner.last_access else None,
        "active": partner.active,
    }


@router.post("/auth/logout")
async def logout(partner: Partner = Depends(verify_api_key)):
    """Log out (invalidate session)."""
    # For API keys, logout just clears rate limit tracking
    if partner.partner_id in auth_manager.rate_limits:
        auth_manager.rate_limits[partner.partner_id] = []

    return {"message": "Logged out successfully"}
