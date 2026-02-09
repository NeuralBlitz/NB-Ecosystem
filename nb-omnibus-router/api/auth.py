"""
API Key Authentication
Partnership-based authentication
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Dict, Optional
import yaml
import os
from pathlib import Path

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def load_partners() -> Dict:
    """Load partner configurations from YAML."""
    config_path = Path(__file__).parent.parent / "config" / "partners.yaml"

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


PARTNER_DB = load_partners()


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> Dict:
    """
    Verify API key belongs to authorized partner.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Partner information dictionary

    Raises:
        HTTPException: If API key is invalid or unauthorized
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Search through partner database
    for partner_id, partner_data in PARTNER_DB.items():
        if partner_data.get("api_key") == api_key:
            # Check if partner is active
            if not partner_data.get("active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key has been deactivated",
                )

            # Return partner info with defaults
            return {
                "partner_id": partner_id,
                "tier": partner_data.get("tier", "basic"),
                "rate_limit": partner_data.get("rate_limit", 100),
                "permissions": partner_data.get("permissions", []),
                "quota_remaining": partner_data.get("quota_remaining", 1000),
            }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or unauthorized API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
