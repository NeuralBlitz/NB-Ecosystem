#!/usr/bin/env python3
"""
NeuralBlitz Partner Onboarding Script
Generated: 2026-02-08

This script automates the partner onboarding process.
"""

import json
import yaml
import argparse
import secrets
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys


class PartnerOnboarding:
    """Manages partner onboarding process."""

    def __init__(self, config_path: str = "config/partners.yaml"):
        self.config_path = Path(config_path)
        self.partners = self.load_partners()

    def load_partners(self) -> Dict:
        """Load existing partners."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                return data.get("partners", {})
        return {}

    def save_partners(self):
        """Save partners to config."""
        config = {"partners": self.partners}

        # Create backup
        if self.config_path.exists():
            backup_path = self.config_path.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            self.config_path.rename(backup_path)

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return f"nb_{secrets.token_hex(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def add_partner(
        self,
        name: str,
        tier: str = "basic",
        permissions: Optional[List[str]] = None,
        quota: Optional[int] = None,
        rate_limit: Optional[int] = None,
        contact: str = "",
        expires_days: int = 90,
    ) -> Dict:
        """Add new partner."""
        partner_id = name.lower().replace(" ", "_").replace("-", "_")

        # Set defaults based on tier
        tier_configs = {
            "enterprise": {"quota": 1000000, "rate_limit": 10000},
            "pro": {"quota": 100000, "rate_limit": 1000},
            "basic": {"quota": 10000, "rate_limit": 100},
        }

        config = tier_configs.get(tier, tier_configs["basic"])

        partner = {
            "api_key": self.generate_api_key(),
            "name": name,
            "tier": tier,
            "active": True,
            "permissions": permissions or ["core"],
            "quota_remaining": quota or config["quota"],
            "rate_limit": rate_limit or config["rate_limit"],
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=expires_days)).isoformat(),
            "contact": contact,
        }

        self.partners[partner_id] = partner

        return {"partner_id": partner_id, **partner}

    def remove_partner(self, partner_id: str) -> bool:
        """Remove partner."""
        if partner_id in self.partners:
            del self.partners[partner_id]
            self.save_partners()
            return True
        return False

    def deactivate_partner(self, partner_id: str) -> bool:
        """Deactivate partner (keep config, revoke access)."""
        if partner_id in self.partners:
            self.partners[partner_id]["active"] = False
            self.save_partners()
            return True
        return False

    def reactivate_partner(self, partner_id: str) -> bool:
        """Reactivate partner."""
        if partner_id in self.partners:
            self.partners[partner_id]["active"] = True
            self.save_partners()
            return True
        return False

    def update_quota(self, partner_id: str, amount: int) -> bool:
        """Update partner quota."""
        if partner_id in self.partners:
            self.partners[partner_id]["quota_remaining"] = amount
            self.save_partners()
            return True
        return False

    def list_partners(self) -> List[Dict]:
        """List all partners."""
        return [
            {
                "partner_id": pid,
                "name": p["name"],
                "tier": p["tier"],
                "active": p["active"],
                "quota_remaining": p["quota_remaining"],
                "expires_at": p.get("expires_at", "N/A"),
            }
            for pid, p in self.partners.items()
        ]

    def generate_credentials_file(
        self, partner_id: str, output_path: str = "."
    ) -> Optional[str]:
        """Generate credentials file for partner."""
        if partner_id not in self.partners:
            return None

        partner = self.partners[partner_id]

        credentials = f"""# NeuralBlitz Partner Credentials
# Generated: {datetime.now().isoformat()}
# ⚠️ KEEP SECURE - DO NOT COMMIT TO VERSION CONTROL

NEURALBLITZ_API_KEY="{partner["api_key"]}"
NEURALBLITZ_PARTNER_ID="{partner_id}"
NEURALBLITZ_TIER="{partner["tier"]}"
NEURALBLITZ_BASE_URL="https://api.neuralblitz.ai"

# Environment variables to set:
# export NEURALBLITZ_API_KEY="{partner["api_key"]}"
# export NEURALBLITZ_PARTNER_ID="{partner_id}"
"""

        filename = Path(output_path) / f"neuralblitz_{partner_id}_credentials.txt"
        with open(filename, "w") as f:
            f.write(credentials)

        return str(filename)

    def generate_onboarding_email(self, partner_id: str) -> Optional[str]:
        """Generate onboarding email template."""
        if partner_id not in self.partners:
            return None

        partner = self.partners[partner_id]

        email = f"""Subject: Welcome to NeuralBlitz - Partner Access Details

Dear {partner["name"]},

Welcome to the NeuralBlitz Partner Program! We're excited to have you on board.

Your Partner Credentials
{"=" * 50}

Partner ID: {partner_id}
API Key: {partner["api_key"]}
Tier: {partner["tier"].title()}
Quota: {partner["quota_remaining"]:,} requests
Rate Limit: {partner["rate_limit"]:,} requests/minute

Getting Started
{"=" * 50}

1. Install the SDK:
   pip install neuralblitz-core

2. Configure your credentials:
   export NEURALBLITZ_API_KEY="{partner["api_key"]}"

3. Make your first API call:
   from neuralblitz_core import NeuralBlitzCore
   nb = NeuralBlitzCore()
   capabilities = nb.get_capabilities()

Documentation
{"=" * 50}

- API Documentation: https://docs.neuralblitz.ai
- SDK Guides: https://docs.neuralblitz.ai/sdk
- Support: support@neuralblitz.ai

Your access expires on: {partner.get("expires_at", "N/A")}

Best regards,
The NeuralBlitz Team
"""

        return email


def main():
    parser = argparse.ArgumentParser(description="NeuralBlitz Partner Onboarding Tool")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add partner command
    add_parser = subparsers.add_parser("add", help="Add new partner")
    add_parser.add_argument("name", help="Partner name")
    add_parser.add_argument(
        "--tier",
        choices=["basic", "pro", "enterprise"],
        default="basic",
        help="Partner tier",
    )
    add_parser.add_argument(
        "--permissions", nargs="+", default=["core"], help="Permissions"
    )
    add_parser.add_argument("--quota", type=int, help="Request quota")
    add_parser.add_argument("--rate-limit", type=int, help="Rate limit per minute")
    add_parser.add_argument("--contact", default="", help="Contact email")
    add_parser.add_argument(
        "--expires", type=int, default=90, help="Expiration in days"
    )

    # List partners command
    subparsers.add_parser("list", help="List all partners")

    # Remove partner command
    remove_parser = subparsers.add_parser("remove", help="Remove partner")
    remove_parser.add_argument("partner_id", help="Partner ID")

    # Deactivate partner command
    deactivate_parser = subparsers.add_parser("deactivate", help="Deactivate partner")
    deactivate_parser.add_argument("partner_id", help="Partner ID")

    # Reactivate partner command
    reactivate_parser = subparsers.add_parser("reactivate", help="Reactivate partner")
    reactivate_parser.add_argument("partner_id", help="Partner ID")

    # Update quota command
    quota_parser = subparsers.add_parser("quota", help="Update partner quota")
    quota_parser.add_argument("partner_id", help="Partner ID")
    quota_parser.add_argument("amount", type=int, help="New quota amount")

    # Credentials command
    creds_parser = subparsers.add_parser(
        "credentials", help="Generate credentials file"
    )
    creds_parser.add_argument("partner_id", help="Partner ID")
    creds_parser.add_argument("--output", default=".", help="Output directory")

    # Email command
    email_parser = subparsers.add_parser("email", help="Generate onboarding email")
    email_parser.add_argument("partner_id", help="Partner ID")

    args = parser.parse_args()

    onboarding = PartnerOnboarding()

    if args.command == "add":
        partner = onboarding.add_partner(
            args.name,
            tier=args.tier,
            permissions=args.permissions,
            quota=args.quota,
            rate_limit=args.rate_limit,
            contact=args.contact,
            expires_days=args.expires,
        )
        onboarding.save_partners()

        print("\n" + "=" * 60)
        print(" Partner Added Successfully!")
        print("=" * 60)
        print(f"\nPartner ID: {partner['partner_id']}")
        print(f"Name: {partner['name']}")
        print(f"Tier: {partner['tier'].title()}")
        print(f"API Key: {partner['api_key']}")
        print(f"Quota: {partner['quota_remaining']:,}")
        print(f"Rate Limit: {partner['rate_limit']:,}/min")
        print(f"Expires: {partner['expires_at']}")
        print("\n" + "-" * 60)
        print("⚠️  IMPORTANT: Copy the API key now - it won't be shown again!")
        print("-" * 60 + "\n")

    elif args.command == "list":
        partners = onboarding.list_partners()
        print("\n" + "=" * 60)
        print(" NeuralBlitz Partners")
        print("=" * 60 + "\n")

        for p in partners:
            status = "✅ Active" if p["active"] else "❌ Inactive"
            print(f"{p['partner_id']:20} | {p['tier']:10} | {status}")
            print(f"{'':20} | {'':10} | Quota: {p['quota_remaining']:,}")
            print()

    elif args.command == "remove":
        if onboarding.remove_partner(args.partner_id):
            print(f"Partner '{args.partner_id}' removed.")
        else:
            print(f"Partner '{args.partner_id}' not found.")

    elif args.command == "deactivate":
        if onboarding.deactivate_partner(args.partner_id):
            print(f"Partner '{args.partner_id}' deactivated.")
        else:
            print(f"Partner '{args.partner_id}' not found.")

    elif args.command == "reactivate":
        if onboarding.reactivate_partner(args.partner_id):
            print(f"Partner '{args.partner_id}' reactivated.")
        else:
            print(f"Partner '{args.partner_id}' not found.")

    elif args.command == "quota":
        if onboarding.update_quota(args.partner_id, args.amount):
            print(f"Quota updated for '{args.partner_id}': {args.amount:,}")
        else:
            print(f"Partner '{args.partner_id}' not found.")

    elif args.command == "credentials":
        filepath = onboarding.generate_credentials_file(args.partner_id, args.output)
        if filepath:
            print(f"Credentials file created: {filepath}")
        else:
            print(f"Partner '{args.partner_id}' not found.")

    elif args.command == "email":
        email = onboarding.generate_onboarding_email(args.partner_id)
        if email:
            print(email)
        else:
            print(f"Partner '{args.partner_id}' not found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
