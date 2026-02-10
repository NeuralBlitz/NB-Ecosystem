"""
Secure Password Hashing Module
Fixes: NBX-2026-0002 - Weak Password Hashing (SHA-256)
Replaces: hashlib.sha256() with bcrypt/Argon2
"""

import bcrypt
import hashlib
from typing import Optional
import secrets
import os


class SecurePasswordHasher:
    """
    Production-grade password hashing using bcrypt with Argon2 fallback.

    This class addresses the critical vulnerability of using SHA-256 for password hashing.
    SHA-256 is designed for speed, making it vulnerable to GPU-based cracking attacks.
    bcrypt is intentionally slow and includes salt, making it suitable for password storage.
    """

    def __init__(self, algorithm: str = "bcrypt"):
        """
        Initialize the password hasher.

        Args:
            algorithm: "bcrypt" (recommended) or "argon2" (fallback)
        """
        self.algorithm = algorithm
        self.bcrypt_rounds = 12  # Current recommendation: 12+ rounds

    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.

        Args:
            password: Plain-text password

        Returns:
            Salted hash string suitable for storage

        Raises:
            ValueError: If password is empty or too weak
        """
        if not password:
            raise ValueError("Password cannot be empty")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        if self.algorithm == "bcrypt":
            return self._hash_bcrypt(password)
        elif self.algorithm == "argon2":
            return self._hash_argon2(password)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _hash_bcrypt(self, password: str) -> str:
        """Hash password using bcrypt."""
        try:
            # Generate salt with specified rounds
            salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
            # Hash the password
            hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
            return hashed.decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Password hashing failed: {e}")

    def _hash_argon2(self, password: str) -> str:
        """Hash password using Argon2 (if available)."""
        try:
            # Try to import argon2
            import argon2
            from argon2 import PasswordHasher

            ph = PasswordHasher(
                time_cost=3,  # Number of iterations
                memory_cost=65536,  # Memory usage in KiB (64MB)
                parallelism=4,  # Number of parallel threads
                hash_len=32,  # Hash length
                salt_len=16,  # Salt length
            )

            return ph.hash(password)
        except ImportError:
            # Fallback to bcrypt if argon2 not available
            return self._hash_bcrypt(password)
        except Exception as e:
            raise RuntimeError(f"Argon2 hashing failed: {e}")

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain-text password to verify
            hashed_password: Stored hash to verify against

        Returns:
            True if password matches, False otherwise
        """
        if not password or not hashed_password:
            return False

        try:
            if self.algorithm == "bcrypt":
                return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
            elif self.algorithm == "argon2":
                return self._verify_argon2(password, hashed_password)
        except Exception:
            # Always return False on error to prevent timing attacks
            return False

        return False

    def _verify_argon2(self, password: str, hashed_password: str) -> bool:
        """Verify password using Argon2."""
        try:
            import argon2
            from argon2 import PasswordHasher

            ph = PasswordHasher()
            try:
                ph.verify(hashed_password, password)
                return True
            except argon2.exceptions.VerifyMismatchError:
                return False
        except ImportError:
            # Fallback to bcrypt if argon2 not available
            return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a cryptographically secure random password.

        Args:
            length: Length of password to generate

        Returns:
            Secure random password
        """
        # Use secrets module for cryptographic randomness
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def check_password_strength(self, password: str) -> dict:
        """
        Check password strength against common requirements.

        Args:
            password: Password to check

        Returns:
            Dictionary with strength metrics
        """
        score = 0
        feedback = []

        # Length check
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters")

        # Complexity checks
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Include lowercase letters")

        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Include uppercase letters")

        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Include numbers")

        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Include special characters")

        # Common patterns
        if password.lower() not in ["password", "123456", "qwerty", "admin", "letmein"]:
            score += 1
        else:
            feedback.append("Avoid common passwords")

        strength_levels = {
            0: "Very Weak",
            1: "Weak",
            2: "Fair",
            3: "Good",
            4: "Strong",
            5: "Very Strong",
            6: "Excellent",
            7: "Excellent",
        }

        return {
            "score": score,
            "max_score": 7,
            "strength": strength_levels.get(score, "Unknown"),
            "feedback": feedback,
            "is_acceptable": score >= 3,
        }


# Legacy SHA256 migration helper
class LegacyPasswordMigrator:
    """
    Helper for migrating from SHA-256 passwords to bcrypt.
    """

    @staticmethod
    def is_sha256_hash(hash_string: str) -> bool:
        """Check if hash looks like SHA-256 (64 hex chars)."""
        return len(hash_string) == 64 and all(c in "0123456789abcdefABCDEF" for c in hash_string)

    @staticmethod
    def verify_sha256_password(password: str, sha256_hash: str) -> bool:
        """Verify password against SHA-256 hash (for migration only)."""
        computed_hash = hashlib.sha256(password.encode()).hexdigest()
        return computed_hash == sha256_hash

    @staticmethod
    def migrate_password(
        password: str, old_sha256_hash: str, new_hasher: SecurePasswordHasher
    ) -> str:
        """
        Migrate from SHA-256 to secure hash.

        Args:
            password: User's plain password
            old_sha256_hash: Stored SHA-256 hash
            new_hasher: SecurePasswordHasher instance

        Returns:
            New secure hash if migration succeeds

        Raises:
            ValueError: If current password doesn't match old hash
        """
        if not LegacyPasswordMigrator.verify_sha256_password(password, old_sha256_hash):
            raise ValueError("Current password verification failed")

        return new_hasher.hash_password(password)


# Global instances for easy import
password_hasher = SecurePasswordHasher()
legacy_migrator = LegacyPasswordMigrator()


# Backward compatibility functions (replace vulnerable hashlib.sha256 usage)
def hash_password_secure(password: str) -> str:
    """Secure replacement for hashlib.sha256(password.encode()).hexdigest()"""
    return password_hasher.hash_password(password)


def verify_password_secure(password: str, hashed_password: str) -> bool:
    """Secure password verification"""
    return password_hasher.verify_password(password, hashed_password)


def migrate_legacy_password(password: str, old_hash: str) -> str:
    """Migrate from SHA-256 to bcrypt"""
    return legacy_migrator.migrate_password(password, old_hash, password_hasher)


# Usage examples and documentation
if __name__ == "__main__":
    # Example usage
    hasher = SecurePasswordHasher()

    # Test password hashing
    password = "SecureP@ssw0rd123!"
    hashed = hasher.hash_password(password)
    print(f"Hashed password: {hashed}")

    # Test verification
    is_valid = hasher.verify_password(password, hashed)
    print(f"Password valid: {is_valid}")

    # Test password strength
    strength = hasher.check_password_strength(password)
    print(f"Password strength: {strength}")

    # Generate secure password
    secure_pwd = hasher.generate_secure_password()
    print(f"Generated secure password: {secure_pwd}")
