"""
Database Backup and Restore Manager
Handles PostgreSQL database backup operations
"""

import os
import subprocess
import gzip
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage database backups using pg_dump and pg_restore."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize backup manager.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.environ.get(
            "NEURALBLITZ_DATABASE_URL",
            "postgresql://neuralblitz:neuralblitz_password@localhost:5432/neuralblitz",
        )
        self.backup_dir = Path("/home/runner/workspace/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Create a compressed database backup.

        Args:
            output_dir: Directory to save backup (default: backups/)

        Returns:
            Path to backup file or None if failed
        """
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"neuralblitz_backup_{timestamp}.sql.gz"

            if output_dir:
                backup_path = Path(output_dir) / backup_name
            else:
                backup_path = self.backup_dir / backup_name

            backup_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Creating backup: {backup_path}")

            # Run pg_dump and compress
            cmd = [
                "pg_dump",
                "--no-owner",
                "--no-acl",
                "--clean",
                "--if-exists",
                "--verbose",
                self.database_url,
            ]

            with gzip.open(backup_path, "wb") as f:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    logger.error(f"pg_dump failed: {result.stderr.decode()}")
                    return None

                f.write(result.stdout)

            # Verify backup
            if backup_path.exists() and backup_path.stat().st_size > 0:
                logger.info(f"Backup created successfully: {backup_path}")

                # Create metadata file
                metadata_path = backup_path.with_suffix(".json")
                self._create_metadata(metadata_path, backup_path)

                return str(backup_path)
            else:
                logger.error("Backup file is empty")
                return None

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    def restore_backup(self, backup_file: str) -> bool:
        """Restore database from backup.

        Args:
            backup_file: Path to backup file (.sql or .sql.gz)

        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_file)

            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            logger.info(f"Restoring from backup: {backup_file}")

            # Check if compressed
            is_compressed = backup_path.suffix == ".gz"

            # Build restore command
            cmd = ["psql", self.database_url]

            if is_compressed:
                # Decompress and restore
                with gzip.open(backup_path, "rb") as f:
                    result = subprocess.run(cmd, stdin=f, capture_output=True, text=True)
            else:
                # Restore directly
                with open(backup_path, "r") as f:
                    result = subprocess.run(cmd, stdin=f, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Restore failed: {result.stderr}")
                return False

            logger.info("Restore completed successfully")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def list_backups(self) -> list:
        """List available backups.

        Returns:
            List of backup file paths
        """
        backups = []
        for backup_file in self.backup_dir.glob("neuralblitz_backup_*.sql.gz"):
            backups.append(
                {
                    "file": str(backup_file),
                    "size": backup_file.stat().st_size,
                    "created": datetime.fromtimestamp(backup_file.stat().st_mtime),
                }
            )

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Remove old backups, keeping only the most recent.

        Args:
            keep_count: Number of backups to keep

        Returns:
            Number of backups removed
        """
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return 0

        removed = 0
        for backup in backups[keep_count:]:
            try:
                Path(backup["file"]).unlink()
                # Also remove metadata if exists
                metadata = Path(backup["file"]).with_suffix(".json")
                if metadata.exists():
                    metadata.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove old backup: {e}")

        logger.info(f"Cleaned up {removed} old backups")
        return removed

    def _create_metadata(self, metadata_path: Path, backup_path: Path):
        """Create metadata file for backup."""
        import json

        metadata = {
            "backup_file": str(backup_path.name),
            "created_at": datetime.now().isoformat(),
            "size_bytes": backup_path.stat().st_size,
            "database_url": self._mask_password(self.database_url),
            "version": "1.0",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging."""
        try:
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(url)
            if parsed.password:
                masked = parsed._replace(netloc=f"{parsed.username}:****@{parsed.hostname}")
                return urlunparse(masked)
            return url
        except:
            return "****"
