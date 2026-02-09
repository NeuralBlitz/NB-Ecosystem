#!/usr/bin/env python3
"""
NeuralBlitz Migration CLI
Command-line interface for database migration management
"""

import os
import sys
import subprocess
import typer
from typing import Optional
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nb_omnibus_router.db.models import get_engine, init_db

app = typer.Typer(
    name="nb-migrate",
    help="NeuralBlitz Database Migration Manager",
    add_completion=False,
)

# Global options
database_url_option = typer.Option(
    "--database-url",
    "-d",
    help="Database URL (overrides alembic.ini)",
    envvar="NEURALBLITZ_DATABASE_URL",
)

verbose_option = typer.Option("--verbose", "-v", help="Enable verbose output", is_flag=True)


def run_alembic_command(args: list, database_url: Optional[str] = None) -> int:
    """Execute an alembic command."""
    cmd = ["alembic"]

    if database_url:
        # Set environment variable for this command
        env = os.environ.copy()
        env["NEURALBLITZ_DATABASE_URL"] = database_url
        result = subprocess.run(cmd + args, env=env)
    else:
        result = subprocess.run(cmd + args)

    return result.returncode


@app.command()
def create(
    message: str = typer.Argument(..., help="Migration message/description"),
    auto: bool = typer.Option(True, "--auto/--manual", help="Auto-generate from models"),
    database_url: Optional[str] = database_url_option,
    verbose: bool = verbose_option,
):
    """Create a new migration script."""
    typer.echo(f"Creating migration: {message}")

    args = ["revision", "-m", message]

    if auto:
        args.append("--autogenerate")
        typer.echo("Auto-generating from SQLAlchemy models...")

    if verbose:
        args.append("--verbose")

    exit_code = run_alembic_command(args, database_url)

    if exit_code == 0:
        typer.echo("✓ Migration created successfully")

        # Show the latest migration file
        result = subprocess.run(["alembic", "history", "--verbose"], capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            if lines:
                typer.echo(f"\nLatest migration:\n{lines[0]}")
    else:
        typer.echo("✗ Migration creation failed", err=True)
        raise typer.Exit(code=exit_code)


@app.command()
def upgrade(
    revision: str = typer.Argument("head", help="Target revision (default: head)"),
    database_url: Optional[str] = database_url_option,
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before migration"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be migrated without executing"
    ),
    verbose: bool = verbose_option,
):
    """Apply database migrations (upgrade)."""

    if dry_run:
        typer.echo("DRY RUN - No changes will be made")

    typer.echo(f"Upgrading to revision: {revision}")

    if backup and not dry_run:
        typer.echo("Creating backup before migration...")
        # Call backup function
        backup_path = create_backup(database_url)
        if backup_path:
            typer.echo(f"✓ Backup created: {backup_path}")
        else:
            typer.echo("⚠ Backup failed, but continuing with migration...")

    args = ["upgrade"]
    if dry_run:
        args.append("--sql")
    args.append(revision)

    if verbose:
        args.append("--verbose")

    exit_code = run_alembic_command(args, database_url)

    if exit_code == 0:
        typer.echo(f"✓ Successfully upgraded to {revision}")

        # Log migration in history table
        log_migration(revision, "upgrade", database_url)
    else:
        typer.echo("✗ Upgrade failed", err=True)
        raise typer.Exit(code=exit_code)


@app.command()
def downgrade(
    revision: str = typer.Argument(..., help="Target revision (e.g., -1, base, or revision ID)"),
    database_url: Optional[str] = database_url_option,
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before downgrade"
    ),
    force: bool = typer.Option(False, "--force", help="Force downgrade without confirmation"),
    verbose: bool = verbose_option,
):
    """Rollback database migrations (downgrade)."""

    if not force:
        confirm = typer.confirm(
            f"⚠️  WARNING: This will DOWNGRADE the database to {revision}. "
            "Data loss may occur. Continue?"
        )
        if not confirm:
            typer.echo("Downgrade cancelled")
            raise typer.Exit()

    typer.echo(f"Downgrading to revision: {revision}")

    if backup:
        typer.echo("Creating backup before downgrade...")
        backup_path = create_backup(database_url)
        if backup_path:
            typer.echo(f"✓ Backup created: {backup_path}")

    args = ["downgrade", revision]

    if verbose:
        args.append("--verbose")

    exit_code = run_alembic_command(args, database_url)

    if exit_code == 0:
        typer.echo(f"✓ Successfully downgraded to {revision}")
        log_migration(revision, "downgrade", database_url)
    else:
        typer.echo("✗ Downgrade failed", err=True)
        raise typer.Exit(code=exit_code)


@app.command()
def history(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed history"),
    indicate_current: bool = typer.Option(True, "--indicate-current", help="Show current revision"),
):
    """Show migration history."""
    args = ["history"]

    if verbose:
        args.append("--verbose")

    if indicate_current:
        args.append("--indicate-current")

    subprocess.run(args)


@app.command()
def current(
    verbose: bool = verbose_option,
):
    """Show current database revision."""
    args = ["current"]

    if verbose:
        args.append("--verbose")

    subprocess.run(args)


@app.command()
def show(
    revision: str = typer.Argument(..., help="Revision ID to show"),
):
    """Show details of a specific revision."""
    subprocess.run(["alembic", "show", revision])


@app.command()
def stamp(
    revision: str = typer.Argument(..., help="Revision to stamp"),
    database_url: Optional[str] = database_url_option,
):
    """Stamp the database with a specific revision (without running migrations)."""
    typer.echo(f"Stamping database with revision: {revision}")

    exit_code = run_alembic_command(["stamp", revision], database_url)

    if exit_code == 0:
        typer.echo(f"✓ Database stamped to {revision}")
    else:
        typer.echo("✗ Stamp failed", err=True)
        raise typer.Exit(code=exit_code)


@app.command()
def check(
    database_url: Optional[str] = database_url_option,
):
    """Check if database is up to date."""
    typer.echo("Checking database status...")

    # Check current revision
    result = subprocess.run(["alembic", "current"], capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo("✗ Failed to check database status", err=True)
        raise typer.Exit(code=1)

    current_rev = result.stdout.strip()
    typer.echo(f"Current revision: {current_rev}")

    # Check if migrations are pending
    result = subprocess.run(["alembic", "history", "--verbose"], capture_output=True, text=True)

    if "head" in current_rev:
        typer.echo("✓ Database is up to date")
    else:
        typer.echo("⚠ Migrations are pending. Run 'nb-migrate upgrade' to apply.")


@app.command()
def test(
    database_url: Optional[str] = database_url_option,
    verbose: bool = verbose_option,
):
    """Test migrations (upgrade and downgrade cycle)."""
    typer.echo("Testing migrations...")

    # Create backup first
    typer.echo("1. Creating backup...")
    backup_path = create_backup(database_url)
    if not backup_path:
        typer.echo("✗ Failed to create backup", err=True)
        raise typer.Exit(code=1)

    try:
        # Get base revision
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", "test_migration"],
            capture_output=True,
            text=True,
        )

        # Test upgrade
        typer.echo("2. Testing upgrade to head...")
        exit_code = run_alembic_command(["upgrade", "head"], database_url)
        if exit_code != 0:
            raise Exception("Upgrade test failed")

        # Test downgrade
        typer.echo("3. Testing downgrade to base...")
        exit_code = run_alembic_command(["downgrade", "base"], database_url)
        if exit_code != 0:
            raise Exception("Downgrade test failed")

        # Restore
        typer.echo("4. Restoring from backup...")
        restore_from_backup(backup_path, database_url)

        typer.echo("✓ All migration tests passed")

    except Exception as e:
        typer.echo(f"✗ Migration test failed: {e}", err=True)
        # Restore from backup
        restore_from_backup(backup_path, database_url)
        raise typer.Exit(code=1)


@app.command()
def init(
    database_url: Optional[str] = database_url_option,
):
    """Initialize database (create tables without running migrations)."""
    typer.echo("Initializing database...")

    try:
        if database_url:
            engine = get_engine(database_url)
        else:
            engine = get_engine()

        init_db(engine)
        typer.echo("✓ Database initialized successfully")

        # Stamp with base revision
        subprocess.run(["alembic", "stamp", "head"])

    except Exception as e:
        typer.echo(f"✗ Failed to initialize database: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def backup(
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    database_url: Optional[str] = database_url_option,
):
    """Create a database backup."""
    backup_path = create_backup(database_url, output_dir)

    if backup_path:
        typer.echo(f"✓ Backup created: {backup_path}")
    else:
        typer.echo("✗ Backup failed", err=True)
        raise typer.Exit(code=1)


@app.command()
def restore(
    backup_file: str = typer.Argument(..., help="Path to backup file"),
    database_url: Optional[str] = database_url_option,
    force: bool = typer.Option(False, "--force", help="Force restore without confirmation"),
):
    """Restore database from backup."""
    if not force:
        confirm = typer.confirm(
            f"⚠️  WARNING: This will RESTORE the database from {backup_file}. "
            "All current data will be replaced. Continue?"
        )
        if not confirm:
            typer.echo("Restore cancelled")
            raise typer.Exit()

    if restore_from_backup(backup_file, database_url):
        typer.echo("✓ Database restored successfully")
    else:
        typer.echo("✗ Restore failed", err=True)
        raise typer.Exit(code=1)


# Helper functions


def create_backup(
    database_url: Optional[str] = None, output_dir: Optional[str] = None
) -> Optional[str]:
    """Create a database backup using pg_dump."""
    from nb_omnibus_router.db.backup import BackupManager

    manager = BackupManager(database_url)
    return manager.create_backup(output_dir)


def restore_from_backup(backup_file: str, database_url: Optional[str] = None) -> bool:
    """Restore database from backup."""
    from nb_omnibus_router.db.backup import BackupManager

    manager = BackupManager(database_url)
    return manager.restore_backup(backup_file)


def log_migration(revision: str, operation: str, database_url: Optional[str] = None):
    """Log migration execution in database."""
    try:
        from sqlalchemy import text
        from nb_omnibus_router.db.models import get_session, get_engine

        if database_url:
            engine = get_engine(database_url)
        else:
            engine = get_engine()

        session = get_session(engine)

        query = text("""
            INSERT INTO migration_history (revision_id, revision_name, operation, success, executed_by)
            VALUES (:revision, :name, :operation, true, :executed_by)
        """)

        session.execute(
            query,
            {
                "revision": revision,
                "name": f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "operation": operation,
                "executed_by": os.environ.get("USER", "cli"),
            },
        )

        session.commit()
        session.close()

    except Exception as e:
        typer.echo(f"Warning: Failed to log migration: {e}")


if __name__ == "__main__":
    app()
