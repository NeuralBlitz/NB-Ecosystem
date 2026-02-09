# NeuralBlitz Database Migration System

## Overview

NeuralBlitz uses Alembic for database migrations with PostgreSQL as the primary database. The migration system is designed to be idempotent, reversible, and production-ready with comprehensive backup/restore capabilities.

## Architecture

```
alembic/
├── alembic.ini          # Configuration
├── env.py               # Environment setup
├── script.py.mako       # Migration template
├── README               # Documentation
├── migration.log        # Migration logs
└── versions/            # Migration scripts
    ├── 2026_02_09_1145-001_initial_schema.py
    ├── 2026_02_09_1200-002_quantum_neuron_tables.py
    ├── 2026_02_09_1215-003_multi_reality_network_tables.py
    ├── 2026_02_09_1230-004_consciousness_tracking_tables.py
    └── 2026_02_09_1245-005_seed_data.py
```

## Configuration

### Environment Variables

```bash
# Database URL (overrides alembic.ini)
export NEURALBLITZ_DATABASE_URL="postgresql://user:pass@host:port/database"

# API Keys for CLI
export NEURALBLITZ_API_KEY="your-api-key"
```

### alembic.ini

The main configuration file:
- `sqlalchemy.url`: Default database connection
- `file_template`: Migration file naming convention
- `truncate_slug_length`: Maximum migration name length

## Migration Scripts

### 001_initial_schema
**Core tables and extensions**
- PostgreSQL extensions (uuid-ossp, pgcrypto)
- Users and authentication
- User sessions
- Audit logs
- System metrics
- Migration history

### 002_quantum_neuron_tables
**Quantum computing entities**
- Quantum neurons
- Quantum processes
- Quantum entanglements
- State tracking

### 003_multi_reality_network_tables
**Parallel reality simulation**
- Reality networks
- Reality instances
- Reality nodes
- Cross-reality coherence

### 004_consciousness_tracking_tables
**Emergence monitoring**
- Consciousness snapshots (partitioned)
- Consciousness events
- Automatic partition management

### 005_seed_data
**Initial system data**
- Admin user
- Demo users
- Sample neurons
- Demo reality network

## CLI Commands

### Create Migration
```bash
# Auto-generate from models
nb-migrate create "add user preferences"

# Manual migration
nb-migrate create "custom migration" --manual
```

### Apply Migrations
```bash
# Upgrade to latest
nb-migrate upgrade

# Upgrade to specific revision
nb-migrate upgrade 002

# Dry run
nb-migrate upgrade --dry-run
```

### Rollback Migrations
```bash
# Downgrade one step
nb-migrate downgrade -1

# Downgrade to base
nb-migrate downgrade base

# Force without confirmation
nb-migrate downgrade -1 --force
```

### Check Status
```bash
# Current revision
nb-migrate current

# Full history
nb-migrate history --verbose

# Check if up to date
nb-migrate check
```

### Backup Operations
```bash
# Create backup
nb-migrate backup --output /path/to/dir

# Restore from backup
nb-migrate restore /path/to/backup.sql.gz

# List backups
nb-migrate list-backups
```

### Testing
```bash
# Test migrations (upgrade/downgrade cycle)
nb-migrate test

# With custom database URL
nb-migrate test --database-url "postgresql://..."
```

## Best Practices

### Creating Migrations

1. **Always review auto-generated migrations**:
   ```bash
   alembic revision --autogenerate -m "description"
   # Review the generated file before committing
   ```

2. **Make migrations idempotent**:
   ```python
   # Good: Check before creating
   op.execute("CREATE TABLE IF NOT EXISTS ...")
   
   # Use proper constraints
   op.create_index(..., postgresql_if_not_exists=True)
   ```

3. **Handle data migrations separately**:
   - Use batch operations for large datasets
   - Consider table locking implications
   - Add progress logging for long operations

### Rollback Safety

1. **Test downgrades**:
   ```bash
   # Always test the full cycle
   nb-migrate test
   ```

2. **Backup before major migrations**:
   ```bash
   # Automatic backup (default)
   nb-migrate upgrade --backup
   
   # Skip backup (not recommended)
   nb-migrate upgrade --no-backup
   ```

3. **Use transactions**:
   ```python
   # Alembic wraps migrations in transactions by default
   # For PostgreSQL DDL, use:
   op.execute("COMMIT")  # If you need to break transaction
   ```

### Production Deployment

1. **Pre-deployment checklist**:
   - [ ] Test on staging environment
   - [ ] Create backup
   - [ ] Review migration execution time
   - [ ] Check disk space
   - [ ] Schedule during maintenance window

2. **Deployment process**:
   ```bash
   # 1. Backup production
   nb-migrate backup --output /backups/pre-deploy
   
   # 2. Run migrations
   nb-migrate upgrade
   
   # 3. Verify
   nb-migrate check
   ```

3. **Rollback procedure**:
   ```bash
   # 1. Identify target revision
   nb-migrate history
   
   # 2. Downgrade
   nb-migrate downgrade <revision>
   
   # 3. Or restore from backup
   nb-migrate restore /backups/pre-deploy/backup.sql.gz
   ```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Database Migration Tests

on: [push, pull_request]

jobs:
  test-migrations:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: neuralblitz
          POSTGRES_PASSWORD: neuralblitz_password
          POSTGRES_DB: neuralblitz_test
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install alembic psycopg2-binary
      
      - name: Run migration tests
        env:
          NEURALBLITZ_DATABASE_URL: postgresql://neuralblitz:neuralblitz_password@localhost:5432/neuralblitz_test
        run: |
          nb-migrate test
          
      - name: Test upgrade/downgrade cycle
        env:
          NEURALBLITZ_DATABASE_URL: postgresql://neuralblitz:neuralblitz_password@localhost:5432/neuralblitz_test
        run: |
          nb-migrate downgrade base
          nb-migrate upgrade head
```

## Troubleshooting

### Common Issues

1. **Migration fails with locked table**:
   ```sql
   -- Check locks
   SELECT * FROM pg_locks WHERE NOT granted;
   
   -- Terminate blocking process
   SELECT pg_terminate_backend(pid);
   ```

2. **Alembic can't find models**:
   ```python
   # Ensure env.py imports models
   from nb_omnibus_router.db.models import Base
   target_metadata = Base.metadata
   ```

3. **Downgrade fails**:
   - Check for foreign key constraints
   - Verify data dependencies
   - May need manual intervention

### Recovery

1. **Stuck migration**:
   ```bash
   # Stamp to specific version without running
   nb-migrate stamp <revision>
   ```

2. **Corrupted alembic_version**:
   ```sql
   -- Manual fix
   DELETE FROM alembic_version;
   INSERT INTO alembic_version (version_num) VALUES ('<correct_revision>');
   ```

## Database Schema

### Entity Relationship Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│    users    │────<│  user_sessions  │     │   quantum_neurons│
├─────────────┤     ├─────────────────┤     ├──────────────────┤
│ id (PK)     │     │ id (PK)         │     │ id (PK)          │
│ email       │     │ user_id (FK)    │     │ name             │
│ username    │     │ session_token   │     │ state            │
│ role        │     │ expires_at      │     │ current          │
└─────────────┘     └─────────────────┘     └──────────────────┘
       │                                              │
       │                                              │
       ▼                                              ▼
┌─────────────────┐                         ┌──────────────────┐
│   audit_logs    │                         │ quantum_processes│
├─────────────────┤                         ├──────────────────┤
│ id (PK)         │                         │ id (PK)          │
│ user_id (FK)    │                         │ neuron_id (FK)   │
│ action          │                         │ input_data       │
│ resource_type   │                         │ output_data      │
└─────────────────┘                         └──────────────────┘

┌─────────────────────┐     ┌─────────────────────┐
│  reality_networks   │────<│  reality_instances  │
├─────────────────────┤     ├─────────────────────┤
│ id (PK)             │     │ id (PK)             │
│ name                │     │ network_id (FK)     │
│ num_realities       │     │ reality_index       │
│ global_consciousness│     │ consciousness_level │
└─────────────────────┘     └─────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────────┐
                           │    reality_nodes    │
                           ├─────────────────────┤
                           │ id (PK)             │
                           │ reality_id (FK)     │
                           │ node_index          │
                           │ activation_level    │
                           └─────────────────────┘

┌──────────────────────────┐
│  consciousness_snapshots │
├──────────────────────────┤
│ id (PK)                  │
│ network_id (FK)          │
│ consciousness_level      │
│ captured_at              │
└──────────────────────────┘
```

## Support

For migration issues:
1. Check migration logs: `alembic/migration.log`
2. Review troubleshooting section
3. Contact: devops@neuralblitz.ai
