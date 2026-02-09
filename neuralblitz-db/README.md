# NeuralBlitz PostgreSQL Database Schema

## Overview

Comprehensive database schema for the NeuralBlitz ecosystem with ACID compliance, proper indexing, partitioning for time-series data, and automated triggers.

## Directory Structure

```
neuralblitz-db/
├── models/                 # SQLAlchemy ORM models
│   ├── __init__.py
│   ├── base.py            # Base model with common fields
│   ├── users.py           # User accounts and authentication
│   ├── quantum.py         # Quantum neuron states and history
│   ├── reality.py         # Multi-reality network configurations
│   ├── consciousness.py   # Consciousness evolution records
│   ├── audit.py           # GoldenDAG audit entries
│   ├── metrics.py         # Metrics and monitoring data
│   └── api.py             # API keys and partner configurations
├── migrations/            # Alembic migration files
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── seed/                  # Seed data for development
│   ├── __init__.py
│   └── seed_data.py
├── triggers/              # PostgreSQL trigger functions
│   ├── update_timestamp.sql
│   ├── audit_trigger.sql
│   └── neuron_state_trigger.sql
├── views/                 # Database views
│   ├── user_activity_view.sql
│   ├── consciousness_metrics_view.sql
│   └── reality_status_view.sql
└── config/                # Connection pool and configuration
    ├── __init__.py
    ├── database.py
    └── connection_pool.py
```

## Key Features

### 1. ACID Compliance
- All tables use proper transaction boundaries
- Foreign key constraints with appropriate CASCADE behaviors
- Row-level locking for concurrent operations

### 2. Partitioning Strategy
- Time-series tables (metrics, audit logs) use range partitioning by date
- Quantum state history partitioned by week for efficient querying
- Automatic partition management through triggers

### 3. Indexing
- B-tree indexes on primary keys and foreign keys
- GiST indexes for JSONB and array columns
- Partial indexes for filtered queries
- BRIN indexes for time-series data (lightweight)

### 4. Security
- Row-level security (RLS) policies for user data isolation
- Encrypted columns for sensitive data
- Audit logging for all modifications

### 5. Performance
- Connection pooling with PgBouncer configuration
- Read replicas configuration
- Materialized views for complex aggregations

## Schema Design

### Core Tables

1. **users** - User accounts and authentication
2. **api_keys** - API authentication and rate limiting
3. **partner_configs** - Third-party partner configurations
4. **quantum_neurons** - Current quantum neuron states
5. **quantum_neuron_history** - Historical neuron state changes (partitioned)
6. **reality_networks** - Multi-reality network configurations
7. **reality_nodes** - Nodes within reality networks
8. **consciousness_records** - Consciousness evolution tracking
9. **consciousness_snapshots** - Point-in-time consciousness states (partitioned)
10. **golden_dag_audit** - Immutable audit trail (partitioned)
11. **metrics** - Time-series metrics data (partitioned)
12. **monitoring_alerts** - Alert configurations and history

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run migrations: `alembic upgrade head`
3. Seed development data: `python seed/seed_data.py`
4. Verify schema: `psql -d neuralblitz -f verify_schema.sql`

## Maintenance

- Partition management runs automatically via cron
- Index maintenance recommended weekly
- VACUUM ANALYZE after large data imports