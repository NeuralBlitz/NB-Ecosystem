"""2026_02_09_1245-005_seed_data

Seed data migration - Initial system data

Revision ID: 005
Revises: 004
Create Date: 2026-02-09 12:45:00.000000

"""
from typing import Sequence, Union
from datetime import datetime, timedelta
import hashlib
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: str = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def hash_password(password: str) -> str:
    """Simple password hashing for seed data."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a demo API key."""
    return f"nb_{uuid.uuid4().hex}"


def upgrade() -> None:
    """Upgrade schema - Insert seed data."""
    
    # Insert admin user
    admin_id = uuid.uuid4()
    op.execute(f"""
        INSERT INTO users (id, email, username, hashed_password, full_name, role, is_active, is_verified, api_key_hash)
        VALUES (
            '{admin_id}',
            'admin@neuralblitz.ai',
            'admin',
            '{hash_password('admin_secure_pass_2026')}',
            'System Administrator',
            'ADMIN',
            true,
            true,
            '{hash_password(generate_api_key())}'
        );
    """)
    
    # Insert demo user
    demo_id = uuid.uuid4()
    op.execute(f"""
        INSERT INTO users (id, email, username, hashed_password, full_name, role, is_active, is_verified, api_key_hash)
        VALUES (
            '{demo_id}',
            'demo@neuralblitz.ai',
            'demo',
            '{hash_password('demo_secure_pass_2026')}',
            'Demo User',
            'USER',
            true,
            true,
            '{hash_password(generate_api_key())}'
        );
    """)
    
    # Insert partner user
    partner_id = uuid.uuid4()
    op.execute(f"""
        INSERT INTO users (id, email, username, hashed_password, full_name, role, is_active, is_verified, api_key_hash)
        VALUES (
            '{partner_id}',
            'partner@neuralblitz.ai',
            'partner',
            '{hash_password('partner_secure_pass_2026')}',
            'Integration Partner',
            'PARTNER',
            true,
            true,
            '{hash_password(generate_api_key())}'
        );
    """)
    
    # Insert demo quantum neurons
    neurons = [
        ("Default Spiking Neuron", "Standard quantum spiking neuron for general processing", 20.0, 1.0, 0.1, "COHERENT"),
        ("High-Current Neuron", "Neuron optimized for high-input scenarios", 50.0, 1.5, 0.05, "COHERENT"),
        ("Fast Decay Neuron", "Neuron with rapid membrane potential decay", 20.0, 0.5, 0.2, "SUPERPOSITION"),
        ("Entangled Pair Alpha", "Part of entangled neuron pair", 30.0, 1.0, 0.1, "ENTANGLED"),
        ("Entangled Pair Beta", "Part of entangled neuron pair", 30.0, 1.0, 0.1, "ENTANGLED"),
    ]
    
    neuron_ids = []
    for name, desc, current, threshold, decay, state in neurons:
        neuron_id = uuid.uuid4()
        neuron_ids.append(neuron_id)
        op.execute(f"""
            INSERT INTO quantum_neurons (id, name, description, current, threshold, decay_rate, state, parameters)
            VALUES (
                '{neuron_id}',
                '{name}',
                '{desc}',
                {current},
                {threshold},
                {decay},
                '{state}',
                '{{"demo": true, "version": "1.0"}}'::jsonb
            );
        """)
    
    # Create entanglement between pair
    if len(neuron_ids) >= 5:
        op.execute(f"""
            INSERT INTO quantum_entanglements (source_neuron_id, target_neuron_id, entanglement_strength, correlation_coefficient)
            VALUES (
                '{neuron_ids[3]}',
                '{neuron_ids[4]}',
                0.85,
                0.92
            );
        """)
    
    # Insert demo reality network
    network_id = uuid.uuid4()
    op.execute(f"""
        INSERT INTO reality_networks (
            id, name, description, num_realities, nodes_per_reality, 
            global_consciousness, cross_reality_coherence, evolution_cycles, status, configuration
        ) VALUES (
            '{network_id}',
            'Demo Multi-Reality Network',
            'Demonstration network with 4 parallel realities',
            4,
            50,
            0.15,
            0.23,
            100,
            'ACTIVE',
            '{{"demo": true, "auto_evolve": false}}'::jsonb
        );
    """)
    
    # Insert reality instances
    for i in range(4):
        reality_id = uuid.uuid4()
        consciousness = 0.1 + (i * 0.05)
        stability = 0.8 + (i * 0.05)
        op.execute(f"""
            INSERT INTO reality_instances (
                id, network_id, reality_index, name, consciousness_level, 
                stability, node_count, connection_density, status, metadata
            ) VALUES (
                '{reality_id}',
                '{network_id}',
                {i},
                'Reality-{i}',
                {consciousness:.2f},
                {stability:.2f},
                50,
                0.1,
                'ACTIVE',
                '{{"dimension": {i}, "energy": {100 + i * 10}}}'}::jsonb
            );
        """)
    
    # Log migration completion
    op.execute(f"""
        INSERT INTO migration_history (revision_id, revision_name, operation, success, executed_by)
        VALUES ('{revision}', 'seed_data', 'upgrade', true, 'system');
    """)


def downgrade() -> None:
    """Downgrade schema - Remove seed data."""
    
    # Delete seed data (in reverse order of creation)
    op.execute("DELETE FROM consciousness_events WHERE network_id IN (SELECT id FROM reality_networks WHERE configuration->>'demo' = 'true');")
    op.execute("DELETE FROM consciousness_snapshots WHERE network_id IN (SELECT id FROM reality_networks WHERE configuration->>'demo' = 'true');")
    op.execute("DELETE FROM reality_nodes WHERE reality_id IN (SELECT id FROM reality_instances WHERE metadata->>'demo' = 'true');")
    op.execute("DELETE FROM reality_instances WHERE metadata->>'demo' = 'true';")
    op.execute("DELETE FROM reality_networks WHERE configuration->>'demo' = 'true';")
    op.execute("DELETE FROM quantum_entanglements WHERE source_neuron_id IN (SELECT id FROM quantum_neurons WHERE parameters->>'demo' = 'true');")
    op.execute("DELETE FROM quantum_processes WHERE neuron_id IN (SELECT id FROM quantum_neurons WHERE parameters->>'demo' = 'true');")
    op.execute("DELETE FROM quantum_neurons WHERE parameters->>'demo' = 'true';")
    op.execute("DELETE FROM user_sessions WHERE user_id IN (SELECT id FROM users WHERE email LIKE '%@neuralblitz.ai');")
    op.execute("DELETE FROM audit_logs WHERE user_id IN (SELECT id FROM users WHERE email LIKE '%@neuralblitz.ai');")
    op.execute("DELETE FROM users WHERE email IN ('admin@neuralblitz.ai', 'demo@neuralblitz.ai', 'partner@neuralblitz.ai');")
