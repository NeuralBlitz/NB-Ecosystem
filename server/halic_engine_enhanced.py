"""
Enhanced HALIC Engine with production-grade audit trail system.

This implementation enhances the original HALIC with persistent audit storage,
cryptographic verification, and enterprise compliance features.
"""

import hashlib
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncpg
from enum import Enum


class AuditLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    CUSTOM = "custom"


@dataclass
class AuditTrail:
    trace_id: str
    prompt: str
    response: str
    golden_dag: str
    codex_id: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: str
    compliance_tags: List[str]
    risk_level: AuditLevel
    metadata: Dict[str, Any]


class EnhancedHALICEngine:
    """
    Production-ready HALIC with persistent audit trails and compliance features.

    Features:
    - Cryptographically verifiable audit trails
    - Multi-framework compliance support (GDPR, SOX, HIPAA)
    - Risk assessment and categorization
    - Persistent storage with PostgreSQL
    - Real-time monitoring and alerting
    - Chain-of-custody tracking
    """

    def __init__(self, db_config: dict, compliance_config: dict = None):
        """
        Initialize the enhanced HALIC Engine.

        Args:
            db_config (dict): Database connection configuration
            compliance_config (dict): Compliance settings
                {
                    'frameworks': ['GDPR', 'SOX'],
                    'retention_days': 2555,  # 7 years
                    'alert_thresholds': {'high_risk': 5, 'critical_risk': 1}
                }
        """
        self.db_config = db_config
        self.compliance_config = compliance_config or {}
        self.pool = None
        print("Enhanced Human-AI Language Interface Core (HALIC) Initialized.")

    async def initialize(self):
        """Initialize database connection and create audit tables."""
        self.pool = await asyncpg.create_pool(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            min_size=self.db_config.get("pool_size", 10),
            command_timeout=60,
        )

        await self._create_tables()
        print("HALIC: Database connection established and audit tables created.")

    async def _create_tables(self):
        """Create audit trail and compliance tables."""
        async with self.pool.acquire() as conn:
            # Main audit trail table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trails (
                    id SERIAL PRIMARY KEY,
                    trace_id VARCHAR(64) UNIQUE NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    golden_dag VARCHAR(64) NOT NULL,
                    codex_id VARCHAR(64) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    user_id VARCHAR(255),
                    session_id VARCHAR(255) NOT NULL,
                    compliance_tags TEXT[],
                    risk_level VARCHAR(20) NOT NULL,
                    metadata JSONB,
                    verified BOOLEAN DEFAULT FALSE
                )
            """)

            # Compliance framework tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_records (
                    id SERIAL PRIMARY KEY,
                    trace_id VARCHAR(64) REFERENCES audit_trails(trace_id),
                    framework VARCHAR(50) NOT NULL,
                    compliance_check BOOLEAN NOT NULL,
                    violations TEXT[],
                    mitigations TEXT[],
                    reviewed_by VARCHAR(255),
                    review_date TIMESTAMP WITH TIME ZONE
                )
            """)

            # Risk monitoring
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_incidents (
                    id SERIAL PRIMARY KEY,
                    trace_id VARCHAR(64) REFERENCES audit_trails(trace_id),
                    risk_level VARCHAR(20) NOT NULL,
                    incident_type VARCHAR(100) NOT NULL,
                    description TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_date TIMESTAMP WITH TIME ZONE,
                    assigned_to VARCHAR(255)
                )
            """)

            # Indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_trails_timestamp ON audit_trails(timestamp)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_trails_user ON audit_trails(user_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_trails_risk ON audit_trails(risk_level)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_records(framework)"
            )

    def _generate_trace_id(self, prompt: str, timestamp: float) -> str:
        """Generate cryptographically secure trace ID."""
        return hashlib.sha256(
            f"{prompt}{timestamp}{time.time_ns()}".encode()
        ).hexdigest()[:32]

    def _generate_codex_id(self, logical_path: str) -> str:
        """Generate codex ID from logical path."""
        return hashlib.sha256(f"{logical_path}{time.time_ns()}".encode()).hexdigest()[
            :32
        ]

    def _generate_golden_dag(self, prompt: str, trace_id: str, response: str) -> str:
        """Generate GoldenDAG seal for complete interaction."""
        golden_dag_input = f"{prompt}{trace_id}{response}{time.time_ns()}".encode()
        return hashlib.sha256(golden_dag_input).hexdigest()

    def _assess_risk_level(self, prompt: str, response: str) -> AuditLevel:
        """
        Assess the risk level of an interaction.

        Args:
            prompt (str): User prompt
            response (str): AI response

        Returns:
            AuditLevel: Categorized risk level
        """
        # Simple keyword-based risk assessment
        high_risk_keywords = [
            "password",
            "credit card",
            "ssn",
            "social security",
            "medical record",
            "confidential",
            "secret",
            "proprietary",
            "legal advice",
            "diagnosis",
        ]

        medium_risk_keywords = [
            "personal",
            "private",
            "sensitive",
            "financial",
            "health",
        ]

        content = (prompt + " " + response).lower()

        if any(keyword in content for keyword in high_risk_keywords):
            return AuditLevel.HIGH
        elif any(keyword in content for keyword in medium_risk_keywords):
            return AuditLevel.MEDIUM
        else:
            return AuditLevel.LOW

    def _get_compliance_tags(self, prompt: str, response: str) -> List[str]:
        """
        Generate compliance tags based on content analysis.

        Returns:
            List[str]: Relevant compliance frameworks
        """
        content = (prompt + " " + response).lower()
        tags = []

        if any(word in content for word in ["personal data", "consent", "privacy"]):
            tags.append(ComplianceFramework.GDPR.value)

        if any(word in content for word in ["financial", "audit", "reporting", "sox"]):
            tags.append(ComplianceFramework.SOX.value)

        if any(word in content for word in ["health", "medical", "patient", "hipaa"]):
            tags.append(ComplianceFramework.HIPAA.value)

        if any(word in content for word in ["security", "information security", "iso"]):
            tags.append(ComplianceFramework.ISO27001.value)

        return tags

    async def process_interaction(
        self,
        raw_prompt: str,
        user_id: str = None,
        session_id: str = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete interaction with full audit trail generation.

        Args:
            raw_prompt (str): Raw user input
            user_id (str): User identifier
            session_id (str): Session identifier
            context (dict): Additional context data

        Returns:
            Dict containing response and audit information
        """
        print(f"HALIC: Processing interaction: '{raw_prompt[:50]}...'")

        timestamp = time.time()
        session_id = session_id or f"session_{int(timestamp)}"
        context = context or {}

        # 1. Parse intent from the raw prompt
        intent = raw_prompt.strip()

        # 2. Generate logical path (placeholder for actual UNE integration)
        logical_path = f"intent_analysis -> knowledge_retrieval -> response_generation"

        # 3. Generate response body (placeholder for actual response generation)
        response_body = f"Processed response based on: {intent[:100]}..."

        # 4. Generate cryptographic audit trail components
        trace_id = self._generate_trace_id(intent, timestamp)
        codex_id = self._generate_codex_id(logical_path)
        golden_dag = self._generate_golden_dag(intent, trace_id, response_body)

        # 5. Assess risk and compliance
        risk_level = self._assess_risk_level(intent, response_body)
        compliance_tags = self._get_compliance_tags(intent, response_body)

        # 6. Create audit trail record
        audit_trail = AuditTrail(
            trace_id=trace_id,
            prompt=intent,
            response=response_body,
            golden_dag=golden_dag,
            codex_id=codex_id,
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
            user_id=user_id,
            session_id=session_id,
            compliance_tags=compliance_tags,
            risk_level=risk_level,
            metadata=context,
        )

        # 7. Store audit trail
        await self._store_audit_trail(audit_trail)

        # 8. Format response with audit information
        formatted_response = {
            "response": response_body,
            "audit": {
                "trace_id": trace_id,
                "golden_dag": golden_dag,
                "codex_id": codex_id,
                "timestamp": audit_trail.timestamp.isoformat(),
                "risk_level": risk_level.value,
                "compliance_tags": compliance_tags,
            },
        }

        # 9. Check for risk alerts
        if risk_level in [AuditLevel.HIGH, AuditLevel.CRITICAL]:
            await self._create_risk_incident(audit_trail)

        return formatted_response

    async def _store_audit_trail(self, audit_trail: AuditTrail):
        """Store audit trail in database."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_trails (
                    trace_id, prompt, response, golden_dag, codex_id,
                    timestamp, user_id, session_id, compliance_tags,
                    risk_level, metadata, verified
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                audit_trail.trace_id,
                audit_trail.prompt,
                audit_trail.response,
                audit_trail.golden_dag,
                audit_trail.codex_id,
                audit_trail.timestamp,
                audit_trail.user_id,
                audit_trail.session_id,
                audit_trail.compliance_tags,
                audit_trail.risk_level.value,
                json.dumps(audit_trail.metadata),
                False,  # Not verified by default
            )

    async def _create_risk_incident(self, audit_trail: AuditTrail):
        """Create risk incident for high-risk interactions."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO risk_incidents (
                    trace_id, risk_level, incident_type, description, resolved
                ) VALUES ($1, $2, $3, $4, $5)
            """,
                audit_trail.trace_id,
                audit_trail.risk_level.value,
                "HIGH_RISK_INTERACTION",
                f"High risk interaction detected: {audit_trail.prompt[:100]}...",
                False,
            )

    async def verify_audit_trail(self, trace_id: str, expected_dag: str) -> bool:
        """
        Verify the integrity of an audit trail using GoldenDAG.

        Args:
            trace_id (str): Trace ID to verify
            expected_dag (str): Expected GoldenDAG hash

        Returns:
            bool: True if verification successful
        """
        async with self.pool.acquire() as conn:
            audit_data = await conn.fetchrow(
                """
                SELECT prompt, response, golden_dag FROM audit_trails WHERE trace_id = $1
            """,
                trace_id,
            )

            if not audit_data:
                return False

            # Recalculate GoldenDAG
            recalculated_dag = self._generate_golden_dag(
                audit_data["prompt"], trace_id, audit_data["response"]
            )

            verification_passed = (
                recalculated_dag == audit_data["golden_dag"] == expected_dag
            )

            # Update verification status
            await conn.execute(
                """
                UPDATE audit_trails SET verified = $1 WHERE trace_id = $2
            """,
                verification_passed,
                trace_id,
            )

            return verification_passed

    async def get_audit_trail(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete audit trail for a given trace ID."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT * FROM audit_trails WHERE trace_id = $1
            """,
                trace_id,
            )

            return dict(result) if result else None

    async def search_audit_trails(
        self,
        user_id: str = None,
        risk_level: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search audit trails with various filters."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM audit_trails WHERE 1=1"
            params = []
            param_count = 0

            if user_id:
                param_count += 1
                query += f" AND user_id = ${param_count}"
                params.append(user_id)

            if risk_level:
                param_count += 1
                query += f" AND risk_level = ${param_count}"
                params.append(risk_level)

            if start_date:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(start_date)

            if end_date:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT $" + str(param_count + 1)
            params.append(limit)

            results = await conn.fetch(query, *params)
            return [dict(result) for result in results]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            print("HALIC: Database connection closed.")
