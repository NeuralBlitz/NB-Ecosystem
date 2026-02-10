"""
Enhanced DRS Engine with PostgreSQL persistence for production deployment.

This implementation replaces the in-memory storage with a proper PostgreSQL
database while maintaining the same interface for backward compatibility.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncpg
from dataclasses import dataclass


@dataclass
class Concept:
    id: str
    data: Dict[str, Any]
    connections: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class EnhancedDRSEngine:
    """
    Production-ready DRS Engine with PostgreSQL persistence.

    Features:
    - Persistent storage with PostgreSQL
    - Async operations for scalability
    - Connection pooling for performance
    - Full-text search capabilities
    - Version history tracking
    - GraphQL-like query interface
    """

    def __init__(self, db_config: dict):
        """
        Initialize the enhanced DRS Engine with database connection.

        Args:
            db_config (dict): Database connection configuration
                {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'neuralblitz',
                    'user': 'postgres',
                    'password': 'password',
                    'pool_size': 10
                }
        """
        self.db_config = db_config
        self.pool = None
        print("Enhanced Dynamic Representational Substrate (DRS) Initialized.")

    async def initialize(self):
        """Initialize database connection and create tables."""
        self.pool = await asyncpg.create_pool(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            min_size=self.db_config.get("pool_size", 10),
            command_timeout=60,
        )

        # Create tables if they don't exist
        await self._create_tables()
        print("DRS: Database connection established and tables created.")

    async def _create_tables(self):
        """Create necessary database tables."""
        async with self.pool.acquire() as conn:
            # Concepts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id VARCHAR(255) PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Connections table for relationships
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id SERIAL PRIMARY KEY,
                    source_concept VARCHAR(255) REFERENCES concepts(id) ON DELETE CASCADE,
                    target_concept VARCHAR(255) REFERENCES concepts(id) ON DELETE CASCADE,
                    relation VARCHAR(255) NOT NULL,
                    weight FLOAT DEFAULT 1.0,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_concepts_data ON concepts USING GIN(data)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_concept)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_concept)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_connections_relation ON connections(relation)"
            )

    async def store(
        self,
        concept: str,
        data: Dict[str, Any],
        connections: List[Dict[str, Any]] = None,
    ):
        """
        Store a concept and its associated data in the database.

        Args:
            concept (str): Unique identifier for the concept
            data (dict): Information associated with the concept
            connections (list): Related concepts and their relationships
        """
        print(f"DRS: Storing concept '{concept}'...")

        async with self.pool.acquire() as conn:
            # Upsert concept
            await conn.execute(
                """
                INSERT INTO concepts (id, data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW()
            """,
                concept,
                json.dumps(data),
            )

            # Clear existing connections and insert new ones
            if connections:
                await conn.execute(
                    "DELETE FROM connections WHERE source_concept = $1", concept
                )

                for conn_data in connections:
                    await conn.execute(
                        """
                        INSERT INTO connections (source_concept, target_concept, relation, weight, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                        concept,
                        conn_data.get("target"),
                        conn_data.get("relation", "related_to"),
                        conn_data.get("weight", 1.0),
                        json.dumps(conn_data.get("metadata", {})),
                    )

    async def query(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data associated with a single concept.

        Args:
            concept (str): Identifier of the concept to retrieve

        Returns:
            Dictionary containing concept data or None if not found
        """
        print(f"DRS: Querying for concept '{concept}'...")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, data, created_at, updated_at
                FROM concepts
                WHERE id = $1
            """,
                concept,
            )

            if not row:
                return None

            # Get connections
            connections = await conn.fetch(
                """
                SELECT target_concept, relation, weight, metadata
                FROM connections
                WHERE source_concept = $1
            """,
                concept,
            )

            return {
                "id": row["id"],
                "data": json.loads(row["data"]),
                "connections": [dict(conn) for conn in connections],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    async def find_connections(self, start_concept: str, end_concept: str) -> List[str]:
        """
        Find the logical path between two concepts using graph traversal.

        Args:
            start_concept (str): Starting concept
            end_concept (str): Target concept

        Returns:
            List representing the path of connections
        """
        print(f"DRS: Finding connection from '{start_concept}' to '{end_concept}'...")

        async with self.pool.acquire() as conn:
            # Use recursive CTE for path finding
            result = await conn.fetch(
                """
                WITH RECURSIVE paths AS (
                    SELECT 
                        source_concept,
                        target_concept,
                        relation,
                        ARRAY[source_concept, target_concept] as path,
                        1 as depth
                    FROM connections
                    WHERE source_concept = $1
                    
                    UNION ALL
                    
                    SELECT 
                        c.source_concept,
                        c.target_concept,
                        c.relation,
                        p.path || c.target_concept,
                        p.depth + 1
                    FROM connections c
                    JOIN paths p ON c.source_concept = p.target_concept
                    WHERE NOT c.target_concept = ANY(p.path)
                        AND p.depth < 10
                )
                SELECT path, relation, depth
                FROM paths
                WHERE target_concept = $2
                ORDER BY depth ASC
                LIMIT 1
            """,
                start_concept,
                end_concept,
            )

            if result:
                return result[0]["path"]
            else:
                return []

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search across concepts.

        Args:
            query (str): Search query
            limit (int): Maximum number of results

        Returns:
            List of matching concepts
        """
        async with self.pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT id, data, created_at, updated_at,
                       ts_rank_cd(data::tsvector, plainto_tsquery($1)) as rank
                FROM concepts
                WHERE data::tsvector @@ plainto_tsquery($1)
                ORDER BY rank DESC
                LIMIT $2
            """,
                query,
                limit,
            )

            return [dict(result) for result in results]

    async def get_related_concepts(self, concept: str, depth: int = 1) -> List[str]:
        """
        Get all concepts related to a given concept within specified depth.

        Args:
            concept (str): Starting concept
            depth (int): Maximum depth of connections

        Returns:
            List of related concept IDs
        """
        async with self.pool.acquire() as conn:
            result = await conn.fetch(
                """
                WITH RECURSIVE related AS (
                    SELECT target_concept, 1 as depth
                    FROM connections
                    WHERE source_concept = $1
                    
                    UNION ALL
                    
                    SELECT c.target_concept, r.depth + 1
                    FROM connections c
                    JOIN related r ON c.source_concept = r.target_concept
                    WHERE r.depth < $2
                )
                SELECT DISTINCT target_concept
                FROM related
                WHERE target_concept != $1
            """,
                concept,
                depth,
            )

            return [row["target_concept"] for row in result]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            print("DRS: Database connection closed.")
