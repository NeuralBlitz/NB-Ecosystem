"""
Unit tests for the Dynamic Representational Substrate (DRS) Engine.

These tests ensure the reliability of the DRS's core functions: storing,
retrieving, and connecting concepts.
"""

import pytest
from core_engine.drs_engine.drs_manager import DRSEngine

@pytest.fixture
def drs_instance():
    """Provides a clean instance of the DRSEngine for each test."""
    config = {"backend": "in-memory"}
    return DRSEngine(config)

def test_drs_initialization(drs_instance):
    """
    Tests that the DRSEngine initializes with an empty storage.
    """
    assert drs_instance is not None
    assert drs_instance._storage == {}

def test_store_and_query_concept(drs_instance):
    """
    Tests the basic store and query functionality.
    """
    concept_id = "UNE"
    concept_data = {"description": "Universal Neural Engine"}
    drs_instance.store(concept_id, concept_data)

    retrieved = drs_instance.query(concept_id)
    assert retrieved is not None
    assert retrieved["data"]["description"] == "Universal Neural Engine"

def test_query_nonexistent_concept(drs_instance):
    """
    Tests that querying a concept that does not exist returns None.
    """
    retrieved = drs_instance.query("NONEXISTENT_CONCEPT")
    assert retrieved is None

def test_find_direct_connection(drs_instance):
    """
    Tests the ability to find a direct, defined connection between two concepts.
    """
    connections_list = [{"target": "DRS", "relation": "depends_on"}]
    drs_instance.store("UNE", {"description": "The mind"}, connections=connections_list)
    drs_instance.store("DRS", {"description": "The memory"})

    path = drs_instance.find_connections("UNE", "DRS")
    assert path == ["UNE", "depends_on", "DRS"]

def test_find_no_connection(drs_instance):
    """
    Tests the case where two concepts exist but have no defined path.
    """
    drs_instance.store("CONCEPT_A", {})
    drs_instance.store("CONCEPT_B", {})

    # Based on our current placeholder logic for indirect paths
    path = drs_instance.find_connections("CONCEPT_A", "CONCEPT_B")
    assert path == ["CONCEPT_A", "--> (indirect path) -->", "CONCEPT_B"]

def test_find_connection_to_nonexistent_node(drs_instance):
    """
    Tests that finding a connection returns an empty list if a concept doesn't exist.
    """
    drs_instance.store("CONCEPT_A", {})
    path = drs_instance.find_connections("CONCEPT_A", "NONEXISTENT_CONCEPT")
    assert path == []
