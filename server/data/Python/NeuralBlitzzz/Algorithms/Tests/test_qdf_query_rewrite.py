# UAID: NBX-TST-ALG-00003
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: QDF-Aware Query Rewriter
# Verifies the correctness of the NBX-ALG-00003 algorithm.
#
# Core Principle: Epistemic Fidelity (ε₃) - Ensuring temporal relevance is correctly inferred.

import pytest
from Algorithms.Source.qdf_query_rewrite import classify_recency_heuristic, qdf_query_rewrite

# --- Test Cases for classify_recency_heuristic ---

@pytest.mark.parametrize("query, expected_qdf_score", [
    # QDF 5: Highest Recency
    ("what is the latest news right now?", 5),
    ("show me a live feed", 5),
    ("what is happening today?", 5),
    
    # QDF 4: High Recency
    ("recent developments in AI ethics", 4),
    ("what are the newest models in 2025?", 4),
    ("any updates this week?", 4),
    
    # QDF 3: Medium Recency
    ("what happened last month?", 3),
    ("events from a few weeks ago", 3),
    
    # QDF 2: Default/Neutral
    ("explain the synergy engine", 2),
    ("how does photosynthesis work?", 2),
    
    # QDF 1: Low Recency
    ("what were the highlights of last year?", 1),
    ("recap the events of 2024", 1),

    # QDF 0: Historical/Evergreen
    ("tell me the history of the roman empire", 0),
    ("biography of Marie Curie", 0),
    ("origin of the internet", 0),
    ("what happened in the 1980s?", 0),
    
    # Case insensitivity
    ("WHAT IS THE LATEST NEWS?", 5),
])
def test_classify_recency_heuristic(query, expected_qdf_score):
    """
    Tests the heuristic classifier against a variety of queries with
    different temporal intents.
    """
    assert classify_recency_heuristic(query) == expected_qdf_score, \
        f"Query '{query}' was misclassified."

# --- Test Cases for the full qdf_query_rewrite function ---

def test_qdf_rewrite_appends_flag_to_unflagged_queries():
    """
    Verifies that queries without a QDF flag have the correct one appended.
    """
    queries = [
        "latest developments", # Expects QDF=4
        "history of philosophy" # Expects QDF=0
    ]
    expected_rewritten = [
        "latest developments --QDF=4",
        "history of philosophy --QDF=0"
    ]
    assert qdf_query_rewrite(queries) == expected_rewritten

def test_qdf_rewrite_preserves_existing_flags():
    """
    Ensures that if a user explicitly provides a QDF flag, it is
    not overridden by the heuristic. This is a critical user-control feature.
    """
    queries = [
        "latest news --QDF=2", # User wants a broader search despite 'latest'
        "history of computing --QDF=5", # User wants latest info on a historic topic
        "some query --qdf=1", # Test lowercase
    ]
    # The function should not change these strings at all.
    assert qdf_query_rewrite(queries) == queries

def test_qdf_rewrite_handles_empty_list():
    """Tests the edge case of an empty input list."""
    assert qdf_query_rewrite([]) == []

def test_qdf_rewrite_handles_list_with_empty_strings():
    """Tests how the function handles empty or whitespace-only queries."""
    queries = ["", "   "]
    expected = [
        " --QDF=2",       # Empty string gets default QDF score
        "    --QDF=2"     # Whitespace preserved, gets default QDF score
    ]
    assert qdf_query_rewrite(queries) == expected

def test_qdf_rewrite_comprehensive_batch():
    """
    A larger, comprehensive test simulating a batch of mixed queries,
    as might be processed by HALIC.
    """
    batch_queries = [
        "What's the weather like right now?",
        "Explain the origins of the Transcendental Charter.",
        "Give me recent papers on SOPES theory.",
        "Summarize last year's financial report.",
        "NeuralBlitz UEF/SIMI architecture",
        "What are the best movies of the 1990s?",
        "live stream of the keynote address --QDF=5", # Explicit, should be kept
        "Explain the role of the Custodian"
    ]
    
    expected_output = [
        "What's the weather like right now? --QDF=5",
        "Explain the origins of the Transcendental Charter. --QDF=0",
        "Give me recent papers on SOPES theory. --QDF=4",
        "Summarize last year's financial report. --QDF=1",
        "NeuralBlitz UEF/SIMI architecture --QDF=2",
        "What are the best movies of the 1990s? --QDF=0",
        "live stream of the keynote address --QDF=5", # Kept as is
        "Explain the role of the Custodian --QDF=2"
    ]

    rewritten_batch = qdf_query_rewrite(batch_queries)
    assert rewritten_batch == expected_output