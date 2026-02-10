# UAID: NBX-ALG-00003
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: QDF-Aware Query Rewriter
# Part of the InterfaceLayer (HALIC)
#
# Core Principle: Epistemic Fidelity (ε₃) - ensures retrieval relevance.

import re
from typing import List, Tuple

def classify_recency_heuristic(text: str) -> int:
    """
    Analyzes a query string to heuristically determine its temporal intent and assign a
    QDF (Query-Deserved Freshness) score. The score ranges from 5 (most recent)
    to 0 (historical/evergreen).

    This logic is a key component of the HALIC pre-processing pipeline.
    See Codex Universalis, Vol. X, Section X-4 for the QDF matrix.

    Args:
        text (str): The natural language user query.

    Returns:
        int: The calculated QDF score.
    """
    # QDF 5: Highest recency, events happening now or today.
    if re.search(r"\b(live|breaking|right now|what's happening now)\b", text, re.I):
        return 5
    if re.search(r"\b(today|latest update)\b", text, re.I):
        return 5

    # QDF 4: High recency, within the last week or current year for major topics.
    if re.search(r"\b(this week|recent|latest|what's new|newest)\b", text, re.I):
        return 4
    if re.search(r"\b(in 2025|this year)\b", text, re.I): # Assuming current year is 2025
        return 4
    
    # QDF 3: Medium recency, within the last few months.
    if re.search(r"\b(this month|last month|a few weeks ago)\b", text, re.I):
        return 3

    # QDF 1: Low recency, historical but within the last few years.
    if re.search(r"\b(last year|a year ago|in 2024)\b", text, re.I):
        return 1

    # QDF 0: Evergreen or deep history, no recency preference.
    if re.search(r"\b(history of|ancient|origin of|biography of|in the 19\d\ds|in the 18\d\ds)\b", text, re.I):
        return 0
    
    # QDF 2: Default for queries with no clear temporal indicators. A neutral setting.
    return 2

def qdf_query_rewrite(queries: List[str]) -> List[str]:
    """
    Takes a list of user queries and injects a '--QDF=<n>' flag if one does
    not already exist. The QDF score is determined by the classify_recency_heuristic.

    This function respects explicitly provided QDF flags by the user.

    Args:
        queries (List[str]): A list of user-provided query strings.

    Returns:
        List[str]: A list of rewritten queries, now with QDF flags where appropriate.
    """
    rewritten_queries = []
    # Regex to check if a QDF flag already exists in the query
    qdf_pattern = re.compile(r'--QDF=\d', re.I)

    for q in queries:
        if qdf_pattern.search(q):
            # Respect user-provided flag, add it as is.
            rewritten_queries.append(q)
            continue
        
        # Classify and append the new flag
        qdf_score = classify_recency_heuristic(q)
        rewritten_queries.append(f"{q} --QDF={qdf_score}")
        
    return rewritten_queries

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # This block simulates the behavior of this algorithm as it would be
    # invoked by the HALIC subsystem on a batch of incoming prompts.

    print("--- Initiating NeuralBlitz QDF Query Rewriter Simulation ---")

    sample_queries = [
        "What is the latest news on quantum computing?",
        "Tell me about the history of the Roman Empire.",
        "How does the Synergy Engine work in NeuralBlitz?",
        "What were the major events of last year?",
        "Show me a live feed of the market.",
        "Explain the origin of the UEF/SIMI framework.",
        "What are the best practices for AI ethics in 2025?",
        "Show me a biography of Alan Turing.",
        "Find info on the Peloponnesian War --QDF=0" # Explicit flag
    ]

    print("\n[Original User Queries]")
    for query in sample_queries:
        print(f"  - {query}")

    # --- Run the rewrite algorithm ---
    rewritten = qdf_query_rewrite(sample_queries)

    print("\n[HALIC Pre-processed Queries (with QDF flags)]")
    for original, new in zip(sample_queries, rewritten):
        print(f"  - Original: {original}")
        print(f"    Rewritten: {new}\n")

    print("--- Simulation Complete ---")
    print("NBCL verbs would now be dispatched to Veritas/web.run with recency hints.")