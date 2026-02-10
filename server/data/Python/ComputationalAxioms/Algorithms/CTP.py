import networkx as nx
import hashlib
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Tuple

# --- Constants ---

# CTP-Signature (Provenance) generation. Hash data, links, and time.
def create_ctp_signature(data: Dict[str, Any], timestamp: datetime, predecessors: List[str]) -> str:
    """Creates a verifiable hash for a CTP event."""
    # Ensure deterministic serialization order for consistent hashing
    data_str = json.dumps(data, sort_keys=True)
    # Combine data, time, and predecessors to form the provenance signature
    combined_input = f"{data_str}|{timestamp.isoformat()}|{sorted(predecessors)}"
    return hashlib.sha256(combined_input.encode('utf-8')).hexdigest()

# --- Causal Set Model ---

class CTPEvent:
    """Represents a single event node in the CTP graph."""
    def __init__(self, event_id: str, description: str, data: Dict[str, Any], timestamp: datetime):
        self.id = event_id
        self.description = description
        self.data = data
        self.timestamp = timestamp
        # The true signature must be calculated based on predecessors
        self.ctp_signature: str = ""

    def __repr__(self) -> str:
        return f"CTPEvent(id='{self.id}', desc='{self.description}')"

def simulate_decision_chain(num_input_sources: int = 3, num_reasoning_steps: int = 5) -> Tuple[nx.DiGraph, str]:
    """
    Simulates a complex decision-making process and creates a CTP graph.
    The process creates: Genesis Events -> Reasoning Steps -> Decision -> Outcome.
    """
    ctp_graph = nx.DiGraph()
    events: Dict[str, CTPEvent] = {}
    genesis_events: List[str] = []
    current_time = datetime.now()

    # Phase 1: Create initial Genesis Events (Inputs)
    for i in range(num_input_sources):
        event_id = f"input_{i+1}"
        description = f"Initial data source #{i+1}"
        data = {"source": f"DataStream-{i+1}", "value": random.randint(100, 1000)}
        events[event_id] = CTPEvent(event_id, description, data, current_time)
        genesis_events.append(event_id)

    # Phase 2: Create Reasoning Steps (Intermediate Processing)
    for step in range(num_reasoning_steps):
        event_id = f"reason_{step+1}"
        description = f"Step {step+1}: Analyzing data set"
        current_time += timedelta(milliseconds=random.randint(50, 150)) # Increment time
        
        # Determine predecessors. For step 1, predecessors are genesis events.
        # For subsequent steps, predecessors are previous reasoning steps.
        predecessor_ids = genesis_events if step == 0 else [f"reason_{step}"]
        
        data = {"analysis_result": f"Inferred pattern {step}", "temp_data": random.uniform(0.1, 0.9)}
        events[event_id] = CTPEvent(event_id, description, data, current_time)
        
        # Add edges (causal links)
        ctp_graph.add_edges_from([(pred_id, event_id) for pred_id in predecessor_ids])

    # Phase 3: Create Final Outcome
    final_time = current_time + timedelta(milliseconds=random.randint(100, 200))
    final_event_id = "final_decision"
    final_description = "Policy recommendation generated"
    final_data = {"recommendation": "Go-NoGo Policy", "score": random.uniform(0.8, 1.0)}
    predecessor_ids = [f"reason_{num_reasoning_steps}"]
    
    events[final_event_id] = CTPEvent(final_event_id, final_description, final_data, final_time)
    ctp_graph.add_edges_from([(pred_id, final_event_id) for pred_id in predecessor_ids])

    # Calculate and apply Provenance Signatures for all nodes
    for event_id, event in events.items():
        predecessor_ids = list(ctp_graph.predecessors(event_id))
        event.ctp_signature = create_ctp_signature(event.data, event.timestamp, predecessor_ids)

    # Store event objects in graph nodes for traceability
    for event_id, event in events.items():
        ctp_graph.nodes[event_id]['event_object'] = event

    return ctp_graph, final_event_id

# --- Causal Chain Tracing Algorithm ---

def trace_causal_chain(graph: nx.DiGraph, final_event_id: str) -> List[CTPEvent]:
    """
    Recursively back-traces a decision chain to find all contributing events.
    """
    traced_events: Set[str] = set()
    causal_stack: List[str] = [final_event_id]

    while causal_stack:
        current_event_id = causal_stack.pop()

        if current_event_id in traced_events:
            continue
        traced_events.add(current_event_id)

        # Find all direct predecessors of the current event
        for predecessor_id in graph.predecessors(current_event_id):
            causal_stack.append(predecessor_id)

    # Convert node IDs back to CTPEvent objects for analysis
    return [graph.nodes[event_id]['event_object'] for event_id in traced_events]

# --- Verification Routines (Veritas Engine) ---

def verify_provenance_integrity(traced_events: List[CTPEvent], graph: nx.DiGraph) -> bool:
    """Verifies that the provenance signature of each event is valid."""
    print("\n[Audit] Verifying Provenance Signatures...")
    for event in traced_events:
        predecessor_ids = list(graph.predecessors(event.id))
        recalculated_signature = create_ctp_signature(event.data, event.timestamp, predecessor_ids)
        
        if recalculated_signature != event.ctp_signature:
            print(f"  [FAILURE] Provenance mismatch for '{event.id}'. Stored: {event.ctp_signature[:10]}... Recalculated: {recalculated_signature[:10]}...")
            return False
        # print(f"  [SUCCESS] Provenance verified for '{event.id}'.")
    return True

def verify_temporal_invariants(traced_events: List[CTPEvent], graph: nx.DiGraph) -> bool:
    """Verifies that time flow is consistent with causal flow (e.g., cause < effect)."""
    print("\n[Audit] Verifying Temporal Invariants...")
    events_by_id = {event.id: event for event in traced_events}
    
    for event_id, event in events_by_id.items():
        for predecessor_id in graph.predecessors(event_id):
            predecessor_event = events_by_id[predecessor_id]
            if predecessor_event.timestamp > event.timestamp:
                print(f"  [FAILURE] Temporal anomaly detected: '{predecessor_id}' ({predecessor_event.timestamp}) caused '{event_id}' ({event.timestamp}). Time order violated.")
                return False
    return True

# --- Demonstration ---

if __name__ == "__main__":
    # 1. Simulate the decision chain for a specific scenario
    print("Simulating a new decision chain for a system...")
    ctp_graph, final_decision_id = simulate_decision_chain()
    
    # 2. Reconstruct the causal chain back to its sources
    print("\nReconstructing causal chain back to source inputs...")
    traced_events = trace_causal_chain(ctp_graph, final_decision_id)

    # 3. Print the reconstructed chain in reverse chronological order
    traced_events.sort(key=lambda x: x.timestamp, reverse=True)
    print(f"\n--- Causal Chain Reconstructed ({len(traced_events)} events total) ---")
    for i, event in enumerate(traced_events):
        predecessors = list(ctp_graph.predecessors(event.id))
        predecessor_names = ", ".join([p for p in predecessors]) if predecessors else "None"
        print(f"[{i:02}] {event.timestamp.time()} | {event.description:<30} | Predecessors: {predecessor_names}")

    # 4. Perform integrity audit (Veritas Engine checks)
    print("\n--- Auditing Reconstructed Chain ---")
    provenance_ok = verify_provenance_integrity(traced_events, ctp_graph)
    temporal_ok = verify_temporal_invariants(traced_events, ctp_graph)

    if provenance_ok and temporal_ok:
        print("\n[SUCCESS] CTP Audit Passed: Chain integrity verified against provenance and temporal invariants.")
    else:
        print("\n[FAILURE] CTP Audit Failed: Inconsistencies detected in the causal chain.")
