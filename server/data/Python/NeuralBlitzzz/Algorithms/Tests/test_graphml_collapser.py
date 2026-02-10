# UAID: NBX-TST-ALG-00009
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: GraphML Dependency Collapser (graphml_collapser.py, NBX-ALG-00009)
#
# Core Principle: Efficiency - validating the tools that simplify our complex architectures.

import pytest
import networkx as nx
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.graphml_collapser import GraphMLCollapser

# --- Test Fixtures ---

@pytest.fixture
def graphml_file_factory(tmp_path: Path):
    """A factory to create temporary GraphML files from NetworkX graphs."""
    def _create_file(graph: nx.Graph, filename: str) -> Path:
        p = tmp_path / filename
        nx.write_graphml(graph, str(p))
        return p
    return _create_file

# --- Test Cases ---

class TestGraphMLCollapser:

    def test_initialization_success(self, graphml_file_factory):
        """Tests successful initialization with a valid GraphML file."""
        G = nx.Graph([("A", "B")])
        gml_path = graphml_file_factory(G, "simple.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        assert collapser.original_node_count == 2
        assert collapser.original_edge_count == 1

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-005"):
            GraphMLCollapser("non_existent_graph.graphml")

    def test_initialization_malformed_file(self, tmp_path: Path):
        """Tests for ValueError on a corrupted GraphML file."""
        malformed_path = tmp_path / "malformed.graphml"
        malformed_path.write_text("<graphml><node id='n0'>") # Intentionally unclosed tag
        with pytest.raises(ValueError, match="ERR-PARSE-003"):
            GraphMLCollapser(str(malformed_path))

    def test_collapse_simple_linear_chain(self, graphml_file_factory):
        """The most common case: collapses a simple A->B->C->D chain into A->D."""
        G = nx.path_graph(4, create_using=nx.DiGraph) # 0 -> 1 -> 2 -> 3
        gml_path = graphml_file_factory(G, "linear_chain.graphml")
        
        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()
        
        assert nodes_removed == 2 # Nodes 1 and 2 should be removed
        assert collapser.graph.number_of_nodes() == 2
        assert collapser.graph.number_of_edges() == 1
        assert collapser.graph.has_edge(0, 3)
        assert not collapser.graph.has_node(1)
        assert not collapser.graph.has_node(2)

    def test_collapse_preserves_branch_points(self, graphml_file_factory):
        """Tests that a node with degree > 2 is not collapsed."""
        # A -> B -> C -> D
        #          ^
        #          |
        #          E
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "C")])
        gml_path = graphml_file_factory(G, "branch.graphml")

        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()

        assert nodes_removed == 1 # Only node B should be removed
        assert collapser.graph.number_of_nodes() == 4
        assert collapser.graph.has_edge("A", "C") # New edge
        assert collapser.graph.has_edge("E", "C") # Preserved edge
        assert collapser.graph.has_edge("C", "D") # Preserved edge
        assert not collapser.graph.has_node("B")

    def test_collapse_preserves_cycles(self, graphml_file_factory):
        """Tests that nodes within a small cycle (all degree 2) are not collapsed."""
        G = nx.cycle_graph(3) # A triangle graph
        gml_path = graphml_file_factory(G, "cycle.graphml")
        
        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()

        assert nodes_removed == 0
        assert collapser.graph.number_of_nodes() == 3
        assert collapser.graph.number_of_edges() == 3

    def test_collapse_preserves_attributes(self, graphml_file_factory):
        """Tests if node and edge attributes are correctly aggregated onto the new edge."""
        G = nx.path_graph(4) # 0-1-2-3
        # Add attributes to the elements that will be collapsed
        G.nodes[1]['importance'] = 'medium'
        G.edges[(0, 1)]['dependency_type'] = 'runtime'
        G.edges[(1, 2)]['weight'] = 5.0

        gml_path = graphml_file_factory(G, "attributed.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains()

        # The new edge should be between 0 and 2
        assert collapser.graph.has_edge(0, 2)
        new_edge_data = collapser.graph.get_edge_data(0, 2)
        
        # Check for aggregated attributes
        assert new_edge_data['node_importance'] == 'medium'
        assert new_edge_data['dependency_type'] == 'runtime'
        assert new_edge_data['weight'] == 5.0
        assert new_edge_data['collapsed_nodes'] == str([1])

    def test_get_stats_is_accurate(self, graphml_file_factory):
        """Tests the statistics reporting method."""
        G = nx.path_graph(5) # 0-1-2-3-4
        gml_path = graphml_file_factory(G, "stats_test.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains() # Should remove nodes 1, 2, 3

        stats = collapser.get_stats()
        
        assert stats["original_nodes"] == 5
        assert stats["original_edges"] == 4
        assert stats["collapsed_nodes"] == 2
        assert stats["collapsed_edges"] == 1
        assert stats["nodes_removed"] == 3
        assert stats["node_reduction_percent"] == "60.00%"

    def test_save_functionality(self, graphml_file_factory, tmp_path: Path):
        """Tests that the save method creates a file at the correct location."""
        G = nx.path_graph(3)
        gml_path = graphml_file_factory(G, "save_test.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains()

        # Test default save path
        default_save_path = collapser.save()
        assert default_save_path.name == "save_test_collapsed.graphml"
        assert default_save_path.exists()

        # Test custom save path
        custom_save_path = tmp_path / "custom_name.xml"
        collapser.save(str(custom_save_path))
        assert custom_save_path.exists()

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and networkx are installed:
    #    pip install pytest networkx
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_graphml_collapser.py
    
    print("This is a test file. Use 'pytest' to execute it.")