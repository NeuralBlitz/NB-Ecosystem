# UAID: NBX-ALG-00009
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: GraphML Dependency Collapser
# Part of the Custodian and Architecton Subsystems
#
# Core Principle: Efficiency - Optimizing complex structures for faster verification.

import networkx as nx
from pathlib import Path
from typing import Optional

class GraphMLCollapser:
    """
    Optimizes a GraphML file by collapsing linear chains of degree-2 nodes,
    preserving the overall topology while significantly reducing node and edge counts.
    This is critical for pre-processing dependency graphs for faster audits.
    """

    def __init__(self, gml_path: str):
        """
        Initializes the collapser with the path to the input GraphML file.

        Args:
            gml_path (str): The path to the source .graphml or .xml file.
        """
        self.gml_path = Path(gml_path)
        if not self.gml_path.exists():
            raise FileNotFoundError(f"ERR-FS-005: Input graph file not found at '{self.gml_path}'")
        
        try:
            self.graph = nx.read_graphml(self.gml_path)
            self.original_node_count = self.graph.number_of_nodes()
            self.original_edge_count = self.graph.number_of_edges()
        except Exception as e:
            # Catches parsing errors from malformed GraphML
            raise ValueError(f"ERR-PARSE-003: Failed to parse GraphML file '{self.gml_path}'. Reason: {e}")

    def collapse_chains(self, preserve_attributes: bool = True) -> int:
        """
        Iteratively finds and collapses linear chains of degree-2 nodes.
        A node is part of such a chain if it has exactly two neighbors
        (one incoming, one outgoing in a DiGraph, or just two in a Graph)
        and its neighbors are not connected to each other.

        Args:
            preserve_attributes (bool): If True, attempts to merge attributes
                                        from collapsed nodes into the new direct edge.

        Returns:
            int: The number of nodes removed.
        """
        if self.original_node_count == 0:
            return 0
            
        nodes_removed_count = 0
        
        # Iteratively collapse until no more degree-2 nodes can be removed.
        # A while loop is necessary because collapsing one node can change the
        # degree of its neighbors, potentially creating new collapse opportunities.
        while True:
            # Find all nodes with a degree of exactly 2.
            # We operate on a copy of the nodes list as we will be modifying the graph.
            degree2_nodes = [node for node, degree in self.graph.degree() if degree == 2]
            
            if not degree2_nodes:
                break # No more nodes to collapse, exit the loop.
            
            nodes_collapsed_in_pass = 0
            for node in degree2_nodes:
                # Re-check degree in case the graph changed due to a previous collapse in this pass
                if node in self.graph and self.graph.degree(node) == 2:
                    neighbors = list(self.graph.neighbors(node))
                    neighbor1, neighbor2 = neighbors[0], neighbors[1]
                    
                    # Ensure we don't collapse a 3-node cycle (triangle)
                    if not self.graph.has_edge(neighbor1, neighbor2):
                        # Collapse the chain: add a direct edge between the neighbors
                        new_edge_attributes = {}
                        if preserve_attributes:
                            # Aggregate attributes from the two old edges and the node
                            original_edge1_data = self.graph.get_edge_data(neighbor1, node)
                            original_edge2_data = self.graph.get_edge_data(node, neighbor2)
                            node_data = self.graph.nodes[node]
                            
                            new_edge_attributes["collapsed_nodes"] = str([node]) # Record what was collapsed
                            new_edge_attributes.update(original_edge1_data)
                            new_edge_attributes.update(original_edge2_data)
                            new_edge_attributes.update({f"node_{k}":v for k,v in node_data.items()})

                        self.graph.add_edge(neighbor1, neighbor2, **new_edge_attributes)
                        
                        # Remove the original node
                        self.graph.remove_node(node)
                        nodes_collapsed_in_pass += 1
            
            if nodes_collapsed_in_pass == 0:
                # If we went through all degree-2 nodes and couldn't collapse any,
                # it means they are all part of small cycles, so we're done.
                break
            else:
                nodes_removed_count += nodes_collapsed_in_pass
        
        return nodes_removed_count

    def save(self, output_path: Optional[str] = None) -> Path:
        """
        Saves the collapsed graph to a new GraphML file.

        Args:
            output_path (Optional[str]): The path to save the new file. If None,
                                         it will be saved next to the original
                                         with a '_collapsed' suffix.
        Returns:
            Path: The path to the saved file.
        """
        if output_path is None:
            output_path = self.gml_path.parent / f"{self.gml_path.stem}_collapsed.graphml"
        else:
            output_path = Path(output_path)
        
        nx.write_graphml(self.graph, output_path)
        return output_path

    def get_stats(self) -> Dict:
        """Returns statistics about the collapse operation."""
        nodes_removed = self.original_node_count - self.graph.number_of_nodes()
        reduction_pct = (nodes_removed / self.original_node_count * 100) if self.original_node_count > 0 else 0
        return {
            "original_nodes": self.original_node_count,
            "original_edges": self.original_edge_count,
            "collapsed_nodes": self.graph.number_of_nodes(),
            "collapsed_edges": self.graph.number_of_edges(),
            "nodes_removed": nodes_removed,
            "node_reduction_percent": f"{reduction_pct:.2f}%"
        }

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke architecton --optimize_graph --path="/Architectural_Blueprints/dependency_full.graphml"

    print("--- Initiating NeuralBlitz GraphML Collapser ---")

    # Create a dummy GraphML file representing a linear dependency chain
    G_dummy = nx.DiGraph()
    # Chain: A -> B -> C -> D -> E (B, C, D are degree-2)
    # Also add a branching path: C -> F
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('C', 'F')]
    G_dummy.add_edges_from(edges)
    
    # Add some attributes
    G_dummy.nodes['C']['importance'] = 'high'
    G_dummy.edges[('B','C')]['dependency_type'] = 'runtime'
    
    dummy_gml_path = "dependency_test.graphml"
    nx.write_graphml(G_dummy, dummy_gml_path)

    print(f"\nCreated dummy dependency graph '{dummy_gml_path}' with {G_dummy.number_of_nodes()} nodes.")
    print("Graph structure: A -> B -> C -> D -> E, and C -> F")

    try:
        collapser = GraphMLCollapser(dummy_gml_path)
        
        print("\n[Phase 1: Collapsing Chains]")
        nodes_removed = collapser.collapse_chains()
        print(f"Collapse complete. Removed {nodes_removed} linear-chain nodes.")
        
        print("\n[Phase 2: Displaying Statistics]")
        stats = collapser.get_stats()
        print(json.dumps(stats, indent=2))

        print("\n[Phase 3: Saving Collapsed Graph]")
        saved_path = collapser.save()
        print(f"Collapsed graph saved to: {saved_path}")

        # You can inspect the _collapsed.graphml file to see the new direct edge A->E and C remaining.
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the dummy file
        Path(dummy_gml_path).unlink(missing_ok=True)
        if 'saved_path' in locals() and saved_path.exists():
            saved_path.unlink()