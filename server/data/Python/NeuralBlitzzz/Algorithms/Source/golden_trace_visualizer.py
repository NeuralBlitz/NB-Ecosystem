# UAID: NBX-ALG-00011
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Golden Trace Visualizer
# Part of the Custodian and Self-Reflection_Logs Subsystems
#
# Core Principle: Radical Transparency (ε₂) - making the GoldenDAG chain human-readable.

import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from svgwrite import Drawing, rgb

class GoldenTraceVisualizer:
    """
    Renders a GoldenDAG collapse trace (from a CTPV event) into an SVG
    braid diagram, illustrating the flow of provenance and symbolic collapse.
    """

    def __init__(self, trace_json_path: str):
        """
        Initializes the visualizer with the path to the trace log file.

        Args:
            trace_json_path (str): Path to the JSON file containing the trace data.
                                   Expected format: {"edges": [{"parent": "hash1", "child": "hash2", "type": "..."}]}.
        """
        self.trace_path = Path(trace_json_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"ERR-FS-007: Trace file not found at '{self.trace_path}'")

        try:
            self.trace_data = json.loads(self.trace_path.read_text())
            self.graph = self._build_graph()
        except json.JSONDecodeError:
            raise ValueError(f"ERR-PARSE-004: Malformed JSON in trace file '{self.trace_path}'")
        except KeyError:
            raise ValueError(f"ERR-SCHEMA-002: Trace file '{self.trace_path}' is missing the 'edges' key.")

    def _build_graph(self) -> nx.DiGraph:
        """Constructs a NetworkX directed graph from the trace data."""
        G = nx.DiGraph()
        for edge in self.trace_data.get('edges', []):
            parent = edge.get('parent')
            child = edge.get('child')
            if parent and child:
                # Add nodes and edge with attributes from the trace log
                G.add_node(parent, type=edge.get('parent_type', 'default'))
                G.add_node(child, type=edge.get('child_type', 'default'))
                G.add_edge(parent, child, type=edge.get('type', 'linear'))
        return G

    def _get_color_by_type(self, node_type: str) -> str:
        """Maps node or edge types to specific colors for visualization."""
        # Defined in Codex Universalis, Vol V: SemanticMaps Style Guide
        color_map = {
            "genesis": rgb(100, 149, 237, '%'), # CornflowerBlue for starting points
            "collapse": rgb(255, 69, 0, '%'),    # OrangeRed for collapse events
            "persona_state": rgb(60, 179, 113, '%'),# MediumSeaGreen for personas
            "ck_invocation": rgb(218, 112, 214, '%'),# Orchid for CK calls
            "ethical_branch": "red",
            "linear": "grey",
            "default": "black"
        }
        return color_map.get(node_type, "black")

    def render_svg(self, output_svg_path: str, layout_seed: int = 42, k_distance: float = 0.5):
        """
        Renders the graph to an SVG file using a force-directed layout.

        Args:
            output_svg_path (str): The path to save the generated SVG file.
            layout_seed (int): The random seed for the layout algorithm for reproducibility.
            k_distance (float): Optimal distance between nodes in the spring layout.

        Returns:
            Path: The path to the saved SVG file.
        """
        if not self.graph.nodes:
            print("Warning: Graph is empty. Cannot render visualization.")
            return

        # Use a spring layout for a more 'organic' braid-like feel
        pos = nx.spring_layout(self.graph, seed=layout_seed, k=k_distance, iterations=100)
        
        # Determine the canvas size based on the layout
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())
        
        # Add some padding
        padding = 50
        width = (max_x - min_x) * 400 + 2 * padding
        height = (max_y - min_y) * 400 + 2 * padding

        dwg = Drawing(output_svg_path, size=(f"{width}px", f"{height}px"), profile='tiny')
        
        def scale(x, y):
            """Scales and translates node positions to fit the SVG canvas."""
            new_x = (x - min_x) * 400 + padding
            new_y = (y - min_y) * 400 + padding
            return new_x, new_y

        # Draw edges first (so they are in the background)
        for u, v, data in self.graph.edges(data=True):
            start = scale(*pos[u])
            end = scale(*pos[v])
            edge_color = self._get_color_by_type(data.get('type', 'linear'))
            dwg.add(dwg.line(start, end, stroke=edge_color, stroke_width=0.5))

        # Draw nodes
        for node, data in self.graph.nodes(data=True):
            cx, cy = scale(*pos[node])
            node_color = self._get_color_by_type(data.get('type', 'default'))
            # Add a circle for the node
            dwg.add(dwg.circle(center=(cx, cy), r=4, fill=node_color))
            # Add a text label with the truncated hash
            dwg.add(dwg.text(f"{node[:6]}...", insert=(cx + 6, cy + 4), 
                           font_size='8px', fill=rgb(10,10,10,'%')))
        
        dwg.save()
        print(f"Golden Trace visualization saved to: {output_svg_path}")
        return Path(output_svg_path)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke visualizer --trace_file="/CollapseTraces/CT-SIM-BOS-001.json"
    
    print("--- Initiating NeuralBlitz Golden Trace Visualizer ---")

    # Create a dummy trace file for the simulation
    dummy_trace_content = {
        "trace_id": "CT-SIM-BOS-001",
        "description": "Trace of the OQT-BOS Core Genesis Simulation.",
        "edges": [
            {"parent": "D1F2A3B4", "child": "A7D3F8E1", "type": "genesis", "parent_type": "genesis"},
            {"parent": "A7D3F8E1", "child": "B8C7E6D5", "type": "linear", "child_type": "collapse"},
            {"parent": "B8C7E6D5", "child": "C9D0A1B2", "type": "linear", "child_type": "persona_state"},
            {"parent": "A7D3F8E1", "child": "F0A1B2C3", "type": "ethical_branch", "child_type": "ck_invocation"},
            {"parent": "F0A1B2C3", "child": "C9D0A1B2", "type": "linear"}
        ]
    }
    trace_file = Path("dummy_trace.json")
    trace_file.write_text(json.dumps(dummy_trace_content))

    try:
        visualizer = GoldenTraceVisualizer(str(trace_file))
        output_svg = "golden_trace_visualization.svg"
        visualizer.render_svg(output_svg)

        print(f"\nSuccessfully generated SVG visualization '{output_svg}' from the dummy trace.")
        print("Inspect the SVG file to see the braided graph.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy file
        trace_file.unlink(missing_ok=True)
        # In a real run, you'd keep the output SVG. We'll remove it here.
        Path("golden_trace_visualization.svg").unlink(missing_ok=True)