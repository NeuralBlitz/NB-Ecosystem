# UAID: NBX-TST-ALG-00011
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Golden Trace Visualizer (golden_trace_visualizer.py, NBX-ALG-00011)
#
# Core Principle: Radical Transparency (ε₂) - validating our audit visualization tools.

import pytest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the class we are testing
from Algorithms.Source.golden_trace_visualizer import GoldenTraceVisualizer

# --- Test Fixtures ---

@pytest.fixture
def valid_trace_file(tmp_path: Path) -> Path:
    """Creates a valid trace.json file in a temporary directory."""
    trace_content = {
        "trace_id": "CT-TEST-001",
        "description": "A standard test trace with a simple branch.",
        "edges": [
            {"parent": "HASH_GENESIS", "child": "HASH_STEP_1", "type": "genesis", "parent_type": "genesis"},
            {"parent": "HASH_STEP_1", "child": "HASH_STEP_2", "type": "linear", "child_type": "ck_invocation"},
            {"parent": "HASH_STEP_1", "child": "HASH_BRANCH_1", "type": "ethical_branch", "child_type": "persona_state"},
            {"parent": "HASH_STEP_2", "child": "HASH_COLLAPSE", "type": "linear", "child_type": "collapse"},
            {"parent": "HASH_BRANCH_1", "child": "HASH_COLLAPSE", "type": "linear"},
        ]
    }
    trace_path = tmp_path / "valid_trace.json"
    trace_path.write_text(json.dumps(trace_content))
    return trace_path

@pytest.fixture
def empty_trace_file(tmp_path: Path) -> Path:
    """Creates a trace file with no edges."""
    trace_content = {"trace_id": "CT-EMPTY-001", "edges": []}
    trace_path = tmp_path / "empty_trace.json"
    trace_path.write_text(json.dumps(trace_content))
    return trace_path

# --- Test Cases ---

class TestGoldenTraceVisualizer:

    def test_initialization_success(self, valid_trace_file: Path):
        """Tests that the visualizer initializes correctly with a valid trace file."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        assert visualizer.trace_path == valid_trace_file
        assert visualizer.graph is not None
        # Graph should have 5 nodes: GENESIS, STEP_1, STEP_2, BRANCH_1, COLLAPSE
        assert visualizer.graph.number_of_nodes() == 5
        assert visualizer.graph.number_of_edges() == 5

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-007"):
            GoldenTraceVisualizer("non_existent_trace.json")

    def test_initialization_malformed_json(self, tmp_path: Path):
        """Tests for ValueError on corrupted JSON."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{'key': 'not valid json'") # Missing closing brace
        with pytest.raises(ValueError, match="ERR-PARSE-004"):
            GoldenTraceVisualizer(str(malformed_file))
            
    def test_initialization_missing_key(self, tmp_path: Path):
        """Tests for KeyError if 'edges' key is missing."""
        missing_key_file = tmp_path / "missing_key.json"
        missing_key_file.write_text('{"trace_id": "some_id"}')
        with pytest.raises(ValueError, match="ERR-SCHEMA-002"):
            GoldenTraceVisualizer(str(missing_key_file))

    def test_color_mapping(self, valid_trace_file: Path):
        """Unit tests the internal `_get_color_by_type` method."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        assert "rgb(100,149,237)" in visualizer._get_color_by_type("genesis")
        assert "rgb(255,69,0)" in visualizer._get_color_by_type("collapse")
        assert "red" in visualizer._get_color_by_type("ethical_branch")
        assert "black" in visualizer._get_color_by_type("unknown_type")

    def test_render_svg_creates_file(self, valid_trace_file: Path, tmp_path: Path):
        """Tests that `render_svg` successfully creates a non-empty SVG file."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        output_path = tmp_path / "output.svg"
        
        result_path = visualizer.render_svg(str(output_path))
        
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_is_valid_xml(self, valid_trace_file: Path, tmp_path: Path):
        """
        Tests that the output is a well-formed XML/SVG file by attempting to parse it.
        This is a robust way to check for rendering errors.
        """
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        output_path = tmp_path / "output.svg"
        visualizer.render_svg(str(output_path))
        
        try:
            # Attempt to parse the SVG file as XML. If it fails, it will raise an exception.
            ET.parse(str(output_path))
        except ET.ParseError as e:
            pytest.fail(f"The generated SVG file is not well-formed XML: {e}")

    def test_render_svg_with_empty_graph(self, empty_trace_file: Path, tmp_path: Path, capsys):
        """
        Tests that rendering an empty graph does not crash and prints a warning.
        """
        visualizer = GoldenTraceVisualizer(str(empty_trace_file))
        output_path = tmp_path / "empty_output.svg"
        
        visualizer.render_svg(str(output_path))
        
        # The file should not be created for an empty graph
        assert not output_path.exists()
        
        # A warning should be printed to stdout
        captured = capsys.readouterr()
        assert "Graph is empty. Cannot render visualization." in captured.out


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest, networkx, and svgwrite are installed:
    #    pip install pytest networkx svgwrite
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_golden_trace_visualizer.py
    
    print("This is a test file. Use 'pytest' to execute it.")