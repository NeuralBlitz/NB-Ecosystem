# UAID: NBX-TST-ALG-00014
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Bloom Timeline Renderer (bloom_timeline_renderer.py, NBX-ALG-00014)
#
# Core Principle: Radical Transparency (ε₂) - validating our self-reflection visualization tools.

import pytest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the class we are testing
from Algorithms.Source.bloom_timeline_renderer import BloomTimelineRenderer

# --- Test Fixtures ---

@pytest.fixture
def valid_bloom_log_file(tmp_path: Path) -> Path:
    """Creates a valid but out-of-order .jsonl bloom log file."""
    events = [
        {"timestamp": "2025-07-28T12:00:00Z", "sigma_level": 3.1, "entropy": 8.5, "shard_file": "shard_B.npz"},
        {"timestamp": "2025-07-28T10:00:00Z", "sigma_level": 2.6, "entropy": 7.9, "shard_file": "shard_A.npz"},
        {"timestamp": "2025-07-28T14:00:00Z", "sigma_level": 4.5, "entropy": 9.2, "shard_file": "shard_C_HYPERBLOOM.npz"},
    ]
    log_path = tmp_path / "valid_bloom_log.jsonl"
    with log_path.open('w') as f:
        for event in events:
            f.write(json.dumps(event) + '\\n')
    return log_path

@pytest.fixture
def empty_bloom_log_file(tmp_path: Path) -> Path:
    """Creates an empty .jsonl log file."""
    log_path = tmp_path / "empty_bloom_log.jsonl"
    log_path.touch()
    return log_path

@pytest.fixture
def malformed_bloom_log_file(tmp_path: Path) -> Path:
    """Creates a log file with some invalid lines."""
    log_path = tmp_path / "malformed_bloom_log.jsonl"
    log_path.write_text(
        '{"timestamp": "2025-07-28T11:00:00Z", "sigma_level": 2.8, "entropy": 8.1}\\n'
        'this is not json\\n'
        '{"timestamp": "2025-07-28T13:00:00Z"}\\n' # Missing keys
    )
    return log_path

# --- Test Cases ---

class TestBloomTimelineRenderer:

    def test_initialization_success_and_sorting(self, valid_bloom_log_file: Path):
        """Tests successful initialization and that events are sorted correctly by timestamp."""
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        assert renderer.log_path == valid_bloom_log_file
        
        # Check that the events list was populated and sorted
        assert len(renderer.events) == 3
        assert renderer.events[0]["timestamp"] == "2025-07-28T10:00:00Z"
        assert renderer.events[1]["timestamp"] == "2025-07-28T12:00:00Z"
        assert renderer.events[2]["timestamp"] == "2025-07-28T14:00:00Z"

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-009"):
            BloomTimelineRenderer("non_existent_log.jsonl")

    def test_initialization_with_malformed_file(self, malformed_bloom_log_file: Path):
        """Tests that the loader gracefully skips malformed lines and missing keys."""
        renderer = BloomTimelineRenderer(str(malformed_bloom_log_file))
        # It should have loaded only the one valid, complete event line.
        assert len(renderer.events) == 1
        assert renderer.events[0]["sigma_level"] == 2.8

    def test_render_svg_creates_file(self, valid_bloom_log_file: Path, tmp_path: Path):
        """Tests that `render_svg` successfully creates a non-empty SVG file."""
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline.svg"
        
        result_path = renderer.render_svg(str(output_path))
        
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_is_valid_xml(self, valid_bloom_log_file: Path, tmp_path: Path):
        """
        Tests that the output is a well-formed XML/SVG file by attempting to parse it.
        """
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline.svg"
        renderer.render_svg(str(output_path))
        
        try:
            ET.parse(str(output_path))
        except ET.ParseError as e:
            pytest.fail(f"The generated SVG file is not well-formed XML: {e}")

    def test_render_svg_with_no_events(self, empty_bloom_log_file: Path, tmp_path: Path):
        """
        Tests that rendering an empty log produces a valid SVG with a message.
        """
        renderer = BloomTimelineRenderer(str(empty_bloom_log_file))
        output_path = tmp_path / "empty_timeline.svg"
        
        renderer.render_svg(str(output_path))
        
        assert output_path.exists()
        
        # Check for the "No events" text in the SVG content
        svg_content = output_path.read_text()
        assert "No Bloom Events to display" in svg_content

    def test_svg_content_correctness(self, valid_bloom_log_file: Path, tmp_path: Path):
        """
        Parses the generated SVG to check for the correct number of elements,
        verifying that the data was translated into visual components.
        """
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline_content_test.svg"
        renderer.render_svg(str(output_path))

        # SVG uses namespaces, so we must use them in findall
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        tree = ET.parse(str(output_path))
        root = tree.getroot()

        # Should be one vertical line for each of the 3 events
        event_lines = root.findall(".//svg:line[@class='event-line']", namespaces)
        assert len(event_lines) == 3

        # Should be one text label for each of the 3 events
        event_labels = root.findall(".//svg:text[@class='event-label']", namespaces)
        assert len(event_labels) == 3

        # Check if the highest sigma event has the largest radius circle
        circles = root.findall(".//svg:circle[@class='event-marker']", namespaces)
        assert len(circles) == 3
        radii = [float(c.get('r')) for c in circles]
        # The events are sorted by time, so the 3rd event (14:00) had the highest sigma
        # and should have the largest radius.
        assert max(radii) == float(circles[2].get('r'))


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and svgwrite are installed:
    #    pip install pytest svgwrite
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_bloom_timeline_renderer.py
    
    print("This is a test file. Use 'pytest' to execute it.")