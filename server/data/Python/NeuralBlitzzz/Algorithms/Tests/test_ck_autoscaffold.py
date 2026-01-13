# UAID: NBX-TST-ALG-00006
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Capability Kernel Auto-Scaffolder (ck_autoscaffold.py, NBX-ALG-00006)
#
# Core Principle: Recursive Self-Betterment - validating the tools that build our system.

import pytest
import json
from pathlib import Path
from typing import List

# Import the functions we are testing
from Algorithms.Source.ck_autoscaffold import ck_autoscaffold, _to_pascal_case

# --- Test Fixtures ---

@pytest.fixture
def base_ck_dir(tmp_path: Path) -> Path:
    """Provides a temporary base directory for generating kernels into."""
    ck_dir = tmp_path / "CapabilityKernels" / "CK_Classes"
    ck_dir.mkdir(parents=True, exist_ok=True)
    return ck_dir

# --- Test Cases ---

class TestCkAutoscaffolder:

    def test_scaffold_creates_all_files_and_dirs(self, base_ck_dir: Path):
        """
        Happy path test: Verifies that a standard scaffold call creates the complete
        and correct directory and file structure.
        """
        name = "My Awesome Kernel"
        description = "This is a test kernel."
        
        ck_autoscaffold(name=name, description=description, base_dir=str(base_ck_dir))

        # Check for the main directory
        kernel_dir = base_ck_dir / "MyAwesomeKernel"
        assert kernel_dir.is_dir()

        # Check for all expected files
        expected_files = [
            "manifest.json",
            "__init__.py",
            "kernel.py",
            "tests/__init__.py",
            "tests/test_my_awesome_kernel.py"
        ]
        for file in expected_files:
            assert (kernel_dir / file).is_file()

    def test_scaffold_aborts_if_directory_exists(self, base_ck_dir: Path, capsys):
        """
        Safety test: Verifies that the scaffolder does not overwrite an existing
        kernel directory.
        """
        name = "Existing Kernel"
        
        # Create it the first time
        ck_autoscaffold(name=name, description="First version", base_dir=str(base_ck_dir))
        
        # Try to create it a second time
        ck_autoscaffold(name=name, description="Second version", base_dir=str(base_ck_dir))

        captured = capsys.readouterr()
        assert "WARNING: Directory" in captured.out
        assert "already exists. Aborting scaffold." in captured.out

    def test_manifest_content_is_correct(self, base_ck_dir: Path):
        """
        Content validation: Checks if the generated manifest.json contains the
        correct information passed during scaffolding.
        """
        name = "Data Validator CK"
        description = "Validates DRS data schemas."
        tags = ["drs", "validation", "custodian"]
        dependencies = ["NBX-SCHEMA-DRS-NODE-V5"]
        
        ck_autoscaffold(
            name=name,
            description=description,
            tags=tags,
            dependencies=dependencies,
            base_dir=str(base_ck_dir)
        )
        
        manifest_path = base_ck_dir / "DataValidatorCk" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())

        assert manifest_data["Name"] == name
        assert manifest_data["Description"] == description
        assert manifest_data["EntryPoint"] == "DataValidatorCk"
        assert manifest_data["Tags"] == tags
        assert manifest_data["Dependencies"] == dependencies
        assert manifest_data["UAID"].startswith("NBX-KRN-DATA-")

    def test_scaffold_handles_optional_parameters_gracefully(self, base_ck_dir: Path):
        """
        Tests that the scaffolder works correctly when optional parameters
        (tags, dependencies) are not provided.
        """
        name = "Simple Kernel"
        description = "A kernel with no special options."
        
        ck_autoscaffold(name=name, description=description, base_dir=str(base_ck_dir))
        
        manifest_path = base_ck_dir / "SimpleKernel" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        
        # Check that the optional keys have default (empty list) values
        assert manifest_data.get("Tags") == []
        assert manifest_data.get("Dependencies") == []

    def test_kernel_and_test_py_content_is_correct(self, base_ck_dir: Path):
        """
        Content validation: Checks if the Python source files are generated with
        the correct class names and import paths.
        """
        name = "symbolic_math_engine"
        class_name = "SymbolicMathEngine"

        ck_autoscaffold(name=name, description="...", base_dir=str(base_ck_dir))
        
        kernel_path = base_ck_dir / class_name / "kernel.py"
        test_path = base_ck_dir / class_name / "tests" / "test_symbolic_math_engine.py"

        kernel_content = kernel_path.read_text()
        test_content = test_path.read_text()

        # Check if the class is defined correctly in the kernel
        assert f"class {class_name}:" in kernel_content
        
        # Check if the test file imports and uses the correct class
        assert f"from ..kernel import {class_name}" in test_content
        assert f"return {class_name}(context=mock_context)" in test_content


class TestPascalCaseHelper:
    """
    Unit tests for the private helper function `_to_pascal_case`.
    """
    @pytest.mark.parametrize("input_str, expected_output", [
        ("my_kernel_name", "MyKernelName"),
        ("My-Awesome-Kernel", "MyAwesomeKernel"),
        ("A Spaced Name", "ASpacedName"),
        ("singleword", "Singleword"),
        ("AlreadyPascal", "Alreadypascal"), # Note: Simple split/capitalize logic
    ])
    def test_pascal_case_conversion(self, input_str, expected_output):
        assert _to_pascal_case(input_str) == expected_output

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_ck_autoscaffold.py
    
    print("This is a test file. Use 'pytest' to execute it.")