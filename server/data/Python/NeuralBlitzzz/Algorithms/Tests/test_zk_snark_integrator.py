# UAID: NBX-TST-ALG-00019
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: ZkSnarkIntegrator (zk_snark_integrator.py, NBX-ALG-00019)
#
# Core Principle: Radical Transparency (ε₂) & Privacy - validating our zero-knowledge proof tools.

import pytest
import json
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.zk_snark_integrator import ZkSnarkIntegrator

# --- Mocking Dependencies ---

# In a real system, this would be a complex library like py_ecc, circom, or ZoKrates.
# We mock it to test the integration logic without the heavy cryptographic dependency.
class MockZkSnarkLib:
    """A mock of a zk-SNARK cryptographic library."""
    def setup(self, circuit_id: str):
        """Generates dummy proving and verifying keys."""
        if circuit_id == "VALID_CK_EXECUTION":
            return (
                f"proving_key_for_{circuit_id}",
                f"verifying_key_for_{circuit_id}"
            )
        else:
            raise FileNotFoundError("Circuit not found")

    def prove(self, proving_key: str, public_inputs: Dict, private_inputs: Dict) -> Dict:
        """Generates a dummy proof."""
        # A real proof would be a complex object. We simulate it with a dictionary.
        # We embed the inputs to allow the mock verify function to check them.
        return {
            "proof_data": "encrypted_looking_string_of_data_12345",
            "metadata": {
                "used_proving_key": proving_key,
                "expected_public_inputs": public_inputs,
                "is_tampered": False
            }
        }

    def verify(self, verifying_key: str, proof: Dict, public_inputs: Dict) -> bool:
        """Verifies a dummy proof."""
        if not proof or not isinstance(proof, dict) or "metadata" not in proof:
            return False
        if proof["metadata"]["is_tampered"]:
            return False
        if verifying_key != f"verifying_key_for_VALID_CK_EXECUTION":
            return False
        # Check if the public inputs provided for verification match those the proof was generated with.
        if public_inputs != proof["metadata"]["expected_public_inputs"]:
            return False
        
        return True

# --- Test Fixtures ---

@pytest.fixture
def integrator_instance(monkeypatch) -> ZkSnarkIntegrator:
    """Provides a ZkSnarkIntegrator instance with the library dependency mocked out."""
    # This is the crucial step: we replace the real library with our mock object
    # for the duration of the test.
    monkeypatch.setattr(
        "Algorithms.Source.zk_snark_integrator.zksnark_lib",
        MockZkSnarkLib()
    )
    # Initialize the integrator for a valid circuit
    return ZkSnarkIntegrator(circuit_id="VALID_CK_EXECUTION")

# --- Test Cases ---

class TestZkSnarkIntegrator:

    def test_initialization_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests that the integrator initializes correctly and calls the mock setup."""
        assert integrator_instance.circuit_id == "VALID_CK_EXECUTION"
        assert integrator_instance.proving_key == "proving_key_for_VALID_CK_EXECUTION"
        assert integrator_instance.verifying_key == "verifying_key_for_VALID_CK_EXECUTION"

    def test_initialization_failure_bad_circuit(self, monkeypatch):
        """Tests that initialization fails if the circuit does not exist."""
        monkeypatch.setattr(
            "Algorithms.Source.zk_snark_integrator.zksnark_lib",
            MockZkSnarkLib()
        )
        with pytest.raises(ValueError, match="ERR-ZK-001"):
            ZkSnarkIntegrator(circuit_id="INVALID_CIRCUIT")

    def test_generate_proof_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests the successful generation of a proof object."""
        public_inputs = {"output_hash": "abc"}
        private_inputs = {"input_data": "secret"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        assert proof is not None
        assert "proof_data" in proof
        assert proof["metadata"]["expected_public_inputs"] == public_inputs

    def test_verify_proof_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests successful verification of a valid, untampered proof."""
        public_inputs = {"output_hash": "def"}
        private_inputs = {"input_data": "another_secret"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs)
        
        assert is_valid is True

    def test_verify_proof_failure_tampered_proof(self, integrator_instance: ZkSnarkIntegrator):
        """Tests verification failure when the proof object is marked as tampered."""
        public_inputs = {"output_hash": "ghi"}
        private_inputs = {"input_data": "secret3"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        # Simulate tampering
        proof["metadata"]["is_tampered"] = True
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs)
        
        assert is_valid is False

    def test_verify_proof_failure_mismatched_public_inputs(self, integrator_instance: ZkSnarkIntegrator):
        """
        Tests verification failure when the proof is valid but the public inputs
        provided for verification do not match.
        """
        public_inputs_original = {"output_hash": "jkl"}
        private_inputs = {"input_data": "secret4"}
        
        proof = integrator_instance.generate_proof(public_inputs_original, private_inputs)
        
        # Attempt to verify with different public inputs
        public_inputs_for_verification = {"output_hash": "xyz"} # Mismatch
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs_for_verification)
        
        assert is_valid is False
        
    def test_end_to_end_workflow(self, integrator_instance: ZkSnarkIntegrator):
        """
        A single test that simulates the entire correct workflow:
        1. Define public and private inputs.
        2. Generate a proof.
        3. Verify the proof with the correct public inputs.
        The result must be True.
        """
        # The Prover's data
        public_inputs = {"output_hash": "final_hash_123"}
        private_inputs = {"model_weights_seed": 98765, "user_data_id": "user-54321"}
        
        # The Prover generates the proof
        proof_object = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        # The Verifier (e.g., Custodian) receives the proof and the public statement
        # and performs verification.
        is_verified = integrator_instance.verify_proof(proof_object, public_inputs)
        
        assert is_verified is True


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_zk_snark_integrator.py
    
    print("This is a test file. Use 'pytest' to execute it.")