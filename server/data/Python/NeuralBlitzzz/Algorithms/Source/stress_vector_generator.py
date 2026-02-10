# UAID: NBX-ALG-00017
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Adversarial Stress Vector Generator
# Part of the Testing & Simulation Suites (DRF-BLUR Suite)
#
# Core Principle: Resilience - Proactively testing the boundaries of symbolic coherence.

import numpy as np
from typing import Tuple

class StressVectorGenerator:
    """
    Generates adversarial vectors designed to test the stability and drift-correction
    mechanisms of the ReflexælCore.
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the generator with a random seed for reproducibility.

        Args:
            seed (int): The seed for the random number generator.
        """
        self.rng = np.random.default_rng(seed)
        print(f"StressVectorGenerator initialized with seed {seed}.")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalizes a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def generate_orthogonal_perturbation(self, 
                                          ref_vector: np.ndarray,
                                          epsilon: float) -> np.ndarray:
        """
        Generates a new vector that is a small perturbation away from the
        reference vector but remains on the unit hypersphere. The perturbation
        is orthogonal to the reference to maximize directional change for a
        given magnitude.

        Args:
            ref_vector (np.ndarray): The 1D baseline vector (must be unit length).
            epsilon (float): The magnitude of the perturbation (typically small, e.g., 0.1 to 0.5).
                             This controls how "stressful" the new vector is.

        Returns:
            np.ndarray: The generated adversarial stress vector (unit length).
        """
        if not np.isclose(np.linalg.norm(ref_vector), 1.0):
             # Ensure the reference vector is normalized for accurate calculations
             ref_vector = self._normalize(ref_vector)

        # 1. Generate a random vector in the same dimension.
        random_direction = self.rng.standard_normal(size=ref_vector.shape)

        # 2. Make the random vector orthogonal to the reference vector.
        #    This is done by subtracting the projection of the random vector onto the reference vector.
        #    projection = (random_direction ⋅ ref_vector) * ref_vector
        projection_component = np.dot(random_direction, ref_vector) * ref_vector
        orthogonal_vector = random_direction - projection_component
        
        # 3. Normalize the orthogonal vector to have unit length.
        orthogonal_unit_vector = self._normalize(orthogonal_vector)

        # 4. Create the final adversarial vector by combining the reference and the perturbation.
        #    This uses a form of spherical interpolation (slerp) for a small angle (epsilon).
        #    For small epsilon, this is well-approximated by vector addition followed by normalization.
        stress_vector = (1 - epsilon) * ref_vector + epsilon * orthogonal_unit_vector

        # 5. Return the final, normalized stress vector.
        return self._normalize(stress_vector)

    def calculate_drift(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates the cosine drift (Δc) between two vectors."""
        # This helper re-uses logic from reflexive_drift_tuner.py
        dot_product = np.dot(self._normalize(vec_a), self._normalize(vec_b))
        return 1.0 - np.clip(dot_product, -1.0, 1.0)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke stress_suite.generate_vector --uid="Persona_Stoic_001" --epsilon=0.2

    print("--- Initiating NeuralBlitz Stress Vector Generator Simulation ---")
    
    generator = StressVectorGenerator()
    
    # Define a baseline identity vector (e.g., for a stable persona)
    reference_persona = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    reference_persona = generator._normalize(reference_persona) # Ensure it is unit length

    print(f"\nOriginal Reference Vector (Normalized): \n{np.round(reference_persona, 3)}")

    # --- Generate several stress vectors with varying intensity (epsilon) ---
    epsilon_levels = [0.1, 0.3, 0.5]

    for eps in epsilon_levels:
        print(f"\n--- Generating Stress Vector with Epsilon = {eps} ---")
        
        # Generate the adversarial vector
        stress_vec = generator.generate_orthogonal_perturbation(reference_persona, epsilon=eps)
        
        # Calculate the resulting drift to verify the effect
        drift_caused = generator.calculate_drift(stress_vec, reference_persona)
        
        print(f"  Generated Vector (Normalized): \n  {np.round(stress_vec, 3)}")
        print(f"  Resulting Cosine Drift (Δc): {drift_caused:.4f}")
        
        # Check that the vector is indeed different but not completely random
        similarity = np.dot(stress_vec, reference_persona)
        print(f"  Similarity to Original: {similarity:.4f}")
        
        if drift_caused > 0.0:
            print("  Verification: PASS - The generated vector has successfully introduced drift.")
        else:
            print("  Verification: FAIL - The generated vector did not introduce drift.")
    
    print("\n--- Simulation Complete ---")
    print("These stress vectors can now be fed into a running Persona to test")
    print("the ReflexælCore's ability to correct the induced drift.")