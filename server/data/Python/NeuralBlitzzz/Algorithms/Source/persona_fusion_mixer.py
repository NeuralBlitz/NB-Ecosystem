# UAID: NBX-ALG-00008
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Persona Fusion Mixer
# Part of the Synergy Engine (SynE v5.1)
#
# Core Principle: Synergy & Emergence - creating novel capabilities from component parts.

import numpy as np
from typing import List, Dict, Tuple

class PersonaFusionMixer:
    """
    Blends the logit outputs of multiple Personas to create a single,
    fused meta-persona output. Uses the Wasserstein barycenter (via the
    Sinkhorn-Knopp algorithm) to find the optimal probabilistic average.
    """

    def __init__(self, epsilon: float = 0.01, max_iters: int = 50):
        """
        Initializes the mixer with regularization and iteration parameters for
        the Sinkhorn algorithm.

        Args:
            epsilon (float): Regularization strength. Higher values lead to a
                             smoother, more entropic barycenter.
            max_iters (int): Maximum number of iterations for the algorithm to converge.
        """
        self.epsilon = epsilon
        self.max_iters = max_iters

    def _get_cost_matrix(self, vocab_size: int) -> np.ndarray:
        """
        Creates a simple cost matrix where the cost of moving probability
        is the squared distance between token indices. For a real-world use case,
        this would be based on semantic distance in an embedding space.
        """
        indices = np.arange(vocab_size, dtype=np.float32)
        # Reshape to (vocab_size, 1) and (1, vocab_size) to broadcast
        return (indices[:, np.newaxis] - indices[np.newaxis, :])**2

    def fuse_logits(self, 
                    logit_vectors: List[np.ndarray], 
                    weights: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Fuses multiple logit vectors into a single barycentric logit vector.

        Args:
            logit_vectors (List[np.ndarray]): A list of 1D NumPy arrays, each representing
                                             the logit output for a given Persona. All
                                             vectors must have the same length (vocab_size).
            weights (List[float]): A list of weights corresponding to each Persona's
                                   influence. Must sum to 1.0.

        Returns:
            Tuple containing:
            - np.ndarray: The resulting fused logit vector.
            - Dict: Diagnostics including convergence status.
        """
        if not logit_vectors:
            raise ValueError("ERR-INPUT-001: logit_vectors cannot be empty.")
        if len(logit_vectors) != len(weights):
            raise ValueError("ERR-INPUT-002: logit_vectors and weights must have the same length.")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("ERR-INPUT-003: weights must sum to 1.0.")

        vocab_size = logit_vectors[0].shape[0]
        num_personas = len(logit_vectors)

        # Convert logits to probability distributions
        probabilities = np.array([np.exp(logits - np.max(logits)) for logits in logit_vectors])
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        # --- Sinkhorn-Knopp Algorithm for Wasserstein Barycenter ---
        
        # In a high-performance setting, this cost matrix would be pre-computed.
        C = self._get_cost_matrix(vocab_size)
        K = np.exp(-C / self.epsilon)

        # Initialization
        b = np.ones(vocab_size, dtype=np.float32)
        v = np.ones((num_personas, vocab_size), dtype=np.float32)

        for i in range(self.max_iters):
            a_old = b.copy()
            u = probabilities / (K @ b)
            b = np.power(np.prod(np.power(K.T @ u, weights[:, np.newaxis]), axis=0), 1.0 / np.sum(weights))
            
            # Check for convergence
            if np.linalg.norm(a_old - b) / np.linalg.norm(b) < 1e-5:
                break
        
        # The barycenter is our fused probability distribution
        fused_probabilities = b * (K @ a_old)

        # Convert back to logits (adding a small constant for stability)
        fused_logits = np.log(fused_probabilities + 1e-20)
        
        diagnostics = {
            "status": "converged" if i < self.max_iters - 1 else "max_iterations_reached",
            "iterations": i + 1,
            "epsilon": self.epsilon
        }

        return fused_logits, diagnostics


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke synergy_engine --spawn personas=["Stoic", "CreativeMuse"] --mode=fusion

    print("--- Initiating NeuralBlitz Persona Fusion Mixer Simulation ---")

    vocab_size = 10  # A small vocabulary for demonstration
    mixer = PersonaFusionMixer(epsilon=0.1)

    # Persona 1: "Stoic Philosopher" - strongly prefers tokens related to logic and reason (e.g., index 1, 2)
    logits_stoic = np.array([-1.0, 5.0, 4.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    
    # Persona 2: "Creative Muse" - strongly prefers tokens related to art and imagination (e.g., index 7, 8)
    logits_creative = np.array([-3.0, -3.0, -3.0, -2.0, -2.0, 0.0, 1.0, 5.0, 4.0, -1.0], dtype=np.float32)
    
    # Fusion weights - e.g., 70% Stoic, 30% Creative
    fusion_weights = [0.7, 0.3]

    print(f"\nFusing {len(fusion_weights)} personas with weights {fusion_weights}")

    fused_logits, report = mixer.fuse_logits([logits_stoic, logits_creative], fusion_weights)

    # --- Analyze the results ---
    fused_probs = np.exp(fused_logits - np.max(fused_logits))
    fused_probs /= fused_probs.sum()

    print(f"\nFusion Diagnostics: {report}")
    print("\n--- Original & Fused Probabilities ---")
    stoic_probs = np.exp(logits_stoic-np.max(logits_stoic)); stoic_probs/=stoic_probs.sum()
    creative_probs = np.exp(logits_creative-np.max(logits_creative)); creative_probs/=creative_probs.sum()
    
    print(f"Stoic Prob Dist:     {np.round(stoic_probs, 2)}")
    print(f"Creative Prob Dist:  {np.round(creative_probs, 2)}")
    print(f"Fused Prob Dist:     {np.round(fused_probs, 2)}")

    print(f"\nMost likely token from Stoic:       {np.argmax(logits_stoic)}")
    print(f"Most likely token from Creative:  {np.argmax(logits_creative)}")
    print(f"Most likely token from Fused Persona: {np.argmax(fused_logits)}")

    print("\n--- Simulation Complete ---")
    print("The fused persona maintains influence from both parents, creating a novel, blended output.")