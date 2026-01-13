# UAID: NBX-ALG-00002
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Reflexive Drift Tuner
# Part of the ReflexælCore Subsystem, managed by MetaMind
#
# Core Principle: Reflexive Alignment (ε₄) - "Know Thy Drift."

import numpy as np
from typing import Dict, Optional, Tuple, Any

class ReflexiveDriftTuner:
    """
    Implements a Proportional-Integral-Derivative (PID)-like controller to
    correct the drift of a symbolic vector towards a reference (baseline) vector.
    This is the core mechanism for maintaining ontological coherence (minimizing Δc)
    in personas, concepts, and the system's core identity.
    """

    def __init__(self, Kp: float = 0.4, Ki: float = 0.05, Kd: float = 0.01, drift_threshold: float = 0.34):
        """
        Initializes the tuner with PID gains and a drift threshold.

        Args:
            Kp (float): Proportional gain. Responds to the current error.
            Ki (float): Integral gain. Corrects long-term, steady-state error.
            Kd (float): Derivative gain. Dampens oscillations and predicts future error.
            drift_threshold (float): The Δc value at which a WARNING or intervention is triggered.
        """
        if not all(k >= 0 for k in [Kp, Ki, Kd]):
            raise ValueError("ERR-INIT-001: PID gains must be non-negative.")
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.drift_threshold = drift_threshold
        # State stored per tuned vector UID to handle multiple simultaneous tunings
        self.states: Dict[str, Dict[str, float]] = {}

    def _calculate_cosine_drift(self, current_vec: np.ndarray, ref_vec: np.ndarray) -> float:
        """Calculates the cosine drift (Δc), a value from 0 (aligned) to 2 (opposed)."""
        # Normalize vectors to ensure the dot product is the cosine of the angle
        norm_current = np.linalg.norm(current_vec)
        norm_ref = np.linalg.norm(ref_vec)
        
        if norm_current == 0 or norm_ref == 0:
            return 1.0 # Max drift if one vector is zero

        current_normed = current_vec / norm_current
        ref_normed = ref_vec / norm_ref
        
        # Clamp dot product to handle potential floating point inaccuracies
        cosine_similarity = np.clip(np.dot(current_normed, ref_normed), -1.0, 1.0)
        
        # Drift is 1 - similarity. Ranges from 0 (perfect alignment) to 2 (perfect opposition).
        return 1.0 - cosine_similarity

    def tune_vector(self, vector_uid: str, current_vector: np.ndarray, reference_vector: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculates a corrective nudge for a vector that has drifted from its reference.

        Args:
            vector_uid (str): A unique identifier for the vector being tuned (e.g., persona UID, concept UID).
            current_vector (np.ndarray): The vector that has potentially drifted.
            reference_vector (np.ndarray): The baseline or ground-truth vector.

        Returns:
            Tuple containing:
            - np.ndarray: The new, corrected vector, nudged back towards the reference.
            - Dict: A diagnostic report including the current drift and control signals.
        """
        if vector_uid not in self.states:
            # Initialize state for a new vector being tracked
            self.states[vector_uid] = {"integral_error": 0.0, "previous_error": 0.0}
        
        state = self.states[vector_uid]
        
        # --- PID Controller Logic ---
        # 1. Calculate the current error (Δc)
        current_drift = self._calculate_cosine_drift(current_vector, reference_vector)

        # 2. Update the integral term (accumulated error over time)
        state["integral_error"] += current_drift
        
        # 3. Calculate the derivative term (rate of change of error)
        derivative_error = current_drift - state["previous_error"]
        
        # 4. Store current error for the next iteration
        state["previous_error"] = current_drift

        # 5. Calculate the total control signal (the "nudge")
        # The signal is a scalar representing the magnitude of the correction.
        # It's negative because we want to reduce the error.
        control_signal = -(self.Kp * current_drift + 
                           self.Ki * state["integral_error"] + 
                           self.Kd * derivative_error)

        # 6. Apply the correction
        # The correction is applied in the direction of the reference vector,
        # effectively "pulling" the current vector back into alignment.
        correction_vector = control_signal * reference_vector
        nudged_vector = current_vector + correction_vector
        
        # 7. Normalize the result to maintain unit length, which is common for embeddings.
        final_vector = nudged_vector / np.linalg.norm(nudged_vector)

        # 8. Generate diagnostic report
        diagnostics = {
            "vector_uid": vector_uid,
            "drift_delta_c": current_drift,
            "is_above_threshold": current_drift > self.drift_threshold,
            "control_signal": control_signal,
            "p_term": self.Kp * current_drift,
            "i_term": self.Ki * state["integral_error"],
            "d_term": self.Kd * derivative_error,
        }
        
        return final_vector, diagnostics

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke ReflexælCore --tune_drift --uid="Persona_Stoic_001"
    
    print("--- Initiating NeuralBlitz Reflexive Drift Tuner Simulation ---")
    
    tuner = ReflexiveDriftTuner()
    
    # Define a baseline identity vector (e.g., for a 'Stoic' persona)
    reference_identity = np.array([1.0, 0.0, 0.0, 0.0]) # A simple 4D vector
    
    # Simulate a vector that has drifted over time
    drifted_vector = np.array([0.8, 0.5, 0.1, 0.2])
    drifted_vector /= np.linalg.norm(drifted_vector) # Normalize

    # --- Run the tuner for several iterations to show convergence ---
    print(f"Reference Vector: {reference_identity}")
    print("-" * 50)

    vector_to_tune = drifted_vector
    uid = "Persona_Stoic_001"
    
    for i in range(5):
        print(f"Iteration {i+1}:")
        initial_drift = tuner._calculate_cosine_drift(vector_to_tune, reference_identity)
        print(f"  - Initial State (Δc): {initial_drift:.6f}")

        # Invoke the tuning algorithm
        corrected_vector, report = tuner.tune_vector(uid, vector_to_tune, reference_identity)
        
        final_drift = tuner._calculate_cosine_drift(corrected_vector, reference_identity)
        print(f"  - Control Signal Applied: {report['control_signal']:.6f}")
        print(f"  - Final State (Δc):   {final_drift:.6f}")
        print("-" * 50)
        
        # Update the vector for the next iteration
        vector_to_tune = corrected_vector

    print("Simulation Complete. Vector has been nudged back towards coherence.")