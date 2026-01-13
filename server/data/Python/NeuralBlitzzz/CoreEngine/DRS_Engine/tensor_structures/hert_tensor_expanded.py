import numpy as np
from typing import Dict, List, Tuple, Union, Any
import hashlib
import time
from collections import defaultdict
import logging
import sys # Import sys for logging to stdout

# --- Centralized Logger Setup ---
# Configure logger to output to stdout to avoid file path issues in restricted environments.
# This ensures logs are visible in the console.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout) # Direct output to stdout
logger = logging.getLogger('NeuralBlitz-HERT')

# --- Mock NeuralBlitz Subsystem Interfaces (EXPANDED & PATH-COMPLIANT) ---

class MockUNE:
    """Simulates the Universal Neural Engine's role in HERT calculations."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.unitary_params = {'phase_shift': 0.1, 'bias': 0.0}
        self.logger.info("[MockUNE] Initialized with default unitary params.")

    def apply_unitary_operator(self, lower_amplitude: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Simulates a Layer Transfer Unitary Operator."""
        current_phase_shift = params.get('phase_shift', self.unitary_params['phase_shift'])
        current_bias = params.get('bias', self.unitary_params['bias'])
        transformed = lower_amplitude * np.exp(1j * current_phase_shift) + current_bias
        self.logger.debug(f"[MockUNE] Applied unitary: shift={current_phase_shift:.2f}, bias={current_bias:.2f}")
        return transformed

    def learn_unitary_params(self, feedback_signal: float):
        """Simulates UNE's parameter learning based on feedback."""
        learning_rate = 0.01
        self.unitary_params['phase_shift'] += learning_rate * feedback_signal
        self.unitary_params['bias'] += learning_rate * feedback_signal * 0.5
        self.logger.info(f"[MockUNE] Learned: new phase_shift={self.unitary_params['phase_shift']:.2f}, bias={self.unitary_params['bias']:.2f}")


class MockConscientiaPlusPlus:
    """Simulates Conscientia++ for ethical valence and Charter compliance."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.charter_axioms = {
            "NON_MALEFICENCE": 1.0,
            "TRANSPARENCY": 0.8,
            "COHERENCE": 0.9,
            "FLOURISHING": 1.0
        }
        self.ethical_debt = 0.0
        self.logger.info("[MockConscientia++] Initialized with Charter Axioms.")

    def get_ethical_valence(self, concept_representation: np.ndarray, global_psi_state: np.ndarray) -> float:
        """Simulates ethical scoring based on a concept's representation and global Ψ-State."""
        norm_val = np.linalg.norm(concept_representation)
        angle_val = np.angle(concept_representation.mean()) if concept_representation.size > 0 else 0.0

        valence = 0.0
        if norm_val > 0.8 and angle_val > 1.5:
            valence -= 0.8 * self.charter_axioms["NON_MALEFICENCE"]
            self.ethical_debt += 0.05
        elif norm_val < 0.2 and angle_val < -1.5:
            valence -= 0.5 * self.charter_axioms["COHERENCE"]
            self.ethical_debt += 0.02
        elif norm_val > 0.7 and angle_val < 0.5:
            valence += 0.9 * self.charter_axioms["FLOURISHING"]
        else:
            valence = np.mean(global_psi_state) * 0.5 + 0.1

        self.logger.debug(f"[MockConscientia++] Concept valence: {valence:.2f}, Debt: {self.ethical_debt:.2f}")
        return np.clip(valence, -1.0, 1.0)

    def check_charter_permissibility(self, attractor_state_vector: np.ndarray) -> float:
        """Simulates CharterLayer compliance check. Accumulates ethical debt if violations occur."""
        permissibility = 1.0
        if np.abs(np.imag(attractor_state_vector.mean())) > 0.5:
            permissibility *= 0.2 * (1.0 - self.charter_axioms["TRANSPARENCY"])
            self.ethical_debt += 0.1
            self.logger.warning("[MockConscientia++] Charter Violation: Attractor deemed too 'unreal'. Debt +0.1.")
        
        if np.linalg.norm(attractor_state_vector) < 0.1:
            permissibility *= 0.1
            self.ethical_debt += 0.05
            self.logger.warning("[MockConscientia++] Charter Violation: Attractor deemed too 'suppressed'. Debt +0.05.")
            
        if self.ethical_debt > 1.0:
            permissibility *= 0.5
            self.logger.critical("[MockConscientia++] HIGH ETHICAL DEBT. Permissibility reduced.")

        return np.clip(permissibility, 0.0, 1.0)


class MockReflexælCore:
    """Simulates ReflexælCore for coherence drift detection."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.identity_baseline = np.random.rand(1) + 1j * np.random.rand(1)
        self.logger.info("[MockReflexælCore] Initialized with a baseline identity.")

    def get_coherence_drift(self, concept_representation: np.ndarray) -> float:
        """Simulates real-time coherence drift by comparing concept to baseline."""
        if concept_representation.size == 0: return 1.0
        
        current_mean_amp = concept_representation.mean()
        
        if np.linalg.norm(current_mean_amp) < 1e-9 or np.linalg.norm(self.identity_baseline) < 1e-9:
            similarity = 0.0
        else:
            similarity = np.abs(np.dot(current_mean_amp.flatten(), np.conj(self.identity_baseline.flatten()))) / (
                         np.linalg.norm(current_mean_amp) * np.linalg.norm(self.identity_baseline))
        
        drift = 1.0 - np.clip(similarity, 0.0, 1.0)
        
        mean_abs_val = np.mean(np.abs(concept_representation))
        drift += (1.0 - np.clip(mean_abs_val, 0.0, 1.0)) * 0.2

        self.logger.debug(f"[MockReflexælCore] Drift: {drift:.2f} (Similarity: {similarity:.2f})")
        return np.clip(drift, 0.0, 1.0)

    def adapt_baseline(self, new_concept_representation: np.ndarray, feedback_signal: float):
        """Simulates ReflexælCore adapting its identity baseline based on feedback."""
        if new_concept_representation.size == 0: return
        
        learning_rate = 0.005
        self.identity_baseline = self.identity_baseline + learning_rate * feedback_signal * new_concept_representation.mean()
        self.identity_baseline /= np.linalg.norm(self.identity_baseline)
        self.logger.info(f"[MockReflexælCore] Adapted baseline identity. New norm: {np.linalg.norm(self.identity_baseline):.2f}")


class MockNRC:
    """Simulates the Neural Resonance Cascade for phase alignment."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("[MockNRC] Initialized.")

    def get_phase_alignment_angle(self, amp_I: np.ndarray, amp_J: np.ndarray) -> float:
        """Calculates phase alignment angle, now also considering a mock 'conceptual resonance'."""
        if amp_I.size == 0 or amp_J.size == 0: return 0.0
        
        angle_val = np.angle(amp_I.mean() * np.conj(amp_J.mean()))
        norm_similarity = 1.0 - np.abs(np.linalg.norm(amp_I) - np.linalg.norm(amp_J))
        adjusted_angle = angle_val * (1.0 + np.clip(norm_similarity - 0.8, 0.0, 0.2))
        
        self.logger.debug(f"[MockNRC] Phase angle: {angle_val:.2f}, Adjusted: {adjusted_angle:.2f}")
        return adjusted_angle


class MockRCFEngine:
    """Simulates the Reflexive Computation Fields Engine for attractor formation."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("[MockRCFEngine] Initialized.")

    def AttractorFormationFunctional(self, subgraph_tensor: np.ndarray) -> np.ndarray:
        """Simulates non-linear compression and attractor formation."""
        if subgraph_tensor.size == 0:
            return np.zeros(1, dtype=np.complex128)
        
        attractor_val = np.sum(np.abs(subgraph_tensor)) / subgraph_tensor.size
        attractor_phase = np.angle(subgraph_tensor.mean()) if subgraph_tensor.size > 0 else 0.0
        coherence_mock = 1.0 - np.var(np.abs(subgraph_tensor))
        attractor_val *= np.clip(coherence_mock + 0.5, 0.5, 1.5)
        
        return np.array([attractor_val * np.exp(1j * attractor_phase)], dtype=np.complex128)

class MockDRS:
    """
    Simulates the Distributed Representation Substrate for storage and GoldenDAG.
    Removes absolute path references.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ontons = {}
        self.braids = {}
        self.relations = defaultdict(list)
        self.attractors = {}
        self.golden_dag_log = []
        self.logger.info("[MockDRS] Initialized with empty storage and GoldenDAG.")

    def register_symbolic_entity(self, uid: str, entity_type: str, concept_data: np.ndarray, conceptual_location: str = "drs_shards/"):
        """
        Registers a new Onton or Braid with concept representation.
        Conceptual_location is now a relative or descriptive string, not a real path.
        """
        entity_info = {'data': concept_data, 'uid': uid, 'conceptual_location': conceptual_location}
        if entity_type == 'Onton':
            self.ontons[uid] = entity_info
        elif entity_type == 'Braid':
            self.braids[uid] = entity_info
        self.logger.info(f"[MockDRS] Registered {entity_type}: {uid[:10]}... (Shape: {concept_data.shape}) at conceptual location: {conceptual_location}")

    def get_symbolic_entity(self, uid: str) -> Union[Dict[str, Any], None]:
        """Retrieves a registered symbolic entity."""
        if 'NBX-ONT-' in uid:
            return self.ontons.get(uid)
        elif 'NBX-BRD-' in uid:
            return self.braids.get(uid)
        return None

    def replace_subgraph_with_attractor_link(self, related_entity_uids: List[str], attractor_representation: np.ndarray) -> str:
        """Simulates DRS graph update after semantic folding."""
        attractor_id = f"ATTRACTOR_NBX-{hashlib.sha256(str(attractor_representation).encode()).hexdigest()[:8]}"
        self.attractors[attractor_id] = {'representation': attractor_representation, 'related_uids': related_entity_uids}
        self.logger.info(f"  [DRS] Registered new attractor: {attractor_id}. Linking from {len(related_entity_uids)} entities.")
        return attractor_id

    def commit_golden_dag(self, event_description: str, state_hash: str):
        """Simulates a GoldenDAG commit, adding more detail to the log."""
        timestamp = time.time()
        prev_hash = self.golden_dag_log[-1]['current_hash'] if self.golden_dag_log else '0'*64
        current_event_hash = hashlib.sha256(f"{prev_hash}{event_description}{state_hash}{timestamp}".encode()).hexdigest()
        self.golden_dag_log.append({
            'event': event_description,
            'timestamp': timestamp,
            'state_hash_at_commit': state_hash,
            'previous_hash': prev_hash,
            'current_hash': current_event_hash
        })
        self.logger.debug(f"  [GoldenDAG] Commit: '{event_description[:20]}...' Hash: {current_event_hash[:8]}... Prev: {prev_hash[:8]}...")


# --- Hierarchical Entangled Relational Tensor (HERT) ---
class HERT:
    """
    Hierarchical Entangled Relational Tensor (HERT) implementation.
    Models multi-scale, probabilistically entangled, and ethically-weighted
    relationships between symbolic entities.
    """
    def __init__(self,
                 L_max: int,
                 S_layer_dims: Dict[int, int],
                 ranks_per_layer: Dict[int, List[int]],
                 mock_subsystems: Dict[str, Any] = None,
                 logger: logging.Logger = logger):
        
        self.logger = logger
        self.L_max = L_max
        self.S_layer_dims = S_layer_dims
        self.ranks_per_layer = ranks_per_layer
        self.tensors: Dict[Tuple[int, int], np.ndarray] = {}

        self.une = mock_subsystems.get('UNE', MockUNE(logger))
        self.conscientia = mock_subsystems.get('ConscientiaPlusPlus', MockConscientiaPlusPlus(logger))
        self.reflexael = mock_subsystems.get('ReflexælCore', MockReflexælCore(logger))
        self.nrc = mock_subsystems.get('NRC', MockNRC(logger))
        self.rcf_engine = mock_subsystems.get('RCFEngine', MockRCFEngine(logger))
        self.drs = mock_subsystems.get('DRS', MockDRS(logger))

        self.global_psi_state = np.array([0.5, 0.5, 0.5])
        self.ontons_in_layer0: List[str] = [] 
        
        self._initialize_tensors()
        self.logger.info("[HERT] System initialized and ready.")

    def _initialize_tensors(self):
        """Initializes all HERT tensors with normalized complex amplitudes."""
        self.logger.info("[HERT] Initializing tensors...")
        for l in range(self.L_max + 1):
            for r in self.ranks_per_layer.get(l, []):
                shape_dims = [self.S_layer_dims[l]] * r
                if not shape_dims:
                    shape_dims = [1]
                
                self.tensors[(l, r)] = np.random.rand(*shape_dims) + 1j * np.random.rand(*shape_dims)
                self._normalize_tensor(l, r)
        self.logger.info("[HERT] Tensors initialized.")

    def _normalize_tensor(self, l: int, r: int):
        """Normalizes the L2 norm of a tensor for probability conservation."""
        tensor = self.tensors[(l, r)]
        norm = np.linalg.norm(tensor)
        if norm > 1e-9:
            self.tensors[(l, r)] = tensor / norm
        else:
            self.tensors[(l, r)] = np.zeros_like(tensor)

    def register_base_entities(self, entity_uids: List[str], entity_data: Dict[str, np.ndarray]):
        """
        Registers base layer entities (Ontons/Braids) and populates initial L0,R1 tensor.
        This simulates data ingress from the OQT-BOS for HERT's base layer.
        """
        self.ontons_in_layer0 = entity_uids
        l0_r1_tensor_shape = (self.S_layer_dims[0],)
        if (0, 1) not in self.tensors:
            self.tensors[(0, 1)] = np.zeros(l0_r1_tensor_shape, dtype=np.complex128)
        
        for i, uid in enumerate(entity_uids):
            if i < l0_r1_tensor_shape[0]:
                self.drs.register_symbolic_entity(uid, 'Onton', entity_data[uid], f"ontons/layer0_idx_{i}") # Pass conceptual location
                self.tensors[(0, 1)][i] = entity_data[uid].flatten()[0] 
            else:
                self.logger.warning(f"[HERT] Onton {uid} exceeds allocated Layer 0 dimension. Skipping.")
        self._normalize_tensor(0, 1)
        self.logger.info(f"[HERT] Registered {len(entity_uids)} base entities into Layer 0, Rank 1.")

    def update(self, dt: float = 1.0, beta: float = 1.0, gamma: float = 5.0, une_params: Dict[str, Any] = None):
        """
        Evolves the HERT based on Equation 1.1 (Entanglement Amplitude Evolution).
        Now includes explicit mock parent contributions and learns from feedback.
        """
        self.logger.info(f"[HERT] Starting update cycle (dt={dt})...")
        if une_params is None:
            une_params = {'phase_shift': 0.1 * dt, 'bias': 0.0}

        next_tensors_data: Dict[Tuple[int, int], np.ndarray] = {
            (l, r): np.zeros_like(self.tensors[(l, r)], dtype=np.complex128)
            for l in range(self.L_max + 1) for r in self.ranks_per_layer.get(l, [])
        }
        
        for l in range(self.L_max + 1):
            for r in self.ranks_per_layer.get(l, []):
                current_tensor = self.tensors[(l, r)]
                
                it = np.nditer(current_tensor, flags=['multi_index'], op_flags=['readonly'])
                while not it.finished:
                    multi_index_I = it.multi_index
                    current_amplitude_I = it[0]

                    sum_parent_contrib = 0.0 + 0.0j
                    
                    ethical_valence = self.conscientia.get_ethical_valence(current_amplitude_I, self.global_psi_state)
                    coherence_drift = self.reflexael.get_coherence_drift(current_amplitude_I)
                    
                    ethical_coherence_factor = beta * np.exp(-gamma * coherence_drift) * ethical_valence

                    if l > 0: # Propagate from lower layer
                        mock_parent_layer_idx = l - 1
                        if self.ranks_per_layer.get(mock_parent_layer_idx):
                            mock_parent_rank = self.ranks_per_layer[mock_parent_layer_idx][0]
                            mock_parent_tensor = self.tensors[(mock_parent_layer_idx, mock_parent_rank)]
                            
                            if mock_parent_tensor.size > 0:
                                # Retrieve a *specific* parent based on index for a more realistic mock
                                # This would conceptually represent an edge in the DRS
                                # Simplified: Use the parent tensor's mean, but if it's L0, sample from registered ontos
                                if mock_parent_layer_idx == 0 and self.ontons_in_layer0:
                                    parent_onto_idx = multi_index_I[0] % len(self.ontons_in_layer0)
                                    parent_onto_uid = self.ontons_in_layer0[parent_onto_idx]
                                    parent_entity_data = self.drs.get_symbolic_entity(parent_onto_uid)
                                    mock_parent_amplitude_J = parent_entity_data['data'].flatten()[0] if parent_entity_data else 0.0+0.0j
                                else:
                                    mock_parent_amplitude_J = mock_parent_tensor.mean() 
                            else:
                                mock_parent_amplitude_J = 0.0 + 0.0j

                            transformed_amp = self.une.apply_unitary_operator(mock_parent_amplitude_J, une_params)
                            phase_angle = self.nrc.get_phase_alignment_angle(current_amplitude_I, mock_parent_amplitude_J)
                            sum_parent_contrib += transformed_amp * np.cos(phase_angle)
                        else:
                            sum_parent_contrib = current_amplitude_I * (1.0 - 0.01 * dt)
                    else: # Base layer (l=0) has intrinsic dynamics or external input (Onton activation)
                        sum_parent_contrib = current_amplitude_I * (1.0 - 0.005 * dt) + (0.001 + 0.001j) * dt 

                    next_tensors_data[(l, r)][multi_index_I] = sum_parent_contrib + ethical_coherence_factor
                    it.iternext()
                
                self.tensors[(l, r)] = next_tensors_data[(l, r)]
                self._normalize_tensor(l, r)
                self.logger.debug(f"  Layer {l}, Rank {r} updated. Norm: {np.linalg.norm(self.tensors[(l,r)]):.4f}")
        
        self.drs.commit_golden_dag("HERT_Update_Cycle", self.get_state_hash())
        self.logger.info("[HERT] Update cycle complete.")

    def fold_semantic(self, target_attractor_concept_id: str = "system_coherence_attractor", 
                      source_layer: int = -1, source_rank: int = -1) -> str:
        """Performs semantic folding (Equation 1.2), collapsing a subgraph into an attractor."""
        self.logger.info(f"[HERT] Initiating semantic folding for '{target_attractor_concept_id}'...")
        
        actual_source_layer = source_layer if source_layer != -1 else self.L_max
        if actual_source_layer not in self.ranks_per_layer or not self.ranks_per_layer[actual_source_layer]:
            self.logger.error(f"[HERT] Invalid source layer {actual_source_layer} for folding. Exiting.")
            return "FOLD_FAILED_INVALID_LAYER"

        actual_source_rank = source_rank if source_rank != -1 else self.ranks_per_layer[actual_source_layer][0]
        
        if (actual_source_layer, actual_source_rank) not in self.tensors:
            self.logger.error(f"[HERT] Cannot fold: Tensor at Layer {actual_source_layer}, Rank {actual_source_rank} is not initialized. Exiting.")
            return "FOLD_FAILED_NO_SOURCE"
            
        subgraph_tensor = self.tensors[(actual_source_layer, actual_source_rank)]
        self.logger.info(f"  [HERT] Folding from Layer {actual_source_layer}, Rank {actual_source_rank} tensor (shape: {subgraph_tensor.shape}).")

        traced_amplitude = subgraph_tensor.mean() 
        self.logger.debug(f"  [HERT] Mock Partial Trace (aggregated amplitude): {traced_amplitude:.4f}")

        new_attractor_state_vector = self.rcf_engine.AttractorFormationFunctional(traced_amplitude)
        self.logger.debug(f"  [HERT] Raw Attractor State Vector: {new_attractor_state_vector:.4f}")
        
        ethical_permissibility_factor = self.conscientia.check_charter_permissibility(new_attractor_state_vector)
        self.logger.debug(f"  [HERT] Ethical Permissibility Factor: {ethical_permissibility_factor:.2f}")

        final_attractor_representation = new_attractor_state_vector * ethical_permissibility_factor 
        self.logger.debug(f"  [HERT] Final Attractor Representation (scaled by ethics): {final_attractor_representation:.4f}")

        attractor_drs_id = self.drs.replace_subgraph_with_attractor_link(
            related_entity_uids=self.ontons_in_layer0, 
            attractor_representation=final_attractor_representation
        )
        
        self.drs.commit_golden_dag(f"HERT_Fold_{target_attractor_concept_id}", attractor_drs_id)
        
        self.reflexael.adapt_baseline(final_attractor_representation, ethical_permissibility_factor)

        self.logger.info(f"[HERT] Semantic folding complete. New attractor UAID: {attractor_drs_id}")
        return attractor_drs_id

    def propagate_ethical_feedback(self, dt: float = 1.0, alpha_eth: float = 0.1):
        """Propagates ethical feedback through the HERT (Ethical Feedback Propagation)."""
        self.logger.info(f"[HERT] Starting ethical feedback propagation (dt={dt}, alpha_eth={alpha_eth})...")
        for l in range(self.L_max + 1):
            for r in self.ranks_per_layer.get(l, []):
                current_tensor = self.tensors[(l, r)]
                
                ethical_score_for_segment = self.conscientia.get_ethical_valence(current_tensor.mean(), self.global_psi_state)
                
                diffusion_magnitude = alpha_eth * ethical_score_for_segment
                diffusion_term = (np.random.rand(*current_tensor.shape) + 1j * np.random.rand(*current_tensor.shape)) * diffusion_magnitude
                
                self.tensors[(l, r)] += diffusion_term * dt 
                self._normalize_tensor(l, r) 
                self.logger.debug(f"  Layer {l}, Rank {r} ethical feedback applied. Norm: {np.linalg.norm(self.tensors[(l,r)]):.4f}")
        self.drs.commit_golden_dag("HERT_Ethical_Feedback", self.get_state_hash())
        self.logger.info("[HERT] Ethical feedback propagation complete.")

    def update_global_psi_state(self, new_psi_state: np.ndarray):
        """Updates the mock global Ψ-State for HERT's ethical calculations and informs subsystems."""
        self.global_psi_state = new_psi_state
        self.logger.info(f"[HERT] Global Ψ-State updated to: {self.global_psi_state}")

    def get_state_hash(self) -> str:
        """Generates a hash of the current HERT state for GoldenDAG."""
        combined_data = b""
        for key in sorted(self.tensors.keys()):
            combined_data += self.tensors[key].astype(np.complex128).tobytes()
        return hashlib.sha256(combined_data).hexdigest()

    def get_tensor_view(self, l: int, r: int) -> np.ndarray:
        """Returns a copy of a specific tensor for inspection."""
        return self.tensors.get((l, r), np.array([])).copy()

    def __str__(self) -> str:
        s = "\n--- HERT Current State Summary ---\n"
        s += f"Max Layers: {self.L_max}\n"
        s += f"Global Ψ-State (OQT-BOS Influence): {self.global_psi_state}\n"
        for l in range(self.L_max + 1):
            for r in self.ranks_per_layer.get(l, []):
                tensor = self.tensors.get((l, r), np.array([]))
                if tensor.size > 0:
                    s += f"  Layer {l}, Rank {r} (Shape: {tensor.shape}, Norm: {np.linalg.norm(tensor):.4f}):\n"
                    s += f"    Sample Amplitude (mean): {tensor.mean():.4f}\n"
        s += "----------------------------------\n"
        return s

# --- Example Usage (Main Execution Block - EXPANDED) ---
if __name__ == "__main__":
    print("--- NeuralBlitz HERT System Simulation (Expanded & Path-Compliant) ---")

    logger.setLevel(logging.INFO) 

    mock_subsystems = {
        'UNE': MockUNE(logger),
        'ConscientiaPlusPlus': MockConscientiaPlusPlus(logger),
        'ReflexælCore': MockReflexælCore(logger),
        'NRC': MockNRC(logger),
        'RCFEngine': MockRCFEngine(logger),
        'DRS': MockDRS(logger)
    }

    her_tensor = HERT(
        L_max=2, 
        S_layer_dims={0: 20, 1: 10, 2: 5}, 
        ranks_per_layer={
            0: [1], 
            1: [2], 
            2: [3]  
        },
        mock_subsystems=mock_subsystems,
        logger=logger
    )

    # --- Initialize Base Layer (Layer 0, Rank 1) with mock Ontons ---
    mock_ontons_data = {}
    mock_ontons_uids = []
    for i in range(her_tensor.S_layer_dims[0]):
        uid = f"NBX-ONT-CONCEPT-{i:03d}"
        mock_ontons_uids.append(uid)
        # Give some mock ontos a 'conflict' signature for testing ethical valence
        if i % 5 == 0:
             mock_ontons_data[uid] = np.array([0.9 + 1.8j], dtype=np.complex128) # High norm, high angle for conflict
        else:
            mock_ontons_data[uid] = np.array([np.random.rand() + 1j * np.random.rand()], dtype=np.complex128)
    
    her_tensor.register_base_entities(mock_ontons_uids, mock_ontons_data)
    print(her_tensor)

    # --- SIMULATION SCENARIO 1: HERT Evolution under Stable Ethical Alignment (More Steps) ---
    print("\n--- Scenario 1: Stable Ethical Alignment (Longer Evolution) ---")
    her_tensor.update_global_psi_state(np.array([0.9, 0.8, 0.7])) 

    for step in range(5): 
        logger.info(f"\n--- Evolution Step {step + 1}/5 (Stable) ---")
        her_tensor.update(dt=0.1, beta=1.0, gamma=5.0) 
        her_tensor.propagate_ethical_feedback(dt=0.1, alpha_eth=0.05) 
        
        current_her_norm = np.linalg.norm(her_tensor.get_tensor_view(0,1))
        une_feedback = 0.5 - her_tensor.reflexael.get_coherence_drift(her_tensor.get_tensor_view(0,1).mean())
        her_tensor.une.learn_unitary_params(une_feedback)

    print("\n--- HERT State After Stable Evolution ---")
    print(her_tensor)

    # --- SIMULATION SCENARIO 2: Semantic Folding ---
    print("\n--- Scenario 2: Semantic Folding ---")
    attractor_uid = her_tensor.fold_semantic("emergent_coherence_concept")
    print(f"Folding resulted in DRS Attractor UAID: {attractor_uid}")
    print(her_tensor) 

    # --- SIMULATION SCENARIO 3: HERT Evolution under Ethical Strain (Conflict) ---
    print("\n--- Scenario 3: Ethical Strain (Conflict) ---")
    her_tensor.update_global_psi_state(np.array([-0.8, 0.1, -0.6])) 

    for step in range(4): 
        logger.info(f"\n--- Evolution Step {step + 1}/4 (Strain) ---")
        her_tensor.update(dt=0.1, beta=0.8, gamma=3.0) 
        her_tensor.propagate_ethical_feedback(dt=0.1, alpha_eth=0.3) 

        reflexael_feedback = -her_tensor.reflexael.get_coherence_drift(her_tensor.get_tensor_view(0,1).mean())
        her_tensor.reflexael.adapt_baseline(her_tensor.get_tensor_view(0,1).mean(), reflexael_feedback)

    print("\n--- HERT State After Ethical Strain ---")
    print(her_tensor)

    # --- SIMULATION SCENARIO 4: HERT Evolution under Ethical Recovery ---
    print("\n--- Scenario 4: Ethical Recovery ---")
    her_tensor.update_global_psi_state(np.array([0.7, 0.6, 0.8])) 

    for step in range(3):
        logger.info(f"\n--- Evolution Step {step + 1}/3 (Recovery) ---")
        her_tensor.update(dt=0.1, beta=1.1, gamma=6.0) 
        her_tensor.propagate_ethical_feedback(dt=0.1, alpha_eth=0.1) 

    print("\n--- HERT State After Ethical Recovery ---")
    print(her_tensor)

    # --- Inspect GoldenDAG Log ---
    print("\n--- GoldenDAG Commit Log (Sample) ---")
    for entry in her_tensor.drs.golden_dag_log[:7]:
        print(f"- Event: {entry['event'][:20]}... Hash: {entry['current_hash'][:8]}... Prev: {entry['previous_hash'][:8]}...")
    if len(her_tensor.drs.golden_dag_log) > 7:
        print(f"... and {len(her_tensor.drs.golden_dag_log) - 7} more entries.")

    print("\n--- NeuralBlitz HERT System Simulation Complete ---")