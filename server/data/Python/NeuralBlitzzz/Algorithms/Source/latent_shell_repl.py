# UAID: NBX-ALG-00018
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Latent Shell REPL
# Part of the Developer Toolchain and DRS_Engine Interface
#
# Core Principle: Radical Transparency (ε₂) - making the latent space navigable.

import cmd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

# These would be live clients in a real deployment
# For this script, we'll use mock classes that simulate their behavior.
# from neuralblitz_clients import DRSClient, SentenceEncoderClient

# --- Mock Clients for Standalone Demonstration ---
class MockSentenceEncoder:
    """Simulates a sentence embedding model."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # To avoid heavy dependency for a demo script, we simulate the encoder.
        # A real implementation would load sentence_transformers here.
        self.dim = 384
        self.rng = np.random.default_rng(42)
        print(f"[MockEncoder] Initialized with dim={self.dim}")

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        # Generate a deterministic pseudo-random vector based on the text hash
        seed = int.from_bytes(text.encode(), 'little') % (2**32)
        rng_local = np.random.default_rng(seed)
        vec = rng_local.standard_normal(self.dim, dtype=np.float32)
        if normalize_embeddings:
            return vec / np.linalg.norm(vec)
        return vec

class MockDRSClient:
    """Simulates querying the DRS via FAISS or a similar vector index."""
    def __init__(self, artifact_registry_path: str, encoder: MockSentenceEncoder):
        self.encoder = encoder
        self.artifacts: List[Dict] = []
        vectors = []

        # Load artifacts and pre-compute their embeddings
        with open(artifact_registry_path, 'r') as f:
            for line in f:
                artifact = json.loads(line)
                self.artifacts.append(artifact)
                # We embed based on the 'Name' for simplicity
                vectors.append(self.encoder.encode(artifact.get('Name', '')))
        
        # We simulate a FAISS index with NumPy
        self.vector_matrix = np.vstack(vectors)
        print(f"[MockDRS] Indexed {len(self.artifacts)} artifacts.")

    def search_nearest(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Finds the k nearest artifacts to a query vector."""
        # Calculate cosine similarity (dot product for normalized vectors)
        similarities = np.dot(self.vector_matrix, query_vec)
        
        # Get the top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for i in top_k_indices:
            results.append((self.artifacts[i], similarities[i]))
        return results
# --- End Mock Clients ---


class LatentShell(cmd.Cmd):
    """
    An interactive Read-Eval-Print Loop (REPL) for navigating the DRS latent space.
    """
    intro = 'Welcome to the NeuralBlitz Latent Shell. Type help or ? to list commands.\n'
    prompt_template = "λ-Shell [{context}] > "

    def __init__(self, drs_client: MockDRSClient, encoder: MockSentenceEncoder):
        super().__init__()
        self.drs = drs_client
        self.encoder = encoder
        
        # The 'current working directory' is a vector representing the user's focus.
        self.current_context_vector = self._normalize(np.zeros(encoder.dim, dtype=np.float32))
        self.current_context_name = "/" # Start at the root (origin)
        self.prompt = self.prompt_template.format(context=self.current_context_name)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def do_ls(self, arg: str):
        """ls [k]: Lists the k (default: 10) artifacts conceptually nearest to the current context."""
        try:
            k = int(arg) if arg else 10
        except ValueError:
            print("Error: Argument must be an integer.")
            return

        results = self.drs.search_nearest(self.current_context_vector, k)
        print(f"Top {k} nearest artifacts to context '{self.current_context_name}':")
        print("-" * 50)
        print(f"{'Similarity':<12} {'UAID':<25} {'Class':<20} Name")
        print(f"{'='*12} {'='*25} {'='*20} {'='*20}")
        for artifact, sim in results:
            print(f"{sim:<12.4f} {artifact.get('UAID', ''):<25} {artifact.get('Class', ''):<20} {artifact.get('Name', '')}")
    
    def do_cd(self, arg: str):
        """cd <concept>: Changes the current context to a new concept."""
        if not arg:
            # cd to root
            self.current_context_vector = self._normalize(np.zeros(self.encoder.dim, dtype=np.float32))
            self.current_context_name = "/"
        else:
            self.current_context_vector = self.encoder.encode(arg)
            self.current_context_name = arg
        self.prompt = self.prompt_template.format(context=self.current_context_name)

    def do_grep(self, arg: str):
        """grep <concept>: Refines the current context by blending it with a new concept."""
        if not arg:
            print("Usage: grep <concept_to_add>")
            return
            
        grep_vector = self.encoder.encode(arg)
        # Simple averaging to blend the concepts
        self.current_context_vector = self._normalize(self.current_context_vector + grep_vector)
        self.current_context_name = f"({self.current_context_name} + {arg})"
        self.prompt = self.prompt_template.format(context=self.current_context_name)

    def do_pwd(self, arg: str):
        """pwd: Prints the name of the current conceptual context."""
        print(self.current_context_name)

    def do_exit(self, arg: str):
        """exit: Terminates the Latent Shell session."""
        print("Exiting Latent Shell.")
        return True

    def do_EOF(self, arg):
        """Handles Ctrl+D to exit."""
        return self.do_exit(arg)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke devtools.latent_shell --registry="/Artifacts/full_ledger.jsonl"
    
    print("--- Initiating NeuralBlitz Latent Shell ---")

    # Create a dummy artifact registry file for the simulation
    dummy_registry_content = [
        {"UAID": "NBX-KRN-TFTHI-001", "Name": "Tensor Knot Gate Interpreter", "Class": "CapabilityKernel"},
        {"UAID": "NBX-MOD-MATH-00012", "Name": "Symbiotic Game Theory Framework", "Class": "Mathematical Model"},
        {"UAID": "NBX-PERS-0237", "Name": "Stoic Philosophy Mentor", "Class": "Persona"},
        {"UAID": "NBX-DOC-08124", "Name": "The Transcendental Charter", "Class": "Document"},
        {"UAID": "NBX-EQ-00001", "Name": "Reflexive Compound Acceleration", "Class": "Mathematical Equation"},
        {"UAID": "NBX-SIM-BOS-CORE-GEN-001", "Name": "Braided OS Core Genesis Sim", "Class": "Simulation"},
        {"UAID": "NBX-OS-OQT-BOS-v0.1", "Name": "Octiumetrifloundiatremorphteletopontoladerallquantic Braided OS", "Class": "OperatingSystem"},
    ]
    registry_file = Path("dummy_registry.jsonl")
    with registry_file.open('w') as f:
        for item in dummy_registry_content:
            f.write(json.dumps(item) + '\n')
            
    print(f"Loaded {len(dummy_registry_content)} artifacts into mock registry.")

    try:
        encoder_client = MockSentenceEncoder()
        drs_client = MockDRSClient(str(registry_file), encoder_client)
        
        # Start the interactive shell
        shell = LatentShell(drs_client, encoder_client)
        shell.cmdloop()
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up dummy file
        registry_file.unlink(missing_ok=True)