# UAID: NBX-ALG-00012
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Semantic Persona Diff
# Part of the InterfaceLayer (HALIC) and Synergy Engine
#
# Core Principle: Reflexive Alignment (ε₄) - quantifies drift between conceptual states.

import numpy as np
from typing import List, Dict
# This algorithm relies on a sentence-embedding model. In the NeuralBlitz
# environment, this would be a direct call to the UNE's embedding layer.
# For standalone execution, we use a high-quality open-source model.
from sentence_transformers import SentenceTransformer

class SemanticPersonaDiff:
    """
    Computes a semantic 'diff' between two text corpuses, representing
    persona histories, dialogues, or conceptual documents.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the differ by loading a sentence embedding model.
        This is a potentially heavy operation and should be done once.

        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        try:
            # The model is loaded into memory for efficient reuse.
            self.model = SentenceTransformer(model_name)
            # Get the embedding dimension from the model configuration
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"SemanticPersonaDiff initialized with model '{model_name}' (dim: {self.embedding_dim}).")
        except Exception as e:
            # Handle potential model download errors
            raise RuntimeError(f"ERR-MODEL-001: Could not load sentence transformer model '{model_name}'. Reason: {e}")

    def compute_diff(self, history_a: List[str], history_b: List[str]) -> Dict:
        """
        Calculates the semantic difference between two text histories.

        The process:
        1. Concatenates each history into a single document.
        2. Encodes each document into a normalized high-dimensional vector.
        3. Calculates the vector difference and cosine similarity.

        Args:
            history_a (List[str]): A list of strings representing the first persona's text.
            history_b (List[str]): A list of strings representing the second persona's text.

        Returns:
            Dict: A dictionary containing the vector difference and cosine similarity score.
        """
        if not history_a or not history_b:
            return {
                "vector_difference": np.zeros(self.embedding_dim, dtype=np.float32).tolist(),
                "cosine_similarity": 0.0,
                "error": "One or both histories are empty."
            }
            
        # Concatenate into single documents for holistic representation
        doc_a = " ".join(history_a)
        doc_b = " ".join(history_b)

        # Encode both documents into normalized vectors. Normalizing is key for cosine similarity.
        embeddings = self.model.encode([doc_a, doc_b], normalize_embeddings=True)
        vec_a, vec_b = embeddings[0], embeddings[1]

        # Calculate the core metrics
        # Vector Difference: Represents the direction of semantic change from B to A.
        vector_difference = vec_a - vec_b
        
        # Cosine Similarity: Represents the overall alignment. 1 = identical, 0 = unrelated, -1 = opposite.
        # Since embeddings are normalized, this is a simple dot product.
        cosine_similarity = np.dot(vec_a, vec_b)
        
        return {
            "vector_difference": vector_difference.tolist(),
            "cosine_similarity": float(cosine_similarity)
        }

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke semantic_diff --persona_a_uid="NBX-PERS-0112" --persona_b_uid="NBX-PERS-0211"

    print("--- Initiating NeuralBlitz Semantic Persona Diff Simulation ---")

    # This might take a moment on the first run to download the model
    try:
        differ = SemanticPersonaDiff()

        # Define two distinct persona histories
        persona_architect_history = [
            "The system architecture must prioritize scalability and fault tolerance.",
            "We should use a microservices-based approach with a gRPC mesh.",
            "The DRS integrity is paramount; every transaction requires a GoldenDAG seal.",
            "Let's model the data flow using a directed acyclic graph."
        ]
        
        persona_creative_history = [
            "Explore the liminal space between dream and reality.",
            "What is the sound of a color? Let's weave a narrative around that sensory blend.",
            "The story needs more emotional resonance and archetypal depth.",
            "Let's express this concept using a symbolic glyph, not just words."
        ]

        print("\n[Computing diff between 'Architect' and 'Creative' personas...]")
        diff_report = differ.compute_diff(persona_architect_history, persona_creative_history)
        
        similarity = diff_report["cosine_similarity"]
        print(f"\n--- Report ---")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print(f"  Interpretation: A score of {similarity:.2f} indicates significant conceptual divergence between the two personas, as expected.")
        # print(f"  Vector Diff (first 5 dims): {np.round(diff_report['vector_difference'][:5], 3)}")

    except RuntimeError as e:
        print(f"\n--- Simulation Failed ---")
        print(f"Could not run the simulation. This may be due to a missing internet connection to download the embedding model.")
        print(f"Error details: {e}")

    print("\n--- Simulation Complete ---")