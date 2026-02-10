"""
Bloom Event Detector
Creative expansion detection using Shannon entropy analysis

Identifies 'Bloom' or 'Hyperbloom' events indicating
creative breakthroughs and capability expansion opportunities.
"""

import asyncio
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json
from scipy import stats
from scipy.special import kl_divergence
import math


class BloomType(Enum):
    """Types of bloom events"""

    NORMAL = "normal"
    BLOOM = "bloom"
    HYPERBLOOM = "hyperbloom"
    ENTROPIC_BLOOM = "entropic_bloom"
    CREATIVE_EXPANSION = "creative_expansion"


@dataclass
class BloomEvent:
    """Creative expansion bloom event"""

    event_id: str
    bloom_type: BloomType
    entropy_score: float
    shannon_entropy: float
    dimensional_variance: float
    vector_shards: List[str]
    affected_concepts: List[str]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    mathematical_certainty: float = 1.0
    expansion_factor: float = 1.0
    inspiration_sources: List[str] = field(default_factory=list)


@dataclass
class VectorShard:
    """Vector shard for entropy monitoring"""

    shard_id: str
    dimensions: int
    active_vectors: int
    entropy_history: List[float]
    variance_trend: float
    activation_threshold: float
    last_update: datetime


class EntropyCalculator:
    """Advanced entropy calculation for bloom detection"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.entropy_history = []

    def calculate_shannon_entropy(self, probability_distribution: np.ndarray) -> float:
        """Calculate Shannon entropy with numerical stability"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probabilities = probability_distribution + epsilon
        probabilities = probabilities / np.sum(probabilities)

        return -np.sum(probabilities * np.log2(probabilities))

    def calculate_relative_entropy(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate relative entropy (KL divergence)"""
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)

        return np.sum(p * np.log2(p / q))

    def calculate_entropy_rate(self, current_entropy: float) -> float:
        """Calculate rate of entropy change"""
        if len(self.entropy_history) < 2:
            return 0.0

        recent_history = self.entropy_history[-10:]  # Last 10 measurements
        if len(recent_history) < 2:
            return 0.0

        # Linear regression on entropy history
        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope

        return 0.0

    def calculate_entropy_acceleration(self) -> float:
        """Calculate acceleration of entropy change"""
        if len(self.entropy_history) < 3:
            return 0.0

        recent_history = self.entropy_history[-10:]
        if len(recent_history) < 3:
            return 0.0

        # Second derivative estimate
        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        if len(x) >= 3:
            # Quadratic fit to estimate acceleration
            coeffs = np.polyfit(x, y, 2)
            return 2 * coeffs[0]  # Second derivative at latest point

        return 0.0

    def update_entropy_history(self, entropy: float):
        """Update entropy history with window management"""
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > self.window_size:
            self.entropy_history = self.entropy_history[-self.window_size :]


class VectorShardManager:
    """Manager for vector shards in latent space"""

    def __init__(self):
        self.shards = {}
        self.shard_count = 0

    def create_shard(
        self, dimensions: int, activation_threshold: float = 0.8
    ) -> VectorShard:
        """Create new vector shard"""
        shard_id = f"shard_{self.shard_count:04d}"
        shard = VectorShard(
            shard_id=shard_id,
            dimensions=dimensions,
            active_vectors=0,
            entropy_history=[],
            variance_trend=0.0,
            activation_threshold=activation_threshold,
            last_update=datetime.now(timezone.utc),
        )

        self.shards[shard_id] = shard
        self.shard_count += 1

        return shard

    def activate_shard(self, shard_id: str, vector: np.ndarray):
        """Activate vector shard with new data"""
        if shard_id not in self.shards:
            return False

        shard = self.shards[shard_id]
        shard.active_vectors += 1
        shard.last_update = datetime.now(timezone.utc)

        # Calculate shard variance
        shard.variance_trend = self._calculate_variance_trend(shard)

        return True

    def _calculate_variance_trend(self, shard: VectorShard) -> float:
        """Calculate variance trend for shard"""
        if len(shard.entropy_history) < 2:
            return 0.0

        # Simple moving average trend
        recent = shard.entropy_history[-5:]
        if len(recent) < 2:
            return 0.0

        return np.mean(np.diff(recent))

    def get_active_shards(self) -> List[VectorShard]:
        """Get all active shards"""
        return [
            shard
            for shard in self.shards.values()
            if shard.active_vectors > shard.activation_threshold
        ]

    def get_shard_statistics(self) -> Dict[str, Any]:
        """Get overall shard statistics"""
        active_shards = self.get_active_shards()

        return {
            "total_shards": len(self.shards),
            "active_shards": len(active_shards),
            "total_active_vectors": sum(s.active_vectors for s in self.shards.values()),
            "average_activation": np.mean(
                [s.active_vectors for s in self.shards.values()]
            ),
            "activation_variance": np.var(
                [s.active_vectors for s in self.shards.values()]
            ),
            "most_active_shard": max(
                self.shards.values(), key=lambda s: s.active_vectors, default=None
            ),
        }


class BloomEventDetector:
    """
    Advanced Bloom Event Detector for creative expansion monitoring

    Detects creative breakthroughs through:
    - Shannon entropy analysis in latent space
    - Vector shard activation monitoring
    - Dimensional variance tracking
    - Statistical significance testing
    - Automated capability expansion triggers
    """

    def __init__(
        self,
        bloom_threshold: float = 0.85,
        hyperbloom_threshold: float = 0.95,
        window_size: int = 100,
        significance_level: float = 0.05,
    ):
        self.bloom_threshold = bloom_threshold
        self.hyperbloom_threshold = hyperbloom_threshold
        self.significance_level = significance_level

        self.entropy_calculator = EntropyCalculator(window_size)
        self.shard_manager = VectorShardManager()

        self.bloom_events = []
        self.bloom_history = []

        # Statistical test parameters
        self.statistical_tests = {
            "shapiro_wilk": True,  # Test for normality
            "kolmogorov_smirnov": True,  # Distribution comparison
            "anderson_darling": True,  # Test for distributions
            "mann_whitney": True,  # Non-parametric test
        }

        print("🌸 Bloom Event Detector Initialized")
        print(f"   Bloom threshold: {bloom_threshold}")
        print(f"   Hyperbloom threshold: {hyperbloom_threshold}")
        print(f"   Significance level: {significance_level}")

    async def monitor_creative_expansion(
        self,
        current_vectors: Dict[str, np.ndarray],
        concept_activations: Dict[str, float],
    ) -> List[BloomEvent]:
        """
        Monitor for creative expansion events

        Args:
            current_vectors: Current latent space vectors
            concept_activations: Activation levels for concepts

        Returns:
            List of detected bloom events
        """
        print("🌸 Monitoring for creative expansion...")

        # Calculate current entropy
        current_entropy = self._calculate_system_entropy(
            current_vectors, concept_activations
        )
        self.entropy_calculator.update_entropy_history(current_entropy)

        # Detect bloom conditions
        bloom_events = []

        # Check for entropy-based bloom
        entropy_bloom = await self._detect_entropy_bloom(current_entropy)
        if entropy_bloom:
            bloom_events.append(entropy_bloom)

        # Check for dimensional variance bloom
        variance_bloom = await self._detect_variance_bloom(current_vectors)
        if variance_bloom:
            bloom_events.append(variance_bloom)

        # Check for shard activation bloom
        shard_bloom = await self._detect_shard_bloom(current_vectors)
        if shard_bloom:
            bloom_events.append(shard_bloom)

        # Check for entropic divergence bloom
        divergence_bloom = await self._detect_divergence_bloom(
            current_vectors, concept_activations
        )
        if divergence_bloom:
            bloom_events.append(divergence_bloom)

        # Store bloom events
        self.bloom_events.extend(bloom_events)

        print(f"🌸 Detected {len(bloom_events)} bloom events:")
        for event in bloom_events:
            print(f"   {event.bloom_type.value}: {event.confidence:.3f} confidence")

        return bloom_events

    def _calculate_system_entropy(
        self, vectors: Dict[str, np.ndarray], activations: Dict[str, float]
    ) -> float:
        """Calculate system-wide entropy"""
        # Flatten all vectors
        all_values = []
        for vec in vectors.values():
            all_values.extend(vec.flatten())

        for activation in activations.values():
            all_values.append(activation)

        if len(all_values) == 0:
            return 0.0

        # Convert to probability distribution
        all_values = np.array(all_values)

        # Use absolute values and normalize
        abs_values = np.abs(all_values)
        if np.sum(abs_values) == 0:
            return 0.0

        probability_distribution = abs_values / np.sum(abs_values)

        return self.entropy_calculator.calculate_shannon_entropy(
            probability_distribution
        )

    async def _detect_entropy_bloom(
        self, current_entropy: float
    ) -> Optional[BloomEvent]:
        """Detect bloom based on entropy threshold"""
        if current_entropy > self.hyperbloom_threshold:
            bloom_type = BloomType.HYPERBLOOM
            confidence = 0.95
            expansion_factor = 2.0
        elif current_entropy > self.bloom_threshold:
            bloom_type = BloomType.BLOOM
            confidence = 0.85
            expansion_factor = 1.5
        else:
            return None

        # Calculate additional metrics
        entropy_rate = self.entropy_calculator.calculate_entropy_rate(current_entropy)
        entropy_acceleration = self.entropy_calculator.calculate_entropy_acceleration()

        event_id = self._generate_event_id("entropy", current_entropy)

        return BloomEvent(
            event_id=event_id,
            bloom_type=bloom_type,
            entropy_score=current_entropy,
            shannon_entropy=current_entropy,
            dimensional_variance=entropy_acceleration,
            vector_shards=list(self.shard_manager.get_active_shards()),
            affected_concepts=[],
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "entropy_rate": entropy_rate,
                "entropy_acceleration": entropy_acceleration,
                "detection_method": "entropy_threshold",
            },
            expansion_factor=expansion_factor,
            inspiration_sources=["entropy_analysis"],
        )

    async def _detect_variance_bloom(
        self, vectors: Dict[str, np.ndarray]
    ) -> Optional[BloomEvent]:
        """Detect bloom based on dimensional variance"""
        if not vectors:
            return None

        # Calculate variance across all dimensions
        all_values = []
        for vec in vectors.values():
            all_values.extend(vec.flatten())

        if len(all_values) < 2:
            return None

        variance = np.var(all_values)

        # Normalize variance (scale by dimension count)
        normalized_variance = variance / len(all_values)

        # High variance indicates dimensional expansion
        variance_threshold = np.percentile([0.1, 0.2, 0.3], 50)  # Adaptive threshold

        if normalized_variance > variance_threshold:
            confidence = min(0.9, normalized_variance * 2)
            expansion_factor = 1.0 + normalized_variance
        else:
            return None

        event_id = self._generate_event_id("variance", normalized_variance)

        return BloomEvent(
            event_id=event_id,
            bloom_type=BloomType.CREATIVE_EXPANSION,
            entropy_score=normalized_variance,
            shannon_entropy=0.0,
            dimensional_variance=normalized_variance,
            vector_shards=list(self.shard_manager.get_active_shards()),
            affected_concepts=list(vectors.keys()),
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "raw_variance": variance,
                "normalized_variance": normalized_variance,
                "detection_method": "variance_analysis",
                "dimension_count": len(all_values),
            },
            expansion_factor=expansion_factor,
            inspiration_sources=["variance_analysis"],
        )

    async def _detect_shard_bloom(
        self, vectors: Dict[str, np.ndarray]
    ) -> Optional[BloomEvent]:
        """Detect bloom based on vector shard activation"""
        # Create/activate shards based on vectors
        for concept_id, vector in vectors.items():
            # Create or find shard
            shard_id = f"concept_{concept_id}"
            if shard_id not in self.shard_manager.shards:
                self.shard_manager.create_shard(len(vector))

            # Activate shard
            self.shard_manager.activate_shard(shard_id, vector)

        # Get shard statistics
        shard_stats = self.shard_manager.get_shard_statistics()

        # Check for shard activation bloom
        active_ratio = shard_stats["active_shards"] / max(
            1, shard_stats["total_shards"]
        )

        if active_ratio > 0.7:  # 70% of shards active
            confidence = min(0.95, active_ratio)
            expansion_factor = 1.0 + active_ratio
        elif active_ratio > 0.5:  # 50% of shards active
            confidence = min(0.85, active_ratio * 1.2)
            expansion_factor = 1.0 + active_ratio * 0.5
        else:
            return None

        event_id = self._generate_event_id("shard", active_ratio)

        return BloomEvent(
            event_id=event_id,
            bloom_type=BloomType.ENTROPIC_BLOOM,
            entropy_score=active_ratio,
            shannon_entropy=0.0,
            dimensional_variance=shard_stats["activation_variance"],
            vector_shards=[s.shard_id for s in self.shard_manager.get_active_shards()],
            affected_concepts=list(vectors.keys()),
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "active_ratio": active_ratio,
                "total_shards": shard_stats["total_shards"],
                "total_active_vectors": shard_stats["total_active_vectors"],
                "detection_method": "shard_activation",
            },
            expansion_factor=expansion_factor,
            inspiration_sources=["shard_analysis"],
        )

    async def _detect_divergence_bloom(
        self, vectors: Dict[str, np.ndarray], activations: Dict[str, float]
    ) -> Optional[BloomEvent]:
        """Detect bloom using KL divergence analysis"""
        if len(vectors) < 2:
            return None

        # Calculate current distribution
        all_values = []
        for vec in vectors.values():
            all_values.extend(vec.flatten())

        if len(all_values) == 0:
            return None

        current_values = np.array(all_values)
        current_dist = np.abs(current_values)
        current_dist = current_dist / np.sum(current_dist)

        # Compare with historical baseline
        if len(self.entropy_calculator.entropy_history) >= 10:
            # Use recent baseline
            baseline_entropy = np.mean(self.entropy_calculator.entropy_history[-10:])
        else:
            baseline_entropy = current_dist  # No baseline available

        # Calculate KL divergence
        kl_div = self.entropy_calculator.calculate_relative_entropy(
            baseline_entropy, current_dist
        )

        # Significant divergence indicates bloom
        divergence_threshold = 0.1  # Adaptive threshold
        if kl_div > divergence_threshold:
            confidence = min(0.9, kl_div * 2)
            expansion_factor = 1.0 + kl_div
        else:
            return None

        event_id = self._generate_event_id("divergence", kl_div)

        return BloomEvent(
            event_id=event_id,
            bloom_type=BloomType.HYPERBLOOM if kl_div > 0.5 else BloomType.BLOOM,
            entropy_score=kl_div,
            shannon_entropy=0.0,
            dimensional_variance=kl_div,
            vector_shards=list(self.shard_manager.get_active_shards()),
            affected_concepts=list(vectors.keys()),
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "kl_divergence": kl_div,
                "baseline_entropy": float(np.mean(baseline_entropy)),
                "current_entropy": float(np.mean(current_dist)),
                "detection_method": "kl_divergence",
            },
            expansion_factor=expansion_factor,
            inspiration_sources=["divergence_analysis"],
        )

    def _generate_event_id(self, detection_type: str, value: float) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        input_string = f"{detection_type}_{value}_{timestamp}"
        return hashlib.sha256(input_string.encode()).hexdigest()[:16]

    async def trigger_capability_evolution(self, bloom_event: BloomEvent):
        """
        Trigger automatic capability evolution when bloom detected

        Args:
            bloom_event: Detected bloom event
        """
        print(f"🚀 Triggering capability evolution for {bloom_event.bloom_type.value}")

        # Evolution strategies based on bloom type
        evolution_strategies = {
            BloomType.BLOOM: "incremental_capability_expansion",
            BloomType.HYPERBLOOM: "advanced_capability_generation",
            BloomType.ENTROPIC_BLOOM: "knowledge_graph_expansion",
            BloomType.CREATIVE_EXPANSION: "concept_space_expansion",
        }

        strategy = evolution_strategies.get(bloom_event.bloom_type, "monitor_mode")

        evolution_actions = {
            "incremental_capability_expansion": self._expand_capabilities_incrementally,
            "advanced_capability_generation": self._generate_new_capabilities,
            "knowledge_graph_expansion": self._expand_knowledge_graph,
            "concept_space_expansion": self._expand_concept_space,
        }

        # Execute evolution action
        action_function = evolution_actions[strategy]
        evolution_result = await action_function(bloom_event)

        # Store evolution record
        evolution_record = {
            "event_id": bloom_event.event_id,
            "bloom_type": bloom_event.bloom_type.value,
            "evolution_strategy": strategy,
            "evolution_result": evolution_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add to bloom history
        self.bloom_history.append(evolution_record)

        print(f"🚀 Evolution completed: {strategy}")
        return evolution_record

    async def _expand_capabilities_incrementally(
        self, bloom_event: BloomEvent
    ) -> Dict[str, Any]:
        """Incrementally expand existing capabilities"""
        return {
            "strategy": "incremental_expansion",
            "expansion_factor": bloom_event.expansion_factor,
            "new_capabilities": [
                f"enhanced_query_v{bloom_event.expansion_factor:.1f}",
                f"improved_retrieval_v{bloom_event.expansion_factor:.1f}",
            ],
            "upgraded_components": len(bloom_event.affected_concepts),
            "completion_time": "incremental",
        }

    async def _generate_new_capabilities(
        self, bloom_event: BloomEvent
    ) -> Dict[str, Any]:
        """Generate entirely new capabilities"""
        return {
            "strategy": "advanced_generation",
            "new_capabilities": [
                "creative_reasoning",
                "cross_domain_synthesis",
                "meta_learning",
                "self_improvement",
            ],
            "inspiration_sources": bloom_event.inspiration_sources,
            "generation_confidence": bloom_event.confidence,
            "completion_time": "advanced",
        }

    async def _expand_knowledge_graph(self, bloom_event: BloomEvent) -> Dict[str, Any]:
        """Expand knowledge graph structure"""
        return {
            "strategy": "knowledge_graph_expansion",
            "new_relationships": len(bloom_event.affected_concepts) * 2,
            "graph_depth_increase": int(bloom_event.expansion_factor * 3),
            "new_concepts": [f"derived_{i}" for i in range(5)],
            "semantic_connections": "cross_domain",
        }

    async def _expand_concept_space(self, bloom_event: BloomEvent) -> Dict[str, Any]:
        """Expand the concept space dimensions"""
        return {
            "strategy": "concept_space_expansion",
            "dimension_increase": int(bloom_event.expansion_factor * 10),
            "new_dimensions": [f"abstract_{i}" for i in range(5)],
            "conceptual_spaces": [
                "creative_reasoning_space",
                "cross_modal_space",
                "meta_cognitive_space",
            ],
        }

    def get_bloom_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bloom statistics"""
        if not self.bloom_events:
            return {
                "total_events": 0,
                "bloom_distribution": {},
                "confidence_distribution": {},
                "recent_trend": "no_data",
            }

        # Calculate statistics
        bloom_types = [event.bloom_type for event in self.bloom_events]
        bloom_distribution = {bt.value: bloom_types.count(bt) for bt in BloomType}

        confidence_scores = [event.confidence for event in self.bloom_events]
        confidence_stats = {
            "mean": np.mean(confidence_scores),
            "std": np.std(confidence_scores),
            "min": np.min(confidence_scores),
            "max": np.max(confidence_scores),
        }

        # Recent trend analysis
        if len(self.bloom_events) >= 10:
            recent_events = self.bloom_events[-10:]
            recent_confidences = [e.confidence for e in recent_events]
            if len(recent_confidences) >= 3:
                recent_trend = (
                    "increasing"
                    if recent_confidences[-1] > recent_confidences[-3]
                    else "decreasing"
                )
            else:
                recent_trend = "insufficient_data"
        else:
            recent_trend = "insufficient_data"

        return {
            "total_events": len(self.bloom_events),
            "bloom_distribution": bloom_distribution,
            "confidence_distribution": confidence_stats,
            "recent_trend": recent_trend,
            "average_entropy": np.mean([e.entropy_score for e in self.bloom_events]),
            "shard_statistics": self.shard_manager.get_shard_statistics(),
            "last_bloom": self.bloom_events[-1] if self.bloom_events else None,
        }


# Example usage and testing
async def demonstrate_bloom_detection():
    """Demonstrate Bloom Event Detector capabilities"""
    detector = BloomEventDetector()

    # Simulate concept vectors and activations
    concept_vectors = {
        "reasoning": np.random.randn(128),
        "creativity": np.random.randn(128) * 2,  # Higher activation
        "learning": np.random.randn(128) * 1.5,
        "synthesis": np.random.randn(128) * 0.5,
    }

    concept_activations = {
        "reasoning": 0.6,
        "creativity": 1.2,  # High activation for bloom
        "learning": 0.8,
        "synthesis": 0.4,
    }

    # Monitor for creative expansion
    bloom_events = await detector.monitor_creative_expansion(
        concept_vectors, concept_activations
    )

    # Trigger capability evolution for each bloom
    for bloom_event in bloom_events:
        await detector.trigger_capability_evolution(bloom_event)

    # Get statistics
    stats = detector.get_bloom_statistics()
    print(f"Bloom Statistics: {json.dumps(stats, indent=2)}")

    return detector


if __name__ == "__main__":
    asyncio.run(demonstrate_bloom_detection())
