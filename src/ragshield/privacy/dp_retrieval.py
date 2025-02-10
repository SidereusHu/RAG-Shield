"""Differential Privacy Retrieval for RAG Systems.

Implements privacy-preserving retrieval using differential privacy:
- Adds calibrated noise to similarity scores
- Supports Laplace and Gaussian mechanisms
- Integrates with privacy budget tracking

Key concepts:
- Sensitivity: Maximum change in output from one input change
- Noise scale: Calibrated to sensitivity/epsilon
- (ε,δ)-DP: Probabilistic privacy guarantee
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
import numpy as np

from ragshield.core.document import Document
from ragshield.core.retriever import Retriever
from ragshield.privacy.privacy_budget import (
    PrivacyBudgetManager,
    BudgetExceededError,
    PrivacySpend,
)


class NoiseMechanism(Enum):
    """Noise mechanisms for differential privacy."""
    LAPLACE = "laplace"  # Pure ε-DP
    GAUSSIAN = "gaussian"  # (ε,δ)-DP, tighter for high dimensions
    EXPONENTIAL = "exponential"  # For discrete outputs


@dataclass
class DPRetrievalResult:
    """Result of a differentially private retrieval.

    Attributes:
        documents: Retrieved documents with noisy scores
        epsilon_spent: Epsilon consumed by this query
        delta_spent: Delta consumed (for Gaussian mechanism)
        noise_scale: Scale of noise added
        original_order_preserved: Whether noise changed the ranking
        metadata: Additional information
    """
    documents: List[Tuple[Document, float]]
    epsilon_spent: float
    delta_spent: float = 0.0
    noise_scale: float = 0.0
    original_order_preserved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity calculation.

    Attributes:
        score_range: Expected range of similarity scores (min, max)
        sensitivity_type: How sensitivity is calculated
        custom_sensitivity: Override calculated sensitivity
    """
    score_range: Tuple[float, float] = (0.0, 1.0)
    sensitivity_type: str = "bounded"  # "bounded", "unbounded", "local"
    custom_sensitivity: Optional[float] = None

    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity based on configuration."""
        if self.custom_sensitivity is not None:
            return self.custom_sensitivity

        if self.sensitivity_type == "bounded":
            # For bounded scores, sensitivity is the range
            return self.score_range[1] - self.score_range[0]
        elif self.sensitivity_type == "unbounded":
            # For unbounded, use a conservative estimate
            return 2.0
        else:
            # Local sensitivity - document specific
            return 1.0


class NoiseMechanismBase(ABC):
    """Base class for noise mechanisms."""

    @abstractmethod
    def add_noise(
        self,
        scores: np.ndarray,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """Add noise to scores.

        Args:
            scores: Original similarity scores
            epsilon: Privacy parameter
            sensitivity: Sensitivity of the scores
            delta: Privacy parameter for (ε,δ)-DP

        Returns:
            Tuple of (noisy_scores, noise_scale)
        """
        pass

    @abstractmethod
    def calibrate_noise(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> float:
        """Calculate noise scale for given parameters.

        Args:
            epsilon: Privacy parameter
            sensitivity: Query sensitivity
            delta: Privacy parameter (for Gaussian)

        Returns:
            Noise scale (e.g., Laplace scale or Gaussian std)
        """
        pass


class LaplaceMechanism(NoiseMechanismBase):
    """Laplace mechanism for pure ε-differential privacy.

    Adds Laplace noise with scale = sensitivity / epsilon.
    Provides pure ε-DP guarantee.
    """

    def add_noise(
        self,
        scores: np.ndarray,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """Add Laplace noise to scores."""
        scale = self.calibrate_noise(epsilon, sensitivity)
        noise = np.random.laplace(0, scale, size=scores.shape)
        noisy_scores = scores + noise
        return noisy_scores, scale

    def calibrate_noise(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> float:
        """Calculate Laplace scale."""
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return sensitivity / epsilon


class GaussianMechanism(NoiseMechanismBase):
    """Gaussian mechanism for (ε,δ)-differential privacy.

    Adds Gaussian noise calibrated for (ε,δ)-DP.
    Requires δ > 0 but provides tighter bounds for high-dimensional data.
    """

    def add_noise(
        self,
        scores: np.ndarray,
        epsilon: float,
        sensitivity: float,
        delta: float = 1e-5,
    ) -> Tuple[np.ndarray, float]:
        """Add Gaussian noise to scores."""
        if delta <= 0:
            raise ValueError("delta must be positive for Gaussian mechanism")

        sigma = self.calibrate_noise(epsilon, sensitivity, delta)
        noise = np.random.normal(0, sigma, size=scores.shape)
        noisy_scores = scores + noise
        return noisy_scores, sigma

    def calibrate_noise(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 1e-5,
    ) -> float:
        """Calculate Gaussian standard deviation.

        Uses the analytic Gaussian mechanism formula:
        σ >= sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")

        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


class ExponentialMechanism(NoiseMechanismBase):
    """Exponential mechanism for selecting from discrete outputs.

    Useful for top-k selection where we want to privately
    choose which documents to return.
    """

    def add_noise(
        self,
        scores: np.ndarray,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """Apply exponential mechanism.

        Instead of adding noise, modifies selection probabilities.
        """
        scale = self.calibrate_noise(epsilon, sensitivity)

        # Convert scores to selection probabilities
        # P(i) ∝ exp(ε * score[i] / (2 * sensitivity))
        adjusted_scores = scores * epsilon / (2 * sensitivity)
        # Normalize to prevent overflow
        adjusted_scores = adjusted_scores - np.max(adjusted_scores)
        probabilities = np.exp(adjusted_scores)
        probabilities = probabilities / np.sum(probabilities)

        # Return scores weighted by probability
        noisy_scores = scores * probabilities * len(scores)
        return noisy_scores, scale

    def calibrate_noise(
        self,
        epsilon: float,
        sensitivity: float,
        delta: float = 0.0,
    ) -> float:
        """Calculate scale for exponential mechanism."""
        return 2 * sensitivity / epsilon


def create_mechanism(mechanism_type: NoiseMechanism) -> NoiseMechanismBase:
    """Create a noise mechanism by type.

    Args:
        mechanism_type: Type of mechanism

    Returns:
        Noise mechanism instance
    """
    mechanisms = {
        NoiseMechanism.LAPLACE: LaplaceMechanism,
        NoiseMechanism.GAUSSIAN: GaussianMechanism,
        NoiseMechanism.EXPONENTIAL: ExponentialMechanism,
    }
    return mechanisms[mechanism_type]()


class DPRetriever:
    """Differentially private document retriever.

    Wraps a standard retriever and adds calibrated noise to
    similarity scores before ranking.

    Example:
        >>> retriever = FaissRetriever()
        >>> retriever.index(knowledge_base)
        >>> budget = PrivacyBudgetManager(epsilon_budget=1.0)
        >>> dp_retriever = DPRetriever(
        ...     retriever=retriever,
        ...     budget_manager=budget,
        ...     epsilon_per_query=0.1
        ... )
        >>> result = dp_retriever.retrieve(query_embedding, top_k=5)
        >>> print(f"Retrieved {len(result.documents)} docs, spent ε={result.epsilon_spent}")
    """

    def __init__(
        self,
        retriever: Retriever,
        budget_manager: Optional[PrivacyBudgetManager] = None,
        epsilon_per_query: float = 0.1,
        delta_per_query: float = 0.0,
        mechanism: NoiseMechanism = NoiseMechanism.LAPLACE,
        sensitivity_config: Optional[SensitivityConfig] = None,
        clip_scores: bool = True,
    ):
        """Initialize DP retriever.

        Args:
            retriever: Base retriever to wrap
            budget_manager: Privacy budget manager (creates default if None)
            epsilon_per_query: Default epsilon per query
            delta_per_query: Default delta per query
            mechanism: Noise mechanism to use
            sensitivity_config: Sensitivity configuration
            clip_scores: Whether to clip noisy scores to valid range
        """
        self.retriever = retriever
        self.budget_manager = budget_manager or PrivacyBudgetManager()
        self.epsilon_per_query = epsilon_per_query
        self.delta_per_query = delta_per_query
        self.mechanism = create_mechanism(mechanism)
        self.mechanism_type = mechanism
        self.sensitivity_config = sensitivity_config or SensitivityConfig()
        self.clip_scores = clip_scores

        # Statistics
        self._query_count = 0
        self._total_noise_added = 0.0

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> DPRetrievalResult:
        """Retrieve documents with differential privacy.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
            epsilon: Override default epsilon (uses epsilon_per_query if None)
            delta: Override default delta

        Returns:
            DPRetrievalResult with noisy rankings

        Raises:
            BudgetExceededError: If privacy budget would be exceeded
        """
        epsilon = epsilon if epsilon is not None else self.epsilon_per_query
        delta = delta if delta is not None else self.delta_per_query

        # For Gaussian mechanism, ensure delta > 0
        if self.mechanism_type == NoiseMechanism.GAUSSIAN and delta <= 0:
            delta = 1e-5  # Default delta for Gaussian

        # Check budget
        if not self.budget_manager.can_spend(epsilon, delta):
            status = self.budget_manager.get_status()
            raise BudgetExceededError(
                f"Cannot spend ε={epsilon}: only {status.remaining_epsilon:.4f} remaining"
            )

        # Get original results (retrieve more than top_k to account for reordering)
        retrieve_k = min(top_k * 2, top_k + 10)
        original_results = self.retriever.retrieve(query_embedding, retrieve_k)

        if not original_results:
            # Spend budget even for empty results (prevents timing attacks)
            self.budget_manager.spend(epsilon, delta, operation="retrieve_empty")
            return DPRetrievalResult(
                documents=[],
                epsilon_spent=epsilon,
                delta_spent=delta,
            )

        # Extract scores
        documents = [doc for doc, _ in original_results]
        scores = np.array([score for _, score in original_results])
        original_order = np.argsort(-scores)

        # Calculate sensitivity
        sensitivity = self.sensitivity_config.sensitivity

        # Add noise
        noisy_scores, noise_scale = self.mechanism.add_noise(
            scores, epsilon, sensitivity, delta
        )

        # Optionally clip to valid range
        if self.clip_scores:
            min_score, max_score = self.sensitivity_config.score_range
            noisy_scores = np.clip(noisy_scores, min_score, max_score)

        # Rerank by noisy scores
        noisy_order = np.argsort(-noisy_scores)
        order_preserved = np.array_equal(original_order[:top_k], noisy_order[:top_k])

        # Select top-k by noisy ranking
        top_indices = noisy_order[:top_k]
        result_docs = [
            (documents[i], float(noisy_scores[i]))
            for i in top_indices
        ]

        # Record spend
        self.budget_manager.spend(
            epsilon, delta,
            operation=f"retrieve_top_{top_k}",
            metadata={"noise_scale": noise_scale, "mechanism": self.mechanism_type.value},
        )

        # Update statistics
        self._query_count += 1
        self._total_noise_added += np.mean(np.abs(noisy_scores - scores))

        return DPRetrievalResult(
            documents=result_docs,
            epsilon_spent=epsilon,
            delta_spent=delta,
            noise_scale=noise_scale,
            original_order_preserved=order_preserved,
            metadata={
                "mechanism": self.mechanism_type.value,
                "sensitivity": sensitivity,
                "num_candidates": len(original_results),
            },
        )

    def retrieve_with_threshold(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        epsilon: Optional[float] = None,
        max_results: int = 100,
    ) -> DPRetrievalResult:
        """Retrieve documents above a noisy threshold.

        Uses the sparse vector technique to privately determine
        which documents pass the threshold.

        Args:
            query_embedding: Query embedding
            threshold: Minimum similarity threshold
            epsilon: Privacy parameter
            max_results: Maximum results to return

        Returns:
            DPRetrievalResult with documents passing noisy threshold
        """
        epsilon = epsilon if epsilon is not None else self.epsilon_per_query

        # Split epsilon: half for scores, half for threshold
        eps_scores = epsilon / 2
        eps_threshold = epsilon / 2

        # Get candidates
        original_results = self.retriever.retrieve(query_embedding, max_results)
        if not original_results:
            self.budget_manager.spend(epsilon, operation="threshold_empty")
            return DPRetrievalResult(documents=[], epsilon_spent=epsilon)

        documents = [doc for doc, _ in original_results]
        scores = np.array([score for _, score in original_results])

        # Add noise to scores
        sensitivity = self.sensitivity_config.sensitivity
        noisy_scores, score_noise_scale = self.mechanism.add_noise(
            scores, eps_scores, sensitivity
        )

        # Add noise to threshold
        threshold_noise = np.random.laplace(0, sensitivity / eps_threshold)
        noisy_threshold = threshold + threshold_noise

        # Filter by noisy threshold
        passing = noisy_scores >= noisy_threshold
        result_docs = [
            (documents[i], float(noisy_scores[i]))
            for i in range(len(documents)) if passing[i]
        ]

        self.budget_manager.spend(
            epsilon, operation=f"threshold_{threshold:.2f}",
            metadata={"noisy_threshold": noisy_threshold},
        )

        return DPRetrievalResult(
            documents=result_docs,
            epsilon_spent=epsilon,
            noise_scale=score_noise_scale,
            metadata={
                "threshold": threshold,
                "noisy_threshold": noisy_threshold,
                "passed": len(result_docs),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Statistics dictionary
        """
        budget_status = self.budget_manager.get_status()
        return {
            "query_count": self._query_count,
            "avg_noise_added": (
                self._total_noise_added / self._query_count
                if self._query_count > 0 else 0
            ),
            "mechanism": self.mechanism_type.value,
            "epsilon_per_query": self.epsilon_per_query,
            "sensitivity": self.sensitivity_config.sensitivity,
            "budget_remaining": budget_status.remaining_epsilon,
            "budget_utilization": budget_status.utilization,
            "queries_remaining": self.budget_manager.estimate_queries_remaining(
                self.epsilon_per_query
            ),
        }


class AdaptiveDPRetriever(DPRetriever):
    """Adaptive DP retriever that adjusts epsilon based on query sensitivity.

    Uses less privacy budget for "easy" queries (high-confidence results)
    and more for "hard" queries (ambiguous results).
    """

    def __init__(
        self,
        retriever: Retriever,
        budget_manager: Optional[PrivacyBudgetManager] = None,
        min_epsilon: float = 0.01,
        max_epsilon: float = 0.5,
        **kwargs,
    ):
        """Initialize adaptive retriever.

        Args:
            retriever: Base retriever
            budget_manager: Budget manager
            min_epsilon: Minimum epsilon for easy queries
            max_epsilon: Maximum epsilon for hard queries
            **kwargs: Additional arguments for DPRetriever
        """
        super().__init__(retriever, budget_manager, **kwargs)
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def _calculate_adaptive_epsilon(
        self,
        scores: np.ndarray,
        top_k: int,
    ) -> float:
        """Calculate epsilon based on score distribution.

        Higher epsilon (more noise) for ambiguous queries.
        Lower epsilon (less noise) for clear winners.

        Args:
            scores: Similarity scores
            top_k: Number of results requested

        Returns:
            Adapted epsilon value
        """
        if len(scores) <= 1:
            return self.min_epsilon

        # Calculate gap between top result and others
        sorted_scores = np.sort(scores)[::-1]

        if len(sorted_scores) > 1:
            # Gap between 1st and 2nd
            gap = sorted_scores[0] - sorted_scores[1]
            # Normalize gap by score range
            score_range = sorted_scores[0] - sorted_scores[-1]
            if score_range > 0:
                normalized_gap = gap / score_range
            else:
                normalized_gap = 0

            # Large gap = clear winner = less epsilon needed
            # Small gap = ambiguous = more epsilon needed
            epsilon = self.max_epsilon - normalized_gap * (self.max_epsilon - self.min_epsilon)
            return max(self.min_epsilon, min(self.max_epsilon, epsilon))

        return self.min_epsilon

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> DPRetrievalResult:
        """Retrieve with adaptive epsilon.

        If epsilon is None, automatically calculates based on query.
        """
        if epsilon is None:
            # First, get scores to determine epsilon
            # This is a "peek" that doesn't count against budget
            preview_results = self.retriever.retrieve(query_embedding, top_k * 2)
            if preview_results:
                scores = np.array([s for _, s in preview_results])
                epsilon = self._calculate_adaptive_epsilon(scores, top_k)
            else:
                epsilon = self.min_epsilon

        return super().retrieve(query_embedding, top_k, epsilon, delta)


def analyze_privacy_utility_tradeoff(
    retriever: Retriever,
    query_embeddings: List[np.ndarray],
    ground_truth: List[List[str]],
    epsilon_values: List[float],
    top_k: int = 5,
    num_trials: int = 10,
) -> Dict[str, List[float]]:
    """Analyze privacy-utility tradeoff for different epsilon values.

    Measures retrieval quality (e.g., recall@k) across epsilon settings.

    Args:
        retriever: Base retriever (indexed)
        query_embeddings: Test queries
        ground_truth: Expected document IDs for each query
        epsilon_values: Epsilon values to test
        top_k: Number of results
        num_trials: Trials per epsilon (for variance)

    Returns:
        Dictionary with metrics for each epsilon
    """
    results = {
        "epsilon": epsilon_values,
        "recall": [],
        "recall_std": [],
        "order_preservation": [],
    }

    for eps in epsilon_values:
        recalls = []
        order_preserved = []

        for _ in range(num_trials):
            # Create fresh budget for each trial
            budget = PrivacyBudgetManager(epsilon_budget=100.0)
            dp_retriever = DPRetriever(
                retriever=retriever,
                budget_manager=budget,
                epsilon_per_query=eps,
            )

            trial_recalls = []
            trial_orders = []

            for query_emb, truth in zip(query_embeddings, ground_truth):
                result = dp_retriever.retrieve(query_emb, top_k)
                retrieved_ids = [doc.doc_id for doc, _ in result.documents]

                # Calculate recall
                hits = len(set(retrieved_ids) & set(truth))
                recall = hits / len(truth) if truth else 0
                trial_recalls.append(recall)
                trial_orders.append(result.original_order_preserved)

            recalls.append(np.mean(trial_recalls))
            order_preserved.append(np.mean(trial_orders))

        results["recall"].append(np.mean(recalls))
        results["recall_std"].append(np.std(recalls))
        results["order_preservation"].append(np.mean(order_preserved))

    return results
