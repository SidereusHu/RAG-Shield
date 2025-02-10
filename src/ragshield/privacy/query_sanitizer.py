"""Query Privacy Protection for RAG Systems.

Protects user queries from being revealed to the retrieval server through:
- Query embedding perturbation
- Dummy query mixing (k-anonymity style)
- Query aggregation/batching
- Local differential privacy on queries

Threat model: Semi-honest server that follows protocol but tries to learn queries.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime
import hashlib


class QueryProtectionMethod(Enum):
    """Methods for protecting query privacy."""
    PERTURBATION = "perturbation"  # Add noise to query embedding
    DUMMY_QUERIES = "dummy_queries"  # Mix real query with fake ones
    AGGREGATION = "aggregation"  # Batch queries together
    LOCAL_DP = "local_dp"  # Local differential privacy


@dataclass
class SanitizedQuery:
    """Result of query sanitization.

    Attributes:
        embeddings: Protected query embedding(s)
        real_index: Index of real query (if using dummies)
        method: Protection method used
        privacy_cost: Privacy spent (epsilon)
        metadata: Additional information
    """
    embeddings: np.ndarray  # Shape: (n_queries, embedding_dim)
    real_index: int = 0
    method: QueryProtectionMethod = QueryProtectionMethod.PERTURBATION
    privacy_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_queries(self) -> int:
        """Number of queries (including dummies)."""
        return len(self.embeddings)


class QuerySanitizer(ABC):
    """Abstract base class for query sanitizers."""

    @abstractmethod
    def sanitize(self, query_embedding: np.ndarray) -> SanitizedQuery:
        """Sanitize a query embedding.

        Args:
            query_embedding: Original query embedding

        Returns:
            Sanitized query result
        """
        pass

    @abstractmethod
    def extract_result(
        self,
        sanitized_query: SanitizedQuery,
        all_results: List[List[Tuple[Any, float]]],
    ) -> List[Tuple[Any, float]]:
        """Extract the real result from mixed results.

        Args:
            sanitized_query: The sanitized query used
            all_results: Results for all queries (including dummies)

        Returns:
            Results for the real query
        """
        pass


class PerturbationSanitizer(QuerySanitizer):
    """Sanitizes queries by adding noise to embeddings.

    Adds calibrated noise to query embeddings to prevent exact
    query reconstruction while maintaining utility.

    Example:
        >>> sanitizer = PerturbationSanitizer(noise_scale=0.1)
        >>> sanitized = sanitizer.sanitize(query_embedding)
        >>> # Use sanitized.embeddings[0] for retrieval
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        noise_type: str = "gaussian",
        normalize_after: bool = True,
        epsilon: float = 1.0,
    ):
        """Initialize perturbation sanitizer.

        Args:
            noise_scale: Scale of noise relative to embedding norm
            noise_type: Type of noise ('gaussian' or 'laplace')
            normalize_after: Re-normalize embedding after noise
            epsilon: Privacy parameter for budget tracking
        """
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.normalize_after = normalize_after
        self.epsilon = epsilon

    def sanitize(self, query_embedding: np.ndarray) -> SanitizedQuery:
        """Add noise to query embedding."""
        query = query_embedding.flatten()
        embedding_norm = np.linalg.norm(query)

        # Generate noise
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, 1, query.shape)
        elif self.noise_type == "laplace":
            noise = np.random.laplace(0, 1, query.shape)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        # Scale noise relative to embedding magnitude
        noise = noise * self.noise_scale * embedding_norm

        # Add noise
        noisy_query = query + noise

        # Optionally renormalize
        if self.normalize_after and embedding_norm > 0:
            noisy_query = noisy_query / np.linalg.norm(noisy_query) * embedding_norm

        return SanitizedQuery(
            embeddings=noisy_query.reshape(1, -1),
            real_index=0,
            method=QueryProtectionMethod.PERTURBATION,
            privacy_cost=self.epsilon,
            metadata={
                "noise_scale": self.noise_scale,
                "noise_type": self.noise_type,
                "actual_noise_norm": np.linalg.norm(noise),
            },
        )

    def extract_result(
        self,
        sanitized_query: SanitizedQuery,
        all_results: List[List[Tuple[Any, float]]],
    ) -> List[Tuple[Any, float]]:
        """Extract result (trivial for perturbation)."""
        return all_results[0] if all_results else []


class DummyQuerySanitizer(QuerySanitizer):
    """Mixes real query with dummy queries for k-anonymity.

    Generates plausible dummy queries to hide the real query
    among k-1 decoys. Server cannot distinguish real from fake.

    Example:
        >>> sanitizer = DummyQuerySanitizer(num_dummies=4)
        >>> sanitized = sanitizer.sanitize(query_embedding)
        >>> # Send all 5 queries to server
        >>> # Extract real result using sanitized.real_index
    """

    def __init__(
        self,
        num_dummies: int = 4,
        dummy_generator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        shuffle: bool = True,
    ):
        """Initialize dummy query sanitizer.

        Args:
            num_dummies: Number of dummy queries (k-1 for k-anonymity)
            dummy_generator: Function to generate dummy embeddings
            shuffle: Randomly shuffle query order
        """
        self.num_dummies = num_dummies
        self.dummy_generator = dummy_generator or self._default_dummy_generator
        self.shuffle = shuffle

    def _default_dummy_generator(self, real_query: np.ndarray) -> np.ndarray:
        """Generate a plausible dummy query.

        Creates dummy that has similar statistical properties
        to the real query to be indistinguishable.
        """
        # Generate random direction
        dummy = np.random.randn(*real_query.shape)

        # Match the norm of real query
        real_norm = np.linalg.norm(real_query)
        if real_norm > 0:
            dummy = dummy / np.linalg.norm(dummy) * real_norm

        return dummy

    def sanitize(self, query_embedding: np.ndarray) -> SanitizedQuery:
        """Mix real query with dummies."""
        query = query_embedding.flatten().reshape(1, -1)

        # Generate dummies
        dummies = np.array([
            self.dummy_generator(query.flatten())
            for _ in range(self.num_dummies)
        ])

        # Combine real and dummies
        all_queries = np.vstack([query, dummies])

        # Shuffle if enabled
        if self.shuffle:
            indices = np.random.permutation(len(all_queries))
            all_queries = all_queries[indices]
            real_index = int(np.where(indices == 0)[0][0])
        else:
            real_index = 0

        return SanitizedQuery(
            embeddings=all_queries,
            real_index=real_index,
            method=QueryProtectionMethod.DUMMY_QUERIES,
            privacy_cost=0.0,  # Information-theoretic protection
            metadata={
                "num_dummies": self.num_dummies,
                "k_anonymity": self.num_dummies + 1,
            },
        )

    def extract_result(
        self,
        sanitized_query: SanitizedQuery,
        all_results: List[List[Tuple[Any, float]]],
    ) -> List[Tuple[Any, float]]:
        """Extract the real query's results."""
        if sanitized_query.real_index < len(all_results):
            return all_results[sanitized_query.real_index]
        return []


class SemanticDummyGenerator:
    """Generates semantically plausible dummy queries.

    Uses a pool of pre-computed embeddings from common queries
    to generate dummies that look like real queries.
    """

    def __init__(
        self,
        embedding_pool: Optional[np.ndarray] = None,
        pool_size: int = 1000,
        embedding_dim: int = 768,
    ):
        """Initialize semantic dummy generator.

        Args:
            embedding_pool: Pre-computed embedding pool
            pool_size: Size of random pool if none provided
            embedding_dim: Embedding dimension
        """
        if embedding_pool is not None:
            self.pool = embedding_pool
        else:
            # Generate random pool (in practice, use real query embeddings)
            self.pool = np.random.randn(pool_size, embedding_dim)
            # Normalize
            norms = np.linalg.norm(self.pool, axis=1, keepdims=True)
            self.pool = self.pool / norms

    def generate(self, real_query: np.ndarray) -> np.ndarray:
        """Generate a dummy query from the pool.

        Selects a query from pool that is somewhat similar
        to make traffic analysis harder.
        """
        # Find queries with moderate similarity (not too similar, not too different)
        real_norm = np.linalg.norm(real_query)
        normalized_real = real_query.flatten() / real_norm if real_norm > 0 else real_query.flatten()

        similarities = np.dot(self.pool, normalized_real)

        # Select from middle similarity range
        target_sim = np.random.uniform(0.2, 0.5)
        distances = np.abs(similarities - target_sim)
        best_idx = np.argmin(distances)

        # Scale to match real query's norm
        dummy = self.pool[best_idx] * real_norm
        return dummy


class LocalDPSanitizer(QuerySanitizer):
    """Applies local differential privacy to queries.

    Each user applies randomized response or other LDP mechanism
    locally before sending query to server.

    Example:
        >>> sanitizer = LocalDPSanitizer(epsilon=2.0)
        >>> sanitized = sanitizer.sanitize(query_embedding)
    """

    def __init__(
        self,
        epsilon: float = 2.0,
        mechanism: str = "laplace",
        clip_norm: Optional[float] = None,
    ):
        """Initialize LDP sanitizer.

        Args:
            epsilon: Privacy parameter
            mechanism: LDP mechanism ('laplace', 'gaussian', 'randomized_response')
            clip_norm: Clip embedding to this norm before adding noise
        """
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.clip_norm = clip_norm

    def sanitize(self, query_embedding: np.ndarray) -> SanitizedQuery:
        """Apply local DP to query."""
        query = query_embedding.flatten()
        dim = len(query)

        # Optionally clip norm
        if self.clip_norm is not None:
            norm = np.linalg.norm(query)
            if norm > self.clip_norm:
                query = query / norm * self.clip_norm
            sensitivity = self.clip_norm
        else:
            sensitivity = np.linalg.norm(query)

        # Add noise based on mechanism
        if self.mechanism == "laplace":
            # Laplace mechanism: sensitivity/epsilon per dimension
            scale = sensitivity * np.sqrt(dim) / self.epsilon
            noise = np.random.laplace(0, scale, dim)
        elif self.mechanism == "gaussian":
            # Gaussian mechanism (requires delta)
            delta = 1e-5
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
            noise = np.random.normal(0, sigma, dim)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

        noisy_query = query + noise

        return SanitizedQuery(
            embeddings=noisy_query.reshape(1, -1),
            real_index=0,
            method=QueryProtectionMethod.LOCAL_DP,
            privacy_cost=self.epsilon,
            metadata={
                "mechanism": self.mechanism,
                "sensitivity": sensitivity,
                "noise_norm": np.linalg.norm(noise),
            },
        )

    def extract_result(
        self,
        sanitized_query: SanitizedQuery,
        all_results: List[List[Tuple[Any, float]]],
    ) -> List[Tuple[Any, float]]:
        """Extract result (trivial for LDP)."""
        return all_results[0] if all_results else []


class QueryAggregator:
    """Aggregates multiple queries for batch privacy.

    Collects queries and processes them in batches to reduce
    per-query information leakage through traffic analysis.
    """

    def __init__(
        self,
        batch_size: int = 10,
        timeout_seconds: float = 5.0,
        shuffle_batch: bool = True,
    ):
        """Initialize query aggregator.

        Args:
            batch_size: Number of queries per batch
            timeout_seconds: Max wait time for batch
            shuffle_batch: Shuffle order within batch
        """
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.shuffle_batch = shuffle_batch
        self._pending_queries: List[Tuple[np.ndarray, str]] = []
        self._query_mapping: Dict[str, int] = {}

    def add_query(self, query_embedding: np.ndarray, query_id: str) -> bool:
        """Add a query to the batch.

        Args:
            query_embedding: Query embedding
            query_id: Unique query identifier

        Returns:
            True if batch is ready to process
        """
        self._pending_queries.append((query_embedding, query_id))
        self._query_mapping[query_id] = len(self._pending_queries) - 1
        return len(self._pending_queries) >= self.batch_size

    def get_batch(self) -> Optional[SanitizedQuery]:
        """Get the current batch if ready.

        Returns:
            SanitizedQuery with batch, or None if not ready
        """
        if len(self._pending_queries) < self.batch_size:
            return None

        # Extract batch
        batch_queries = self._pending_queries[:self.batch_size]
        self._pending_queries = self._pending_queries[self.batch_size:]

        embeddings = np.array([q for q, _ in batch_queries])
        query_ids = [qid for _, qid in batch_queries]

        # Shuffle if enabled
        if self.shuffle_batch:
            perm = np.random.permutation(len(embeddings))
            embeddings = embeddings[perm]
            query_ids = [query_ids[i] for i in perm]

        return SanitizedQuery(
            embeddings=embeddings,
            method=QueryProtectionMethod.AGGREGATION,
            metadata={
                "batch_size": len(embeddings),
                "query_ids": query_ids,
            },
        )

    def pending_count(self) -> int:
        """Number of pending queries."""
        return len(self._pending_queries)


class CompositeSanitizer(QuerySanitizer):
    """Combines multiple sanitization methods.

    Applies multiple protection layers for defense in depth.

    Example:
        >>> sanitizer = CompositeSanitizer([
        ...     PerturbationSanitizer(noise_scale=0.05),
        ...     DummyQuerySanitizer(num_dummies=3),
        ... ])
    """

    def __init__(self, sanitizers: List[QuerySanitizer]):
        """Initialize composite sanitizer.

        Args:
            sanitizers: List of sanitizers to apply in order
        """
        if not sanitizers:
            raise ValueError("At least one sanitizer required")
        self.sanitizers = sanitizers

    def sanitize(self, query_embedding: np.ndarray) -> SanitizedQuery:
        """Apply all sanitizers in sequence."""
        current = query_embedding
        total_privacy_cost = 0.0
        metadata = {"layers": []}

        for sanitizer in self.sanitizers:
            result = sanitizer.sanitize(current)
            current = result.embeddings[result.real_index]
            total_privacy_cost += result.privacy_cost
            metadata["layers"].append({
                "method": result.method.value,
                "cost": result.privacy_cost,
            })

        # Final sanitization
        final_result = self.sanitizers[-1].sanitize(current)
        final_result.privacy_cost = total_privacy_cost
        final_result.metadata.update(metadata)

        return final_result

    def extract_result(
        self,
        sanitized_query: SanitizedQuery,
        all_results: List[List[Tuple[Any, float]]],
    ) -> List[Tuple[Any, float]]:
        """Extract result through all layers (reverse order)."""
        current_results = all_results
        for sanitizer in reversed(self.sanitizers):
            if hasattr(sanitizer, 'extract_result'):
                # This is simplified - full implementation would track indices
                current_results = [sanitizer.extract_result(sanitized_query, current_results)]
        return current_results[0] if current_results else []


@dataclass
class QueryPrivacyConfig:
    """Configuration for query privacy protection.

    Attributes:
        method: Protection method to use
        perturbation_scale: Noise scale for perturbation
        num_dummies: Number of dummy queries
        epsilon: Privacy parameter
        enable_aggregation: Whether to use query batching
    """
    method: QueryProtectionMethod = QueryProtectionMethod.PERTURBATION
    perturbation_scale: float = 0.1
    num_dummies: int = 4
    epsilon: float = 1.0
    enable_aggregation: bool = False
    aggregation_batch_size: int = 10


def create_query_sanitizer(config: QueryPrivacyConfig) -> QuerySanitizer:
    """Create a query sanitizer from configuration.

    Args:
        config: Privacy configuration

    Returns:
        Configured sanitizer
    """
    if config.method == QueryProtectionMethod.PERTURBATION:
        return PerturbationSanitizer(
            noise_scale=config.perturbation_scale,
            epsilon=config.epsilon,
        )
    elif config.method == QueryProtectionMethod.DUMMY_QUERIES:
        return DummyQuerySanitizer(num_dummies=config.num_dummies)
    elif config.method == QueryProtectionMethod.LOCAL_DP:
        return LocalDPSanitizer(epsilon=config.epsilon)
    else:
        return PerturbationSanitizer()
