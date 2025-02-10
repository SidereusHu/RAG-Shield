"""Privacy Guard for RAG Systems.

Integrates all privacy components into a unified protection layer:
- Differential Privacy for retrieval scores
- Query sanitization for query privacy
- Privacy budget management

Provides a high-level API for:
- Privacy-preserving retrieval
- Query protection
- Budget tracking and alerts
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time

from ragshield.core.document import Document
from ragshield.core.retriever import Retriever
from ragshield.privacy.privacy_budget import (
    PrivacyBudgetManager,
    PrivacyAccountant,
    BudgetStatus,
    BudgetExceededError,
    CompositionType,
)
from ragshield.privacy.dp_retrieval import (
    DPRetriever,
    DPRetrievalResult,
    NoiseMechanism,
    SensitivityConfig,
)
from ragshield.privacy.query_sanitizer import (
    QuerySanitizer,
    PerturbationSanitizer,
    DummyQuerySanitizer,
    LocalDPSanitizer,
    QueryProtectionMethod,
    SanitizedQuery,
    create_query_sanitizer,
    QueryPrivacyConfig,
)


class PrivacyLevel(Enum):
    """Predefined privacy protection levels."""
    MINIMAL = "minimal"  # Low privacy, high utility
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"  # Maximum privacy, lower utility


@dataclass
class PrivacyConfig:
    """Configuration for privacy guard.

    Attributes:
        epsilon_budget: Total epsilon budget
        delta_budget: Total delta budget
        epsilon_per_query: Default epsilon per query
        noise_mechanism: Mechanism for DP
        query_protection: Query protection method
        query_perturbation_scale: Noise scale for query perturbation
        num_dummy_queries: Number of dummy queries
        enable_query_protection: Whether to protect queries
        composition_type: How to compose privacy
        warning_callback: Called when budget low
    """
    epsilon_budget: float = 1.0
    delta_budget: float = 1e-5
    epsilon_per_query: float = 0.1
    noise_mechanism: NoiseMechanism = NoiseMechanism.LAPLACE
    query_protection: QueryProtectionMethod = QueryProtectionMethod.PERTURBATION
    query_perturbation_scale: float = 0.1
    num_dummy_queries: int = 4
    enable_query_protection: bool = True
    composition_type: CompositionType = CompositionType.BASIC
    warning_callback: Optional[Callable[[BudgetStatus], None]] = None


@dataclass
class PrivateRetrievalResult:
    """Result of a privacy-preserving retrieval.

    Attributes:
        documents: Retrieved documents with scores
        epsilon_spent: Total epsilon spent
        delta_spent: Total delta spent
        query_protected: Whether query was protected
        budget_status: Current budget status
        timing: Operation timing information
        metadata: Additional information
    """
    documents: List[Tuple[Document, float]]
    epsilon_spent: float
    delta_spent: float = 0.0
    query_protected: bool = False
    budget_status: Optional[BudgetStatus] = None
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrivacyGuard:
    """High-level privacy protection for RAG systems.

    Provides a unified interface for privacy-preserving retrieval
    with automatic budget management and query protection.

    Example:
        >>> retriever = FaissRetriever()
        >>> retriever.index(knowledge_base)
        >>> guard = PrivacyGuard(retriever)
        >>> result = guard.retrieve(query_embedding, top_k=5)
        >>> print(f"Retrieved {len(result.documents)} docs")
        >>> print(f"Budget remaining: {result.budget_status.remaining_epsilon}")
    """

    def __init__(
        self,
        retriever: Retriever,
        config: Optional[PrivacyConfig] = None,
    ):
        """Initialize privacy guard.

        Args:
            retriever: Base retriever (should be indexed)
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.retriever = retriever

        # Initialize budget manager
        self.budget_manager = PrivacyBudgetManager(
            epsilon_budget=self.config.epsilon_budget,
            delta_budget=self.config.delta_budget,
            composition=self.config.composition_type,
            on_exhausted=self._on_budget_exhausted,
        )

        # Initialize DP retriever
        self.dp_retriever = DPRetriever(
            retriever=retriever,
            budget_manager=self.budget_manager,
            epsilon_per_query=self.config.epsilon_per_query,
            mechanism=self.config.noise_mechanism,
        )

        # Initialize query sanitizer
        if self.config.enable_query_protection:
            query_config = QueryPrivacyConfig(
                method=self.config.query_protection,
                perturbation_scale=self.config.query_perturbation_scale,
                num_dummies=self.config.num_dummy_queries,
            )
            self.query_sanitizer = create_query_sanitizer(query_config)
        else:
            self.query_sanitizer = None

        # Statistics
        self._query_count = 0
        self._total_epsilon_spent = 0.0
        self._warning_issued = False

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        epsilon: Optional[float] = None,
        protect_query: Optional[bool] = None,
    ) -> PrivateRetrievalResult:
        """Perform privacy-preserving retrieval.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
            epsilon: Override default epsilon
            protect_query: Override query protection setting

        Returns:
            PrivateRetrievalResult with protected results

        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        timing = {}
        start_time = time.time()

        # Determine settings
        epsilon = epsilon or self.config.epsilon_per_query
        protect_query = (
            protect_query if protect_query is not None
            else self.config.enable_query_protection
        )

        # Check budget before proceeding
        if not self.budget_manager.can_spend(epsilon):
            raise BudgetExceededError(
                f"Insufficient budget for epsilon={epsilon}"
            )

        # Optionally protect query
        query_to_use = query_embedding
        query_metadata = {}

        if protect_query and self.query_sanitizer is not None:
            sanitize_start = time.time()
            sanitized = self.query_sanitizer.sanitize(query_embedding)
            query_to_use = sanitized.embeddings[sanitized.real_index]
            timing["query_sanitization"] = time.time() - sanitize_start
            query_metadata = {
                "protection_method": sanitized.method.value,
                "query_privacy_cost": sanitized.privacy_cost,
            }

        # Perform DP retrieval
        retrieve_start = time.time()
        dp_result = self.dp_retriever.retrieve(
            query_to_use, top_k=top_k, epsilon=epsilon
        )
        timing["dp_retrieval"] = time.time() - retrieve_start

        # Update statistics
        self._query_count += 1
        self._total_epsilon_spent += dp_result.epsilon_spent

        # Check for budget warning
        status = self.budget_manager.get_status()
        if status.utilization > 0.8 and not self._warning_issued:
            self._warning_issued = True
            if self.config.warning_callback:
                self.config.warning_callback(status)

        timing["total"] = time.time() - start_time

        return PrivateRetrievalResult(
            documents=dp_result.documents,
            epsilon_spent=dp_result.epsilon_spent,
            delta_spent=dp_result.delta_spent,
            query_protected=protect_query,
            budget_status=status,
            timing=timing,
            metadata={
                **dp_result.metadata,
                **query_metadata,
                "noise_scale": dp_result.noise_scale,
                "order_preserved": dp_result.original_order_preserved,
            },
        )

    def retrieve_batch(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 5,
        epsilon_per_query: Optional[float] = None,
    ) -> List[PrivateRetrievalResult]:
        """Retrieve for multiple queries with privacy.

        Useful for batch processing with shared privacy budget.

        Args:
            query_embeddings: List of query embeddings
            top_k: Number of documents per query
            epsilon_per_query: Epsilon for each query

        Returns:
            List of results for each query
        """
        epsilon = epsilon_per_query or self.config.epsilon_per_query

        # Check if we have enough budget for all queries
        total_epsilon = epsilon * len(query_embeddings)
        if not self.budget_manager.can_spend(total_epsilon):
            status = self.budget_manager.get_status()
            max_queries = int(status.remaining_epsilon / epsilon)
            raise BudgetExceededError(
                f"Can only process {max_queries} of {len(query_embeddings)} queries"
            )

        results = []
        for query_emb in query_embeddings:
            result = self.retrieve(query_emb, top_k=top_k, epsilon=epsilon)
            results.append(result)

        return results

    def estimate_queries_remaining(self) -> int:
        """Estimate number of queries that can be made.

        Returns:
            Estimated query count
        """
        return self.budget_manager.estimate_queries_remaining(
            self.config.epsilon_per_query
        )

    def get_budget_status(self) -> BudgetStatus:
        """Get current privacy budget status.

        Returns:
            Budget status
        """
        return self.budget_manager.get_status()

    def get_stats(self) -> Dict[str, Any]:
        """Get privacy guard statistics.

        Returns:
            Statistics dictionary
        """
        status = self.get_budget_status()
        return {
            "query_count": self._query_count,
            "total_epsilon_spent": self._total_epsilon_spent,
            "avg_epsilon_per_query": (
                self._total_epsilon_spent / self._query_count
                if self._query_count > 0 else 0
            ),
            "budget_utilization": status.utilization,
            "budget_remaining": status.remaining_epsilon,
            "queries_remaining": self.estimate_queries_remaining(),
            "query_protection_enabled": self.config.enable_query_protection,
            "noise_mechanism": self.config.noise_mechanism.value,
            "dp_retriever_stats": self.dp_retriever.get_stats(),
        }

    def reset_budget(self) -> None:
        """Reset privacy budget to initial state.

        Warning: This should only be used in specific scenarios
        where budget reset is acceptable (e.g., new session).
        """
        self.budget_manager.reset()
        self._warning_issued = False
        self._query_count = 0
        self._total_epsilon_spent = 0.0

    def _on_budget_exhausted(self, manager: PrivacyBudgetManager) -> None:
        """Handle budget exhaustion."""
        if self.config.warning_callback:
            self.config.warning_callback(manager.get_status())


class PrivacyMonitor:
    """Monitors privacy spending across multiple guards.

    Useful for tracking privacy across a system with multiple
    retrieval endpoints.
    """

    def __init__(self, total_epsilon: float = 10.0):
        """Initialize monitor.

        Args:
            total_epsilon: Total system-wide epsilon budget
        """
        self.total_epsilon = total_epsilon
        self._guards: Dict[str, PrivacyGuard] = {}
        self._accountant = PrivacyAccountant(total_epsilon=total_epsilon)

    def register_guard(
        self,
        name: str,
        guard: PrivacyGuard,
        epsilon_allocation: float,
    ) -> None:
        """Register a privacy guard for monitoring.

        Args:
            name: Unique name for the guard
            guard: Privacy guard to monitor
            epsilon_allocation: Epsilon allocated to this guard
        """
        self._guards[name] = guard
        self._accountant.allocate_pool(name, epsilon_allocation)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system-wide privacy status.

        Returns:
            Status across all guards
        """
        guard_stats = {}
        total_spent = 0.0

        for name, guard in self._guards.items():
            stats = guard.get_stats()
            guard_stats[name] = stats
            total_spent += stats["total_epsilon_spent"]

        return {
            "total_epsilon_budget": self.total_epsilon,
            "total_epsilon_spent": total_spent,
            "remaining": self.total_epsilon - total_spent,
            "utilization": total_spent / self.total_epsilon,
            "guards": guard_stats,
        }


def create_privacy_guard(
    retriever: Retriever,
    level: PrivacyLevel = PrivacyLevel.MEDIUM,
    epsilon_budget: Optional[float] = None,
    **kwargs,
) -> PrivacyGuard:
    """Create a privacy guard with preset configuration.

    Args:
        retriever: Base retriever
        level: Privacy protection level
        epsilon_budget: Override total budget
        **kwargs: Additional configuration options

    Returns:
        Configured PrivacyGuard
    """
    # Preset configurations
    presets = {
        PrivacyLevel.MINIMAL: PrivacyConfig(
            epsilon_budget=10.0,
            epsilon_per_query=1.0,
            enable_query_protection=False,
            noise_mechanism=NoiseMechanism.LAPLACE,
        ),
        PrivacyLevel.LOW: PrivacyConfig(
            epsilon_budget=5.0,
            epsilon_per_query=0.5,
            enable_query_protection=False,
            noise_mechanism=NoiseMechanism.LAPLACE,
        ),
        PrivacyLevel.MEDIUM: PrivacyConfig(
            epsilon_budget=2.0,
            epsilon_per_query=0.2,
            enable_query_protection=True,
            query_protection=QueryProtectionMethod.PERTURBATION,
            query_perturbation_scale=0.1,
            noise_mechanism=NoiseMechanism.LAPLACE,
        ),
        PrivacyLevel.HIGH: PrivacyConfig(
            epsilon_budget=1.0,
            epsilon_per_query=0.1,
            enable_query_protection=True,
            query_protection=QueryProtectionMethod.PERTURBATION,
            query_perturbation_scale=0.2,
            noise_mechanism=NoiseMechanism.GAUSSIAN,
            delta_budget=1e-3,  # Supports ~100 queries at delta=1e-5 each
        ),
        PrivacyLevel.MAXIMUM: PrivacyConfig(
            epsilon_budget=0.5,
            epsilon_per_query=0.05,
            enable_query_protection=True,
            query_protection=QueryProtectionMethod.DUMMY_QUERIES,
            num_dummy_queries=9,  # 10-anonymity
            noise_mechanism=NoiseMechanism.GAUSSIAN,
            delta_budget=1e-3,  # Supports ~100 queries at delta=1e-5 each
        ),
    }

    config = presets[level]

    # Apply overrides
    if epsilon_budget is not None:
        config.epsilon_budget = epsilon_budget

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return PrivacyGuard(retriever, config)


@dataclass
class PrivacyReport:
    """Privacy analysis report.

    Summarizes privacy spending and provides recommendations.
    """
    total_queries: int
    total_epsilon_spent: float
    average_epsilon_per_query: float
    budget_utilization: float
    queries_remaining: int
    recommendations: List[str]
    detailed_history: List[Dict[str, Any]]


def generate_privacy_report(guard: PrivacyGuard) -> PrivacyReport:
    """Generate a privacy analysis report.

    Args:
        guard: Privacy guard to analyze

    Returns:
        PrivacyReport with analysis and recommendations
    """
    stats = guard.get_stats()
    status = guard.get_budget_status()
    history = guard.budget_manager.get_history()

    recommendations = []

    # Analyze spending pattern
    if stats["avg_epsilon_per_query"] > 0.2:
        recommendations.append(
            "Consider reducing epsilon_per_query for more queries"
        )

    if status.utilization > 0.9:
        recommendations.append(
            "Budget nearly exhausted - consider increasing budget or reducing per-query epsilon"
        )

    if not guard.config.enable_query_protection:
        recommendations.append(
            "Enable query protection for additional privacy"
        )

    if guard.config.noise_mechanism == NoiseMechanism.LAPLACE:
        recommendations.append(
            "Consider Gaussian mechanism for tighter bounds with small delta"
        )

    # Build history summary
    detailed_history = [
        {
            "operation": spend.operation,
            "epsilon": spend.epsilon,
            "timestamp": spend.timestamp.isoformat(),
        }
        for spend in history[-100:]  # Last 100 operations
    ]

    return PrivacyReport(
        total_queries=stats["query_count"],
        total_epsilon_spent=stats["total_epsilon_spent"],
        average_epsilon_per_query=stats["avg_epsilon_per_query"],
        budget_utilization=status.utilization,
        queries_remaining=stats["queries_remaining"],
        recommendations=recommendations,
        detailed_history=detailed_history,
    )
