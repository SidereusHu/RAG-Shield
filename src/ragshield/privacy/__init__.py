"""Privacy protection components for RAG systems.

This module provides differential privacy and query privacy protections:

- Privacy Budget Management: Track and limit privacy spending
- Differential Privacy Retrieval: Add calibrated noise to retrieval scores
- Query Sanitization: Protect user queries from server observation
- Privacy Guard: Unified high-level protection layer

Example:
    >>> from ragshield.privacy import PrivacyGuard, create_privacy_guard, PrivacyLevel
    >>> from ragshield.core.retriever import SimpleRetriever
    >>>
    >>> # Create retriever and index
    >>> retriever = SimpleRetriever()
    >>> retriever.index(knowledge_base)
    >>>
    >>> # Create privacy guard with medium protection
    >>> guard = create_privacy_guard(retriever, level=PrivacyLevel.MEDIUM)
    >>>
    >>> # Perform private retrieval
    >>> result = guard.retrieve(query_embedding, top_k=5)
    >>> print(f"Found {len(result.documents)} documents")
    >>> print(f"Privacy spent: ε={result.epsilon_spent}")
    >>> print(f"Budget remaining: {result.budget_status.remaining_epsilon}")
"""

# Privacy Budget
from ragshield.privacy.privacy_budget import (
    PrivacyBudgetManager,
    PrivacyAccountant,
    PrivacySpend,
    BudgetStatus,
    BudgetExceededError,
    CompositionType,
)

# Differential Privacy Retrieval
from ragshield.privacy.dp_retrieval import (
    DPRetriever,
    DPRetrievalResult,
    NoiseMechanism,
    SensitivityConfig,
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    AdaptiveDPRetriever,
    analyze_privacy_utility_tradeoff,
)

# Query Sanitization
from ragshield.privacy.query_sanitizer import (
    QuerySanitizer,
    PerturbationSanitizer,
    DummyQuerySanitizer,
    LocalDPSanitizer,
    CompositeSanitizer,
    QueryAggregator,
    SemanticDummyGenerator,
    QueryProtectionMethod,
    SanitizedQuery,
    QueryPrivacyConfig,
    create_query_sanitizer,
)

# Privacy Guard
from ragshield.privacy.guard import (
    PrivacyGuard,
    PrivacyConfig,
    PrivacyLevel,
    PrivacyMonitor,
    PrivateRetrievalResult,
    PrivacyReport,
    create_privacy_guard,
    generate_privacy_report,
)

__all__ = [
    # Budget Management
    "PrivacyBudgetManager",
    "PrivacyAccountant",
    "PrivacySpend",
    "BudgetStatus",
    "BudgetExceededError",
    "CompositionType",
    # DP Retrieval
    "DPRetriever",
    "DPRetrievalResult",
    "NoiseMechanism",
    "SensitivityConfig",
    "LaplaceMechanism",
    "GaussianMechanism",
    "ExponentialMechanism",
    "AdaptiveDPRetriever",
    "analyze_privacy_utility_tradeoff",
    # Query Sanitization
    "QuerySanitizer",
    "PerturbationSanitizer",
    "DummyQuerySanitizer",
    "LocalDPSanitizer",
    "CompositeSanitizer",
    "QueryAggregator",
    "SemanticDummyGenerator",
    "QueryProtectionMethod",
    "SanitizedQuery",
    "QueryPrivacyConfig",
    "create_query_sanitizer",
    # Privacy Guard
    "PrivacyGuard",
    "PrivacyConfig",
    "PrivacyLevel",
    "PrivacyMonitor",
    "PrivateRetrievalResult",
    "PrivacyReport",
    "create_privacy_guard",
    "generate_privacy_report",
]
