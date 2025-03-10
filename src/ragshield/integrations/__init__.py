"""RAG-Shield Integrations Module.

Provides integrations with popular RAG frameworks and
production deployment utilities.

Supported Frameworks:
- LangChain: Document loaders, vector stores, retrievers
- LlamaIndex: Indices, query engines, retrievers

Production Utilities:
- Configuration management with environment variables
- Structured security logging
- Metrics collection and monitoring
"""

from ragshield.integrations.base import (
    FrameworkType,
    IntegrationConfig,
    SecureDocument,
    BaseRAGIntegration,
)

from ragshield.integrations.langchain_adapter import (
    LangChainIntegration,
    SecureRetriever,
    SecureVectorStore,
    SecureDocumentLoader,
    create_langchain_integration,
    LANGCHAIN_AVAILABLE,
)

from ragshield.integrations.llamaindex_adapter import (
    LlamaIndexIntegration,
    SecureLlamaRetriever,
    SecureLlamaIndex,
    SecureQueryEngine,
    create_llamaindex_integration,
    LLAMAINDEX_AVAILABLE,
)

from ragshield.integrations.config import (
    ConfigSource,
    DetectionConfig,
    DefenseSettings,
    PrivacyConfig,
    MonitoringConfig,
    ForensicsConfig,
    RAGShieldConfig,
    ConfigManager,
    PROFILES,
    get_config,
    create_config_manager,
)

from ragshield.integrations.logging import (
    SecurityEventType,
    SecurityEvent,
    SecurityLogger,
    MetricsCollector,
    SecurityMetrics,
    get_logger,
    get_metrics,
    configure_logging,
)


__all__ = [
    # Base
    "FrameworkType",
    "IntegrationConfig",
    "SecureDocument",
    "BaseRAGIntegration",
    # LangChain
    "LangChainIntegration",
    "SecureRetriever",
    "SecureVectorStore",
    "SecureDocumentLoader",
    "create_langchain_integration",
    "LANGCHAIN_AVAILABLE",
    # LlamaIndex
    "LlamaIndexIntegration",
    "SecureLlamaRetriever",
    "SecureLlamaIndex",
    "SecureQueryEngine",
    "create_llamaindex_integration",
    "LLAMAINDEX_AVAILABLE",
    # Configuration
    "ConfigSource",
    "DetectionConfig",
    "DefenseSettings",
    "PrivacyConfig",
    "MonitoringConfig",
    "ForensicsConfig",
    "RAGShieldConfig",
    "ConfigManager",
    "PROFILES",
    "get_config",
    "create_config_manager",
    # Logging
    "SecurityEventType",
    "SecurityEvent",
    "SecurityLogger",
    "MetricsCollector",
    "SecurityMetrics",
    "get_logger",
    "get_metrics",
    "configure_logging",
]
