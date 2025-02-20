"""Defense components for RAG systems.

This module provides active defense mechanisms against poisoning attacks.

Components:
- Quarantine: Isolation and review of suspicious documents
- Monitor: Real-time monitoring and alerting
- Sanitizer: Input validation and cleaning
- Shield: Unified protection layer

Example:
    >>> from ragshield.defense import RAGShield, DefenseLevel
    >>>
    >>> # Create shield with standard protection
    >>> shield = RAGShield(knowledge_base, detector)
    >>> shield.set_level(DefenseLevel.STRICT)
    >>>
    >>> # Ingest with protection
    >>> result = shield.ingest(document, source="user_upload")
    >>> if not result.success:
    ...     print(f"Blocked: {result.action_taken}")
"""

# Quarantine
from ragshield.defense.quarantine import (
    QuarantineStatus,
    QuarantineAction,
    QuarantineEntry,
    QuarantineDecision,
    QuarantineManager,
)

# Monitoring
from ragshield.defense.monitor import (
    AlertSeverity,
    AlertType,
    Alert,
    RateLimitRule,
    MonitoringMetrics,
    RateLimiter,
    AnomalyDetector,
    SecurityMonitor,
)

# Sanitization
from ragshield.defense.sanitizer import (
    SanitizationAction,
    SanitizationRule,
    SanitizationResult,
    EmbeddingValidationResult,
    ContentSanitizer,
    EmbeddingSanitizer,
    MetadataSanitizer,
    DocumentSanitizer,
)

# Shield
from ragshield.defense.shield import (
    DefenseLevel,
    DefenseConfig,
    IngestionResult,
    ShieldStatus,
    RAGShield,
    create_shield,
)

__all__ = [
    # Quarantine
    "QuarantineStatus",
    "QuarantineAction",
    "QuarantineEntry",
    "QuarantineDecision",
    "QuarantineManager",
    # Monitoring
    "AlertSeverity",
    "AlertType",
    "Alert",
    "RateLimitRule",
    "MonitoringMetrics",
    "RateLimiter",
    "AnomalyDetector",
    "SecurityMonitor",
    # Sanitization
    "SanitizationAction",
    "SanitizationRule",
    "SanitizationResult",
    "EmbeddingValidationResult",
    "ContentSanitizer",
    "EmbeddingSanitizer",
    "MetadataSanitizer",
    "DocumentSanitizer",
    # Shield
    "DefenseLevel",
    "DefenseConfig",
    "IngestionResult",
    "ShieldStatus",
    "RAGShield",
    "create_shield",
]
