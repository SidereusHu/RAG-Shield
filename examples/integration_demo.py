#!/usr/bin/env python3
"""RAG-Shield Integration Demo.

Demonstrates framework integrations, configuration management,
and production deployment features.
"""

from ragshield.defense import DefenseLevel
from ragshield.integrations import (
    # Configuration
    RAGShieldConfig,
    DetectionConfig,
    DefenseSettings,
    PrivacyConfig,
    MonitoringConfig,
    ConfigManager,
    get_config,
    create_config_manager,
    PROFILES,
    # Logging
    SecurityEventType,
    SecurityEvent,
    SecurityLogger,
    SecurityMetrics,
    get_logger,
    get_metrics,
    configure_logging,
    # Base integration
    IntegrationConfig,
    SecureDocument,
    # Framework availability
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
)


def demo_configuration():
    """Demo: Configuration management."""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Management")
    print("=" * 60)

    # Create custom configuration
    config = RAGShieldConfig(
        detection=DetectionConfig(
            enabled=True,
            preset="strict",
        ),
        defense=DefenseSettings(
            level=DefenseLevel.STRICT,
            auto_quarantine=True,
        ),
        privacy=PrivacyConfig(
            enabled=True,
            epsilon=0.5,
        ),
        monitoring=MonitoringConfig(
            enabled=True,
            log_level="INFO",
            rate_limit_requests=100,
        ),
    )

    print("\nCustom Configuration:")
    print(f"  Detection preset: {config.detection.preset}")
    print(f"  Defense level: {config.defense.level.value}")
    print(f"  Privacy epsilon: {config.privacy.epsilon}")
    print(f"  Rate limit: {config.monitoring.rate_limit_requests} req/min")

    # Convert to JSON
    json_config = config.to_json()
    print(f"\nJSON representation:\n{json_config[:200]}...")


def demo_profiles():
    """Demo: Configuration profiles."""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Profiles")
    print("=" * 60)

    print("\nAvailable profiles:")
    for name, profile in PROFILES.items():
        print(f"\n  {name}:")
        print(f"    Detection: {profile.detection.preset}")
        print(f"    Defense: {profile.defense.level.value}")
        print(f"    Log level: {profile.monitoring.log_level}")

    # Get specific profile
    prod_config = get_config("production")
    print(f"\n\nProduction profile defense level: {prod_config.defense.level.value}")


def demo_config_manager():
    """Demo: Configuration manager."""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Manager")
    print("=" * 60)

    # Create manager
    manager = ConfigManager(load_env=False)

    print("\nDefault values:")
    print(f"  detection.enabled: {manager.get('detection.enabled')}")
    print(f"  defense.level: {manager.get('defense.level').value}")
    print(f"  privacy.epsilon: {manager.get('privacy.epsilon')}")

    # Override at runtime
    print("\nOverriding values at runtime...")
    manager.set("detection.preset", "strict")
    manager.set("privacy.epsilon", 0.1)

    print(f"  detection.preset: {manager.get('detection.preset')}")
    print(f"  privacy.epsilon: {manager.get('privacy.epsilon')}")

    # Validate
    errors = manager.validate()
    print(f"\nValidation errors: {len(errors)}")


def demo_security_logging():
    """Demo: Security logging."""
    print("\n" + "=" * 60)
    print("DEMO: Security Logging")
    print("=" * 60)

    # Create logger
    logger = SecurityLogger(name="demo_logger", level="INFO")

    # Log various events
    print("\nLogging security events...")

    logger.poison_detected(
        doc_id="doc_001",
        confidence=0.85,
        details={"method": "perplexity", "source": "upload"}
    )

    logger.document_blocked(
        doc_id="doc_002",
        reason="High threat score",
        source="api_upload"
    )

    logger.document_quarantined(
        doc_id="doc_003",
        threat_score=0.72
    )

    logger.query_processed(
        query_id="q_001",
        latency_ms=15.5,
        results_count=5
    )

    # Get recent events
    events = logger.get_recent_events(count=10)
    print(f"\nRecent events: {len(events)}")

    for event in events:
        print(f"  [{event.severity}] {event.event_type.value}: {event.message[:50]}...")


def demo_metrics():
    """Demo: Metrics collection."""
    print("\n" + "=" * 60)
    print("DEMO: Metrics Collection")
    print("=" * 60)

    metrics = SecurityMetrics()

    # Simulate activity
    print("\nSimulating security operations...")

    # Detection
    for i in range(20):
        is_poisoned = i % 5 == 0  # 20% poison rate
        metrics.record_detection(
            is_poisoned=is_poisoned,
            confidence=0.3 + (i * 0.03),
            latency_ms=5 + (i * 0.5)
        )

    # Ingestion
    for i in range(15):
        accepted = i % 4 != 0  # 75% acceptance rate
        metrics.record_document_ingested(accepted=accepted)

    # Queries
    for i in range(10):
        metrics.record_query(
            latency_ms=10 + (i * 2),
            results_count=3 + (i % 5)
        )

    # Set gauges
    metrics.set_quarantine_size(5)
    metrics.set_knowledge_base_size(1000)

    # Get summary
    summary = metrics.get_summary()

    print("\nMetrics Summary:")
    print(f"  Uptime: {summary['uptime_seconds']:.1f}s")
    print(f"\n  Detection:")
    print(f"    Total: {summary['detection']['total']}")
    print(f"    Poisoned: {summary['detection']['poisoned']}")
    print(f"    Poison rate: {summary['detection']['poison_rate']:.1%}")

    print(f"\n  Ingestion:")
    print(f"    Total: {summary['ingestion']['total']}")
    print(f"    Rejected: {summary['ingestion']['rejected']}")
    print(f"    Rejection rate: {summary['ingestion']['rejection_rate']:.1%}")

    print(f"\n  Queries:")
    print(f"    Total: {summary['queries']['total']}")

    print(f"\n  System:")
    print(f"    Quarantine size: {summary['quarantine_size']}")
    print(f"    Knowledge base size: {summary['knowledge_base_size']}")


def demo_secure_document():
    """Demo: Secure document wrapper."""
    print("\n" + "=" * 60)
    print("DEMO: Secure Document Wrapper")
    print("=" * 60)

    # Create secure document
    doc = SecureDocument(
        content="This is a sample document for RAG processing.",
        metadata={"source": "api", "user": "admin"},
        doc_id="secure_001",
        is_verified=True,
        threat_score=0.15,
    )

    print(f"\nSecure Document:")
    print(f"  ID: {doc.doc_id}")
    print(f"  Content: {doc.content[:50]}...")
    print(f"  Verified: {doc.is_verified}")
    print(f"  Threat score: {doc.threat_score}")

    # Add security flags
    doc.security_flags.append("scanned")
    doc.security_flags.append("sanitized")
    print(f"  Security flags: {doc.security_flags}")

    # Convert to RAG-Shield Document
    rag_doc = doc.to_document()
    print(f"\n  Converted to Document: {rag_doc.doc_id}")


def demo_framework_integration():
    """Demo: Framework integration setup."""
    print("\n" + "=" * 60)
    print("DEMO: Framework Integration")
    print("=" * 60)

    print("\nFramework availability:")
    print(f"  LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not installed'}")
    print(f"  LlamaIndex: {'Available' if LLAMAINDEX_AVAILABLE else 'Not installed'}")

    # Create integration config
    config = IntegrationConfig(
        defense_level=DefenseLevel.STANDARD,
        detector_preset="default",
        enable_detection=True,
        enable_sanitization=True,
        enable_monitoring=True,
        auto_quarantine=True,
    )

    print(f"\nIntegration Configuration:")
    print(f"  Defense level: {config.defense_level.value}")
    print(f"  Detector preset: {config.detector_preset}")
    print(f"  Detection enabled: {config.enable_detection}")
    print(f"  Auto quarantine: {config.auto_quarantine}")

    if LANGCHAIN_AVAILABLE:
        print("\n  LangChain integration ready!")
        print("  Use: create_langchain_integration() to create wrapper")
    else:
        print("\n  To use LangChain integration:")
        print("    pip install langchain langchain-core")

    if LLAMAINDEX_AVAILABLE:
        print("\n  LlamaIndex integration ready!")
        print("  Use: create_llamaindex_integration() to create wrapper")
    else:
        print("\n  To use LlamaIndex integration:")
        print("    pip install llama-index")


def demo_production_setup():
    """Demo: Production deployment setup."""
    print("\n" + "=" * 60)
    print("DEMO: Production Setup Guide")
    print("=" * 60)

    print("""
Production Deployment Checklist:

1. Configuration
   - Use 'production' or 'high_security' profile
   - Set environment variables for sensitive settings
   - Enable config validation

2. Logging
   - Configure structured logging
   - Set appropriate log levels
   - Add file handlers for persistence

3. Monitoring
   - Enable metrics collection
   - Set up alerts for anomalies
   - Configure rate limiting

4. Integration
   - Wrap all vector stores/retrievers
   - Enable threat callbacks
   - Set up quarantine review workflow

Example setup code:
""")

    print("""
    from ragshield.integrations import (
        create_config_manager,
        configure_logging,
        get_metrics,
    )

    # Load configuration
    config_manager = create_config_manager(
        config_file="config/ragshield.json",
        profile="production",
        load_env=True,
    )

    # Configure logging
    logger = configure_logging(
        level="INFO",
        json_output=True,
        log_file="logs/security.log",
    )

    # Get metrics for monitoring
    metrics = get_metrics()

    # Ready for production!
""")


def main():
    """Run all integration demos."""
    print("\n" + "#" * 60)
    print("#  RAG-Shield Integration & Deployment Demo")
    print("#" * 60)

    demo_configuration()
    demo_profiles()
    demo_config_manager()
    demo_security_logging()
    demo_metrics()
    demo_secure_document()
    demo_framework_integration()
    demo_production_setup()

    print("\n" + "=" * 60)
    print("All integration demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
