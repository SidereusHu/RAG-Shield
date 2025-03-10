"""Tests for RAG-Shield integrations module."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from ragshield.core.document import Document
from ragshield.defense import DefenseLevel

from ragshield.integrations import (
    # Base
    FrameworkType,
    IntegrationConfig,
    SecureDocument,
    # Configuration
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
    # Logging
    SecurityEventType,
    SecurityEvent,
    SecurityLogger,
    MetricsCollector,
    SecurityMetrics,
    get_logger,
    get_metrics,
    # Framework availability
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
)


# ============================================================================
# Base Integration Tests
# ============================================================================

class TestIntegrationConfig:
    """Tests for IntegrationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IntegrationConfig()

        assert config.defense_level == DefenseLevel.STANDARD
        assert config.detector_preset == "default"
        assert config.enable_detection is True
        assert config.enable_sanitization is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegrationConfig(
            defense_level=DefenseLevel.STRICT,
            detector_preset="strict",
            enable_detection=True,
            auto_quarantine=False,
        )

        assert config.defense_level == DefenseLevel.STRICT
        assert config.detector_preset == "strict"
        assert config.auto_quarantine is False


class TestSecureDocument:
    """Tests for SecureDocument."""

    def test_creation(self):
        """Test secure document creation."""
        doc = SecureDocument(
            content="Test content",
            metadata={"source": "test"},
            is_verified=True,
            threat_score=0.1,
        )

        assert doc.content == "Test content"
        assert doc.is_verified is True
        assert doc.threat_score == 0.1

    def test_to_document(self):
        """Test conversion to Document."""
        secure_doc = SecureDocument(
            content="Test content",
            doc_id="test_123",
            embedding=[0.1, 0.2, 0.3],
        )

        doc = secure_doc.to_document()

        assert isinstance(doc, Document)
        assert doc.doc_id == "test_123"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]

    def test_from_document(self):
        """Test creation from Document."""
        doc = Document(
            doc_id="test_123",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
        )

        secure_doc = SecureDocument.from_document(
            doc,
            is_verified=True,
            threat_score=0.2,
        )

        assert secure_doc.content == "Test content"
        assert secure_doc.doc_id == "test_123"
        assert secure_doc.is_verified is True
        assert secure_doc.threat_score == 0.2

    def test_security_flags(self):
        """Test security flags."""
        doc = SecureDocument(content="Test")
        doc.security_flags.append("potential_poison")
        doc.security_flags.append("high_perplexity")

        assert "potential_poison" in doc.security_flags
        assert len(doc.security_flags) == 2


# ============================================================================
# Configuration Tests
# ============================================================================

class TestDetectionConfig:
    """Tests for DetectionConfig."""

    def test_default(self):
        """Test default detection config."""
        config = DetectionConfig()

        assert config.enabled is True
        assert config.preset == "default"

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = DetectionConfig(preset="strict")
        d = config.to_dict()

        assert d["preset"] == "strict"
        assert d["enabled"] is True


class TestDefenseSettings:
    """Tests for DefenseSettings."""

    def test_default(self):
        """Test default defense settings."""
        settings = DefenseSettings()

        assert settings.level == DefenseLevel.STANDARD
        assert settings.auto_quarantine is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        settings = DefenseSettings(level=DefenseLevel.STRICT)
        d = settings.to_dict()

        assert d["level"] == "strict"


class TestRAGShieldConfig:
    """Tests for RAGShieldConfig."""

    def test_default(self):
        """Test default configuration."""
        config = RAGShieldConfig()

        assert config.detection.enabled is True
        assert config.defense.level == DefenseLevel.STANDARD
        assert config.privacy.enabled is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = RAGShieldConfig()
        d = config.to_dict()

        assert "detection" in d
        assert "defense" in d
        assert "privacy" in d
        assert "monitoring" in d
        assert "forensics" in d

    def test_to_json(self):
        """Test JSON conversion."""
        config = RAGShieldConfig()
        json_str = config.to_json()

        parsed = json.loads(json_str)
        assert "detection" in parsed

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "detection": {"enabled": False, "preset": "strict"},
            "defense": {"level": "paranoid", "auto_quarantine": False},
        }

        config = RAGShieldConfig.from_dict(data)

        assert config.detection.enabled is False
        assert config.detection.preset == "strict"
        assert config.defense.level == DefenseLevel.PARANOID

    def test_save_and_load(self):
        """Test saving and loading config."""
        config = RAGShieldConfig(
            detection=DetectionConfig(preset="strict"),
            defense=DefenseSettings(level=DefenseLevel.PARANOID),
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            loaded = RAGShieldConfig.load(f.name)

        assert loaded.detection.preset == "strict"
        assert loaded.defense.level == DefenseLevel.PARANOID


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_default_config(self):
        """Test default configuration."""
        manager = ConfigManager(load_env=False)

        assert manager.config.detection.enabled is True

    def test_get_dotted_key(self):
        """Test getting values by dotted key."""
        manager = ConfigManager(load_env=False)

        assert manager.get("detection.enabled") is True
        assert manager.get("defense.level") == DefenseLevel.STANDARD
        assert manager.get("nonexistent.key", "default") == "default"

    def test_set_value(self):
        """Test setting values."""
        manager = ConfigManager(load_env=False)

        manager.set("detection.enabled", False)
        assert manager.get("detection.enabled") is False

    def test_reset(self):
        """Test resetting configuration."""
        manager = ConfigManager(load_env=False)

        manager.set("detection.enabled", False)
        manager.reset()

        assert manager.get("detection.enabled") is True

    def test_validate(self):
        """Test configuration validation."""
        manager = ConfigManager(load_env=False)

        # Valid config
        errors = manager.validate()
        assert len(errors) == 0

        # Invalid epsilon
        manager.config.privacy.epsilon = -1
        errors = manager.validate()
        assert any("epsilon" in e.lower() for e in errors)

    def test_profiles(self):
        """Test configuration profiles."""
        assert "development" in PROFILES
        assert "production" in PROFILES
        assert "high_security" in PROFILES

        dev_config = PROFILES["development"]
        assert dev_config.defense.level == DefenseLevel.MINIMAL

        hs_config = PROFILES["high_security"]
        assert hs_config.defense.level == DefenseLevel.PARANOID


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_production(self):
        """Test getting production profile."""
        config = get_config("production")

        assert config.defense.level == DefenseLevel.STANDARD

    def test_get_development(self):
        """Test getting development profile."""
        config = get_config("development")

        assert config.defense.level == DefenseLevel.MINIMAL

    def test_get_unknown(self):
        """Test getting unknown profile."""
        config = get_config("unknown_profile")

        # Should return default config
        assert config.defense.level == DefenseLevel.STANDARD


# ============================================================================
# Logging Tests
# ============================================================================

class TestSecurityEvent:
    """Tests for SecurityEvent."""

    def test_creation(self):
        """Test event creation."""
        event = SecurityEvent(
            event_type=SecurityEventType.POISON_DETECTED,
            severity="WARNING",
            message="Test poison detected",
            doc_id="doc_123",
        )

        assert event.event_type == SecurityEventType.POISON_DETECTED
        assert event.severity == "WARNING"
        assert event.doc_id == "doc_123"

    def test_to_dict(self):
        """Test dictionary conversion."""
        event = SecurityEvent(
            event_type=SecurityEventType.DOCUMENT_BLOCKED,
            message="Blocked",
            details={"reason": "suspicious"},
        )

        d = event.to_dict()

        assert d["event_type"] == "document_blocked"
        assert d["message"] == "Blocked"
        assert d["details"]["reason"] == "suspicious"

    def test_to_json(self):
        """Test JSON conversion."""
        event = SecurityEvent(
            event_type=SecurityEventType.QUERY_PROCESSED,
            message="Query OK",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "query_processed"


class TestSecurityLogger:
    """Tests for SecurityLogger."""

    def test_creation(self):
        """Test logger creation."""
        logger = SecurityLogger(name="test_logger", level="DEBUG")

        assert logger.logger.name == "test_logger"

    def test_log_event(self):
        """Test logging an event."""
        logger = SecurityLogger(name="test", level="DEBUG")

        event = SecurityEvent(
            event_type=SecurityEventType.POISON_DETECTED,
            message="Test",
        )

        logger.log_event(event)

        # Check event history
        events = logger.get_recent_events(count=10)
        assert len(events) == 1
        assert events[0].event_type == SecurityEventType.POISON_DETECTED

    def test_poison_detected(self):
        """Test poison_detected convenience method."""
        logger = SecurityLogger(name="test")

        logger.poison_detected("doc_123", 0.85, {"method": "perplexity"})

        events = logger.get_recent_events()
        assert len(events) == 1
        assert events[0].event_type == SecurityEventType.POISON_DETECTED
        assert events[0].doc_id == "doc_123"

    def test_document_blocked(self):
        """Test document_blocked convenience method."""
        logger = SecurityLogger(name="test")

        logger.document_blocked("doc_123", "High threat score")

        events = logger.get_recent_events()
        assert events[0].event_type == SecurityEventType.DOCUMENT_BLOCKED

    def test_event_callback(self):
        """Test event callbacks."""
        logger = SecurityLogger(name="test")
        received_events = []

        def callback(event):
            received_events.append(event)

        logger.add_callback(callback)
        logger.poison_detected("doc_123", 0.9)

        assert len(received_events) == 1
        assert received_events[0].doc_id == "doc_123"

    def test_filter_by_type(self):
        """Test filtering events by type."""
        logger = SecurityLogger(name="test")

        logger.poison_detected("doc_1", 0.8)
        logger.document_blocked("doc_2", "blocked")
        logger.poison_detected("doc_3", 0.9)

        poison_events = logger.get_recent_events(
            event_type=SecurityEventType.POISON_DETECTED
        )

        assert len(poison_events) == 2


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment(self):
        """Test incrementing counters."""
        collector = MetricsCollector()

        collector.increment("requests")
        collector.increment("requests")
        collector.increment("requests", 5)

        assert collector.get_counter("requests") == 7

    def test_increment_with_labels(self):
        """Test incrementing with labels."""
        collector = MetricsCollector()

        collector.increment("requests", labels={"method": "GET"})
        collector.increment("requests", labels={"method": "POST"})
        collector.increment("requests", labels={"method": "GET"})

        assert collector.get_counter("requests", labels={"method": "GET"}) == 2
        assert collector.get_counter("requests", labels={"method": "POST"}) == 1

    def test_set_gauge(self):
        """Test setting gauges."""
        collector = MetricsCollector()

        collector.set_gauge("queue_size", 10)
        assert collector.get_gauge("queue_size") == 10

        collector.set_gauge("queue_size", 5)
        assert collector.get_gauge("queue_size") == 5

    def test_observe_histogram(self):
        """Test observing histogram values."""
        collector = MetricsCollector()

        for i in range(100):
            collector.observe("latency", i)

        stats = collector.get_histogram_stats("latency")

        assert stats["count"] == 100
        assert stats["min"] == 0
        assert stats["max"] == 99
        assert stats["avg"] == 49.5

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()

        collector.increment("counter1")
        collector.set_gauge("gauge1", 10)
        collector.observe("hist1", 5)

        metrics = collector.get_all_metrics()

        assert "uptime_seconds" in metrics
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        collector.increment("counter")
        collector.reset()

        assert collector.get_counter("counter") == 0


class TestSecurityMetrics:
    """Tests for SecurityMetrics."""

    def test_record_detection(self):
        """Test recording detection."""
        metrics = SecurityMetrics()

        metrics.record_detection(is_poisoned=True, confidence=0.8, latency_ms=10)
        metrics.record_detection(is_poisoned=False, confidence=0.2, latency_ms=5)

        summary = metrics.get_summary()

        assert summary["detection"]["total"] == 2
        assert summary["detection"]["poisoned"] == 1

    def test_record_ingestion(self):
        """Test recording ingestion."""
        metrics = SecurityMetrics()

        metrics.record_document_ingested(accepted=True)
        metrics.record_document_ingested(accepted=True)
        metrics.record_document_ingested(accepted=False)

        summary = metrics.get_summary()

        assert summary["ingestion"]["total"] == 3
        assert summary["ingestion"]["rejected"] == 1

    def test_record_query(self):
        """Test recording queries."""
        metrics = SecurityMetrics()

        metrics.record_query(latency_ms=10, results_count=5)
        metrics.record_query(latency_ms=20, results_count=3)

        summary = metrics.get_summary()

        assert summary["queries"]["total"] == 2

    def test_gauges(self):
        """Test gauge metrics."""
        metrics = SecurityMetrics()

        metrics.set_quarantine_size(10)
        metrics.set_knowledge_base_size(1000)

        summary = metrics.get_summary()

        assert summary["quarantine_size"] == 10
        assert summary["knowledge_base_size"] == 1000


class TestGlobalLogging:
    """Tests for global logging functions."""

    def test_get_logger(self):
        """Test getting default logger."""
        logger = get_logger()

        assert isinstance(logger, SecurityLogger)

    def test_get_metrics(self):
        """Test getting default metrics."""
        metrics = get_metrics()

        assert isinstance(metrics, SecurityMetrics)


# ============================================================================
# Framework Availability Tests
# ============================================================================

class TestFrameworkAvailability:
    """Tests for framework availability checks."""

    def test_langchain_check(self):
        """Test LangChain availability check."""
        # Should be a boolean
        assert isinstance(LANGCHAIN_AVAILABLE, bool)

    def test_llamaindex_check(self):
        """Test LlamaIndex availability check."""
        # Should be a boolean
        assert isinstance(LLAMAINDEX_AVAILABLE, bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config
        config = RAGShieldConfig(
            detection=DetectionConfig(preset="strict"),
            defense=DefenseSettings(level=DefenseLevel.PARANOID),
            privacy=PrivacyConfig(epsilon=0.5),
        )

        # Save to file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config.save(f.name)
            path = f.name

        # Create manager from file
        manager = create_config_manager(config_file=path, load_env=False)

        # Verify loaded correctly
        assert manager.get("detection.preset") == "strict"
        assert manager.get("defense.level") == DefenseLevel.PARANOID
        assert manager.get("privacy.epsilon") == 0.5

        # Override at runtime
        manager.set("detection.enabled", False)
        assert manager.get("detection.enabled") is False

        # Validate
        errors = manager.validate()
        assert len(errors) == 0


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        logger = SecurityLogger(name="integration_test")
        metrics = SecurityMetrics()

        # Log various events
        logger.poison_detected("doc_1", 0.85)
        metrics.record_detection(is_poisoned=True, confidence=0.85, latency_ms=10)

        logger.document_blocked("doc_2", "High threat")
        metrics.record_document_ingested(accepted=False)

        logger.query_processed("q_1", 15.5, 5)
        metrics.record_query(latency_ms=15.5, results_count=5)

        # Verify events
        events = logger.get_recent_events()
        assert len(events) == 3

        # Verify metrics
        summary = metrics.get_summary()
        assert summary["detection"]["total"] == 1
        assert summary["ingestion"]["rejected"] == 1
        assert summary["queries"]["total"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
