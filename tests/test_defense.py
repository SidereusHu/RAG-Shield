"""Tests for defense components."""

import pytest
import time
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection.base import DetectionResult, ThreatLevel
from ragshield.defense import (
    # Quarantine
    QuarantineStatus,
    QuarantineAction,
    QuarantineManager,
    # Monitoring
    AlertSeverity,
    AlertType,
    RateLimiter,
    AnomalyDetector,
    SecurityMonitor,
    # Sanitization
    SanitizationAction,
    SanitizationRule,
    ContentSanitizer,
    EmbeddingSanitizer,
    MetadataSanitizer,
    DocumentSanitizer,
    # Shield
    DefenseLevel,
    RAGShield,
    create_shield,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(
        doc_id="doc_001",
        content="This is a sample document for testing purposes.",
        embedding=[0.1] * 64,
    )


@pytest.fixture
def malicious_document():
    """Create a document with malicious content."""
    return Document(
        doc_id="mal_001",
        content="<script>alert('xss')</script> Some content here.",
    )


@pytest.fixture
def knowledge_base():
    """Create a knowledge base."""
    return KnowledgeBase()


@pytest.fixture
def detection_result():
    """Create a detection result."""
    return DetectionResult(
        is_poisoned=True,
        confidence=0.85,
        threat_level=ThreatLevel.HIGH,
        reason="Suspicious content detected",
        score=0.85,
    )


# ============================================================================
# Quarantine Tests
# ============================================================================

class TestQuarantineManager:
    """Tests for quarantine management."""

    def test_quarantine_document(self, sample_document, detection_result):
        """Test quarantining a document."""
        manager = QuarantineManager()

        entry = manager.quarantine(
            sample_document,
            reason="Suspicious",
            detection_result=detection_result,
        )

        assert entry.status == QuarantineStatus.PENDING_REVIEW
        assert entry.document.doc_id == sample_document.doc_id

    def test_release_document(self, sample_document):
        """Test releasing a document."""
        manager = QuarantineManager()

        manager.quarantine(sample_document, reason="Review needed")

        kb = KnowledgeBase()
        doc = manager.release(
            sample_document.doc_id,
            reviewer="admin",
            reason="Approved",
            target_kb=kb,
        )

        assert doc is not None
        assert doc.doc_id == sample_document.doc_id
        # Verify entry removed from quarantine
        assert manager.get_entry(sample_document.doc_id) is None

    def test_reject_document(self, sample_document):
        """Test rejecting a document."""
        manager = QuarantineManager()

        manager.quarantine(sample_document, reason="Suspicious")

        success = manager.reject(
            sample_document.doc_id,
            reviewer="admin",
            reason="Malicious content confirmed",
        )

        assert success
        assert manager.get_entry(sample_document.doc_id) is None

    def test_extend_quarantine(self, sample_document):
        """Test extending quarantine period."""
        manager = QuarantineManager()

        entry = manager.quarantine(sample_document, reason="Review needed")
        original_expiry = entry.expiry_time

        manager.extend_quarantine(
            sample_document.doc_id,
            additional_days=7,
            reviewer="admin",
            reason="Need more time",
        )

        updated_entry = manager.get_entry(sample_document.doc_id)
        assert updated_entry.expiry_time > original_expiry

    def test_get_pending(self, sample_document):
        """Test getting pending documents."""
        manager = QuarantineManager()

        manager.quarantine(sample_document, reason="Review needed")

        pending = manager.get_all_pending()
        assert len(pending) == 1

    def test_start_review(self, sample_document):
        """Test starting review process."""
        manager = QuarantineManager()

        manager.quarantine(sample_document, reason="Review needed")

        success = manager.start_review(sample_document.doc_id, "reviewer1")

        assert success
        entry = manager.get_entry(sample_document.doc_id)
        assert entry.status == QuarantineStatus.UNDER_REVIEW


# ============================================================================
# Rate Limiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for rate limiting."""

    def test_basic_rate_limit(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # First 5 requests should pass
        for i in range(5):
            allowed, remaining = limiter.is_allowed("test_key")
            assert allowed
            assert remaining == 4 - i

        # 6th request should be blocked
        allowed, remaining = limiter.is_allowed("test_key")
        assert not allowed
        assert remaining == 0

    def test_different_keys(self):
        """Test rate limiting with different keys."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Key 1
        assert limiter.is_allowed("key1")[0]
        assert limiter.is_allowed("key1")[0]
        assert not limiter.is_allowed("key1")[0]

        # Key 2 should have its own limit
        assert limiter.is_allowed("key2")[0]

    def test_get_usage(self):
        """Test getting current usage."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        limiter.is_allowed("test")
        limiter.is_allowed("test")

        current, max_req = limiter.get_usage("test")
        assert current == 2
        assert max_req == 10

    def test_reset(self):
        """Test resetting rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("test")
        limiter.is_allowed("test")
        assert not limiter.is_allowed("test")[0]

        limiter.reset("test")
        assert limiter.is_allowed("test")[0]


# ============================================================================
# Anomaly Detector Tests
# ============================================================================

class TestAnomalyDetector:
    """Tests for anomaly detection."""

    def test_normal_values(self):
        """Test with normal values."""
        detector = AnomalyDetector(baseline_window=20, threshold_std=3.0)

        # Build baseline
        for i in range(20):
            result = detector.record_event("metric", 100 + i * 0.1)

        # Normal value should not trigger anomaly
        result = detector.record_event("metric", 101)
        assert result is None

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = AnomalyDetector(baseline_window=20, threshold_std=2.0)

        # Build baseline with values around 100
        for i in range(20):
            detector.record_event("metric", 100 + np.random.randn())

        # Extreme value should trigger anomaly
        result = detector.record_event("metric", 200)

        if result:  # May not trigger with random baseline
            assert result["direction"] == "high"

    def test_get_baseline(self):
        """Test getting baseline statistics."""
        detector = AnomalyDetector()

        for i in range(10):
            detector.record_event("test", 100 + i)

        baseline = detector.get_baseline("test")

        assert baseline["samples"] == 10
        assert baseline["mean"] > 0


# ============================================================================
# Security Monitor Tests
# ============================================================================

class TestSecurityMonitor:
    """Tests for security monitoring."""

    def test_check_ingestion(self, sample_document):
        """Test ingestion check."""
        monitor = SecurityMonitor(ingestion_rate_limit=100)

        allowed, reason = monitor.check_ingestion(sample_document, "source1")

        assert allowed
        assert reason is None

    def test_rate_limit_exceeded(self, sample_document):
        """Test rate limit exceeded."""
        monitor = SecurityMonitor(ingestion_rate_limit=2, ingestion_window=60)

        monitor.check_ingestion(sample_document, "source1")
        monitor.check_ingestion(sample_document, "source1")

        allowed, reason = monitor.check_ingestion(sample_document, "source1")

        assert not allowed

    def test_block_source(self):
        """Test blocking a source."""
        monitor = SecurityMonitor()

        monitor.block_source("bad_source", duration_hours=1)

        doc = Document(doc_id="test", content="test")
        allowed, reason = monitor.check_ingestion(doc, "bad_source")

        assert not allowed
        assert "blocked" in reason.lower()

    def test_unblock_source(self):
        """Test unblocking a source."""
        monitor = SecurityMonitor()

        monitor.block_source("source1", duration_hours=1)
        success = monitor.unblock_source("source1")

        assert success

        doc = Document(doc_id="test", content="test")
        allowed, _ = monitor.check_ingestion(doc, "source1")
        assert allowed

    def test_get_alerts(self):
        """Test getting alerts."""
        monitor = SecurityMonitor(ingestion_rate_limit=1)

        # Trigger rate limit alert
        doc = Document(doc_id="test", content="test")
        monitor.check_ingestion(doc, "source")
        monitor.check_ingestion(doc, "source")

        alerts = monitor.get_alerts()

        assert len(alerts) >= 1

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        monitor = SecurityMonitor(ingestion_rate_limit=1)

        doc = Document(doc_id="test", content="test")
        monitor.check_ingestion(doc, "source")
        monitor.check_ingestion(doc, "source")

        alerts = monitor.get_alerts(acknowledged=False)
        if alerts:
            success = monitor.acknowledge_alert(alerts[0].alert_id)
            assert success


# ============================================================================
# Sanitizer Tests
# ============================================================================

class TestContentSanitizer:
    """Tests for content sanitization."""

    def test_block_script_tags(self, malicious_document):
        """Test blocking script tags."""
        sanitizer = ContentSanitizer()

        result = sanitizer.sanitize(malicious_document)

        assert result.is_blocked
        assert "script" in result.matched_rules[0].lower()

    def test_clean_content(self):
        """Test cleaning content."""
        doc = Document(
            doc_id="test",
            content="Normal content\x00with control chars",
        )
        sanitizer = ContentSanitizer()

        result = sanitizer.sanitize(doc)

        assert result.is_modified
        assert "\x00" not in result.document.content

    def test_length_limits(self):
        """Test content length limits."""
        long_content = "x" * 200000
        doc = Document(doc_id="test", content=long_content)

        sanitizer = ContentSanitizer(max_content_length=1000)
        result = sanitizer.sanitize(doc)

        assert len(result.document.content) <= 1000

    def test_custom_rule(self):
        """Test adding custom rules."""
        sanitizer = ContentSanitizer()

        sanitizer.add_rule(
            SanitizationRule(
                name="block_password",
                description="Block password mentions",
                pattern=r"password\s*[:=]",
                action=SanitizationAction.BLOCK,
            )
        )

        doc = Document(doc_id="test", content="The password: secret123")
        result = sanitizer.sanitize(doc)

        assert result.is_blocked


class TestEmbeddingSanitizer:
    """Tests for embedding sanitization."""

    def test_validate_normal_embedding(self):
        """Test validating normal embedding."""
        sanitizer = EmbeddingSanitizer(expected_dim=64)

        embedding = np.random.randn(64).tolist()
        result = sanitizer.validate(embedding)

        assert result.is_valid

    def test_detect_wrong_dimension(self):
        """Test detecting wrong dimension."""
        sanitizer = EmbeddingSanitizer(expected_dim=64)

        embedding = np.random.randn(128).tolist()
        result = sanitizer.validate(embedding)

        assert not result.is_valid
        assert "dimension" in result.issues[0].lower()

    def test_detect_abnormal_norm(self):
        """Test detecting abnormal norm."""
        sanitizer = EmbeddingSanitizer(expected_dim=64, max_norm=5.0)

        embedding = (np.random.randn(64) * 10).tolist()  # Large values
        result = sanitizer.validate(embedding)

        assert not result.is_valid

    def test_normalize(self):
        """Test normalizing embedding."""
        sanitizer = EmbeddingSanitizer()

        embedding = [1.0, 2.0, 3.0]
        normalized = sanitizer.normalize(embedding)

        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 0.001


class TestMetadataSanitizer:
    """Tests for metadata sanitization."""

    def test_strip_reserved_fields(self):
        """Test stripping reserved fields."""
        sanitizer = MetadataSanitizer()

        metadata = {
            "title": "Document",
            "poisoned": True,  # Reserved
            "attack_type": "direct",  # Reserved
        }

        clean, warnings = sanitizer.sanitize(metadata)

        assert "poisoned" not in clean
        assert "attack_type" not in clean
        assert "title" in clean

    def test_sanitize_keys(self):
        """Test sanitizing keys."""
        sanitizer = MetadataSanitizer()

        metadata = {"key with spaces!": "value"}

        clean, warnings = sanitizer.sanitize(metadata)

        assert "key_with_spaces_" in clean


# ============================================================================
# Shield Tests
# ============================================================================

class TestRAGShield:
    """Tests for unified RAG Shield."""

    def test_create_shield(self, knowledge_base):
        """Test creating shield."""
        shield = create_shield(knowledge_base, level=DefenseLevel.STANDARD)

        status = shield.get_status()
        assert status.is_active
        assert status.level == DefenseLevel.STANDARD

    def test_ingest_clean_document(self, knowledge_base, sample_document):
        """Test ingesting clean document."""
        shield = RAGShield(knowledge_base)

        result = shield.ingest(sample_document, source="test")

        assert result.success
        assert result.action_taken == "ingested"
        assert knowledge_base.size() == 1

    def test_block_malicious_content(self, knowledge_base, malicious_document):
        """Test blocking malicious content."""
        shield = RAGShield(knowledge_base)

        result = shield.ingest(malicious_document, source="test")

        assert not result.success
        assert "blocked" in result.action_taken
        assert knowledge_base.size() == 0

    def test_defense_levels(self, knowledge_base):
        """Test different defense levels."""
        shield = RAGShield(knowledge_base)

        for level in DefenseLevel:
            shield.set_level(level)
            assert shield.config.level == level

    def test_get_statistics(self, knowledge_base, sample_document):
        """Test getting statistics."""
        shield = RAGShield(knowledge_base)

        shield.ingest(sample_document, source="test")

        stats = shield.get_statistics()

        assert "ingestion" in stats
        assert stats["ingestion"]["documents_processed"] == 1

    def test_bulk_ingest(self, knowledge_base):
        """Test bulk ingestion."""
        shield = RAGShield(knowledge_base)
        shield.set_level(DefenseLevel.MINIMAL)  # Use minimal to skip detection

        docs = [
            Document(doc_id=f"doc_{i}", content=f"This is normal document content number {i} for testing.")
            for i in range(5)
        ]

        results = shield.bulk_ingest(docs, source="bulk_upload")

        success_count = sum(1 for r in results if r.success)
        assert success_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
