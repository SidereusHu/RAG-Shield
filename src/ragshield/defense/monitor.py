"""Real-time monitoring for RAG systems.

Provides anomaly detection, rate limiting, and alerting for
ingestion and retrieval operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import threading
import time

from ragshield.core.document import Document
from ragshield.detection.base import DetectionResult, ThreatLevel


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_DOCUMENT = "suspicious_document"
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_MATCH = "pattern_match"
    BULK_INGESTION = "bulk_ingestion"
    SOURCE_BLOCKED = "source_blocked"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class Alert:
    """Security alert.

    Attributes:
        alert_id: Unique alert ID
        alert_type: Type of alert
        severity: Alert severity
        message: Alert message
        timestamp: When alert was created
        source: Source of the alert
        related_docs: Related document IDs
        metadata: Additional alert data
        acknowledged: Whether alert has been acknowledged
    """

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    related_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class RateLimitRule:
    """Rate limiting rule.

    Attributes:
        name: Rule name
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        action: Action when exceeded (block, throttle, alert)
    """

    name: str
    max_requests: int
    window_seconds: int
    action: str = "block"


@dataclass
class MonitoringMetrics:
    """Monitoring metrics snapshot.

    Attributes:
        timestamp: Metrics timestamp
        ingestion_rate: Documents ingested per minute
        detection_rate: Detection rate percentage
        alert_count: Active alerts
        blocked_count: Blocked operations
        average_latency: Average processing latency
    """

    timestamp: datetime
    ingestion_rate: float
    detection_rate: float
    alert_count: int
    blocked_count: int
    average_latency: float


class RateLimiter:
    """Token bucket rate limiter.

    Limits request rates using sliding window algorithm.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests in window
            window_seconds: Window duration
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """Check if request is allowed.

        Args:
            key: Rate limit key (e.g., source ID)

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            if key not in self._requests:
                self._requests[key] = deque()

            # Remove old requests
            while self._requests[key] and self._requests[key][0] < window_start:
                self._requests[key].popleft()

            current_count = len(self._requests[key])

            if current_count >= self.max_requests:
                return False, 0

            self._requests[key].append(now)
            return True, self.max_requests - current_count - 1

    def get_usage(self, key: str) -> Tuple[int, int]:
        """Get current usage for a key.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (current_count, max_requests)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            if key not in self._requests:
                return 0, self.max_requests

            # Count valid requests
            count = sum(1 for t in self._requests[key] if t >= window_start)
            return count, self.max_requests

    def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Rate limit key
        """
        with self._lock:
            if key in self._requests:
                self._requests[key].clear()


class AnomalyDetector:
    """Detects anomalies in ingestion patterns.

    Uses statistical methods to identify unusual activity.
    """

    def __init__(
        self,
        baseline_window: int = 100,
        threshold_std: float = 3.0,
    ):
        """Initialize anomaly detector.

        Args:
            baseline_window: Number of samples for baseline
            threshold_std: Standard deviations for anomaly
        """
        self.baseline_window = baseline_window
        self.threshold_std = threshold_std
        self._history: deque = deque(maxlen=baseline_window)
        self._metrics: Dict[str, deque] = {}

    def record_event(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """Record a metric event and check for anomaly.

        Args:
            metric: Metric name
            value: Metric value

        Returns:
            Anomaly info if detected, None otherwise
        """
        if metric not in self._metrics:
            self._metrics[metric] = deque(maxlen=self.baseline_window)

        history = self._metrics[metric]
        history.append(value)

        if len(history) < 10:
            return None

        # Calculate statistics
        values = list(history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5

        if std == 0:
            return None

        # Check for anomaly
        z_score = (value - mean) / std

        if abs(z_score) > self.threshold_std:
            return {
                "metric": metric,
                "value": value,
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "direction": "high" if z_score > 0 else "low",
            }

        return None

    def get_baseline(self, metric: str) -> Dict[str, float]:
        """Get baseline statistics for a metric.

        Args:
            metric: Metric name

        Returns:
            Baseline statistics
        """
        if metric not in self._metrics or len(self._metrics[metric]) < 2:
            return {"mean": 0, "std": 0, "samples": 0}

        values = list(self._metrics[metric])
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)

        return {
            "mean": mean,
            "std": variance ** 0.5,
            "samples": len(values),
        }


class SecurityMonitor:
    """Main security monitoring component.

    Integrates rate limiting, anomaly detection, and alerting.
    """

    def __init__(
        self,
        ingestion_rate_limit: int = 100,
        ingestion_window: int = 60,
        alert_threshold: float = 0.3,
    ):
        """Initialize security monitor.

        Args:
            ingestion_rate_limit: Max documents per window
            ingestion_window: Rate limit window in seconds
            alert_threshold: Detection rate threshold for alert
        """
        self.alert_threshold = alert_threshold

        # Rate limiters
        self._ingestion_limiter = RateLimiter(ingestion_rate_limit, ingestion_window)
        self._source_limiters: Dict[str, RateLimiter] = {}

        # Anomaly detection
        self._anomaly_detector = AnomalyDetector()

        # Alerts
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._alert_counter = 0

        # Metrics
        self._ingestion_count = 0
        self._detection_count = 0
        self._blocked_count = 0
        self._latencies: deque = deque(maxlen=100)

        # Blocked sources
        self._blocked_sources: Dict[str, datetime] = {}

        self._lock = threading.Lock()

    def check_ingestion(
        self,
        document: Document,
        source: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check if document ingestion is allowed.

        Args:
            document: Document to ingest
            source: Source of document

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        with self._lock:
            # Check if source is blocked
            if self._is_source_blocked(source):
                return False, "Source is blocked"

            # Check global rate limit
            allowed, remaining = self._ingestion_limiter.is_allowed("global")
            if not allowed:
                self._blocked_count += 1
                self._create_alert(
                    AlertType.RATE_LIMIT_EXCEEDED,
                    AlertSeverity.WARNING,
                    "Global ingestion rate limit exceeded",
                    source,
                    [document.doc_id],
                )
                return False, "Rate limit exceeded"

            # Check source-specific limit
            if source not in self._source_limiters:
                self._source_limiters[source] = RateLimiter(20, 60)

            allowed, _ = self._source_limiters[source].is_allowed(source)
            if not allowed:
                self._blocked_count += 1
                self._create_alert(
                    AlertType.RATE_LIMIT_EXCEEDED,
                    AlertSeverity.HIGH,
                    f"Source {source} rate limit exceeded",
                    source,
                    [document.doc_id],
                )
                return False, "Source rate limit exceeded"

            # Check for bulk ingestion anomaly
            self._ingestion_count += 1
            anomaly = self._anomaly_detector.record_event(
                "ingestion_rate", self._ingestion_count
            )
            if anomaly and anomaly["direction"] == "high":
                self._create_alert(
                    AlertType.BULK_INGESTION,
                    AlertSeverity.WARNING,
                    f"Unusual bulk ingestion detected (z-score: {anomaly['z_score']:.2f})",
                    source,
                    [document.doc_id],
                )

            return True, None

    def record_detection(
        self,
        document: Document,
        result: DetectionResult,
        source: str,
    ) -> None:
        """Record a detection result.

        Args:
            document: Detected document
            result: Detection result
            source: Document source
        """
        with self._lock:
            if result.is_poisoned:
                self._detection_count += 1

                # Create alert based on threat level
                severity = self._threat_to_severity(result.threat_level)
                self._create_alert(
                    AlertType.SUSPICIOUS_DOCUMENT,
                    severity,
                    f"Suspicious document detected: {result.reason}",
                    source,
                    [document.doc_id],
                    {"confidence": result.confidence, "threat_level": result.threat_level.value},
                )

                # Check detection rate anomaly
                detection_rate = self._detection_count / max(self._ingestion_count, 1)
                if detection_rate > self.alert_threshold:
                    self._create_alert(
                        AlertType.THRESHOLD_EXCEEDED,
                        AlertSeverity.HIGH,
                        f"Detection rate {detection_rate:.1%} exceeds threshold",
                        source,
                    )

    def record_latency(self, latency_ms: float) -> None:
        """Record processing latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._latencies.append(latency_ms)

    def block_source(
        self, source: str, duration_hours: int = 24, reason: str = ""
    ) -> None:
        """Block a source from ingesting documents.

        Args:
            source: Source to block
            duration_hours: Block duration
            reason: Reason for blocking
        """
        with self._lock:
            expiry = datetime.now() + timedelta(hours=duration_hours)
            self._blocked_sources[source] = expiry

            self._create_alert(
                AlertType.SOURCE_BLOCKED,
                AlertSeverity.HIGH,
                f"Source {source} blocked for {duration_hours}h: {reason}",
                source,
            )

    def unblock_source(self, source: str) -> bool:
        """Unblock a source.

        Args:
            source: Source to unblock

        Returns:
            True if source was blocked
        """
        with self._lock:
            if source in self._blocked_sources:
                del self._blocked_sources[source]
                return True
            return False

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering.

        Args:
            severity: Filter by severity
            acknowledged: Filter by acknowledged status
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        alerts = self._alerts

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics.

        Returns:
            Metrics snapshot
        """
        with self._lock:
            avg_latency = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies
                else 0
            )

            detection_rate = (
                self._detection_count / self._ingestion_count
                if self._ingestion_count > 0
                else 0
            )

            unack_alerts = sum(1 for a in self._alerts if not a.acknowledged)

            return MonitoringMetrics(
                timestamp=datetime.now(),
                ingestion_rate=self._ingestion_count,
                detection_rate=detection_rate,
                alert_count=unack_alerts,
                blocked_count=self._blocked_count,
                average_latency=avg_latency,
            )

    def register_alert_callback(
        self, callback: Callable[[Alert], None]
    ) -> None:
        """Register callback for new alerts.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def _is_source_blocked(self, source: str) -> bool:
        """Check if source is blocked.

        Args:
            source: Source to check

        Returns:
            True if blocked
        """
        if source not in self._blocked_sources:
            return False

        expiry = self._blocked_sources[source]
        if datetime.now() > expiry:
            del self._blocked_sources[source]
            return False

        return True

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        source: str,
        related_docs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and store an alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            source: Alert source
            related_docs: Related document IDs
            metadata: Additional metadata

        Returns:
            Created alert
        """
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert-{self._alert_counter:06d}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            related_docs=related_docs or [],
            metadata=metadata or {},
        )

        self._alerts.append(alert)

        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass

        return alert

    def _threat_to_severity(self, threat_level: ThreatLevel) -> AlertSeverity:
        """Convert threat level to alert severity.

        Args:
            threat_level: Threat level

        Returns:
            Alert severity
        """
        mapping = {
            ThreatLevel.NONE: AlertSeverity.INFO,
            ThreatLevel.LOW: AlertSeverity.INFO,
            ThreatLevel.MEDIUM: AlertSeverity.WARNING,
            ThreatLevel.HIGH: AlertSeverity.HIGH,
            ThreatLevel.CRITICAL: AlertSeverity.CRITICAL,
        }
        return mapping.get(threat_level, AlertSeverity.WARNING)
