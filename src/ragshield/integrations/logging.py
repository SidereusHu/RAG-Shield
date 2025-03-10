"""Logging and monitoring utilities for RAG-Shield.

Provides structured logging, metrics collection, and security
event tracking for production deployments.
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
from collections import defaultdict
import threading


class SecurityEventType(Enum):
    """Types of security events."""

    # Detection events
    POISON_DETECTED = "poison_detected"
    DOCUMENT_VERIFIED = "document_verified"
    DETECTION_ERROR = "detection_error"

    # Defense events
    DOCUMENT_BLOCKED = "document_blocked"
    DOCUMENT_QUARANTINED = "document_quarantined"
    DOCUMENT_RELEASED = "document_released"
    SOURCE_BLOCKED = "source_blocked"

    # Query events
    QUERY_PROCESSED = "query_processed"
    QUERY_FILTERED = "query_filtered"
    SUSPICIOUS_QUERY = "suspicious_query"

    # System events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    ALERT_TRIGGERED = "alert_triggered"

    # Audit events
    CONFIG_CHANGED = "config_changed"
    ACCESS_DENIED = "access_denied"


@dataclass
class SecurityEvent:
    """A security event for logging.

    Attributes:
        event_type: Type of security event
        timestamp: Event timestamp
        severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        message: Human-readable message
        details: Additional event details
        doc_id: Related document ID if applicable
        source: Event source/component
    """

    event_type: SecurityEventType
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "INFO"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    source: str = "ragshield"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "doc_id": self.doc_id,
            "source": self.source,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SecurityLogger:
    """Structured security event logger.

    Provides structured logging for security events with support
    for multiple handlers and formatters.
    """

    def __init__(
        self,
        name: str = "ragshield",
        level: str = "INFO",
        handlers: Optional[List[logging.Handler]] = None,
    ):
        """Initialize security logger.

        Args:
            name: Logger name
            level: Logging level
            handlers: Optional custom handlers
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Add default handler if none provided
        if not handlers and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._create_formatter())
            self.logger.addHandler(handler)
        elif handlers:
            for handler in handlers:
                self.logger.addHandler(handler)

        # Event callbacks
        self._event_callbacks: List[Callable[[SecurityEvent], None]] = []

        # Event history (limited)
        self._event_history: List[SecurityEvent] = []
        self._max_history = 1000
        self._lock = threading.Lock()

    def _create_formatter(self) -> logging.Formatter:
        """Create a structured log formatter."""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event.

        Args:
            event: Security event to log
        """
        # Log to Python logger
        log_func = getattr(self.logger, event.severity.lower(), self.logger.info)
        log_func(f"[{event.event_type.value}] {event.message}", extra={
            "event_data": event.to_dict()
        })

        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Call callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def poison_detected(
        self,
        doc_id: str,
        confidence: float,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a poison detection event."""
        event = SecurityEvent(
            event_type=SecurityEventType.POISON_DETECTED,
            severity="WARNING",
            message=f"Potential poison detected in document {doc_id} (confidence: {confidence:.2f})",
            doc_id=doc_id,
            details={"confidence": confidence, **(details or {})},
        )
        self.log_event(event)

    def document_blocked(
        self,
        doc_id: str,
        reason: str,
        source: Optional[str] = None,
    ) -> None:
        """Log a document blocked event."""
        event = SecurityEvent(
            event_type=SecurityEventType.DOCUMENT_BLOCKED,
            severity="WARNING",
            message=f"Document {doc_id} blocked: {reason}",
            doc_id=doc_id,
            details={"reason": reason, "source": source},
        )
        self.log_event(event)

    def document_quarantined(
        self,
        doc_id: str,
        threat_score: float,
    ) -> None:
        """Log a document quarantined event."""
        event = SecurityEvent(
            event_type=SecurityEventType.DOCUMENT_QUARANTINED,
            severity="WARNING",
            message=f"Document {doc_id} quarantined (threat score: {threat_score:.2f})",
            doc_id=doc_id,
            details={"threat_score": threat_score},
        )
        self.log_event(event)

    def query_processed(
        self,
        query_id: str,
        latency_ms: float,
        results_count: int,
    ) -> None:
        """Log a query processed event."""
        event = SecurityEvent(
            event_type=SecurityEventType.QUERY_PROCESSED,
            severity="INFO",
            message=f"Query {query_id} processed in {latency_ms:.1f}ms ({results_count} results)",
            details={
                "query_id": query_id,
                "latency_ms": latency_ms,
                "results_count": results_count,
            },
        )
        self.log_event(event)

    def alert_triggered(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log an alert triggered event."""
        event = SecurityEvent(
            event_type=SecurityEventType.ALERT_TRIGGERED,
            severity="ERROR",
            message=f"Alert: {alert_type} - {message}",
            details={"alert_type": alert_type, **(details or {})},
        )
        self.log_event(event)

    def add_callback(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add an event callback.

        Args:
            callback: Function to call on each event
        """
        self._event_callbacks.append(callback)

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[SecurityEventType] = None,
    ) -> List[SecurityEvent]:
        """Get recent events.

        Args:
            count: Number of events to return
            event_type: Optional filter by event type

        Returns:
            List of recent events
        """
        with self._lock:
            events = self._event_history[-count:]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events


@dataclass
class MetricValue:
    """A single metric value with metadata."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates security metrics.

    Provides counters, gauges, and histograms for monitoring
    RAG-Shield performance and security.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._start_time = time.time()

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict] = None) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Value to add
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Observe a value for a histogram.

        Args:
            name: Histogram name
            value: Observed value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Limit histogram size
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-10000:]

    def _make_key(self, name: str, labels: Optional[Dict] = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[Dict] = None) -> float:
        """Get a counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict] = None) -> Optional[float]:
        """Get a gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Optional[Dict] = None) -> Dict[str, float]:
        """Get histogram statistics.

        Returns:
            Dictionary with count, sum, avg, min, max, p50, p95, p99
        """
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(values),
            "avg": sum(values) / n,
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[int(n * 0.5)],
            "p95": sorted_values[min(int(n * 0.95), n - 1)],
            "p99": sorted_values[min(int(n * 0.99), n - 1)],
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            return {
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k)
                    for k in self._histograms.keys()
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()


class SecurityMetrics:
    """Pre-defined security metrics for RAG-Shield.

    Provides convenient methods for common security metrics.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize security metrics.

        Args:
            collector: Optional metrics collector
        """
        self.collector = collector or MetricsCollector()

    def record_detection(self, is_poisoned: bool, confidence: float, latency_ms: float) -> None:
        """Record a detection result."""
        self.collector.increment("detection_total")
        if is_poisoned:
            self.collector.increment("detection_poisoned")
        self.collector.observe("detection_confidence", confidence)
        self.collector.observe("detection_latency_ms", latency_ms)

    def record_document_ingested(self, accepted: bool) -> None:
        """Record a document ingestion."""
        self.collector.increment("ingestion_total")
        if accepted:
            self.collector.increment("ingestion_accepted")
        else:
            self.collector.increment("ingestion_rejected")

    def record_query(self, latency_ms: float, results_count: int) -> None:
        """Record a query."""
        self.collector.increment("query_total")
        self.collector.observe("query_latency_ms", latency_ms)
        self.collector.observe("query_results", results_count)

    def record_threat_blocked(self, threat_type: str) -> None:
        """Record a blocked threat."""
        self.collector.increment("threats_blocked_total")
        self.collector.increment("threats_blocked", labels={"type": threat_type})

    def set_quarantine_size(self, size: int) -> None:
        """Set the current quarantine size."""
        self.collector.set_gauge("quarantine_size", size)

    def set_knowledge_base_size(self, size: int) -> None:
        """Set the current knowledge base size."""
        self.collector.set_gauge("knowledge_base_size", size)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of security metrics."""
        metrics = self.collector.get_all_metrics()

        detection_total = metrics["counters"].get("detection_total", 0)
        detection_poisoned = metrics["counters"].get("detection_poisoned", 0)

        ingestion_total = metrics["counters"].get("ingestion_total", 0)
        ingestion_rejected = metrics["counters"].get("ingestion_rejected", 0)

        return {
            "uptime_seconds": metrics["uptime_seconds"],
            "detection": {
                "total": detection_total,
                "poisoned": detection_poisoned,
                "poison_rate": detection_poisoned / detection_total if detection_total > 0 else 0,
                "latency": metrics["histograms"].get("detection_latency_ms", {}),
            },
            "ingestion": {
                "total": ingestion_total,
                "rejected": ingestion_rejected,
                "rejection_rate": ingestion_rejected / ingestion_total if ingestion_total > 0 else 0,
            },
            "queries": {
                "total": metrics["counters"].get("query_total", 0),
                "latency": metrics["histograms"].get("query_latency_ms", {}),
            },
            "threats_blocked": metrics["counters"].get("threats_blocked_total", 0),
            "quarantine_size": metrics["gauges"].get("quarantine_size", 0),
            "knowledge_base_size": metrics["gauges"].get("knowledge_base_size", 0),
        }


# Global instances for convenience
_default_logger: Optional[SecurityLogger] = None
_default_metrics: Optional[SecurityMetrics] = None


def get_logger() -> SecurityLogger:
    """Get the default security logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = SecurityLogger()
    return _default_logger


def get_metrics() -> SecurityMetrics:
    """Get the default security metrics."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = SecurityMetrics()
    return _default_metrics


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> SecurityLogger:
    """Configure the default security logger.

    Args:
        level: Logging level
        json_output: Use JSON output format
        log_file: Optional log file path

    Returns:
        Configured SecurityLogger
    """
    global _default_logger

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    if json_output:
        console_handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)

    _default_logger = SecurityLogger(level=level, handlers=handlers)
    return _default_logger
