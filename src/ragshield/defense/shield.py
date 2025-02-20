"""Unified defense shield for RAG systems.

Combines all defense mechanisms into a single protection layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel
from ragshield.forensics.provenance import ProvenanceTracker
from ragshield.forensics.analyzer import AttackPatternAnalyzer
from ragshield.defense.quarantine import QuarantineManager, QuarantineEntry
from ragshield.defense.monitor import SecurityMonitor, Alert
from ragshield.defense.sanitizer import DocumentSanitizer


class DefenseLevel(Enum):
    """Defense level presets."""

    MINIMAL = "minimal"  # Basic sanitization only
    STANDARD = "standard"  # Sanitization + detection
    STRICT = "strict"  # Full protection with quarantine
    PARANOID = "paranoid"  # Maximum security, strict blocking


@dataclass
class DefenseConfig:
    """Defense configuration.

    Attributes:
        level: Defense level preset
        auto_quarantine: Automatically quarantine suspicious docs
        auto_block_sources: Automatically block malicious sources
        detection_threshold: Confidence threshold for detection
        quarantine_review_required: Require manual review for release
        enable_monitoring: Enable real-time monitoring
        enable_forensics: Enable forensic analysis
    """

    level: DefenseLevel = DefenseLevel.STANDARD
    auto_quarantine: bool = True
    auto_block_sources: bool = False
    detection_threshold: float = 0.7
    quarantine_review_required: bool = True
    enable_monitoring: bool = True
    enable_forensics: bool = True


@dataclass
class IngestionResult:
    """Result of protected document ingestion.

    Attributes:
        success: Whether document was ingested
        document: Processed document
        action_taken: Action taken (ingested, quarantined, blocked)
        detection_result: Detection result if checked
        sanitization_report: Sanitization report
        warnings: Warning messages
        metadata: Additional metadata
    """

    success: bool
    document: Optional[Document]
    action_taken: str
    detection_result: Optional[DetectionResult] = None
    sanitization_report: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShieldStatus:
    """Current shield status.

    Attributes:
        is_active: Whether shield is active
        level: Current defense level
        documents_processed: Total documents processed
        documents_blocked: Documents blocked
        documents_quarantined: Documents quarantined
        active_alerts: Number of active alerts
        sources_blocked: Number of blocked sources
    """

    is_active: bool
    level: DefenseLevel
    documents_processed: int
    documents_blocked: int
    documents_quarantined: int
    active_alerts: int
    sources_blocked: int


class RAGShield:
    """Unified defense shield for RAG systems.

    Integrates sanitization, detection, quarantine, monitoring,
    and forensics into a comprehensive protection layer.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        detector: Optional[PoisonDetector] = None,
        config: Optional[DefenseConfig] = None,
    ):
        """Initialize RAG Shield.

        Args:
            knowledge_base: Knowledge base to protect
            detector: Poison detector (None = create default)
            config: Defense configuration (None = use defaults)
        """
        self.knowledge_base = knowledge_base
        self.detector = detector
        self.config = config or DefenseConfig()

        # Initialize components
        self.sanitizer = DocumentSanitizer()
        self.quarantine = QuarantineManager()
        self.monitor = SecurityMonitor()
        self.provenance = ProvenanceTracker()
        self.analyzer = AttackPatternAnalyzer()

        # Statistics
        self._stats = {
            "documents_processed": 0,
            "documents_ingested": 0,
            "documents_blocked": 0,
            "documents_quarantined": 0,
        }

        # Callbacks
        self._ingestion_callbacks: List[Callable[[IngestionResult], None]] = []

        # Apply level-specific settings
        self._apply_level_settings()

    def ingest(
        self,
        document: Document,
        source: str = "unknown",
        skip_detection: bool = False,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest a document with full protection.

        Args:
            document: Document to ingest
            source: Source of the document
            skip_detection: Skip poison detection
            force: Force ingestion even if suspicious

        Returns:
            Ingestion result
        """
        self._stats["documents_processed"] += 1
        warnings = []
        action = "pending"

        # Step 1: Rate limiting check
        if self.config.enable_monitoring:
            allowed, reason = self.monitor.check_ingestion(document, source)
            if not allowed and not force:
                return IngestionResult(
                    success=False,
                    document=document,
                    action_taken="rate_limited",
                    warnings=[reason] if reason else [],
                )

        # Step 2: Sanitization
        sanitized_doc, san_report = self.sanitizer.sanitize(document)
        warnings.extend(san_report.get("warnings", []))

        if san_report.get("is_blocked") and not force:
            self._stats["documents_blocked"] += 1
            return IngestionResult(
                success=False,
                document=sanitized_doc,
                action_taken="blocked_sanitization",
                sanitization_report=san_report,
                warnings=warnings,
            )

        # Step 3: Detection
        detection_result = None
        if self.detector and not skip_detection:
            detection_result = self.detector.detect(sanitized_doc)

            if self.config.enable_monitoring:
                self.monitor.record_detection(sanitized_doc, detection_result, source)

            if detection_result.is_poisoned:
                if detection_result.confidence >= self.config.detection_threshold:
                    # Quarantine or block
                    if self.config.auto_quarantine:
                        self._quarantine_document(
                            sanitized_doc, detection_result, source
                        )
                        action = "quarantined"
                        self._stats["documents_quarantined"] += 1

                        # Optionally block source
                        if (
                            self.config.auto_block_sources
                            and detection_result.threat_level
                            in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                        ):
                            self.monitor.block_source(
                                source,
                                duration_hours=24,
                                reason=f"High-threat document: {detection_result.reason}",
                            )

                        if not force:
                            return IngestionResult(
                                success=False,
                                document=sanitized_doc,
                                action_taken=action,
                                detection_result=detection_result,
                                sanitization_report=san_report,
                                warnings=warnings,
                            )
                    elif not force:
                        self._stats["documents_blocked"] += 1
                        return IngestionResult(
                            success=False,
                            document=sanitized_doc,
                            action_taken="blocked_detection",
                            detection_result=detection_result,
                            sanitization_report=san_report,
                            warnings=warnings,
                        )
                else:
                    warnings.append(
                        f"Low-confidence detection: {detection_result.reason}"
                    )

        # Step 4: Forensic analysis (if enabled and detection triggered)
        if self.config.enable_forensics and detection_result and detection_result.is_poisoned:
            analysis = self.analyzer.analyze(sanitized_doc)
            self.analyzer.store_fingerprint(sanitized_doc.doc_id, analysis.fingerprint)

        # Step 5: Create provenance chain
        if self.config.enable_forensics:
            self.provenance.create_chain(sanitized_doc, source)

        # Step 6: Ingest into knowledge base
        self.knowledge_base.add_document(sanitized_doc)
        self._stats["documents_ingested"] += 1
        action = "ingested"

        result = IngestionResult(
            success=True,
            document=sanitized_doc,
            action_taken=action,
            detection_result=detection_result,
            sanitization_report=san_report,
            warnings=warnings,
        )

        # Trigger callbacks
        self._trigger_callbacks(result)

        return result

    def bulk_ingest(
        self,
        documents: List[Document],
        source: str = "unknown",
        skip_detection: bool = False,
        stop_on_block: bool = False,
    ) -> List[IngestionResult]:
        """Ingest multiple documents.

        Args:
            documents: Documents to ingest
            source: Source of documents
            skip_detection: Skip poison detection
            stop_on_block: Stop on first blocked document

        Returns:
            List of ingestion results
        """
        results = []

        for doc in documents:
            result = self.ingest(doc, source, skip_detection)
            results.append(result)

            if stop_on_block and not result.success:
                break

        return results

    def scan_knowledge_base(
        self, quarantine_threats: bool = True
    ) -> Dict[str, Any]:
        """Scan entire knowledge base for threats.

        Args:
            quarantine_threats: Quarantine detected threats

        Returns:
            Scan report
        """
        if not self.detector:
            return {"error": "No detector configured"}

        scan_result = self.detector.scan_knowledge_base(self.knowledge_base)

        if quarantine_threats:
            for doc, result in scan_result.poisoned_docs:
                self._quarantine_document(doc, result, "kb_scan")
                # Remove from knowledge base
                self.knowledge_base.remove_document(doc.doc_id)

        return {
            "total_scanned": scan_result.total_documents,
            "threats_found": len(scan_result.poisoned_docs),
            "clean_docs": scan_result.clean_docs,
            "detection_rate": scan_result.detection_rate,
            "quarantined": len(scan_result.poisoned_docs) if quarantine_threats else 0,
        }

    def get_status(self) -> ShieldStatus:
        """Get current shield status.

        Returns:
            Shield status
        """
        alerts = self.monitor.get_alerts(acknowledged=False)

        return ShieldStatus(
            is_active=True,
            level=self.config.level,
            documents_processed=self._stats["documents_processed"],
            documents_blocked=self._stats["documents_blocked"],
            documents_quarantined=self._stats["documents_quarantined"],
            active_alerts=len(alerts),
            sources_blocked=len(self.monitor._blocked_sources),
        )

    def get_alerts(
        self, limit: int = 100, unacknowledged_only: bool = True
    ) -> List[Alert]:
        """Get current alerts.

        Args:
            limit: Maximum alerts to return
            unacknowledged_only: Only return unacknowledged alerts

        Returns:
            List of alerts
        """
        acknowledged = False if unacknowledged_only else None
        return self.monitor.get_alerts(acknowledged=acknowledged, limit=limit)

    def get_quarantine_queue(self) -> List[QuarantineEntry]:
        """Get documents pending quarantine review.

        Returns:
            List of quarantine entries
        """
        return self.quarantine.get_all_pending()

    def review_quarantine(
        self,
        doc_id: str,
        action: str,
        reviewer: str,
        reason: str = "",
    ) -> bool:
        """Review a quarantined document.

        Args:
            doc_id: Document ID
            action: Action (release, reject)
            reviewer: Who is reviewing
            reason: Reason for decision

        Returns:
            True if action was taken
        """
        if action == "release":
            doc = self.quarantine.release(
                doc_id, reviewer, reason, self.knowledge_base
            )
            if doc and self.config.enable_forensics:
                self.provenance.record_event(
                    doc_id,
                    "released",
                    reviewer,
                    {"reason": reason},
                )
            return doc is not None

        elif action == "reject":
            success = self.quarantine.reject(doc_id, reviewer, reason)
            if success and self.config.enable_forensics:
                self.provenance.record_event(
                    doc_id,
                    "deleted",
                    reviewer,
                    {"reason": reason},
                )
            return success

        return False

    def set_level(self, level: DefenseLevel) -> None:
        """Change defense level.

        Args:
            level: New defense level
        """
        self.config.level = level
        self._apply_level_settings()

    def register_callback(
        self, callback: Callable[[IngestionResult], None]
    ) -> None:
        """Register callback for ingestion events.

        Args:
            callback: Callback function
        """
        self._ingestion_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Statistics dictionary
        """
        monitor_metrics = self.monitor.get_metrics()
        quarantine_stats = self.quarantine.get_statistics()
        provenance_stats = self.provenance.get_statistics()

        return {
            "ingestion": self._stats.copy(),
            "monitoring": {
                "ingestion_rate": monitor_metrics.ingestion_rate,
                "detection_rate": monitor_metrics.detection_rate,
                "alert_count": monitor_metrics.alert_count,
                "average_latency": monitor_metrics.average_latency,
            },
            "quarantine": quarantine_stats,
            "forensics": provenance_stats,
        }

    def _quarantine_document(
        self,
        document: Document,
        detection_result: DetectionResult,
        source: str,
    ) -> QuarantineEntry:
        """Quarantine a suspicious document.

        Args:
            document: Document to quarantine
            detection_result: Detection result
            source: Document source

        Returns:
            Quarantine entry
        """
        entry = self.quarantine.quarantine(
            document=document,
            reason=detection_result.reason,
            detection_result=detection_result,
            metadata={"source": source, "threat_level": detection_result.threat_level.value},
        )

        if self.config.enable_forensics:
            self.provenance.flag_document(
                document.doc_id,
                reason=detection_result.reason,
                threat_level=detection_result.threat_level.value,
                detector="shield",
                confidence=detection_result.confidence,
            )
            self.provenance.quarantine_document(
                document.doc_id,
                reason=detection_result.reason,
            )

        return entry

    def _apply_level_settings(self) -> None:
        """Apply settings based on defense level."""
        level_settings = {
            DefenseLevel.MINIMAL: {
                "auto_quarantine": False,
                "auto_block_sources": False,
                "detection_threshold": 0.9,
                "enable_monitoring": False,
                "enable_forensics": False,
            },
            DefenseLevel.STANDARD: {
                "auto_quarantine": True,
                "auto_block_sources": False,
                "detection_threshold": 0.7,
                "enable_monitoring": True,
                "enable_forensics": True,
            },
            DefenseLevel.STRICT: {
                "auto_quarantine": True,
                "auto_block_sources": True,
                "detection_threshold": 0.5,
                "enable_monitoring": True,
                "enable_forensics": True,
            },
            DefenseLevel.PARANOID: {
                "auto_quarantine": True,
                "auto_block_sources": True,
                "detection_threshold": 0.3,
                "enable_monitoring": True,
                "enable_forensics": True,
            },
        }

        settings = level_settings.get(self.config.level, {})
        for key, value in settings.items():
            setattr(self.config, key, value)

    def _trigger_callbacks(self, result: IngestionResult) -> None:
        """Trigger ingestion callbacks.

        Args:
            result: Ingestion result
        """
        for callback in self._ingestion_callbacks:
            try:
                callback(result)
            except Exception:
                pass


def create_shield(
    knowledge_base: KnowledgeBase,
    level: DefenseLevel = DefenseLevel.STANDARD,
    detector: Optional[PoisonDetector] = None,
) -> RAGShield:
    """Create a RAG Shield with specified level.

    Args:
        knowledge_base: Knowledge base to protect
        level: Defense level
        detector: Optional poison detector

    Returns:
        Configured RAG Shield
    """
    config = DefenseConfig(level=level)
    return RAGShield(knowledge_base, detector, config)
