"""Quarantine system for suspicious documents.

Provides isolation and staged review for potentially malicious documents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection.base import DetectionResult


class QuarantineStatus(Enum):
    """Status of quarantined document."""

    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class QuarantineAction(Enum):
    """Actions that can be taken on quarantined documents."""

    RELEASE = "release"  # Release to knowledge base
    DELETE = "delete"  # Permanently delete
    EXTEND = "extend"  # Extend quarantine period
    ESCALATE = "escalate"  # Escalate for higher review


@dataclass
class QuarantineEntry:
    """Entry for a quarantined document.

    Attributes:
        document: The quarantined document
        reason: Why it was quarantined
        detection_result: Detection result that triggered quarantine
        quarantine_time: When it was quarantined
        expiry_time: When quarantine expires
        status: Current status
        reviewer: Who is reviewing (if any)
        review_notes: Notes from reviewer
        metadata: Additional metadata
    """

    document: Document
    reason: str
    detection_result: Optional[DetectionResult]
    quarantine_time: datetime
    expiry_time: datetime
    status: QuarantineStatus = QuarantineStatus.PENDING_REVIEW
    reviewer: Optional[str] = None
    review_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuarantineDecision:
    """Decision made on a quarantined document.

    Attributes:
        action: Action to take
        reviewer: Who made the decision
        reason: Reason for decision
        timestamp: When decision was made
    """

    action: QuarantineAction
    reviewer: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class QuarantineManager:
    """Manages document quarantine.

    Provides isolation, review workflow, and release/deletion of documents.
    """

    def __init__(
        self,
        default_quarantine_days: int = 7,
        auto_expire_action: QuarantineAction = QuarantineAction.DELETE,
        max_quarantine_size: int = 1000,
    ):
        """Initialize quarantine manager.

        Args:
            default_quarantine_days: Default quarantine period
            auto_expire_action: Action when quarantine expires
            max_quarantine_size: Maximum documents in quarantine
        """
        self.default_quarantine_days = default_quarantine_days
        self.auto_expire_action = auto_expire_action
        self.max_quarantine_size = max_quarantine_size

        self._quarantine: Dict[str, QuarantineEntry] = {}
        self._decision_log: List[QuarantineDecision] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "quarantine": [],
            "release": [],
            "delete": [],
            "expire": [],
        }
        self._lock = threading.Lock()

    def quarantine(
        self,
        document: Document,
        reason: str,
        detection_result: Optional[DetectionResult] = None,
        quarantine_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantineEntry:
        """Quarantine a document.

        Args:
            document: Document to quarantine
            reason: Reason for quarantine
            detection_result: Optional detection result
            quarantine_days: Days to quarantine (None = default)
            metadata: Additional metadata

        Returns:
            Quarantine entry

        Raises:
            ValueError: If quarantine is full
        """
        with self._lock:
            if len(self._quarantine) >= self.max_quarantine_size:
                raise ValueError("Quarantine is full")

            days = quarantine_days or self.default_quarantine_days
            now = datetime.now()

            entry = QuarantineEntry(
                document=document,
                reason=reason,
                detection_result=detection_result,
                quarantine_time=now,
                expiry_time=now + timedelta(days=days),
                metadata=metadata or {},
            )

            self._quarantine[document.doc_id] = entry
            self._trigger_callbacks("quarantine", entry)

            return entry

    def release(
        self,
        doc_id: str,
        reviewer: str,
        reason: str,
        target_kb: Optional[KnowledgeBase] = None,
    ) -> Optional[Document]:
        """Release a document from quarantine.

        Args:
            doc_id: Document ID
            reviewer: Who approved release
            reason: Reason for release
            target_kb: Optional knowledge base to add document to

        Returns:
            Released document or None if not found
        """
        with self._lock:
            entry = self._quarantine.get(doc_id)
            if not entry:
                return None

            # Update status
            entry.status = QuarantineStatus.APPROVED
            entry.reviewer = reviewer
            entry.review_notes.append(f"Released by {reviewer}: {reason}")

            # Log decision
            decision = QuarantineDecision(
                action=QuarantineAction.RELEASE,
                reviewer=reviewer,
                reason=reason,
            )
            self._decision_log.append(decision)

            # Remove from quarantine
            document = entry.document
            del self._quarantine[doc_id]

            # Add to target knowledge base if provided
            if target_kb:
                target_kb.add_document(document)

            self._trigger_callbacks("release", entry)
            return document

    def reject(
        self, doc_id: str, reviewer: str, reason: str, delete: bool = True
    ) -> bool:
        """Reject and optionally delete a quarantined document.

        Args:
            doc_id: Document ID
            reviewer: Who rejected
            reason: Reason for rejection
            delete: Whether to delete the document

        Returns:
            True if successful
        """
        with self._lock:
            entry = self._quarantine.get(doc_id)
            if not entry:
                return False

            entry.status = QuarantineStatus.REJECTED
            entry.reviewer = reviewer
            entry.review_notes.append(f"Rejected by {reviewer}: {reason}")

            action = QuarantineAction.DELETE if delete else QuarantineAction.ESCALATE
            decision = QuarantineDecision(
                action=action,
                reviewer=reviewer,
                reason=reason,
            )
            self._decision_log.append(decision)

            if delete:
                del self._quarantine[doc_id]
                self._trigger_callbacks("delete", entry)

            return True

    def extend_quarantine(
        self, doc_id: str, additional_days: int, reviewer: str, reason: str
    ) -> bool:
        """Extend quarantine period for a document.

        Args:
            doc_id: Document ID
            additional_days: Days to add
            reviewer: Who extended
            reason: Reason for extension

        Returns:
            True if successful
        """
        with self._lock:
            entry = self._quarantine.get(doc_id)
            if not entry:
                return False

            entry.expiry_time += timedelta(days=additional_days)
            entry.review_notes.append(
                f"Extended {additional_days} days by {reviewer}: {reason}"
            )

            decision = QuarantineDecision(
                action=QuarantineAction.EXTEND,
                reviewer=reviewer,
                reason=reason,
            )
            self._decision_log.append(decision)

            return True

    def start_review(self, doc_id: str, reviewer: str) -> bool:
        """Mark document as under review.

        Args:
            doc_id: Document ID
            reviewer: Who is reviewing

        Returns:
            True if successful
        """
        with self._lock:
            entry = self._quarantine.get(doc_id)
            if not entry:
                return False

            if entry.status != QuarantineStatus.PENDING_REVIEW:
                return False

            entry.status = QuarantineStatus.UNDER_REVIEW
            entry.reviewer = reviewer
            entry.review_notes.append(f"Review started by {reviewer}")

            return True

    def get_entry(self, doc_id: str) -> Optional[QuarantineEntry]:
        """Get quarantine entry for a document.

        Args:
            doc_id: Document ID

        Returns:
            Quarantine entry or None
        """
        return self._quarantine.get(doc_id)

    def get_all_pending(self) -> List[QuarantineEntry]:
        """Get all documents pending review.

        Returns:
            List of pending entries
        """
        return [
            entry
            for entry in self._quarantine.values()
            if entry.status == QuarantineStatus.PENDING_REVIEW
        ]

    def get_expired(self) -> List[QuarantineEntry]:
        """Get all expired quarantine entries.

        Returns:
            List of expired entries
        """
        now = datetime.now()
        return [
            entry
            for entry in self._quarantine.values()
            if entry.expiry_time < now
        ]

    def process_expired(self) -> List[QuarantineEntry]:
        """Process all expired entries based on auto_expire_action.

        Returns:
            List of processed entries
        """
        expired = self.get_expired()

        for entry in expired:
            entry.status = QuarantineStatus.EXPIRED

            if self.auto_expire_action == QuarantineAction.DELETE:
                self.reject(
                    entry.document.doc_id,
                    reviewer="system",
                    reason="Quarantine expired",
                    delete=True,
                )
            elif self.auto_expire_action == QuarantineAction.RELEASE:
                self.release(
                    entry.document.doc_id,
                    reviewer="system",
                    reason="Quarantine expired - auto-release",
                )

            self._trigger_callbacks("expire", entry)

        return expired

    def get_statistics(self) -> Dict[str, Any]:
        """Get quarantine statistics.

        Returns:
            Statistics dictionary
        """
        status_counts = {}
        for status in QuarantineStatus:
            status_counts[status.value] = sum(
                1 for e in self._quarantine.values() if e.status == status
            )

        expired_count = len(self.get_expired())

        return {
            "total_quarantined": len(self._quarantine),
            "status_breakdown": status_counts,
            "expired_count": expired_count,
            "capacity_used": len(self._quarantine) / self.max_quarantine_size,
            "total_decisions": len(self._decision_log),
        }

    def register_callback(
        self, event: str, callback: Callable[[QuarantineEntry], None]
    ) -> None:
        """Register callback for quarantine events.

        Args:
            event: Event type (quarantine, release, delete, expire)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, entry: QuarantineEntry) -> None:
        """Trigger callbacks for an event.

        Args:
            event: Event type
            entry: Quarantine entry
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(entry)
            except Exception:
                pass  # Don't let callback errors break the flow

    def bulk_release(
        self,
        doc_ids: List[str],
        reviewer: str,
        reason: str,
        target_kb: Optional[KnowledgeBase] = None,
    ) -> List[Document]:
        """Release multiple documents from quarantine.

        Args:
            doc_ids: Document IDs to release
            reviewer: Who approved
            reason: Reason for release
            target_kb: Optional target knowledge base

        Returns:
            List of released documents
        """
        released = []
        for doc_id in doc_ids:
            doc = self.release(doc_id, reviewer, reason, target_kb)
            if doc:
                released.append(doc)
        return released

    def bulk_reject(
        self, doc_ids: List[str], reviewer: str, reason: str
    ) -> int:
        """Reject multiple documents.

        Args:
            doc_ids: Document IDs to reject
            reviewer: Who rejected
            reason: Reason for rejection

        Returns:
            Number of documents rejected
        """
        count = 0
        for doc_id in doc_ids:
            if self.reject(doc_id, reviewer, reason):
                count += 1
        return count
