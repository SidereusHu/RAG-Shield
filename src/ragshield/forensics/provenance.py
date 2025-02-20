"""Attack provenance tracking for RAG systems.

Tracks document origin, chain of custody, and modification history
to enable forensic analysis of poisoning attacks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json

from ragshield.core.document import Document


class ProvenanceEventType(Enum):
    """Types of provenance events."""

    CREATED = "created"
    INGESTED = "ingested"
    MODIFIED = "modified"
    FLAGGED = "flagged"
    QUARANTINED = "quarantined"
    RELEASED = "released"
    DELETED = "deleted"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class ProvenanceEvent:
    """A single event in document provenance chain.

    Attributes:
        event_type: Type of provenance event
        timestamp: When the event occurred
        actor: Who/what triggered the event
        details: Additional event details
        content_hash: Hash of document content at this point
        previous_hash: Hash linking to previous event
    """

    event_type: ProvenanceEventType
    timestamp: datetime
    actor: str
    details: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    previous_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of this event for chain integrity."""
        data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "details": self.details,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "details": self.details,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
            "event_hash": self.compute_hash(),
        }


@dataclass
class ProvenanceChain:
    """Chain of custody for a document.

    Maintains a tamper-evident log of all events in document lifecycle.
    """

    doc_id: str
    events: List[ProvenanceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(
        self,
        event_type: ProvenanceEventType,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
    ) -> ProvenanceEvent:
        """Add a new event to the chain.

        Args:
            event_type: Type of event
            actor: Who triggered the event
            details: Event details
            content_hash: Current document content hash

        Returns:
            The created event
        """
        previous_hash = self.events[-1].compute_hash() if self.events else None

        event = ProvenanceEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            actor=actor,
            details=details or {},
            content_hash=content_hash,
            previous_hash=previous_hash,
        )

        self.events.append(event)
        return event

    def verify_chain(self) -> bool:
        """Verify integrity of the provenance chain.

        Returns:
            True if chain is intact, False if tampered
        """
        for i, event in enumerate(self.events):
            if i == 0:
                if event.previous_hash is not None:
                    return False
            else:
                expected_hash = self.events[i - 1].compute_hash()
                if event.previous_hash != expected_hash:
                    return False
        return True

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of all events.

        Returns:
            List of event dictionaries in chronological order
        """
        return [event.to_dict() for event in self.events]

    def get_origin(self) -> Optional[ProvenanceEvent]:
        """Get the origin (creation) event.

        Returns:
            First event in chain or None
        """
        return self.events[0] if self.events else None

    def get_current_state(self) -> Optional[ProvenanceEvent]:
        """Get the most recent event.

        Returns:
            Last event in chain or None
        """
        return self.events[-1] if self.events else None


class ProvenanceTracker:
    """Tracks provenance for all documents in a knowledge base.

    Maintains chains of custody and provides forensic analysis capabilities.
    """

    def __init__(self, system_id: str = "rag-shield"):
        """Initialize provenance tracker.

        Args:
            system_id: Identifier for this system
        """
        self.system_id = system_id
        self.chains: Dict[str, ProvenanceChain] = {}
        self._event_index: Dict[str, List[str]] = {}  # event_type -> doc_ids

    def create_chain(
        self,
        document: Document,
        source: str,
        actor: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceChain:
        """Create a new provenance chain for a document.

        Args:
            document: Document to track
            source: Origin source of the document
            actor: Who added the document
            details: Additional details

        Returns:
            New provenance chain
        """
        chain = ProvenanceChain(
            doc_id=document.doc_id,
            metadata={"source": source, "original_hash": document.hash},
        )

        # Add creation event
        event_details = {"source": source}
        if details:
            event_details.update(details)

        chain.add_event(
            event_type=ProvenanceEventType.CREATED,
            actor=actor,
            details=event_details,
            content_hash=document.hash,
        )

        # Add ingestion event
        chain.add_event(
            event_type=ProvenanceEventType.INGESTED,
            actor=self.system_id,
            details={"system": self.system_id},
            content_hash=document.hash,
        )

        self.chains[document.doc_id] = chain
        self._index_event(ProvenanceEventType.CREATED, document.doc_id)
        self._index_event(ProvenanceEventType.INGESTED, document.doc_id)

        return chain

    def record_event(
        self,
        doc_id: str,
        event_type: ProvenanceEventType,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
    ) -> Optional[ProvenanceEvent]:
        """Record an event for a document.

        Args:
            doc_id: Document identifier
            event_type: Type of event
            actor: Who triggered the event
            details: Event details
            content_hash: Current content hash

        Returns:
            Created event or None if chain not found
        """
        chain = self.chains.get(doc_id)
        if not chain:
            return None

        event = chain.add_event(
            event_type=event_type,
            actor=actor,
            details=details,
            content_hash=content_hash,
        )

        self._index_event(event_type, doc_id)
        return event

    def flag_document(
        self,
        doc_id: str,
        reason: str,
        threat_level: str,
        detector: str,
        confidence: float,
    ) -> Optional[ProvenanceEvent]:
        """Record that a document was flagged as suspicious.

        Args:
            doc_id: Document identifier
            reason: Why it was flagged
            threat_level: Severity level
            detector: Which detector flagged it
            confidence: Detection confidence

        Returns:
            Created event or None
        """
        return self.record_event(
            doc_id=doc_id,
            event_type=ProvenanceEventType.FLAGGED,
            actor=detector,
            details={
                "reason": reason,
                "threat_level": threat_level,
                "confidence": confidence,
            },
        )

    def quarantine_document(
        self, doc_id: str, reason: str, actor: str = "system"
    ) -> Optional[ProvenanceEvent]:
        """Record that a document was quarantined.

        Args:
            doc_id: Document identifier
            reason: Why it was quarantined
            actor: Who initiated quarantine

        Returns:
            Created event or None
        """
        return self.record_event(
            doc_id=doc_id,
            event_type=ProvenanceEventType.QUARANTINED,
            actor=actor,
            details={"reason": reason},
        )

    def get_chain(self, doc_id: str) -> Optional[ProvenanceChain]:
        """Get provenance chain for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Provenance chain or None
        """
        return self.chains.get(doc_id)

    def get_documents_by_event(
        self, event_type: ProvenanceEventType
    ) -> List[str]:
        """Get all documents with a specific event type.

        Args:
            event_type: Event type to search for

        Returns:
            List of document IDs
        """
        return self._event_index.get(event_type.value, [])

    def get_flagged_documents(self) -> List[str]:
        """Get all flagged documents.

        Returns:
            List of document IDs
        """
        return self.get_documents_by_event(ProvenanceEventType.FLAGGED)

    def get_quarantined_documents(self) -> List[str]:
        """Get all quarantined documents.

        Returns:
            List of document IDs
        """
        return self.get_documents_by_event(ProvenanceEventType.QUARANTINED)

    def verify_all_chains(self) -> Dict[str, bool]:
        """Verify integrity of all provenance chains.

        Returns:
            Dictionary of doc_id -> verification result
        """
        results = {}
        for doc_id, chain in self.chains.items():
            results[doc_id] = chain.verify_chain()
        return results

    def get_documents_by_source(self, source: str) -> List[str]:
        """Get all documents from a specific source.

        Args:
            source: Source to filter by

        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc_id, chain in self.chains.items():
            if chain.metadata.get("source") == source:
                doc_ids.append(doc_id)
        return doc_ids

    def get_documents_by_actor(self, actor: str) -> List[str]:
        """Get all documents with events from a specific actor.

        Args:
            actor: Actor to filter by

        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc_id, chain in self.chains.items():
            for event in chain.events:
                if event.actor == actor:
                    doc_ids.append(doc_id)
                    break
        return doc_ids

    def export_chains(self) -> Dict[str, Any]:
        """Export all provenance chains.

        Returns:
            Dictionary of all chains
        """
        return {
            doc_id: {
                "metadata": chain.metadata,
                "events": chain.get_timeline(),
                "verified": chain.verify_chain(),
            }
            for doc_id, chain in self.chains.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance statistics.

        Returns:
            Statistics dictionary
        """
        event_counts = {}
        for event_type in ProvenanceEventType:
            event_counts[event_type.value] = len(
                self._event_index.get(event_type.value, [])
            )

        return {
            "total_documents": len(self.chains),
            "event_counts": event_counts,
            "all_chains_valid": all(self.verify_all_chains().values()),
        }

    def _index_event(self, event_type: ProvenanceEventType, doc_id: str) -> None:
        """Index an event for fast lookup.

        Args:
            event_type: Event type
            doc_id: Document ID
        """
        key = event_type.value
        if key not in self._event_index:
            self._event_index[key] = []
        if doc_id not in self._event_index[key]:
            self._event_index[key].append(doc_id)
