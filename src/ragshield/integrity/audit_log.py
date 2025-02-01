"""Verifiable Audit Log for RAG operations.

A tamper-evident log that records all operations on the knowledge base:
- Document additions/deletions
- Retrieval queries
- Integrity checks

Properties:
- Append-only: Entries cannot be removed or modified
- Chain integrity: Each entry links to the previous (hash chain)
- Verifiable: Any tampering can be detected
- Non-repudiable: Operations can be attributed and proven

Similar to blockchain structure but optimized for audit purposes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import json
import time


class OperationType(Enum):
    """Types of operations that can be logged."""
    DOCUMENT_ADD = "document_add"
    DOCUMENT_REMOVE = "document_remove"
    DOCUMENT_UPDATE = "document_update"
    QUERY = "query"
    RETRIEVAL = "retrieval"
    INTEGRITY_CHECK = "integrity_check"
    INTEGRITY_VIOLATION = "integrity_violation"
    POISON_DETECTED = "poison_detected"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEntry:
    """A single entry in the audit log.

    Attributes:
        sequence_number: Sequential index of this entry
        timestamp: Unix timestamp when entry was created
        operation_type: Type of operation being logged
        operation_data: Details of the operation
        previous_hash: Hash of the previous entry (chain link)
        entry_hash: Hash of this entry (computed on creation)
        signature: Optional digital signature for non-repudiation
    """
    sequence_number: int
    timestamp: float
    operation_type: OperationType
    operation_data: Dict[str, Any]
    previous_hash: str
    entry_hash: str = ""
    signature: Optional[str] = None

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this entry."""
        data = {
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "operation_type": self.operation_type.value,
            "operation_data": self.operation_data,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def verify_hash(self) -> bool:
        """Verify that the entry hash is correct."""
        return self.entry_hash == self._compute_hash()

    def to_dict(self) -> dict:
        """Convert entry to dictionary."""
        return {
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "operation_type": self.operation_type.value,
            "operation_data": self.operation_data,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        """Create entry from dictionary."""
        return cls(
            sequence_number=data["sequence_number"],
            timestamp=data["timestamp"],
            operation_type=OperationType(data["operation_type"]),
            operation_data=data["operation_data"],
            previous_hash=data["previous_hash"],
            entry_hash=data["entry_hash"],
            signature=data.get("signature"),
        )

    def __str__(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp)
        return f"[{self.sequence_number}] {dt.isoformat()} - {self.operation_type.value}"


class AuditLog:
    """Append-only audit log with hash chain integrity.

    Example:
        >>> log = AuditLog()
        >>> log.log_operation(OperationType.DOCUMENT_ADD, {"doc_id": "123"})
        >>> log.verify_chain()
        True
    """

    GENESIS_HASH = "0" * 64  # Genesis block previous hash

    def __init__(self):
        """Initialize empty audit log."""
        self.entries: List[AuditEntry] = []
        self._last_hash = self.GENESIS_HASH

    def log_operation(
        self,
        operation_type: OperationType,
        operation_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> AuditEntry:
        """Log an operation.

        Args:
            operation_type: Type of operation
            operation_data: Operation details
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            Created audit entry
        """
        if timestamp is None:
            timestamp = time.time()

        entry = AuditEntry(
            sequence_number=len(self.entries),
            timestamp=timestamp,
            operation_type=operation_type,
            operation_data=operation_data,
            previous_hash=self._last_hash,
        )

        self.entries.append(entry)
        self._last_hash = entry.entry_hash

        return entry

    def log_document_add(self, doc_id: str, content_hash: str, **kwargs) -> AuditEntry:
        """Log document addition."""
        return self.log_operation(
            OperationType.DOCUMENT_ADD,
            {"doc_id": doc_id, "content_hash": content_hash, **kwargs},
        )

    def log_document_remove(self, doc_id: str, reason: str = "") -> AuditEntry:
        """Log document removal."""
        return self.log_operation(
            OperationType.DOCUMENT_REMOVE,
            {"doc_id": doc_id, "reason": reason},
        )

    def log_query(self, query: str, query_hash: str, user_id: Optional[str] = None) -> AuditEntry:
        """Log a query operation."""
        return self.log_operation(
            OperationType.QUERY,
            {"query_hash": query_hash, "user_id": user_id},
        )

    def log_retrieval(
        self,
        query_hash: str,
        retrieved_doc_ids: List[str],
        scores: Optional[List[float]] = None,
    ) -> AuditEntry:
        """Log retrieval results."""
        return self.log_operation(
            OperationType.RETRIEVAL,
            {
                "query_hash": query_hash,
                "retrieved_doc_ids": retrieved_doc_ids,
                "scores": scores,
                "num_results": len(retrieved_doc_ids),
            },
        )

    def log_integrity_check(self, check_type: str, result: bool, details: str = "") -> AuditEntry:
        """Log integrity check."""
        return self.log_operation(
            OperationType.INTEGRITY_CHECK,
            {"check_type": check_type, "result": result, "details": details},
        )

    def log_integrity_violation(
        self,
        violation_type: str,
        affected_docs: List[str],
        severity: str = "high",
    ) -> AuditEntry:
        """Log integrity violation."""
        return self.log_operation(
            OperationType.INTEGRITY_VIOLATION,
            {
                "violation_type": violation_type,
                "affected_docs": affected_docs,
                "severity": severity,
            },
        )

    def log_poison_detected(
        self,
        doc_id: str,
        detection_method: str,
        confidence: float,
    ) -> AuditEntry:
        """Log poison detection."""
        return self.log_operation(
            OperationType.POISON_DETECTED,
            {
                "doc_id": doc_id,
                "detection_method": detection_method,
                "confidence": confidence,
            },
        )

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire log chain.

        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self.entries:
            return True

        # Check genesis entry
        if self.entries[0].previous_hash != self.GENESIS_HASH:
            return False

        # Verify each entry's hash
        for entry in self.entries:
            if not entry.verify_hash():
                return False

        # Verify chain links
        for i in range(1, len(self.entries)):
            if self.entries[i].previous_hash != self.entries[i - 1].entry_hash:
                return False

        return True

    def find_tampering(self) -> List[int]:
        """Find entries that may have been tampered with.

        Returns:
            List of sequence numbers of potentially tampered entries
        """
        tampered = []

        for entry in self.entries:
            if not entry.verify_hash():
                tampered.append(entry.sequence_number)

        # Check chain links
        for i in range(1, len(self.entries)):
            if self.entries[i].previous_hash != self.entries[i - 1].entry_hash:
                tampered.append(self.entries[i].sequence_number)

        return sorted(set(tampered))

    def get_entries_by_type(self, operation_type: OperationType) -> List[AuditEntry]:
        """Get all entries of a specific type.

        Args:
            operation_type: Type to filter by

        Returns:
            List of matching entries
        """
        return [e for e in self.entries if e.operation_type == operation_type]

    def get_entries_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> List[AuditEntry]:
        """Get entries within a time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            List of entries in range
        """
        return [
            e for e in self.entries
            if start_time <= e.timestamp <= end_time
        ]

    def get_entry(self, sequence_number: int) -> Optional[AuditEntry]:
        """Get entry by sequence number.

        Args:
            sequence_number: Entry index

        Returns:
            Entry if found, None otherwise
        """
        if 0 <= sequence_number < len(self.entries):
            return self.entries[sequence_number]
        return None

    def get_latest_hash(self) -> str:
        """Get hash of the latest entry."""
        return self._last_hash

    def get_stats(self) -> dict:
        """Get log statistics."""
        type_counts = {}
        for e in self.entries:
            t = e.operation_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_entries": len(self.entries),
            "operation_counts": type_counts,
            "chain_valid": self.verify_chain(),
            "first_timestamp": self.entries[0].timestamp if self.entries else None,
            "last_timestamp": self.entries[-1].timestamp if self.entries else None,
        }

    def export(self) -> List[dict]:
        """Export log to list of dictionaries."""
        return [e.to_dict() for e in self.entries]

    def export_json(self) -> str:
        """Export log to JSON string."""
        return json.dumps(self.export(), indent=2)

    @classmethod
    def from_export(cls, data: List[dict]) -> "AuditLog":
        """Import log from exported data.

        Args:
            data: List of entry dictionaries

        Returns:
            Reconstructed AuditLog
        """
        log = cls()
        for entry_dict in data:
            entry = AuditEntry.from_dict(entry_dict)
            log.entries.append(entry)
            log._last_hash = entry.entry_hash
        return log

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __str__(self) -> str:
        return f"AuditLog(entries={len(self.entries)}, valid={self.verify_chain()})"


class AuditLogVerifier:
    """Independent verifier for audit logs.

    Can verify logs without having the original AuditLog instance.
    """

    @staticmethod
    def verify_log(entries: List[dict]) -> Tuple[bool, List[str]]:
        """Verify a log from its serialized form.

        Args:
            entries: List of entry dictionaries

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not entries:
            return True, []

        # Check genesis
        if entries[0].get("previous_hash") != AuditLog.GENESIS_HASH:
            issues.append("Invalid genesis entry")

        # Verify each entry
        for i, entry_dict in enumerate(entries):
            entry = AuditEntry.from_dict(entry_dict)

            # Verify hash
            if not entry.verify_hash():
                issues.append(f"Entry {i}: Hash mismatch")

            # Verify chain link (except first entry)
            if i > 0:
                expected_prev = entries[i - 1]["entry_hash"]
                if entry.previous_hash != expected_prev:
                    issues.append(f"Entry {i}: Chain link broken")

        return len(issues) == 0, issues

    @staticmethod
    def verify_entry(entry_dict: dict, expected_prev_hash: str) -> Tuple[bool, str]:
        """Verify a single entry.

        Args:
            entry_dict: Entry dictionary
            expected_prev_hash: Expected previous hash

        Returns:
            Tuple of (is_valid, error_message)
        """
        entry = AuditEntry.from_dict(entry_dict)

        if not entry.verify_hash():
            return False, "Entry hash mismatch"

        if entry.previous_hash != expected_prev_hash:
            return False, "Chain link mismatch"

        return True, ""

    @staticmethod
    def generate_proof(entries: List[dict], index: int) -> dict:
        """Generate proof that an entry exists in the log.

        Args:
            entries: Full log entries
            index: Index of entry to prove

        Returns:
            Proof dictionary
        """
        if index < 0 or index >= len(entries):
            raise ValueError(f"Index {index} out of range")

        # Include entry and sufficient context for verification
        proof = {
            "entry": entries[index],
            "index": index,
            "prev_hash": entries[index - 1]["entry_hash"] if index > 0 else AuditLog.GENESIS_HASH,
            "next_hash": entries[index + 1]["previous_hash"] if index + 1 < len(entries) else None,
            "log_length": len(entries),
            "final_hash": entries[-1]["entry_hash"],
        }

        return proof

    @staticmethod
    def verify_proof(proof: dict) -> bool:
        """Verify an entry existence proof.

        Args:
            proof: Proof dictionary

        Returns:
            True if proof is valid
        """
        entry = AuditEntry.from_dict(proof["entry"])

        # Verify entry hash
        if not entry.verify_hash():
            return False

        # Verify chain link to previous
        if entry.previous_hash != proof["prev_hash"]:
            return False

        # Verify chain link to next (if exists)
        if proof["next_hash"] is not None:
            if proof["next_hash"] != entry.entry_hash:
                return False

        return True
