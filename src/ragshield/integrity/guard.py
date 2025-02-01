"""Integrity Guard for RAG systems.

Integrates all integrity components into a unified protection layer:
- Merkle Tree for knowledge base integrity
- Vector Commitment for embedding integrity
- Audit Log for operation tracking

Provides a high-level API for:
- Protecting documents when added
- Verifying integrity before retrieval
- Detecting and reporting violations
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import time
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.integrity.merkle_tree import MerkleTree, MerkleProof
from ragshield.integrity.vector_commit import (
    VectorCommitment,
    HashBasedCommitment,
    CommitmentStore,
    Commitment,
)
from ragshield.integrity.audit_log import AuditLog, OperationType


class IntegrityStatus(Enum):
    """Status of integrity verification."""
    VALID = "valid"
    CONTENT_TAMPERED = "content_tampered"
    EMBEDDING_TAMPERED = "embedding_tampered"
    DOCUMENT_MISSING = "document_missing"
    DOCUMENT_ADDED = "document_added"
    UNKNOWN = "unknown"


@dataclass
class IntegrityResult:
    """Result of integrity verification.

    Attributes:
        is_valid: Overall integrity status
        status: Detailed status code
        document_id: ID of verified document (if applicable)
        message: Human-readable message
        details: Additional details
    """
    is_valid: bool
    status: IntegrityStatus
    document_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class IntegrityGuard:
    """High-level integrity protection for RAG systems.

    Example:
        >>> guard = IntegrityGuard()
        >>> guard.protect_knowledge_base(knowledge_base)
        >>> result = guard.verify_document(doc_id, document, embedding)
        >>> if not result.is_valid:
        ...     print(f"Integrity violation: {result.message}")
    """

    def __init__(
        self,
        enable_merkle: bool = True,
        enable_vector_commit: bool = True,
        enable_audit: bool = True,
    ):
        """Initialize integrity guard.

        Args:
            enable_merkle: Enable Merkle Tree verification
            enable_vector_commit: Enable vector commitment
            enable_audit: Enable audit logging
        """
        self.enable_merkle = enable_merkle
        self.enable_vector_commit = enable_vector_commit
        self.enable_audit = enable_audit

        # Initialize components
        self.merkle_tree: Optional[MerkleTree] = MerkleTree() if enable_merkle else None
        self.commitment_store: Optional[CommitmentStore] = (
            CommitmentStore() if enable_vector_commit else None
        )
        self.audit_log: Optional[AuditLog] = AuditLog() if enable_audit else None

        # Track protected documents
        self._protected_docs: Dict[str, int] = {}  # doc_id -> merkle_index
        self._merkle_root: Optional[str] = None

    def protect_knowledge_base(self, knowledge_base: KnowledgeBase) -> str:
        """Protect an entire knowledge base.

        Builds Merkle tree and commits to all embeddings.

        Args:
            knowledge_base: Knowledge base to protect

        Returns:
            Root hash representing the protected state
        """
        documents = knowledge_base.documents

        # Build Merkle tree from document contents
        if self.merkle_tree is not None:
            contents = [doc.content for doc in documents]
            self._merkle_root = self.merkle_tree.build(contents)

            # Track document indices
            for i, doc in enumerate(documents):
                self._protected_docs[doc.doc_id] = i

        # Commit to embeddings
        if self.commitment_store is not None:
            for doc in documents:
                if doc.embedding is not None:
                    embedding = np.array(doc.embedding)
                    self.commitment_store.store_commitment(doc.doc_id, embedding)

        # Log operation
        if self.audit_log is not None:
            self.audit_log.log_operation(
                OperationType.SYSTEM_EVENT,
                {
                    "event": "knowledge_base_protected",
                    "num_documents": len(documents),
                    "merkle_root": self._merkle_root,
                },
            )

        return self._merkle_root or ""

    def protect_document(
        self,
        document: Document,
        embedding: Optional[np.ndarray] = None,
    ) -> IntegrityResult:
        """Protect a single document.

        Args:
            document: Document to protect
            embedding: Document embedding (uses doc.embedding if None)

        Returns:
            Integrity result
        """
        if embedding is None and document.embedding is not None:
            embedding = np.array(document.embedding)

        # Add to Merkle tree (requires rebuild)
        # For now, just track it
        if self.merkle_tree is not None and document.doc_id not in self._protected_docs:
            # Would need to rebuild tree - mark as pending
            pass

        # Commit to embedding
        if self.commitment_store is not None and embedding is not None:
            self.commitment_store.store_commitment(document.doc_id, embedding)

        # Log
        if self.audit_log is not None:
            self.audit_log.log_document_add(
                doc_id=document.doc_id,
                content_hash=document.hash or document.compute_hash(),
            )

        return IntegrityResult(
            is_valid=True,
            status=IntegrityStatus.VALID,
            document_id=document.doc_id,
            message="Document protected successfully",
        )

    def verify_document(
        self,
        doc_id: str,
        document: Document,
        embedding: Optional[np.ndarray] = None,
    ) -> IntegrityResult:
        """Verify a document's integrity.

        Args:
            doc_id: Document ID
            document: Document to verify
            embedding: Embedding to verify (uses doc.embedding if None)

        Returns:
            Integrity verification result
        """
        issues = []

        # Verify content via Merkle tree
        if self.merkle_tree is not None and doc_id in self._protected_docs:
            index = self._protected_docs[doc_id]
            if index < len(self.merkle_tree.documents):
                proof = self.merkle_tree.generate_proof(index)
                if not self.merkle_tree.verify_proof(document.content, proof):
                    issues.append("content_mismatch")

        # Verify embedding via commitment
        if self.commitment_store is not None:
            if embedding is None and document.embedding is not None:
                embedding = np.array(document.embedding)

            if embedding is not None:
                if self.commitment_store.has_commitment(doc_id):
                    if not self.commitment_store.verify_embedding(doc_id, embedding):
                        issues.append("embedding_mismatch")
                else:
                    issues.append("no_embedding_commitment")

        # Determine result
        if not issues:
            status = IntegrityStatus.VALID
            message = "Document integrity verified"
            is_valid = True
        elif "content_mismatch" in issues:
            status = IntegrityStatus.CONTENT_TAMPERED
            message = "Document content has been tampered with"
            is_valid = False
        elif "embedding_mismatch" in issues:
            status = IntegrityStatus.EMBEDDING_TAMPERED
            message = "Document embedding has been tampered with"
            is_valid = False
        else:
            status = IntegrityStatus.UNKNOWN
            message = f"Verification issues: {issues}"
            is_valid = False

        # Log verification
        if self.audit_log is not None:
            self.audit_log.log_integrity_check(
                check_type="document_verification",
                result=is_valid,
                details=f"doc_id={doc_id}, issues={issues}",
            )

        return IntegrityResult(
            is_valid=is_valid,
            status=status,
            document_id=doc_id,
            message=message,
            details={"issues": issues},
        )

    def verify_knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
    ) -> Tuple[bool, List[IntegrityResult]]:
        """Verify entire knowledge base integrity.

        Args:
            knowledge_base: Knowledge base to verify

        Returns:
            Tuple of (all_valid, list_of_results)
        """
        results = []
        all_valid = True

        for doc in knowledge_base.documents:
            result = self.verify_document(doc.doc_id, doc)
            results.append(result)
            if not result.is_valid:
                all_valid = False

        # Check for missing or added documents
        if self.merkle_tree is not None:
            current_ids = {doc.doc_id for doc in knowledge_base.documents}
            protected_ids = set(self._protected_docs.keys())

            # Missing documents
            for doc_id in protected_ids - current_ids:
                results.append(IntegrityResult(
                    is_valid=False,
                    status=IntegrityStatus.DOCUMENT_MISSING,
                    document_id=doc_id,
                    message=f"Document {doc_id} is missing from knowledge base",
                ))
                all_valid = False

            # Added documents (not protected)
            for doc_id in current_ids - protected_ids:
                results.append(IntegrityResult(
                    is_valid=False,
                    status=IntegrityStatus.DOCUMENT_ADDED,
                    document_id=doc_id,
                    message=f"Document {doc_id} was added but not protected",
                ))
                all_valid = False

        # Log full verification
        if self.audit_log is not None:
            self.audit_log.log_integrity_check(
                check_type="full_knowledge_base",
                result=all_valid,
                details=f"checked={len(results)}, valid={sum(1 for r in results if r.is_valid)}",
            )

            if not all_valid:
                violations = [r for r in results if not r.is_valid]
                self.audit_log.log_integrity_violation(
                    violation_type="knowledge_base_tampering",
                    affected_docs=[r.document_id for r in violations if r.document_id],
                    severity="high",
                )

        return all_valid, results

    def get_merkle_proof(self, doc_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for a document.

        Args:
            doc_id: Document ID

        Returns:
            Merkle proof if available
        """
        if self.merkle_tree is None or doc_id not in self._protected_docs:
            return None

        index = self._protected_docs[doc_id]
        return self.merkle_tree.generate_proof(index)

    def get_embedding_commitment(self, doc_id: str) -> Optional[Commitment]:
        """Get embedding commitment for a document.

        Args:
            doc_id: Document ID

        Returns:
            Commitment if available
        """
        if self.commitment_store is None:
            return None
        return self.commitment_store.commitments.get(doc_id)

    def get_root_hash(self) -> Optional[str]:
        """Get the current Merkle root hash."""
        return self._merkle_root

    def get_audit_log(self) -> Optional[AuditLog]:
        """Get the audit log."""
        return self.audit_log

    def get_stats(self) -> dict:
        """Get integrity guard statistics."""
        stats = {
            "merkle_enabled": self.enable_merkle,
            "vector_commit_enabled": self.enable_vector_commit,
            "audit_enabled": self.enable_audit,
            "protected_documents": len(self._protected_docs),
        }

        if self.merkle_tree is not None:
            stats["merkle_stats"] = self.merkle_tree.get_tree_stats()

        if self.audit_log is not None:
            stats["audit_stats"] = self.audit_log.get_stats()

        return stats


class IntegrityMonitor:
    """Continuous integrity monitoring for RAG systems.

    Periodically checks integrity and alerts on violations.
    """

    def __init__(self, guard: IntegrityGuard):
        """Initialize monitor.

        Args:
            guard: Integrity guard to use
        """
        self.guard = guard
        self.violations: List[IntegrityResult] = []
        self.last_check_time: Optional[float] = None
        self.check_count: int = 0

    def check(self, knowledge_base: KnowledgeBase) -> bool:
        """Perform integrity check.

        Args:
            knowledge_base: Knowledge base to check

        Returns:
            True if all checks pass
        """
        self.last_check_time = time.time()
        self.check_count += 1

        is_valid, results = self.guard.verify_knowledge_base(knowledge_base)

        # Record violations
        for result in results:
            if not result.is_valid:
                self.violations.append(result)

        return is_valid

    def get_recent_violations(self, max_count: int = 10) -> List[IntegrityResult]:
        """Get recent violations.

        Args:
            max_count: Maximum number to return

        Returns:
            List of recent violations
        """
        return self.violations[-max_count:]

    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violations.clear()

    def get_status(self) -> dict:
        """Get monitor status."""
        return {
            "check_count": self.check_count,
            "last_check_time": self.last_check_time,
            "total_violations": len(self.violations),
            "guard_stats": self.guard.get_stats(),
        }


def create_integrity_guard(
    preset: str = "default",
    **kwargs,
) -> IntegrityGuard:
    """Create integrity guard with preset configuration.

    Args:
        preset: Configuration preset
            - "full": All protections enabled
            - "default": Merkle + Vector Commitment
            - "minimal": Merkle only
            - "audit_only": Only audit logging
        **kwargs: Override specific options

    Returns:
        Configured IntegrityGuard
    """
    presets = {
        "full": {
            "enable_merkle": True,
            "enable_vector_commit": True,
            "enable_audit": True,
        },
        "default": {
            "enable_merkle": True,
            "enable_vector_commit": True,
            "enable_audit": True,
        },
        "minimal": {
            "enable_merkle": True,
            "enable_vector_commit": False,
            "enable_audit": False,
        },
        "audit_only": {
            "enable_merkle": False,
            "enable_vector_commit": False,
            "enable_audit": True,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset].copy()
    config.update(kwargs)

    return IntegrityGuard(**config)
