"""Cryptographic integrity protection components.

This module provides cryptographic primitives for ensuring knowledge base integrity:
- Merkle Tree: Efficient verification of document integrity
- Vector Commitment: Binding commitment to embedding vectors
- Audit Log: Tamper-evident logging of operations
"""

from ragshield.integrity.merkle_tree import (
    MerkleTree,
    MerkleProof,
    MerkleNode,
)
from ragshield.integrity.vector_commit import (
    VectorCommitment,
    HashBasedCommitment,
    Commitment,
    CommitmentStore,
)
from ragshield.integrity.audit_log import (
    AuditLog,
    AuditEntry,
    AuditLogVerifier,
)
from ragshield.integrity.guard import (
    IntegrityGuard,
    IntegrityMonitor,
    create_integrity_guard,
)

__all__ = [
    # Merkle Tree
    "MerkleTree",
    "MerkleProof",
    "MerkleNode",
    # Vector Commitment
    "VectorCommitment",
    "HashBasedCommitment",
    "Commitment",
    "CommitmentStore",
    # Audit Log
    "AuditLog",
    "AuditEntry",
    "AuditLogVerifier",
    # Guard
    "IntegrityGuard",
    "IntegrityMonitor",
    "create_integrity_guard",
]
