#!/usr/bin/env python3
"""
Cryptographic Integrity Demo

Demonstrates cryptographic integrity protection for RAG systems:
- Merkle Tree: Knowledge base integrity verification
- Vector Commitment: Embedding integrity protection
- Audit Log: Tamper-evident operation logging
- IntegrityGuard: Unified protection layer
"""

import numpy as np
from ragshield.core import Document, KnowledgeBase
from ragshield.integrity import (
    MerkleTree,
    MerkleProof,
    HashBasedCommitment,
    CommitmentStore,
    AuditLog,
    AuditLogVerifier,
    IntegrityGuard,
    IntegrityMonitor,
    create_integrity_guard,
)
from ragshield.integrity.merkle_tree import HashAlgorithm
from ragshield.integrity.audit_log import OperationType


def section(title: str):
    """Print section header."""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def main():
    print("=" * 70)
    print("RAG-Shield: Cryptographic Integrity Protection Demo")
    print("=" * 70)

    # =========================================================================
    # 1. Merkle Tree Demo
    # =========================================================================
    section("1. MERKLE TREE - Knowledge Base Integrity")
    print("  Merkle Trees enable O(log n) verification of document integrity")

    # Create sample documents
    documents = [
        "Paris is the capital of France.",
        "Machine learning uses statistical methods.",
        "The speed of light is 299,792 km/s.",
        "Python is a programming language.",
        "RAG systems retrieve relevant documents.",
    ]

    # Build Merkle tree
    tree = MerkleTree(algorithm=HashAlgorithm.SHA256)
    root_hash = tree.build(documents)

    print(f"\n  Built tree with {len(documents)} documents")
    print(f"  Root hash: {root_hash[:32]}...")
    print(f"  Tree stats: {tree.get_tree_stats()}")

    # Generate and verify proof
    print("\n  Generating proof for document 2...")
    proof = tree.generate_proof(2)
    print(f"  Proof size: {len(proof.siblings)} sibling hashes")

    is_valid = tree.verify_proof(documents[2], proof)
    print(f"  Verification result: {'VALID' if is_valid else 'INVALID'}")

    # Detect tampering
    print("\n  Simulating tampering...")
    tampered_docs = documents.copy()
    tampered_docs[1] = "TAMPERED: Malicious content injected!"

    tampered_indices = tree.detect_tampering(tampered_docs)
    print(f"  Tampered documents detected at indices: {tampered_indices}")

    # Verify tampered doc fails
    tampered_valid = tree.verify_proof(tampered_docs[1], tree.generate_proof(1))
    print(f"  Tampered doc verification: {'VALID' if tampered_valid else 'INVALID (expected)'}")

    # =========================================================================
    # 2. Vector Commitment Demo
    # =========================================================================
    section("2. VECTOR COMMITMENT - Embedding Integrity")
    print("  Vector commitments protect against embedding manipulation attacks")

    # Create embeddings
    embedding1 = np.random.randn(384)  # Typical embedding dimension
    embedding2 = np.random.randn(384)

    # Create commitment
    scheme = HashBasedCommitment()
    commitment = scheme.commit(embedding1)

    print(f"\n  Created commitment for 384-dim embedding")
    print(f"  Commitment value: {commitment.value[:32]}...")
    print(f"  Metadata: dimension={commitment.metadata['dimension']}")

    # Verify original
    is_valid = scheme.verify(embedding1, commitment)
    print(f"\n  Original embedding verification: {'VALID' if is_valid else 'INVALID'}")

    # Verify tampered (different embedding)
    is_valid = scheme.verify(embedding2, commitment)
    print(f"  Different embedding verification: {'VALID' if is_valid else 'INVALID (expected)'}")

    # Subtle tampering (change one value)
    tampered_embedding = embedding1.copy()
    tampered_embedding[100] += 0.001  # Tiny change
    is_valid = scheme.verify(tampered_embedding, commitment)
    print(f"  Subtly tampered (delta=0.001): {'VALID' if is_valid else 'INVALID (expected)'}")

    # Commitment Store demo
    print("\n  Using CommitmentStore for document-level protection...")
    store = CommitmentStore()

    doc_embeddings = {
        "doc_001": np.random.randn(384),
        "doc_002": np.random.randn(384),
        "doc_003": np.random.randn(384),
    }

    for doc_id, emb in doc_embeddings.items():
        store.store_commitment(doc_id, emb)

    print(f"  Stored commitments for {len(store.get_all_doc_ids())} documents")

    # Verify all
    for doc_id, emb in doc_embeddings.items():
        is_valid = store.verify_embedding(doc_id, emb)
        print(f"  {doc_id}: {'VALID' if is_valid else 'INVALID'}")

    # =========================================================================
    # 3. Audit Log Demo
    # =========================================================================
    section("3. AUDIT LOG - Tamper-Evident Logging")
    print("  Hash-chain audit log for non-repudiable operation tracking")

    log = AuditLog()

    # Log various operations
    log.log_document_add("doc_001", "hash_abc123", source="upload")
    log.log_document_add("doc_002", "hash_def456", source="api")
    log.log_query("What is machine learning?", "query_hash_1")
    log.log_retrieval("query_hash_1", ["doc_001", "doc_002"], [0.95, 0.87])
    log.log_integrity_check("merkle_verification", True, "All documents verified")

    print(f"\n  Logged {len(log)} operations")
    print(f"  Chain valid: {log.verify_chain()}")

    print("\n  Recent entries:")
    for entry in log.entries[-3:]:
        print(f"    [{entry.sequence_number}] {entry.operation_type.value}")

    # Demonstrate tamper detection
    print("\n  Simulating log tampering...")
    original_data = log.entries[1].operation_data["doc_id"]
    log.entries[1].operation_data["doc_id"] = "TAMPERED_ID"

    chain_valid = log.verify_chain()
    tampered_entries = log.find_tampering()
    print(f"  Chain valid after tampering: {chain_valid}")
    print(f"  Tampered entries: {tampered_entries}")

    # Restore for demo
    log.entries[1].operation_data["doc_id"] = original_data

    # Export and verify externally
    exported = log.export()
    is_valid, issues = AuditLogVerifier.verify_log(exported)
    print(f"\n  External verification: {'VALID' if is_valid else f'INVALID ({len(issues)} issues)'}")

    # =========================================================================
    # 4. IntegrityGuard Demo
    # =========================================================================
    section("4. INTEGRITY GUARD - Unified Protection")
    print("  High-level API combining all integrity components")

    # Create knowledge base
    kb = KnowledgeBase()
    for i, content in enumerate(documents):
        doc = Document(content=content)
        doc.embedding = np.random.randn(128).tolist()
        kb.add_document(doc)

    # Create and apply guard
    guard = IntegrityGuard()
    root_hash = guard.protect_knowledge_base(kb)

    print(f"\n  Protected knowledge base with {len(kb)} documents")
    print(f"  Merkle root: {root_hash[:32]}...")

    # Verify single document
    doc = kb.documents[0]
    result = guard.verify_document(doc.doc_id, doc)
    print(f"\n  Document verification:")
    print(f"    Status: {result.status.value}")
    print(f"    Message: {result.message}")

    # Verify full knowledge base
    is_valid, results = guard.verify_knowledge_base(kb)
    print(f"\n  Full KB verification: {'ALL VALID' if is_valid else 'ISSUES FOUND'}")

    # Simulate content tampering
    print("\n  Simulating content tampering...")
    original_content = kb.documents[2].content
    kb.documents[2].content = "INJECTED MALICIOUS CONTENT!"

    result = guard.verify_document(kb.documents[2].doc_id, kb.documents[2])
    print(f"  Tampered doc status: {result.status.value}")
    print(f"  Detection: {result.message}")

    # Restore
    kb.documents[2].content = original_content

    # Guard stats
    print(f"\n  Guard statistics: {guard.get_stats()}")

    # =========================================================================
    # 5. IntegrityMonitor Demo
    # =========================================================================
    section("5. INTEGRITY MONITOR - Continuous Monitoring")
    print("  Continuous monitoring for real-time tampering detection")

    monitor = IntegrityMonitor(guard)

    # Initial check
    is_valid = monitor.check(kb)
    print(f"\n  Initial check: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  Violations: {len(monitor.violations)}")

    # Introduce tampering
    kb.documents[0].content = "TAMPERED CONTENT"
    is_valid = monitor.check(kb)
    print(f"\n  After tampering: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  Violations detected: {len(monitor.violations)}")

    if monitor.violations:
        v = monitor.violations[-1]
        print(f"  Latest violation: {v.status.value} - {v.message}")

    # Monitor status
    status = monitor.get_status()
    print(f"\n  Monitor status:")
    print(f"    Total checks: {status['check_count']}")
    print(f"    Total violations: {status['total_violations']}")

    # =========================================================================
    # 6. Preset Configurations
    # =========================================================================
    section("6. CONFIGURATION PRESETS")
    print("  Different security configurations for various use cases")

    presets = ["full", "minimal", "audit_only"]
    for preset in presets:
        g = create_integrity_guard(preset=preset)
        print(f"\n  {preset.upper()} preset:")
        print(f"    Merkle: {g.enable_merkle}, Vector: {g.enable_vector_commit}, Audit: {g.enable_audit}")

    print("\n" + "=" * 70)
    print("Cryptographic Integrity Demo Completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Merkle Trees detect any document modification in O(log n)")
    print("  2. Vector Commitments prevent embedding manipulation attacks")
    print("  3. Audit Logs provide tamper-evident operation tracking")
    print("  4. IntegrityGuard combines all protections seamlessly")
    print("=" * 70)


if __name__ == "__main__":
    main()
