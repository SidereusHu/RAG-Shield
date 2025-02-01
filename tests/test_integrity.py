"""Tests for integrity protection components."""

import pytest
import numpy as np
import time
from ragshield.core import Document, KnowledgeBase
from ragshield.integrity import (
    MerkleTree,
    MerkleProof,
    MerkleNode,
    HashBasedCommitment,
    Commitment,
    CommitmentStore,
    AuditLog,
    AuditEntry,
    AuditLogVerifier,
    IntegrityGuard,
    IntegrityMonitor,
    create_integrity_guard,
)
from ragshield.integrity.merkle_tree import HashAlgorithm
from ragshield.integrity.audit_log import OperationType
from ragshield.integrity.guard import IntegrityStatus


class TestMerkleTree:
    """Tests for Merkle Tree implementation."""

    def test_tree_creation(self):
        """Test creating a Merkle tree."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3", "doc4"]
        root = tree.build(documents)

        assert root is not None
        assert len(root) == 64  # SHA256 hex
        assert len(tree) == 4

    def test_empty_tree(self):
        """Test that empty document list raises error."""
        tree = MerkleTree()
        with pytest.raises(ValueError):
            tree.build([])

    def test_single_document(self):
        """Test tree with single document."""
        tree = MerkleTree()
        root = tree.build(["single doc"])

        assert root is not None
        assert len(tree) == 1

    def test_proof_generation(self):
        """Test generating Merkle proof."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3", "doc4"]
        tree.build(documents)

        proof = tree.generate_proof(0)

        assert isinstance(proof, MerkleProof)
        assert proof.leaf_index == 0
        assert proof.root_hash == tree.get_root_hash()
        assert len(proof.siblings) > 0

    def test_proof_verification(self):
        """Test verifying a valid proof."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3", "doc4"]
        tree.build(documents)

        for i, doc in enumerate(documents):
            proof = tree.generate_proof(i)
            assert tree.verify_proof(doc, proof) is True

    def test_tampered_content_detection(self):
        """Test that tampered content fails verification."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3", "doc4"]
        tree.build(documents)

        proof = tree.generate_proof(0)

        # Try to verify with wrong content
        assert tree.verify_proof("tampered doc1", proof) is False

    def test_static_verification(self):
        """Test static proof verification."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3"]
        tree.build(documents)

        proof = tree.generate_proof(1)

        # Verify statically (without tree instance)
        assert MerkleTree.verify_proof_static("doc2", proof) is True
        assert MerkleTree.verify_proof_static("wrong", proof) is False

    def test_detect_tampering(self):
        """Test detecting tampered documents."""
        tree = MerkleTree()
        original_docs = ["doc1", "doc2", "doc3"]
        tree.build(original_docs)

        # Tamper with one document
        tampered_docs = ["doc1", "TAMPERED", "doc3"]
        tampered_indices = tree.detect_tampering(tampered_docs)

        assert 1 in tampered_indices
        assert 0 not in tampered_indices
        assert 2 not in tampered_indices

    def test_proof_serialization(self):
        """Test proof to/from JSON."""
        tree = MerkleTree()
        tree.build(["doc1", "doc2", "doc3"])

        proof = tree.generate_proof(1)
        json_str = proof.to_json()
        restored = MerkleProof.from_json(json_str)

        assert restored.leaf_hash == proof.leaf_hash
        assert restored.root_hash == proof.root_hash
        assert restored.siblings == proof.siblings

    def test_different_hash_algorithms(self):
        """Test with different hash algorithms."""
        documents = ["doc1", "doc2", "doc3"]

        for algo in [HashAlgorithm.SHA256, HashAlgorithm.SHA3_256, HashAlgorithm.BLAKE2B]:
            tree = MerkleTree(algorithm=algo)
            root = tree.build(documents)
            assert root is not None

            proof = tree.generate_proof(0)
            assert tree.verify_proof("doc1", proof) is True

    def test_odd_number_documents(self):
        """Test tree with odd number of documents."""
        tree = MerkleTree()
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        root = tree.build(documents)

        assert root is not None
        # All documents should be verifiable
        for i in range(5):
            proof = tree.generate_proof(i)
            assert tree.verify_proof(documents[i], proof) is True


class TestVectorCommitment:
    """Tests for Vector Commitment scheme."""

    def test_commitment_creation(self):
        """Test creating a commitment."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0, 4.0])

        commitment = scheme.commit(vector)

        assert isinstance(commitment, Commitment)
        assert commitment.value is not None
        assert commitment.metadata["dimension"] == 4

    def test_commitment_verification(self):
        """Test verifying a valid commitment."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0, 4.0])

        commitment = scheme.commit(vector)

        assert scheme.verify(vector, commitment) is True

    def test_tampered_vector_detection(self):
        """Test that tampered vectors fail verification."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0, 4.0])

        commitment = scheme.commit(vector)

        # Tamper with vector
        tampered = np.array([1.0, 2.0, 3.0, 5.0])  # Changed last value

        assert scheme.verify(tampered, commitment) is False

    def test_opening_proof(self):
        """Test opening a commitment at a position."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0, 4.0])

        commitment = scheme.commit(vector)
        proof = scheme.open(vector, position=2, commitment=commitment)

        assert proof.position == 2
        assert proof.value == 3.0
        assert scheme.verify_opening(proof, commitment) is True

    def test_wrong_dimension_fails(self):
        """Test that wrong dimension vector fails verification."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0])

        commitment = scheme.commit(vector)

        # Try with different dimension
        wrong_dim = np.array([1.0, 2.0, 3.0, 4.0])
        assert scheme.verify(wrong_dim, commitment) is False

    def test_commitment_store(self):
        """Test CommitmentStore functionality."""
        store = CommitmentStore()

        # Store commitments
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])

        store.store_commitment("doc1", embedding1)
        store.store_commitment("doc2", embedding2)

        # Verify
        assert store.verify_embedding("doc1", embedding1) is True
        assert store.verify_embedding("doc2", embedding2) is True
        assert store.verify_embedding("doc1", embedding2) is False

    def test_commitment_serialization(self):
        """Test commitment to/from JSON."""
        scheme = HashBasedCommitment()
        vector = np.array([1.0, 2.0, 3.0])

        commitment = scheme.commit(vector)
        json_str = commitment.to_json()
        restored = Commitment.from_json(json_str)

        assert restored.value == commitment.value
        assert restored.metadata == commitment.metadata


class TestAuditLog:
    """Tests for Audit Log."""

    def test_log_creation(self):
        """Test creating an audit log."""
        log = AuditLog()
        assert len(log) == 0
        assert log.verify_chain() is True

    def test_log_operation(self):
        """Test logging an operation."""
        log = AuditLog()

        entry = log.log_operation(
            OperationType.DOCUMENT_ADD,
            {"doc_id": "123", "content_hash": "abc"},
        )

        assert len(log) == 1
        assert entry.operation_type == OperationType.DOCUMENT_ADD
        assert entry.sequence_number == 0

    def test_chain_integrity(self):
        """Test that chain maintains integrity."""
        log = AuditLog()

        log.log_document_add("doc1", "hash1")
        log.log_document_add("doc2", "hash2")
        log.log_query("query", "queryhash")

        assert len(log) == 3
        assert log.verify_chain() is True

        # Each entry should link to previous
        assert log.entries[1].previous_hash == log.entries[0].entry_hash
        assert log.entries[2].previous_hash == log.entries[1].entry_hash

    def test_tampered_entry_detection(self):
        """Test detecting tampered entries."""
        log = AuditLog()

        log.log_document_add("doc1", "hash1")
        log.log_document_add("doc2", "hash2")

        # Tamper with an entry
        log.entries[0].operation_data["doc_id"] = "tampered"

        assert log.verify_chain() is False
        tampered = log.find_tampering()
        assert 0 in tampered

    def test_log_export_import(self):
        """Test exporting and importing log."""
        log = AuditLog()
        log.log_document_add("doc1", "hash1")
        log.log_integrity_check("test", True, "OK")

        # Export
        exported = log.export()

        # Import into new log
        restored = AuditLog.from_export(exported)

        assert len(restored) == 2
        assert restored.verify_chain() is True
        assert restored.entries[0].operation_data["doc_id"] == "doc1"

    def test_entry_types(self):
        """Test various log entry types."""
        log = AuditLog()

        log.log_document_add("doc1", "hash1")
        log.log_document_remove("doc1", "test removal")
        log.log_query("test query", "qhash")
        log.log_retrieval("qhash", ["doc1", "doc2"], [0.9, 0.8])
        log.log_integrity_check("merkle", True, "OK")
        log.log_integrity_violation("tampering", ["doc1"], "high")
        log.log_poison_detected("doc2", "semantic", 0.95)

        assert len(log) == 7
        assert log.verify_chain() is True

    def test_query_entries(self):
        """Test querying entries."""
        log = AuditLog()

        log.log_document_add("doc1", "hash1")
        log.log_document_add("doc2", "hash2")
        log.log_query("q1", "qh1")
        log.log_document_remove("doc1", "")

        # By type
        adds = log.get_entries_by_type(OperationType.DOCUMENT_ADD)
        assert len(adds) == 2

        queries = log.get_entries_by_type(OperationType.QUERY)
        assert len(queries) == 1

    def test_verifier_static(self):
        """Test static verification."""
        log = AuditLog()
        log.log_document_add("doc1", "hash1")
        log.log_document_add("doc2", "hash2")

        exported = log.export()

        is_valid, issues = AuditLogVerifier.verify_log(exported)
        assert is_valid is True
        assert len(issues) == 0

    def test_verifier_tampered(self):
        """Test verifier detects tampering."""
        log = AuditLog()
        log.log_document_add("doc1", "hash1")
        log.log_document_add("doc2", "hash2")

        exported = log.export()
        # Tamper
        exported[0]["operation_data"]["doc_id"] = "tampered"

        is_valid, issues = AuditLogVerifier.verify_log(exported)
        assert is_valid is False
        assert len(issues) > 0


class TestIntegrityGuard:
    """Tests for IntegrityGuard."""

    def test_guard_creation(self):
        """Test creating integrity guard."""
        guard = IntegrityGuard()
        assert guard.enable_merkle is True
        assert guard.enable_vector_commit is True
        assert guard.enable_audit is True

    def test_protect_knowledge_base(self):
        """Test protecting a knowledge base."""
        guard = IntegrityGuard()

        kb = KnowledgeBase()
        for i in range(5):
            doc = Document(content=f"Document {i}")
            doc.embedding = np.random.randn(10).tolist()
            kb.add_document(doc)

        root_hash = guard.protect_knowledge_base(kb)

        assert root_hash is not None
        assert len(root_hash) == 64

    def test_verify_document(self):
        """Test verifying a protected document."""
        guard = IntegrityGuard()

        kb = KnowledgeBase()
        doc = Document(content="Test document")
        doc.embedding = np.random.randn(10).tolist()
        kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        # Verify original
        result = guard.verify_document(doc.doc_id, doc)
        assert result.is_valid is True
        assert result.status == IntegrityStatus.VALID

    def test_detect_content_tampering(self):
        """Test detecting content tampering."""
        guard = IntegrityGuard()

        kb = KnowledgeBase()
        doc = Document(content="Original content")
        doc.embedding = np.random.randn(10).tolist()
        kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        # Tamper with content
        tampered_doc = Document(content="Tampered content")
        tampered_doc.doc_id = doc.doc_id
        tampered_doc.embedding = doc.embedding

        result = guard.verify_document(doc.doc_id, tampered_doc)
        assert result.is_valid is False
        assert result.status == IntegrityStatus.CONTENT_TAMPERED

    def test_detect_embedding_tampering(self):
        """Test detecting embedding tampering."""
        guard = IntegrityGuard()

        kb = KnowledgeBase()
        doc = Document(content="Test document")
        doc.embedding = np.random.randn(10).tolist()
        kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        # Keep content same but change embedding
        tampered_embedding = np.random.randn(10)

        result = guard.verify_document(
            doc.doc_id,
            doc,
            embedding=tampered_embedding,
        )
        assert result.is_valid is False
        assert result.status == IntegrityStatus.EMBEDDING_TAMPERED

    def test_verify_full_knowledge_base(self):
        """Test verifying entire knowledge base."""
        guard = IntegrityGuard()

        kb = KnowledgeBase()
        for i in range(5):
            doc = Document(content=f"Document {i}")
            doc.embedding = np.random.randn(10).tolist()
            kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        is_valid, results = guard.verify_knowledge_base(kb)
        assert is_valid is True
        assert all(r.is_valid for r in results)

    def test_create_guard_presets(self):
        """Test creating guard with presets."""
        full = create_integrity_guard(preset="full")
        assert full.enable_merkle is True
        assert full.enable_vector_commit is True
        assert full.enable_audit is True

        minimal = create_integrity_guard(preset="minimal")
        assert minimal.enable_merkle is True
        assert minimal.enable_vector_commit is False
        assert minimal.enable_audit is False

        audit_only = create_integrity_guard(preset="audit_only")
        assert audit_only.enable_merkle is False
        assert audit_only.enable_audit is True


class TestIntegrityMonitor:
    """Tests for IntegrityMonitor."""

    def test_monitor_check(self):
        """Test monitor check."""
        guard = IntegrityGuard()
        monitor = IntegrityMonitor(guard)

        kb = KnowledgeBase()
        doc = Document(content="Test")
        doc.embedding = np.random.randn(10).tolist()
        kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        is_valid = monitor.check(kb)
        assert is_valid is True
        assert monitor.check_count == 1

    def test_monitor_violations(self):
        """Test monitor tracks violations."""
        guard = IntegrityGuard()
        monitor = IntegrityMonitor(guard)

        kb = KnowledgeBase()
        doc = Document(content="Original")
        doc.embedding = np.random.randn(10).tolist()
        kb.add_document(doc)

        guard.protect_knowledge_base(kb)

        # Tamper
        kb.documents[0].content = "Tampered"

        is_valid = monitor.check(kb)
        assert is_valid is False
        assert len(monitor.violations) > 0
