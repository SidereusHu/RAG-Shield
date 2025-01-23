"""Tests for core RAG system components."""

import pytest
import numpy as np
from ragshield.core import (
    Document,
    DocumentMetadata,
    KnowledgeBase,
    RAGSystem,
)
from ragshield.core.embedder import MockEmbedder
from ragshield.core.retriever import SimpleRetriever


class TestDocument:
    """Tests for Document class."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(content="Test content")
        assert doc.content == "Test content"
        assert doc.doc_id is not None
        assert doc.hash is not None

    def test_document_hash(self):
        """Test document hash computation."""
        doc = Document(content="Test content")
        expected_hash = doc.compute_hash()
        assert doc.hash == expected_hash

    def test_document_integrity_verification(self):
        """Test document integrity verification."""
        doc = Document(content="Test content")
        assert doc.verify_integrity() is True

        # Tamper with content
        doc.content = "Modified content"
        assert doc.verify_integrity() is False

    def test_document_with_metadata(self):
        """Test document with metadata."""
        metadata = DocumentMetadata(
            source="test_source",
            author="test_author",
            tags=["tag1", "tag2"],
        )
        doc = Document(content="Test content", metadata=metadata)
        assert doc.metadata.source == "test_source"
        assert doc.metadata.author == "test_author"
        assert len(doc.metadata.tags) == 2

    def test_document_serialization(self):
        """Test document to_dict and from_dict."""
        doc = Document(content="Test content")
        doc_dict = doc.to_dict()

        restored = Document.from_dict(doc_dict)
        assert restored.content == doc.content
        assert restored.doc_id == doc.doc_id
        assert restored.hash == doc.hash

    def test_document_str_representation(self):
        """Test document string representation."""
        doc = Document(content="Short content")
        str_repr = str(doc)
        assert "Short content" in str_repr

        # Long content should be truncated
        long_doc = Document(content="A" * 100)
        str_repr = str(long_doc)
        assert "..." in str_repr


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""

    def test_empty_knowledge_base(self):
        """Test empty knowledge base."""
        kb = KnowledgeBase()
        assert kb.size() == 0
        assert len(kb) == 0

    def test_add_document(self):
        """Test adding document to knowledge base."""
        kb = KnowledgeBase()
        doc = Document(content="Test content")
        kb.add_document(doc)
        assert kb.size() == 1

    def test_add_duplicate_document(self):
        """Test adding duplicate document raises error."""
        kb = KnowledgeBase()
        doc = Document(content="Test content")
        kb.add_document(doc)

        with pytest.raises(ValueError):
            kb.add_document(doc)

    def test_get_document(self):
        """Test getting document by ID."""
        kb = KnowledgeBase()
        doc = Document(content="Test content")
        kb.add_document(doc)

        retrieved = kb.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.content == doc.content

    def test_remove_document(self):
        """Test removing document."""
        kb = KnowledgeBase()
        doc = Document(content="Test content")
        kb.add_document(doc)

        assert kb.remove_document(doc.doc_id) is True
        assert kb.size() == 0
        assert kb.get_document(doc.doc_id) is None

    def test_remove_nonexistent_document(self):
        """Test removing nonexistent document."""
        kb = KnowledgeBase()
        assert kb.remove_document("nonexistent") is False

    def test_clear(self):
        """Test clearing knowledge base."""
        kb = KnowledgeBase()
        for i in range(5):
            kb.add_document(Document(content=f"Content {i}"))
        assert kb.size() == 5

        kb.clear()
        assert kb.size() == 0

    def test_iteration(self):
        """Test iterating over knowledge base."""
        kb = KnowledgeBase()
        contents = ["Content 1", "Content 2", "Content 3"]
        for content in contents:
            kb.add_document(Document(content=content))

        retrieved_contents = [doc.content for doc in kb]
        assert retrieved_contents == contents


class TestEmbedder:
    """Tests for Embedder classes."""

    def test_mock_embedder_single(self):
        """Test mock embedder with single text."""
        embedder = MockEmbedder(embedding_dim=384)
        embedding = embedder.embed("Test text")
        assert embedding.shape == (1, 384)

    def test_mock_embedder_batch(self):
        """Test mock embedder with batch of texts."""
        embedder = MockEmbedder(embedding_dim=384)
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape == (3, 384)

    def test_mock_embedder_consistency(self):
        """Test mock embedder produces consistent embeddings."""
        embedder = MockEmbedder(embedding_dim=384)
        embedding1 = embedder.embed("Test text")
        embedding2 = embedder.embed("Test text")
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_mock_embedder_different_texts(self):
        """Test mock embedder produces different embeddings for different texts."""
        embedder = MockEmbedder(embedding_dim=384)
        embedding1 = embedder.embed("Text 1")
        embedding2 = embedder.embed("Text 2")
        assert not np.allclose(embedding1, embedding2)


class TestRetriever:
    """Tests for Retriever classes."""

    def test_simple_retriever_index(self):
        """Test indexing with simple retriever."""
        kb = KnowledgeBase()
        embedder = MockEmbedder(embedding_dim=64)

        # Add documents with embeddings
        for i in range(5):
            doc = Document(content=f"Document {i}")
            doc.embedding = embedder.embed(f"Document {i}")[0].tolist()
            kb.add_document(doc)

        retriever = SimpleRetriever()
        retriever.index(kb)
        assert retriever.embeddings is not None

    def test_simple_retriever_retrieve(self):
        """Test retrieval with simple retriever."""
        kb = KnowledgeBase()
        embedder = MockEmbedder(embedding_dim=64)

        # Add documents
        for i in range(5):
            doc = Document(content=f"Document {i}")
            doc.embedding = embedder.embed(f"Document {i}")[0].tolist()
            kb.add_document(doc)

        retriever = SimpleRetriever()
        retriever.index(kb)

        # Query
        query_embedding = embedder.embed("Document 2")[0]
        results = retriever.retrieve(query_embedding, top_k=3)

        assert len(results) == 3
        # Should return document, score tuples
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_retriever_empty_kb(self):
        """Test retriever with empty knowledge base."""
        kb = KnowledgeBase()
        retriever = SimpleRetriever()

        with pytest.raises(ValueError):
            retriever.index(kb)


class TestRAGSystem:
    """Tests for RAGSystem class."""

    def test_rag_system_creation(self):
        """Test RAG system creation."""
        rag = RAGSystem()
        assert rag.embedder is not None
        assert rag.retriever is not None
        assert len(rag.knowledge_base) == 0

    def test_add_document(self):
        """Test adding document to RAG system."""
        rag = RAGSystem()
        doc = rag.add_document("Test content")
        assert doc.content == "Test content"
        assert doc.embedding is not None
        assert len(rag) == 1

    def test_add_documents(self):
        """Test adding multiple documents."""
        rag = RAGSystem()
        docs = rag.add_documents(["Content 1", "Content 2", "Content 3"])
        assert len(docs) == 3
        assert len(rag) == 3

    def test_retrieve(self):
        """Test document retrieval."""
        rag = RAGSystem()
        rag.add_documents([
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany.",
        ])

        results = rag.retrieve("What is the capital of France?", top_k=2)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieve_with_scores(self):
        """Test retrieval with scores."""
        rag = RAGSystem()
        rag.add_documents([
            "Paris is the capital of France.",
            "London is the capital of England.",
        ])

        results = rag.retrieve_with_scores("capital of France", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_remove_document(self):
        """Test removing document from RAG system."""
        rag = RAGSystem()
        doc = rag.add_document("Test content")
        assert len(rag) == 1

        removed = rag.remove_document(doc.doc_id)
        assert removed is True
        assert len(rag) == 0

    def test_clear(self):
        """Test clearing RAG system."""
        rag = RAGSystem()
        rag.add_documents(["Content 1", "Content 2"])
        assert len(rag) == 2

        rag.clear()
        assert len(rag) == 0

    def test_get_stats(self):
        """Test getting RAG system statistics."""
        rag = RAGSystem()
        rag.add_documents(["Content 1", "Content 2"])

        stats = rag.get_stats()
        assert stats["num_documents"] == 2
        assert "embedder" in stats
        assert "retriever" in stats
