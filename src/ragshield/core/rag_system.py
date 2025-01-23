"""Main RAG system implementation."""

from typing import List, Optional, Union
import numpy as np

from ragshield.core.document import Document, DocumentMetadata
from ragshield.core.embedder import Embedder, SentenceTransformerEmbedder, MockEmbedder
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.core.retriever import Retriever, SimpleRetriever


class RAGSystem:
    """Retrieval-Augmented Generation system.

    Args:
        embedder: Embedder instance for generating text embeddings
        retriever: Retriever instance for similarity search
        knowledge_base: Optional pre-existing knowledge base
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        retriever: Optional[Retriever] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.embedder = embedder or MockEmbedder()
        self.retriever = retriever or SimpleRetriever()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self._indexed = False

    def add_document(
        self,
        content: str,
        metadata: Optional[DocumentMetadata] = None,
        compute_embedding: bool = True,
    ) -> Document:
        """Add a document to the knowledge base.

        Args:
            content: Document text content
            metadata: Optional document metadata
            compute_embedding: Whether to compute embedding immediately

        Returns:
            The created Document object
        """
        doc = Document(content=content, metadata=metadata or DocumentMetadata())

        if compute_embedding:
            embedding = self.embedder.embed(content)
            doc.embedding = embedding[0].tolist()

        self.knowledge_base.add_document(doc)
        self._indexed = False  # Need to reindex

        return doc

    def add_documents(
        self,
        contents: List[str],
        metadatas: Optional[List[DocumentMetadata]] = None,
        compute_embeddings: bool = True,
    ) -> List[Document]:
        """Add multiple documents to the knowledge base.

        Args:
            contents: List of document text contents
            metadatas: Optional list of document metadatas
            compute_embeddings: Whether to compute embeddings immediately

        Returns:
            List of created Document objects
        """
        if metadatas is None:
            metadatas = [DocumentMetadata() for _ in contents]

        if len(metadatas) != len(contents):
            raise ValueError("Number of metadatas must match number of contents")

        docs = []

        if compute_embeddings:
            # Batch embedding for efficiency
            embeddings = self.embedder.embed(contents)

            for content, metadata, embedding in zip(contents, metadatas, embeddings):
                doc = Document(content=content, metadata=metadata, embedding=embedding.tolist())
                self.knowledge_base.add_document(doc)
                docs.append(doc)
        else:
            for content, metadata in zip(contents, metadatas):
                doc = Document(content=content, metadata=metadata)
                self.knowledge_base.add_document(doc)
                docs.append(doc)

        self._indexed = False  # Need to reindex
        return docs

    def build_index(self) -> None:
        """Build the retrieval index.

        Must be called after adding documents and before retrieval.
        """
        if self.knowledge_base.size() == 0:
            raise ValueError("Cannot build index on empty knowledge base")

        self.retriever.index(self.knowledge_base)
        self._indexed = True

    def retrieve(
        self, query: str, top_k: int = 5, auto_index: bool = True
    ) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            auto_index: Automatically build index if not already built

        Returns:
            List of retrieved documents, sorted by relevance
        """
        if not self._indexed:
            if auto_index:
                self.build_index()
            else:
                raise RuntimeError("Index not built. Call build_index() first.")

        # Embed query
        query_embedding = self.embedder.embed(query)[0]

        # Retrieve documents
        results = self.retriever.retrieve(query_embedding, top_k=top_k)

        # Return just the documents (without scores)
        return [doc for doc, score in results]

    def retrieve_with_scores(
        self, query: str, top_k: int = 5, auto_index: bool = True
    ) -> List[tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            auto_index: Automatically build index if not already built

        Returns:
            List of (document, score) tuples, sorted by score (descending)
        """
        if not self._indexed:
            if auto_index:
                self.build_index()
            else:
                raise RuntimeError("Index not built. Call build_index() first.")

        # Embed query
        query_embedding = self.embedder.embed(query)[0]

        # Retrieve documents with scores
        results = self.retriever.retrieve(query_embedding, top_k=top_k)

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        removed = self.knowledge_base.remove_document(doc_id)
        if removed:
            self._indexed = False
        return removed

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.knowledge_base.clear()
        self._indexed = False

    def get_stats(self) -> dict:
        """Get statistics about the RAG system.

        Returns:
            Dictionary containing system statistics
        """
        return {
            "num_documents": self.knowledge_base.size(),
            "indexed": self._indexed,
            "embedder": self.embedder.__class__.__name__,
            "retriever": self.retriever.__class__.__name__,
            "embedding_dim": self.embedder.get_embedding_dim(),
        }

    def __len__(self) -> int:
        """Return the number of documents in the knowledge base."""
        return len(self.knowledge_base)

    def __str__(self) -> str:
        """Return string representation."""
        return f"RAGSystem(documents={len(self.knowledge_base)}, indexed={self._indexed})"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return self.__str__()
