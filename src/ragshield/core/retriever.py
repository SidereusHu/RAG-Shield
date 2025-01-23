"""Retrieval components for RAG system."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase


class Retriever(ABC):
    """Abstract base class for document retrievers."""

    @abstractmethod
    def index(self, knowledge_base: KnowledgeBase) -> None:
        """Index the knowledge base for retrieval.

        Args:
            knowledge_base: Knowledge base to index
        """
        pass

    @abstractmethod
    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve

        Returns:
            List of (document, similarity_score) tuples
        """
        pass


class FaissRetriever(Retriever):
    """FAISS-based retriever for efficient similarity search.

    Args:
        metric: Distance metric ('cosine' or 'l2')
    """

    def __init__(self, metric: str = "cosine"):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        self.metric = metric
        self.faiss = faiss
        self.index = None
        self.knowledge_base = None

    def index(self, knowledge_base: KnowledgeBase) -> None:
        """Index the knowledge base using FAISS.

        Args:
            knowledge_base: Knowledge base to index
        """
        if knowledge_base.size() == 0:
            raise ValueError("Cannot index empty knowledge base")

        # Get embeddings
        embeddings = np.array(knowledge_base.get_embeddings(), dtype=np.float32)

        # Create FAISS index
        dimension = embeddings.shape[1]

        if self.metric == "cosine":
            # Normalize embeddings for cosine similarity
            self.faiss.normalize_L2(embeddings)
            self.index = self.faiss.IndexFlatIP(dimension)  # Inner product = cosine after norm
        elif self.metric == "l2":
            self.index = self.faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Add embeddings to index
        self.index.add(embeddings)
        self.knowledge_base = knowledge_base

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve

        Returns:
            List of (document, similarity_score) tuples sorted by score (descending)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index() first.")

        # Reshape query
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            self.faiss.normalize_L2(query)

        # Search
        top_k = min(top_k, self.knowledge_base.size())
        scores, indices = self.index.search(query, top_k)

        # Convert to list of (document, score) tuples
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # -1 indicates no more results
                doc = self.knowledge_base.documents[idx]
                # Convert L2 distance to similarity if needed
                if self.metric == "l2":
                    score = -score  # Negate for sorting (smaller distance = higher similarity)
                results.append((doc, float(score)))

        return results


class SimpleRetriever(Retriever):
    """Simple retriever using numpy for small knowledge bases."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.embeddings = None
        self.knowledge_base = None

    def index(self, knowledge_base: KnowledgeBase) -> None:
        """Index the knowledge base.

        Args:
            knowledge_base: Knowledge base to index
        """
        if knowledge_base.size() == 0:
            raise ValueError("Cannot index empty knowledge base")

        self.embeddings = np.array(knowledge_base.get_embeddings(), dtype=np.float32)
        self.knowledge_base = knowledge_base

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve

        Returns:
            List of (document, similarity_score) tuples sorted by score (descending)
        """
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")

        query = query_embedding.reshape(1, -1)

        # Compute similarities
        if self.metric == "cosine":
            # Cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            embeddings_norm = self.embeddings / (
                np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(embeddings_norm, query_norm.T).flatten()
        elif self.metric == "l2":
            # L2 distance (negated for sorting)
            distances = np.linalg.norm(self.embeddings - query, axis=1)
            scores = -distances
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Convert to list of (document, score) tuples
        results = []
        for idx in top_indices:
            doc = self.knowledge_base.documents[idx]
            results.append((doc, float(scores[idx])))

        return results
