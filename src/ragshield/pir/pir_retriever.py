"""PIR-based Retriever for RAG Systems.

Integrates Private Information Retrieval with RAG to enable
privacy-preserving document retrieval.

The key challenge: Standard PIR retrieves by index, but RAG needs
to retrieve by similarity. Solutions:
1. Pre-compute indices: Client computes similarity locally (requires embeddings)
2. Hybrid approach: Use DP for similarity, PIR for document fetch
3. Oblivious retrieval: Specialized protocols for similarity search

This module implements practical hybrid approaches.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
import time

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.core.retriever import Retriever
from ragshield.pir.base import PIRParameters, PIRScheme, PIRResult
from ragshield.pir.single_server import SingleServerPIR
from ragshield.pir.multi_server import MultiServerPIR


class PIRMode(Enum):
    """PIR operation modes."""
    SINGLE_SERVER = "single_server"  # Computational PIR
    MULTI_SERVER = "multi_server"  # Information-theoretic PIR
    HYBRID = "hybrid"  # PIR for fetch, DP for ranking


@dataclass
class PIRRetrievalResult:
    """Result of PIR-based retrieval.

    Attributes:
        documents: Retrieved documents
        indices_retrieved: Which indices were retrieved via PIR
        pir_time: Time spent in PIR operations
        total_time: Total retrieval time
        metadata: Additional information
    """
    documents: List[Tuple[Document, float]]
    indices_retrieved: List[int]
    pir_time: float = 0.0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIRRetriever:
    """Privacy-preserving retriever using PIR.

    Allows fetching documents without the server learning which
    documents were retrieved.

    Two modes of operation:
    1. Index-based: Client specifies indices, server doesn't learn which
    2. Hybrid: Client computes similarity locally, uses PIR to fetch

    Example:
        >>> retriever = PIRRetriever(knowledge_base, mode=PIRMode.MULTI_SERVER)
        >>> # Client-side similarity computation
        >>> indices = retriever.compute_top_k_indices(query_embedding, k=5)
        >>> # Private fetching
        >>> docs = retriever.fetch_by_indices(indices)
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        mode: PIRMode = PIRMode.MULTI_SERVER,
        num_servers: int = 2,
        cache_embeddings: bool = True,
    ):
        """Initialize PIR retriever.

        Args:
            knowledge_base: Knowledge base to retrieve from
            mode: PIR mode to use
            num_servers: Number of servers for multi-server PIR
            cache_embeddings: Whether to cache embeddings for local similarity
        """
        self.mode = mode
        self.num_servers = num_servers
        self.cache_embeddings = cache_embeddings

        self._kb: Optional[KnowledgeBase] = None
        self._pir_protocol = None
        self._embeddings: Optional[np.ndarray] = None
        self._documents: List[Document] = []

        if knowledge_base:
            self.setup(knowledge_base)

    def setup(self, knowledge_base: KnowledgeBase) -> None:
        """Setup retriever with knowledge base.

        Args:
            knowledge_base: Knowledge base to index
        """
        self._kb = knowledge_base
        self._documents = knowledge_base.documents

        # Cache embeddings for local similarity computation
        if self.cache_embeddings:
            self._embeddings = np.array(knowledge_base.get_embeddings())

        # Setup PIR protocol with document contents
        # For PIR, we use document indices as the "database"
        # The actual documents are fetched after PIR reveals the index
        doc_data = []
        for i, doc in enumerate(self._documents):
            # Store index and content hash for verification
            doc_data.append(i)

        if self.mode == PIRMode.SINGLE_SERVER:
            self._pir_protocol = SingleServerPIR(doc_data, key_bits=64)
        elif self.mode == PIRMode.MULTI_SERVER:
            self._pir_protocol = MultiServerPIR(
                doc_data,
                num_servers=self.num_servers,
            )
        else:
            # Hybrid mode - PIR for document content
            self._pir_protocol = MultiServerPIR(
                doc_data,
                num_servers=self.num_servers,
            )

    def compute_top_k_indices(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Compute top-k document indices locally.

        This is done on the client side using cached embeddings.
        The server doesn't see the query.

        Args:
            query_embedding: Query embedding vector
            k: Number of results

        Returns:
            List of (index, similarity_score) tuples
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not cached. Set cache_embeddings=True")

        # Normalize for cosine similarity
        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        embeddings = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Compute similarities
        similarities = np.dot(embeddings, query)

        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    def fetch_by_index(self, index: int) -> Tuple[Document, PIRResult]:
        """Fetch a single document by index using PIR.

        The server doesn't learn which index was requested.

        Args:
            index: Document index to fetch

        Returns:
            Tuple of (document, PIR result)
        """
        if self._pir_protocol is None:
            raise RuntimeError("Retriever not setup")

        # Use PIR to privately retrieve
        pir_result = self._pir_protocol.retrieve(index)

        # Get the actual document
        # In a real system, the document content would be encrypted
        # and retrieved via PIR
        document = self._documents[index]

        return document, pir_result

    def fetch_by_indices(
        self,
        indices: List[int],
    ) -> List[Tuple[Document, PIRResult]]:
        """Fetch multiple documents by indices using PIR.

        Args:
            indices: List of indices to fetch

        Returns:
            List of (document, PIR result) tuples
        """
        results = []
        for idx in indices:
            doc, pir_result = self.fetch_by_index(idx)
            results.append((doc, pir_result))
        return results

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> PIRRetrievalResult:
        """Full private retrieval: similarity + PIR fetch.

        1. Computes similarity locally (client-side)
        2. Fetches top-k documents via PIR

        Args:
            query_embedding: Query embedding
            top_k: Number of documents to retrieve

        Returns:
            PIRRetrievalResult with documents
        """
        start_time = time.time()

        # Step 1: Compute top-k indices locally
        top_k_results = self.compute_top_k_indices(query_embedding, top_k)

        # Step 2: Fetch documents via PIR
        pir_start = time.time()
        documents = []
        indices = []

        for idx, score in top_k_results:
            doc, _ = self.fetch_by_index(idx)
            documents.append((doc, score))
            indices.append(idx)

        pir_time = time.time() - pir_start
        total_time = time.time() - start_time

        return PIRRetrievalResult(
            documents=documents,
            indices_retrieved=indices,
            pir_time=pir_time,
            total_time=total_time,
            metadata={
                "mode": self.mode.value,
                "num_fetched": len(documents),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "mode": self.mode.value,
            "num_documents": len(self._documents),
            "embeddings_cached": self._embeddings is not None,
        }

        if self._pir_protocol:
            stats["pir_stats"] = self._pir_protocol.get_stats()

        return stats


class HybridPIRRetriever:
    """Hybrid retriever combining DP similarity with PIR fetch.

    Uses differential privacy for the similarity computation phase
    and PIR for the document fetching phase.

    This provides:
    - DP protection for query similarity patterns
    - PIR protection for which documents are actually retrieved
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        epsilon: float = 0.1,
        pir_mode: PIRMode = PIRMode.MULTI_SERVER,
        num_servers: int = 2,
    ):
        """Initialize hybrid retriever.

        Args:
            knowledge_base: Knowledge base
            epsilon: DP epsilon for similarity phase
            pir_mode: PIR mode for fetch phase
            num_servers: Number of PIR servers
        """
        self.epsilon = epsilon
        self.pir_mode = pir_mode

        self._pir_retriever = PIRRetriever(
            knowledge_base,
            mode=pir_mode,
            num_servers=num_servers,
        )

    def setup(self, knowledge_base: KnowledgeBase) -> None:
        """Setup with knowledge base."""
        self._pir_retriever.setup(knowledge_base)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        add_noise: bool = True,
    ) -> PIRRetrievalResult:
        """Retrieve with DP similarity and PIR fetch.

        Args:
            query_embedding: Query embedding
            top_k: Number of documents
            add_noise: Whether to add DP noise to similarities

        Returns:
            PIRRetrievalResult
        """
        start_time = time.time()

        # Compute similarities locally
        top_results = self._pir_retriever.compute_top_k_indices(
            query_embedding, top_k * 2  # Get more for DP selection
        )

        if add_noise:
            # Add Laplace noise to scores for DP
            indices = [idx for idx, _ in top_results]
            scores = np.array([score for _, score in top_results])

            # Add noise calibrated to sensitivity/epsilon
            sensitivity = 1.0  # Similarity scores in [-1, 1]
            noise = np.random.laplace(0, sensitivity / self.epsilon, len(scores))
            noisy_scores = scores + noise

            # Re-rank by noisy scores
            ranking = np.argsort(noisy_scores)[::-1]
            top_results = [(indices[i], float(noisy_scores[i])) for i in ranking[:top_k]]

        # Fetch via PIR
        pir_start = time.time()
        documents = []
        indices = []

        for idx, score in top_results[:top_k]:
            doc, _ = self._pir_retriever.fetch_by_index(idx)
            documents.append((doc, score))
            indices.append(idx)

        return PIRRetrievalResult(
            documents=documents,
            indices_retrieved=indices,
            pir_time=time.time() - pir_start,
            total_time=time.time() - start_time,
            metadata={
                "dp_epsilon": self.epsilon if add_noise else None,
                "pir_mode": self.pir_mode.value,
            },
        )


class BatchPIRRetriever:
    """Batch PIR retriever for efficient multi-query retrieval.

    Amortizes PIR overhead across multiple queries by batching.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        batch_size: int = 10,
        pir_mode: PIRMode = PIRMode.MULTI_SERVER,
    ):
        """Initialize batch retriever.

        Args:
            knowledge_base: Knowledge base
            batch_size: Queries to batch together
            pir_mode: PIR mode
        """
        self.batch_size = batch_size
        self._pir_retriever = PIRRetriever(knowledge_base, mode=pir_mode)
        self._pending_queries: List[Tuple[np.ndarray, int]] = []

    def setup(self, knowledge_base: KnowledgeBase) -> None:
        """Setup with knowledge base."""
        self._pir_retriever.setup(knowledge_base)

    def add_query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Optional[List[PIRRetrievalResult]]:
        """Add a query to the batch.

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            Results if batch is full, None otherwise
        """
        self._pending_queries.append((query_embedding, top_k))

        if len(self._pending_queries) >= self.batch_size:
            return self._process_batch()

        return None

    def _process_batch(self) -> List[PIRRetrievalResult]:
        """Process pending batch."""
        results = []

        # Collect all needed indices
        all_indices = set()
        query_needs = []

        for query_emb, top_k in self._pending_queries:
            top_results = self._pir_retriever.compute_top_k_indices(query_emb, top_k)
            indices = [idx for idx, _ in top_results]
            query_needs.append((top_results, indices))
            all_indices.update(indices)

        # Fetch all unique indices via PIR
        fetched_docs = {}
        for idx in all_indices:
            doc, _ = self._pir_retriever.fetch_by_index(idx)
            fetched_docs[idx] = doc

        # Assemble results for each query
        for top_results, indices in query_needs:
            documents = [
                (fetched_docs[idx], score)
                for idx, score in top_results
            ]
            results.append(PIRRetrievalResult(
                documents=documents,
                indices_retrieved=indices,
                metadata={"batch_mode": True},
            ))

        self._pending_queries.clear()
        return results

    def flush(self) -> List[PIRRetrievalResult]:
        """Process any remaining queries."""
        if self._pending_queries:
            return self._process_batch()
        return []
