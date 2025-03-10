"""LangChain integration for RAG-Shield.

Provides secure wrappers for LangChain retrievers, vector stores,
and document loaders with built-in threat detection and defense.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection import PoisonDetector
from ragshield.integrations.base import (
    BaseRAGIntegration,
    FrameworkType,
    IntegrationConfig,
    SecureDocument,
    SecureRetrieverMixin,
)


# Type stubs for LangChain (avoid hard dependency)
try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LCDocument = Any
    BaseRetriever = Any
    VectorStore = Any
    CallbackManagerForRetrieverRun = Any


def require_langchain(func: Callable) -> Callable:
    """Decorator to check LangChain availability."""
    def wrapper(*args, **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this feature. "
                "Install with: pip install langchain langchain-core"
            )
        return func(*args, **kwargs)
    return wrapper


class LangChainIntegration(BaseRAGIntegration):
    """LangChain integration for RAG-Shield.

    Provides secure wrappers for LangChain components with
    built-in threat detection, sanitization, and monitoring.

    Example:
        ```python
        from ragshield.integrations import LangChainIntegration, IntegrationConfig
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings

        # Create integration
        integration = LangChainIntegration(
            config=IntegrationConfig(defense_level=DefenseLevel.STANDARD)
        )

        # Wrap existing vector store
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        secure_store = integration.wrap_vector_store(vectorstore)

        # Use securely
        results = secure_store.similarity_search("query")
        ```
    """

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """Initialize LangChain integration."""
        super().__init__(config=config, knowledge_base=knowledge_base)

    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        return FrameworkType.LANGCHAIN

    @require_langchain
    def wrap_retriever(self, retriever: "BaseRetriever") -> "SecureRetriever":
        """Wrap a LangChain retriever with security.

        Args:
            retriever: LangChain BaseRetriever instance

        Returns:
            SecureRetriever wrapper
        """
        return SecureRetriever(
            base_retriever=retriever,
            detector=self._detector,
            config=self.config,
            stats=self._stats,
        )

    @require_langchain
    def wrap_vector_store(self, vector_store: "VectorStore") -> "SecureVectorStore":
        """Wrap a LangChain vector store with security.

        Args:
            vector_store: LangChain VectorStore instance

        Returns:
            SecureVectorStore wrapper
        """
        return SecureVectorStore(
            base_store=vector_store,
            detector=self._detector,
            config=self.config,
            stats=self._stats,
        )

    @require_langchain
    def convert_document(self, lc_doc: "LCDocument") -> Document:
        """Convert LangChain document to RAG-Shield document.

        Args:
            lc_doc: LangChain Document

        Returns:
            RAG-Shield Document
        """
        return Document(
            doc_id=lc_doc.metadata.get("doc_id", f"lc_{hash(lc_doc.page_content) % 10000:04x}"),
            content=lc_doc.page_content,
            metadata=lc_doc.metadata,
        )

    @require_langchain
    def to_langchain_document(self, doc: Document) -> "LCDocument":
        """Convert RAG-Shield document to LangChain document.

        Args:
            doc: RAG-Shield Document

        Returns:
            LangChain Document
        """
        metadata = {**doc.metadata, "doc_id": doc.doc_id}
        return LCDocument(page_content=doc.content, metadata=metadata)

    @require_langchain
    def secure_documents(
        self,
        documents: List["LCDocument"],
    ) -> List["LCDocument"]:
        """Filter and secure a list of LangChain documents.

        Args:
            documents: List of LangChain documents

        Returns:
            Filtered list with suspicious documents removed
        """
        if not self._detector:
            return documents

        secure_docs = []
        for lc_doc in documents:
            doc = self.convert_document(lc_doc)
            result = self._detector.detect(doc)

            self._stats["documents_processed"] += 1

            if result.is_poisoned and result.confidence > 0.8:
                self._stats["threats_detected"] += 1
                if self.config.on_threat_callback:
                    self.config.on_threat_callback({
                        "doc_id": doc.doc_id,
                        "threat_score": result.confidence,
                        "action": "filtered",
                    })
                continue

            # Add security metadata
            lc_doc.metadata["_security"] = {
                "verified": not result.is_poisoned,
                "threat_score": result.confidence,
            }
            secure_docs.append(lc_doc)

        return secure_docs

    @require_langchain
    def create_secure_loader(
        self,
        loader_class: Type,
        *args,
        **kwargs,
    ) -> "SecureDocumentLoader":
        """Create a secure document loader wrapper.

        Args:
            loader_class: LangChain loader class
            *args: Loader arguments
            **kwargs: Loader keyword arguments

        Returns:
            SecureDocumentLoader wrapper
        """
        base_loader = loader_class(*args, **kwargs)
        return SecureDocumentLoader(
            base_loader=base_loader,
            integration=self,
        )


class SecureRetriever(SecureRetrieverMixin):
    """Secure wrapper for LangChain retrievers.

    Adds threat detection and filtering to retrieval results.
    """

    def __init__(
        self,
        base_retriever: "BaseRetriever",
        detector: Optional[PoisonDetector],
        config: IntegrationConfig,
        stats: Dict[str, int],
    ):
        """Initialize secure retriever.

        Args:
            base_retriever: LangChain retriever to wrap
            detector: Poison detector
            config: Integration config
            stats: Statistics dictionary
        """
        self.base_retriever = base_retriever
        self.detector = detector
        self.config = config
        self._stats = stats

    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional["CallbackManagerForRetrieverRun"] = None,
    ) -> List["LCDocument"]:
        """Get relevant documents with security filtering.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            Filtered list of relevant documents
        """
        self._stats["queries_processed"] += 1

        # Get base results
        if run_manager:
            results = self.base_retriever.get_relevant_documents(
                query, run_manager=run_manager
            )
        else:
            results = self.base_retriever.get_relevant_documents(query)

        # Apply security filtering
        if self.detector and self.config.enable_detection:
            results = self._filter_results(results)

        return results

    def _filter_results(
        self,
        results: List["LCDocument"],
    ) -> List["LCDocument"]:
        """Filter results for security threats."""
        if not self.detector:
            return results

        filtered = []
        for doc in results:
            rag_doc = Document(
                doc_id=doc.metadata.get("doc_id", "unknown"),
                content=doc.page_content,
                metadata=doc.metadata,
            )

            detection = self.detector.detect(rag_doc)

            if detection.is_poisoned and detection.confidence > 0.85:
                self._stats["threats_detected"] += 1
                continue

            doc.metadata["_threat_score"] = detection.confidence
            filtered.append(doc)

        return filtered

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional["CallbackManagerForRetrieverRun"] = None,
    ) -> List["LCDocument"]:
        """Async version of get_relevant_documents."""
        # For simplicity, use sync version
        return self.get_relevant_documents(query, run_manager=run_manager)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base retriever."""
        return getattr(self.base_retriever, name)


class SecureVectorStore:
    """Secure wrapper for LangChain vector stores.

    Adds security checks to document ingestion and retrieval.
    """

    def __init__(
        self,
        base_store: "VectorStore",
        detector: Optional[PoisonDetector],
        config: IntegrationConfig,
        stats: Dict[str, int],
    ):
        """Initialize secure vector store.

        Args:
            base_store: LangChain vector store to wrap
            detector: Poison detector
            config: Integration config
            stats: Statistics dictionary
        """
        self.base_store = base_store
        self.detector = detector
        self.config = config
        self._stats = stats

    def add_documents(
        self,
        documents: List["LCDocument"],
        **kwargs,
    ) -> List[str]:
        """Add documents with security screening.

        Args:
            documents: Documents to add
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        if not self.detector or not self.config.enable_detection:
            return self.base_store.add_documents(documents, **kwargs)

        # Filter suspicious documents
        safe_docs = []
        for doc in documents:
            self._stats["documents_processed"] += 1

            rag_doc = Document(
                doc_id=doc.metadata.get("doc_id", f"doc_{len(safe_docs)}"),
                content=doc.page_content,
                metadata=doc.metadata,
            )

            detection = self.detector.detect(rag_doc)

            if detection.is_poisoned and detection.confidence > 0.8:
                self._stats["threats_detected"] += 1
                self._stats["documents_blocked"] += 1

                if self.config.on_threat_callback:
                    self.config.on_threat_callback({
                        "doc_id": rag_doc.doc_id,
                        "threat_score": detection.confidence,
                        "action": "blocked",
                    })
                continue

            safe_docs.append(doc)

        if not safe_docs:
            return []

        return self.base_store.add_documents(safe_docs, **kwargs)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs,
    ) -> List[str]:
        """Add texts with security screening.

        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        metadatas = metadatas or [{} for _ in texts]

        # Convert to documents for screening
        documents = [
            LCDocument(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        return self.add_documents(documents, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> List["LCDocument"]:
        """Search with security filtering.

        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            Filtered search results
        """
        self._stats["queries_processed"] += 1

        # Get more results than needed for filtering
        results = self.base_store.similarity_search(
            query, k=k * 2, **kwargs
        )

        if not self.detector or not self.config.enable_detection:
            return results[:k]

        # Filter and return top k
        filtered = []
        for doc in results:
            rag_doc = Document(
                doc_id=doc.metadata.get("doc_id", "unknown"),
                content=doc.page_content,
            )

            detection = self.detector.detect(rag_doc)

            if detection.is_poisoned and detection.confidence > 0.85:
                self._stats["threats_detected"] += 1
                continue

            doc.metadata["_threat_score"] = detection.confidence
            filtered.append(doc)

            if len(filtered) >= k:
                break

        return filtered

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> List[tuple]:
        """Search with scores and security filtering.

        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        self._stats["queries_processed"] += 1

        results = self.base_store.similarity_search_with_score(
            query, k=k * 2, **kwargs
        )

        if not self.detector or not self.config.enable_detection:
            return results[:k]

        filtered = []
        for doc, score in results:
            rag_doc = Document(
                doc_id=doc.metadata.get("doc_id", "unknown"),
                content=doc.page_content,
            )

            detection = self.detector.detect(rag_doc)

            if detection.is_poisoned and detection.confidence > 0.85:
                self._stats["threats_detected"] += 1
                continue

            doc.metadata["_threat_score"] = detection.confidence
            filtered.append((doc, score))

            if len(filtered) >= k:
                break

        return filtered

    def as_retriever(self, **kwargs) -> SecureRetriever:
        """Convert to secure retriever.

        Args:
            **kwargs: Retriever arguments

        Returns:
            SecureRetriever instance
        """
        base_retriever = self.base_store.as_retriever(**kwargs)
        return SecureRetriever(
            base_retriever=base_retriever,
            detector=self.detector,
            config=self.config,
            stats=self._stats,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base store."""
        return getattr(self.base_store, name)


class SecureDocumentLoader:
    """Secure wrapper for LangChain document loaders.

    Screens loaded documents for potential threats.
    """

    def __init__(
        self,
        base_loader: Any,
        integration: LangChainIntegration,
    ):
        """Initialize secure loader.

        Args:
            base_loader: LangChain document loader
            integration: LangChain integration instance
        """
        self.base_loader = base_loader
        self.integration = integration

    def load(self) -> List["LCDocument"]:
        """Load and screen documents.

        Returns:
            Filtered list of documents
        """
        documents = self.base_loader.load()
        return self.integration.secure_documents(documents)

    def load_and_split(self, text_splitter: Any = None) -> List["LCDocument"]:
        """Load, split, and screen documents.

        Args:
            text_splitter: Optional text splitter

        Returns:
            Filtered list of split documents
        """
        if text_splitter:
            documents = self.base_loader.load_and_split(text_splitter)
        else:
            documents = self.base_loader.load_and_split()

        return self.integration.secure_documents(documents)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base loader."""
        return getattr(self.base_loader, name)


def create_langchain_integration(
    defense_level: "DefenseLevel" = None,
    detector_preset: str = "default",
    **kwargs,
) -> LangChainIntegration:
    """Convenience function to create LangChain integration.

    Args:
        defense_level: Security level
        detector_preset: Detector preset
        **kwargs: Additional config options

    Returns:
        Configured LangChainIntegration
    """
    from ragshield.defense import DefenseLevel as DL

    config = IntegrationConfig(
        defense_level=defense_level or DL.STANDARD,
        detector_preset=detector_preset,
        **kwargs,
    )
    return LangChainIntegration(config=config)
