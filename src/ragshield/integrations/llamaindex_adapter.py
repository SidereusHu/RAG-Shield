"""LlamaIndex integration for RAG-Shield.

Provides secure wrappers for LlamaIndex indices, retrievers,
and query engines with built-in threat detection and defense.
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


# Type stubs for LlamaIndex (avoid hard dependency)
try:
    from llama_index.core import Document as LIDocument
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.retrievers import BaseRetriever as LIBaseRetriever
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.query_engine import BaseQueryEngine

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LIDocument = Any
    NodeWithScore = Any
    TextNode = Any
    LIBaseRetriever = Any
    BaseIndex = Any
    BaseQueryEngine = Any


def require_llamaindex(func: Callable) -> Callable:
    """Decorator to check LlamaIndex availability."""
    def wrapper(*args, **kwargs):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this feature. "
                "Install with: pip install llama-index"
            )
        return func(*args, **kwargs)
    return wrapper


class LlamaIndexIntegration(BaseRAGIntegration):
    """LlamaIndex integration for RAG-Shield.

    Provides secure wrappers for LlamaIndex components with
    built-in threat detection, sanitization, and monitoring.

    Example:
        ```python
        from ragshield.integrations import LlamaIndexIntegration, IntegrationConfig
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        # Create integration
        integration = LlamaIndexIntegration(
            config=IntegrationConfig(defense_level=DefenseLevel.STANDARD)
        )

        # Load documents securely
        documents = SimpleDirectoryReader("data").load_data()
        secure_docs = integration.secure_documents(documents)

        # Create index with secure documents
        index = VectorStoreIndex.from_documents(secure_docs)

        # Wrap query engine for secure querying
        query_engine = integration.wrap_query_engine(index.as_query_engine())
        ```
    """

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """Initialize LlamaIndex integration."""
        super().__init__(config=config, knowledge_base=knowledge_base)

    def get_framework_type(self) -> FrameworkType:
        """Get the framework type."""
        return FrameworkType.LLAMAINDEX

    @require_llamaindex
    def wrap_retriever(self, retriever: "LIBaseRetriever") -> "SecureLlamaRetriever":
        """Wrap a LlamaIndex retriever with security.

        Args:
            retriever: LlamaIndex BaseRetriever instance

        Returns:
            SecureLlamaRetriever wrapper
        """
        return SecureLlamaRetriever(
            base_retriever=retriever,
            detector=self._detector,
            config=self.config,
            stats=self._stats,
        )

    @require_llamaindex
    def wrap_vector_store(self, index: "BaseIndex") -> "SecureLlamaIndex":
        """Wrap a LlamaIndex index with security.

        Args:
            index: LlamaIndex BaseIndex instance

        Returns:
            SecureLlamaIndex wrapper
        """
        return SecureLlamaIndex(
            base_index=index,
            detector=self._detector,
            config=self.config,
            stats=self._stats,
        )

    @require_llamaindex
    def wrap_query_engine(
        self,
        query_engine: "BaseQueryEngine",
    ) -> "SecureQueryEngine":
        """Wrap a LlamaIndex query engine with security.

        Args:
            query_engine: LlamaIndex query engine

        Returns:
            SecureQueryEngine wrapper
        """
        return SecureQueryEngine(
            base_engine=query_engine,
            detector=self._detector,
            config=self.config,
            stats=self._stats,
        )

    @require_llamaindex
    def convert_document(self, li_doc: "LIDocument") -> Document:
        """Convert LlamaIndex document to RAG-Shield document.

        Args:
            li_doc: LlamaIndex Document

        Returns:
            RAG-Shield Document
        """
        doc_id = li_doc.doc_id or f"li_{hash(li_doc.text) % 10000:04x}"
        metadata = li_doc.metadata if hasattr(li_doc, 'metadata') else {}

        return Document(
            doc_id=doc_id,
            content=li_doc.text,
            metadata=metadata,
        )

    @require_llamaindex
    def to_llamaindex_document(self, doc: Document) -> "LIDocument":
        """Convert RAG-Shield document to LlamaIndex document.

        Args:
            doc: RAG-Shield Document

        Returns:
            LlamaIndex Document
        """
        return LIDocument(
            text=doc.content,
            doc_id=doc.doc_id,
            metadata=doc.metadata,
        )

    @require_llamaindex
    def secure_documents(
        self,
        documents: List["LIDocument"],
    ) -> List["LIDocument"]:
        """Filter and secure a list of LlamaIndex documents.

        Args:
            documents: List of LlamaIndex documents

        Returns:
            Filtered list with suspicious documents removed
        """
        if not self._detector:
            return documents

        secure_docs = []
        for li_doc in documents:
            doc = self.convert_document(li_doc)
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
            if not hasattr(li_doc, 'metadata') or li_doc.metadata is None:
                li_doc.metadata = {}
            li_doc.metadata["_security"] = {
                "verified": not result.is_poisoned,
                "threat_score": result.confidence,
            }
            secure_docs.append(li_doc)

        return secure_docs

    @require_llamaindex
    def secure_nodes(
        self,
        nodes: List["NodeWithScore"],
    ) -> List["NodeWithScore"]:
        """Filter and secure a list of retrieved nodes.

        Args:
            nodes: List of NodeWithScore objects

        Returns:
            Filtered list with suspicious nodes removed
        """
        if not self._detector:
            return nodes

        secure_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            content = node.get_content() if hasattr(node, 'get_content') else str(node)

            doc = Document(
                doc_id=node.node_id if hasattr(node, 'node_id') else "unknown",
                content=content,
            )

            result = self._detector.detect(doc)

            if result.is_poisoned and result.confidence > 0.85:
                self._stats["threats_detected"] += 1
                continue

            secure_nodes.append(node_with_score)

        return secure_nodes


class SecureLlamaRetriever(SecureRetrieverMixin):
    """Secure wrapper for LlamaIndex retrievers.

    Adds threat detection and filtering to retrieval results.
    """

    def __init__(
        self,
        base_retriever: "LIBaseRetriever",
        detector: Optional[PoisonDetector],
        config: IntegrationConfig,
        stats: Dict[str, int],
    ):
        """Initialize secure retriever."""
        self.base_retriever = base_retriever
        self.detector = detector
        self.config = config
        self._stats = stats

    def retrieve(self, query: str) -> List["NodeWithScore"]:
        """Retrieve nodes with security filtering.

        Args:
            query: Search query

        Returns:
            Filtered list of nodes with scores
        """
        self._stats["queries_processed"] += 1

        # Get base results
        results = self.base_retriever.retrieve(query)

        # Apply security filtering
        if self.detector and self.config.enable_detection:
            results = self._filter_nodes(results)

        return results

    def _filter_nodes(
        self,
        nodes: List["NodeWithScore"],
    ) -> List["NodeWithScore"]:
        """Filter nodes for security threats."""
        if not self.detector:
            return nodes

        filtered = []
        for node_with_score in nodes:
            node = node_with_score.node
            content = node.get_content() if hasattr(node, 'get_content') else str(node)

            doc = Document(
                doc_id=getattr(node, 'node_id', 'unknown'),
                content=content,
            )

            detection = self.detector.detect(doc)

            if detection.is_poisoned and detection.confidence > 0.85:
                self._stats["threats_detected"] += 1
                continue

            filtered.append(node_with_score)

        return filtered

    async def aretrieve(self, query: str) -> List["NodeWithScore"]:
        """Async version of retrieve."""
        return self.retrieve(query)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base retriever."""
        return getattr(self.base_retriever, name)


class SecureLlamaIndex:
    """Secure wrapper for LlamaIndex indices.

    Adds security checks to document ingestion and retrieval.
    """

    def __init__(
        self,
        base_index: "BaseIndex",
        detector: Optional[PoisonDetector],
        config: IntegrationConfig,
        stats: Dict[str, int],
    ):
        """Initialize secure index."""
        self.base_index = base_index
        self.detector = detector
        self.config = config
        self._stats = stats

    def insert(self, document: "LIDocument", **kwargs) -> None:
        """Insert document with security screening.

        Args:
            document: Document to insert
            **kwargs: Additional arguments
        """
        if not self.detector or not self.config.enable_detection:
            self.base_index.insert(document, **kwargs)
            return

        self._stats["documents_processed"] += 1

        # Check document
        doc = Document(
            doc_id=document.doc_id or "unknown",
            content=document.text,
        )

        detection = self.detector.detect(doc)

        if detection.is_poisoned and detection.confidence > 0.8:
            self._stats["threats_detected"] += 1
            self._stats["documents_blocked"] += 1

            if self.config.on_threat_callback:
                self.config.on_threat_callback({
                    "doc_id": doc.doc_id,
                    "threat_score": detection.confidence,
                    "action": "blocked",
                })
            return

        self.base_index.insert(document, **kwargs)

    def insert_nodes(self, nodes: List["TextNode"], **kwargs) -> None:
        """Insert nodes with security screening.

        Args:
            nodes: Nodes to insert
            **kwargs: Additional arguments
        """
        if not self.detector or not self.config.enable_detection:
            self.base_index.insert_nodes(nodes, **kwargs)
            return

        safe_nodes = []
        for node in nodes:
            self._stats["documents_processed"] += 1

            content = node.get_content() if hasattr(node, 'get_content') else str(node)
            doc = Document(
                doc_id=getattr(node, 'node_id', 'unknown'),
                content=content,
            )

            detection = self.detector.detect(doc)

            if detection.is_poisoned and detection.confidence > 0.8:
                self._stats["threats_detected"] += 1
                self._stats["documents_blocked"] += 1
                continue

            safe_nodes.append(node)

        if safe_nodes:
            self.base_index.insert_nodes(safe_nodes, **kwargs)

    def as_retriever(self, **kwargs) -> SecureLlamaRetriever:
        """Get a secure retriever from the index.

        Args:
            **kwargs: Retriever arguments

        Returns:
            SecureLlamaRetriever instance
        """
        base_retriever = self.base_index.as_retriever(**kwargs)
        return SecureLlamaRetriever(
            base_retriever=base_retriever,
            detector=self.detector,
            config=self.config,
            stats=self._stats,
        )

    def as_query_engine(self, **kwargs) -> "SecureQueryEngine":
        """Get a secure query engine from the index.

        Args:
            **kwargs: Query engine arguments

        Returns:
            SecureQueryEngine instance
        """
        base_engine = self.base_index.as_query_engine(**kwargs)
        return SecureQueryEngine(
            base_engine=base_engine,
            detector=self.detector,
            config=self.config,
            stats=self._stats,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base index."""
        return getattr(self.base_index, name)


class SecureQueryEngine:
    """Secure wrapper for LlamaIndex query engines.

    Adds security filtering to query results.
    """

    def __init__(
        self,
        base_engine: "BaseQueryEngine",
        detector: Optional[PoisonDetector],
        config: IntegrationConfig,
        stats: Dict[str, int],
    ):
        """Initialize secure query engine."""
        self.base_engine = base_engine
        self.detector = detector
        self.config = config
        self._stats = stats

    def query(self, query_str: str) -> Any:
        """Query with security checks.

        Args:
            query_str: Query string

        Returns:
            Query response
        """
        self._stats["queries_processed"] += 1

        # Get response from base engine
        response = self.base_engine.query(query_str)

        # Check source nodes if available
        if hasattr(response, 'source_nodes') and self.detector:
            secure_nodes = []
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                content = node.get_content() if hasattr(node, 'get_content') else str(node)

                doc = Document(
                    doc_id=getattr(node, 'node_id', 'unknown'),
                    content=content,
                )

                detection = self.detector.detect(doc)

                if detection.is_poisoned and detection.confidence > 0.9:
                    self._stats["threats_detected"] += 1
                    # Mark as potentially compromised
                    if hasattr(response, 'metadata'):
                        response.metadata = response.metadata or {}
                        response.metadata['_security_warning'] = (
                            "Response may be influenced by potentially malicious content"
                        )
                    continue

                secure_nodes.append(node_with_score)

            response.source_nodes = secure_nodes

        return response

    async def aquery(self, query_str: str) -> Any:
        """Async query with security checks."""
        return self.query(query_str)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base engine."""
        return getattr(self.base_engine, name)


def create_llamaindex_integration(
    defense_level: "DefenseLevel" = None,
    detector_preset: str = "default",
    **kwargs,
) -> LlamaIndexIntegration:
    """Convenience function to create LlamaIndex integration.

    Args:
        defense_level: Security level
        detector_preset: Detector preset
        **kwargs: Additional config options

    Returns:
        Configured LlamaIndexIntegration
    """
    from ragshield.defense import DefenseLevel as DL

    config = IntegrationConfig(
        defense_level=defense_level or DL.STANDARD,
        detector_preset=detector_preset,
        **kwargs,
    )
    return LlamaIndexIntegration(config=config)
